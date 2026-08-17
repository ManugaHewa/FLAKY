"""
collect_data.py

Pulls historical pytest run results from a GitHub repo's Actions workflow
runs, and writes them out as JSONL rows of:

    {
        "test_name": "tests/test_foo.py::test_bar",
        "commit_sha": "abc123",
        "run_id": 123456,
        "attempt": 1,
        "outcome": "failed" | "passed",
        "error_message": "AssertionError: ..." | null,
        "duration_sec": null,
        "timestamp": "2026-08-01T12:00:00Z"
    }

Key idea for labeling flakiness (used later in features.py):
if the SAME test on the SAME commit fails on attempt 1 but passes on a
rerun (attempt 2+), that's strong evidence the test is flaky rather than
a real regression.

IMPORTANT: we parse the raw GitHub Actions JOB LOG (plain text), not
JUnit XML. pytest's JUnit XML output does NOT reliably capture rerun
attempts -- only the final outcome is recorded, with no indication a
retry happened (see pytest-rerunfailures issue #97 upstream). The plain
console log is more reliable here because pytest-rerunfailures prints an
explicit "RERUN <test_id>" line for every retry, which is exactly the
signal we need.

USAGE:
    python collect_data.py --repo apache/airflow --workflows ci.yml \
        --job-name "Test provider packages" --max-runs 100 \
        --output data/raw_runs.jsonl

NOTE ON LOG FORMAT: pytest's exact console output format (verbose vs.
default, whether test node IDs appear inline) varies by repo config. The
regexes in parse_pytest_log are a reasonable starting point but you
SHOULD sanity-check them against one real downloaded log from your
target repo/job before trusting the output -- print a few raw log lines
first (see `--debug-print-lines` flag) and adjust the patterns if needed.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"

# --- Log line patterns -----------------------------------------------------
# pytest-rerunfailures prints a line like:
#   RERUN tests/test_foo.py::test_bar
# and the final short summary section prints lines like:
#   FAILED tests/test_foo.py::test_bar - AssertionError: ...
#   PASSED tests/test_foo.py::test_bar
# GitHub Actions log lines are also prefixed with a timestamp, e.g.:
#   2026-08-01T12:00:03.4512310Z FAILED tests/test_foo.py::test_bar - ...
# We strip that prefix before matching.
TIMESTAMP_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*"
)
RERUN_RE = re.compile(r"^RERUN\s+(?P<test>\S+::\S+)")
FAILED_RE = re.compile(r"^FAILED\s+(?P<test>\S+::\S+)(?:\s+-\s+(?P<msg>.*))?")
PASSED_RE = re.compile(r"^PASSED\s+(?P<test>\S+::\S+)")
# Verbose-mode inline result lines, e.g. "tests/test_foo.py::test_bar PASSED"
INLINE_RE = re.compile(
    r"^(?P<test>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|RERUN|ERROR)\b"
)


def get_session() -> requests.Session:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("WARNING: no GITHUB_TOKEN set -- you will hit low rate limits.",
              file=sys.stderr)
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def list_workflow_runs(session: requests.Session, repo: str, workflow_file: str,
                        max_runs: int) -> list[dict]:
    """Page through GitHub Actions runs for a given workflow file."""
    runs = []
    page = 1
    per_page = min(100, max_runs)
    with tqdm(total=max_runs, desc=f"Listing runs for {workflow_file}") as pbar:
        while len(runs) < max_runs:
            url = f"{GITHUB_API}/repos/{repo}/actions/workflows/{workflow_file}/runs"
            resp = session.get(url, params={
                "per_page": per_page,
                "page": page,
                "status": "completed",
            })
            resp.raise_for_status()
            batch = resp.json().get("workflow_runs", [])
            if not batch:
                break
            runs.extend(batch)
            pbar.update(len(batch))
            page += 1
            time.sleep(0.2)  # be polite to the API / rate limits
    return runs[:max_runs]


def find_job_id(session: requests.Session, repo: str, run_id: int,
                 job_name: str) -> int | None:
    """
    A workflow run can contain many jobs (matrix builds, parallel shards).
    We only want the one job whose name matches --job-name (substring match,
    case-insensitive) so we're not mixing unrelated test suites together.
    """
    url = f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/jobs"
    resp = session.get(url, params={"per_page": 100})
    resp.raise_for_status()
    jobs = resp.json().get("jobs", [])
    for job in jobs:
        if job_name.lower() in job.get("name", "").lower():
            return job["id"]
    return None


def download_job_log(session: requests.Session, repo: str, job_id: int) -> str | None:
    """
    Downloads the plain-text log for a single job. GitHub returns a
    redirect to a blob-storage URL; `requests` follows it automatically.
    Returns None if the log isn't available (e.g. job was skipped/cancelled).
    """
    url = f"{GITHUB_API}/repos/{repo}/actions/jobs/{job_id}/logs"
    resp = session.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.text


def parse_pytest_log(log_text: str, run_id: int, commit_sha: str,
                      timestamp: str) -> list[dict]:
    """
    Parses a plain-text pytest log into per-test-attempt rows.

    Strategy: scan line by line, stripping the GitHub Actions timestamp
    prefix, and match RERUN / FAILED / PASSED lines (both the short-summary
    style and the inline verbose style). We track how many times we've
    seen RERUN for a given test to assign an `attempt` number.

    error_message is best-effort: whatever trails "FAILED <test> - " on
    the same line. Full tracebacks span multiple lines in real logs; for
    a first pass this one-line message is usually enough signal for the
    error_msg_similarity feature. Revisit if you want full tracebacks.
    """
    rerun_counts: dict[str, int] = {}
    rows = []

    for raw_line in log_text.splitlines():
        line = TIMESTAMP_PREFIX_RE.sub("", raw_line).strip()

        m = RERUN_RE.match(line)
        if m:
            test = m.group("test")
            rerun_counts[test] = rerun_counts.get(test, 0) + 1
            rows.append({
                "test_name": test,
                "commit_sha": commit_sha,
                "run_id": run_id,
                "attempt": rerun_counts[test],  # attempt that just failed
                "outcome": "failed",
                "error_message": None,
                "duration_sec": None,
                "timestamp": timestamp,
            })
            continue

        m = FAILED_RE.match(line)
        if m:
            test = m.group("test")
            attempt = rerun_counts.get(test, 0) + 1
            rows.append({
                "test_name": test,
                "commit_sha": commit_sha,
                "run_id": run_id,
                "attempt": attempt,
                "outcome": "failed",
                "error_message": m.group("msg"),
                "duration_sec": None,
                "timestamp": timestamp,
            })
            continue

        m = PASSED_RE.match(line)
        if m:
            test = m.group("test")
            attempt = rerun_counts.get(test, 0) + 1
            rows.append({
                "test_name": test,
                "commit_sha": commit_sha,
                "run_id": run_id,
                "attempt": attempt,
                "outcome": "passed",
                "error_message": None,
                "duration_sec": None,
                "timestamp": timestamp,
            })
            continue

        m = INLINE_RE.match(line)
        if m and m.group("outcome") != "RERUN":
            test = m.group("test")
            attempt = rerun_counts.get(test, 0) + 1
            rows.append({
                "test_name": test,
                "commit_sha": commit_sha,
                "run_id": run_id,
                "attempt": attempt,
                "outcome": "failed" if m.group("outcome") in ("FAILED", "ERROR") else "passed",
                "error_message": None,
                "duration_sec": None,
                "timestamp": timestamp,
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. apache/airflow")
    parser.add_argument("--workflows", required=True,
                         help="workflow filename, e.g. ci.yml")
    parser.add_argument("--job-name", required=True,
                         help="substring to match one job within each run, "
                              "e.g. 'Test provider packages' -- check the "
                              "Actions tab on GitHub for exact job names")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--output", default="data/raw_runs.jsonl")
    parser.add_argument("--debug-print-lines", type=int, default=0,
                         help="print the first N raw log lines from the "
                              "first job and exit, without writing output. "
                              "Use this FIRST to check the log format "
                              "matches the regexes before trusting a full run.")
    args = parser.parse_args()

    session = get_session()
    runs = list_workflow_runs(session, args.repo, args.workflows, args.max_runs)
    print(f"Found {len(runs)} completed runs.")

    if args.debug_print_lines:
        for run in runs:
            job_id = find_job_id(session, args.repo, run["id"], args.job_name)
            if job_id is None:
                continue
            log_text = download_job_log(session, args.repo, job_id)
            if not log_text:
                continue
            print(f"--- First {args.debug_print_lines} lines of job {job_id} "
                  f"(run {run['id']}) ---")
            for line in log_text.splitlines()[:args.debug_print_lines]:
                print(line)
            return
        print("No job matched --job-name in the runs checked. "
              "Double check the exact job name in the GitHub Actions UI.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_jobs_found = 0
    with open(args.output, "w") as f:
        for run in tqdm(runs, desc="Processing runs"):
            commit_sha = run["head_sha"]
            timestamp = run["created_at"]

            job_id = find_job_id(session, args.repo, run["id"], args.job_name)
            if job_id is None:
                continue
            log_text = download_job_log(session, args.repo, job_id)
            if not log_text:
                continue
            n_jobs_found += 1

            rows = parse_pytest_log(log_text, run["id"], commit_sha, timestamp)
            for row in rows:
                f.write(json.dumps(row) + "\n")
                n_rows += 1
            time.sleep(0.2)

    print(f"Matched job in {n_jobs_found}/{len(runs)} runs.")
    print(f"Wrote {n_rows} test-run rows to {args.output}")
    if n_rows == 0:
        print(
            "No rows were parsed -- this almost always means the regex "
            "patterns don't match this repo's log format. Rerun with "
            "--debug-print-lines 200 and inspect the output, then adjust "
            "RERUN_RE / FAILED_RE / PASSED_RE / INLINE_RE accordingly.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()