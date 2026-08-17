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
        "duration_sec": 0.42,
        "timestamp": "2026-08-01T12:00:00Z"
    }

Key idea for labeling flakiness (used later in features.py):
if the SAME test on the SAME commit fails on attempt 1 but passes on a
rerun (attempt 2+), that's strong evidence the test is flaky rather than
a real regression. Repos using the `pytest-rerunfailures` plugin (rerun
markers like "RERUN" in the JUnit XML / log output) make this easy to spot.

USAGE:
    python collect_data.py --repo pytest-dev/pytest --workflows ci.yml \
        --max-runs 500 --output data/raw_runs.jsonl

NOTE: This is a skeleton. The GitHub API calls for listing workflow runs
and downloading logs are implemented; the actual pytest-output parsing
(parse_pytest_log) is left as a TODO since log format varies by repo
(plain text vs. JUnit XML artifact vs. pytest-json-report). Start with a
repo that uploads a JUnit XML artifact -- it's far easier to parse
reliably than scraping raw log text.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

GITHUB_API = "https://api.github.com"


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


def get_run_attempts(session: requests.Session, repo: str, run: dict) -> list[int]:
    """A run can have multiple attempts if it was rerun. Returns attempt numbers."""
    return list(range(1, run.get("run_attempt", 1) + 1))


def download_junit_xml_for_attempt(session: requests.Session, repo: str,
                                    run_id: int, attempt: int) -> bytes | None:
    """
    TODO: implement artifact download + unzip for the JUnit XML report.
    Steps:
      1. GET /repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/artifacts
      2. Find the artifact matching your repo's test-report naming
         (e.g. "pytest-results", "junit.xml")
      3. GET the artifact's archive_download_url, unzip in memory
      4. Return the raw XML bytes
    Left unimplemented here because artifact naming is repo-specific --
    check the target repo's workflow YAML to see what it uploads.
    """
    raise NotImplementedError(
        "Implement artifact download for your target repo's JUnit XML output."
    )


def parse_pytest_log(xml_bytes: bytes, run_id: int, attempt: int,
                      commit_sha: str, timestamp: str) -> list[dict]:
    """
    TODO: parse JUnit XML (e.g. with xml.etree.ElementTree) into per-test
    rows matching the schema described in the module docstring.

    JUnit XML structure to expect:
        <testsuite>
          <testcase classname="..." name="..." time="0.42">
            <failure message="AssertionError: ...">traceback...</failure>
          </testcase>
          ...
        </testsuite>

    A <testcase> with no <failure>/<error> child = passed.
    """
    raise NotImplementedError("Implement JUnit XML parsing here.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="e.g. pytest-dev/pytest")
    parser.add_argument("--workflows", required=True,
                         help="workflow filename, e.g. ci.yml")
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--output", default="data/raw_runs.jsonl")
    args = parser.parse_args()

    session = get_session()
    runs = list_workflow_runs(session, args.repo, args.workflows, args.max_runs)
    print(f"Found {len(runs)} completed runs.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(args.output, "w") as f:
        for run in tqdm(runs, desc="Processing runs"):
            commit_sha = run["head_sha"]
            timestamp = run["created_at"]
            for attempt in get_run_attempts(session, args.repo, run):
                try:
                    xml_bytes = download_junit_xml_for_attempt(
                        session, args.repo, run["id"], attempt
                    )
                    if xml_bytes is None:
                        continue
                    rows = parse_pytest_log(
                        xml_bytes, run["id"], attempt, commit_sha, timestamp
                    )
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                        n_rows += 1
                except NotImplementedError:
                    # Skeleton mode -- surface this clearly instead of
                    # silently producing an empty dataset.
                    print(
                        "NotImplementedError: fill in download_junit_xml_for_attempt "
                        "and parse_pytest_log for your target repo, then rerun.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

    print(f"Wrote {n_rows} test-run rows to {args.output}")


if __name__ == "__main__":
    main()
