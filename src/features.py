"""
features.py

Turns raw per-test-run rows (from collect_data.py) into a feature table
suitable for training a classifier, plus labels.

LABELING STRATEGY (the ground truth we bootstrap from):
For a given (test_name, commit_sha), if the test FAILED on attempt 1 and
PASSED on a later attempt (same commit, same code) -- that's a rerun-flip,
and we label that failure event as flaky = 1.
If a test failed and there was no later passing attempt on that commit
(i.e. the failure "stuck"), we label it flaky = 0 (treated as a real
regression signal) for training purposes.

This is a heuristic, not ground truth -- note this limitation in your
writeup. A stronger validation set: cross-check against tests the repo's
maintainers have manually tagged as "known flaky" in GitHub issues
(search issues for "flaky" + test name).

FEATURES (per failure event, computed using only PRIOR history --
never leak future information):
- hist_failure_rate       : fraction of the test's last N runs that failed
- hist_rerun_flip_rate    : fraction of the test's past failures that were
                             themselves rerun-flips (flaky begets flaky)
- error_msg_similarity    : how similar this failure's error message is to
                             the test's other historical failure messages
                             (real bugs tend to repeat the same message;
                             flaky failures are often more varied -- timeouts,
                             connection errors, race conditions, etc.)
- duration_zscore         : how unusual this run's duration was vs. the
                             test's historical duration distribution
- n_prior_failures        : how many times this test has failed before
- days_since_last_failure : recency signal
"""

import argparse
import json
from collections import defaultdict

import pandas as pd
from difflib import SequenceMatcher


def load_raw_runs(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def label_rerun_flips(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (test_name, commit_sha) group, mark failures that were
    followed by a passing attempt on the same commit as flaky=1.
    Failures with no later pass on the same commit are flaky=0.
    """
    df = df.copy()
    df["flaky"] = 0

    grouped = df.groupby(["test_name", "commit_sha"])
    for (_test, _commit), group in grouped:
        group_sorted = group.sort_values("attempt")
        outcomes = group_sorted["outcome"].tolist()
        if "failed" in outcomes and "passed" in outcomes:
            # at least one failed attempt was followed by / mixed with a pass
            fail_idxs = group_sorted[group_sorted["outcome"] == "failed"].index
            df.loc[fail_idxs, "flaky"] = 1

    # Only failures are meaningful training examples (passes aren't "flaky or not")
    return df[df["outcome"] == "failed"].reset_index(drop=True)


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through failures in chronological order, computing features
    from each test's PRIOR history only (no leakage from future runs).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    history = defaultdict(list)  # test_name -> list of past run dicts
    feature_rows = []

    for _, row in df.iterrows():
        test_name = row["test_name"]
        past = history[test_name]
        past_failures = [r for r in past if r["outcome"] == "failed"]

        # hist_failure_rate: over last 20 runs (or fewer if history is short)
        recent = past[-20:]
        hist_failure_rate = (
            sum(1 for r in recent if r["outcome"] == "failed") / len(recent)
            if recent else 0.0
        )

        # hist_rerun_flip_rate: of past FAILURES, how many were themselves flaky
        hist_rerun_flip_rate = (
            sum(1 for r in past_failures if r.get("flaky") == 1) / len(past_failures)
            if past_failures else 0.0
        )

        # error message similarity vs. past failure messages (max similarity)
        error_msg_similarity = 0.0
        if past_failures and row.get("error_message"):
            sims = [
                text_similarity(row["error_message"], r.get("error_message", ""))
                for r in past_failures
            ]
            error_msg_similarity = max(sims) if sims else 0.0

        # duration z-score vs. historical durations for this test
        durations = [r["duration_sec"] for r in past if r.get("duration_sec") is not None]
        if len(durations) >= 2:
            mean_d = sum(durations) / len(durations)
            std_d = (sum((d - mean_d) ** 2 for d in durations) / len(durations)) ** 0.5
            duration_zscore = (
                (row["duration_sec"] - mean_d) / std_d if std_d > 0 else 0.0
            )
        else:
            duration_zscore = 0.0

        n_prior_failures = len(past_failures)

        days_since_last_failure = (
            (row["timestamp"] - past_failures[-1]["timestamp"]).days
            if past_failures else -1  # -1 = "no prior failure"
        )

        feature_rows.append({
            "test_name": test_name,
            "commit_sha": row["commit_sha"],
            "timestamp": row["timestamp"],
            "hist_failure_rate": hist_failure_rate,
            "hist_rerun_flip_rate": hist_rerun_flip_rate,
            "error_msg_similarity": error_msg_similarity,
            "duration_zscore": duration_zscore,
            "n_prior_failures": n_prior_failures,
            "days_since_last_failure": days_since_last_failure,
            "flaky": row["flaky"],
        })

        history[test_name].append(row.to_dict())

    return pd.DataFrame(feature_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw_runs.jsonl from collect_data.py")
    parser.add_argument("--output", required=True, help="path to write features.csv")
    args = parser.parse_args()

    df = load_raw_runs(args.input)
    df = label_rerun_flips(df)
    print(f"Loaded {len(df)} failure events.")
    print(f"Flaky label distribution:\n{df['flaky'].value_counts()}")

    features = build_features(df)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
