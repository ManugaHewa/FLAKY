"""
predict_cli.py

Given a trained model and a set of NEW failing tests (already run through
features.py to get the same feature columns used at training time), prints
a flakiness score per test.

USAGE:
    python predict_cli.py --model models/baseline.pkl --failures data/new_failures.csv

Input format for --failures: a CSV with a "test_name" column plus the same
feature columns produced by features.py (hist_failure_rate,
hist_rerun_flip_rate, error_msg_similarity, duration_zscore,
n_prior_failures, days_since_last_failure).

In a real CI integration, this step would run automatically: a GitHub
Action triggers on test failure, computes fresh features for the failing
test using the repo's run history, and calls this script (or an equivalent
function) to decide whether to auto-quarantine the test or alert a human.
"""

import argparse
import pickle
import sys

import click
import pandas as pd


@click.command()
@click.option("--model", "model_path", required=True, help="Path to trained model .pkl")
@click.option("--failures", "failures_path", required=True,
              help="CSV of new failures with feature columns")
@click.option("--threshold", default=0.5, help="Probability threshold for flagging as flaky")
def main(model_path, failures_path, threshold):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(failures_path)
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"ERROR: input file is missing required feature columns: {missing}",
              file=sys.stderr)
        print("Run features.py on your new failure data first.", file=sys.stderr)
        sys.exit(1)

    probs = model.predict_proba(df[feature_columns])[:, 1]  # P(flaky)
    df["flaky_probability"] = probs
    df["verdict"] = df["flaky_probability"].apply(
        lambda p: "LIKELY FLAKY" if p >= threshold else "likely real bug"
    )

    results = df[["test_name", "flaky_probability", "verdict"]].sort_values(
        "flaky_probability", ascending=False
    )

    print(f"\n{'Test':60s} {'P(flaky)':>10s}  Verdict")
    print("-" * 90)
    for _, row in results.iterrows():
        marker = "\u26a0\ufe0f " if row["verdict"] == "LIKELY FLAKY" else "   "
        print(f"{marker}{row['test_name']:57s} {row['flaky_probability']:>9.1%}  {row['verdict']}")

    n_flagged = (results["verdict"] == "LIKELY FLAKY").sum()
    print(f"\n{n_flagged} of {len(results)} failing test(s) flagged as likely flaky "
          f"(threshold={threshold}).")


if __name__ == "__main__":
    main()
