"""
train.py

Trains a baseline classifier on the feature table produced by features.py,
evaluates it with precision/recall/F1 (not just accuracy -- flaky tests
are usually a minority class, so accuracy alone is misleading), and saves
the model + feature column list to disk.

Two model options:
  --model logreg   : Logistic Regression (fast, interpretable coefficients)
  --model xgboost   : Gradient boosted trees (usually higher recall)

USAGE:
    python train.py --input data/features.csv --model-out models/baseline.pkl \
        --model logreg
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

FEATURE_COLUMNS = [
    "hist_failure_rate",
    "hist_rerun_flip_rate",
    "error_msg_similarity",
    "duration_zscore",
    "n_prior_failures",
    "days_since_last_failure",
]


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    IMPORTANT: split by TIME, not randomly. Randomly splitting would let
    the model "see the future" (train on a later run of a test, test on
    an earlier one), which inflates your reported performance and won't
    hold up under scrutiny in an interview. Train on the earliest 80% of
    events chronologically, test on the most recent 20%.
    """
    cutoff = int(len(df) * (1 - test_frac))
    return df.iloc[:cutoff], df.iloc[cutoff:]


def train_logreg(X_train, y_train):
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = load_features(args.input)
    if len(df) < 20:
        print(
            f"WARNING: only {len(df)} labeled examples. This is too small "
            "for a meaningful train/test split -- collect more history "
            "(more repos, more runs, or a longer time window) before "
            "trusting these numbers."
        )

    train_df, test_df = time_based_split(df, args.test_frac)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["flaky"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["flaky"]

    if args.model == "logreg":
        model = train_logreg(X_train, y_train)
    else:
        model = train_xgboost(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Evaluation (held-out, most recent 20% of events) ===")
    print(classification_report(y_test, y_pred, target_names=["real_bug", "flaky"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nPrecision (flaky): {precision_score(y_test, y_pred):.3f}")
    print(f"Recall (flaky):    {recall_score(y_test, y_pred):.3f}")
    print(f"F1 (flaky):        {f1_score(y_test, y_pred):.3f}")

    if args.model == "logreg":
        print("\nFeature coefficients (higher = more predictive of 'flaky'):")
        for name, coef in sorted(
            zip(FEATURE_COLUMNS, model.coef_[0]), key=lambda x: -abs(x[1])
        ):
            print(f"  {name:28s} {coef:+.3f}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, f)
    print(f"\nSaved model to {args.model_out}")


if __name__ == "__main__":
    main()
