# Flaky Test Detector

A tool that analyzes pytest CI history and predicts whether a failing test is
**flaky** (fails intermittently, unrelated to the code change) or a **real
regression** (a genuine bug introduced by a commit).

## Problem

Flaky tests waste engineering time: developers re-run CI, second-guess their
own changes, and eventually start ignoring failures altogether ("just rerun
it"). This tool uses a repo's own test-run history to flag which failing
tests are statistically likely to be flaky, so teams can quarantine them
instead of chasing ghosts.

## How it works

1. **Collect** — pull historical pytest run results (pass/fail per test per
   commit) from a repo's GitHub Actions history, focusing on repos using
   `pytest-rerunfailures`, where a test that fails then passes on rerun
   (same commit) is a strong flaky signal.
2. **Featurize** — turn each test's history into a feature vector: failure
   rate, rerun-flip rate, error-message similarity across failures, timing
   variance, etc.
3. **Train** — fit a simple, explainable baseline model (logistic
   regression / gradient boosted trees) to classify tests as flaky vs. not.
4. **Predict** — a CLI tool that takes a repo's recent CI output and prints
   flakiness scores per failing test.

## Project structure

```
flaky-test-detector/
├── src/
│   ├── collect_data.py   # pulls test run history from GitHub Actions
│   ├── features.py       # turns raw run history into ML features
│   ├── train.py          # trains and evaluates the baseline model
│   └── predict_cli.py    # CLI: score a new set of failures
├── data/                 # raw + processed datasets (gitignored)
├── models/               # saved trained models (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GITHUB_TOKEN=your_personal_access_token   # needed for data collection
```

## Usage

```bash
# 1. Collect raw test run history for a repo
python src/collect_data.py --repo pytest-dev/pytest --workflows ci.yml --max-runs 500

# 2. Build features + labels from the raw data
python src/features.py --input data/raw_runs.jsonl --output data/features.csv

# 3. Train the baseline model
python src/train.py --input data/features.csv --model-out models/baseline.pkl

# 4. Score new failing tests
python src/predict_cli.py --model models/baseline.pkl --failures data/new_failures.jsonl
```

## Status

This is an early skeleton. See TODOs in each file for what's stubbed vs.
implemented. Next milestones: (1) validate rerun-flip labeling against a
real repo's known-flaky issue tags, (2) get precision/recall on a held-out
set of commits, (3) package as a GitHub Action.

## Results (fill in once you have them)

- Repo(s) tested against:
- Precision / Recall on held-out set:
- Example output:
