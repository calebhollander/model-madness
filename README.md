# Model Madness

End-to-end machine learning for NCAA March Madness: historical game data, feature engineering, calibrated models, and Kaggle-ready submissions for **P(Team A beats Team B)**.

## Results

Submitted to [**March Machine Learning Mania 2026**](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview) (Kaggle). **Rank 627 / 3,462** (~**top 18%**) on the public leaderboard (log loss).

## What’s in this repo

- **Pipeline code** — Load Kaggle CSVs, build team- and matchup-level features (difference design), time-based validation by season, training, calibration, ensembling, and `submission.csv` generation.
- **Models** — Logistic regression, XGBoost, blended and calibrated for probability outputs.
- **Notebooks** — EDA, feature work, modeling, and optional bracket simulation under `march-machine-learning-mania-2026/notebooks/`.

Design goals match the competition: **log loss**, **no leakage** (season-ordered splits, no random row splits), and **probabilities** clipped/calibrated for submission.

## Features and how they were chosen

Team stats come from **detailed** regular-season box scores (possessions estimate, Four Factors–style rates, efficiencies), rolled up to season means and to **last-10-games** form, plus **tournament seed**. Models consume **matchup differences** (Team A − Team B), not raw team vectors.

**Logistic regression** narrows the diff columns with **greedy forward selection** on time-based validation **log loss**: a six-feature baseline (seed, margin, win rate, off/def/net efficiency diffs) plus candidates tried in order; **ORB% diff** and **free-throw rate diff** were kept on a representative run, while eFG%, turnover, and last-10 net-efficiency diffs were dropped when they failed to improve validation loss. **XGBoost** uses the full candidate pool by default; at inference, logreg reads the saved subset from JSON while XGB uses the full column list.

Full write-up (definitions, rationale, logreg vs XGB): **[Feature set and selection](march-machine-learning-mania-2026/README.md#feature-set-and-selection)** in the competition project README.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Jupyter

## Repository layout

```
model-madness/
└── march-machine-learning-mania-2026/   # Main project (code, data dirs, notebooks)
    ├── src/                               # Pipeline modules
    ├── notebooks/                         # 01–04 workflow notebooks
    ├── data/                              # raw / interim / processed (see project README)
    ├── requirements.txt
    └── README.md                          # Detailed pipeline & runbook
```

## Getting started

Full setup, data expectations, run order, and outputs are documented here:

**[march-machine-learning-mania-2026/README.md](march-machine-learning-mania-2026/README.md)**

Short version:

```bash
cd march-machine-learning-mania-2026
pip install -r requirements.txt
```

Place competition CSVs under `data/raw/` (or adjust `src/config.py` as described in the project README).

## Data

Raw Kaggle competition data is **not** committed here. Download from the [competition Data tab](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and follow the paths in the subproject README.
