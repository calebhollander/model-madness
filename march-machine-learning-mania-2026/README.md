# March Machine Learning Mania 2026

Production-quality ML pipeline for predicting NCAA March Madness game outcomes in the Kaggle competition of the same name.

## Project overview

This repository implements an end-to-end pipeline that trains on historical NCAA basketball data and predicts the probability that Team A beats Team B in tournament matchups. The system is optimized for **log loss** (the competition metric), produces a valid Kaggle submission file, and optionally simulates the tournament bracket.

## Problem definition

- **Task:** For each (Season, Team A, Team B) in the tournament, predict **P(Team A beats Team B)**.
- **Data:** Kaggle provides Men’s and Women’s data: teams, seasons, regular-season and tournament game results, seeds, bracket slots, and optional supplementary files (Massey ordinals, spellings, etc.).
- **Goal:** Minimize log loss on held-out tournament outcomes; submit a CSV of IDs and probabilities for the competition’s test set.

## Competition format

- **Submission file:** CSV with columns `ID` and `Pred`.
  - `ID`: `Season_TeamID1_TeamID2` (e.g. `2026_1101_1102`). Order is fixed by the competition.
  - `Pred`: probability that the first team (TeamID1) wins.
- **Evaluation:** Log loss. Predictions must be probabilities in (0, 1); clipping extreme values is recommended to avoid severe penalties.

## Pipeline steps

1. **Data loading** — Read Kaggle CSVs into clean DataFrames (`src/data_loading.py`).
2. **Feature engineering** — Build season-level team statistics from regular-season (and optionally other) data (`src/features.py`).
3. **Matchup generation** — Convert team-level features into pairwise rows with **feature differences** (Team A stat minus Team B stat) and target (`src/matchup_builder.py`).
4. **Model training** — Train logistic regression and XGBoost on matchup diffs (`src/train_logreg.py`, `src/train_xgb.py`). An Elo-style ratings module exists as a stub (`src/ratings.py`) for future blending.
5. **Validation** — Time-based cross-validation by season (train on past seasons, validate on a single future season); never use a random split (`src/validation.py`).
6. **Calibration and ensembling** — Calibrate probabilities and blend model outputs (`src/calibrate.py`, `src/ensemble.py`).
7. **Submission generation** — Read the sample submission, generate predictions, clip probabilities, and write `submission.csv` (`src/submit.py`).
8. **Optional: Bracket simulation** — Simulate the tournament bracket from predicted probabilities (`src/simulate.py`).

## Feature engineering philosophy

- **Team-level first:** One row per (Season, TeamID) with aggregate stats (e.g. win rate, points per game, points allowed, efficiency, seed rank).
- **Matchup-level by differences:** For each (Season, Team A, Team B), use **differences** (A − B) of those stats as model inputs. This keeps features comparable and reduces leakage.
- **No future data:** Use only information available before the game (e.g. regular-season games for that season only; no tournament games in the training target for that season).

## Feature set and selection

### What gets built from the raw data

Features start from **detailed** regular-season results (not compact boxes), reshaped to one row per team per game (`src/game_results.py`, `src/features.py`). Per-game helpers use standard basketball analytics conventions:

- **Possessions** — Approximated from FGA, offensive rebounds, turnovers, and FTA (0.475 weight), averaged with the opponent’s possession estimate for that game.
- **Shooting and ball security** — Effective FG% (counts made threes), turnover rate (TO per possession), offensive rebound rate (OR vs opponent DR), free-throw rate (FTA / FGA).
- **Efficiency** — Offensive / defensive / net efficiency (points per 100 estimated possessions).

Those per-game fields are aggregated to **one row per (Season, TeamID)** in two horizons:

1. **Season-long means** — Games played, win rate, average scoring margin, average points for/against, mean off/def/net efficiency, and mean eFG%, TOV%, ORB%, FTR.
2. **Recent form** — Same style of stats over the **last *N* regular-season games** (default *N* = 10) before the aggregation step: recent win rate, margin, and efficiencies (`get_recent_features` in `src/features.py`).

**Tournament seed** is merged in as a numeric seed (`seed_num`) from the competition seed string.

For modeling, **only matchup-level inputs** are used: for each ordered pair (Team A, Team B), every numeric team stat becomes **Team A − Team B** (`src/matchup_builder.py`). That is the full diff feature pool (e.g. `seed_diff`, `win_pct_diff`, efficiency diffs, Four Factors–style diffs, and recent-form diffs). The constant `DIFF_FEATURE_COLUMNS` lists every diff column the pipeline can emit.

### How the set was narrowed (logistic regression)

Logistic regression uses **greedy forward selection on validation log loss**, implemented in `train_logreg()` (`src/train_logreg.py`), on the same **time-ordered train/validation split** as the rest of the pipeline (`split_matchup_train_val` in `src/splits.py`—past seasons for training, a held-out season for validation—never a random row split).

1. **Baseline block (always included first)** — Strong, interpretable signals that bracket committee and season performance already summarize: `seed_diff`, `avg_margin_diff`, `off_eff_diff`, `def_eff_diff`, `net_eff_diff`, `win_pct_diff`.
2. **Candidate block (tried in a fixed order)** — Each candidate is **appended to the working feature set** (baseline plus any candidates already kept earlier in the list); it **stays** only if validation log loss improves versus the best score seen so far:
   - `efg_pct_diff`, `tov_pct_diff`, `orb_pct_diff`, `ftr_diff` (Four Factors–style differences at season level).
   - `last10_net_eff_diff` (recent net efficiency gap; equivalent to `lastx_net_eff_diff` when the recent window is 10 games).

On a representative training run (see `notebooks/03_modeling.ipynb`), that procedure **rejected** eFG and TOV diffs (they did not beat validation loss when added after the baseline), **kept** ORB% and FTR diffs (they helped), and **rejected** the last-10 net efficiency diff (it hurt—likely redundant with season net efficiency and margin). The resulting **eight-feature** logistic model is what gets saved to `models/logreg_*_features.json` for inference.

This is intentionally **simple and loss-driven**: no separate filter for correlation; the split decides whether a candidate earns its parameters under log loss.

### XGBoost and inference

**XGBoost** is trained on the **full default column list** in `src/train_logreg.py`—baseline plus **all** candidate diffs (`FEATURES`), i.e. a wider set than the greedily selected logistic subset. Gain-based importances (printed in training and optionally `outputs/feature_importance.csv`) typically rank `seed_diff`, margin, and efficiency diffs highly, with secondary mass on the Four Factors and recent-form diffs.

At Stage 2 inference (`src/predict_2026.py`), **logistic regression** uses the **saved feature order** from JSON (the selected subset), while **XGBoost** uses the **full `FEATURES` list**, so the two models do not always consume identical columns even though they share the same underlying team feature table.

## Modeling approach

- **Logistic regression:** Simple baseline on feature differences; outputs probabilities.
- **XGBoost:** Gradient boosting on the same features; tune with time-based CV; keep complexity moderate (e.g. shallow trees) given dataset size.
- **Elo-style ratings:** Scaffold only in `src/ratings.py` (not used in the default submission path).
- **Ensemble:** Default submission blends **logistic regression and XGBoost** probabilities (equal or weighted), then clips; see `src/submit.py` and `src/ensemble.py`.

## Evaluation

- **Metric:** Log loss (Kaggle’s metric).
- **Validation:** Time-based cross-validation by season (e.g. train on seasons &lt; Y, validate on season Y). Never use a random train/val split to avoid leakage and overoptimistic scores.
- **Best practices:** Always predict probabilities; calibrate and clip extremes (e.g. to [1e-15, 1−1e-15]) before submission.

## How to run the pipeline

1. **Environment**
   ```bash
   cd march-machine-learning-mania-2026
   pip install -r requirements.txt
   ```

2. **Data**
   - Place Kaggle data in `data/raw/` (or keep CSVs in the project root and set `DATA_RAW = PROJECT_ROOT` in `src/config.py`).

3. **Execution order**
   - Load data → build team stats → build matchup dataset (with feature diffs).
   - Run time-based CV and train models (logreg, XGB).
   - Calibrate and ensemble → generate predictions for the sample submission → write `submission.csv`.
   - Optionally run bracket simulation and write results to `outputs/simulation_results.csv`.

4. **Outputs**
   - `models/` — Saved models (e.g. joblib/pickle).
   - `outputs/cv_scores.txt` — Cross-validation log loss per fold.
   - `outputs/feature_importance.csv` — XGBoost feature importance.
   - `outputs/simulation_results.csv` — Optional simulation summary.
   - `submission.csv` — Final Kaggle submission (e.g. in project root or `outputs/`).

## Rules and best practices (enforced in code and docs)

- **Always** use time-based validation (by season); never a random split.
- **Always** predict probabilities, not just winners.
- **Always** use feature **differences** (Team A − Team B) for matchup-level models.
- **Avoid data leakage:** no future games when building features or training.
- **Keep models relatively simple** given the small dataset size.
- **Calibrate** probabilities before submission when possible.
- **Clip** extreme predictions to avoid log loss penalties.

## Future improvements

- Richer features (e.g. Massey ordinals, conference strength, recent form).
- Neural networks or other models with careful regularization.
- External data (injuries, travel, etc.) if allowed.
- Stacking or meta-learning across seasons.
- Separate Men’s vs Women’s pipelines or shared feature design.

## Repository structure

```
march-machine-learning-mania-2026/
├── data/
│   ├── raw/          # Kaggle CSVs (or point config here)
│   ├── interim/      # Feature tables, intermediate data
│   └── processed/    # Final training sets, submission-ready data
├── notebooks/        # 01_eda, 02_feature_engineering, 03_modeling, 04_simulation
├── src/              # Pipeline modules (config, data_loading, features, ...)
├── models/           # Saved models
├── outputs/          # cv_scores, feature_importance, simulation_results
├── requirements.txt
└── README.md
```
