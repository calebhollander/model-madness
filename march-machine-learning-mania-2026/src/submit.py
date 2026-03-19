"""
Kaggle Stage 2 submission generator.

Builds matchup feature differences for each `SampleSubmissionStage2.csv` ID,
loads your trained logistic regression + XGBoost models, ensembles their
probabilities, and writes a submission CSV with columns `ID` and `Pred`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src import config
from src.ensemble import ensemble_predict
from src.matchup_builder import DIFF_FEATURE_COLUMNS, build_submission_matchups as _build_matchups
from src.train_logreg import FEATURES as XGB_FEATURE_ORDER


# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------

LOGREG_WEIGHT_DEFAULT = 0.5
XGB_WEIGHT_DEFAULT = 0.5

DEFAULT_OUTPUT_PATH = config.DATA_PROCESSED / "submission.csv"

# Team-level columns required to build matchup diff features.
REQUIRED_TEAM_FEATURE_COLUMNS: Sequence[str] = (
    "Season",
    "TeamID",
    "seed_num",
    "win_pct",
    "avg_margin",
    "off_eff",
    "def_eff",
    "net_eff",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "ftr",
    "lastx_win_pct",
    "lastx_avg_margin",
    "lastx_off_eff",
    "lastx_def_eff",
    "lastx_net_eff",
)


def parse_submission_ids(sample_df: pd.DataFrame, id_column: str = "ID") -> pd.DataFrame:
    """
    Parse `Season_TeamA_TeamB` IDs into separate integer columns.

    Parameters
    ----------
    sample_df:
        DataFrame loaded from `SampleSubmissionStage2.csv`.
    id_column:
        Name of the submission ID column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: `[id_column, Season, TeamA, TeamB]`.
    """
    if id_column not in sample_df.columns:
        raise KeyError(f"Missing column {id_column!r} in sample submission")

    ids = sample_df[id_column].astype(str)
    parts = ids.str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(f"Expected ID format Season_TeamA_TeamB in column {id_column!r}")

    return pd.DataFrame(
        {
            id_column: sample_df[id_column].values,
            "Season": parts.iloc[:, 0].astype(int).values,
            "TeamA": parts.iloc[:, 1].astype(int).values,
            "TeamB": parts.iloc[:, 2].astype(int).values,
        }
    )


def prepare_team_feature_views(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and lightly normalize the saved team feature table.

    Returns a copy so inference doesn't mutate the loaded CSV DataFrame.
    """
    missing = [c for c in REQUIRED_TEAM_FEATURE_COLUMNS if c not in team_stats_df.columns]
    if missing:
        raise ValueError(
            "team_stats_df is missing required columns for matchup feature diffs: "
            + ", ".join(missing)
        )

    df = team_stats_df.copy()
    # Keep dtypes stable for merges with integer TeamIDs.
    df["Season"] = df["Season"].astype(int)
    df["TeamID"] = df["TeamID"].astype(int)
    # Ensure seed_num is numeric for diffs.
    df["seed_num"] = pd.to_numeric(df["seed_num"], errors="coerce")
    return df


def build_submission_matchups(
    ids_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    *,
    feature_columns: Optional[list[str]] = None,
    id_column: str = "ID",
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build submission matchup rows with diff features.

    Uses the same diff logic as training (`src.matchup_builder`):
    - merge TeamA and TeamB features by `Season` + TeamID
    - compute `TeamA - TeamB` diffs for the known diff columns
    """
    if id_column not in ids_df.columns:
        raise KeyError(f"ids_df is missing column {id_column!r}")

    sample_ids_df = ids_df[[id_column]].copy()
    matchup_df = _build_matchups(
        sample_ids_df,
        team_stats_df,
        feature_columns=feature_columns,
        id_column=id_column,
        season=season,
    )

    missing_diff = [c for c in DIFF_FEATURE_COLUMNS if c not in matchup_df.columns]
    if missing_diff:
        raise ValueError(f"Built matchup_df is missing expected diff columns: {missing_diff}")

    return matchup_df


def load_logreg_model(
    logreg_model_path: Path,
    logreg_features_path: Path,
) -> Tuple[object, list[str]]:
    """
    Load a saved sklearn logistic regression model and its ordered feature list.
    """
    features_list = json.loads(logreg_features_path.read_text(encoding="utf-8"))
    model = joblib.load(logreg_model_path)
    return model, list(features_list)


def load_xgb_model(xgb_model_path: Path) -> object:
    """
    Load a saved XGBoost model.

    Preference order:
    - `xgb_men.pkl` via joblib (most reliable for predict_proba in this repo)
    - `xgb_men.json` via xgboost's `load_model` fallback
    """
    xgb_model_path = Path(xgb_model_path)
    if not xgb_model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {xgb_model_path}")

    suffix = xgb_model_path.suffix.lower()
    if suffix in {".pkl", ".joblib"}:
        return joblib.load(xgb_model_path)

    if suffix == ".json":
        import xgboost as xgb  # local import: only needed for json fallback

        model = xgb.XGBClassifier()
        model.load_model(str(xgb_model_path))
        return model

    # Best-effort fallback (in case you stored a pickled estimator with an odd suffix).
    return joblib.load(xgb_model_path)


def predict_submission(
    matchup_df: pd.DataFrame,
    *,
    logreg_model: object,
    logreg_features: list[str],
    xgb_model: object,
    xgb_feature_order: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    """
    Predict final probabilities for each matchup row.
    """
    weights = list(weights)
    if len(weights) != 2:
        raise ValueError("weights must be [logreg_weight, xgb_weight]")

    missing_lr = [c for c in logreg_features if c not in matchup_df.columns]
    missing_xgb = [c for c in xgb_feature_order if c not in matchup_df.columns]
    if missing_lr:
        raise ValueError(f"matchup_df is missing logistic regression feature columns: {missing_lr}")
    if missing_xgb:
        raise ValueError(f"matchup_df is missing XGBoost feature columns: {missing_xgb}")

    X_lr = matchup_df[logreg_features]
    X_xgb = matchup_df[list(xgb_feature_order)]

    if X_lr.isna().any().any() or X_xgb.isna().any().any():
        n_lr = int(X_lr.isna().any(axis=1).sum())
        n_xgb = int(X_xgb.isna().any(axis=1).sum())
        raise ValueError(
            f"NaNs in feature matrix (logreg rows with any NaN: {n_lr}, xgb rows with any NaN: {n_xgb}). "
            "Check team_stats coverage."
        )

    pred_lr = logreg_model.predict_proba(X_lr)[:, 1].astype(np.float64, copy=False)
    pred_xgb = xgb_model.predict_proba(X_xgb)[:, 1].astype(np.float64, copy=False)

    blended = ensemble_predict([pred_lr, pred_xgb], weights=weights)
    return np.clip(blended, config.SUBMISSION_PROB_MIN, config.SUBMISSION_PROB_MAX)


def write_submission(
    matchup_ids: pd.Series,
    preds: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Write submission CSV with exactly columns: `ID`, `Pred`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({"ID": matchup_ids.values, "Pred": preds})
    out.to_csv(output_path, index=False)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="March Machine Learning Mania 2026 - Kaggle Stage 2 submission")
    parser.add_argument("--nrows", type=int, default=None, help="Only process first N rows of sample submission")
    parser.add_argument("--sample", type=Path, default=config.SAMPLE_SUBMISSION_STAGE2, help="SampleSubmissionStage2.csv path")
    parser.add_argument(
        "--team-stats",
        type=Path,
        default=config.DATA_INTERIM / "men_team_features.csv",
        help="Saved team feature table (men_team_features.csv) path",
    )
    parser.add_argument("--logreg-model", type=Path, default=config.MODELS_DIR / "logreg_men.pkl")
    parser.add_argument("--logreg-features", type=Path, default=config.LOGREG_FEATURE_NAMES_JSON)
    parser.add_argument("--xgb-model", type=Path, default=config.MODELS_DIR / "xgb_men.pkl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--season", type=int, default=None, help="Override season inferred from IDs")
    parser.add_argument("--logreg-weight", type=float, default=LOGREG_WEIGHT_DEFAULT)
    parser.add_argument("--xgb-weight", type=float, default=XGB_WEIGHT_DEFAULT)

    args = parser.parse_args(argv)

    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()

    sample_df = pd.read_csv(args.sample, nrows=args.nrows)
    ids_df = parse_submission_ids(sample_df, id_column="ID")

    seasons = sorted(ids_df["Season"].unique().tolist())
    if args.season is not None:
        use_season = int(args.season)
        if seasons != [use_season]:
            raise ValueError(f"Provided --season={use_season} but IDs contain seasons {seasons}")
    else:
        if len(seasons) != 1:
            raise ValueError(f"Multiple seasons in sample IDs {seasons}; pass --season to disambiguate")
        use_season = int(seasons[0])

    team_stats_df = pd.read_csv(args.team_stats)
    team_stats_df = prepare_team_feature_views(team_stats_df)

    matchup_df = build_submission_matchups(ids_df, team_stats_df, feature_columns=None, id_column="ID", season=use_season)

    logreg_model, logreg_features = load_logreg_model(args.logreg_model, args.logreg_features)
    xgb_model = load_xgb_model(args.xgb_model)

    preds = predict_submission(
        matchup_df,
        logreg_model=logreg_model,
        logreg_features=logreg_features,
        xgb_model=xgb_model,
        xgb_feature_order=XGB_FEATURE_ORDER,
        weights=[args.logreg_weight, args.xgb_weight],
    )

    out_path = write_submission(matchup_df["ID"], preds, args.output)
    print(f"Wrote {out_path} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
