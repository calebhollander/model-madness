"""
Kaggle Stage 2 submission pipeline for men + women.

This module can:
1) Build women/men artifacts via the same processing+training pipeline.
2) Generate a combined Stage 2 `submission.csv` with both men and women IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src import config
from src.ensemble import ensemble_predict
from src.features import get_all_team_features
from src.matchup_builder import DIFF_FEATURE_COLUMNS, build_matchup_data, build_submission_matchups as _build_matchups
from src.train_logreg import FEATURES as SHARED_FEATURE_ORDER
from src.train_logreg import save_logreg_artifacts, train_logreg
from src.train_xgb import train_xgb


LOGREG_WEIGHT_DEFAULT = 0.5
XGB_WEIGHT_DEFAULT = 0.5
DEFAULT_OUTPUT_PATH = config.DATA_PROCESSED / "submission.csv"

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
    """Parse IDs like `2026_1101_1102` into `Season`, `TeamA`, `TeamB`."""
    if id_column not in sample_df.columns:
        raise KeyError(f"Missing column {id_column!r} in sample submission")
    ids = sample_df[id_column].astype(str)
    parts = ids.str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(f"Expected ID format Season_TeamA_TeamB in column {id_column!r}")
    out = pd.DataFrame(
        {
            id_column: sample_df[id_column].values,
            "Season": parts.iloc[:, 0].astype(int).values,
            "TeamA": parts.iloc[:, 1].astype(int).values,
            "TeamB": parts.iloc[:, 2].astype(int).values,
        }
    )
    return out


def validate_submission_id_order(ids_df: pd.DataFrame) -> None:
    """Ensure TeamA < TeamB so Pred is P(first team wins) in Kaggle-required order."""
    bad = ids_df["TeamA"] >= ids_df["TeamB"]
    if bad.any():
        examples = ids_df.loc[bad, "ID"].head(5).tolist()
        raise ValueError(
            "Submission IDs must have TeamA < TeamB (lower TeamID first). "
            f"Found invalid IDs, e.g. {examples}"
        )


def prepare_team_feature_views(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Validate required team-level columns and normalize merge dtypes."""
    missing = [c for c in REQUIRED_TEAM_FEATURE_COLUMNS if c not in team_stats_df.columns]
    if missing:
        raise ValueError("Missing required team feature columns: " + ", ".join(missing))
    df = team_stats_df.copy()
    df["Season"] = df["Season"].astype(int)
    df["TeamID"] = df["TeamID"].astype(int)
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
    """Build submission matchup rows using training-identical diff-feature logic."""
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
) -> Tuple[Optional[object], list[str]]:
    """
    Load saved logistic model + feature order.

    Falls back to shared training feature list if the feature JSON is missing.
    """
    model: Optional[object] = None
    if logreg_model_path.exists():
        model = joblib.load(logreg_model_path)

    if logreg_features_path.exists():
        features_list = json.loads(logreg_features_path.read_text(encoding="utf-8"))
    else:
        features_list = list(SHARED_FEATURE_ORDER)
    return model, list(features_list)


def load_xgb_model(xgb_model_path: Path) -> Optional[object]:
    """
    Load a saved XGBoost model from `.pkl/.joblib` or `.json`.
    """
    xgb_model_path = Path(xgb_model_path)
    if not xgb_model_path.exists():
        return None
    suffix = xgb_model_path.suffix.lower()
    if suffix in {".pkl", ".joblib"}:
        return joblib.load(xgb_model_path)
    if suffix == ".json":
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(xgb_model_path))
        return model
    return joblib.load(xgb_model_path)


def predict_submission(
    matchup_df: pd.DataFrame,
    *,
    logreg_model: Optional[object],
    logreg_features: Sequence[str],
    xgb_model: Optional[object],
    xgb_feature_order: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    """Predict probabilities and blend enabled models."""
    if len(weights) != 2:
        raise ValueError("weights must be [logreg_weight, xgb_weight]")
    logreg_weight, xgb_weight = float(weights[0]), float(weights[1])

    pred_list: list[np.ndarray] = []
    used_weights: list[float] = []

    if logreg_model is not None and logreg_weight > 0:
        missing_lr = [c for c in logreg_features if c not in matchup_df.columns]
        if missing_lr:
            raise ValueError(f"Missing logistic feature columns: {missing_lr}")
        X_lr = matchup_df[list(logreg_features)]
        pred_lr = logreg_model.predict_proba(X_lr)[:, 1].astype(np.float64, copy=False)
        pred_list.append(pred_lr)
        used_weights.append(logreg_weight)

    if xgb_model is not None and xgb_weight > 0:
        missing_xgb = [c for c in xgb_feature_order if c not in matchup_df.columns]
        if missing_xgb:
            raise ValueError(f"Missing XGBoost feature columns: {missing_xgb}")
        X_xgb = matchup_df[list(xgb_feature_order)]
        pred_xgb = xgb_model.predict_proba(X_xgb)[:, 1].astype(np.float64, copy=False)
        pred_list.append(pred_xgb)
        used_weights.append(xgb_weight)

    if not pred_list:
        raise ValueError("No active model predictions: provide at least one model with weight > 0")

    blended = ensemble_predict(pred_list, weights=used_weights)
    return np.clip(blended, config.SUBMISSION_PROB_MIN, config.SUBMISSION_PROB_MAX)


def write_submission(matchup_ids: pd.Series, preds: np.ndarray, output_path: Path) -> Path:
    """Write final Kaggle CSV with exactly `ID,Pred`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": matchup_ids.values, "Pred": preds}).to_csv(output_path, index=False)
    return output_path


def _presence_key_set(team_stats_df: pd.DataFrame) -> set[tuple[int, int]]:
    """Return available `(Season, TeamID)` keys for division assignment."""
    return set(
        team_stats_df[["Season", "TeamID"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )


def split_ids_by_division(
    ids_df: pd.DataFrame,
    men_team_stats: pd.DataFrame,
    women_team_stats: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Split sample IDs into men/women partitions by team coverage in feature tables.
    """
    men_keys = _presence_key_set(men_team_stats)
    women_keys = _presence_key_set(women_team_stats)

    men_mask = ids_df.apply(
        lambda r: (int(r["Season"]), int(r["TeamA"])) in men_keys and (int(r["Season"]), int(r["TeamB"])) in men_keys,
        axis=1,
    )
    women_mask = ids_df.apply(
        lambda r: (int(r["Season"]), int(r["TeamA"])) in women_keys and (int(r["Season"]), int(r["TeamB"])) in women_keys,
        axis=1,
    )

    both = men_mask & women_mask
    neither = ~men_mask & ~women_mask
    if both.any():
        examples = ids_df.loc[both, "ID"].head(5).tolist()
        raise ValueError(f"Ambiguous division assignment for IDs: {examples}")
    if neither.any():
        examples = ids_df.loc[neither, "ID"].head(5).tolist()
        raise ValueError(f"Could not assign IDs to men/women from team feature coverage: {examples}")

    return {
        "men": ids_df.loc[men_mask].copy(),
        "women": ids_df.loc[women_mask].copy(),
    }


def prepare_division_artifacts(
    division: str,
    *,
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
) -> None:
    """Run feature -> matchup -> logreg/xgb training pipeline and save artifacts."""
    team_features = get_all_team_features(
        season_min=season_min,
        season_max=season_max,
        num_games=10,
        division=division,
    )
    team_features_path = config.DATA_INTERIM / f"{division}_team_features.csv"
    team_features_path.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(team_features_path, index=False)

    matchups = build_matchup_data(
        season_min=season_min,
        season_max=season_max,
        division=division,
    )
    matchup_path = config.DATA_INTERIM / f"{division}_train_matchups.csv"
    matchup_path.parent.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(matchup_path, index=False)

    lr_model, lr_features = train_logreg(division=division)
    save_logreg_artifacts(lr_model, lr_features, division=division)
    train_xgb(division=division, save_model=True)

    print(f"Prepared {division} artifacts:")
    print(f"  - {team_features_path}")
    print(f"  - {matchup_path}")
    print(f"  - {config.MODELS_DIR / f'logreg_{division}.pkl'}")
    print(f"  - {config.MODELS_DIR / f'logreg_{division}_features.json'}")
    print(f"  - {config.MODELS_DIR / f'xgb_{division}.pkl'}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="March Machine Learning Mania 2026 - combined men+women Stage 2 submission")
    parser.add_argument("--nrows", type=int, default=None, help="Only process first N rows from sample submission")
    parser.add_argument("--sample", type=Path, default=config.SAMPLE_SUBMISSION_STAGE2, help="SampleSubmissionStage2.csv path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Final output path for ID,Pred")
    parser.add_argument("--season", type=int, default=None, help="Override season inferred from IDs")
    parser.add_argument("--logreg-weight", type=float, default=LOGREG_WEIGHT_DEFAULT)
    parser.add_argument("--xgb-weight", type=float, default=XGB_WEIGHT_DEFAULT)

    parser.add_argument("--men-team-stats", type=Path, default=config.DATA_INTERIM / "men_team_features.csv")
    parser.add_argument("--women-team-stats", type=Path, default=config.DATA_INTERIM / "women_team_features.csv")
    parser.add_argument("--men-logreg-model", type=Path, default=config.MODELS_DIR / "logreg_men.pkl")
    parser.add_argument("--women-logreg-model", type=Path, default=config.MODELS_DIR / "logreg_women.pkl")
    parser.add_argument("--men-logreg-features", type=Path, default=config.MODELS_DIR / "logreg_men_features.json")
    parser.add_argument("--women-logreg-features", type=Path, default=config.MODELS_DIR / "logreg_women_features.json")
    parser.add_argument("--men-xgb-model", type=Path, default=config.MODELS_DIR / "xgb_men.pkl")
    parser.add_argument("--women-xgb-model", type=Path, default=config.MODELS_DIR / "xgb_women.pkl")

    parser.add_argument(
        "--prepare-division",
        type=str,
        default="none",
        choices=["none", "men", "women", "all"],
        help="Optionally build artifacts before prediction.",
    )
    parser.add_argument("--season-min", type=int, default=None, help="Optional lower bound for training prep")
    parser.add_argument("--season-max", type=int, default=None, help="Optional upper bound for training prep")
    args = parser.parse_args(argv)

    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()

    if args.prepare_division in {"men", "all"}:
        prepare_division_artifacts("men", season_min=args.season_min, season_max=args.season_max)
    if args.prepare_division in {"women", "all"}:
        prepare_division_artifacts("women", season_min=args.season_min, season_max=args.season_max)

    sample_df = pd.read_csv(args.sample, nrows=args.nrows)
    ids_df = parse_submission_ids(sample_df, id_column="ID")
    validate_submission_id_order(ids_df)

    seasons = sorted(ids_df["Season"].unique().tolist())
    if args.season is not None:
        use_season = int(args.season)
        if seasons != [use_season]:
            raise ValueError(f"Provided --season={use_season} but IDs contain seasons {seasons}")
    else:
        if len(seasons) != 1:
            raise ValueError(f"Multiple seasons in sample IDs {seasons}; pass --season to disambiguate")
        use_season = int(seasons[0])

    men_team_stats = prepare_team_feature_views(pd.read_csv(args.men_team_stats))
    women_team_stats = prepare_team_feature_views(pd.read_csv(args.women_team_stats))
    split = split_ids_by_division(ids_df, men_team_stats=men_team_stats, women_team_stats=women_team_stats)

    out = pd.DataFrame({"ID": ids_df["ID"].values, "Pred": np.nan}, index=ids_df.index)
    for division in ("men", "women"):
        ids_part = split[division]
        if ids_part.empty:
            continue
        team_stats = men_team_stats if division == "men" else women_team_stats
        matchup_df = build_submission_matchups(
            ids_part,
            team_stats,
            feature_columns=None,
            id_column="ID",
            season=use_season,
        )

        logreg_model_path = args.men_logreg_model if division == "men" else args.women_logreg_model
        logreg_features_path = args.men_logreg_features if division == "men" else args.women_logreg_features
        xgb_model_path = args.men_xgb_model if division == "men" else args.women_xgb_model

        logreg_model, logreg_features = load_logreg_model(logreg_model_path, logreg_features_path)
        xgb_model = load_xgb_model(xgb_model_path)

        preds = predict_submission(
            matchup_df,
            logreg_model=logreg_model,
            logreg_features=logreg_features,
            xgb_model=xgb_model,
            xgb_feature_order=SHARED_FEATURE_ORDER,
            weights=[args.logreg_weight, args.xgb_weight],
        )
        out.loc[ids_part.index, "Pred"] = preds

    if out["Pred"].isna().any():
        missing = int(out["Pred"].isna().sum())
        raise ValueError(f"{missing} submission rows are missing predictions after men/women merge")

    out_path = write_submission(out["ID"], out["Pred"].to_numpy(dtype=np.float64), args.output)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
