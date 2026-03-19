"""
2026 Stage 2 inference: build matchup diffs from the sample submission and run
saved logistic regression + XGBoost models.

Reads `SampleSubmissionStage2.csv`, merges `men_team_features.csv`, writes
`outputs/predictions_2026_logreg_xgb.csv` (ID, pred_logreg, pred_xgb).

Stage 2 IDs may reference TeamIDs not in the men features file; matchup_builder
fills those sides with **season medians** so inference does not hit NaNs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.matchup_builder import build_submission_matchups
from src.train_logreg import FEATURES as XGB_FEATURE_ORDER


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, config.SUBMISSION_PROB_MIN, config.SUBMISSION_PROB_MAX)


def predict_from_matchup_dataframe(
    matchup_df: pd.DataFrame,
    *,
    logreg_model_path: Optional[Path] = None,
    logreg_features_path: Optional[Path] = None,
    xgb_model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    write_csv: bool = False,
    id_column: str = "ID",
) -> pd.DataFrame:
    """
    Run saved logreg + XGB on rows produced by ``build_submission_matchups``
    (must include diff feature columns and ``id_column``).

    When ``write_csv`` is True, ``output_path`` must be set (or defaults to
    ``config.PREDICTIONS_2026_LOGREG_XGB``).
    """
    logreg_model_path = Path(logreg_model_path or config.MODELS_DIR / "logreg_men.pkl")
    logreg_features_path = Path(logreg_features_path or config.LOGREG_FEATURE_NAMES_JSON)
    xgb_model_path = Path(xgb_model_path or config.MODELS_DIR / "xgb_men.pkl")

    logreg_features: list[str] = json.loads(
        logreg_features_path.read_text(encoding="utf-8")
    )
    logreg_model = joblib.load(logreg_model_path)
    xgb_model = joblib.load(xgb_model_path)

    X_lr = matchup_df[logreg_features]
    X_xgb = matchup_df[list(XGB_FEATURE_ORDER)]

    if X_lr.isna().any().any() or X_xgb.isna().any().any():
        n_lr = int(X_lr.isna().any(axis=1).sum())
        n_xgb = int(X_xgb.isna().any(axis=1).sum())
        raise ValueError(
            f"NaNs in feature matrix (logreg rows with any NaN: {n_lr}, "
            f"xgb rows with any NaN: {n_xgb}). Check team_stats coverage."
        )

    pred_logreg = _clip_probs(logreg_model.predict_proba(X_lr)[:, 1].astype(np.float64))
    pred_xgb = _clip_probs(xgb_model.predict_proba(X_xgb)[:, 1].astype(np.float64))

    out = pd.DataFrame(
        {
            id_column: matchup_df[id_column],
            "pred_logreg": pred_logreg,
            "pred_xgb": pred_xgb,
        }
    )
    if write_csv:
        out_path = Path(output_path or config.PREDICTIONS_2026_LOGREG_XGB)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


def predict_stage2_matchups(
    sample_sub_path: Optional[Path] = None,
    team_stats_path: Optional[Path] = None,
    logreg_model_path: Optional[Path] = None,
    logreg_features_path: Optional[Path] = None,
    xgb_model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    *,
    season: int = 2026,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Stage 2 template IDs, build diff features, run both models, clip
    probabilities, and write the combined CSV.

    Parameters
    ----------
    nrows
        If set, only read this many rows from the sample file (for quick tests).
    season
        Season filter for team stats and ID consistency check.
    """
    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()

    sample_sub_path = Path(sample_sub_path or config.SAMPLE_SUBMISSION_STAGE2)
    team_stats_path = Path(team_stats_path or config.DATA_INTERIM / "men_team_features.csv")
    logreg_model_path = Path(logreg_model_path or config.MODELS_DIR / "logreg_men.pkl")
    logreg_features_path = Path(logreg_features_path or config.LOGREG_FEATURE_NAMES_JSON)
    xgb_model_path = Path(xgb_model_path or config.MODELS_DIR / "xgb_men.pkl")
    output_path = Path(output_path or config.PREDICTIONS_2026_LOGREG_XGB)

    sample_df = pd.read_csv(sample_sub_path, nrows=nrows)
    team_stats = pd.read_csv(team_stats_path)
    matchup_df = build_submission_matchups(
        sample_df,
        team_stats,
        feature_columns=None,
        id_column="ID",
        season=season,
    )

    out = predict_from_matchup_dataframe(
        matchup_df,
        logreg_model_path=logreg_model_path,
        logreg_features_path=logreg_features_path,
        xgb_model_path=xgb_model_path,
        output_path=output_path,
        write_csv=True,
        id_column="ID",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="2026 Stage 2 logreg + XGB predictions.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Only process first N rows of the sample file (smoke test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: config.PREDICTIONS_2026_LOGREG_XGB).",
    )
    args = parser.parse_args()
    out_path = Path(args.output or config.PREDICTIONS_2026_LOGREG_XGB)
    predict_stage2_matchups(nrows=args.nrows, output_path=args.output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
