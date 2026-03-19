"""
Train logistic regression model for win probability.

Role: Fit logistic regression on matchup feature differences; output
probabilities (not just class labels). Used in ensemble with XGBoost and
rating model. Keep model simple; calibrate probabilities before submission.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import joblib

from src.data_loading import load_matchup_training_data
from src.splits import split_matchup_train_val

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src import config

import pandas as pd
import numpy as np

# Input: X (feature matrix, e.g. matchup diffs), y (0/1 target).
# Output: fitted model object; predict_proba returns P(Team A wins).

BASELINE_FEATURES = [
    "seed_diff",
    "avg_margin_diff",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "win_pct_diff",
]

# Try these one at a time (greedy forward selection) and keep only if validation improves.
CANDIDATE_FEATURES = [
    "efg_pct_diff",
    "tov_pct_diff",
    "orb_pct_diff",
    "ftr_diff",
    "last10_net_eff_diff",
]

# Default feature set used when not running selection explicitly.
FEATURES = BASELINE_FEATURES + CANDIDATE_FEATURES


def _fit_and_score_logreg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_params: dict,
    features: list[str],
) -> tuple[float, LogisticRegression, np.ndarray]:
    X_train = train_df[features]
    y_train = train_df["target"]

    X_val = val_df[features]
    y_val = val_df["target"]

    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, y_pred)
    return loss, model, y_pred





def train_logreg(
    X = None,
    Y = None,
    sample_weight=None,
    run_feature_selection: bool = True,
    division: str = "men",
    **kwargs,
) -> tuple[LogisticRegression, list[str]]:
    """
    Train logistic regression. kwargs can override config.MODEL_PARAMS['logreg'].

    Returns
    -------
    model, feature_names
        Fitted classifier and the ordered feature columns used at fit time
        (needed for inference when greedy selection is on).
    """

    # In notebooks, `src.config` can be imported before edits land, leaving the
    # module cached without newer helpers. Fall back to direct mkdirs so model
    # training isn't blocked by a stale import.
    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()
    else:
        for p in (config.DATA_INTERIM, config.DATA_PROCESSED, config.MODELS_DIR, config.OUTPUTS_DIR):
            p.mkdir(parents=True, exist_ok=True)

    model_params = config.MODEL_PARAMS["logreg"]
    model_params.update(kwargs)

    df = load_matchup_training_data(division=division)

    train_df, val_df = split_matchup_train_val(df)

    # If external X/Y are supplied, do the simple fit path and skip selection.
    if X is not None and Y is not None:
        model = LogisticRegression(**model_params)
        model.fit(X, Y, sample_weight=sample_weight)
        if hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"x{i}" for i in range(X.shape[1])]
        return model, names

    selected_features = list(BASELINE_FEATURES)

    base_loss, _, _ = _fit_and_score_logreg(train_df, val_df, model_params, selected_features)
    print(f"Baseline ({len(selected_features)} feats) log loss: {base_loss}")

    if run_feature_selection:
        best_loss = base_loss
        for feat in CANDIDATE_FEATURES:
            trial_features = selected_features + [feat]
            trial_loss, _, _ = _fit_and_score_logreg(train_df, val_df, model_params, trial_features)

            if trial_loss < best_loss:
                selected_features.append(feat)
                best_loss = trial_loss
                print(f"KEEP  + {feat:22s} -> {trial_loss}")
            else:
                print(f"DROP  - {feat:22s} -> {trial_loss}")
    else:
        selected_features = list(FEATURES)

    val_loss, model, y_pred = _fit_and_score_logreg(train_df, val_df, model_params, selected_features)
    print(f"Final ({len(selected_features)} feats) log loss: {val_loss}")
    print(f"Final features: {selected_features}")

    val_df.loc[:, "pred"] = y_pred
    preview_cols = ["target", "pred"] + selected_features
    print(val_df[preview_cols].head(60))

    coef_df = (
        pd.DataFrame({"feature": selected_features, "coefficient": model.coef_[0]})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    print(coef_df)

    return model, selected_features


def save_logreg_artifacts(
    model: LogisticRegression,
    feature_names: list[str],
    model_path: Optional[Union[str, Path]] = None,
    features_path: Optional[Union[str, Path]] = None,
    division: str = "men",
) -> tuple[Path, Path]:
    """
    Persist the sklearn estimator and the ordered feature list for inference
    (joblib + JSON). Paths default to `config.MODELS_DIR` filenames.
    """
    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()
    else:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path or config.MODELS_DIR / f"logreg_{division}.pkl")
    features_path = Path(features_path or config.MODELS_DIR / f"logreg_{division}_features.json")
    joblib.dump(model, model_path)
    features_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    return model_path, features_path


def predict_proba(model: Any, X) -> np.ndarray:
    """Return P(Team A wins) per row (scikit-learn positive class column)."""
    return model.predict_proba(X)[:, 1]
