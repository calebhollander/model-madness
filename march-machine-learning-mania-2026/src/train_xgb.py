"""
Train XGBoost classifier for win probability.

Role: Fit XGBoost on matchup feature differences; report validation log loss and
feature importances (gain-based defaults from the booster).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import log_loss

from src import config
from src.data_loading import load_matchup_training_data
from src.splits import split_matchup_train_val
from src.train_logreg import FEATURES


def train_xgb(
    features: Optional[list[str]] = None,
    save_model: bool = True,
    **kwargs: Any,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier on the matchup frame using the same time-based
    train/val split as logistic regression (`split_matchup_train_val`).

    Hyperparameters start from `config.MODEL_PARAMS['xgb']`; ``kwargs`` override
    those entries (e.g. for notebook experiments).

    The model is trained in a validation-aware way via ``eval_set``. When
    `early_stopping_rounds` is present in ``config.MODEL_PARAMS['xgb']``, we
    pass it through to ``model.fit``.

    Uses `FEATURES` from `train_logreg` (full baseline + candidate diff columns)
    unless ``features`` is passed explicitly.

    Prints validation log loss and a sorted gain-importance table, then
    returns the fitted estimator.
    """
    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()
    else:
        for p in (
            config.DATA_INTERIM,
            config.DATA_PROCESSED,
            config.MODELS_DIR,
            config.OUTPUTS_DIR,
        ):
            p.mkdir(parents=True, exist_ok=True)

    model_params = dict(config.MODEL_PARAMS["xgb"])
    model_params.update(kwargs)
    # XGBoost 2.x ignores `use_label_encoder`; dropping it avoids a runtime warning.
    model_params.pop("use_label_encoder", None)
    early_stopping_rounds = model_params.pop("early_stopping_rounds", None)

    feat_list = list(features) if features is not None else list(FEATURES)

    df = load_matchup_training_data()
    train_df, val_df = split_matchup_train_val(df)

    X_train = train_df[feat_list]
    y_train = train_df["target"]
    X_val = val_df[feat_list]
    y_val = val_df["target"]

    model = xgb.XGBClassifier(**model_params)
    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(X_train, y_train, **fit_kwargs)

    # Clip probabilities before log loss to avoid any log(0) edge cases.
    y_pred = model.predict_proba(X_val)[:, 1].astype(np.float64)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    val_loss = log_loss(y_val, y_pred)
    print(f"XGBoost validation log loss: {val_loss:.6f}")

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        print(f"best_iteration: {best_iteration}")

    importance = get_feature_importance(model, feat_list, importance_type="gain")
    print(importance)

    if save_model:
        xgb_json_path = config.MODELS_DIR / "xgb_men.json"
        model.save_model(str(xgb_json_path))
        # Preserve backward compatibility with existing inference code that
        # currently loads the pickled estimator.
        xgb_pkl_path = config.MODELS_DIR / "xgb_men.pkl"
        joblib.dump(model, xgb_pkl_path)

    return model


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(Team A wins) per row (positive class column from `predict_proba`)."""
    return model.predict_proba(X)[:, 1]


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    *,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Build a pandas DataFrame of booster-based feature importances.

    We use XGBoost's underlying booster scores (e.g. ``importance_type="gain"``)
    rather than ``feature_importances_`` so the values match the exact booster
    scoring used during training.
    """
    booster = model.get_booster()
    raw_scores = booster.get_score(importance_type=importance_type)

    importance_by_feature: dict[str, float] = {name: 0.0 for name in feature_names}
    for key, value in raw_scores.items():
        # XGBoost uses f0, f1, ... when feature names are not explicitly attached.
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if 0 <= idx < len(feature_names):
                importance_by_feature[feature_names[idx]] = float(value)
            continue

        # If the booster already stores real feature names, respect them.
        if key in importance_by_feature:
            importance_by_feature[key] = float(value)

    df = pd.DataFrame(
        {"feature": list(importance_by_feature.keys()), "importance": list(importance_by_feature.values())}
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
