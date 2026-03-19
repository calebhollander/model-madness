"""
Train XGBoost classifier for win probability.

Role: Fit XGBoost on matchup feature differences; report validation log loss and
feature importances (gain-based defaults from the booster).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from src import config
from src.data_loading import load_matchup_training_data
from src.splits import split_matchup_train_val
from src.train_logreg import FEATURES


def train_xgb(
    features: Optional[list[str]] = None,
    **kwargs: Any,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier on the matchup frame using the same time-based
    train/val split as logistic regression (`split_matchup_train_val`).

    Hyperparameters start from `config.MODEL_PARAMS['xgb']`; ``kwargs`` override
    those entries (e.g. for notebook experiments).

    Uses `FEATURES` from `train_logreg` (full baseline + candidate diff columns)
    unless ``features`` is passed explicitly.

    Prints validation log loss and a sorted importance table, then returns the
    fitted estimator.
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

    feat_list = list(features) if features is not None else list(FEATURES)

    df = load_matchup_training_data()
    train_df, val_df = split_matchup_train_val(df)

    X_train = train_df[feat_list]
    y_train = train_df["target"]
    X_val = val_df[feat_list]
    y_val = val_df["target"]

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    print("XGBoost Log Loss:", log_loss(y_val, y_pred))

    importance = get_feature_importance(model, feat_list)
    print(importance)

    return model


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return P(Team A wins) per row (positive class column from `predict_proba`)."""
    return model.predict_proba(X)[:, 1]


def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a pandas DataFrame of feature importances from the fitted XGBoost
    estimator (`feature_importances_` matches the order of columns at fit time).
    """
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
