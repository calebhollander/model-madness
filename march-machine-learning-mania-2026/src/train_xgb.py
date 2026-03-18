"""
Train XGBoost classifier for win probability.

Role: Fit XGBoost on matchup feature differences; output probabilities.
Optionally export feature importance to outputs/feature_importance.csv.
Use time-based CV for tuning; keep model size small (e.g. max_depth 4).
"""

from typing import Any, Optional

# Input: X, y; optional validation set for early stopping.
# Output: fitted booster; predict_proba; optional importance DataFrame.


def train_xgb(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    **kwargs,
) -> Any:
    """
    Train XGBoost classifier. kwargs can override config.MODEL_PARAMS['xgb'].
    Returns fitted model with .predict_proba(X). Use eval_set for early stopping if provided.
    """
    raise NotImplementedError


def predict_proba(model: Any, X) -> "np.ndarray":
    """Return P(Team A wins) for each row."""
    raise NotImplementedError


def get_feature_importance(model: Any, feature_names: list) -> "pd.DataFrame":
    """Return DataFrame with feature names and importance; can be saved to outputs/feature_importance.csv."""
    raise NotImplementedError
