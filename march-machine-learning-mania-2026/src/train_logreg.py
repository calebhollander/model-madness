"""
Train logistic regression model for win probability.

Role: Fit logistic regression on matchup feature differences; output
probabilities (not just class labels). Used in ensemble with XGBoost and
rating model. Keep model simple; calibrate probabilities before submission.
"""

from typing import Any, Optional

# Input: X (feature matrix, e.g. matchup diffs), y (0/1 target).
# Output: fitted model object; predict_proba returns P(Team A wins).


def train_logreg(
    X_train,
    y_train,
    sample_weight=None,
    **kwargs,
) -> Any:
    """
    Train logistic regression. kwargs can override config.MODEL_PARAMS['logreg'].
    Returns fitted classifier with .predict_proba(X).
    """
    raise NotImplementedError


def predict_proba(model: Any, X) -> "np.ndarray":
    """Return P(Team A wins) for each row; shape (n_samples,) or (n_samples, 2) second col = P(win)."""
    raise NotImplementedError
