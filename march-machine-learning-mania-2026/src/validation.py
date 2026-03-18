"""
Time-based cross-validation by season.

Role: Split data by season so that we never train on future seasons to predict
past (e.g. train on seasons < Y, validate on Y). ALWAYS use time-based splits;
never use random train/val split to avoid leakage and overoptimistic metrics.
"""

from typing import List, Tuple, Any

# Input: DataFrame with Season column; optional list of seasons.
# Output: List of (train_indices, val_indices) or (train_seasons, val_season) per fold.


def get_time_splits(
    seasons: List[int],
    n_splits: int = 5,
    min_train_seasons: int = 5,
) -> List[Tuple[Any, Any]]:
    """
    Generate time-based CV splits. Each fold: train on past seasons, validate on one season.
    Returns list of (train_seasons, val_season) or (train_idx, val_idx) for indexing.
    """
    raise NotImplementedError


def evaluate_cv(
    model,
    X,
    y,
    splits,
    metric: str = "log_loss",
    sample_weight=None,
):
    """
    Run cross-validation: for each split, fit on train and evaluate on val.
    Returns dict or list of per-fold scores; metric should be log loss (Kaggle metric).
    """
    raise NotImplementedError
