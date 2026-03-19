"""
Time-based train/validation splits for matchup-level training data.

Use season columns so validation always reflects future tournaments, not a
random row split (scikit-learn `train_test_split` would leak season structure).
"""

from __future__ import annotations

import pandas as pd

# Same cutoff as the original logistic-regression notebook path: train on
# seasons through this year, validate on all later seasons in the CSV.
DEFAULT_TRAIN_SEASON_MAX = 2018


def split_matchup_train_val(
    df: pd.DataFrame,
    *,
    train_season_max: int = DEFAULT_TRAIN_SEASON_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split matchup rows into train and validation by `Season`.

    Parameters
    ----------
    df
        Matchup training frame (must include a `Season` column).
    train_season_max
        Training rows satisfy `Season <= train_season_max`; validation is
        strictly later seasons.

    Returns
    -------
    train_df, val_df
        Copies of the input slices so downstream code can add columns safely.
    """
    train_df = df.loc[df["Season"] <= train_season_max].copy()
    val_df = df.loc[df["Season"] > train_season_max].copy()
    return train_df, val_df
