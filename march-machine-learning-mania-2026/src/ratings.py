"""
Elo-style (or similar) rating model for team strength.

Role: Compute a single rating per (Season, TeamID) from game results (e.g.
regular season). Ratings are used as features (e.g. rating difference in
matchups) and can be converted to win probability for ensembling. Updates
must use only past games (no future data / no leakage).
"""

from typing import Optional

import pandas as pd

# Input: DataFrame of games (Season, DayNum or order, WTeamID, LTeamID, optionally WScore, LScore).
# Output: DataFrame with (Season, TeamID, rating) or (Season, TeamID, elo, ...).


def compute_ratings(
    games_df,
    initial_rating: float = 1500.0,
    k: float = 20.0,
    use_margin: bool = False,
) -> pd.DataFrame:
    """
    Compute Elo-style ratings from game results. One row per (Season, TeamID).
    games_df must have WTeamID, LTeamID, Season; optional DayNum for order.
    No future games may influence past ratings (time-order enforced).
    """
    raise NotImplementedError


def get_rating_features(ratings_df, season: int, team_id: int) -> dict:
    """Look up rating(s) for a (Season, TeamID). Return dict for feature row or single value."""
    raise NotImplementedError


def rating_to_probability(rating_a: float, rating_b: float) -> float:
    """Convert two ratings to P(Team A beats Team B) via logistic (e.g. 1 / (1 + 10^((Rb-Ra)/400)))."""
    raise NotImplementedError
