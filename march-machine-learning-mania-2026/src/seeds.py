"""
Seed parsing and seed-based features.

Role: Parse Kaggle seed strings (e.g. 'W01', 'X16') into numeric values for
modeling. Optional: derive seed strength or region indicators. Used when
building team-level features and matchup diffs.
"""

import re
from typing import Union

# Input: seed string from MNCAATourneySeeds (e.g. "W01").
# Output: integer 1--16 (or 1--N) for bracket position; optional one-hot or strength feature.


def parse_seed(seed_str: str) -> int:
    """
    Parse seed string (e.g. 'W01', 'X16') to numeric seed rank.
    Returns integer 1--16 (or 1--N depending on bracket size).
    """
    if not seed_str or not isinstance(seed_str, str):
        raise ValueError("seed_str must be a non-empty string")
    match = re.match(r"^[A-Za-z](\d+)$", seed_str.strip())
    if not match:
        raise ValueError(f"Cannot parse seed: {seed_str}")
    return int(match.group(1))


def get_seed_features(seeds_series):
    """
    From a Series of seed strings, return a DataFrame of numeric seed features
    (e.g. seed_rank, optional region dummy). Used for team-level feature table.
    """
    raise NotImplementedError
