"""
Data loading: read Kaggle CSVs and return clean DataFrames.

Role: First step in the pipeline. Loads teams, seasons, regular-season results,
tourney results, seeds, slots, and optional Massey ordinals / spellings.
Outputs are used by feature engineering and matchup building. No transformation
beyond basic read and optional type casting.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

DATA_RAW = Path('..') / "data" / "raw"
# Expected inputs: paths under config.DATA_RAW (or project root).
# Outputs: pandas DataFrames with consistent dtypes and no duplicate index issues.


def load_teams(data_dir: Optional[Path] = None):
    """Load MTeams.csv (and optionally WTeams.csv). Returns DataFrame with TeamID, TeamName, etc."""
    return pd.read_csv(DATA_RAW / "MTeams.csv")


def load_seasons(data_dir: Optional[Path] = None):
    """Load MSeasons.csv. Returns DataFrame with Season, DayZero, etc."""
    return pd.read_csv(DATA_RAW / "MSeasons.csv")


def load_tourney_results(data_dir: Optional[Path] = None, compact: bool = True):
    """Load MNCAATourneyCompactResults (or Detailed). Returns Season, WTeamID, LTeamID, (WScore, LScore)."""
    return pd.read_csv(DATA_RAW / "MNCAATourneyDetailedResults.csv")


def load_regular_season_results(data_dir: Optional[Path] = None, compact: bool = False):
    """Load MRegularSeasonCompactResults (or Detailed). Returns game-level rows with WTeamID, LTeamID, scores."""
    return (pd.read_csv(DATA_RAW / "MRegularSeasonCompactResults.csv") if compact else pd.read_csv(DATA_RAW / "MRegularSeasonDetailedResults.csv"))


def load_tourney_seeds(data_dir: Optional[Path] = None):
    """Load MNCAATourneySeeds.csv. Returns Season, Seed, TeamID."""
    return pd.read_csv(DATA_RAW / "MNCAATourneySeeds.csv")


def load_tourney_slots(data_dir: Optional[Path] = None):
    """Load MNCAATourneySlots.csv. Returns slot structure for bracket (Season, Slot, StrongSeed, WeakSeed)."""
    raise NotImplementedError


def load_all_raw(data_dir: Optional[Path] = None):
    """
    Load all required Kaggle CSVs into a dict or named tuple of DataFrames.
    Keys: teams, seasons, tourney_results, regular_season_results, tourney_seeds, tourney_slots, (optional) massey, spellings).
    """
    raise NotImplementedError
