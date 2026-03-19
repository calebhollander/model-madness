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

# Resolve data paths relative to this file so imports work from notebooks/scripts/tests
# regardless of current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = _PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = _PROJECT_ROOT / "data" / "interim"
# Expected inputs: paths under config.DATA_RAW (or project root).
# Outputs: pandas DataFrames with consistent dtypes and no duplicate index issues.


def load_teams(data_dir: Optional[Path] = None):
    """Load MTeams.csv (and optionally WTeams.csv). Returns DataFrame with TeamID, TeamName, etc."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MTeams.csv"))


def load_seasons(data_dir: Optional[Path] = None):
    """Load MSeasons.csv. Returns DataFrame with Season, DayZero, etc."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MSeasons.csv"))


def load_tourney_results(data_dir: Optional[Path] = None, compact: bool = True):
    """Load MNCAATourneyCompactResults (or Detailed). Returns Season, WTeamID, LTeamID, (WScore, LScore)."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MNCAATourneyDetailedResults.csv"))


def load_regular_season_results(data_dir: Optional[Path] = None, compact: bool = False):
    """Load MRegularSeasonCompactResults (or Detailed). Returns game-level rows with WTeamID, LTeamID, scores."""
    data_dir = data_dir or DATA_RAW
    return (
        pd.read_csv(str(data_dir / "MRegularSeasonCompactResults.csv"))
        if compact
        else pd.read_csv(str(data_dir / "MRegularSeasonDetailedResults.csv"))
    )


def load_tourney_seeds(data_dir: Optional[Path] = None):
    """Load MNCAATourneySeeds.csv. Returns Season, Seed, TeamID."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MNCAATourneySeeds.csv"))


def load_tourney_slots(data_dir: Optional[Path] = None):
    """Load MNCAATourneySlots.csv. Returns slot structure for bracket (Season, Slot, StrongSeed, WeakSeed)."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MNCAATourneySlots.csv"))

def load_team_names(data_dir: Optional[Path] = None):
    """Load MTeamSpellings.csv (columns ``TeamNameSpelling``, ``TeamID``)."""
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MTeamSpellings.csv"))

def load_all_raw(data_dir: Optional[Path] = None):
    """
    Load all required Kaggle CSVs into a dict or named tuple of DataFrames.
    Keys: teams, seasons, tourney_results, regular_season_results, tourney_seeds, tourney_slots, (optional) massey, spellings).
    """

    return {
        "teams": load_teams(data_dir),
        "seasons": load_seasons(data_dir),
        "tourney_results": load_tourney_results(data_dir),
        "regular_season_results": load_regular_season_results(data_dir),
        "tourney_seeds": load_tourney_seeds(data_dir),
        "tourney_slots": load_tourney_slots(data_dir),
    }

def load_matchup_data(data_dir: Optional[Path] = None):
    """
    Load detailedtournament matchup data.
    """
    data_dir = data_dir or DATA_RAW
    return pd.read_csv(str(data_dir / "MNCAATourneyDetailedResults.csv"))

def load_season_data(data_dir: Optional[Path] = None):
    """
    Load season data from the interim directory.
    """
    data_dir = data_dir or DATA_INTERIM
    return pd.read_csv(str(data_dir / "men_team_features.csv"))

def load_matchup_training_data(data_dir: Optional[Path] = None):
    """
    Load matchup training data from the interim directory.
    """
    data_dir = data_dir or DATA_INTERIM
    df = pd.read_csv(str(data_dir / "men_train_matchups.csv"))
    # Alias for models that expect last-10 naming; same as lastx when recent form uses 10 games.
    if "last10_net_eff_diff" not in df.columns and "lastx_net_eff_diff" in df.columns:
        df = df.copy()
        df["last10_net_eff_diff"] = df["lastx_net_eff_diff"]
    return df