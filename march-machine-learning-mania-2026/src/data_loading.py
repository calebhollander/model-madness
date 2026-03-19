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


def _division_token(division: str) -> str:
    """Map a logical division string to Kaggle filename token."""
    d = str(division).strip().lower()
    if d in {"men", "m"}:
        return "M"
    if d in {"women", "w"}:
        return "W"
    raise ValueError(f"Unsupported division {division!r}; expected 'men' or 'women'")


def load_teams(data_dir: Optional[Path] = None, division: str = "men"):
    """Load MTeams.csv or WTeams.csv by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    return pd.read_csv(str(data_dir / f"{token}Teams.csv"))


def load_seasons(data_dir: Optional[Path] = None, division: str = "men"):
    """Load MSeasons.csv or WSeasons.csv by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    return pd.read_csv(str(data_dir / f"{token}Seasons.csv"))


def load_tourney_results(
    data_dir: Optional[Path] = None,
    compact: bool = False,
    division: str = "men",
):
    """Load NCAA tourney results (compact or detailed) by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    stem = f"{token}NCAATourneyCompactResults.csv" if compact else f"{token}NCAATourneyDetailedResults.csv"
    return pd.read_csv(str(data_dir / stem))


def load_regular_season_results(
    data_dir: Optional[Path] = None,
    compact: bool = False,
    division: str = "men",
):
    """Load regular-season results (compact or detailed) by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    stem = f"{token}RegularSeasonCompactResults.csv" if compact else f"{token}RegularSeasonDetailedResults.csv"
    return pd.read_csv(str(data_dir / stem))


def load_tourney_seeds(data_dir: Optional[Path] = None, division: str = "men"):
    """Load NCAA tourney seeds by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    return pd.read_csv(str(data_dir / f"{token}NCAATourneySeeds.csv"))


def load_tourney_slots(data_dir: Optional[Path] = None, division: str = "men"):
    """Load NCAA tourney slots by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    return pd.read_csv(str(data_dir / f"{token}NCAATourneySlots.csv"))


def load_team_names(data_dir: Optional[Path] = None, division: str = "men"):
    """Load team spellings CSV by division."""
    data_dir = data_dir or DATA_RAW
    token = _division_token(division)
    return pd.read_csv(str(data_dir / f"{token}TeamSpellings.csv"))


def load_all_raw(data_dir: Optional[Path] = None, division: str = "men"):
    """
    Load all required Kaggle CSVs into a dict or named tuple of DataFrames.
    Keys: teams, seasons, tourney_results, regular_season_results, tourney_seeds, tourney_slots, (optional) massey, spellings).
    """

    return {
        "teams": load_teams(data_dir, division=division),
        "seasons": load_seasons(data_dir, division=division),
        "tourney_results": load_tourney_results(data_dir, division=division),
        "regular_season_results": load_regular_season_results(data_dir, division=division),
        "tourney_seeds": load_tourney_seeds(data_dir, division=division),
        "tourney_slots": load_tourney_slots(data_dir, division=division),
    }


def load_matchup_data(data_dir: Optional[Path] = None, division: str = "men"):
    """
    Load detailedtournament matchup data.
    """
    data_dir = data_dir or DATA_RAW
    return load_tourney_results(data_dir=data_dir, compact=False, division=division)


def load_season_data(data_dir: Optional[Path] = None, division: str = "men"):
    """
    Load season data from the interim directory.
    """
    data_dir = data_dir or DATA_INTERIM
    return pd.read_csv(str(data_dir / f"{division}_team_features.csv"))


def load_matchup_training_data(data_dir: Optional[Path] = None, division: str = "men"):
    """
    Load matchup training data from the interim directory.
    """
    data_dir = data_dir or DATA_INTERIM
    df = pd.read_csv(str(data_dir / f"{division}_train_matchups.csv"))
    # Alias for models that expect last-10 naming; same as lastx when recent form uses 10 games.
    if "last10_net_eff_diff" not in df.columns and "lastx_net_eff_diff" in df.columns:
        df = df.copy()
        df["last10_net_eff_diff"] = df["lastx_net_eff_diff"]
    return df