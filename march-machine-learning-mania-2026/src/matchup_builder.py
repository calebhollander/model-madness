"""
Matchup generation: convert team-level features to pairwise dataset.

Role: For each (Season, TeamA, TeamB) with a known outcome (from tourney results),
build one row with FEATURE DIFFERENCES (Team A stat minus Team B stat). Target:
1 if Team A (first team in ID order or WTeamID) wins, else 0. For submission,
generate (Season, TeamA, TeamB) from sample submission and apply same diff logic.
Always use differences—never raw team stats—to keep features comparable and avoid leakage.
"""

from typing import Optional

import pandas as pd

from src.data_loading import load_season_data, load_matchup_data

# Input: team_stats DataFrame (Season, TeamID, feat1, feat2, ...), games_with_outcomes (Season, WTeamID, LTeamID).
# Output: DataFrame with Season, TeamID_A, TeamID_B, diff_feat1, diff_feat2, ..., target (0/1).


def _load_filtered_sources(
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
    division: str = "men",
):
    season_features = load_season_data(division=division)
    tourney_games = load_matchup_data(division=division)

    if season_min is not None:
        season_features = season_features[season_features["Season"] >= season_min]
        tourney_games = tourney_games[tourney_games["Season"] >= season_min]
    if season_max is not None:
        season_features = season_features[season_features["Season"] <= season_max]
        tourney_games = tourney_games[tourney_games["Season"] <= season_max]

    return season_features, tourney_games


def _build_labeled_matchups(tourney_games: pd.DataFrame) -> pd.DataFrame:
    # 1 for win, 0 for loss
    winner_matchups = pd.DataFrame(
        {
            "Season": tourney_games["Season"],
            "DayNum": tourney_games["DayNum"],
            "TeamA": tourney_games["WTeamID"],
            "TeamB": tourney_games["LTeamID"],
            "target": 1,
        }
    )

    loser_matchups = pd.DataFrame(
        {
            "Season": tourney_games["Season"],
            "DayNum": tourney_games["DayNum"],
            "TeamA": tourney_games["LTeamID"],
            "TeamB": tourney_games["WTeamID"],
            "target": 0,
        }
    )

    return pd.concat([winner_matchups, loser_matchups])


def _split_team_features(season_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_a_features = season_features.rename(
        columns={
            "TeamID": "TeamA",
            "games_played": "games_played_A",
            "win_pct": "win_pct_A",
            "avg_margin": "avg_margin_A",
            "avg_points_for": "avg_points_for_A",
            "avg_points_against": "avg_points_against_A",
            "off_eff": "off_eff_A",
            "def_eff": "def_eff_A",
            "net_eff": "net_eff_A",
            "efg_pct": "efg_pct_A",
            "tov_pct": "tov_pct_A",
            "orb_pct": "orb_pct_A",
            "ftr": "ftr_A",
            "lastx_win_pct": "lastx_win_pct_A",
            "lastx_avg_margin": "lastx_avg_margin_A",
            "lastx_off_eff": "lastx_off_eff_A",
            "lastx_def_eff": "lastx_def_eff_A",
            "lastx_net_eff": "lastx_net_eff_A",
            "seed_num": "seed_num_A",
        }
    )

    team_b_features = season_features.rename(
        columns={
            "TeamID": "TeamB",
            "games_played": "games_played_B",
            "win_pct": "win_pct_B",
            "avg_margin": "avg_margin_B",
            "avg_points_for": "avg_points_for_B",
            "avg_points_against": "avg_points_against_B",
            "off_eff": "off_eff_B",
            "def_eff": "def_eff_B",
            "net_eff": "net_eff_B",
            "efg_pct": "efg_pct_B",
            "tov_pct": "tov_pct_B",
            "orb_pct": "orb_pct_B",
            "ftr": "ftr_B",
            "lastx_win_pct": "lastx_win_pct_B",
            "lastx_avg_margin": "lastx_avg_margin_B",
            "lastx_off_eff": "lastx_off_eff_B",
            "lastx_def_eff": "lastx_def_eff_B",
            "lastx_net_eff": "lastx_net_eff_B",
            "seed_num": "seed_num_B",
        }
    )

    return team_a_features, team_b_features


def _attach_team_features(
    matchups: pd.DataFrame,
    team_a_features: pd.DataFrame,
    team_b_features: pd.DataFrame,
) -> pd.DataFrame:
    matchups = matchups.merge(team_a_features, on=["Season", "TeamA"], how="left")
    matchups = matchups.merge(team_b_features, on=["Season", "TeamB"], how="left")
    return matchups


def _impute_missing_team_sides_from_season_stats(
    matchups: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    After a left merge, fill NaN ``*_A`` / ``*_B`` columns using per-column
    medians from ``stats`` for that season.

    Kaggle Stage 2 IDs can include TeamIDs not present in a single league file
    (e.g. women's IDs when only ``men_team_features`` is available). Without this,
    ``_add_feature_differences`` produces NaNs and sklearn/XGB inference fails.
    """
    for col in stats.columns:
        if col in ("Season", "TeamID"):
            continue
        if not pd.api.types.is_numeric_dtype(stats[col]):
            continue
        ca, cb = f"{col}_A", f"{col}_B"
        if ca not in matchups.columns or cb not in matchups.columns:
            continue
        med = stats[col].median()
        if pd.isna(med):
            med = 16.0 if col == "seed_num" else 0.0
        matchups[ca] = matchups[ca].fillna(med)
        matchups[cb] = matchups[cb].fillna(med)
    return matchups


def _add_feature_differences(matchups: pd.DataFrame) -> pd.DataFrame:
    matchups["seed_diff"] = matchups["seed_num_A"] - matchups["seed_num_B"]
    matchups["win_pct_diff"] = matchups["win_pct_A"] - matchups["win_pct_B"]
    matchups["avg_margin_diff"] = matchups["avg_margin_A"] - matchups["avg_margin_B"]
    matchups["off_eff_diff"] = matchups["off_eff_A"] - matchups["off_eff_B"]
    matchups["def_eff_diff"] = matchups["def_eff_A"] - matchups["def_eff_B"]
    matchups["net_eff_diff"] = matchups["net_eff_A"] - matchups["net_eff_B"]
    matchups["efg_pct_diff"] = matchups["efg_pct_A"] - matchups["efg_pct_B"]
    matchups["tov_pct_diff"] = matchups["tov_pct_A"] - matchups["tov_pct_B"]
    matchups["orb_pct_diff"] = matchups["orb_pct_A"] - matchups["orb_pct_B"]
    matchups["ftr_diff"] = matchups["ftr_A"] - matchups["ftr_B"]
    matchups["lastx_win_pct_diff"] = matchups["lastx_win_pct_A"] - matchups["lastx_win_pct_B"]
    matchups["lastx_avg_margin_diff"] = matchups["lastx_avg_margin_A"] - matchups["lastx_avg_margin_B"]
    matchups["lastx_off_eff_diff"] = matchups["lastx_off_eff_A"] - matchups["lastx_off_eff_B"]
    matchups["lastx_def_eff_diff"] = matchups["lastx_def_eff_A"] - matchups["lastx_def_eff_B"]
    matchups["lastx_net_eff_diff"] = matchups["lastx_net_eff_A"] - matchups["lastx_net_eff_B"]
    # Same values as lastx_net_eff_diff when recent form uses num_games=10 (see get_recent_features).
    matchups["last10_net_eff_diff"] = matchups["lastx_net_eff_diff"]
    return matchups


# Diff columns shared by training CSV and submission-time matchup rows.
DIFF_FEATURE_COLUMNS = [
    "seed_diff",
    "win_pct_diff",
    "avg_margin_diff",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "efg_pct_diff",
    "tov_pct_diff",
    "orb_pct_diff",
    "ftr_diff",
    "lastx_win_pct_diff",
    "lastx_avg_margin_diff",
    "lastx_off_eff_diff",
    "lastx_def_eff_diff",
    "lastx_net_eff_diff",
    "last10_net_eff_diff",
]


def _select_training_columns(matchups: pd.DataFrame) -> pd.DataFrame:
    return matchups[
        [
            "Season",
            "DayNum",
            "TeamA",
            "TeamB",
            "target",
            *DIFF_FEATURE_COLUMNS,
        ]
    ].copy()


def build_matchup_data(
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
    division: str = "men",
):
    """
    Build pairwise rows: for each game, compute feature_diffs (TeamA - TeamB)
    and target (1 if TeamA wins, 0 else). TeamA/TeamB ordering must be consistent
    with submission ID (e.g. lower TeamID first).
    """

    season_features, tourney_games = _load_filtered_sources(
        season_min=season_min,
        season_max=season_max,
        division=division,
    )
    matchups = _build_labeled_matchups(tourney_games)
    team_a_features, team_b_features = _split_team_features(season_features)
    matchups = _attach_team_features(matchups, team_a_features, team_b_features)
    matchups = _add_feature_differences(matchups)
    training_df = _select_training_columns(matchups)

    return training_df


def build_submission_matchups(
    sample_submission_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    *,
    id_column: str = "ID",
    season: Optional[int] = None,
) -> pd.DataFrame:
    """
    From SampleSubmissionStage2 (ID = Season_TeamID1_TeamID2), parse IDs and
    build matchup rows with the same diff features as training. TeamA/TeamB are
    the first and second team IDs in the ID string (Kaggle order), not numeric sort.

    Parameters
    ----------
    sample_submission_df
        Rows with at least ``id_column`` (e.g. ``2026_1101_1102``).
    team_stats
        Per-team season table (``Season``, ``TeamID``, stats, ``seed_num``) as in
        ``men_team_features.csv``. TeamIDs absent from this table (common for
        Stage 2 when IDs mix leagues) get that column's **season median** on each
        side so diff features stay finite.
    feature_columns
        If set, subset of diff columns to keep in the output (must exist after
        merge). If ``None``, all ``DIFF_FEATURE_COLUMNS`` are returned.
    id_column
        Name of the submission ID column.
    season
        If set, restrict ``team_stats`` to this season. If ``None``, use the
        single season implied by the submission IDs (raises if multiple seasons).
    """
    if id_column not in sample_submission_df.columns:
        raise KeyError(f"Missing column {id_column!r} in sample submission")

    ids = sample_submission_df[id_column].astype(str)
    parts = ids.str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(f"Expected ID format Season_TeamA_TeamB; got bad row in {id_column}")

    matchups = pd.DataFrame(
        {
            id_column: sample_submission_df[id_column].values,
            "Season": parts.iloc[:, 0].astype(int),
            "TeamA": parts.iloc[:, 1].astype(int),
            "TeamB": parts.iloc[:, 2].astype(int),
        }
    )

    seasons = matchups["Season"].unique()
    if season is not None:
        use_season = season
        if not (seasons == use_season).all():
            raise ValueError(
                f"season={use_season} but submission IDs contain seasons {sorted(seasons.tolist())}"
            )
    else:
        if len(seasons) != 1:
            raise ValueError(
                "Multiple seasons in submission IDs; pass season= explicitly or split the file."
            )
        use_season = int(seasons[0])

    stats = team_stats.loc[team_stats["Season"] == use_season].copy()
    team_a_features, team_b_features = _split_team_features(stats)
    matchups = _attach_team_features(matchups, team_a_features, team_b_features)
    # NaNs from unknown TeamIDs (e.g. W teams when only men stats exist) or
    # missing seed_num rows—fill from season medians before diffs.
    matchups = _impute_missing_team_sides_from_season_stats(matchups, stats)

    matchups = _add_feature_differences(matchups)

    missing_a = matchups["win_pct_A"].isna()
    missing_b = matchups["win_pct_B"].isna()
    if (missing_a | missing_b).any():
        n_bad = int((missing_a | missing_b).sum())
        print(
            f"Warning: {n_bad} matchup rows missing team stats after merge "
            f"(Season={use_season})."
        )

    cols_out = [id_column, "Season", "TeamA", "TeamB"]
    if feature_columns is None:
        diff_cols = list(DIFF_FEATURE_COLUMNS)
    else:
        missing = set(feature_columns) - set(DIFF_FEATURE_COLUMNS)
        if missing:
            raise ValueError(f"Unknown feature_columns (not diff features): {sorted(missing)}")
        diff_cols = list(feature_columns)

    for c in diff_cols:
        if c not in matchups.columns:
            raise KeyError(f"Expected diff column {c!r} after feature merge")

    return matchups[cols_out + diff_cols].copy()
