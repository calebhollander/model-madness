"""
Feature engineering: build season-level team statistics.

Role: From raw game and seed data, produce one row per (Season, TeamID) with
aggregate stats: win rate, points per game, points allowed, efficiency, seed
rank, optional rating. These are the inputs to matchup_builder; matchups
use FEATURE DIFFERENCES (Team A stat minus Team B stat), not raw team stats,
to avoid leakage and improve generalization.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

# Input: DataFrames from data_loading (games, seeds, optional ratings).
# Output: DataFrame with columns [Season, TeamID, win_pct, ppg, papg, ...].
# Best practice: only use data available before tournament (e.g. regular season only for that season).


def build_regular_season_team_stats(
    team_games: pd.DataFrame,
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build season-level team stats from long-format regular-season games.

    Expects one row per (Season, DayNum, TeamID, OppTeamID) with box-score columns
    like `points_for`, `points_against`, `fga`, `fta`, `or_`, `to`, and opponent
    counterparts (as produced by `game_results.make_long_regular_season_results`).
    """
    df = team_games
    if season_min is not None:
        df = df[df["Season"] >= season_min]
    if season_max is not None:
        df = df[df["Season"] <= season_max]
    
    season_features = (
        df.groupby(["Season", "TeamID"])
        .agg(
            games_played=("win", "size"),
            win_pct=("win", "mean"),
            avg_margin=("margin", "mean"),
            avg_points_for=("points_for", "mean"),
            avg_points_against=("points_against", "mean"),
            off_eff=("off_eff", "mean"),
            def_eff=("def_eff", "mean"),
            net_eff=("net_eff", "mean"),
            efg_pct=("efg_pct", "mean"),
            tov_pct=("tov_pct", "mean"),
            orb_pct=("orb_pct", "mean"),
            ftr=("ftr", "mean"),
        )
        .reset_index()
    )

    return season_features


def build_team_season_stats(
    regular_season_results,
    tourney_seeds=None,
    ratings_df=None,
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
):
    """
    Build one row per (Season, TeamID) with season-level stats.
    Use only regular-season games for that season (no future games, no tourney games in train target).
    Returns DataFrame with Season, TeamID, and feature columns.
    """
    raise NotImplementedError


def get_feature_columns() -> list:
    """Return list of feature column names produced by build_team_season_stats (for matchup diff)."""
    raise NotImplementedError  

def build_advanced_features(team_games: pd.DataFrame, season_min: Optional[int] = None, season_max: Optional[int] = None) -> pd.DataFrame:
    """
    Build advanced features from team games. Return a DataFrame with the advanced features.
    """
    df = team_games

    if season_min is not None:
        df = df[df["Season"] >= season_min]
    if season_max is not None:
        df = df[df["Season"] <= season_max]

    df["margin"] = df["points_for"] - df["points_against"]

    # Possessions estimate (standard box-score approximation).
    team_poss = df["fga"] - df["or_"] + df["to"] + 0.475 * df["fta"]
    opp_poss = df["opp_fga"] - df["opp_or"] + df["opp_to"] + 0.475 * df["opp_fta"]
    df["possessions"] = (team_poss + opp_poss) / 2

    # Effective field goal %.
    df["efg_pct"] = (df["fgm"] + 0.5 * df["fgm3"]) / df["fga"]

    # Turnover rate.
    df["tov_pct"] = df["to"] / df["possessions"]

    # Offensive rebounding rate.
    df["orb_pct"] = df["or_"] / (df["or_"] + df["opp_dr"])

    # Free throw rate.
    df["ftr"] = df["fta"] / df["fga"]

    # Efficiency (points per 100 possessions).
    df["off_eff"] = 100 * df["points_for"] / df["possessions"]
    df["def_eff"] = 100 * df["points_against"] / df["possessions"]
    df["net_eff"] = df["off_eff"] - df["def_eff"]

    return df

def get_recent_features(team_games: pd.DataFrame, num_games: Optional[int] = 10, season_min: Optional[int] = None, season_max: Optional[int] = None) -> pd.DataFrame:
    """
    Get recent features from team games. Return a DataFrame with the recent features.
    """
    df = team_games
    if season_min is not None:
        df = df[df["Season"] >= season_min]
    if season_max is not None:
        df = df[df["Season"] <= season_max]
    
    recent_features = (
        df.sort_values(["Season","TeamID","DayNum"])
        .groupby(["Season", "TeamID"], group_keys = False)
        .tail(num_games)
        .groupby(["Season", "TeamID"])
        .agg(
            lastx_win_pct = ("win", "mean"),
            lastx_avg_margin = ("margin", "mean"),
            lastx_off_eff = ("off_eff", "mean"),
            lastx_def_eff = ("def_eff", "mean"),
            lastx_net_eff = ("net_eff", "mean"),
        )
        .reset_index()
    )

    return recent_features

def combine_features(season_features: pd.DataFrame, recent_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combine season and recent features. Return a DataFrame with the combined features.
    """
    return pd.merge(season_features, recent_features, on = ["Season", "TeamID"], how = "left")

