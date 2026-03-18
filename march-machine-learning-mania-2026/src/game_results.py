"""
Game-level result shaping helpers.

Role: Turn raw regular-season detailed results into a "long" format with one
row per (Season, DayNum, TeamID, OppTeamID, win) including both team and
opponent box-score stats. This is the canonical place for winner/loser row
extraction so notebooks and feature code can reuse the same logic.
"""

from typing import Optional

import pandas as pd


TEAM_COLS = {
    "Season": "Season",
    "DayNum": "DayNum",
    "TeamID": "TeamID",
    "OppTeamID": "OppTeamID",
    "win": "win",
    "points_for": "points_for",
    "points_against": "points_against",
    "fgm": "fgm",
    "fga": "fga",
    "fgm3": "fgm3",
    "fga3": "fga3",
    "ftm": "ftm",
    "fta": "fta",
    "or_": "or_",
    "dr": "dr",
    "ast": "ast",
    "to": "to",
    "stl": "stl",
    "blk": "blk",
    "pf": "pf",
    "opp_fgm": "opp_fgm",
    "opp_fga": "opp_fga",
    "opp_fgm3": "opp_fgm3",
    "opp_fga3": "opp_fga3",
    "opp_ftm": "opp_ftm",
    "opp_fta": "opp_fta",
    "opp_or": "opp_or",
    "opp_dr": "opp_dr",
    "opp_ast": "opp_ast",
    "opp_to": "opp_to",
    "opp_stl": "opp_stl",
    "opp_blk": "opp_blk",
    "opp_pf": "opp_pf",
}


def make_long_regular_season_results(
    games: pd.DataFrame,
    season_min: Optional[int] = None,
    season_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert detailed regular-season results to long format.

    Input columns (from MRegularSeasonDetailedResults):
        Season, DayNum, WTeamID, WScore, LTeamID, LScore, WFGM, WFGA, WFGM3, WFGA3,
        WFTM, WFTA, WOR, WDR, WAst, WTO, WStl, WBlk, WPF,
        LFGM, LFGA, LFGM3, LFGA3, LFTM, LFTA, LOR, LDR, LAst, LTO, LStl, LBlk, LPF

    Output columns:
        Season, DayNum, TeamID, OppTeamID, win,
        points_for, points_against,
        fgm, fga, fgm3, fga3, ftm, fta, or_, dr, ast, to, stl, blk, pf,
        opp_fgm, opp_fga, opp_fgm3, opp_fga3, opp_ftm, opp_fta,
        opp_or, opp_dr, opp_ast, opp_to, opp_stl, opp_blk, opp_pf
    """
    df = games
    if season_min is not None:
        df = df[df["Season"] >= season_min]
    if season_max is not None:
        df = df[df["Season"] <= season_max]

    winner_rows = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["WTeamID"],
            "OppTeamID": df["LTeamID"],
            "win": 1,
            "points_for": df["WScore"],
            "points_against": df["LScore"],
            "fgm": df["WFGM"],
            "fga": df["WFGA"],
            "fgm3": df["WFGM3"],
            "fga3": df["WFGA3"],
            "ftm": df["WFTM"],
            "fta": df["WFTA"],
            "or_": df["WOR"],
            "dr": df["WDR"],
            "ast": df["WAst"],
            "to": df["WTO"],
            "stl": df["WStl"],
            "blk": df["WBlk"],
            "pf": df["WPF"],
            "opp_fgm": df["LFGM"],
            "opp_fga": df["LFGA"],
            "opp_fgm3": df["LFGM3"],
            "opp_fga3": df["LFGA3"],
            "opp_ftm": df["LFTM"],
            "opp_fta": df["LFTA"],
            "opp_or": df["LOR"],
            "opp_dr": df["LDR"],
            "opp_ast": df["LAst"],
            "opp_to": df["LTO"],
            "opp_stl": df["LStl"],
            "opp_blk": df["LBlk"],
            "opp_pf": df["LPF"],
        }
    )

    loser_rows = pd.DataFrame(
        {
            "Season": df["Season"],
            "DayNum": df["DayNum"],
            "TeamID": df["LTeamID"],
            "OppTeamID": df["WTeamID"],
            "win": 0,
            "points_for": df["LScore"],
            "points_against": df["WScore"],
            "fgm": df["LFGM"],
            "fga": df["LFGA"],
            "fgm3": df["LFGM3"],
            "fga3": df["LFGA3"],
            "ftm": df["LFTM"],
            "fta": df["LFTA"],
            "or_": df["LOR"],
            "dr": df["LDR"],
            "ast": df["LAst"],
            "to": df["LTO"],
            "stl": df["LStl"],
            "blk": df["LBlk"],
            "pf": df["LPF"],
            "opp_fgm": df["WFGM"],
            "opp_fga": df["WFGA"],
            "opp_fgm3": df["WFGM3"],
            "opp_fga3": df["WFGA3"],
            "opp_ftm": df["WFTM"],
            "opp_fta": df["WFTA"],
            "opp_or": df["WOR"],
            "opp_dr": df["WDR"],
            "opp_ast": df["WAst"],
            "opp_to": df["WTO"],
            "opp_stl": df["WStl"],
            "opp_blk": df["WBlk"],
            "opp_pf": df["WPF"],
        }
    )

    long_df = pd.concat([winner_rows, loser_rows], ignore_index=True)
    # Sort for readability; not required for correctness.
    long_df = long_df.sort_values(["Season", "DayNum", "TeamID", "OppTeamID"]).reset_index(drop=True)
    return long_df
