"""
Matchup generation: convert team-level features to pairwise dataset.

Role: For each (Season, TeamA, TeamB) with a known outcome (from tourney results),
build one row with FEATURE DIFFERENCES (Team A stat minus Team B stat). Target:
1 if Team A (first team in ID order or WTeamID) wins, else 0. For submission,
generate (Season, TeamA, TeamB) from sample submission and apply same diff logic.
Always use differences—never raw team stats—to keep features comparable and avoid leakage.
"""

from typing import Optional

# Input: team_stats DataFrame (Season, TeamID, feat1, feat2, ...), games_with_outcomes (Season, WTeamID, LTeamID).
# Output: DataFrame with Season, TeamID_A, TeamID_B, diff_feat1, diff_feat2, ..., target (0/1).


def build_matchup_dataset(
    team_stats,
    games_with_outcomes,
    feature_columns: list,
    target_name: str = "target",
):
    """
    Build pairwise rows: for each game, compute feature_diffs (TeamA - TeamB)
    and target (1 if TeamA wins, 0 else). TeamA/TeamB ordering must be consistent
    with submission ID (e.g. lower TeamID first).
    """
    raise NotImplementedError


def build_submission_matchups(sample_submission_df, team_stats, feature_columns: list):
    """
    From SampleSubmissionStage2 (ID = Season_TeamID1_TeamID2), parse IDs and
    build matchup rows with same diff features. No target column.
    """
    raise NotImplementedError
