"""
Submission generation: read sample submission, predict, write submission.csv.

Role: Load SampleSubmissionStage2.csv (or Stage1), parse IDs into Season,
TeamID1, TeamID2; generate predictions using the trained ensemble (or single
model); clip probabilities; write submission.csv with columns ID, Pred.
"""

from pathlib import Path
from typing import Union

# Input: path to sample submission; model or ensemble predict function; optional path to team_stats/features.
# Output: submission.csv with ID and Pred (probability).


def load_sample_submission(path: Union[str, Path]):
    """Load sample submission CSV. Returns DataFrame with ID column (e.g. 2026_1101_1102)."""
    raise NotImplementedError


def generate_predictions(
    ids_series,
    model_or_ensemble,
    matchup_df=None,
    feature_columns: list = None,
):
    """
    For each ID, get matchup row (or build from IDs), run model predict_proba,
    return array of probabilities. Must match order of ids_series.
    """
    raise NotImplementedError


def write_submission(ids, preds, path: Union[str, Path]) -> Path:
    """Write submission file: ID, Pred. Clip preds to (0, 1) before writing."""
    raise NotImplementedError
