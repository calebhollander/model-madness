"""
Optional bracket simulation from predicted probabilities.

Role: Given a probability function for P(Team A beats Team B) and the tourney
slot structure, simulate the tournament (e.g. sample winner per game or take
argmax). Output: bracket outcome, champion, optional summary to outputs/simulation_results.csv.
"""

from pathlib import Path
from typing import Callable, Optional

# Input: prob_fn(season, team_a, team_b) -> float; slots DataFrame (Season, Slot, StrongSeed, WeakSeed).
# Output: simulated bracket (e.g. dict of slot -> winner), optional CSV of results.


def load_bracket_slots(data_dir=None) -> "pd.DataFrame":
    """Load MNCAATourneySlots (and seeds) to get slot -> (StrongSeed, WeakSeed) per season."""
    raise NotImplementedError


def simulate_tournament(
    prob_fn: Callable[[int, int, int], float],
    slots_df,
    seeds_df,
    season: int,
    method: str = "sample",
) -> dict:
    """
    Simulate one tournament: for each slot, resolve Strong vs Weak (by seed -> team),
    get prob(Strong wins), then sample or take argmax. Return dict slot -> winning TeamID.
    """
    raise NotImplementedError


def run_simulation(
    prob_fn: Callable[[int, int, int], float],
    seasons: list = None,
    output_path: Optional[Path] = None,
):
    """Run simulation for given seasons; optionally write results to outputs/simulation_results.csv."""
    raise NotImplementedError
