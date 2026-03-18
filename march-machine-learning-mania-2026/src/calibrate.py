"""
Probability calibration and clipping before submission.

Role: Raw model probabilities are often poorly calibrated. Calibrate (e.g. Platt
scaling or isotonic regression) on a holdout or CV set. Then clip extreme
probabilities to [min_p, max_p] to avoid log loss penalties (Kaggle metric).
"""

from typing import Literal

# Input: predicted probabilities, optional true labels for calibration fit.
# Output: calibrated and/or clipped probabilities in (0, 1).


def calibrate_probs(
    probs,
    y_true=None,
    method: Literal["platt", "isotonic", "none"] = "platt",
):
    """
    Calibrate probabilities. If method != 'none', fit calibrator on (probs, y_true)
    and return calibrated probs. Use time-based split for calibration fit (no leakage).
    """
    raise NotImplementedError


def clip_probs(probs, min_p: float = 1e-15, max_p: float = 1.0 - 1e-15):
    """
    Clip probabilities to [min_p, max_p]. Avoids log(0) or log(1) which
    blow up log loss. Use config.SUBMISSION_PROB_MIN / SUBMISSION_PROB_MAX.
    """
    raise NotImplementedError
