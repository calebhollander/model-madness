"""
Central configuration for the March Madness ML pipeline.

Single source of truth for paths (raw, interim, processed data; models; outputs)
and constants (random seed, model hyperparameters). All pipeline scripts should
import from here.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (relative to this file's parent's parent = project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw Kaggle data: use DATA_RAW if CSVs are in data/raw/; else point to project root.
DATA_RAW = PROJECT_ROOT / "data" / "raw"
# If raw CSVs live in project root (march-machine-learning-mania-2026/), use:
# DATA_RAW = PROJECT_ROOT

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# -----------------------------------------------------------------------------
# Reproducibility and model parameters
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# Placeholder: tune via validation. Always use time-based CV; never random split.
MODEL_PARAMS = {
    "logreg": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
    },
    "xgb": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": RANDOM_SEED,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    },
}

# Submission / calibration
SUBMISSION_PROB_MIN = 1e-15  # Clip lower bound to avoid log loss blow-up
SUBMISSION_PROB_MAX = 1.0 - 1e-15  # Clip upper bound
