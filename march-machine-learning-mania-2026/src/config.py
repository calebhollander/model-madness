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

# Kaggle Stage 2 template (pairwise IDs for the competition season).
SAMPLE_SUBMISSION_STAGE2 = DATA_RAW / "SampleSubmissionStage2.csv"
# Artifacts for 2026 inference (see `train_logreg.save_logreg_artifacts`, `predict_2026`).
LOGREG_FEATURE_NAMES_JSON = MODELS_DIR / "logreg_men_features.json"
PREDICTIONS_2026_LOGREG_XGB = OUTPUTS_DIR / "predictions_2026_logreg_xgb.csv"


def ensure_project_dirs() -> None:
    """Create standard output directories if missing."""
    for p in (DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
        p.mkdir(parents=True, exist_ok=True)

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
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_SEED,
    },
}

# Submission / calibration
SUBMISSION_PROB_MIN = 1e-15  # Clip lower bound to avoid log loss blow-up
SUBMISSION_PROB_MAX = 1.0 - 1e-15  # Clip upper bound
