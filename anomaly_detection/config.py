from pathlib import Path


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DIR = REPO_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

FEATURE_DATA = PROCESSED_DIR / "feature_engineered_energy_data.csv"
CLEANED_DATA = PROCESSED_DIR / "cleaned_energy_data.csv"
RAW_DATA = RAW_DIR / "Tetuan_City_power_consumption.csv"


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

MODELS_DIR = MODULE_DIR / "models"

REPORTS_DIR = MODULE_DIR / "docs"
PUBLIC_RESULTS_DIR = REPORTS_DIR / "results"

FIGURES_DIR = MODULE_DIR / "notebooks" / "figures"

INTERNAL_RESULTS_DIR = MODULE_DIR / "internal_results"
PREDICTIONS_DIR = INTERNAL_RESULTS_DIR / "predictions"
TUNING_DIR = INTERNAL_RESULTS_DIR / "tuning"

# Files intended for other project components/dashboard.
OUTPUT_DIR = MODULE_DIR / "outputs"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

TARGETS = {
    "Zone_1": "Zone_1_Power_Consumption",
    "Zone_2": "Zone_2_Power_Consumption",
    "Zone_3": "Zone_3_Power_Consumption",
}


# Dataset contains 52,416 rows at 10-minute frequency.
TOTAL_ROWS = 52_416

# Same chronological boundaries used by forecasting/regression.
TRAIN_END_ROW = 36_691
VALIDATION_END_ROW = 44_553

SPLIT_ROWS = {
    "train": (0, TRAIN_END_ROW),
    "validation": (TRAIN_END_ROW, VALIDATION_END_ROW),
    "test": (VALIDATION_END_ROW, TOTAL_ROWS),
}


# 144 x 10-minute observations = 24 hours.
SEASONAL_PERIOD = 144

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Isolation Forest features
# ---------------------------------------------------------------------------

# These are the features that already exist in:
# data/processed/feature_engineered_energy_data.csv
#
# {Z} is replaced by Zone_1 / Zone_2 / Zone_3.

ISOLATION_FOREST_FEATURE_TEMPLATE = [
    "{Z}_Power_Consumption",
    "{Z}_lag_1",
    "{Z}_lag_6",
    "{Z}_lag_144",
    "{Z}_roll_mean_1h",
    "{Z}_roll_std_1h",
    "{Z}_roll_min_1h",
    "{Z}_roll_max_1h",
    "{Z}_roll_mean_24h",
    "{Z}_roll_std_24h",
]


# Time/weather contextual variables.
CONTEXT_FEATURES = [
    "Hour",
    "Is_Weekend",
    "Is_Peak_Hour",
    "Temperature",
    "Humidity",
    "Temp_Humidity_Index",
]


# ---------------------------------------------------------------------------
# Default anomaly-detection parameters
# ---------------------------------------------------------------------------

ISOLATION_FOREST_CONTAMINATION = 0.02

ZSCORE_THRESHOLD = 3.0

MAD_ZSCORE_THRESHOLD = 3.5

SYNTHETIC_ANOMALY_RATE = 0.02


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

CONTAMINATION_CANDIDATES = [
    0.005,
    0.01,
    0.02,
    0.05,
]

ZSCORE_THRESHOLD_CANDIDATES = [
    2.5,
    3.0,
    3.5,
    4.0,
]

MAD_ZSCORE_THRESHOLD_CANDIDATES = [
    3.0,
    3.5,
    4.0,
    5.0,
]


# ---------------------------------------------------------------------------
# Rolling feature configuration
# ---------------------------------------------------------------------------

WINDOW_1H = 6
WINDOW_24H = 144

MAD_WINDOW = 144
MAD_MIN_PERIODS = 36

# Prevent extremely small standard deviations / MAD values from generating
# meaningless huge scores during very stable periods.
MIN_SCALE_FRACTION = 0.02


# ---------------------------------------------------------------------------
# Raw-vs-cleaned configuration
# ---------------------------------------------------------------------------

CAPPING_TOLERANCE_WATTS = 1.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create all output directories used by the anomaly-detection module."""

    directories = [
        MODELS_DIR,
        PUBLIC_RESULTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        TUNING_DIR,
        OUTPUT_DIR,
    ]

    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


def validate_repo_layout() -> None:
    """Validate that the required preprocessing output exists."""

    if not FEATURE_DATA.exists():
        raise FileNotFoundError(
            "Required preprocessing output was not found:\n"
            f"- {FEATURE_DATA}\n\n"
            "Make sure anomaly_detection/ is located directly inside "
            "the repository root and that the preprocessing team has "
            "generated feature_engineered_energy_data.csv."
        )

    ensure_output_dirs()