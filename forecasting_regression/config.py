from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FEATURE_DATA = PROCESSED_DIR / "feature_engineered_energy_data.csv"

# All forecasting/regression outputs remain inside this folder.
MODELS_DIR = MODULE_DIR / "models"
REPORTS_DIR = MODULE_DIR / "docs"
PUBLIC_RESULTS_DIR = REPORTS_DIR / "results"
NOTEBOOK_DIR = MODULE_DIR / "notebooks"
FIGURES_DIR = NOTEBOOK_DIR / "figures"
INTERNAL_RESULTS_DIR = MODULE_DIR / "internal_results"
PREDICTIONS_DIR = INTERNAL_RESULTS_DIR / "predictions"
TUNING_DIR = INTERNAL_RESULTS_DIR / "tuning"

TARGETS = {
    "Zone_1": "Zone_1_Power_Consumption",
    "Zone_2": "Zone_2_Power_Consumption",
    "Zone_3": "Zone_3_Power_Consumption",
}

# Exact chronological boundaries documented in the team evaluation plan.
TRAIN_END_ROW = 36691
VALIDATION_END_ROW = 44553
TOTAL_ROWS = 52416
SPLIT_ROWS = {
    "train": (0, TRAIN_END_ROW),
    "validation": (TRAIN_END_ROW, VALIDATION_END_ROW),
    "test": (VALIDATION_END_ROW, TOTAL_ROWS),
}

SEASONAL_PERIOD = 144  # 10-minute observations per day: 6 * 24
RANDOM_STATE = 42
PROPHET_REGRESSORS = ["Temperature", "Humidity", "Is_Peak_Hour", "Is_Weekend"]


def ensure_output_dirs() -> None:
    for path in [
        MODELS_DIR / "arima_sarima",
        MODELS_DIR / "prophet",
        MODELS_DIR / "regression",
        PUBLIC_RESULTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
        TUNING_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_repo_layout() -> None:
    required = [
        FEATURE_DATA,
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "validation.csv",
        PROCESSED_DIR / "test.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required preprocessing outputs were not found:\n- " + "\n- ".join(missing)
        )
    ensure_output_dirs()
