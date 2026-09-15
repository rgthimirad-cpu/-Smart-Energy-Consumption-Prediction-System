from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURE_DATA,
    PROCESSED_DIR,
    SPLIT_ROWS,
    TARGETS,
    TRAIN_END_ROW,
    validate_repo_layout,
)


def load_feature_data() -> pd.DataFrame:
    validate_repo_layout()
    df = pd.read_csv(FEATURE_DATA, parse_dates=["DateTime"])
    if len(df) != 52416:
        raise ValueError(f"Expected 52,416 rows in feature data, found {len(df):,}.")
    return df.sort_values("DateTime").reset_index(drop=True)


def load_unscaled_time_series_splits() -> dict[str, pd.DataFrame]:
    """Return the unscaled 70/15/15 partitions defined in the team plan."""
    df = load_feature_data()
    return {
        name: df.iloc[start:end].copy().reset_index(drop=True)
        for name, (start, end) in SPLIT_ROWS.items()
    }


def load_scaled_regression_splits() -> dict[str, pd.DataFrame]:
    """Load the preprocessing team's scaled train/validation/test files."""
    validate_repo_layout()
    return {
        name: pd.read_csv(PROCESSED_DIR / f"{name}.csv", parse_dates=["DateTime"])
        for name in ["train", "validation", "test"]
    }


def target_min_max() -> dict[str, tuple[float, float]]:
    """Reproduce target MinMax scaling from the original training partition."""
    feat = load_feature_data().iloc[:TRAIN_END_ROW]
    return {
        target: (float(feat[target].min()), float(feat[target].max()))
        for target in TARGETS.values()
    }


def inverse_target(values, target: str, min_max: dict[str, tuple[float, float]] | None = None):
    params = min_max or target_min_max()
    col_min, col_max = params[target]
    arr = np.asarray(values, dtype=float)
    return arr * (col_max - col_min) + col_min


def regression_xy(df: pd.DataFrame, target: str):
    """Prepare scaled regression inputs while excluding all current zone targets.

    Lagged and rolling zone features remain available because they are based on
    past observations. The three raw current-consumption columns are removed to
    avoid target and cross-zone current-value leakage.
    """
    drop_cols = ["DateTime", *TARGETS.values()]
    X = df.drop(columns=drop_cols).copy()
    y = df[target].astype(float).copy()
    dates = df["DateTime"].copy()

    non_numeric = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric regression features found: {non_numeric}")
    X = X.astype(float)
    if X.isna().any().any() or y.isna().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values found in regression data. Feature columns: {bad}")
    return X, y, dates


def combine_scaled_train_validation(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train, validation], ignore_index=True)
