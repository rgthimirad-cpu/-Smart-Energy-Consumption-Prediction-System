from __future__ import annotations

import pandas as pd

from .config import (
    CLEANED_DATA,
    FEATURE_DATA,
    RAW_DATA,
    SPLIT_ROWS,
    TARGETS,
    TOTAL_ROWS,
    validate_repo_layout,
)


# ---------------------------------------------------------------------------
# Feature-engineered data
# ---------------------------------------------------------------------------

def load_feature_data() -> pd.DataFrame:
    """
    Load the full unscaled feature-engineered dataset.

    The anomaly detector operates on the original Watts values and the
    leakage-safe lag/rolling features created by the preprocessing team.
    """

    validate_repo_layout()

    df = pd.read_csv(
        FEATURE_DATA,
        parse_dates=["DateTime"],
    )

    if len(df) != TOTAL_ROWS:
        raise ValueError(
            f"Expected {TOTAL_ROWS:,} rows in feature data, "
            f"found {len(df):,}."
        )

    if "DateTime" not in df.columns:
        raise ValueError("Feature data must contain a DateTime column.")

    df = (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )

    return df


# ---------------------------------------------------------------------------
# Cleaned data
# ---------------------------------------------------------------------------

def load_cleaned_data() -> pd.DataFrame | None:
    """
    Load cleaned data if available.

    This is optional and is mainly useful for diagnostics comparing the
    preprocessing output against the original raw data.
    """

    if not CLEANED_DATA.exists():
        return None

    df = pd.read_csv(
        CLEANED_DATA,
        parse_dates=["DateTime"],
    )

    return (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

_RAW_COLUMN_MAP = {
    "Wind Speed": "Wind_Speed",
    "general diffuse flows": "General_Diffuse_Flows",
    "diffuse flows": "Diffuse_Flows",
    "Zone 1 Power Consumption": "Zone_1_Power_Consumption",
    "Zone 2 Power Consumption": "Zone_2_Power_Consumption",
    "Zone 3 Power Consumption": "Zone_3_Power_Consumption",
}


def load_raw_data() -> pd.DataFrame | None:
    """
    Load the original raw dataset.

    The raw dataset uses the original Kaggle naming convention. Repeated
    whitespace is normalized before column names are mapped.

    Returns None when the raw file is unavailable.
    """

    if not RAW_DATA.exists():
        return None

    df = pd.read_csv(RAW_DATA)

    # Normalize repeated whitespace.
    df.columns = [
        " ".join(str(column).split())
        for column in df.columns
    ]

    df = df.rename(columns=_RAW_COLUMN_MAP)

    if "DateTime" not in df.columns:
        raise ValueError(
            "Raw dataset does not contain a DateTime column."
        )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        format="%m/%d/%Y %H:%M",
        errors="raise",
    )

    missing = [
        column
        for column in TARGETS.values()
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Raw data is missing expected power-consumption columns: "
            f"{missing}"
        )

    return (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------------

def split_by_row(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the dataset using the project's agreed chronological boundaries.

    Returns:
        {
            "train": ...,
            "validation": ...,
            "test": ...
        }
    """

    if len(df) != TOTAL_ROWS:
        raise ValueError(
            f"Expected {TOTAL_ROWS:,} rows before splitting, "
            f"found {len(df):,}."
        )

    return {
        name: df.iloc[start:end].copy().reset_index(drop=True)
        for name, (start, end) in SPLIT_ROWS.items()
    }


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def zone_feature_columns(
    zone: str,
    feature_template: list[str],
) -> list[str]:
    """Expand {Z} in a feature template."""

    return [
        feature.format(Z=zone)
        for feature in feature_template
    ]


def eligible_mask(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    Return rows for which all requested features are available.
    """

    missing = [
        column
        for column in feature_cols
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Required feature columns are missing: "
            f"{missing}"
        )

    return df[feature_cols].notna().all(axis=1)


def build_zone_matrix(
    df: pd.DataFrame,
    zone: str,
    feature_template: list[str],
    context_features: list[str],
):
    """
    Build the Isolation Forest feature matrix for a zone.
    """

    feature_cols = (
        zone_feature_columns(zone, feature_template)
        + context_features
    )

    mask = eligible_mask(df, feature_cols)

    X = df.loc[
        mask,
        feature_cols,
    ].astype(float)

    return X, mask, feature_cols