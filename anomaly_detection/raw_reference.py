from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CAPPING_TOLERANCE_WATTS,
    MAD_ZSCORE_THRESHOLD,
)
from .data import load_raw_data
from .methods import mad_flags


def compare_raw_vs_cleaned(
    df: pd.DataFrame,
    zone_column: str,
    threshold: float = MAD_ZSCORE_THRESHOLD,
) -> pd.DataFrame | None:
    """
    Compare the feature-engineered/cleaned value with the original raw value.

    This provides an additional signal for extreme observations that may have
    been suppressed by upstream preprocessing.

    The output distinguishes:
        Raw_Value_Watts
        Raw_MAD_Zscore
        Raw_Anomaly_Flag
        Suppressed_Amount_Watts
        Changed_By_Preprocessing

    Note:
        A difference between raw and cleaned is evidence that preprocessing
        changed the value. It is called Changed_By_Preprocessing rather than
        Capped_By_Preprocessing to avoid assuming that every future change
        necessarily comes from IQR capping.
    """

    raw = load_raw_data()

    if raw is None:
        return None

    if zone_column not in raw.columns:
        raise KeyError(
            f"Raw data does not contain {zone_column}."
        )

    raw_values = raw[zone_column].astype(float)

    raw_mad_z, raw_mad_flag = mad_flags(
        raw_values,
        threshold=threshold,
    )

    raw_side = pd.DataFrame(
        {
            "DateTime": raw["DateTime"],
            "Raw_Value_Watts": raw_values,
            "Raw_MAD_Zscore": raw_mad_z,
            "Raw_Anomaly_Flag": raw_mad_flag,
        }
    )

    cleaned_side = (
        df[
            [
                "DateTime",
                zone_column,
            ]
        ]
        .rename(
            columns={
                zone_column: "Cleaned_Value_Watts"
            }
        )
    )

    merged = cleaned_side.merge(
        raw_side,
        on="DateTime",
        how="left",
    )

    merged["Suppressed_Amount_Watts"] = (
        merged["Raw_Value_Watts"]
        - merged["Cleaned_Value_Watts"]
    )

    merged["Changed_By_Preprocessing"] = (
        merged["Suppressed_Amount_Watts"]
        .abs()
        .gt(CAPPING_TOLERANCE_WATTS)
        .fillna(False)
    )

    return merged[
        [
            "DateTime",
            "Raw_Value_Watts",
            "Raw_MAD_Zscore",
            "Raw_Anomaly_Flag",
            "Suppressed_Amount_Watts",
            "Changed_By_Preprocessing",
        ]
    ]