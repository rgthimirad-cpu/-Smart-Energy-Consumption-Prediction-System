from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CONTEXT_FEATURES,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_FEATURE_TEMPLATE,
    MAD_ZSCORE_THRESHOLD,
    RANDOM_STATE,
    ZSCORE_THRESHOLD,
)
from .data import (
    build_zone_matrix,
    split_by_row,
)
from .methods import (
    IsolationForestDetector,
    mad_flags,
    statistical_threshold_flags,
)
from .raw_reference import (
    compare_raw_vs_cleaned,
)


HANDOFF_COLUMNS = [
    "DateTime",
    "Zone",
    "Value_Watts",

    "Isolation_Forest_Score",
    "Isolation_Forest_Flag",

    "Zscore_1h",
    "Zscore_24h",
    "Statistical_Flag",

    "MAD_Zscore",
    "MAD_Flag",

    "Deviation_vs_Yesterday_Watts",
    "Anomaly_Severity",

    "Raw_MAD_Zscore",
    "Raw_Anomaly_Flag",

    "Signals_Agreeing",
    "Severity_Level",
    "Is_Anomaly",

    "Raw_Value_Watts",
    "Changed_By_Preprocessing",
    "Suppressed_Amount_Watts",

    "Eligible_For_Scoring",
]


def _severity_level(
    signals_agreeing: pd.Series,
    extreme: pd.Series,
    raw_anomaly: pd.Series,
    changed_by_preprocessing: pd.Series,
) -> pd.Series:
    """
    Assign dashboard severity.

    Normal:
        no detector fires.

    Warning:
        exactly one detector fires.

    Critical:
        two or more detector signals agree,
        OR a detector is extremely strong,
        OR raw data independently indicates an anomaly,
        OR preprocessing changed a value that was already flagged.
    """

    level = pd.Series(
        "Normal",
        index=signals_agreeing.index,
        dtype="object",
    )

    one_signal = (
        signals_agreeing == 1
    )

    level.loc[one_signal] = "Warning"

    critical = (
        (signals_agreeing >= 2)
        | extreme
        | raw_anomaly
        | (
            changed_by_preprocessing
            & (signals_agreeing >= 1)
        )
    )

    level.loc[critical] = "Critical"

    return level


def score_full_dataset(
    df: pd.DataFrame,
    zone: str,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    mad_threshold: float = MAD_ZSCORE_THRESHOLD,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Score the entire dataset.

    Isolation Forest:
        fit only on the chronological training split.

    Statistical:
        uses precomputed leakage-safe rolling statistics.

    MAD:
        uses trailing robust statistics.

    Raw reference:
        optionally checks the original raw data for anomalies suppressed
        during preprocessing.
    """

    target = f"{zone}_Power_Consumption"

    # ------------------------------------------------------------------
    # Train Isolation Forest
    # ------------------------------------------------------------------

    splits = split_by_row(df)

    X_train, _, _ = build_zone_matrix(
        splits["train"],
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    if X_train.empty:
        raise ValueError(
            f"No eligible training rows for {zone}."
        )

    detector = IsolationForestDetector(
        contamination=contamination,
        random_state=random_state,
    )

    detector.fit(X_train)

    # ------------------------------------------------------------------
    # Score entire dataset
    # ------------------------------------------------------------------

    X_full, eligible, _ = build_zone_matrix(
        df,
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    if_score = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if_flag = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    if_score.loc[eligible] = (
        detector.score(X_full)
    )

    if_flag.loc[eligible] = (
        detector.flag(X_full)
    )

    # ------------------------------------------------------------------
    # Statistical detector
    # ------------------------------------------------------------------

    stat = statistical_threshold_flags(
        df,
        zone,
        threshold=zscore_threshold,
    )

    stat_flag = (
        stat.flag
        .fillna(False)
        .astype(bool)
    )

    # ------------------------------------------------------------------
    # MAD detector
    # ------------------------------------------------------------------

    mad_z, mad_flag = mad_flags(
        df[target],
        threshold=mad_threshold,
    )

    mad_flag = (
        mad_flag
        .fillna(False)
        .astype(bool)
    )

    # ------------------------------------------------------------------
    # Raw data detector
    # ------------------------------------------------------------------

    raw_compare = compare_raw_vs_cleaned(
        df,
        target,
        threshold=mad_threshold,
    )

    # ------------------------------------------------------------------
    # Default raw fields
    # ------------------------------------------------------------------

    if raw_compare is None:

        raw_mad_z = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        raw_flag = pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

        raw_value = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        changed_by_preprocessing = pd.Series(
            False,
            index=df.index,
            dtype=bool,
        )

        suppressed_amount = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    else:

        raw_compare = (
            raw_compare
            .set_index("DateTime")
            .reindex(df["DateTime"])
            .reset_index()
        )

        raw_mad_z = (
            raw_compare["Raw_MAD_Zscore"]
            .set_axis(df.index)
        )

        raw_flag = (
            raw_compare["Raw_Anomaly_Flag"]
            .fillna(False)
            .astype(bool)
            .set_axis(df.index)
        )

        raw_value = (
            raw_compare["Raw_Value_Watts"]
            .set_axis(df.index)
        )

        changed_by_preprocessing = (
            raw_compare[
                "Changed_By_Preprocessing"
            ]
            .fillna(False)
            .astype(bool)
            .set_axis(df.index)
        )

        suppressed_amount = (
            raw_compare[
                "Suppressed_Amount_Watts"
            ]
            .set_axis(df.index)
        )

    # ------------------------------------------------------------------
    # Detector agreement
    # ------------------------------------------------------------------

    signals_agreeing = (
        if_flag.astype(int)
        + stat_flag.astype(int)
        + mad_flag.astype(int)
        + raw_flag.astype(int)
    )

    # Recall-oriented combined anomaly flag.
    combined_flag = (
        signals_agreeing > 0
    )

    # ------------------------------------------------------------------
    # Extreme signal
    # ------------------------------------------------------------------

    extreme = (
        (
            stat.severity
            .fillna(0)
            > 2 * zscore_threshold
        )
        |
        (
            mad_z.abs()
            .fillna(0)
            > 2 * mad_threshold
        )
        |
        (
            raw_mad_z.abs()
            .fillna(0)
            > 2 * mad_threshold
        )
    )

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    severity_level = _severity_level(
        signals_agreeing=signals_agreeing,
        extreme=extreme,
        raw_anomaly=raw_flag,
        changed_by_preprocessing=changed_by_preprocessing,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    result = pd.DataFrame(
        {
            "DateTime": df["DateTime"],
            "Zone": zone,
            "Value_Watts": df[target],

            "Isolation_Forest_Score": if_score,
            "Isolation_Forest_Flag": if_flag,

            "Zscore_1h": stat.zscore_1h,
            "Zscore_24h": stat.zscore_24h,
            "Statistical_Flag": stat_flag,

            "MAD_Zscore": mad_z,
            "MAD_Flag": mad_flag,

            "Deviation_vs_Yesterday_Watts": (
                stat.vs_yesterday
            ),

            "Anomaly_Severity": stat.severity,

            "Raw_MAD_Zscore": raw_mad_z,
            "Raw_Anomaly_Flag": raw_flag,

            "Signals_Agreeing": signals_agreeing,
            "Severity_Level": severity_level,
            "Is_Anomaly": combined_flag,

            "Raw_Value_Watts": raw_value,
            "Changed_By_Preprocessing": (
                changed_by_preprocessing
            ),
            "Suppressed_Amount_Watts": (
                suppressed_amount
            ),

            "Eligible_For_Scoring": eligible,
        }
    )

    return result[HANDOFF_COLUMNS]