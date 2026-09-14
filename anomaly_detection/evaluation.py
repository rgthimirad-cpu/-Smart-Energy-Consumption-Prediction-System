from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from .config import (
    CONTAMINATION_CANDIDATES,
    CONTEXT_FEATURES,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_FEATURE_TEMPLATE,
    MAD_ZSCORE_THRESHOLD,
    MAD_ZSCORE_THRESHOLD_CANDIDATES,
    RANDOM_STATE,
    SYNTHETIC_ANOMALY_RATE,
    ZSCORE_THRESHOLD,
    ZSCORE_THRESHOLD_CANDIDATES,
)
from .data import (
    build_zone_matrix,
)
from .methods import (
    IsolationForestDetector,
    inject_synthetic_anomalies,
    mad_flags,
    recompute_zone_features,
    statistical_threshold_flags,
)


BENCHMARK_COLUMNS = [
    "Model",
    "Zone",
    "Precision",
    "Recall",
    "F1",
    "Injected_Anomalies",
    "Flagged",
    "Notes",
]


def _metric_row(
    model_name: str,
    zone: str,
    truth: pd.Series,
    prediction,
    notes: str,
) -> dict:
    """Create a benchmark result row."""

    truth_array = (
        truth
        .astype(int)
        .to_numpy()
    )

    prediction_array = (
        np.asarray(prediction)
        .astype(int)
    )

    return {
        "Model": model_name,
        "Zone": zone,
        "Precision": float(
            precision_score(
                truth_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                truth_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                truth_array,
                prediction_array,
                zero_division=0,
            )
        ),
        "Injected_Anomalies": int(
            truth_array.sum()
        ),
        "Flagged": int(
            prediction_array.sum()
        ),
        "Notes": notes,
    }


def _prepare_synthetic_zone_data(
    base_df: pd.DataFrame,
    zone: str,
    random_state: int,
    rate: float,
):
    """
    Inject anomalies into the target column and rebuild the same zone
    features used by production Isolation Forest.
    """

    target = f"{zone}_Power_Consumption"

    injected_values, labels = (
        inject_synthetic_anomalies(
            base_df[target],
            rate=rate,
            random_state=random_state,
        )
    )

    synthetic_df = base_df.copy()

    synthetic_df[target] = injected_values

    # Rebuild all target-dependent lag/rolling features.
    synthetic_df = recompute_zone_features(
        synthetic_df,
        zone,
    )

    return synthetic_df, labels


def run_synthetic_benchmark(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    zone: str,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    zscore_threshold: float = ZSCORE_THRESHOLD,
    mad_threshold: float = MAD_ZSCORE_THRESHOLD,
    rate: float = SYNTHETIC_ANOMALY_RATE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Evaluate all three detectors against synthetic anomalies.

    IMPORTANT:
        The Isolation Forest feature space here is the SAME feature space
        used in production.

    The model is fitted only on clean training data.

    Synthetic anomalies are injected only into evaluation data.
    """

    target = f"{zone}_Power_Consumption"

    # ------------------------------------------------------------------
    # Training matrix
    # ------------------------------------------------------------------

    X_train, train_mask, feature_columns = build_zone_matrix(
        train_df,
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    if X_train.empty:
        raise ValueError(
            f"No eligible training rows available for {zone}."
        )

    # ------------------------------------------------------------------
    # Synthetic evaluation data
    # ------------------------------------------------------------------

    synthetic_df, labels = (
        _prepare_synthetic_zone_data(
            evaluation_df,
            zone,
            random_state,
            rate,
        )
    )

    X_eval, eval_mask, _ = build_zone_matrix(
        synthetic_df,
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    labels_eval = labels.loc[
        eval_mask
    ]

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------

    detector = IsolationForestDetector(
        contamination=contamination,
        random_state=random_state,
    )

    detector.fit(X_train)

    if_flags = detector.flag(
        X_eval
    )

    # ------------------------------------------------------------------
    # Statistical z-score
    # ------------------------------------------------------------------

    stat = statistical_threshold_flags(
        synthetic_df,
        zone,
        threshold=zscore_threshold,
    )

    stat_flags = (
        stat.flag
        .loc[eval_mask]
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # MAD
    # ------------------------------------------------------------------

    mad_z, mad_flag = mad_flags(
        synthetic_df[target],
        threshold=mad_threshold,
    )

    mad_flags_eval = (
        mad_flag
        .loc[eval_mask]
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # Benchmark results
    # ------------------------------------------------------------------

    rows = []

    rows.append(
        _metric_row(
            "Isolation Forest",
            zone,
            labels_eval,
            if_flags,
            (
                "Same production feature space; "
                f"contamination={contamination}; "
                "fit on clean train only"
            ),
        )
    )

    rows.append(
        _metric_row(
            "Statistical Threshold",
            zone,
            labels_eval,
            stat_flags,
            (
                f"|z| > {zscore_threshold}; "
                "rolling 1h/24h mean/std"
            ),
        )
    )

    rows.append(
        _metric_row(
            "Robust MAD Z-score",
            zone,
            labels_eval,
            mad_flags_eval,
            (
                f"|MAD z| > {mad_threshold}; "
                "trailing rolling MAD"
            ),
        )
    )

    result = pd.DataFrame(
        rows,
        columns=BENCHMARK_COLUMNS,
    )

    result["Notes"] = (
        result["Notes"]
        + f"; synthetic anomaly rate={rate}"
    )

    return result


def tune_all_thresholds(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    zone: str,
    contamination_candidates: list[float] = CONTAMINATION_CANDIDATES,
    zscore_candidates: list[float] = ZSCORE_THRESHOLD_CANDIDATES,
    mad_candidates: list[float] = MAD_ZSCORE_THRESHOLD_CANDIDATES,
    random_state: int = RANDOM_STATE,
    rate: float = SYNTHETIC_ANOMALY_RATE,
) -> dict:
    """
    Tune parameters against a synthetic benchmark on VALIDATION data.

    The TEST set is deliberately not used here.

    Returns:
        {
            "contamination": ...,
            "zscore_threshold": ...,
            "mad_threshold": ...,
            "grid": ...
        }
    """

    target = f"{zone}_Power_Consumption"

    # ------------------------------------------------------------------
    # Create one fixed synthetic validation dataset.
    #
    # Every candidate is therefore evaluated against exactly the same
    # injected anomaly locations and magnitudes.
    # ------------------------------------------------------------------

    synthetic_validation, labels = (
        _prepare_synthetic_zone_data(
            validation_df,
            zone,
            random_state,
            rate,
        )
    )

    # ------------------------------------------------------------------
    # Production feature matrices
    # ------------------------------------------------------------------

    X_train, _, _ = build_zone_matrix(
        train_df,
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    X_validation, validation_mask, _ = build_zone_matrix(
        synthetic_validation,
        zone,
        ISOLATION_FOREST_FEATURE_TEMPLATE,
        CONTEXT_FEATURES,
    )

    labels_validation = labels.loc[
        validation_mask
    ]

    # ------------------------------------------------------------------
    # Calculate statistical and MAD scores ONCE.
    # ------------------------------------------------------------------

    stat_default = statistical_threshold_flags(
        synthetic_validation,
        zone,
        threshold=1.0,
    )

    stat_severity = (
        stat_default.severity
        .loc[validation_mask]
        .to_numpy()
    )

    mad_z, _ = mad_flags(
        synthetic_validation[target],
        threshold=1.0,
    )

    mad_abs = (
        mad_z.abs()
        .loc[validation_mask]
        .to_numpy()
    )

    truth = (
        labels_validation
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # Tune Isolation Forest
    # ------------------------------------------------------------------

    if_rows = []

    for contamination in contamination_candidates:

        detector = IsolationForestDetector(
            contamination=contamination,
            random_state=random_state,
        )

        detector.fit(X_train)

        prediction = detector.flag(
            X_validation
        ).astype(int)

        if_rows.append(
            {
                "Model": "Isolation Forest",
                "Zone": zone,
                "Parameter": "contamination",
                "Parameter_Value": contamination,
                "Precision": precision_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "F1": f1_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Tune ordinary z-score
    # ------------------------------------------------------------------

    z_rows = []

    for threshold in zscore_candidates:

        prediction = (
            stat_severity > threshold
        ).astype(int)

        z_rows.append(
            {
                "Model": "Statistical Threshold",
                "Zone": zone,
                "Parameter": "zscore_threshold",
                "Parameter_Value": threshold,
                "Precision": precision_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "F1": f1_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Tune MAD
    # ------------------------------------------------------------------

    mad_rows = []

    for threshold in mad_candidates:

        prediction = (
            mad_abs > threshold
        ).astype(int)

        mad_rows.append(
            {
                "Model": "Robust MAD Z-score",
                "Zone": zone,
                "Parameter": "mad_threshold",
                "Parameter_Value": threshold,
                "Precision": precision_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
                "F1": f1_score(
                    truth,
                    prediction,
                    zero_division=0,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Combine tuning grid
    # ------------------------------------------------------------------

    grid = pd.DataFrame(
        if_rows + z_rows + mad_rows
    )

    # ------------------------------------------------------------------
    # Select best parameter for each detector
    #
    # idxmax() returns the first maximum, making selection deterministic.
    # ------------------------------------------------------------------

    if_grid = grid[
        grid["Model"] == "Isolation Forest"
    ]

    z_grid = grid[
        grid["Model"] == "Statistical Threshold"
    ]

    mad_grid = grid[
        grid["Model"] == "Robust MAD Z-score"
    ]

    best_contamination = float(
        if_grid.loc[
            if_grid["F1"].idxmax(),
            "Parameter_Value",
        ]
    )

    best_zscore = float(
        z_grid.loc[
            z_grid["F1"].idxmax(),
            "Parameter_Value",
        ]
    )

    best_mad = float(
        mad_grid.loc[
            mad_grid["F1"].idxmax(),
            "Parameter_Value",
        ]
    )

    return {
        "contamination": best_contamination,
        "zscore_threshold": best_zscore,
        "mad_threshold": best_mad,
        "grid": grid,
    }