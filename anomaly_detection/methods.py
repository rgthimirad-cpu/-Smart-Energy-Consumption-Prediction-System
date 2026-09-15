from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import (
    MAD_MIN_PERIODS,
    MAD_WINDOW,
    MIN_SCALE_FRACTION,
    RANDOM_STATE,
    WINDOW_1H,
    WINDOW_24H,
)


# ============================================================================
# Statistical anomaly detection
# ============================================================================

@dataclass
class StatisticalFlags:
    """Container for statistical anomaly-detection outputs."""

    zscore_1h: pd.Series
    zscore_24h: pd.Series
    vs_yesterday: pd.Series
    flag: pd.Series
    severity: pd.Series


def _zscore_severity(
    value: pd.Series,
    mean_1h: pd.Series,
    std_1h: pd.Series,
    mean_24h: pd.Series,
    std_24h: pd.Series,
    threshold: float,
    min_std_fraction: float = MIN_SCALE_FRACTION,
):
    """
    Calculate leakage-safe rolling z-scores.

    The rolling features supplied to this function are expected to have
    already been calculated using previous observations only.
    """

    overall_std = float(value.std())

    if not np.isfinite(overall_std) or overall_std <= 0:
        overall_std = 1.0

    floor = min_std_fraction * overall_std

    std_1h = (
        std_1h.astype(float)
        .clip(lower=floor)
        .replace(0, np.nan)
    )

    std_24h = (
        std_24h.astype(float)
        .clip(lower=floor)
        .replace(0, np.nan)
    )

    z_1h = (
        value.astype(float) - mean_1h.astype(float)
    ) / std_1h

    z_24h = (
        value.astype(float) - mean_24h.astype(float)
    ) / std_24h

    severity = pd.concat(
        [
            z_1h.abs(),
            z_24h.abs(),
        ],
        axis=1,
    ).max(axis=1)

    flag = (
        severity > threshold
    ).fillna(False)

    return z_1h, z_24h, severity, flag


def statistical_threshold_flags(
    df: pd.DataFrame,
    zone: str,
    threshold: float = 3.0,
) -> StatisticalFlags:
    """Detect anomalies using precomputed leakage-safe rolling features."""

    value = df[
        f"{zone}_Power_Consumption"
    ].astype(float)

    mean_1h = df[
        f"{zone}_roll_mean_1h"
    ].astype(float)

    std_1h = df[
        f"{zone}_roll_std_1h"
    ].astype(float)

    mean_24h = df[
        f"{zone}_roll_mean_24h"
    ].astype(float)

    std_24h = df[
        f"{zone}_roll_std_24h"
    ].astype(float)

    lag_144 = df[
        f"{zone}_lag_144"
    ].astype(float)

    (
        z_1h,
        z_24h,
        severity,
        flag,
    ) = _zscore_severity(
        value=value,
        mean_1h=mean_1h,
        std_1h=std_1h,
        mean_24h=mean_24h,
        std_24h=std_24h,
        threshold=threshold,
    )

    return StatisticalFlags(
        zscore_1h=z_1h,
        zscore_24h=z_24h,
        vs_yesterday=value - lag_144,
        flag=flag,
        severity=severity,
    )


# ============================================================================
# Feature recomputation for synthetic evaluation
# ============================================================================

def recompute_zone_features(
    df: pd.DataFrame,
    zone: str,
) -> pd.DataFrame:
    """
    Recompute target-dependent lag and rolling features after synthetic
    anomaly injection.

    The current observation is excluded from rolling baselines by shifting
    the series before calculating rolling statistics.
    """

    value_col = f"{zone}_Power_Consumption"

    if value_col not in df.columns:
        raise KeyError(
            f"Missing target column: {value_col}"
        )

    value = df[value_col].astype(float)

    shifted = value.shift(1)

    result = df.copy()

    result[f"{zone}_lag_1"] = value.shift(1)
    result[f"{zone}_lag_6"] = value.shift(6)
    result[f"{zone}_lag_144"] = value.shift(WINDOW_24H)

    result[f"{zone}_roll_mean_1h"] = (
        shifted
        .rolling(WINDOW_1H)
        .mean()
    )

    result[f"{zone}_roll_std_1h"] = (
        shifted
        .rolling(WINDOW_1H)
        .std()
    )

    result[f"{zone}_roll_min_1h"] = (
        shifted
        .rolling(WINDOW_1H)
        .min()
    )

    result[f"{zone}_roll_max_1h"] = (
        shifted
        .rolling(WINDOW_1H)
        .max()
    )

    result[f"{zone}_roll_mean_24h"] = (
        shifted
        .rolling(WINDOW_24H)
        .mean()
    )

    result[f"{zone}_roll_std_24h"] = (
        shifted
        .rolling(WINDOW_24H)
        .std()
    )

    return result


def statistical_threshold_flags_simple(
    frame: pd.DataFrame,
    threshold: float = 3.0,
) -> StatisticalFlags:
    """
    Statistical detector for compact benchmark frames.

    Expected columns:
        value
        roll_mean_1h
        roll_std_1h
        roll_mean_24h
        roll_std_24h
        lag_144
    """

    (
        z_1h,
        z_24h,
        severity,
        flag,
    ) = _zscore_severity(
        value=frame["value"],
        mean_1h=frame["roll_mean_1h"],
        std_1h=frame["roll_std_1h"],
        mean_24h=frame["roll_mean_24h"],
        std_24h=frame["roll_std_24h"],
        threshold=threshold,
    )

    vs_yesterday = (
        frame["value"]
        - frame["lag_144"]
    )

    return StatisticalFlags(
        zscore_1h=z_1h,
        zscore_24h=z_24h,
        vs_yesterday=vs_yesterday,
        flag=flag,
        severity=severity,
    )


# ============================================================================
# Robust MAD detector
# ============================================================================

MAD_CONSTANT = 0.6744897501960817


def rolling_mad_zscore(
    value: pd.Series,
    window: int = MAD_WINDOW,
    min_periods: int = MAD_MIN_PERIODS,
    min_mad_fraction: float = MIN_SCALE_FRACTION,
) -> pd.Series:
    """
    Calculate a trailing robust MAD z-score.

    The current observation is excluded from the rolling baseline.
    """

    value = value.astype(float)

    shifted = value.shift(1)

    rolling_median = (
        shifted
        .rolling(
            window=window,
            min_periods=min_periods,
        )
        .median()
    )

    def mad_from_window(
        window_values: np.ndarray,
    ) -> float:

        if len(window_values) == 0:
            return np.nan

        median = np.median(window_values)

        return float(
            np.median(
                np.abs(
                    window_values - median
                )
            )
        )

    rolling_mad = (
        shifted
        .rolling(
            window=window,
            min_periods=min_periods,
        )
        .apply(
            mad_from_window,
            raw=True,
        )
    )

    valid_values = value.dropna().to_numpy()

    if len(valid_values) == 0:
        global_mad = 1.0
    else:
        global_mad = float(
            np.median(
                np.abs(
                    valid_values
                    - np.median(valid_values)
                )
            )
        )

    if (
        not np.isfinite(global_mad)
        or global_mad <= 0
    ):
        global_mad = 1.0

    floor = min_mad_fraction * global_mad

    rolling_mad = (
        rolling_mad
        .clip(lower=floor)
        .replace(0, np.nan)
    )

    z = (
        MAD_CONSTANT
        * (value - rolling_median)
        / rolling_mad
    )

    return z


def mad_flags(
    value: pd.Series,
    threshold: float = 3.5,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """Return MAD z-score and boolean anomaly flag."""

    z = rolling_mad_zscore(
        value,
        **kwargs,
    )

    flag = (
        z.abs() > threshold
    ).fillna(False)

    return z, flag


# ============================================================================
# Isolation Forest
# ============================================================================

class IsolationForestDetector:
    """
    Isolation Forest detector with StandardScaler.

    n_jobs=1 is intentional.

    On Windows, using all CPU workers through joblib can cause long process
    overhead and Tkinter/Matplotlib cleanup messages when the process is
    interrupted. A single worker is more stable for this project and the
    dataset size is still manageable.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        random_state: int = RANDOM_STATE,
        n_estimators: int = 100,
        n_jobs: int = 1,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.scaler = StandardScaler()

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
    ) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest."""

        if X_train.empty:
            raise ValueError(
                "Cannot fit Isolation Forest on an empty dataset."
            )

        X_scaled = self.scaler.fit_transform(
            X_train.to_numpy(dtype=float)
        )

        self.model.fit(X_scaled)

        return self

    def score(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return anomaly score.

        Higher values indicate more anomalous observations.
        """

        X_scaled = self.scaler.transform(
            X.to_numpy(dtype=float)
        )

        return -self.model.decision_function(
            X_scaled
        )

    def flag(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Return Isolation Forest anomaly flags."""

        X_scaled = self.scaler.transform(
            X.to_numpy(dtype=float)
        )

        return (
            self.model.predict(X_scaled) == -1
        )


# ============================================================================
# Synthetic anomaly generation
# ============================================================================

def inject_synthetic_anomalies(
    series: pd.Series,
    rate: float = 0.02,
    spike_multiplier_range: tuple[float, float] = (
        2.5,
        5.0,
    ),
    dip_multiplier_range: tuple[float, float] = (
        0.05,
        0.4,
    ),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series]:
    """
    Inject synthetic spikes and dips.

    Returns:
        injected_series
        ground_truth_labels
    """

    rng = np.random.default_rng(
        random_state
    )

    values = (
        series
        .to_numpy(dtype=float)
        .copy()
    )

    n = len(values)

    if n == 0:
        return (
            series.copy(),
            pd.Series(
                dtype=int,
                index=series.index,
                name="is_synthetic_anomaly",
            ),
        )

    n_inject = max(
        1,
        int(n * rate),
    )

    n_inject = min(
        n_inject,
        n,
    )

    indices = rng.choice(
        n,
        size=n_inject,
        replace=False,
    )

    labels = np.zeros(
        n,
        dtype=int,
    )

    for index in indices:

        if rng.random() < 0.5:
            multiplier = rng.uniform(
                *spike_multiplier_range
            )
        else:
            multiplier = rng.uniform(
                *dip_multiplier_range
            )

        values[index] *= multiplier
        labels[index] = 1

    injected = pd.Series(
        values,
        index=series.index,
        name=series.name,
    )

    label_series = pd.Series(
        labels,
        index=series.index,
        name="is_synthetic_anomaly",
    )

    return injected, label_series
