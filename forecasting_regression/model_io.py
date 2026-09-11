from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def _trend_for(order: tuple[int, int, int]) -> str:
    return "c" if order[1] == 0 else "n"


def _seasonal_difference(values, period: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if period <= 0:
        raise ValueError("seasonal_period must be a positive integer.")
    if len(arr) <= period:
        raise ValueError(
            f"At least {period + 1} observations are required for seasonal differencing."
        )
    return arr[period:] - arr[:-period]


def load_forecasting_artifact(path: str | Path) -> dict[str, Any]:
    """Load a compact ARIMA/SARIMA artifact saved by arima_sarima.py."""
    artifact = joblib.load(path)
    required = {"format_version", "Model", "Zone", "order", "params", "training_series"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Saved artifact is missing required fields: {sorted(missing)}")
    return artifact


def rebuild_fitted_result(artifact: dict[str, Any]):
    """Rebuild a statsmodels fitted result from saved training data and parameters."""
    family = str(artifact["Model"]).upper()
    order = tuple(int(x) for x in artifact["order"])
    params = np.asarray(artifact["params"], dtype=float)
    training = np.asarray(artifact["training_series"], dtype=float)

    if family == "ARIMA":
        model = ARIMA(training, order=order, trend=_trend_for(order))
        return model.filter(params)

    if family == "SARIMA":
        seasonal_order = tuple(int(x) for x in artifact["seasonal_order"])
        p_seasonal, d_seasonal, q_seasonal, period = seasonal_order
        if (p_seasonal, d_seasonal, q_seasonal) != (0, 1, 0):
            raise ValueError(
                "This project stores the efficient SARIMA form with seasonal_order=(0, 1, 0, m)."
            )
        differenced = _seasonal_difference(training, period)
        model = ARIMA(differenced, order=order, trend=_trend_for(order))
        return model.filter(params)

    raise ValueError(f"Unsupported model family: {family}")


def forecast_from_artifact(
    artifact_or_path: dict[str, Any] | str | Path,
    steps: int = 1,
) -> np.ndarray:
    """Forecast in original Watts from a saved compact ARIMA/SARIMA artifact."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    artifact = (
        load_forecasting_artifact(artifact_or_path)
        if isinstance(artifact_or_path, (str, Path))
        else artifact_or_path
    )
    family = str(artifact["Model"]).upper()
    fitted = rebuild_fitted_result(artifact)

    if family == "ARIMA":
        return np.asarray(fitted.forecast(steps=steps), dtype=float)

    if family == "SARIMA":
        seasonal_order = tuple(int(x) for x in artifact["seasonal_order"])
        period = seasonal_order[3]
        history = list(np.asarray(artifact["training_series"], dtype=float))
        diff_forecast = np.asarray(fitted.forecast(steps=steps), dtype=float)
        output: list[float] = []

        for i, diff_value in enumerate(diff_forecast):
            # For y[t] - y[t-m], the seasonal base may come from the observed
            # training history or from an earlier forecast when steps > m.
            base_index = len(history) + i - period
            if base_index < len(history):
                seasonal_base = history[base_index]
            else:
                seasonal_base = output[base_index - len(history)]
            output.append(float(seasonal_base + diff_value))

        return np.asarray(output, dtype=float)

    raise ValueError(f"Unsupported model family: {family}")
