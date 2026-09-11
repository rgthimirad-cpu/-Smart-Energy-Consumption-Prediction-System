from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PUBLIC_COLUMNS = ["Model", "Zone", "RMSE", "MAE", "MAPE", "Notes"]


def metrics_watts(y_true, y_pred) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Metric shape mismatch: {true.shape} vs {pred.shape}")
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    nonzero = np.abs(true) > 1e-12
    mape = float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100.0)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def public_result_row(model: str, zone: str, y_true, y_pred, notes: str) -> dict:
    scores = metrics_watts(y_true, y_pred)
    return {
        "Model": model,
        "Zone": zone,
        "RMSE": scores["RMSE"],
        "MAE": scores["MAE"],
        "MAPE": scores["MAPE"],
        "Notes": notes,
    }


def validation_row(model: str, zone: str, y_true, y_pred, params: dict) -> dict:
    scores = metrics_watts(y_true, y_pred)
    return {
        "Model": model,
        "Zone": zone,
        "RMSE": scores["RMSE"],
        "MAE": scores["MAE"],
        "MAPE": scores["MAPE"],
        "Parameters": json.dumps(params, sort_keys=True, default=str),
    }


def prediction_frame(dates, model: str, zone: str, split: str, y_true, y_pred) -> pd.DataFrame:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return pd.DataFrame({
        "DateTime": pd.to_datetime(dates),
        "Model": model,
        "Zone": zone,
        "Split": split,
        "Actual_Watts": true,
        "Predicted_Watts": pred,
        "Error_Watts": pred - true,
        "Absolute_Error_Watts": np.abs(pred - true),
    })


def save_public_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PUBLIC_COLUMNS).to_csv(path, index=False)


def save_csv(rows_or_df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False)
