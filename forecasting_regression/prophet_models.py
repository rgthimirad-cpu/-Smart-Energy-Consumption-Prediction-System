from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROPHET_REGRESSORS,
    PUBLIC_RESULTS_DIR,
    TARGETS,
    TUNING_DIR,
    ensure_output_dirs,
)
from .data import load_unscaled_time_series_splits
from .evaluation import prediction_frame, public_result_row, save_csv, save_public_results, validation_row

# The candidate set explicitly includes a baseline and enhanced seasonal/regressor models.
PROPHET_CANDIDATES = [
    {
        "name": "baseline",
        "use_regressors": False,
        "use_holidays": False,
        "daily_fourier_order": 10,
        "weekly_fourier_order": 3,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
    },
    {
        "name": "regressors_tuned_a",
        "use_regressors": True,
        "use_holidays": False,
        "daily_fourier_order": 12,
        "weekly_fourier_order": 5,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 5.0,
    },
    {
        "name": "regressors_tuned_holidays",
        "use_regressors": True,
        "use_holidays": True,
        "daily_fourier_order": 16,
        "weekly_fourier_order": 6,
        "changepoint_prior_scale": 0.10,
        "seasonality_prior_scale": 10.0,
    },
]


def _imports():
    try:
        from prophet import Prophet
        from prophet.serialize import model_to_json
    except ImportError as exc:
        raise RuntimeError(
            "Prophet is not installed. Install forecasting_regression/requirements.txt before running this module."
        ) from exc
    return Prophet, model_to_json


def _frame(df: pd.DataFrame, target: str, use_regressors: bool) -> pd.DataFrame:
    cols = ["DateTime", target] + (PROPHET_REGRESSORS if use_regressors else [])
    out = df[cols].copy().rename(columns={"DateTime": "ds", target: "y"})
    return out


def _future_frame(df: pd.DataFrame, use_regressors: bool) -> pd.DataFrame:
    cols = ["DateTime"] + (PROPHET_REGRESSORS if use_regressors else [])
    return df[cols].copy().rename(columns={"DateTime": "ds"})


def _build(Prophet, cfg: dict):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=cfg["changepoint_prior_scale"],
        seasonality_prior_scale=cfg["seasonality_prior_scale"],
    )
    model.add_seasonality(name="daily", period=1, fourier_order=cfg["daily_fourier_order"])
    model.add_seasonality(name="weekly", period=7, fourier_order=cfg["weekly_fourier_order"])
    if cfg["use_regressors"]:
        for regressor in PROPHET_REGRESSORS:
            model.add_regressor(regressor)
    if cfg["use_holidays"]:
        # Morocco is the location of the Tetouan City dataset. If the installed
        # holidays package does not provide this calendar, the model remains
        # valid without the optional holiday component.
        try:
            model.add_country_holidays(country_name="MA")
        except Exception:
            pass
    return model


def run_prophet() -> None:
    Prophet, model_to_json = _imports()
    ensure_output_dirs()
    splits = load_unscaled_time_series_splits()
    train, val, test = splits["train"], splits["validation"], splits["test"]
    train_val = pd.concat([train, val], ignore_index=True)

    public_rows, tuning_rows, prediction_parts = [], [], []
    model_dir = MODELS_DIR / "prophet"

    for zone, target in TARGETS.items():
        print(f"\n{target}: Prophet")
        best_cfg, best_rmse, best_val_pred = None, np.inf, None

        for cfg in PROPHET_CANDIDATES:
            model = _build(Prophet, cfg)
            model.fit(_frame(train, target, cfg["use_regressors"]))
            val_forecast = model.predict(_future_frame(val, cfg["use_regressors"]))
            val_pred = val_forecast["yhat"].to_numpy(dtype=float)
            row = validation_row("Prophet", target, val[target], val_pred, cfg)
            tuning_rows.append(row)
            if row["RMSE"] < best_rmse:
                best_rmse, best_cfg, best_val_pred = row["RMSE"], cfg.copy(), val_pred

        prediction_parts.append(prediction_frame(
            val["DateTime"], "Prophet", target, "validation", val[target], best_val_pred
        ))

        final_model = _build(Prophet, best_cfg)
        final_model.fit(_frame(train_val, target, best_cfg["use_regressors"]))
        test_forecast = final_model.predict(_future_frame(test, best_cfg["use_regressors"]))
        test_pred = test_forecast["yhat"].to_numpy(dtype=float)

        with open(model_dir / f"prophet_zone{zone[-1]}.json", "w", encoding="utf-8") as handle:
            handle.write(model_to_json(final_model))

        prediction_parts.append(prediction_frame(
            test["DateTime"], "Prophet", target, "test", test[target], test_pred
        ))
        public_rows.append(public_result_row(
            "Prophet", target, test[target], test_pred,
            "best validation configuration=" + json.dumps(best_cfg, sort_keys=True),
        ))

    save_public_results(public_rows, PUBLIC_RESULTS_DIR / "results_prophet.csv")
    save_csv(tuning_rows, TUNING_DIR / "prophet_validation.csv")
    pd.concat(prediction_parts, ignore_index=True).to_csv(
        PREDICTIONS_DIR / "predictions_prophet.csv", index=False
    )


if __name__ == "__main__":
    run_prophet()
