from __future__ import annotations

import gc
import warnings

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    PUBLIC_RESULTS_DIR,
    SEASONAL_PERIOD,
    TARGETS,
    TUNING_DIR,
    ensure_output_dirs,
)
from .data import load_unscaled_time_series_splits
from .evaluation import prediction_frame, public_result_row, validation_row

warnings.filterwarnings("ignore")

# Small, reproducible grids keep tuning practical at the original 10-minute frequency.
ARIMA_CANDIDATES = [(1, 0, 0), (2, 0, 0), (1, 0, 1)]
SARIMA_ARIMA_CANDIDATES = [(1, 0, 0), (2, 0, 0), (1, 0, 1)]

# Efficient SARIMA candidates. P=Q=0 is retained so the model can be fit as an
# ARIMA model after exact seasonal differencing. Both daily and weekly seasonal
# periods are validated rather than hard-coding a single seasonal setting.
SARIMA_SEASONAL_CANDIDATES = [
    (0, 1, 0, SEASONAL_PERIOD),       # daily seasonality: 144 ten-minute samples
    (0, 1, 0, SEASONAL_PERIOD * 7),   # weekly seasonality: 1008 samples
]


def _trend_for(order: tuple[int, int, int]) -> str:
    return "c" if order[1] == 0 else "n"


def _fit_arima(y, order):
    return ARIMA(
        np.asarray(y, dtype=float),
        order=order,
        trend=_trend_for(order),
    ).fit(method_kwargs={"maxiter": 30})


def _arima_one_step_predictions(fit, future):
    future = np.asarray(future, dtype=float)
    extended = fit.extend(future)
    return np.asarray(extended.fittedvalues, dtype=float)


def _seasonal_difference(y, m: int):
    arr = np.asarray(y, dtype=float)
    if len(arr) <= m:
        raise ValueError(f"At least {m + 1} observations are required for seasonal period {m}.")
    return arr[m:] - arr[:-m]


def _fit_sarima(y, order, seasonal_order):
    """Fit SARIMA(p,d,q)(0,1,0,m) using exact seasonal differencing.

    With P=Q=0 and D=1, fitting ARIMA(p,d,q) to y[t]-y[t-m] is
    mathematically equivalent to SARIMA(p,d,q)(0,1,0,m). This preserves the
    original 10-minute data while avoiding the large state-space model that a
    direct SARIMAX fit with m=144 or m=1008 would create.
    """
    p_seasonal, d_seasonal, q_seasonal, m = seasonal_order
    if (p_seasonal, d_seasonal, q_seasonal) != (0, 1, 0):
        raise ValueError(
            "Efficient SARIMA implementation requires seasonal_order=(0, 1, 0, m)."
        )
    z = _seasonal_difference(y, m)
    return _fit_arima(z, order)


def _sarima_one_step_predictions(fit, history, future, seasonal_order):
    _, _, _, m = seasonal_order
    history = np.asarray(history, dtype=float)
    future = np.asarray(future, dtype=float)
    combined = np.concatenate([history, future])
    z_future = _seasonal_difference(combined, m)[-len(future):]
    pred_z = np.asarray(fit.extend(z_future).fittedvalues, dtype=float)
    seasonal_base = np.array(
        [combined[len(history) + i - m] for i in range(len(future))],
        dtype=float,
    )
    return pred_z + seasonal_base


def _compact_artifact(
    model_family: str,
    zone: str,
    order,
    fitted_result,
    training_series,
    seasonal_order=None,
):
    """Create a compact, reloadable forecasting artifact.

    The fitted statsmodels result itself is not saved because remove_data()
    makes later forecasting unreliable, while keeping the full result produces
    very large files. Saving the fitted parameters plus the original training
    series allows model_io.py to rebuild the fitted state when needed.
    """
    artifact = {
        "format_version": 1,
        "Model": model_family,
        "Zone": zone,
        "order": tuple(int(x) for x in order),
        "params": np.asarray(fitted_result.params, dtype=float),
        "training_series": np.asarray(training_series, dtype=np.float32),
        "training_rows": int(len(training_series)),
    }
    if seasonal_order is not None:
        artifact["seasonal_order"] = tuple(int(x) for x in seasonal_order)
        artifact["seasonal_period"] = int(seasonal_order[3])
    return artifact


def _select_arima(train_y, val_y, zone):
    rows, best_order, best_rmse = [], None, np.inf
    for order in ARIMA_CANDIDATES:
        fit = _fit_arima(train_y, order)
        pred = _arima_one_step_predictions(fit, val_y)
        row = validation_row("ARIMA", zone, val_y, pred, {"order": order})
        row["AIC"] = float(fit.aic)
        rows.append(row)
        if row["RMSE"] < best_rmse:
            best_rmse, best_order = row["RMSE"], order
        del fit, pred
        gc.collect()
    return best_order, rows


def _select_sarima(train_y, val_y, zone):
    rows = []
    best_order = None
    best_seasonal_order = None
    best_rmse = np.inf

    for seasonal_order in SARIMA_SEASONAL_CANDIDATES:
        for order in SARIMA_ARIMA_CANDIDATES:
            fit = _fit_sarima(train_y, order, seasonal_order)
            pred = _sarima_one_step_predictions(
                fit,
                train_y,
                val_y,
                seasonal_order,
            )
            params = {"order": order, "seasonal_order": seasonal_order}
            row = validation_row("SARIMA", zone, val_y, pred, params)
            row["AIC"] = float(fit.aic)
            rows.append(row)

            if row["RMSE"] < best_rmse:
                best_rmse = row["RMSE"]
                best_order = order
                best_seasonal_order = seasonal_order

            del fit, pred
            gc.collect()

    return best_order, best_seasonal_order, rows


def _upsert(path, rows_or_df, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if len(old) and len(new_df):
            incoming = set(map(tuple, new_df[keys].astype(str).to_numpy()))
            keep = [tuple(row) not in incoming for row in old[keys].astype(str).to_numpy()]
            new_df = pd.concat([old.loc[keep], new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def run_arima_sarima(
    zones: list[str] | None = None,
    families: list[str] | None = None,
) -> None:
    ensure_output_dirs()
    splits = load_unscaled_time_series_splits()
    train, val, test = splits["train"], splits["validation"], splits["test"]

    selected_zones = zones or list(TARGETS.keys())
    selected_families = set(families or ["ARIMA", "SARIMA"])
    invalid = selected_families - {"ARIMA", "SARIMA"}
    if invalid:
        raise ValueError(f"Unknown model families: {sorted(invalid)}")

    model_dir = MODELS_DIR / "arima_sarima"
    public_path = PUBLIC_RESULTS_DIR / "results_arima_sarima.csv"
    tuning_path = TUNING_DIR / "arima_sarima_validation.csv"
    prediction_path = PREDICTIONS_DIR / "predictions_arima_sarima.csv"

    for zone in selected_zones:
        if zone not in TARGETS:
            raise ValueError(f"Unknown zone: {zone}")

        target = TARGETS[zone]
        print(f"\n{target}: ARIMA/SARIMA")
        train_y = train[target].to_numpy(dtype=float)
        val_y = val[target].to_numpy(dtype=float)
        test_y = test[target].to_numpy(dtype=float)
        train_val_y = np.concatenate([train_y, val_y])

        zone_public, zone_tuning, zone_predictions = [], [], []

        if "ARIMA" in selected_families:
            best_arima, rows = _select_arima(train_y, val_y, target)
            zone_tuning.extend(rows)

            arima_train_fit = _fit_arima(train_y, best_arima)
            val_pred = _arima_one_step_predictions(arima_train_fit, val_y)
            zone_predictions.append(
                prediction_frame(
                    val["DateTime"],
                    "ARIMA",
                    target,
                    "validation",
                    val_y,
                    val_pred,
                )
            )

            arima_final = _fit_arima(train_val_y, best_arima)
            test_pred = _arima_one_step_predictions(arima_final, test_y)
            zone_predictions.append(
                prediction_frame(
                    test["DateTime"],
                    "ARIMA",
                    target,
                    "test",
                    test_y,
                    test_pred,
                )
            )

            artifact = _compact_artifact(
                "ARIMA",
                target,
                best_arima,
                arima_final,
                train_val_y,
            )
            joblib.dump(
                artifact,
                model_dir / f"arima_zone{zone[-1]}.pkl",
                compress=5,
            )

            zone_public.append(
                public_result_row(
                    "ARIMA",
                    target,
                    test_y,
                    test_pred,
                    f"order={best_arima}; validation-selected; rolling one-step evaluation; 10-minute data",
                )
            )
            del arima_train_fit, arima_final, val_pred, test_pred, artifact
            gc.collect()

        if "SARIMA" in selected_families:
            best_sarima, best_seasonal_order, rows = _select_sarima(
                train_y,
                val_y,
                target,
            )
            zone_tuning.extend(rows)

            sarima_train_fit = _fit_sarima(
                train_y,
                best_sarima,
                best_seasonal_order,
            )
            val_pred = _sarima_one_step_predictions(
                sarima_train_fit,
                train_y,
                val_y,
                best_seasonal_order,
            )
            zone_predictions.append(
                prediction_frame(
                    val["DateTime"],
                    "SARIMA",
                    target,
                    "validation",
                    val_y,
                    val_pred,
                )
            )

            sarima_final = _fit_sarima(
                train_val_y,
                best_sarima,
                best_seasonal_order,
            )
            test_pred = _sarima_one_step_predictions(
                sarima_final,
                train_val_y,
                test_y,
                best_seasonal_order,
            )
            zone_predictions.append(
                prediction_frame(
                    test["DateTime"],
                    "SARIMA",
                    target,
                    "test",
                    test_y,
                    test_pred,
                )
            )

            artifact = _compact_artifact(
                "SARIMA",
                target,
                best_sarima,
                sarima_final,
                train_val_y,
                seasonal_order=best_seasonal_order,
            )
            joblib.dump(
                artifact,
                model_dir / f"sarima_zone{zone[-1]}.pkl",
                compress=5,
            )

            zone_public.append(
                public_result_row(
                    "SARIMA",
                    target,
                    test_y,
                    test_pred,
                    f"order={best_sarima}, seasonal_order={best_seasonal_order}; validation-selected; rolling one-step evaluation; 10-minute data",
                )
            )
            del sarima_train_fit, sarima_final, val_pred, test_pred, artifact
            gc.collect()

        if zone_public:
            _upsert(public_path, pd.DataFrame(zone_public), ["Model", "Zone"])
        if zone_tuning:
            _upsert(
                tuning_path,
                pd.DataFrame(zone_tuning),
                ["Model", "Zone", "Parameters"],
            )
        if zone_predictions:
            _upsert(
                prediction_path,
                pd.concat(zone_predictions, ignore_index=True),
                ["DateTime", "Model", "Zone", "Split"],
            )

        print(f"Saved {target} checkpoint.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate ARIMA/SARIMA models.")
    parser.add_argument("--zone", choices=list(TARGETS.keys()), help="Run one zone only.")
    parser.add_argument("--family", choices=["ARIMA", "SARIMA"], help="Run one model family only.")
    args = parser.parse_args()
    run_arima_sarima(
        [args.zone] if args.zone else None,
        [args.family] if args.family else None,
    )
