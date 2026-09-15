from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from .config import (
    MODELS_DIR,
    PREDICTIONS_DIR,
    PUBLIC_RESULTS_DIR,
    RANDOM_STATE,
    TARGETS,
    TUNING_DIR,
    ensure_output_dirs,
)
from .data import (
    combine_scaled_train_validation,
    inverse_target,
    load_scaled_regression_splits,
    regression_xy,
    target_min_max,
)
from .evaluation import prediction_frame, public_result_row, save_csv, save_public_results, validation_row

RF_CANDIDATES = [
    {"n_estimators": 20, "max_depth": 10, "min_samples_leaf": 1, "max_features": 0.6, "max_samples": 0.5},
    {"n_estimators": 40, "max_depth": 14, "min_samples_leaf": 1, "max_features": 0.7, "max_samples": 0.6},
]

XGB_CANDIDATES = [
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
]



def _rf(params):
    return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)


def _xgb(params):
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _select(model_name, candidates, builder, X_train, y_train, X_val, y_val_watts, target, scales, zone):
    rows, best_params, best_rmse = [], None, np.inf
    for params in candidates:
        model = builder(params).fit(X_train, y_train)
        pred_watts = inverse_target(model.predict(X_val), target, scales)
        row = validation_row(model_name, zone, y_val_watts, pred_watts, params)
        rows.append(row)
        if row["RMSE"] < best_rmse:
            best_rmse, best_params = row["RMSE"], params
    return best_params, rows


def _upsert(path, rows_or_df, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if len(old) and len(new):
            incoming = set(map(tuple, new[keys].astype(str).to_numpy()))
            keep = [tuple(row) not in incoming for row in old[keys].astype(str).to_numpy()]
            new = pd.concat([old.loc[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def run_regression(zones: list[str] | None = None) -> None:
    ensure_output_dirs()
    splits = load_scaled_regression_splits()
    train, val, test = splits["train"], splits["validation"], splits["test"]
    train_val = combine_scaled_train_validation(train, val)
    scales = target_min_max()

    public_rows, tuning_rows, prediction_parts, importance_rows = [], [], [], []
    model_dir = MODELS_DIR / "regression"

    selected = zones or list(TARGETS.keys())
    for zone in selected:
        if zone not in TARGETS:
            raise ValueError(f"Unknown zone: {zone}")
        target = TARGETS[zone]
        print(f"\n{target}: regression models")
        X_train, y_train, _ = regression_xy(train, target)
        X_val, y_val, val_dates = regression_xy(val, target)
        X_test, y_test, test_dates = regression_xy(test, target)
        X_train_val, y_train_val, _ = regression_xy(train_val, target)

        y_val_watts = inverse_target(y_val, target, scales)
        y_test_watts = inverse_target(y_test, target, scales)

        # Linear Regression
        linear_train = LinearRegression().fit(X_train, y_train)
        val_pred_watts = inverse_target(linear_train.predict(X_val), target, scales)
        tuning_rows.append(validation_row("Linear Regression", target, y_val_watts, val_pred_watts, {}))
        prediction_parts.append(prediction_frame(
            val_dates, "Linear Regression", target, "validation", y_val_watts, val_pred_watts
        ))

        linear_final = LinearRegression().fit(X_train_val, y_train_val)
        test_pred_watts = inverse_target(linear_final.predict(X_test), target, scales)
        joblib.dump(linear_final, model_dir / f"linear_zone{zone[-1]}.joblib", compress=5)
        prediction_parts.append(prediction_frame(
            test_dates, "Linear Regression", target, "test", y_test_watts, test_pred_watts
        ))
        public_rows.append(public_result_row(
            "Linear Regression", target, y_test_watts, test_pred_watts,
            "Scaled preprocessing features; current zone targets excluded; predictions inverse-transformed to Watts",
        ))

        # Random Forest
        best_rf, rows = _select(
            "Random Forest", RF_CANDIDATES, _rf, X_train, y_train, X_val,
            y_val_watts, target, scales, target,
        )
        tuning_rows.extend(rows)
        rf_train = _rf(best_rf).fit(X_train, y_train)
        val_pred_watts = inverse_target(rf_train.predict(X_val), target, scales)
        prediction_parts.append(prediction_frame(
            val_dates, "Random Forest", target, "validation", y_val_watts, val_pred_watts
        ))
        rf_final = _rf(best_rf).fit(X_train_val, y_train_val)
        test_pred_watts = inverse_target(rf_final.predict(X_test), target, scales)
        joblib.dump(rf_final, model_dir / f"rf_zone{zone[-1]}.joblib", compress=5)
        prediction_parts.append(prediction_frame(
            test_dates, "Random Forest", target, "test", y_test_watts, test_pred_watts
        ))
        public_rows.append(public_result_row(
            "Random Forest", target, y_test_watts, test_pred_watts,
            f"best validation parameters={best_rf}; predictions inverse-transformed to Watts",
        ))
        importance_rows.extend({
            "Model": "Random Forest", "Zone": target, "Feature": feature, "Importance": float(importance)
        } for feature, importance in zip(X_train_val.columns, rf_final.feature_importances_))

        # XGBoost
        best_xgb, rows = _select(
            "XGBoost", XGB_CANDIDATES, _xgb, X_train, y_train, X_val,
            y_val_watts, target, scales, target,
        )
        tuning_rows.extend(rows)
        xgb_train = _xgb(best_xgb).fit(X_train, y_train)
        val_pred_watts = inverse_target(xgb_train.predict(X_val), target, scales)
        prediction_parts.append(prediction_frame(
            val_dates, "XGBoost", target, "validation", y_val_watts, val_pred_watts
        ))
        xgb_final = _xgb(best_xgb).fit(X_train_val, y_train_val)
        test_pred_watts = inverse_target(xgb_final.predict(X_test), target, scales)
        joblib.dump(xgb_final, model_dir / f"xgb_zone{zone[-1]}.joblib", compress=5)
        prediction_parts.append(prediction_frame(
            test_dates, "XGBoost", target, "test", y_test_watts, test_pred_watts
        ))
        public_rows.append(public_result_row(
            "XGBoost", target, y_test_watts, test_pred_watts,
            f"best validation parameters={best_xgb}; predictions inverse-transformed to Watts",
        ))
        importance_rows.extend({
            "Model": "XGBoost", "Zone": target, "Feature": feature, "Importance": float(importance)
        } for feature, importance in zip(X_train_val.columns, xgb_final.feature_importances_))

    _upsert(PUBLIC_RESULTS_DIR / "results_regression.csv", pd.DataFrame(public_rows), ["Model", "Zone"])
    _upsert(TUNING_DIR / "regression_validation.csv", pd.DataFrame(tuning_rows), ["Model", "Zone", "Parameters"])
    _upsert(PUBLIC_RESULTS_DIR / "feature_importance_regression.csv", pd.DataFrame(importance_rows), ["Model", "Zone", "Feature"])
    _upsert(
        PREDICTIONS_DIR / "predictions_regression.csv",
        pd.concat(prediction_parts, ignore_index=True),
        ["DateTime", "Model", "Zone", "Split"],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate regression models.")
    parser.add_argument("--zone", choices=list(TARGETS.keys()), help="Run one zone only.")
    args = parser.parse_args()
    run_regression([args.zone] if args.zone else None)
