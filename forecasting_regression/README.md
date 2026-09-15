# Time-Series Forecasting and Regression

This folder contains the complete classical forecasting and regression component for the Smart Energy Consumption Prediction System.

## Scope

The implementation covers:

- ARIMA
- SARIMA
- Prophet
- Linear Regression
- Random Forest Regression
- XGBoost Regression

All three power-consumption zones are modeled separately. Validation data is used for model and hyperparameter selection, and the held-out test split is used for final evaluation. RMSE and MAE are reported in original Watts. MAPE is also saved because it is part of the agreed result format.

## Required Input from Data Engineering

Place this folder directly in the main repository root. It reads the preprocessing outputs from:

```text
data/processed/feature_engineered_energy_data.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

The preprocessing files are read only and are not modified.

## Folder Structure

```text
forecasting_regression/
├── README.md
├── requirements.txt
├── run_all.py
├── arima_sarima.py
├── prophet_models.py
├── regression_models.py
├── diagnostics.py
├── comparison.py
├── data.py
├── evaluation.py
├── config.py
├── models/
│   ├── arima_sarima/
│   ├── prophet/
│   └── regression/
├── notebooks/
│   ├── 01_arima_sarima.ipynb
│   ├── 02_prophet_models.ipynb
│   ├── 03_regression_models.ipynb
│   └── figures/
├── docs/
│   ├── EVALUATION_CONTRACT.md
│   ├── forecasting_regression_report.md
│   └── results/
└── internal_results/
    ├── predictions/
    └── tuning/
```

Everything created by this component stays inside the `forecasting_regression` folder. The only external dependency is the preprocessing data in `data/processed/`.

## Installation

Run from the main repository root:

```bash
pip install -r forecasting_regression/requirements.txt
```

## Run the Complete Pipeline

From the main repository root:

```bash
python -m forecasting_regression.run_all
```

The complete pipeline performs:

1. ADF stationarity tests and ACF/PACF analysis.
2. ARIMA and SARIMA tuning, training, forecasting, and evaluation.
3. Prophet tuning, training, forecasting, and evaluation.
4. Linear Regression, Random Forest, and XGBoost training and evaluation.
5. Final RMSE/MAE/MAPE comparison and report generation.

Individual components can be run separately:

```bash
python -m forecasting_regression.diagnostics
python -m forecasting_regression.arima_sarima
python -m forecasting_regression.prophet_models
python -m forecasting_regression.regression_models
python -m forecasting_regression.comparison
```

## Evaluation Rules

1. Report RMSE and MAE in original Watts for every model.
2. Preserve the chronological train/validation/test partitions.
3. Use validation only for tuning and model selection, and test only for final evaluation.
4. Save public results with the exact columns `Model, Zone, RMSE, MAE, MAPE, Notes`.
5. Exclude current raw zone-consumption targets from regression predictors to avoid leakage.
6. Use the original 10-minute time-series frequency for ARIMA/SARIMA/Prophet.

See `forecasting_regression/docs/EVALUATION_CONTRACT.md` for the complete evaluation protocol.
