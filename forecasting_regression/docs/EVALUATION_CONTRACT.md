# Evaluation Contract — Time-Series Forecasting and Regression

## Purpose

This contract keeps ARIMA/SARIMA, Prophet, and regression results directly comparable when the three model families use different preprocessing outputs.

## Rule 1 — Report Errors in Original Watts

Every final model result must report RMSE and MAE on the original power-consumption scale.

- ARIMA, SARIMA, and Prophet use the unscaled feature-engineered dataset, so their targets and predictions are already in original units.
- Linear Regression, Random Forest, and XGBoost use the scaled preprocessing files. Their predictions are inverse-transformed using the target minimum and maximum learned from the training partition before metrics are calculated.

MAPE is also saved because the shared result format includes it.

## Rule 2 — Preserve the Chronological Splits

For the unscaled time-series dataset, use the documented row boundaries:

| Split | Row range | DateTime range |
|---|---:|---|
| Train | `[0:36691]` | 2017-01-01 00:00 to 2017-09-12 19:00 |
| Validation | `[36691:44553]` | 2017-09-12 19:10 to 2017-11-06 09:20 |
| Test | `[44553:52416]` | 2017-11-06 09:30 to 2017-12-30 23:50 |

Regression uses `train.csv`, `validation.csv`, and `test.csv` directly. The preprocessing notebook removes the first 144 training rows with incomplete lag history, so the supplied scaled training file starts at 2017-01-02 00:00. It must not be resliced or shuffled.

## Rule 3 — Use Validation for Selection and Test for Final Evaluation

- Train candidate configurations on the training partition.
- Select the configuration with the lowest validation RMSE.
- Refit the selected configuration using train + validation when appropriate.
- Report final metrics only on the held-out test partition.

## Rule 4 — Shared Public Result Format

Each model-family result CSV uses exactly these columns:

```text
Model, Zone, RMSE, MAE, MAPE, Notes
```

Required result files:

```text
forecasting_regression/docs/results/results_arima_sarima.csv
forecasting_regression/docs/results/results_prophet.csv
forecasting_regression/docs/results/results_regression.csv
```

## Rule 5 — Leakage Control for Regression

For a zone target, the regression feature matrix excludes all three current raw zone-consumption columns:

```text
Zone_1_Power_Consumption
Zone_2_Power_Consumption
Zone_3_Power_Consumption
```

Lag and rolling features are retained because the preprocessing notebook constructs them from previous observations.

## Forecasting Protocol

ARIMA and SARIMA are evaluated with rolling one-step-ahead forecasts. This matches the information pattern of the regression models, whose lagged features contain the most recently observed consumption values.

SARIMA uses the original 10-minute series with daily seasonal period `m = 144`. Stationarity is assessed with the Augmented Dickey-Fuller test, and ACF/PACF plots are generated before model selection.

Prophet uses the required `ds` and `y` format, daily and weekly seasonality, and candidate configurations that include the extra regressors `Temperature`, `Humidity`, `Is_Peak_Hour`, and `Is_Weekend`. An optional Morocco holiday component is also tested.
