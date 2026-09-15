# Time-Series Forecasting and Regression Comparison Report

## Objective

This work implements and compares ARIMA, SARIMA, Prophet, Linear Regression, Random Forest, and XGBoost for the three power-consumption zones. Model selection uses validation data, while the final comparison uses the held-out test period. RMSE, MAE, and MAPE are reported in original power-consumption units (Watts for RMSE/MAE).

## Evaluation Protocol

- Time-series models use the unscaled feature-engineered dataset at the original 10-minute frequency.
- Regression models use the preprocessing team's scaled train/validation/test files and are inverse-transformed to Watts before scoring.
- Current raw Zone 1/2/3 consumption columns are excluded from regression predictors to prevent target leakage; lagged and rolling historical features are retained.
- Hyperparameters are selected using the validation split only. Final models are then fitted on train + validation and evaluated on test.
- ARIMA/SARIMA are evaluated as rolling one-step-ahead forecasts, consistent with the availability of lagged observations used by regression models.

## Test Results

| Model             | Zone                     |     RMSE |      MAE |   MAPE | Notes                                                                                                                                                                                                                                    |
|:------------------|:-------------------------|---------:|---------:|-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARIMA             | Zone_1_Power_Consumption |  381.096 |  255.861 |  0.899 | order=(2, 0, 0); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                                                        |
| ARIMA             | Zone_2_Power_Consumption |  341.139 |  215.229 |  0.937 | order=(2, 0, 0); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                                                        |
| ARIMA             | Zone_3_Power_Consumption |  289.493 |  165.621 |  1.467 | order=(2, 0, 0); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                                                        |
| Linear Regression | Zone_1_Power_Consumption |  365.588 |  260.853 |  0.934 | Scaled preprocessing features; current zone targets excluded; predictions inverse-transformed to Watts                                                                                                                                   |
| Linear Regression | Zone_2_Power_Consumption |  315.773 |  199.889 |  0.869 | Scaled preprocessing features; current zone targets excluded; predictions inverse-transformed to Watts                                                                                                                                   |
| Linear Regression | Zone_3_Power_Consumption |  318.479 |  215.707 |  1.929 | Scaled preprocessing features; current zone targets excluded; predictions inverse-transformed to Watts                                                                                                                                   |
| Prophet           | Zone_1_Power_Consumption | 2570.634 | 2001.613 |  6.773 | best validation configuration={"changepoint_prior_scale": 0.1, "daily_fourier_order": 16, "name": "regressors_tuned_holidays", "seasonality_prior_scale": 10.0, "use_holidays": true, "use_regressors": true, "weekly_fourier_order": 6} |
| Prophet           | Zone_2_Power_Consumption | 2727.599 | 2243.168 | 10.509 | best validation configuration={"changepoint_prior_scale": 0.1, "daily_fourier_order": 16, "name": "regressors_tuned_holidays", "seasonality_prior_scale": 10.0, "use_holidays": true, "use_regressors": true, "weekly_fourier_order": 6} |
| Prophet           | Zone_3_Power_Consumption | 2843.882 | 2164.581 | 18.972 | best validation configuration={"changepoint_prior_scale": 0.1, "daily_fourier_order": 16, "name": "regressors_tuned_holidays", "seasonality_prior_scale": 10.0, "use_holidays": true, "use_regressors": true, "weekly_fourier_order": 6} |
| Random Forest     | Zone_1_Power_Consumption |  364.810 |  250.037 |  0.883 | best validation parameters={'n_estimators': 40, 'max_depth': 14, 'min_samples_leaf': 1, 'max_features': 0.7, 'max_samples': 0.6}; predictions inverse-transformed to Watts                                                               |
| Random Forest     | Zone_2_Power_Consumption |  314.705 |  209.273 |  0.914 | best validation parameters={'n_estimators': 40, 'max_depth': 14, 'min_samples_leaf': 1, 'max_features': 0.7, 'max_samples': 0.6}; predictions inverse-transformed to Watts                                                               |
| Random Forest     | Zone_3_Power_Consumption |  371.298 |  243.600 |  2.267 | best validation parameters={'n_estimators': 40, 'max_depth': 14, 'min_samples_leaf': 1, 'max_features': 0.7, 'max_samples': 0.6}; predictions inverse-transformed to Watts                                                               |
| SARIMA            | Zone_1_Power_Consumption |  342.587 |  227.697 |  0.803 | order=(1, 0, 1), seasonal_order=(0, 1, 0, 144); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                         |
| SARIMA            | Zone_2_Power_Consumption |  321.037 |  198.199 |  0.866 | order=(1, 0, 0), seasonal_order=(0, 1, 0, 144); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                         |
| SARIMA            | Zone_3_Power_Consumption |  295.927 |  160.500 |  1.414 | order=(1, 0, 1), seasonal_order=(0, 1, 0, 144); validation-selected; rolling one-step evaluation; 10-minute data                                                                                                                         |
| XGBoost           | Zone_1_Power_Consumption |  325.391 |  226.503 |  0.813 | best validation parameters={'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.9, 'colsample_bytree': 0.9}; predictions inverse-transformed to Watts                                                             |
| XGBoost           | Zone_2_Power_Consumption |  300.783 |  200.948 |  0.867 | best validation parameters={'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9}; predictions inverse-transformed to Watts                                                             |
| XGBoost           | Zone_3_Power_Consumption |  355.968 |  230.041 |  2.204 | best validation parameters={'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9, 'colsample_bytree': 0.9}; predictions inverse-transformed to Watts                                                             |

## Average Performance Across Zones

| Model             |     RMSE |      MAE |   MAPE |
|:------------------|---------:|---------:|-------:|
| SARIMA            |  319.850 |  195.465 |  1.027 |
| XGBoost           |  327.381 |  219.164 |  1.295 |
| Linear Regression |  333.280 |  225.483 |  1.244 |
| ARIMA             |  337.242 |  212.237 |  1.101 |
| Random Forest     |  350.271 |  234.303 |  1.355 |
| Prophet           | 2714.038 | 2136.454 | 12.085 |

## Best-Performing Model

The lowest average test RMSE is achieved by **SARIMA** (319.850 W). Its average MAE is 195.465 W and average MAPE is 1.027%.

The result indicates that this model provides the strongest overall predictive accuracy under the common evaluation protocol. The reason is assessed from measured test errors rather than model complexity alone: lower RMSE shows fewer/lower large errors, while lower MAE reflects lower typical absolute error. For tree-based regression models, the saved feature-importance table can be used to identify which lag, rolling, weather, and time features contributed most strongly. For SARIMA and Prophet, performance can be interpreted in relation to the strong daily/weekly seasonal structure identified during time-series analysis.

## Best Model by Zone

- Zone_1_Power_Consumption: **XGBoost** (RMSE 325.391 W)
- Zone_2_Power_Consumption: **XGBoost** (RMSE 300.783 W)
- Zone_3_Power_Consumption: **ARIMA** (RMSE 289.493 W)

## Notes on Comparability

All reported errors are on the original power-consumption scale. The models do not use exactly the same predictor information: ARIMA/SARIMA model the target history directly, Prophet uses calendar/seasonal structure plus the specified external regressors, and regression models use engineered lag/rolling/weather/time predictors. This difference is retained because it is part of the required model families, and it should be considered when interpreting why a model performs better.

## Completion Status

All required model families are present in the result table.
