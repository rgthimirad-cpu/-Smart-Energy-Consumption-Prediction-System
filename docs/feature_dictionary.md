# Feature Dictionary — Smart Energy Consumption Prediction System

This document describes every column produced by the Data Engineering & Preprocessing pipeline (`notebooks/data_preprocessing.ipynb`), across the following files in `data/processed/`:

- `cleaned_energy_data.csv` — cleaned, pre-feature-engineering
- `feature_engineered_energy_data.csv` — full feature set, unscaled
- `train.csv` / `validation.csv` / `test.csv` — feature-complete, scaled (chronological 70/15/15 split)

## Original / Source Columns

| Feature | Data Type | Source / Formula | Intended Usage |
|---|---|---|---|
| `DateTime` | datetime | Parsed from raw `DateTime` string (`MM/DD/YYYY HH:MM`) | Index/join key for all teams; x-axis for dashboard time-series views |
| `Temperature` | float | Raw sensor reading (°C) | Regression/LSTM input feature; dashboard weather panel |
| `Humidity` | float | Raw sensor reading (%) | Regression/LSTM input feature; dashboard weather panel |
| `Wind_Speed` | float | Raw sensor reading | Regression/LSTM input feature |
| `General_Diffuse_Flows` | float | Raw solar sensor reading | Weather/solar-irradiance proxy feature |
| `Diffuse_Flows` | float | Raw solar sensor reading | Weather/solar-irradiance proxy feature |
| `Zone_1_Power_Consumption` | float | Raw target (Watts), IQR-capped | **Primary forecasting target** for Zone 1; anomaly detection base signal |
| `Zone_2_Power_Consumption` | float | Raw target (Watts), IQR-capped | **Primary forecasting target** for Zone 2; anomaly detection base signal |
| `Zone_3_Power_Consumption` | float | Raw target (Watts), IQR-capped | **Primary forecasting target** for Zone 3; anomaly detection base signal |

## Time-Based Features

*(present in `feature_engineered_energy_data.csv`, `train.csv`, `validation.csv`, `test.csv` — not in `cleaned_energy_data.csv`)*

| Feature | Data Type | Formula/Source | Intended Usage |
|---|---|---|---|
| `Hour` | int (0–23) | `DateTime.hour` | Cyclical/peak-demand modeling; dashboard hourly filters |
| `Day_of_Week` | int (0–6, Mon=0) | `DateTime.dayofweek` | Weekly seasonality feature for regression/LSTM |
| `Month` | int (1–12) | `DateTime.month` | Monthly/seasonal trend feature |
| `Quarter` | int (1–4) | `DateTime.quarter` | Coarse seasonal grouping; dashboard quarterly rollups |
| `Is_Weekend` | int (0/1) | 1 if `Day_of_Week` in {Sat, Sun} | Captures weekday/weekend consumption shift |
| `Season_Winter` / `Season_Spring` / `Season_Summer` / `Season_Autumn` | int (0/1, one-hot) | Derived from `Month` (Northern Hemisphere) | Seasonal regime feature for regression/clustering |
| `Is_Peak_Hour` | int (0/1) | 1 if `18:00 <= Hour < 22:00` | **Peak-demand prediction** target/feature; dashboard peak-window highlighting |

## Lag Features (per zone: `Zone_1`, `Zone_2`, `Zone_3`)

| Feature pattern | Data Type | Formula/Source | Intended Usage |
|---|---|---|---|
| `{Zone}_lag_1` | float | Value 10 minutes prior (`shift(1)`) | Short-horizon forecasting input; strongest autocorrelation feature |
| `{Zone}_lag_6` | float | Value 1 hour prior (`shift(6)`) | Medium-horizon forecasting input |
| `{Zone}_lag_144` | float | Value 24 hours prior (`shift(144)`) | Daily-seasonality forecasting input; anomaly detection baseline ("same time yesterday") |

## Rolling Window Statistics (per zone, computed on `shift(1)` to avoid leakage)

| Feature pattern | Data Type | Formula/Source | Intended Usage |
|---|---|---|---|
| `{Zone}_roll_mean_1h` / `_24h` | float | Rolling mean over trailing 6 / 144 rows | Local trend feature; smoothing for regression/LSTM |
| `{Zone}_roll_std_1h` / `_24h` | float | Rolling std over trailing 6 / 144 rows | **Volatility feature — key input for anomaly detection** (spikes relative to recent variance) |
| `{Zone}_roll_min_1h` / `_24h` | float | Rolling min over trailing 6 / 144 rows | Local floor; anomaly lower-bound reference |
| `{Zone}_roll_max_1h` / `_24h` | float | Rolling max over trailing 6 / 144 rows | Local ceiling; anomaly upper-bound / **peak-demand reference** |

## Weather Interaction Feature

| Feature | Data Type | Formula/Source | Intended Usage |
|---|---|---|---|
| `Temp_Humidity_Index` | float | Simplified heat-index: `T + 0.5555*(0.06*RH*T - 10 - RH)` | Captures combined thermal-comfort driver of cooling load; regression/LSTM input |

## File-by-File Notes

- **`cleaned_energy_data.csv`** — missing values/duplicates handled, outliers capped (IQR, k=1.5) on the three zone columns. No engineered features yet. Use this if you want the cleaned series in original units without the modeling-specific columns (e.g. as an anomaly detection baseline).
- **`feature_engineered_energy_data.csv`** — all of the above, unscaled. Best for EDA, dashboarding, and clustering, since values remain human-readable (Watts, °C, etc.).
- **`train.csv` / `validation.csv` / `test.csv`** — chronological 70% / 15% / 15% split (no shuffling, to avoid time-series leakage). All numeric columns except `DateTime` are scaled with `MinMaxScaler` fit only on `train.csv`. `train.csv` has the first 144 rows (24h) dropped, since they lack full lag/rolling history; `validation.csv` and `test.csv` keep every row, with any edge-case NaNs forward/back-filled so every timestamp is scoreable in a live setting.

## Which File Should Each Team Use?

| Team | Recommended file(s) |
|---|---|
| Forecasting / Regression / LSTM-GRU | `train.csv`, `validation.csv`, `test.csv` |
| Anomaly Detection | `cleaned_energy_data.csv` or `feature_engineered_energy_data.csv` (unscaled); cross-reference `data/raw/` for pre-capping extremes |
| Consumption Clustering | `feature_engineered_energy_data.csv` (has `Season_*`, `Hour`, `Is_Peak_Hour`) |
| Dashboard | `feature_engineered_energy_data.csv` (unscaled, full `DateTime`) |
