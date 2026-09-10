# Smart Energy Consumption Prediction System — Data Engineering & Preprocessing- 

This repository contains the **Data Engineering & Preprocessing** component of the Smart Energy Consumption Prediction System group project. It sources, cleans, and feature-engineers the Tetuan City power consumption dataset, then hands off a documented, split, and scaled dataset to the Forecasting, Anomaly Detection, Clustering, and Dashboard teams.

## Project Structure

```
Smart-Energy-Consumption-Prediction-System/
│
├── data/
│   ├── raw/
│   │   └── Tetuan_City_power_consumption.csv
│   │
│   └── processed/
│       ├── cleaned_energy_data.csv
│       ├── feature_engineered_energy_data.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── figures/
│
├── docs/
│   └── feature_dictionary.md
│
├── README.md
└── requirements.txt
```

## Dataset

**Source:** Tetuan City power consumption (Tetouan, Morocco) — 52,416 rows of 10-minute interval smart meter readings across three distribution zones, alongside weather parameters (Temperature, Humidity, Wind Speed, and two diffuse-flow solar readings).

## Pipeline Overview

The notebook `notebooks/data_preprocessing.ipynb` runs five stages:

1. **Data Exploration & Quality Analysis (EDA)** — shape, dtypes, missing/duplicate checks, descriptive statistics, and three exploratory figures (time-series by zone, weather/zone correlation heatmap, daily & monthly consumption profiles).
2. **Preprocessing & Outlier Handling** — datetime parsing, defensive missing-value/duplicate handling, IQR-based outlier detection with before/after boxplots, and treatment via capping (winsorizing) so the fixed 10-minute time grid stays intact. Produces `cleaned_energy_data.csv`.
3. **Advanced Feature Engineering** — time-based features (hour, day of week, month, quarter, weekend flag, season, peak-hour flag), lag features (10 min / 1 hr / 24 hr) per zone, rolling window statistics (1 hr / 24 hr mean/std/min/max) per zone, and a temperature–humidity interaction index. Produces `feature_engineered_energy_data.csv`.
4. **Train / Validation / Test Split & Scaling** — chronological 70% / 15% / 15% split (no shuffling, to prevent time-series leakage), with `MinMaxScaler` fit only on the training set. Produces `train.csv`, `validation.csv`, `test.csv`.
5. **Exports & Deliverables** — all processed CSVs saved to `data/processed/`, all figures saved to `notebooks/figures/`, and the full data dictionary written to `docs/feature_dictionary.md`.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/data_preprocessing.ipynb
```

Run all cells top to bottom. The notebook resolves paths relative to its own location (`notebooks/`), so it must be run from inside that folder (which is the default when opened via Jupyter from the project root).

## Deliverables

| File | Description |
|---|---|
| `data/processed/cleaned_energy_data.csv` | Cleaned dataset (missing values, duplicates, outliers handled), before feature engineering |
| `data/processed/feature_engineered_energy_data.csv` | Full feature-engineered dataset, unscaled |
| `data/processed/train.csv` | Training split (70%), scaled |
| `data/processed/validation.csv` | Validation split (15%), scaled |
| `data/processed/test.csv` | Test split (15%), scaled |
| `notebooks/figures/` | Exported PNGs of all EDA visualizations |
| `docs/feature_dictionary.md` | Full data dictionary — every feature's type, formula/source, and intended downstream use |

See the table at the end of `docs/feature_dictionary.md` — in short: modeling teams (Forecasting/Regression/LSTM-GRU) use `train.csv`/`validation.csv`/`test.csv`; Anomaly Detection and Dashboard teams use the unscaled `cleaned_energy_data.csv` or `feature_engineered_energy_data.csv`; Clustering uses `feature_engineered_energy_data.csv`.

