# Smart Energy Consumption Prediction System

A full-stack machine learning system for real-time energy consumption **forecasting**, **anomaly detection**, **peak demand prediction**, **consumer clustering**, and **optimization recommendations** — powered by a FastAPI backend and a live interactive dashboard.

---

## Project Structure

```
Smart-Energy-Consumption-Prediction-System/
│
├── data/
│   ├── raw/
│   │   └── Tetuan_City_power_consumption.csv
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
├── forecasting_regression/       ← Regression forecasting models
├── lstm_gru/                     ← LSTM / GRU deep learning models
├── anomaly_detection/            ← Isolation Forest + Z-score detection
├── clustering/                   ← K-Means consumer segmentation
│
├── api/                          ← FastAPI REST backend
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── forecasting.py
│   │   ├── anomalies.py
│   │   ├── peak_demand.py
│   │   ├── clustering.py
│   │   ├── optimization.py
│   │   ├── ingest.py
│   │   └── system.py
│   ├── schemas/
│   └── services/
│
├── dashboard/                    ← Interactive frontend dashboard
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── api.js                ← Central API client (all endpoints)
│       ├── app.js                ← SPA navigation controller
│       ├── forecasting.js        ← Member 2: forecast charts
│       ├── anomalies.js          ← Member 2: anomaly & peak visuals
│       ├── clustering.js         ← Member 3: cluster scatter plots
│       └── recommendations.js   ← Member 3: recommendation cards
│
├── run_api.py                    ← API server launcher
├── requirements.txt
└── README.md
```

---

## System Overview

### 1. Data Engineering & Preprocessing

**Dataset:** Tetuan City power consumption (Tetouan, Morocco) — 52,416 rows of 10-minute interval smart meter readings across three distribution zones, with weather features (Temperature, Humidity, Wind Speed, Diffuse Flow).

The notebook `notebooks/data_preprocessing.ipynb` runs five stages:

1. **EDA** — shape, dtypes, missing/duplicate checks, descriptive statistics, time-series and correlation plots.
2. **Preprocessing & Outlier Handling** — datetime parsing, IQR-based outlier detection with winsorizing. Produces `cleaned_energy_data.csv`.
3. **Feature Engineering** — time-based, lag (10 min / 1 hr / 24 hr), rolling window statistics, and temperature–humidity interaction features. Produces `feature_engineered_energy_data.csv`.
4. **Train / Validation / Test Split & Scaling** — chronological 70% / 15% / 15% split with `MinMaxScaler` fit only on the training set. Produces `train.csv`, `validation.csv`, `test.csv`.
5. **Exports** — all processed CSVs and figures saved to their respective directories.

---

### 2. 📈 Forecasting (Regression & LSTM/GRU)

ML models trained to predict energy consumption across zones. Outputs used by the dashboard's Forecasting section.

---

### 3. Anomaly Detection

Multi-algorithm consensus using **Isolation Forest**, **3-sigma Z-score**, and **MAD robust Z-score**. Flags unusual consumption events with severity ratings: `NORMAL` → `LOW` → `MEDIUM` → `HIGH` → `CRITICAL`.

---

### 4. Peak Demand Prediction

Predicts whether total demand in the next 10-minute interval will exceed the 90th-percentile peak threshold.

| Metric | Validation Result |
|---|---:|
| Accuracy | 97.97% |
| Precision | 58.94% |
| Recall | 91.36% |
| F1 Score | 71.66% |
| ROC-AUC | 99.34% |
| PR-AUC | 83.85% |

**Best model:** Logistic Regression — recent lag and rolling features are the top predictors.

---

### 5. Clustering & Recommendations

K-Means consumer segmentation into 4 cluster profiles, each with efficiency ratings, usage behavior summaries, and targeted energy optimization recommendations with estimated kW and cost savings.

---

### 6. Dashboard (EnergyIQ)

A styled, dark-themed single-page dashboard that connects to the live API and displays outputs from every model team.

**Sections:**
| Section | Visuals |
|---|---|
| Forecasting | Actual vs Predicted dual-line chart, confidence bounds, KPI cards, forward forecast form |
| Anomalies & Peaks | Anomaly time-series with marker overlays, event feed, peak demand alert cards |
| Clusters & Recs | Cluster scatter plot, segment cards, "Classify My Meter" form, recommendation cards, action plan timeline |

---

## How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Start the API backend

```bash
py run_api.py
```

API will be available at **http://localhost:8000**  
Interactive docs at **http://localhost:8000/docs**

### Step 3 — Start the Dashboard

```bash
py -m http.server 3000 --directory dashboard
```

Open **http://localhost:3000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/forecasting/actual-vs-predicted` | Historical actual vs predicted series |
| `POST` | `/api/v1/forecasting/predict` | Generate forward multi-step forecast |
| `GET` | `/api/v1/anomalies` | List flagged anomaly events |
| `GET` | `/api/v1/anomalies/summary` | Anomaly detection aggregate stats |
| `POST` | `/api/v1/anomalies/detect` | On-demand single reading detection |
| `GET` | `/api/v1/peak-demand/upcoming-peaks` | Upcoming predicted peak windows |
| `POST` | `/api/v1/peak-demand/predict` | Next-interval peak probability |
| `GET` | `/api/v1/clustering/groups` | All cluster profiles and centroids |
| `GET` | `/api/v1/clustering/distribution` | Cluster record distribution |
| `POST` | `/api/v1/clustering/classify` | Classify a meter reading into a cluster |
| `GET` | `/api/v1/optimization/recommendations` | Energy optimization recommendations |
| `POST` | `/api/v1/optimization/action-plan` | Generate custom operator action plan |
| `POST` | `/api/v1/ingest/readings` | Ingest live smart meter readings |
| `GET` | `/api/v1/system/status` | System health and model load status |
| `POST` | `/api/v1/system/refresh` | Trigger on-demand model refresh |

---

## Key Deliverables

| File / Folder | Description |
|---|---|
| `data/processed/cleaned_energy_data.csv` | Cleaned dataset |
| `data/processed/feature_engineered_energy_data.csv` | Full feature-engineered dataset |
| `data/processed/train.csv` / `validation.csv` / `test.csv` | Scaled chronological splits |
| `docs/feature_dictionary.md` | Full data dictionary for all features |
| `api/` | FastAPI REST backend serving all model outputs |
| `dashboard/` | Interactive EnergyIQ frontend dashboard |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Deep Learning | LSTM / GRU (see `lstm_gru/`) |
| API Backend | FastAPI, Uvicorn, Pydantic v2 |
| Dashboard Frontend | HTML5, Vanilla CSS, Vanilla JS (ES Modules) |
| Charts | Chart.js 4 |
| Notebook | Jupyter |
