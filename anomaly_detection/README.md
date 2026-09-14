# Anomaly Detection

This is the anomaly detection half of the "Anomaly Detection & Peak-Demand Prediction" team's deliverable. Peak-demand prediction is out of scope here — this folder flags unusual spikes/drops in consumption and hands off a scored file for the dashboard team (and for your teammate's peak-demand work, if useful as context).

Every claim below was verified by actually running this against the project's real data (`data/processed/feature_engineered_energy_data.csv`, `data/raw/Tetuan_City_power_consumption.csv`), not just written from the schema.

## Approach

Three independent detectors, run per zone (`Zone_1`, `Zone_2`, `Zone_3`), combined into a tiered severity:

| Signal                                   | Catches                                                                                     | Precision on synthetic benchmark\*       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Isolation Forest**                     | Multivariate context — value wrong for the weather/hour, even if not extreme alone          | ~0.03–0.05                               |
| **Rolling z-score** (mean/std, 1h & 24h) | Simple deviation from recent trend                                                          | ~0.44–0.49                               |
| **Robust MAD z-score** (median/MAD)      | Same idea as z-score, but resistant to being thrown off by the very spikes it's looking for | ~0.33–1.00                               |
| **Raw-vs-cleaned capping check**         | Real extreme values erased by upstream IQR capping                                          | n/a (deterministic, not threshold-tuned) |

\*See exact numbers per zone in `docs/results/results_anomaly_detection_benchmark.csv` after running.

**All rolling windows are trailing** — every `.rolling()` call is preceded by `.shift(1)`, and `center=True` is never used. A centered window needs future data to score "now," which breaks live/streaming scoring and silently reintroduces the same look-ahead leakage the preprocessing team explicitly avoided elsewhere in this project. Every score here only ever uses the past.

### Every threshold is tuned, not guessed

Isolation Forest's `contamination`, the z-score threshold, and the MAD threshold are each chosen by running a small grid through the synthetic-injection benchmark (see below) and keeping whichever value gets the best F1 — not by a proxy heuristic like matching an observed flag rate to a target rate.

### Why the raw-vs-cleaned check matters

`cleaned_energy_data.csv` (and everything derived from it) IQR-caps (k=1.5) each zone's values. Checked directly against this project's data:

| Zone   | Raw max (Watts) | Cleaned max (Watts) | Suppressed        |
| ------ | --------------- | ------------------- | ----------------- |
| Zone 1 | 52,204          | 52,204              | none              |
| Zone 2 | 37,409          | 36,313              | ~1,100            |
| Zone 3 | 47,598          | 34,366              | **~13,200 (28%)** |

Zone 3's single biggest real spike (2017-07-24 20:10) is invisible to any detector that only ever sees the capped file — it's been clipped to look like a normal-ish value. `raw_reference.py` loads `data/raw/Tetuan_City_power_consumption.csv` (renaming its inconsistent original column names — e.g. `"Zone 2  Power Consumption"` with a stray double space — to match the rest of the pipeline), compares it row-by-row against the cleaned value, and flags/recovers exactly this kind of suppressed spike. This runs automatically if `data/raw/` is present; it's skipped gracefully (with a warning) if not.

### Evaluation: synthetic anomaly injection

There are no labeled anomalies in this dataset, so a copy of the _test_ series has synthetic spikes/dips injected at ~2% of rows, and Precision/Recall/F1 are computed against those known injection points. This is the standard way to benchmark unsupervised anomaly detectors without ground truth, and it's what actually picks every threshold used in production, not just what gets reported afterward.

### Severity, not just a boolean

Each row gets `Signals_Agreeing` (0–3) and a `Severity_Level`:

- **Normal** — nothing fired.
- **Warning** — exactly one detector fired.
- **Critical** — two or more detectors agree, OR one signal is extreme (>2x its threshold) even alone, OR the row was both capped by preprocessing _and_ already flagged.

`Is_Anomaly` stays a simple "did anything fire" boolean for convenience, but `Severity_Level` is what you'd actually filter/sort a dashboard by.

## Folder Structure

```text
anomaly_detection/
├── README.md
├── requirements.txt
├── run_all.py
├── config.py
├── data.py            # feature-engineered + cleaned + raw loaders (raw column renaming lives here)
├── methods.py          # z-score, robust MAD z-score, Isolation Forest, synthetic injection
├── raw_reference.py     # raw-vs-cleaned IQR-capping comparison
├── evaluation.py         # synthetic-injection benchmark + threshold tuning
├── pipeline.py            # combines all signals into the final scored output
├── visualization.py
├── notebooks/
│   ├── 01_anomaly_detection.ipynb
│   └── figures/
├── docs/results/
│   └── results_anomaly_detection_benchmark.csv
├── internal_results/
│   ├── predictions/
│   │   └── predictions_anomaly_detection.csv
│   └── tuning/
│       └── anomaly_detection_threshold_grid.csv
└── outputs/              # <- hand this to Dashboard / Peak-Demand
    ├── anomalies_zone_1.csv
    ├── anomalies_zone_2.csv
    ├── anomalies_zone_3.csv
    └── anomalies_all_zones.csv
```

## Setup

Place this folder in the repo root, next to `forecasting_regression/`, so it can read:

```text
data/processed/feature_engineered_energy_data.csv
data/raw/Tetuan_City_power_consumption.csv   (optional but recommended — enables the capping check)
```

```bash
pip install -r anomaly_detection/requirements.txt
```

## Run

```bash
python -m anomaly_detection.run_all
```

Or open `notebooks/01_anomaly_detection.ipynb` for the same pipeline with inline results and plots.

## Output Contract (for Dashboard / Peak-Demand teammate)

Each row of `outputs/anomalies_<zone>.csv` is one timestamp for one zone:

| Column                                                                    | Meaning                                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `DateTime`, `Zone`, `Value_Watts`                                         | Identity + the (possibly capped) consumption value           |
| `Isolation_Forest_Score` / `_Flag`                                        | Multivariate signal                                          |
| `Zscore_1h` / `Zscore_24h` / `Statistical_Flag`                           | Rolling mean/std deviation                                   |
| `MAD_Zscore` / `MAD_Flag`                                                 | Robust median/MAD deviation — highest-precision of the three |
| `Deviation_vs_Yesterday_Watts`                                            | Value minus same time yesterday                              |
| `Signals_Agreeing`                                                        | 0–3                                                          |
| `Severity_Level`                                                          | **Normal / Warning / Critical** — filter/sort on this        |
| `Is_Anomaly`                                                              | Any detector fired                                           |
| `Raw_Value_Watts` / `Capped_By_Preprocessing` / `Suppressed_Amount_Watts` | Whether preprocessing altered this row, and by how much      |
| `Eligible_For_Scoring`                                                    | False for the first 24h (not enough history yet)             |

For a stricter, higher-confidence list: filter on `Severity_Level == "Critical"`, or `MAD_Flag == True` alone (highest measured precision of the three detectors).
