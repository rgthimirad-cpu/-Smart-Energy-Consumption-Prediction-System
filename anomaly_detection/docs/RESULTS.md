# Anomaly Detection — Results and Observations

This document records the verified results obtained from the anomaly detection pipeline after running the complete system against the project's feature-engineered energy-consumption dataset.

The results in this document are based on an actual execution of:

```bash
python -m anomaly_detection.run_all
```

The pipeline completed successfully for Zone 1, Zone 2, and Zone 3.

---

# 1\. Dataset Summary

The pipeline loaded:

```plaintext
52,416 rows
```

covering:

```plaintext
2017-01-01 00:00:00
to
2017-12-30 23:50:00
```

The data was divided chronologically into training, validation, and test sets.

| Split      |   Rows | Start            | End              |
| ---------- | -----: | ---------------- | ---------------- |
| Train      | 36,691 | 2017-01-01 00:00 | 2017-09-12 19:00 |
| Validation |  7,862 | 2017-09-12 19:10 | 2017-11-06 09:20 |
| Test       |  7,863 | 2017-11-06 09:30 | 2017-12-30 23:50 |

The split preserves temporal ordering and prevents future observations from being used during model training.

---

# 2\. Selected Parameters

The anomaly detection thresholds were selected automatically during the tuning stage using the synthetic anomaly benchmark.

| Zone   | Isolation Forest Contamination | Statistical Z-score | MAD Z-score |
| ------ | -----------------------------: | ------------------: | ----------: |
| Zone 1 |                           0.02 |                 4.0 |         3.0 |
| Zone 2 |                           0.05 |                 4.0 |         3.0 |
| Zone 3 |                           0.05 |                 4.0 |         5.0 |

The parameters are therefore zone-specific rather than assuming that every consumption zone has identical anomaly behaviour.

---

# 3\. Synthetic Benchmark

Because the original dataset does not contain ground-truth anomaly labels, synthetic anomalies were injected into the test data.

Each zone contained:

```plaintext
156 injected anomalies
```

Precision, recall, and F1 were then calculated against the known injected anomaly locations.

---

# 4\. Zone 1 Results

## 4.1 Benchmark Performance

| Detector              |  Precision | Recall |         F1 | Flagged |
| --------------------- | ---------: | -----: | ---------: | ------: |
| Isolation Forest      |     0.0327 | 0.1603 |     0.0543 |     765 |
| Statistical Threshold |     0.4896 | 0.9038 |     0.6351 |     288 |
| Robust MAD Z-score    | **1.0000** | 0.5962 | **0.7470** |      93 |

### Observation

The Robust MAD detector performed best for Zone 1 based on F1.

It achieved:

```plaintext
Precision = 1.0000
Recall    = 0.5962
F1        = 0.7470
```

This indicates that the MAD detector produced a highly precise anomaly list, although it did not identify every injected anomaly.

The statistical detector achieved substantially higher recall:

```plaintext
90.38%
```

but at the cost of lower precision.

Isolation Forest performed poorly as a standalone detector, with:

```plaintext
Precision = 3.27%
F1        = 5.43%
```

---

## 4.2 Full Dataset

| Measure                    | Result |
| -------------------------- | -----: |
| Eligible rows              | 52,272 |
| Flagged anomalies          |  2,212 |
| Flagged percentage         |  4.23% |
| Critical observations      |    142 |
| Preprocessing-changed rows |      0 |

### Observation

Zone 1 has the lowest overall anomaly rate among the three zones.

Only 4.23% of eligible observations were flagged.

No observations were changed by the preprocessing capping stage.

---

# 5\. Zone 2 Results

## 5.1 Benchmark Performance

| Detector              |  Precision |     Recall |         F1 | Flagged |
| --------------------- | ---------: | ---------: | ---------: | ------: |
| Isolation Forest      |     0.0241 |     0.4167 |     0.0456 |   2,693 |
| Statistical Threshold |     0.4778 | **0.8974** |     0.6236 |     293 |
| Robust MAD Z-score    | **1.0000** |     0.4551 | **0.6256** |      71 |

### Observation

The Robust MAD detector achieved perfect precision on the synthetic benchmark:

```plaintext
Precision = 1.0000
```

However, its recall was only 45.51%.

The statistical detector achieved much higher recall:

```plaintext
89.74%
```

and produced an F1 score of 0.6236.

The two statistical detectors therefore provide complementary behaviour:

- Statistical threshold → better coverage

- MAD → higher precision

Isolation Forest again produced a very large number of false positives relative to the number of injected anomalies.

---

## 5.2 Full Dataset

| Measure                    | Result |
| -------------------------- | -----: |
| Eligible rows              | 52,272 |
| Flagged anomalies          |  5,559 |
| Flagged percentage         | 10.63% |
| Critical observations      |    440 |
| Preprocessing-changed rows |      7 |

### Observation

Zone 2 has a considerably higher anomaly rate than Zone 1.

Approximately one in ten eligible observations was flagged.

Only seven observations were modified by preprocessing, so preprocessing capping has a relatively small impact on the Zone 2 results.

---

# 6\. Zone 3 Results

## 6.1 Benchmark Performance

| Detector              | Precision |     Recall |         F1 | Flagged |
| --------------------- | --------: | ---------: | ---------: | ------: |
| Isolation Forest      |    0.0264 |     0.5192 |     0.0503 |   3,063 |
| Statistical Threshold |    0.4357 | **0.8910** | **0.5853** |     319 |
| Robust MAD Z-score    |    0.4417 |     0.6795 |     0.5354 |     240 |

### Observation

Zone 3 differs from Zones 1 and 2.

The Statistical Threshold detector achieved the best F1:

```plaintext
F1 = 0.5853
```

and the highest recall:

```plaintext
Recall = 89.10%
```

The Robust MAD detector performed slightly better in precision:

```plaintext
44.17%
```

versus:

```plaintext
43.57%
```

for the statistical detector.

However, its recall was lower.

Isolation Forest again had very low precision.

---

## 6.2 Full Dataset

| Measure                    | Result |
| -------------------------- | -----: |
| Eligible rows              | 52,272 |
| Flagged anomalies          |  7,581 |
| Flagged percentage         | 14.50% |
| Critical observations      |  1,817 |
| Preprocessing-changed rows |  1,191 |

### Observation

Zone 3 has the highest anomaly rate:

```plaintext
14.50%
```

It also has by far the largest number of observations affected by preprocessing.

This makes Zone 3 the most important zone for further investigation.

---

# 7\. Cross-Zone Comparison

| Metric                |    Zone 1 |     Zone 2 |     Zone 3 |
| --------------------- | --------: | ---------: | ---------: |
| Eligible rows         |    52,272 |     52,272 |     52,272 |
| Flagged anomalies     |     2,212 |      5,559 |      7,581 |
| Flagged %             | **4.23%** | **10.63%** | **14.50%** |
| Critical              |       142 |        440 |      1,817 |
| Preprocessing changed |         0 |          7 |      1,191 |

### Main observation

The anomaly burden increases substantially from Zone 1 to Zone 3.

```plaintext
Zone 1 → 4.23%
Zone 2 → 10.63%
Zone 3 → 14.50%
```

Zone 3 therefore requires the greatest attention during anomaly review.

---

# 8\. Detector Comparison

The benchmark demonstrates that no single detector dominates across all zones.

## Isolation Forest

Isolation Forest produced:

- low precision,

- relatively high numbers of flagged observations,

- low F1 scores.

Its main value is therefore as a **multivariate supporting signal** rather than as a standalone high-confidence detector.

---

## Statistical Threshold

The rolling statistical detector produced consistently high recall:

| Zone   | Recall |
| ------ | -----: |
| Zone 1 | 90.38% |
| Zone 2 | 89.74% |
| Zone 3 | 89.10% |

This makes it useful when the objective is to capture a large proportion of unusual observations.

---

## Robust MAD

MAD produced very high precision for Zones 1 and 2:

| Zone   | Precision |
| ------ | --------: |
| Zone 1 |   100.00% |
| Zone 2 |   100.00% |
| Zone 3 |    44.17% |

Therefore, MAD should not be described as universally superior.

Its effectiveness varies by zone.

For Zone 1, it produced the best F1.

For Zone 2, it achieved very high precision and a slightly higher F1 than the statistical detector.

For Zone 3, the statistical detector produced the better F1.

---

# 9\. Preprocessing Observations

One of the important findings from the anomaly detection pipeline is the effect of upstream IQR capping.

The number of observations changed by preprocessing was:

```plaintext
Zone 1:     0
Zone 2:     7
Zone 3: 1,191
```

Zone 3 is therefore substantially more affected by preprocessing than the other zones.

This is important because an anomaly detector operating only on the cleaned dataset may fail to recognize the original magnitude of some extreme events.

The pipeline therefore retains information such as:

```plaintext
Raw_Value_Watts
Capped_By_Preprocessing
Suppressed_Amount_Watts
```

to make these cases identifiable.

---

# 10\. Zone 3 Preprocessing Observation

Zone 3 contains the strongest evidence that preprocessing can hide extreme consumption events.

The raw-versus-cleaned comparison indicates that extreme Zone 3 observations were substantially affected by IQR capping.

Consequently, an observation can be statistically ordinary in the cleaned dataset while having originally been a very large raw consumption event.

This is why the final output should not be interpreted using only:

```plaintext
Value_Watts
```

The following fields should also be considered:

```plaintext
Raw_Value_Watts
Capped_By_Preprocessing
Suppressed_Amount_Watts
```

especially when investigating Critical observations.

---

# 11\. Severity Results

The pipeline combines the detector outputs into:

```plaintext
Signals_Agreeing
Severity_Level
Is_Anomaly
```

The resulting severity categories are:

| Severity | Interpretation                                                                              |
| -------- | ------------------------------------------------------------------------------------------- |
| Normal   | No detector fired                                                                           |
| Warning  | One detector fired                                                                          |
| Critical | Multiple signals agree or the observation satisfies the pipeline's extreme-anomaly criteria |

The number of Critical observations was:

| Zone   | Critical |
| ------ | -------: |
| Zone 1 |      142 |
| Zone 2 |      440 |
| Zone 3 |    1,817 |

Zone 3 therefore has substantially more high-severity observations than the other zones.

---

# 12\. Detector Agreement Visualizations

The pipeline successfully generated and verified all three detector-agreement plots:

```plaintext
notebooks/figures/detector_agreement_zone_1.png
notebooks/figures/detector_agreement_zone_2.png
notebooks/figures/detector_agreement_zone_3.png
```

The files were verified using:

```powershell
Test-Path .\anomaly_detection\notebooks\figures\detector_agreement_zone_1.png
Test-Path .\anomaly_detection\notebooks\figures\detector_agreement_zone_2.png
Test-Path .\anomaly_detection\notebooks\figures\detector_agreement_zone_3.png
```

All three returned:

```plaintext
True
```

---

# 13\. Other Generated Visualizations

The following plots were also successfully generated:

```plaintext
anomalies_zone_1.png
anomalies_zone_2.png
anomalies_zone_3.png

severity_distribution_zone_1.png
severity_distribution_zone_2.png
severity_distribution_zone_3.png
```

All visualization files are stored under:

```plaintext
anomaly_detection/notebooks/figures/
```

---

# 14\. Generated Data Files

The successful run produced the following dashboard/project handoff files:

```plaintext
anomaly_detection/outputs/
├── anomalies_zone_1.csv
├── anomalies_zone_2.csv
├── anomalies_zone_3.csv
└── anomalies_all_zones.csv
```

The combined file is:

```plaintext
anomaly_detection/outputs/anomalies_all_zones.csv
```

This is the main file intended for downstream dashboard/project integration.

---

# 15\. Benchmark Output

The exact benchmark results are stored in:

```plaintext
anomaly_detection/docs/results/results_anomaly_detection_benchmark.csv
```

This CSV should be treated as the authoritative machine-readable benchmark record.

It contains the detector-level:

- Precision

- Recall

- F1

- Injected anomaly count

- Flagged count

- Benchmark notes

for each zone.

---

# 16\. Tuning Output

The complete threshold-search results are stored in:

```plaintext
anomaly_detection/internal_results/tuning/anomaly_detection_threshold_grid.csv
```

This provides the detailed parameter combinations evaluated during tuning.

The selected parameters reported in this document correspond to the best-performing configuration selected by the tuning procedure.

---

# 17\. Internal Prediction Output

The complete combined row-level predictions are stored in:

```plaintext
anomaly_detection/internal_results/predictions/predictions_anomaly_detection.csv
```

This file is intended for internal analysis and reproducibility.

---

# 18\. Recommended Interpretation

The results should not be interpreted as proof that every flagged observation is a real-world anomaly because the original dataset does not contain manually verified anomaly labels.

The synthetic benchmark measures how effectively the detectors recover artificially injected anomalies.

The full-dataset anomaly counts therefore represent:

> observations identified as unusual by the implemented detection rules,

rather than confirmed physical failures or confirmed abnormal consumption events.

For operational review, higher-confidence observations should receive priority.

Recommended filtering:

```plaintext
Severity_Level == "Critical"
```

For robust statistical investigation:

```plaintext
MAD_Flag == True
```

For preprocessing investigation:

```plaintext
Capped_By_Preprocessing == True
```

---

# 19\. Main Findings

The verified run produced the following main findings:

### Finding 1 — Zone 3 has the highest anomaly burden

Zone 3 had:

```plaintext
7,581 flagged observations
14.50% of eligible observations
```

compared with:

```plaintext
Zone 1: 2,212 / 4.23%
Zone 2: 5,559 / 10.63%
```

---

### Finding 2 — Zone 3 has substantially more preprocessing changes

Zone 3 had:

```plaintext
1,191 preprocessing-changed observations
```

while Zone 1 had none and Zone 2 had only seven.

This makes raw-vs-cleaned inspection particularly important for Zone 3.

---

### Finding 3 — Statistical detection provides consistently high recall

The statistical detector achieved approximately 89–90% recall across all three zones in the synthetic benchmark.

This makes it useful for broad anomaly screening.

---

### Finding 4 — MAD provides high precision in Zones 1 and 2

MAD achieved:

```plaintext
Zone 1 precision = 100%
Zone 2 precision = 100%
```

on the synthetic benchmark.

However, this behaviour did not generalize equally to Zone 3.

---

### Finding 5 — Detector performance is zone-dependent

The best-performing detector by F1 differed between zones.

| Zone   | Best F1 Detector      |     F1 |
| ------ | --------------------- | -----: |
| Zone 1 | Robust MAD            | 0.7470 |
| Zone 2 | Robust MAD            | 0.6256 |
| Zone 3 | Statistical Threshold | 0.5853 |

Therefore, using independently tuned parameters per zone is justified.

---

### Finding 6 — Isolation Forest is most useful as a supporting signal

Isolation Forest had substantially lower precision and F1 than the statistical detectors in the synthetic benchmark.

Its multivariate signal is nevertheless useful when combined with other detectors for agreement-based severity classification.

---

# 20\. Final Conclusion

The anomaly detection pipeline successfully completed on the project's real feature-engineered dataset.

It produced:

- tuned anomaly detection parameters,

- synthetic benchmark metrics,

- per-zone anomaly predictions,

- combined anomaly results,

- severity classifications,

- detector-agreement visualizations,

- severity-distribution visualizations,

- raw-vs-cleaned preprocessing information.

The most important operational finding is that **Zone 3 requires the greatest investigation**, due to its combination of:

```plaintext
14.50% flagged observations
1,817 Critical observations
1,191 preprocessing-changed observations
```

The results also demonstrate that anomaly detection performance differs between zones. Consequently, the pipeline uses zone-specific threshold tuning rather than applying one universal threshold to all consumption zones.

The generated CSV outputs under:

```plaintext
anomaly_detection/outputs/
```

are the primary handoff artifacts for downstream dashboard and peak-demand analysis.

The benchmark CSV under:

```plaintext
anomaly_detection/docs/results/results_anomaly_detection_benchmark.csv
```

provides the detailed machine-readable evaluation results.

---

# 21\. Result Artifacts

The verified run generated:

```plaintext
outputs/
├── anomalies_zone_1.csv
├── anomalies_zone_2.csv
├── anomalies_zone_3.csv
└── anomalies_all_zones.csv

docs/results/
└── results_anomaly_detection_benchmark.csv

internal_results/tuning/
└── anomaly_detection_threshold_grid.csv

internal_results/predictions/
└── predictions_anomaly_detection.csv

notebooks/figures/
├── anomalies_zone_1.png
├── anomalies_zone_2.png
├── anomalies_zone_3.png
├── detector_agreement_zone_1.png
├── detector_agreement_zone_2.png
├── detector_agreement_zone_3.png
├── severity_distribution_zone_1.png
├── severity_distribution_zone_2.png
└── severity_distribution_zone_3.png
```

All listed artifacts were generated by the successful pipeline execution.
