"""
Configuration settings for Smart Energy Consumption Prediction System API.
Handles path resolution for models, datasets, cache settings, and default parameters.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
ANOMALY_DIR = BASE_DIR / "anomaly_detection" / "outputs"
CLUSTERING_DIR = BASE_DIR / "clustering"
FORECASTING_DIR = BASE_DIR / "forecasting_regression"

# Dataset paths
CLEANED_DATA_PATH = DATA_DIR / "cleaned_energy_data.csv"
FEATURE_DATA_PATH = DATA_DIR / "feature_engineered_energy_data.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
ANOMALY_ALL_ZONES_PATH = ANOMALY_DIR / "anomalies_all_zones.csv"
CLUSTER_RESULTS_PATH = CLUSTERING_DIR / "outputs" / "energy_cluster_results.csv"
CLUSTER_RECOMMENDATIONS_PATH = CLUSTERING_DIR / "outputs" / "recommendations.json"

# Model paths
KMEANS_MODEL_PATH = CLUSTERING_DIR / "models" / "kmeans_model.pkl"
KMEANS_SCALER_PATH = CLUSTERING_DIR / "models" / "scaler.pkl"
XGB_ZONE1_PATH = FORECASTING_DIR / "models" / "regression" / "xgb_zone1.joblib"
XGB_ZONE2_PATH = FORECASTING_DIR / "models" / "regression" / "xgb_zone2.joblib"
XGB_ZONE3_PATH = FORECASTING_DIR / "models" / "regression" / "xgb_zone3.joblib"

# API Metadata & Defaults
API_TITLE = "Smart Energy Consumption Prediction API"
API_DESCRIPTION = (
    "RESTful API serving ML model outputs for time-series forecasting, "
    "anomaly detection, peak-demand prediction, consumption clustering, "
    "and actionable energy optimization recommendations."
)
API_VERSION = "1.0.0"

# Domain Thresholds
PEAK_DEMAND_THRESHOLD_KW = 90000.0  # 90th percentile threshold for total demand (~90k kW)
DEFAULT_HORIZON_STEPS = 144        # 24 hours of 10-minute intervals
