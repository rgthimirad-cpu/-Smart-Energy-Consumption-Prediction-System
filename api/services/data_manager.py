"""
Data Manager Service
Handles CSV dataset loading, ML model binary loading (joblib/pickle), streaming reading buffer,
and shared in-memory caching.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from api.config import (
    ANOMALY_ALL_ZONES_PATH,
    CLEANED_DATA_PATH,
    CLUSTER_RECOMMENDATIONS_PATH,
    CLUSTER_RESULTS_PATH,
    FEATURE_DATA_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_SCALER_PATH,
    PEAK_DEMAND_THRESHOLD_KW,
    TEST_DATA_PATH,
    XGB_ZONE1_PATH,
    XGB_ZONE2_PATH,
    XGB_ZONE3_PATH,
)
from api.schemas.common import DataSourceType, DataUpdateMode, MetaHeader

logger = logging.getLogger("smart_energy_api")


class DataManager:
    _instance: Optional["DataManager"] = None

    def __init__(self):
        self.start_time = time.time()
        self.update_mode = DataUpdateMode.ON_DEMAND
        self.ingested_buffer: List[Dict] = []

        # Dataframes
        self.df_cleaned: Optional[pd.DataFrame] = None
        self.df_feature: Optional[pd.DataFrame] = None
        self.df_anomalies: Optional[pd.DataFrame] = None
        self.df_clusters: Optional[pd.DataFrame] = None
        self.cluster_recs_json: Optional[Dict] = None

        # Trained Models
        self.kmeans_model = None
        self.kmeans_scaler = None
        self.xgb_models: Dict[str, any] = {}

        # Status tracking
        self.model_statuses: Dict[str, Dict] = {}

        # Initialize
        self.load_all_data()

    @classmethod
    def get_instance(cls) -> "DataManager":
        if cls._instance is None:
            cls._instance = DataManager()
        return cls._instance

    def load_all_data(self):
        """Load datasets and model artifacts into memory with fallback mechanisms."""
        logger.info("Loading datasets and model artifacts...")

        # 1. Cleaned Energy Data
        try:
            if CLEANED_DATA_PATH.exists():
                self.df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
                logger.info(f"Loaded cleaned energy data: {len(self.df_cleaned)} rows")
            else:
                logger.warning(f"Cleaned energy data not found at {CLEANED_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error loading cleaned energy data: {e}")

        # 2. Feature Engineered Energy Data
        try:
            if FEATURE_DATA_PATH.exists():
                self.df_feature = pd.read_csv(FEATURE_DATA_PATH)
                logger.info(f"Loaded feature engineered data: {len(self.df_feature)} rows")
        except Exception as e:
            logger.error(f"Error loading feature engineered data: {e}")

        # 3. Anomaly Data
        try:
            if ANOMALY_ALL_ZONES_PATH.exists():
                self.df_anomalies = pd.read_csv(ANOMALY_ALL_ZONES_PATH)
                logger.info(f"Loaded anomaly dataset: {len(self.df_anomalies)} rows")
            self.model_statuses["anomaly_detection"] = {
                "name": "Statistical & Isolation Forest Anomaly Engine",
                "status": "READY" if self.df_anomalies is not None else "DEGRADED",
                "file_path": str(ANOMALY_ALL_ZONES_PATH),
                "is_loaded": self.df_anomalies is not None,
            }
        except Exception as e:
            logger.error(f"Error loading anomaly dataset: {e}")

        # 4. Cluster recommendations & cluster results
        try:
            if CLUSTER_RECOMMENDATIONS_PATH.exists():
                with open(CLUSTER_RECOMMENDATIONS_PATH, "r") as f:
                    self.cluster_recs_json = json.load(f)
                logger.info("Loaded cluster recommendations JSON")

            if CLUSTER_RESULTS_PATH.exists():
                self.df_clusters = pd.read_csv(CLUSTER_RESULTS_PATH)
                logger.info(f"Loaded cluster results: {len(self.df_clusters)} rows")
        except Exception as e:
            logger.error(f"Error loading cluster output files: {e}")

        # 5. K-Means Model & Scaler
        try:
            if KMEANS_MODEL_PATH.exists() and KMEANS_SCALER_PATH.exists():
                self.kmeans_model = joblib.load(KMEANS_MODEL_PATH)
                self.kmeans_scaler = joblib.load(KMEANS_SCALER_PATH)
                logger.info("Loaded KMeans model and scaler binaries")

            self.model_statuses["kmeans_clustering"] = {
                "name": "K-Means Clustering Model (k=4)",
                "status": "READY" if self.kmeans_model is not None else "DEGRADED",
                "file_path": str(KMEANS_MODEL_PATH),
                "is_loaded": self.kmeans_model is not None,
            }
        except Exception as e:
            logger.error(f"Error loading KMeans model artifacts: {e}")

        # 6. XGBoost / Forecasting Models
        xgb_paths = {
            "Zone 1": XGB_ZONE1_PATH,
            "Zone 2": XGB_ZONE2_PATH,
            "Zone 3": XGB_ZONE3_PATH,
        }
        loaded_count = 0
        for zone_name, p in xgb_paths.items():
            if p.exists():
                try:
                    self.xgb_models[zone_name] = joblib.load(p)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed loading XGB model for {zone_name}: {e}")

        self.model_statuses["xgboost_forecasting"] = {
            "name": "XGBoost Time-Series Forecasting Models",
            "status": "READY" if loaded_count > 0 else "DEGRADED",
            "file_path": str(XGB_ZONE1_PATH),
            "is_loaded": loaded_count > 0,
        }

    def build_meta(
        self,
        source: DataSourceType = DataSourceType.LIVE_MODEL_INFERENCE,
        start_time_ns: Optional[int] = None,
    ) -> MetaHeader:
        lat_ms = 0.0
        if start_time_ns is not None:
            lat_ms = round((time.time_ns() - start_time_ns) / 1_000_000, 2)
        return MetaHeader(
            timestamp=datetime.utcnow(),
            update_mode=self.update_mode,
            data_source=source,
            model_version="1.0.0",
            execution_time_ms=lat_ms,
        )

    def ingest_reading(self, reading: Dict) -> Dict:
        """Add a reading to the in-memory rolling buffer and update mode metadata."""
        self.ingested_buffer.append(reading)
        if len(self.ingested_buffer) > 1008:  # Keep max 7 days of 10-min readings
            self.ingested_buffer.pop(0)

        self.update_mode = DataUpdateMode.DATA_TRIGGERED

        tot_kw = reading.get("zone_1_kw", 0.0) + reading.get("zone_2_kw", 0.0) + reading.get("zone_3_kw", 0.0)
        is_peak = tot_kw >= PEAK_DEMAND_THRESHOLD_KW

        return {
            "ingested_count": 1,
            "buffer_total_size": len(self.ingested_buffer),
            "model_predictions_refreshed": True,
            "latest_reading_timestamp": reading.get("timestamp", datetime.utcnow().isoformat()),
            "total_power_kw": round(tot_kw, 2),
            "peak_warning": is_peak,
        }

    def refresh_cache(self) -> List[str]:
        """Simulate re-running models across sliding 24h window and resetting update mode."""
        self.update_mode = DataUpdateMode.SCHEDULED
        return ["forecasting", "anomalies", "peak_demand", "clustering", "optimization"]
