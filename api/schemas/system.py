"""
Pydantic schemas for Data Ingestion, System Status, and Data Refresh API endpoints.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import DataUpdateMode, MetaHeader


class SmartMeterReading(BaseModel):
    timestamp: str = Field(..., description="Timestamp of meter reading in ISO format", example="2026-09-20 14:10:00")
    zone_1_kw: float = Field(..., description="Zone 1 power reading in kW", example=29100.0)
    zone_2_kw: float = Field(..., description="Zone 2 power reading in kW", example=23400.0)
    zone_3_kw: float = Field(..., description="Zone 3 power reading in kW", example=24800.0)
    temperature_c: float = Field(..., description="Ambient outdoor temperature in °C", example=27.4)
    humidity_pct: float = Field(..., description="Relative humidity %", example=58.2)
    wind_speed_ms: Optional[float] = Field(1.5, description="Wind speed in m/s", example=1.5)


class ReadingIngestRequest(BaseModel):
    readings: List[SmartMeterReading] = Field(..., description="Batch or single list of incoming 10-min smart meter readings")
    trigger_model_updates: bool = Field(True, description="Whether to immediately re-calculate predictions and cache state")


class ReadingIngestResponse(BaseModel):
    meta: MetaHeader
    ingested_count: int = Field(..., description="Count of successfully ingested meter readings", example=1)
    buffer_total_size: int = Field(..., description="Total active readings in rolling buffer", example=145)
    model_predictions_refreshed: bool = Field(..., description="Whether live predictions were recalculated", example=True)
    latest_reading_timestamp: str = Field(..., description="Timestamp of latest reading in buffer", example="2026-09-20 14:10:00")
    total_power_kw: float = Field(..., description="Total power across all zones in latest reading", example=77300.0)
    peak_warning: bool = Field(..., description="Peak warning flag for latest reading", example=False)


class RefreshSystemResponse(BaseModel):
    meta: MetaHeader
    status: str = Field("SUCCESS", description="Refresh operation result status", example="SUCCESS")
    models_refreshed: List[str] = Field(..., description="List of models re-executed", example=["forecasting", "anomalies", "peak_demand", "clustering"])
    records_processed: int = Field(..., description="Count of historical records processed", example=52416)
    refreshed_at: str = Field(..., description="Timestamp of refresh operation", example="2026-09-20 21:47:00")


class ModelHealthStatus(BaseModel):
    name: str = Field(..., description="Model name", example="XGBoost Zone Forecasting")
    status: str = Field(..., description="Model health status: 'READY', 'DEGRADED', or 'OFFLINE'", example="READY")
    file_path: str = Field(..., description="Path to underlying model asset or data file", example="forecasting_regression/models/regression/xgb_zone1.joblib")
    is_loaded: bool = Field(..., description="Boolean indicating if model binary is loaded in memory", example=True)


class SystemStatusResponse(BaseModel):
    meta: MetaHeader
    system_name: str = Field("Smart Energy Consumption Prediction System API", description="System identifier")
    api_version: str = Field("1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="API service uptime in seconds", example=3600.0)
    current_update_mode: DataUpdateMode = Field(..., description="Active data flow update strategy", example="on_demand")
    buffer_readings_count: int = Field(..., description="Number of readings currently held in memory buffer", example=144)
    dataset_records_count: int = Field(..., description="Total historical dataset records loaded", example=52416)
    models: Dict[str, ModelHealthStatus]
