"""
Pydantic schemas for Anomaly Detection API endpoints.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import MetaHeader


class AnomalySeverity(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AnomalyEvent(BaseModel):
    event_id: str = Field(..., description="Unique identifier for the anomaly event", example="ANO-20260920-0012")
    timestamp: str = Field(..., description="Timestamp of flagged unusual reading", example="2026-08-15 14:30:00")
    zone: str = Field(..., description="Affected zone", example="Zone 1")
    observed_value_kw: float = Field(..., description="Actual measured power consumption in kW", example=58900.0)
    expected_baseline_kw: float = Field(..., description="Expected baseline normal consumption in kW", example=34200.0)
    deviation_kw: float = Field(..., description="Absolute power deviation from normal baseline in kW", example=24700.0)
    severity: AnomalySeverity = Field(..., description="Severity level rating", example="CRITICAL")
    isolation_forest_score: float = Field(..., description="Isolation Forest anomaly score (-1.0 to 1.0)", example=-0.285)
    zscore_24h: float = Field(..., description="24-hour rolling window Z-score", example=4.12)
    mad_zscore: float = Field(..., description="Median Absolute Deviation (MAD) robust Z-score", example=5.84)
    signals_agreeing: int = Field(..., description="Number of anomaly detection algorithms agreeing on flag", example=3)
    explanation: str = Field(..., description="Human-readable root cause explanation", example="Sudden 72% spike in Zone 1 during peak ambient temperature (34.5°C)")


class AnomalyListResponse(BaseModel):
    meta: MetaHeader
    total_anomalies: int = Field(..., description="Total count of anomalies matching filter criteria", example=42)
    critical_count: int = Field(..., description="Count of critical severity events", example=5)
    high_count: int = Field(..., description="Count of high severity events", example=12)
    medium_count: int = Field(..., description="Count of medium severity events", example=15)
    low_count: int = Field(..., description="Count of low severity events", example=10)
    events: List[AnomalyEvent]


class AnomalyDetectRequest(BaseModel):
    timestamp: str = Field(default="2026-09-20 12:00:00", description="Timestamp of incoming reading", example="2026-09-20 12:00:00")
    zone: str = Field(..., description="Zone identifier: 'Zone 1', 'Zone 2', or 'Zone 3'", example="Zone 1")
    value_kw: float = Field(..., description="Energy reading in kW", example=54000.0)
    rolling_24h_mean_kw: Optional[float] = Field(None, description="24h rolling average kw baseline", example=32000.0)
    rolling_24h_std_kw: Optional[float] = Field(None, description="24h rolling standard deviation", example=4500.0)
    temperature_c: Optional[float] = Field(None, description="Ambient temperature (°C)", example=31.2)


class AnomalyDetectResponse(BaseModel):
    meta: MetaHeader
    is_anomaly: bool = Field(..., description="Boolean flag indicating if the reading is anomalous", example=True)
    severity: AnomalySeverity = Field(..., description="Evaluated severity category", example="HIGH")
    isolation_forest_flag: bool = Field(..., description="Isolation Forest algorithm flag", example=True)
    statistical_zscore_flag: bool = Field(..., description="Statistical 3-sigma Z-score flag", example=True)
    mad_zscore_flag: bool = Field(..., description="MAD robust Z-score flag", example=True)
    deviation_vs_baseline_kw: float = Field(..., description="Deviation from expected rolling mean", example=22000.0)
    recommendation: str = Field(..., description="Immediate recommended action for operators", example="Inspect Zone 1 transformer and industrial cooling units for unannounced heavy startup")


class AnomalySummaryResponse(BaseModel):
    meta: MetaHeader
    total_records_analyzed: int = Field(..., description="Total 10-min records evaluated", example=52416)
    total_anomalies_detected: int = Field(..., description="Total anomaly records found across all zones", example=1845)
    anomaly_rate_pct: float = Field(..., description="Percentage of dataset flagged as anomalous", example=3.52)
    zone_breakdown: dict = Field(..., description="Anomaly counts by zone", example={"Zone 1": 712, "Zone 2": 580, "Zone 3": 553})
