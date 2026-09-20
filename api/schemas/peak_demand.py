"""
Pydantic schemas for Peak-Demand Prediction API endpoints.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import MetaHeader


class PeakRiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PeakWindow(BaseModel):
    window_id: str = Field(..., description="Unique window identifier", example="PEAK-WIN-0815-1400")
    start_time: str = Field(..., description="Start timestamp of predicted peak period", example="2026-08-15 14:00:00")
    end_time: str = Field(..., description="End timestamp of predicted peak period", example="2026-08-15 18:30:00")
    duration_minutes: int = Field(..., description="Duration of peak window in minutes", example=270)
    peak_probability: float = Field(..., description="Probability of exceeding peak threshold (0.0 to 1.0)", example=0.94)
    expected_peak_kw: float = Field(..., description="Highest expected total demand during peak window in kW", example=98450.0)
    threshold_kw: float = Field(..., description="Peak demand classification threshold in kW", example=90000.0)
    risk_level: PeakRiskLevel = Field(..., description="Risk rating of peak event", example="CRITICAL")
    contributing_factors: List[str] = Field(..., description="Main drivers of peak demand", example=["High temperature (34°C)", "Zone 1 & 3 simultaneous afternoon heating", "Weekday afternoon commercial load"])


class UpcomingPeaksResponse(BaseModel):
    meta: MetaHeader
    hours_evaluated: int = Field(..., description="Number of future hours analyzed", example=24)
    threshold_kw: float = Field(..., description="Applied peak threshold in kW", example=90000.0)
    peaks_detected_count: int = Field(..., description="Total peak windows detected in lookahead window", example=2)
    highest_probability: float = Field(..., description="Maximum peak probability found", example=0.94)
    upcoming_windows: List[PeakWindow]


class PeakPredictRequest(BaseModel):
    timestamp: Optional[str] = Field("2026-09-20 14:00:00", description="Target prediction timestamp", example="2026-09-20 14:00:00")
    total_power_kw: float = Field(..., description="Current total energy demand (Zone 1 + Zone 2 + Zone 3) in kW", example=88500.0)
    lag_10m_kw: Optional[float] = Field(None, description="Previous 10-minute demand in kW", example=87200.0)
    lag_1h_kw: Optional[float] = Field(None, description="Demand 1 hour ago in kW", example=82100.0)
    rolling_24h_mean_kw: Optional[float] = Field(None, description="24-hour rolling mean in kW", example=71000.0)
    temperature_c: float = Field(28.0, description="Current outdoor temperature (°C)", example=32.5)
    is_weekend: bool = Field(False, description="Is weekend flag", example=False)


class PeakPredictResponse(BaseModel):
    meta: MetaHeader
    timestamp: str = Field(..., description="Evaluated timestamp", example="2026-09-20 14:00:00")
    predicted_total_kw: float = Field(..., description="Predicted total demand in next 10 minutes (kW)", example=91250.0)
    peak_probability: float = Field(..., description="Probability of exceeding peak threshold", example=0.88)
    is_peak_warning: bool = Field(..., description="Binary peak alert flag", example=True)
    threshold_kw: float = Field(..., description="System peak threshold in kW", example=90000.0)
    risk_level: PeakRiskLevel = Field(..., description="Categorical risk assessment", example="HIGH")
    advisory_message: str = Field(..., description="Operator advisory notice", example="Peak demand expected to cross 90,000 kW threshold in next interval. Prepare load-shedding protocol.")
