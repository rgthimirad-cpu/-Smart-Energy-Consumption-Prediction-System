"""
Pydantic schemas for Time-Series Forecasting API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import MetaHeader


class ForecastingDataPoint(BaseModel):
    timestamp: str = Field(..., description="Timestamp in ISO 8601 format", example="2026-09-20 12:00:00")
    actual_kw: Optional[float] = Field(None, description="Actual energy consumption in kW", example=45230.5)
    predicted_kw: float = Field(..., description="Predicted energy consumption in kW", example=46100.2)
    residual_kw: Optional[float] = Field(None, description="Difference between actual and predicted (Actual - Predicted)", example=-869.7)
    lower_bound_kw: float = Field(..., description="95% Confidence Interval Lower Bound", example=43795.0)
    upper_bound_kw: float = Field(..., description="95% Confidence Interval Upper Bound", example=48405.4)


class ZoneForecastSummary(BaseModel):
    zone: str = Field(..., description="Zone identifier", example="Zone 1")
    mean_actual_kw: Optional[float] = Field(None, description="Average actual kW consumption over period", example=32150.4)
    mean_predicted_kw: float = Field(..., description="Average predicted kW consumption over period", example=32400.1)
    peak_predicted_kw: float = Field(..., description="Highest predicted single reading kW", example=48200.0)
    mae_kw: Optional[float] = Field(None, description="Mean Absolute Error in kW", example=650.3)
    mape_pct: Optional[float] = Field(None, description="Mean Absolute Percentage Error (%)", example=2.15)


class ForecastingResponse(BaseModel):
    meta: MetaHeader
    zone: str = Field(..., description="Queried zone (Zone 1, Zone 2, Zone 3, or Total)", example="Total")
    total_records: int = Field(..., description="Number of forecasting data points returned", example=144)
    summary: ZoneForecastSummary
    series: List[ForecastingDataPoint]


class ForecastRequest(BaseModel):
    zone: str = Field("Total", description="Target zone: 'Zone 1', 'Zone 2', 'Zone 3', or 'Total'", example="Total")
    horizon_steps: int = Field(144, description="Forward forecast horizon (1 step = 10 minutes; 144 = 24 hours)", ge=1, le=1008, example=144)
    temperature_c: Optional[float] = Field(None, description="Optional projected ambient temperature (°C)", example=28.5)
    humidity_pct: Optional[float] = Field(None, description="Optional projected relative humidity (%)", example=55.0)
    wind_speed_ms: Optional[float] = Field(None, description="Optional projected wind speed (m/s)", example=1.2)
