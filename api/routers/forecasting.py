"""
FastAPI Router for Time-Series Forecasting endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from api.schemas.forecasting import ForecastRequest, ForecastingResponse
from api.services.forecasting_service import get_actual_vs_predicted, predict_horizon

router = APIRouter(prefix="/api/v1/forecasting", tags=["Time-Series Forecasting"])


@router.get(
    "/actual-vs-predicted",
    response_model=ForecastingResponse,
    summary="Get Historical Actual vs Predicted Energy Series",
    description="Returns time-series comparison of actual measured consumption vs ML model predictions with 95% confidence bounds.",
)
def get_forecasting_series(
    zone: str = Query("Total", description="Zone selection: 'Zone 1', 'Zone 2', 'Zone 3', or 'Total'"),
    limit: int = Query(144, ge=1, le=1008, description="Number of 10-minute intervals to return (144 = 24h)"),
    start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD HH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD HH:MM:SS)"),
):
    valid_zones = ["Zone 1", "Zone 2", "Zone 3", "Total"]
    if zone not in valid_zones:
        raise HTTPException(status_code=400, detail=f"Invalid zone '{zone}'. Must be one of {valid_zones}")
    return get_actual_vs_predicted(zone=zone, limit=limit, start_date=start_date, end_date=end_date)


@router.post(
    "/predict",
    response_model=ForecastingResponse,
    summary="Generate Forward Multi-Step Forecast",
    description="Executes forward time-series prediction over requested horizon given target environmental weather inputs.",
)
def forecast_predict(body: ForecastRequest):
    return predict_horizon(body)
