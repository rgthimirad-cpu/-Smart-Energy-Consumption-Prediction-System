"""
FastAPI Router for Peak-Demand Prediction endpoints.
"""

from fastapi import APIRouter, Query

from api.config import PEAK_DEMAND_THRESHOLD_KW
from api.schemas.peak_demand import (
    PeakPredictRequest,
    PeakPredictResponse,
    UpcomingPeaksResponse,
)
from api.services.peak_demand_service import (
    get_upcoming_peaks,
    predict_peak_demand,
)

router = APIRouter(prefix="/api/v1/peak-demand", tags=["Peak-Demand Prediction"])


@router.get(
    "/upcoming-peaks",
    response_model=UpcomingPeaksResponse,
    summary="Get Upcoming Predicted Peak Windows",
    description="Identifies predicted peak energy demand periods and risk probability over the lookahead horizon.",
)
def get_peaks_lookahead(
    hours_ahead: int = Query(24, ge=1, le=168, description="Lookahead window in hours (1-168)"),
    threshold_kw: float = Query(PEAK_DEMAND_THRESHOLD_KW, gt=0.0, description="Custom total kW peak threshold"),
):
    return get_upcoming_peaks(hours_ahead=hours_ahead, threshold_kw=threshold_kw)


@router.post(
    "/predict",
    response_model=PeakPredictResponse,
    summary="Predict Next Interval Peak Probability",
    description="Calculates probability of exceeding total peak threshold in the next interval using current power demand and lag features.",
)
def predict_peak_event(body: PeakPredictRequest):
    return predict_peak_demand(body)
