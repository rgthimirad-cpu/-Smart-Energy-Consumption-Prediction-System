"""
FastAPI Router for Anomaly Detection endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from api.schemas.anomalies import (
    AnomalyDetectRequest,
    AnomalyDetectResponse,
    AnomalyListResponse,
    AnomalySummaryResponse,
)
from api.services.anomaly_service import (
    detect_anomaly,
    get_anomalies,
    get_anomaly_summary,
)

router = APIRouter(prefix="/api/v1/anomalies", tags=["Anomaly Detection"])


@router.get(
    "",
    response_model=AnomalyListResponse,
    summary="Get Flagged Anomaly Consumption Events",
    description="Query historical flagged anomaly events filtered by zone, severity rating, and time window.",
)
def list_anomalies(
    zone: Optional[str] = Query(None, description="Filter by zone: 'Zone 1', 'Zone 2', 'Zone 3'"),
    severity: Optional[str] = Query(None, description="Filter by severity: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'"),
    start_date: Optional[str] = Query(None, description="Start timestamp filter"),
    end_date: Optional[str] = Query(None, description="End timestamp filter"),
    limit: int = Query(50, ge=1, le=500, description="Maximum anomaly records to return"),
):
    if severity and severity.upper() not in ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        raise HTTPException(status_code=400, detail=f"Invalid severity level '{severity}'")
    return get_anomalies(zone=zone, severity=severity, start_date=start_date, end_date=end_date, limit=limit)


@router.post(
    "/detect",
    response_model=AnomalyDetectResponse,
    summary="Run On-Demand Anomaly Detection",
    description="Evaluates a single smart meter reading in real time using MAD robust Z-score, 3-sigma Z-score, and Isolation Forest score heuristics.",
)
def run_anomaly_detection(body: AnomalyDetectRequest):
    return detect_anomaly(body)


@router.get(
    "/summary",
    response_model=AnomalySummaryResponse,
    summary="Get Anomaly Detection System Summary",
    description="Returns aggregate anomaly detection statistics and zone breakdown across the analyzed dataset.",
)
def get_anomalies_summary_stats():
    return get_anomaly_summary()
