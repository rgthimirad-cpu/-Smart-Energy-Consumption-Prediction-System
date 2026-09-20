"""
FastAPI Router for Data Ingestion & Live Reading Stream endpoints.
"""

from fastapi import APIRouter, HTTPException

from api.schemas.system import ReadingIngestRequest, ReadingIngestResponse
from api.services.data_manager import DataManager

router = APIRouter(prefix="/api/v1/ingest", tags=["Data Ingestion & Flow Management"])


@router.post(
    "/readings",
    response_model=ReadingIngestResponse,
    summary="Ingest Smart Meter Readings (Data-Triggered Update)",
    description="Ingests streaming 10-minute smart meter readings, updates rolling time-series buffers, and triggers immediate model state refresh.",
)
def ingest_meter_readings(body: ReadingIngestRequest):
    if not body.readings or len(body.readings) == 0:
        raise HTTPException(status_code=422, detail="Readings payload cannot be empty")

    dm = DataManager.get_instance()
    last_res = None

    for r in body.readings:
        last_res = dm.ingest_reading(r.model_dump())

    t0 = None
    meta = dm.build_meta(start_time_ns=t0)
    return ReadingIngestResponse(
        meta=meta,
        ingested_count=len(body.readings),
        buffer_total_size=last_res["buffer_total_size"],
        model_predictions_refreshed=True,
        latest_reading_timestamp=last_res["latest_reading_timestamp"],
        total_power_kw=last_res["total_power_kw"],
        peak_warning=last_res["peak_warning"],
    )
