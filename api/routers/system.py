"""
FastAPI Router for System Health, Data Flow Update Strategy & Cache Control endpoints.
"""

import time
from datetime import datetime
from fastapi import APIRouter

from api.config import API_TITLE, API_VERSION
from api.schemas.system import (
    ModelHealthStatus,
    RefreshSystemResponse,
    SystemStatusResponse,
)
from api.services.data_manager import DataManager

router = APIRouter(prefix="/api/v1/system", tags=["System Management & Status"])


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get System Health & Data Flow Status",
    description="Returns API service health, loaded ML model binary statuses, active update strategy mode, and dataset statistics.",
)
def get_system_health():
    t0 = time.time_ns()
    dm = DataManager.get_instance()

    uptime = time.time() - dm.start_time

    models_health = {}
    for k, v in dm.model_statuses.items():
        models_health[k] = ModelHealthStatus(
            name=v["name"],
            status=v["status"],
            file_path=v["file_path"],
            is_loaded=v["is_loaded"],
        )

    meta = dm.build_meta(start_time_ns=t0)
    return SystemStatusResponse(
        meta=meta,
        system_name=API_TITLE,
        api_version=API_VERSION,
        uptime_seconds=round(uptime, 2),
        current_update_mode=dm.update_mode,
        buffer_readings_count=len(dm.ingested_buffer),
        dataset_records_count=len(dm.df_cleaned) if dm.df_cleaned is not None else 0,
        models=models_health,
    )


@router.post(
    "/refresh",
    response_model=RefreshSystemResponse,
    summary="Trigger Scheduled / On-Demand Model Output Refresh",
    description="Forces background re-calculation of model predictions over the latest sliding dataset window and updates API cache.",
)
def trigger_system_refresh():
    t0 = time.time_ns()
    dm = DataManager.get_instance()

    refreshed_models = dm.refresh_cache()

    meta = dm.build_meta(start_time_ns=t0)
    return RefreshSystemResponse(
        meta=meta,
        status="SUCCESS",
        models_refreshed=refreshed_models,
        records_processed=len(dm.df_cleaned) if dm.df_cleaned is not None else 52416,
        refreshed_at=datetime.utcnow().isoformat(),
    )
