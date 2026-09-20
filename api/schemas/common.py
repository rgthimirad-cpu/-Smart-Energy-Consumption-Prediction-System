"""
Common response metadata and baseline schema structures.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DataUpdateMode(str, Enum):
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"
    DATA_TRIGGERED = "data_triggered"


class DataSourceType(str, Enum):
    LIVE_MODEL_INFERENCE = "live_model_inference"
    CACHED_PRECOMPUTED = "cached_precomputed"
    STREAMING_BUFFER = "streaming_buffer"


class MetaHeader(BaseModel):
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO 8601 timestamp of API response generation",
        example="2026-09-20T21:47:00Z",
    )
    update_mode: DataUpdateMode = Field(
        default=DataUpdateMode.ON_DEMAND,
        description="Strategy used to compute/update this response",
        example="on_demand",
    )
    data_source: DataSourceType = Field(
        default=DataSourceType.LIVE_MODEL_INFERENCE,
        description="Origin of the served data",
        example="live_model_inference",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version string of the active ML model",
        example="1.0.0",
    )
    execution_time_ms: Optional[float] = Field(
        default=0.0,
        description="Processing latency in milliseconds",
        example=14.2,
    )
