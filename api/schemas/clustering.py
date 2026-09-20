"""
Pydantic schemas for Consumption Clustering API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import MetaHeader


class ClusterProfileStatistics(BaseModel):
    total_power_mean_kw: float = Field(..., description="Average total power consumption in kW", example=90529.63)
    zone_1_mean_kw: float = Field(..., description="Average Zone 1 power consumption in kW", example=37747.27)
    zone_2_mean_kw: float = Field(..., description="Average Zone 2 power consumption in kW", example=25486.51)
    zone_3_mean_kw: float = Field(..., description="Average Zone 3 power consumption in kW", example=27295.85)
    temperature_mean_c: float = Field(..., description="Average ambient temperature (°C)", example=26.78)
    humidity_mean_pct: float = Field(..., description="Average humidity (%)", example=59.04)
    peak_hour_percentage: float = Field(..., description="Percentage of cluster instances occurring during peak hours", example=17.25)
    weekend_percentage: float = Field(..., description="Percentage of cluster instances occurring on weekends", example=28.70)


class ClusterProfile(BaseModel):
    cluster_id: int = Field(..., description="Cluster numerical label (0 to k-1)", example=0)
    cluster_name: str = Field(..., description="Human-readable cluster descriptive title", example="High-Temperature Seasonal Heavy Consumers")
    short_code: str = Field(..., description="System short identifier code", example="SUMMER_HVAC_HEAVY")
    efficiency_rating: str = Field(..., description="Efficiency grade rating (Grade A to D)", example="Grade C")
    efficiency_level: str = Field(..., description="Efficiency category description", example="Low (Weather-Sensitive Heavy Load)")
    ui_color_theme: str = Field(..., description="Hex color code for dashboard UI visualization", example="#EF4444")
    usage_behavior: str = Field(..., description="Summary of consumption behavior", example="High total electricity consumption across all zones heavily driven by elevated outdoor temperatures.")
    dominant_time_window: str = Field(..., description="Primary time of day when cluster is active", example="12:00 - 23:00")
    environmental_trigger: str = Field(..., description="Key weather or environmental trigger", example="High Ambient Temperature (> 25°C)")
    sample_count: int = Field(..., description="Total records in dataset belonging to cluster", example=8013)
    sample_percentage: float = Field(..., description="Share of total records (%)", example=15.29)
    statistics: ClusterProfileStatistics
    key_patterns: List[str] = Field(..., description="Observed energy patterns", example=["Strong correlation with temperature", "Zone 3 elevated cooling load"])


class ClusterListResponse(BaseModel):
    meta: MetaHeader
    total_records_analyzed: int = Field(..., description="Total records analyzed by K-Means model", example=52415)
    clusters_count: int = Field(..., description="Number of distinct clusters", example=4)
    clusters: List[ClusterProfile]


class ClusterClassifyRequest(BaseModel):
    zone_1_kw: float = Field(..., description="Zone 1 power consumption in kW", example=35000.0)
    zone_2_kw: float = Field(..., description="Zone 2 power consumption in kW", example=22000.0)
    zone_3_kw: float = Field(..., description="Zone 3 power consumption in kW", example=24000.0)
    temperature_c: float = Field(25.0, description="Ambient temperature in °C", example=27.5)
    humidity_pct: float = Field(60.0, description="Relative humidity %", example=58.0)
    is_peak_hour: bool = Field(False, description="Is peak hour flag", example=True)


class ClusterClassifyResponse(BaseModel):
    meta: MetaHeader
    assigned_cluster_id: int = Field(..., description="Assigned K-Means cluster ID (0-3)", example=0)
    cluster_name: str = Field(..., description="Assigned cluster name", example="High-Temperature Seasonal Heavy Consumers")
    short_code: str = Field(..., description="Assigned cluster code", example="SUMMER_HVAC_HEAVY")
    efficiency_rating: str = Field(..., description="Cluster efficiency grade", example="Grade C")
    ui_color_theme: str = Field(..., description="UI hex color code", example="#EF4444")
    distance_to_centroid: float = Field(..., description="Euclidean distance to cluster centroid in normalized feature space", example=0.342)
    top_recommendation: str = Field(..., description="Immediate top optimization advice for this cluster profile", example="Optimize HVAC setpoints and pre-cool facility before peak afternoon heat.")


class ClusterDistributionItem(BaseModel):
    cluster_id: int
    cluster_name: str
    count: int
    percentage: float
    ui_color_theme: str


class ClusterDistributionResponse(BaseModel):
    meta: MetaHeader
    total_records: int
    distribution: List[ClusterDistributionItem]
