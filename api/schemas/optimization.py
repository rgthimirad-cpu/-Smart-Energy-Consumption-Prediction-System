"""
Pydantic schemas for Optimization Recommendations API endpoints.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from api.schemas.common import MetaHeader


class RecCategory(str, Enum):
    HVAC_OPTIMIZATION = "HVAC_OPTIMIZATION"
    LOAD_SHIFTING = "LOAD_SHIFTING"
    SOLAR_INTEGRATION = "SOLAR_INTEGRATION"
    VAMPIRE_LOAD_REDUCTION = "VAMPIRE_LOAD_REDUCTION"
    GENERAL_EFFICIENCY = "GENERAL_EFFICIENCY"


class RecPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RecommendationItem(BaseModel):
    recommendation_id: str = Field(..., description="Unique recommendation ID", example="REC-CLUST-001")
    target_cluster_id: int = Field(..., description="Target cluster ID (0 to 3)", example=0)
    target_cluster_name: str = Field(..., description="Target cluster name", example="High-Temperature Seasonal Heavy Consumers")
    title: str = Field(..., description="Short recommendation title", example="HVAC Thermal Pre-Cooling & Setpoint Optimization")
    category: RecCategory = Field(..., description="Optimization category", example="HVAC_OPTIMIZATION")
    priority: RecPriority = Field(..., description="Implementation urgency priority", example="HIGH")
    action_summary: str = Field(..., description="Clear suggested action", example="Optimize HVAC setpoints, pre-cool facilities before afternoon heat, and insulate Zone 3 thermal loads.")
    estimated_power_saving_kw: float = Field(..., description="Estimated reduction in peak power demand (kW)", example=4200.0)
    estimated_cost_saving_pct: float = Field(..., description="Estimated percentage cost savings on energy bill", example=14.5)
    implementation_steps: List[str] = Field(..., description="Concrete execution steps", example=[
        "Raise thermostat setpoints by 1.5°C during 12:00-17:00 peak hours",
        "Run chillers during early morning low-tariff hours (04:00-08:00)",
        "Inspect Zone 3 thermal insulation and duct leakages"
    ])


class RecommendationsResponse(BaseModel):
    meta: MetaHeader
    total_recommendations: int = Field(..., description="Total count of active recommendations", example=4)
    total_potential_power_saving_kw: float = Field(..., description="Cumulative potential peak kW savings", example=12400.0)
    max_cost_saving_pct: float = Field(..., description="Maximum single recommendation cost saving %", example=18.5)
    recommendations: List[RecommendationItem]


class ActionPlanRequest(BaseModel):
    current_zone_1_kw: float = Field(..., description="Current Zone 1 power reading in kW", example=38000.0)
    current_zone_2_kw: float = Field(..., description="Current Zone 2 power reading in kW", example=26000.0)
    current_zone_3_kw: float = Field(..., description="Current Zone 3 power reading in kW", example=28000.0)
    ambient_temp_c: float = Field(..., description="Current ambient temperature in °C", example=29.0)
    allow_load_shifting: bool = Field(True, description="Whether facility allows shifting flexible loads", example=True)


class ActionPlanResponse(BaseModel):
    meta: MetaHeader
    current_total_kw: float = Field(..., description="Current total baseline kW", example=92000.0)
    optimized_target_kw: float = Field(..., description="Projected total kW after executing action plan", example=79500.0)
    projected_power_reduction_kw: float = Field(..., description="Immediate demand reduction in kW", example=12500.0)
    projected_bill_reduction_pct: float = Field(..., description="Projected energy bill reduction (%)", example=13.6)
    immediate_actions: List[str] = Field(..., description="Ordered list of immediate steps for operators", example=[
        "Pre-cool Zone 3 thermal spaces immediately prior to 13:00 temperature peak",
        "Shift non-essential pump operations in Zone 2 to night hours (22:00-06:00)",
        "Audit standby loads across Zone 1 office circuits"
    ])
    category_breakdown: List[RecommendationItem]
