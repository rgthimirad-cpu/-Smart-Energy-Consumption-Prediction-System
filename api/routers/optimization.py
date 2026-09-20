"""
FastAPI Router for Optimization Recommendations endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Query

from api.schemas.optimization import (
    ActionPlanRequest,
    ActionPlanResponse,
    RecommendationsResponse,
)
from api.services.optimization_service import (
    generate_action_plan,
    get_recommendations,
)

router = APIRouter(prefix="/api/v1/optimization", tags=["Optimization Recommendations"])


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    summary="Get Energy Optimization Recommendations",
    description="Returns categorized energy saving recommendations (HVAC setpoints, load shifting, solar PV, vampire load reduction) with estimated kW & cost savings.",
)
def list_optimization_recommendations(
    cluster_id: Optional[int] = Query(None, ge=0, le=3, description="Filter by target cluster ID (0-3)"),
    min_savings_pct: float = Query(0.0, ge=0.0, le=100.0, description="Filter recommendations above minimum cost saving %"),
):
    return get_recommendations(cluster_id=cluster_id, min_savings_pct=min_savings_pct)


@router.post(
    "/action-plan",
    response_model=ActionPlanResponse,
    summary="Generate Custom Operator Action Plan",
    description="Calculates immediate power reduction target and step-by-step action plan based on current live demand vector.",
)
def create_operator_action_plan(body: ActionPlanRequest):
    return generate_action_plan(body)
