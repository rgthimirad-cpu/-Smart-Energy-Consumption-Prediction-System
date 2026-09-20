"""
Optimization Recommendation Service
Provides categorized actionable energy saving recommendations and real-time custom action plans.
"""

import time
from typing import List, Optional

from api.schemas.common import DataSourceType
from api.schemas.optimization import (
    ActionPlanRequest,
    ActionPlanResponse,
    RecCategory,
    RecPriority,
    RecommendationItem,
    RecommendationsResponse,
)
from api.services.data_manager import DataManager

BASE_RECOMMENDATIONS = [
    RecommendationItem(
        recommendation_id="REC-HVAC-001",
        target_cluster_id=0,
        target_cluster_name="High-Temperature Seasonal Heavy Consumers",
        title="HVAC Thermal Pre-Cooling & Dynamic Setpoint Control",
        category=RecCategory.HVAC_OPTIMIZATION,
        priority=RecPriority.CRITICAL,
        action_summary="Optimize HVAC setpoints, pre-cool facilities before afternoon heat, and insulate Zone 3 thermal loads.",
        estimated_power_saving_kw=5200.0,
        estimated_cost_saving_pct=18.5,
        implementation_steps=[
            "Raise cooling setpoints by 1.5°C during peak afternoon window (12:00-17:00)",
            "Pre-cool building envelope during early morning off-peak tariff hours (04:00-08:00)",
            "Inspect Zone 3 thermal insulation and seals on commercial chillers",
        ],
    ),
    RecommendationItem(
        recommendation_id="REC-LOAD-002",
        target_cluster_id=1,
        target_cluster_name="Evening Peak Heavy Consumers",
        title="Residential & Commercial Load Shifting Strategy",
        category=RecCategory.LOAD_SHIFTING,
        priority=RecPriority.HIGH,
        action_summary="Shift heavy appliances, dishwashers, and EV charging away from peak evening hours (6 PM - 10 PM) to off-peak night hours.",
        estimated_power_saving_kw=3800.0,
        estimated_cost_saving_pct=14.2,
        implementation_steps=[
            "Schedule EV charging and heavy equipment for 23:00-06:00 window",
            "Implement automated demand-response signaling for residential smart thermostats",
            "Stagger commercial cooking equipment startup sequences",
        ],
    ),
    RecommendationItem(
        recommendation_id="REC-SOLAR-003",
        target_cluster_id=2,
        target_cluster_name="Daytime High Active Load",
        title="Daytime Solar PV Array & Daylight Harvesting Integration",
        category=RecCategory.SOLAR_INTEGRATION,
        priority=RecPriority.HIGH,
        action_summary="Deploy rooftop solar PV arrays and daylight harvesting during active daytime operating hours.",
        estimated_power_saving_kw=4500.0,
        estimated_cost_saving_pct=16.0,
        implementation_steps=[
            "Install rooftop solar PV arrays matched to Zone 1 & 2 daytime demand profile",
            "Integrate automated daylight dimming controls for commercial lighting",
            "Align high-energy industrial batch processes with solar production peak (11:00-14:00)",
        ],
    ),
    RecommendationItem(
        recommendation_id="REC-VAMP-004",
        target_cluster_id=3,
        target_cluster_name="Off-Peak Baseline Consumers",
        title="Nighttime Standby & Vampire Load Elimination",
        category=RecCategory.VAMPIRE_LOAD_REDUCTION,
        priority=RecPriority.MEDIUM,
        action_summary="Audit nighttime standby vampire loads and maintain efficient off-peak baseline power usage.",
        estimated_power_saving_kw=1900.0,
        estimated_cost_saving_pct=8.5,
        implementation_steps=[
            "Install smart power strips to auto-cutoff non-essential office peripherals overnight",
            "Audit Zone 1 transformer idle losses and standby lighting power",
            "Schedule main server room auxiliary cooling cycling during low ambient night hours",
        ],
    ),
]


def get_recommendations(
    cluster_id: Optional[int] = None,
    min_savings_pct: float = 0.0,
) -> RecommendationsResponse:
    t0 = time.time_ns()
    dm = DataManager.get_instance()

    recs = []
    for r in BASE_RECOMMENDATIONS:
        if cluster_id is not None and r.target_cluster_id != cluster_id:
            continue
        if r.estimated_cost_saving_pct < min_savings_pct:
            continue
        recs.append(r)

    total_kw_saving = sum(r.estimated_power_saving_kw for r in recs)
    max_pct_saving = max((r.estimated_cost_saving_pct for r in recs), default=0.0)

    meta = dm.build_meta(source=DataSourceType.CACHED_PRECOMPUTED, start_time_ns=t0)
    return RecommendationsResponse(
        meta=meta,
        total_recommendations=len(recs),
        total_potential_power_saving_kw=round(total_kw_saving, 2),
        max_cost_saving_pct=round(max_pct_saving, 2),
        recommendations=recs,
    )


def generate_action_plan(req: ActionPlanRequest) -> ActionPlanResponse:
    t0 = time.time_ns()
    dm = DataManager.get_instance()

    tot_kw = req.current_zone_1_kw + req.current_zone_2_kw + req.current_zone_3_kw

    # Estimate reduction based on temperature and load shifting potential
    hvac_save = max(0.0, (req.ambient_temp_c - 22.0) * 450.0)
    shift_save = tot_kw * 0.08 if req.allow_load_shifting else 0.0
    tot_saving = hvac_save + shift_save + 1500.0

    target_kw = max(20000.0, tot_kw - tot_saving)
    bill_pct = round((tot_saving / max(tot_kw, 1.0)) * 100.0, 1)

    steps = [
        f"Raise HVAC thermostat setpoints by 1.5°C immediately to counter high ambient ({req.ambient_temp_c}°C) thermal load.",
        "De-energize non-essential office lighting and auxiliary pump circuits across Zone 1.",
    ]
    if req.allow_load_shifting:
        steps.append("De-prioritize EV charging stations until off-peak night window (23:00).")
    steps.append("Verify Zone 3 baseline transformer efficiency.")

    meta = dm.build_meta(source=DataSourceType.LIVE_MODEL_INFERENCE, start_time_ns=t0)
    return ActionPlanResponse(
        meta=meta,
        current_total_kw=round(tot_kw, 2),
        optimized_target_kw=round(target_kw, 2),
        projected_power_reduction_kw=round(tot_saving, 2),
        projected_bill_reduction_pct=bill_pct,
        immediate_actions=steps,
        category_breakdown=BASE_RECOMMENDATIONS,
    )
