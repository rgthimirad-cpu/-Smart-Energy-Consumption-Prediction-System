"""
Script to generate sample request & response JSON files for each model endpoint.
Provides static reference fixtures for the frontend dashboard team.
"""

import json
from pathlib import Path
from api.services.forecasting_service import get_actual_vs_predicted, predict_horizon
from api.services.anomaly_service import get_anomalies, detect_anomaly
from api.services.peak_demand_service import get_upcoming_peaks, predict_peak_demand
from api.services.clustering_service import get_cluster_groups, classify_profile
from api.services.optimization_service import get_recommendations, generate_action_plan
from api.schemas import (
    ForecastRequest,
    AnomalyDetectRequest,
    PeakPredictRequest,
    ClusterClassifyRequest,
    ActionPlanRequest,
)

SAMPLES_DIR = Path(__file__).resolve().parent


def generate_all_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting sample JSON fixtures to {SAMPLES_DIR}...")

    # 1. Forecasting
    f_req = ForecastRequest(zone="Total", horizon_steps=6, temperature_c=28.5)
    f_res = predict_horizon(f_req)
    f_actual_res = get_actual_vs_predicted(zone="Total", limit=6)
    with open(SAMPLES_DIR / "forecasting_sample.json", "w") as f:
        json.dump({
            "sample_request": f_req.model_dump(),
            "sample_response_forecast_predict": f_res.model_dump(mode="json"),
            "sample_response_actual_vs_predicted": f_actual_res.model_dump(mode="json"),
        }, f, indent=2)

    # 2. Anomalies
    a_req = AnomalyDetectRequest(timestamp="2026-09-20 14:30:00", zone="Zone 1", value_kw=58900.0, rolling_24h_mean_kw=32000.0)
    a_res = detect_anomaly(a_req)
    a_list_res = get_anomalies(limit=3)
    with open(SAMPLES_DIR / "anomalies_sample.json", "w") as f:
        json.dump({
            "sample_request": a_req.model_dump(),
            "sample_response_detect": a_res.model_dump(mode="json"),
            "sample_response_list": a_list_res.model_dump(mode="json"),
        }, f, indent=2)

    # 3. Peak Demand
    p_req = PeakPredictRequest(timestamp="2026-09-20 14:00:00", total_power_kw=88500.0, temperature_c=32.5)
    p_res = predict_peak_demand(p_req)
    p_upcoming = get_upcoming_peaks(hours_ahead=24)
    with open(SAMPLES_DIR / "peak_demand_sample.json", "w") as f:
        json.dump({
            "sample_request": p_req.model_dump(),
            "sample_response_predict": p_res.model_dump(mode="json"),
            "sample_response_upcoming": p_upcoming.model_dump(mode="json"),
        }, f, indent=2)

    # 4. Clustering
    c_req = ClusterClassifyRequest(zone_1_kw=37000.0, zone_2_kw=25000.0, zone_3_kw=27000.0, temperature_c=27.5, is_peak_hour=True)
    c_res = classify_profile(c_req)
    c_groups = get_cluster_groups()
    with open(SAMPLES_DIR / "clustering_sample.json", "w") as f:
        json.dump({
            "sample_request": c_req.model_dump(),
            "sample_response_classify": c_res.model_dump(mode="json"),
            "sample_response_groups": c_groups.model_dump(mode="json"),
        }, f, indent=2)

    # 5. Optimization
    o_req = ActionPlanRequest(current_zone_1_kw=38000.0, current_zone_2_kw=26000.0, current_zone_3_kw=28000.0, ambient_temp_c=29.0)
    o_res = generate_action_plan(o_req)
    o_recs = get_recommendations()
    with open(SAMPLES_DIR / "optimization_sample.json", "w") as f:
        json.dump({
            "sample_request": o_req.model_dump(),
            "sample_response_action_plan": o_res.model_dump(mode="json"),
            "sample_response_recommendations": o_recs.model_dump(mode="json"),
        }, f, indent=2)

    print("All sample JSON fixtures exported successfully.")

if __name__ == "__main__":
    generate_all_samples()
