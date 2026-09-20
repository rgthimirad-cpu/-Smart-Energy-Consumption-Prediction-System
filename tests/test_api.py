"""
Pytest Test Suite for Smart Energy Consumption Prediction System API.
Tests all endpoints, schemas, validation rules, data ingestion triggers, and model serving logic.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "documentation" in data
    assert "/docs" in data["documentation"]


# 1. Forecasting Endpoint Tests
def test_forecasting_actual_vs_predicted():
    res = client.get("/api/v1/forecasting/actual-vs-predicted?zone=Zone 1&limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["zone"] == "Zone 1"
    assert data["total_records"] == 10
    assert "summary" in data
    assert len(data["series"]) == 10
    point = data["series"][0]
    assert "actual_kw" in point
    assert "predicted_kw" in point
    assert "lower_bound_kw" in point
    assert "upper_bound_kw" in point


def test_forecasting_invalid_zone():
    res = client.get("/api/v1/forecasting/actual-vs-predicted?zone=InvalidZone")
    assert res.status_code == 400
    assert "Invalid zone" in res.json()["detail"]


def test_forecasting_predict():
    payload = {
        "zone": "Total",
        "horizon_steps": 12,
        "temperature_c": 29.5,
        "humidity_pct": 55.0,
    }
    res = client.post("/api/v1/forecasting/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] == 12
    assert data["summary"]["zone"] == "Total"
    assert len(data["series"]) == 12


# 2. Anomaly Detection Endpoint Tests
def test_anomalies_list():
    res = client.get("/api/v1/anomalies?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "total_anomalies" in data
    assert "events" in data
    assert len(data["events"]) <= 5


def test_anomalies_detect():
    payload = {
        "timestamp": "2026-09-20 14:30:00",
        "zone": "Zone 1",
        "value_kw": 62000.0,
        "rolling_24h_mean_kw": 32000.0,
        "rolling_24h_std_kw": 4500.0,
    }
    res = client.post("/api/v1/anomalies/detect", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_anomaly"] is True
    assert data["severity"] in ["MEDIUM", "HIGH", "CRITICAL"]
    assert "recommendation" in data


def test_anomalies_summary():
    res = client.get("/api/v1/anomalies/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_records_analyzed" in data
    assert "zone_breakdown" in data


# 3. Peak Demand Endpoint Tests
def test_peak_demand_upcoming():
    res = client.get("/api/v1/peak-demand/upcoming-peaks?hours_ahead=24")
    assert res.status_code == 200
    data = res.json()
    assert data["hours_evaluated"] == 24
    assert len(data["upcoming_windows"]) >= 1
    win = data["upcoming_windows"][0]
    assert "peak_probability" in win
    assert "risk_level" in win


def test_peak_demand_predict():
    payload = {
        "timestamp": "2026-09-20 14:00:00",
        "total_power_kw": 92500.0,
        "temperature_c": 33.0,
    }
    res = client.post("/api/v1/peak-demand/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["peak_probability"] >= 0.5
    assert data["is_peak_warning"] is True
    assert data["risk_level"] in ["HIGH", "CRITICAL"]


# 4. Clustering Endpoint Tests
def test_clustering_groups():
    res = client.get("/api/v1/clustering/groups")
    assert res.status_code == 200
    data = res.json()
    assert data["clusters_count"] == 4
    assert len(data["clusters"]) == 4
    cluster0 = data["clusters"][0]
    assert "cluster_name" in cluster0
    assert "ui_color_theme" in cluster0
    assert "statistics" in cluster0


def test_clustering_classify():
    payload = {
        "zone_1_kw": 38000.0,
        "zone_2_kw": 26000.0,
        "zone_3_kw": 28000.0,
        "temperature_c": 28.5,
        "humidity_pct": 58.0,
        "is_peak_hour": True,
    }
    res = client.post("/api/v1/clustering/classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["assigned_cluster_id"] in [0, 1, 2, 3]
    assert "efficiency_rating" in data
    assert "top_recommendation" in data


def test_clustering_distribution():
    res = client.get("/api/v1/clustering/distribution")
    assert res.status_code == 200
    data = res.json()
    assert len(data["distribution"]) == 4


# 5. Optimization Recommendations Endpoint Tests
def test_optimization_recommendations():
    res = client.get("/api/v1/optimization/recommendations")
    assert res.status_code == 200
    data = res.json()
    assert data["total_recommendations"] >= 1
    rec = data["recommendations"][0]
    assert "estimated_power_saving_kw" in rec
    assert "estimated_cost_saving_pct" in rec
    assert "implementation_steps" in rec


def test_optimization_action_plan():
    payload = {
        "current_zone_1_kw": 38000.0,
        "current_zone_2_kw": 26000.0,
        "current_zone_3_kw": 28000.0,
        "ambient_temp_c": 31.0,
        "allow_load_shifting": True,
    }
    res = client.post("/api/v1/optimization/action-plan", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["current_total_kw"] == 92000.0
    assert data["optimized_target_kw"] < 92000.0
    assert len(data["immediate_actions"]) >= 1


# 6. Data Ingestion & Flow Strategy Tests
def test_data_ingestion_stream():
    payload = {
        "readings": [
            {
                "timestamp": "2026-09-20 14:10:00",
                "zone_1_kw": 29000.0,
                "zone_2_kw": 23000.0,
                "zone_3_kw": 24000.0,
                "temperature_c": 27.5,
                "humidity_pct": 59.0,
            }
        ],
        "trigger_model_updates": True,
    }
    res = client.post("/api/v1/ingest/readings", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["ingested_count"] == 1
    assert data["model_predictions_refreshed"] is True
    assert data["meta"]["update_mode"] == "data_triggered"


def test_data_ingestion_empty_payload():
    res = client.post("/api/v1/ingest/readings", json={"readings": []})
    assert res.status_code == 422


# 7. System Management Tests
def test_system_status():
    res = client.get("/api/v1/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["api_version"] == "1.0.0"
    assert "models" in data
    assert "anomaly_detection" in data["models"]
    assert "kmeans_clustering" in data["models"]


def test_system_refresh():
    res = client.post("/api/v1/system/refresh")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "models_refreshed" in data
    assert data["meta"]["update_mode"] == "scheduled"
