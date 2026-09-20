"""
Export all service modules.
"""

from api.services.anomaly_service import (
    detect_anomaly,
    get_anomalies,
    get_anomaly_summary,
)
from api.services.clustering_service import (
    classify_profile,
    get_cluster_distribution,
    get_cluster_groups,
)
from api.services.data_manager import DataManager
from api.services.forecasting_service import (
    get_actual_vs_predicted,
    predict_horizon,
)
from api.services.optimization_service import (
    generate_action_plan,
    get_recommendations,
)
from api.services.peak_demand_service import (
    get_upcoming_peaks,
    predict_peak_demand,
)
