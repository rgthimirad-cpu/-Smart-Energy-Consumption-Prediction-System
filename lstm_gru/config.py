# LSTM/GRU Shared Configuration


# ============================================================
# Time-Series Configuration
# ============================================================

LOOKBACK = 144       # Previous 24 hours (144 x 10 minutes)
HORIZON = 1          # Predict the next 10-minute value


# ============================================================
# Target Columns
# ============================================================

ZONES = {
    "Zone_1": "Zone_1_Power_Consumption",
    "Zone_2": "Zone_2_Power_Consumption",
    "Zone_3": "Zone_3_Power_Consumption",
}


# ============================================================
# Reproducibility
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Evaluation Metrics
# ============================================================

METRICS = [
    "RMSE",
    "MAE",
    "MAPE",
    "R2"
]


# ============================================================
# Common Input Features
# Created by the Data Engineering Team
# ============================================================

COMMON_FEATURES = [
    "Day_of_Week",
    "Diffuse_Flows",
    "General_Diffuse_Flows",
    "Hour",
    "Humidity",
    "Is_Peak_Hour",
    "Is_Weekend",
    "Month",
    "Quarter",
    "Season_Autumn",
    "Season_Spring",
    "Season_Summer",
    "Season_Winter",
    "Temp_Humidity_Index",
    "Temperature",
    "Wind_Speed",
]


# ============================================================
# Historical Features for Each Zone
# Created by the Data Engineering Team
# ============================================================

ZONE_FEATURES = {
    "Zone_1": [
        "Zone_1_lag_1",
        "Zone_1_lag_6",
        "Zone_1_lag_144",
        "Zone_1_roll_max_1h",
        "Zone_1_roll_max_24h",
        "Zone_1_roll_mean_1h",
        "Zone_1_roll_mean_24h",
        "Zone_1_roll_min_1h",
        "Zone_1_roll_min_24h",
        "Zone_1_roll_std_1h",
        "Zone_1_roll_std_24h",
    ],

    "Zone_2": [
        "Zone_2_lag_1",
        "Zone_2_lag_6",
        "Zone_2_lag_144",
        "Zone_2_roll_max_1h",
        "Zone_2_roll_max_24h",
        "Zone_2_roll_mean_1h",
        "Zone_2_roll_mean_24h",
        "Zone_2_roll_min_1h",
        "Zone_2_roll_min_24h",
        "Zone_2_roll_std_1h",
        "Zone_2_roll_std_24h",
    ],

    "Zone_3": [
        "Zone_3_lag_1",
        "Zone_3_lag_6",
        "Zone_3_lag_144",
        "Zone_3_roll_max_1h",
        "Zone_3_roll_max_24h",
        "Zone_3_roll_mean_1h",
        "Zone_3_roll_mean_24h",
        "Zone_3_roll_min_1h",
        "Zone_3_roll_min_24h",
        "Zone_3_roll_std_1h",
        "Zone_3_roll_std_24h",
    ],
}


# ============================================================
# Feature Selection Helper
# ============================================================

def get_zone_features(zone):
    """
    Return common features plus historical features
    belonging to the selected zone.
    """

    return COMMON_FEATURES + ZONE_FEATURES[zone]