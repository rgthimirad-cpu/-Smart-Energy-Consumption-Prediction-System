'''
GRU Training, Tuning, and Evaluation Pipeline

For each zone:
    1. Train every candidate GRU architecture on the training split.
    2. Evaluate each on the validation split - accuracy (original Watts)
    AND single-sample inference latency.
    3. Select the architecture with the lowest validation RMSE among those
    that are not disproportionately slow (see gru_config.LATENCY_TOLERANCE).
    4. Refit the selected architecture on train + validation combined.
    5. Evaluate once on the held-out test split: RMSE/MAE/MAPE/R2 in original Watts,
    plus response-time metrics for the full test set (total inference time, per-sample latency, throughput).
    6. Save the model, the tuning log, the final results row, and the test-set predictions.

Run from the repository root:

    python lstm_gru/gru/train_gru.py
                or
    python -m lstm_gru.gru.train_gru
'''

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from lstm_gru.config import ZONES, RANDOM_STATE, LOOKBACK, HORIZON
from lstm_gru.data_harness import prepare_zone_data
from lstm_gru.data_loader import ROOT, load_datasets
from lstm_gru.windowing import create_sequences
from lstm_gru.evaluation import calculate_metrics
from lstm_gru.scaling import inverse_transform_target
from lstm_gru.gru.gru_model import (
    build_gru_model,
    measure_single_sample_latency,
    measure_full_set_inference
)
from lstm_gru.gru.gru_config import (
    GRU_ARCHITECTURES,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LATENCY_TOLERANCE,
    TIMING_REPEATS
)

# Paths
GRU_DIR = ROOT / "lstm_gru" / "gru"
MODELS_DIR = GRU_DIR / "models"
DOCS_DIR = GRU_DIR / "docs"
RESULTS_DIR = DOCS_DIR / "results"
INTERNAL_DIR = GRU_DIR / "internal_results"
TUNING_DIR = INTERNAL_DIR / "tuning"
PREDICTIONS_DIR = INTERNAL_DIR / "predictions"


def ensure_output_dirs():
    for path in [MODELS_DIR, RESULTS_DIR, TUNING_DIR, PREDICTIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
