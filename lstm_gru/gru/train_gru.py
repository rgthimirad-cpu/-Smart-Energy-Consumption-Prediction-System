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


def set_seeds():
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def search_architectures(zone, data, architectures, max_epochs):
    """
    Train every candidate architecture on the training split and
    evaluate each on the validation split.

    Returns
    -------
    (pandas.DataFrame, dict)
        A results table (one row per architecture) and a dict of
        {architecture_name: best_epoch} for the winning epoch count
        each candidate reached under early stopping.
    """

    target_column = data["target"]
    input_shape = data["X_train"].shape[1:]

    records = []
    best_epochs = {}

    for architecture in architectures:
        print(f"\n[{zone}] Training candidate: {architecture['name']}")

        set_seeds()
        model = build_gru_model(architecture, input_shape)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )

        history = model.fit(
            data["X_train"],
            data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=max_epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
            verbose=2
        )

        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        val_pred_scaled = model.predict(data["X_val"], verbose=0).reshape(-1)
        val_pred = inverse_transform_target(val_pred_scaled, target_column)
        val_true = inverse_transform_target(data["y_val"], target_column)

        metrics = calculate_metrics(val_true, val_pred)
        latency_ms = measure_single_sample_latency(
            model, data["X_val"], repeats=TIMING_REPEATS
        )

        records.append(
            {
                "Model": architecture["name"],
                "Zone": target_column,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "R2": metrics["R2"],
                "Single_Sample_Latency_ms": latency_ms,
                "Best_Epoch": best_epoch,
                "Parameters": json.dumps(architecture)
            }
        )

        best_epochs[architecture["name"]] = best_epoch

        # Free memory before the next candidate.
        keras.backend.clear_session()

    return pd.DataFrame(records), best_epochs


def select_best_architecture(results_df):
    """
    Pick the architecture with the lowest validation RMSE among those
    whose single-sample latency is within LATENCY_TOLERANCE times the
    fastest candidate's latency.

    This keeps accuracy as the primary objective while ruling out
    disproportionately slow candidates first - i.e. "maximise
    accuracy while keeping response time low", made explicit and
    auditable rather than left as an implicit trade-off.
    """

    fastest_latency = results_df["Single_Sample_Latency_ms"].min()
    eligible = results_df[
        results_df["Single_Sample_Latency_ms"] <= fastest_latency *
        LATENCY_TOLERANCE
    ]
    best_row = eligible.sort_values("RMSE").iloc[0]
    return best_row


def refit_on_train_and_validation(data, architecture, best_epoch):
    """
    Retrain the selected architecture on train + validation combined,
    for the epoch count found during architecture search. No
    validation split is held out here, so there is no early stopping
    - the epoch count itself is the tuned hyperparameter carried over.
    """

    train_df, validation_df, _ = load_datasets()
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)

    X_combined, y_combined = create_sequences(
        combined_df, data["features"], data["target"], LOOKBACK, HORIZON
    )

    set_seeds()
    model = build_gru_model(architecture, X_combined.shape[1:])
    model.fit(
        X_combined,
        y_combined,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        verbose=2
    )

    return model


def evaluate_on_test(model, data, architecture_name):
    """
    Final, one-time evaluation on the held-out test split: accuracy
    metrics in original Watts, plus response-time metrics for
    inference over the entire test set.
    """

    target_column = data["target"]

    timing = measure_full_set_inference(
        model, data["X_test"], batch_size=BATCH_SIZE)
    test_pred = inverse_transform_target(timing["predictions"], target_column)
    test_true = inverse_transform_target(data["y_test"], target_column)

    metrics = calculate_metrics(test_true, test_pred)
    single_latency_ms = measure_single_sample_latency(
        model, data["X_test"], repeats=TIMING_REPEATS
    )

    result_row = {
        "Model": "GRU",
        "Zone": target_column,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "Test_Set_Total_Inference_Time_Sec": timing["total_seconds"],
        "Avg_Inference_Time_Per_Sample_ms": timing["avg_ms_per_sample"],
        "Throughput_Samples_Per_Sec": timing["throughput_samples_per_sec"],
        "Single_Sample_Latency_ms": single_latency_ms,
        "Test_Set_Size": len(test_true),
        "Notes": (
            f"best architecture={architecture_name}; "
            "predictions inverse-transformed to Watts; "
            "refit on train+validation before test evaluation"
        )
    }

    return result_row, test_true, test_pred
