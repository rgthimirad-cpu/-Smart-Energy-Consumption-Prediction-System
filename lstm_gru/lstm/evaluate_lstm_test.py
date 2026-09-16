from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model

from lstm_gru.config import ZONES
from lstm_gru.data_harness import prepare_zone_data
from lstm_gru.data_loader import load_datasets
from lstm_gru.scaling import inverse_transform_target
from lstm_gru.evaluation import calculate_metrics


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "lstm_gru" / "lstm" / "models"
RESULT_DIR = ROOT / "lstm_gru" / "lstm" / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Evaluate one zone
# ============================================================

def evaluate_zone(zone):

    print("\n" + "=" * 60)
    print(f"FINAL TEST EVALUATION: {zone}")
    print("=" * 60)

    # --------------------------------------------------------
    # Prepare shared test sequences
    # --------------------------------------------------------

    data = prepare_zone_data(zone)

    X_test = data["X_test"]
    y_test = data["y_test"]

    target_column = data["target"]

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # --------------------------------------------------------
    # Load the best tuned LSTM model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        f"{zone}_best_lstm.keras"
    )

    model = load_model(model_path)

    # --------------------------------------------------------
    # Predict test set
    # --------------------------------------------------------

    y_pred_scaled = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)

    # --------------------------------------------------------
    # Convert back to original electricity units
    # --------------------------------------------------------

    y_test_original = inverse_transform_target(
        y_test,
        target_column
    )

    y_pred_original = inverse_transform_target(
        y_pred_scaled,
        target_column
    )

    # --------------------------------------------------------
    # Calculate final test metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test_original,
        y_pred_original
    )

    print("\nFinal Test Metrics")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metric_result = {
        "Zone": zone,
        "Model": "LSTM",
        **metrics
    }

    metrics_df = pd.DataFrame(
        [metric_result]
    )

    metrics_path = (
        RESULT_DIR /
        f"{zone}_lstm_test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save actual vs predicted values
    # --------------------------------------------------------

    _, _, test_df = load_datasets()

    if len(test_df) != len(y_test_original):
        raise ValueError(
            "Test timestamp count does not match "
            "the number of predictions."
        )

    predictions_df = pd.DataFrame({
        "DateTime": test_df["DateTime"],
        "Actual": y_test_original,
        "Predicted": y_pred_original
    })

    predictions_path = (
        RESULT_DIR /
        f"{zone}_lstm_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("\nMetrics saved to:")
    print(metrics_path)

    print("\nPredictions saved to:")
    print(predictions_path)

    return metric_result


# ============================================================
# Evaluate all three zones
# ============================================================

def evaluate_all_zones():

    all_results = []

    for zone in ZONES:

        result = evaluate_zone(zone)

        all_results.append(result)

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_results
    )

    summary_path = (
        RESULT_DIR /
        "lstm_test_metrics_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("ALL LSTM TEST EVALUATIONS COMPLETE")
    print("=" * 60)

    print("\n", summary_df.to_string(index=False))

    print("\nCombined summary saved to:")
    print(summary_path)


if __name__ == "__main__":
    evaluate_all_zones()