from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model

from lstm_gru.data_harness import prepare_zone_data
from lstm_gru.scaling import inverse_transform_target
from lstm_gru.evaluation import calculate_metrics


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "lstm_gru" / "lstm" / "models"
RESULT_DIR = ROOT / "lstm_gru" / "lstm" / "results"


def evaluate_baseline(zone="Zone_1"):

    print(f"\nEvaluating baseline LSTM for {zone}...")

    data = prepare_zone_data(zone)

    X_val = data["X_val"]
    y_val = data["y_val"]

    target_column = data["target"]

    model_path = MODEL_DIR / f"{zone}_baseline_lstm.keras"

    model = load_model(model_path)

    y_pred_scaled = model.predict(
        X_val,
        verbose=0
    ).reshape(-1)

    y_val_original = inverse_transform_target(
        y_val,
        target_column
    )

    y_pred_original = inverse_transform_target(
        y_pred_scaled,
        target_column
    )

    metrics = calculate_metrics(
        y_val_original,
        y_pred_original
    )

    print("\nValidation Metrics")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    result = {
        "Zone": zone,
        "Model": "Baseline LSTM",
        "Units": 64,
        "Dropout": 0.2,
        "Learning_Rate": 0.001,
        "Batch_Size": 64,
        **metrics
    }

    result_df = pd.DataFrame([result])

    output_path = (
        RESULT_DIR /
        f"{zone}_baseline_validation_metrics.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\nMetrics saved to:")
    print(output_path)


if __name__ == "__main__":
    evaluate_baseline("Zone_1")