from pathlib import Path
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from lstm_gru.data_harness import prepare_zone_data


ROOT = Path(__file__).resolve().parents[2]




def measure_response_time(
    zone="Zone_1",
    warmup_runs=10,
    measurement_runs=100
):
    """
    Measure single-sample inference response time
    for the best trained LSTM model.
    """

    print(f"\nLoading data for {zone}...")

    data = prepare_zone_data(zone)

    # Use validation data only.
    X_val = data["X_val"]

    print("Loading best LSTM model...")

    model_path = (
    ROOT
    / "lstm_gru"
    / "lstm"
    / "models"
    / f"{zone}_best_lstm.keras"
)
    model = load_model(model_path)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------
    # TensorFlow's first few predictions can be slower because
    # some internal setup happens during initial execution.
    # Therefore, warm-up runs are excluded from timing.

    for i in range(warmup_runs):
        sample = X_val[i:i + 1]

        _ = model(
            sample,
            training=False
        ).numpy()

    # --------------------------------------------------------
    # Measure response time
    # --------------------------------------------------------

    response_times = []

    for i in range(measurement_runs):

        sample = X_val[
            warmup_runs + i:
            warmup_runs + i + 1
        ]

        start_time = time.perf_counter()

        _ = model(
            sample,
            training=False
        ).numpy()

        end_time = time.perf_counter()

        elapsed_ms = (
            end_time - start_time
        ) * 1000

        response_times.append(elapsed_ms)

    response_times = np.asarray(response_times)

    average_ms = np.mean(response_times)
    median_ms = np.median(response_times)
    min_ms = np.min(response_times)
    max_ms = np.max(response_times)

    print("\nLSTM Response Time")
    print("-----------------------------")
    print(f"Samples measured: {measurement_runs}")
    print(f"Average: {average_ms:.4f} ms")
    print(f"Median:  {median_ms:.4f} ms")
    print(f"Minimum: {min_ms:.4f} ms")
    print(f"Maximum: {max_ms:.4f} ms")


        # --------------------------------------------------------
    # Save response-time results
    # --------------------------------------------------------

    result = {
        "Zone": zone,
        "Average_Response_Time_ms": average_ms,
        "Median_Response_Time_ms": median_ms,
        "Minimum_Response_Time_ms": min_ms,
        "Maximum_Response_Time_ms": max_ms,
        "Samples_Measured": measurement_runs
    }

    result_df = pd.DataFrame([result])

    result_path = (
        ROOT
        / "lstm_gru"
        / "lstm"
        / "results"
        / f"{zone}_lstm_response_time.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    print("\nResponse-time results saved to:")
    print(result_path)
    
if __name__ == "__main__":
    measure_response_time("Zone_1")