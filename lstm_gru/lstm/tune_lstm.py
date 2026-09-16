from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping

from lstm_gru.config import RANDOM_STATE
from lstm_gru.data_harness import prepare_zone_data
from lstm_gru.lstm_model import build_lstm_model
from lstm_gru.scaling import inverse_transform_target
from lstm_gru.evaluation import calculate_metrics


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Output paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "lstm_gru" / "lstm" / "models"
RESULT_DIR = ROOT / "lstm_gru" / "lstm" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TUNING_CONFIGS = [
    {
        "name": "run_01",
        "units": (32,),
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "name": "run_02",
        "units": (64,),
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "name": "run_03",
        "units": (128,),
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "name": "run_04",
        "units": (64, 32),
        "dropout": 0.2,
        "learning_rate": 0.001,
    },
    {
        "name": "run_05",
        "units": (64, 32),
        "dropout": 0.3,
        "learning_rate": 0.001,
    },
    {
        "name": "run_06",
        "units": (64, 32),
        "dropout": 0.3,
        "learning_rate": 0.0005,
    },
]
def run_experiment(
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    target_column
):
    """
    Train and evaluate one LSTM configuration.
    """

    print("\n" + "=" * 60)
    print("Starting:", config["name"])
    print("Units:", config["units"])
    print("Dropout:", config["dropout"])
    print("Learning rate:", config["learning_rate"])
    print("=" * 60)

    # Clear previous TensorFlow model from memory.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # Build model using this configuration.
    model = build_lstm_model(
        input_shape=X_train.shape[1:],
        units=config["units"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1
    )

    training_time = time.time() - start_time

    # Find best validation epoch.
    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    best_val_loss = float(
        np.min(history.history["val_loss"])
    )

    # Predict validation set.
    y_pred_scaled = model.predict(
        X_val,
        verbose=0
    ).reshape(-1)

    # Convert actual and predicted values
    # back to original electricity units.
    y_val_original = inverse_transform_target(
        y_val,
        target_column
    )

    y_pred_original = inverse_transform_target(
        y_pred_scaled,
        target_column
    )

    # Calculate required metrics.
    metrics = calculate_metrics(
        y_val_original,
        y_pred_original
    )

    result = {
        "Run": config["name"],
        "Layers": len(config["units"]),
        "Units": "-".join(
            str(unit) for unit in config["units"]
        ),
        "Dropout": config["dropout"],
        "Learning_Rate": config["learning_rate"],
        "Batch_Size": 64,
        "Best_Epoch": best_epoch,
        "Best_Val_Loss": best_val_loss,
        "Training_Time_Seconds": training_time,
        **metrics
    }

    print("\nResult:")
    print(result)

    return model, result
def tune_lstm(zone="Zone_1"):

    print(f"\nPreparing tuning data for {zone}...")

    data = prepare_zone_data(zone)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    target_column = data["target"]

    results_path = (
        RESULT_DIR /
        f"{zone}_lstm_tuning_results.csv"
    )

    best_model_path = (
        MODEL_DIR /
        f"{zone}_best_lstm.keras"
    )

    # --------------------------------------------------------
    # Resume previous tuning if results already exist
    # --------------------------------------------------------

    if results_path.exists():

        existing_df = pd.read_csv(results_path)

        results = existing_df.to_dict("records")

        completed_runs = set(
            existing_df["Run"].tolist()
        )

        best_rmse = float(
            existing_df["RMSE"].min()
        )

        print("\nExisting tuning results found.")
        print("Completed runs:", completed_runs)

    else:

        results = []
        completed_runs = set()
        best_rmse = float("inf")

    # --------------------------------------------------------
    # Run unfinished experiments
    # --------------------------------------------------------

    for config in TUNING_CONFIGS:

        if config["name"] in completed_runs:

            print(
                f"\nSkipping {config['name']} "
                "(already completed)"
            )

            continue

        model, result = run_experiment(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            target_column
        )

        results.append(result)

        # Save results immediately after every experiment.
        results_df = pd.DataFrame(results)

        results_df.to_csv(
            results_path,
            index=False
        )

        print(
            f"\nCheckpoint saved after "
            f"{config['name']}"
        )

        # Save best model immediately.
        if result["RMSE"] < best_rmse:

            best_rmse = result["RMSE"]

            model.save(
                best_model_path
            )

            print(
                "New best model saved."
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["RMSE"].idxmin()
    ]

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)

    print("\nBest configuration:")
    print(best_row.to_dict())

    print("\nTuning results saved to:")
    print(results_path)

    print("\nBest model saved to:")
    print(best_model_path)


if __name__ == "__main__":
    tune_lstm("Zone_1")