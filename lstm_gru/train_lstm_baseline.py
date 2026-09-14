from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping

from lstm_gru.config import RANDOM_STATE
from lstm_gru.data_harness import prepare_zone_data
from lstm_gru.lstm_model import build_lstm_model


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Output folders
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "lstm_gru" / "models"
RESULT_DIR = ROOT / "lstm_gru" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Baseline Training
# ============================================================

def train_baseline(zone="Zone_1"):

    print(f"\nPreparing data for {zone}...")

    data = prepare_zone_data(zone)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # --------------------------------------------------------
    # Build baseline model
    # --------------------------------------------------------

    model = build_lstm_model(
        input_shape=X_train.shape[1:],
        units=64,
        dropout=0.2,
        learning_rate=0.001
    )

    model.summary()

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / f"{zone}_baseline_lstm.keras"

    model.save(model_path)

    print("\nModel saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history.history)

    history_path = (
        RESULT_DIR /
        f"{zone}_baseline_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print("\nTraining history saved to:")
    print(history_path)


if __name__ == "__main__":
    train_baseline("Zone_1")