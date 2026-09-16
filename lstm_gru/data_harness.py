# LSTM/GRU Shared Data Harness

import numpy as np

from lstm_gru.config import (
    LOOKBACK,
    HORIZON,
    ZONES,
    get_zone_features,
)
from lstm_gru.data_loader import load_datasets
from lstm_gru.windowing import create_sequences


def create_split_sequences(
    history_df,
    split_df,
    feature_columns,
    target_column,
    lookback=LOOKBACK,
    horizon=HORIZON,
):
    """
    Create sequences for one data split while preserving
    historical context from the previous split.

    Only targets belonging to split_df are generated.
    The history_df is used only as past input context.
    """

    # Keep only the last LOOKBACK rows from the history.
    history_tail = history_df.tail(lookback)

    # Combine historical context with the current split.
    combined_df = np.concatenate(
        [
            history_tail.index.to_numpy(),
            split_df.index.to_numpy(),
        ]
    )

    # Reindex is handled by concatenating actual dataframe rows.
    combined_df = None

    import pandas as pd

    combined_df = pd.concat(
        [history_tail, split_df],
        ignore_index=True
    )

    # Number of rows belonging to the current split.
    history_length = len(history_tail)

    X_values = combined_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_values = combined_df[target_column].to_numpy(
        dtype=np.float32
    )

    X = []
    y = []

    # Generate one prediction for each valid target in split_df.
    for i in range(
        history_length,
        len(combined_df) - horizon + 1
    ):
        start = i - lookback
        target_index = i + horizon - 1

        X.append(
            X_values[start:i]
        )

        y.append(
            y_values[target_index]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


def prepare_zone_data(zone):
    """
    Prepare train, validation and test sequences
    for one electricity consumption zone.
    """

    if zone not in ZONES:
        raise ValueError(
            f"Unknown zone: {zone}. "
            f"Available zones: {list(ZONES.keys())}"
        )

    train_df, validation_df, test_df = load_datasets()

    target_column = ZONES[zone]
    feature_columns = get_zone_features(zone)

    # Training sequences.
    X_train, y_train = create_sequences(
        train_df,
        feature_columns,
        target_column,
        LOOKBACK,
        HORIZON,
    )

    # Validation sequences use the end of training data
    # as historical context.
    X_val, y_val = create_split_sequences(
        train_df,
        validation_df,
        feature_columns,
        target_column,
        LOOKBACK,
        HORIZON,
    )

    # Test sequences use the end of validation data
    # as historical context.
    X_test, y_test = create_split_sequences(
        validation_df,
        test_df,
        feature_columns,
        target_column,
        LOOKBACK,
        HORIZON,
    )

    return {
        "zone": zone,
        "target": target_column,
        "features": feature_columns,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":

    data = prepare_zone_data("Zone_1")

    print("Zone:", data["zone"])
    print("Target:", data["target"])
    print("Number of features:", len(data["features"]))

    print("\nSequence shapes:")
    print("X_train:", data["X_train"].shape)
    print("y_train:", data["y_train"].shape)
    print("X_val:", data["X_val"].shape)
    print("y_val:", data["y_val"].shape)
    print("X_test:", data["X_test"].shape)
    print("y_test:", data["y_test"].shape)