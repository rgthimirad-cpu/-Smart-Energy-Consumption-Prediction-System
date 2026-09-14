# LSTM/GRU Scaling Utilities

from pathlib import Path

import joblib
import numpy as np


# Project paths
ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = ROOT / "lstm_gru" / "artifacts" / "minmax_scaler.pkl"


def load_scaler():
    """
    Load the saved MinMaxScaler and its column order.
    """

    artifact = joblib.load(SCALER_PATH)

    scaler = artifact["scaler"]
    columns = artifact["columns"]

    return scaler, columns


def inverse_transform_target(values, target_column):
    """
    Convert scaled target values back to the original
    electricity-consumption scale.

    Parameters
    ----------
    values : array-like
        Scaled prediction or actual target values.

    target_column : str
        Target column name, for example:
        'Zone_1_Power_Consumption'

    Returns
    -------
    numpy.ndarray
        Values converted back to the original scale.
    """

    scaler, columns = load_scaler()

    if target_column not in columns:
        raise ValueError(
            f"Target column '{target_column}' "
            f"was not found in the scaler columns."
        )

    values = np.asarray(
        values,
        dtype=np.float64
    )

    # Find the position of the target column
    target_index = columns.index(target_column)

    # MinMaxScaler:
    # X_scaled = X * scale_ + min_
    # Therefore:
    # X = (X_scaled - min_) / scale_

    original_values = (
        values - scaler.min_[target_index]
    ) / scaler.scale_[target_index]

    return original_values


if __name__ == "__main__":

    target = "Zone_1_Power_Consumption"

    # Example scaled values
    test_values = np.array([
        0.2,
        0.5,
        0.8
    ])

    original_values = inverse_transform_target(
        test_values,
        target
    )

    print("Target:", target)
    print("Scaled values:", test_values)
    print("Original values:", original_values)