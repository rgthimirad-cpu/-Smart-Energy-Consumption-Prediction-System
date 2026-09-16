# LSTM/GRU Preprocessing Utilities

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "lstm_gru" / "artifacts"

FEATURE_DATA = DATA_DIR / "feature_engineered_energy_data.csv"
SCALER_PATH = ARTIFACT_DIR / "minmax_scaler.pkl"


def build_and_save_scaler():
    """
    Rebuild the MinMaxScaler using the same training data
    and scaling procedure used by the Data Engineering team.
    """

    print("Loading feature-engineered dataset...")

    df = pd.read_csv(
        FEATURE_DATA,
        parse_dates=["DateTime"]
    )

    print("Full dataset shape:", df.shape)

    # The original preprocessing used the first 70% as training data.
    train_end = 36691

    train_df = df.iloc[:train_end].copy()

    print("Training portion before removing NaN rows:",
          train_df.shape)

    # Find lag and rolling feature columns.
    lag_cols = [
        column for column in train_df.columns
        if "_lag_" in column
    ]

    rolling_cols = [
        column for column in train_df.columns
        if "_roll_" in column
    ]

    # Remove rows containing NaN values caused by
    # lag and rolling calculations.
    train_df = train_df.dropna(
        subset=lag_cols + rolling_cols
    ).reset_index(drop=True)

    print("Training data used for scaler:",
          train_df.shape)

    # DateTime was excluded from scaling.
    scale_columns = [
        column for column in train_df.columns
        if column != "DateTime"
    ]

    print("Number of columns being scaled:",
          len(scale_columns))

    # Fit scaler ONLY on training data.
    scaler = MinMaxScaler()

    scaler.fit(train_df[scale_columns])

    # Save scaler and column order together.
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "scaler": scaler,
            "columns": scale_columns
        },
        SCALER_PATH
    )

    print("\nScaler successfully saved!")
    print("Location:", SCALER_PATH)

    return scaler, scale_columns


if __name__ == "__main__":
    build_and_save_scaler()