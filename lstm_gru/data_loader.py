# Data Loading and Basic Verification

from pathlib import Path
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"


def load_datasets():
    """
    Load the prepared train, validation and test datasets.
    """

    train_path = DATA_DIR / "train.csv"
    validation_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_path, parse_dates=["DateTime"])
    validation_df = pd.read_csv(
        validation_path,
        parse_dates=["DateTime"]
    )
    test_df = pd.read_csv(test_path, parse_dates=["DateTime"])

    return train_df, validation_df, test_df


def verify_datasets(train_df, validation_df, test_df):
    """
    Check dataset sizes, columns and DateTime ordering.
    """

    print("Dataset shapes:")
    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Test:", test_df.shape)

    print("\nDateTime ranges:")
    print("Train:", train_df["DateTime"].min(), "to", train_df["DateTime"].max())
    print(
        "Validation:",
        validation_df["DateTime"].min(),
        "to",
        validation_df["DateTime"].max()
    )
    print("Test:", test_df["DateTime"].min(), "to", test_df["DateTime"].max())

    print("\nDateTime ordering:")
    print("Train sorted:", train_df["DateTime"].is_monotonic_increasing)
    print("Validation sorted:", validation_df["DateTime"].is_monotonic_increasing)
    print("Test sorted:", test_df["DateTime"].is_monotonic_increasing)

    print("\nMissing values:")
    print("Train:", train_df.isna().sum().sum())
    print("Validation:", validation_df.isna().sum().sum())
    print("Test:", test_df.isna().sum().sum())


if __name__ == "__main__":
    train, validation, test = load_datasets()
    verify_datasets(train, validation, test)