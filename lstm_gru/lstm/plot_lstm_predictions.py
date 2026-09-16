from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = ROOT / "lstm_gru" / "lstm" / "results"
PLOT_DIR = ROOT / "lstm_gru" / "lstm" / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_zone_predictions(zone):

    prediction_path = (
        RESULT_DIR /
        f"{zone}_lstm_test_predictions.csv"
    )

    df = pd.read_csv(
        prediction_path,
        parse_dates=["DateTime"]
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["DateTime"],
        df["Actual"],
        label="Actual"
    )

    plt.plot(
        df["DateTime"],
        df["Predicted"],
        label="Predicted"
    )

    plt.title(
        f"{zone} LSTM - Actual vs Predicted Test Consumption"
    )

    plt.xlabel("DateTime")
    plt.ylabel("Power Consumption")

    plt.legend()
    plt.tight_layout()

    output_path = (
        PLOT_DIR /
        f"{zone}_lstm_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("Saved:")
    print(output_path)


def plot_all_zones():

    for zone in [
        "Zone_1",
        "Zone_2",
        "Zone_3"
    ]:
        plot_zone_predictions(zone)


if __name__ == "__main__":
    plot_all_zones()