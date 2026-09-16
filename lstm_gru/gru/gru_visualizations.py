'''
Create GRU evaluation plots from the saved result artifacts.

Run from the repository root with::

	python -m lstm_gru.gru.gru_visualizations
'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GRU_DIR = Path(__file__).resolve().parent
RESULTS_PATH = GRU_DIR / "docs" / "results" / "results_gru.csv"
PREDICTIONS_PATH = (
    GRU_DIR / "internal_results" / "predictions" / "predictions_gru.csv"
)
PLOTS_DIR = GRU_DIR / "plots"


def load_results():
    """Load and validate the final metrics and test predictions."""
    results = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["DateTime"])

    required_result_columns = {
        "Zone",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Test_Set_Total_Inference_Time_Sec",
        "Avg_Inference_Time_Per_Sample_ms",
        "Throughput_Samples_Per_Sec",
        "Single_Sample_Latency_ms",
    }
    required_prediction_columns = {
        "DateTime",
        "Zone",
        "Actual",
        "Predicted",
    }

    missing_results = required_result_columns - set(results.columns)
    missing_predictions = required_prediction_columns - \
        set(predictions.columns)
    if missing_results or missing_predictions:
        raise ValueError(
            "Missing required columns: "
            f"results={sorted(missing_results)}, "
            f"predictions={sorted(missing_predictions)}"
        )

    return results, predictions


def plot_actual_vs_predicted(predictions):
    """Save one time-series actual-vs-predicted plot for each zone."""
    for zone, zone_predictions in predictions.groupby("Zone", sort=True):
        zone_predictions = zone_predictions.sort_values("DateTime")
        figure, axis = plt.subplots(figsize=(13, 5))
        axis.plot(
            zone_predictions["DateTime"],
            zone_predictions["Actual"],
            label="Actual",
            color="#173f5f",
            linewidth=1.2,
        )
        axis.plot(
            zone_predictions["DateTime"],
            zone_predictions["Predicted"],
            label="Predicted",
            color="#ed553b",
            linewidth=1.0,
            alpha=0.9,
        )
        axis.set_title(f"GRU Actual vs Predicted Power Consumption - {zone}")
        axis.set_xlabel("Date and time")
        axis.set_ylabel("Power consumption (Watts)")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(
            PLOTS_DIR / f"actual_vs_predicted_{zone.lower()}.png", dpi=180)
        plt.close(figure)


def plot_metrics_comparison(results):
    """Save native-scale accuracy and timing metrics in one figure."""
    metrics = [
        ("RMSE", "RMSE", "{:.2f}"),
        ("MAE", "MAE", "{:.2f}"),
        ("MAPE", "MAPE (%)", "{:.2f}"),
        (
            "Test_Set_Total_Inference_Time_Sec",
            "Total inference time (sec)",
            "{:.3f}",
        ),
        (
            "Avg_Inference_Time_Per_Sample_ms",
            "Average inference time (ms)",
            "{:.3f}",
        ),
        (
            "Throughput_Samples_Per_Sec",
            "Throughput (samples/sec)",
            "{:.2f}",
        ),
        (
            "Single_Sample_Latency_ms",
            "Single-sample latency (ms)",
            "{:.3f}",
        ),
    ]

    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    colors = ["#173f5f", "#20639b", "#3caea3"]

    for axis, (column, title, value_format) in zip(axes, metrics):
        bars = axis.bar(results["Zone"], results[column], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(
            bars,
            labels=[value_format.format(value) for value in results[column]],
            padding=3,
            fontsize=8,
        )

    axes[-1].axis("off")
    figure.suptitle(
        "GRU Accuracy and Inference-Time Metrics by Zone", fontsize=15)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "metrics_comparison_by_zone.png", dpi=180)
    plt.close(figure)


def plot_r2_comparison(results):
    """Save R2 values for all zones in their own chart."""
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(results["Zone"], results["R2"], color="#3caea3", width=0.6)
    axis.set_title("GRU R2 Score Comparison by Zone")
    axis.set_xlabel("Zone")
    axis.set_ylabel("R2 score")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.4f", padding=3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "r2_comparison_by_zone.png", dpi=180)
    plt.close(figure)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results, predictions = load_results()
    plot_actual_vs_predicted(predictions)
    plot_metrics_comparison(results)
    plot_r2_comparison(results)
    print(f"Saved GRU plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
