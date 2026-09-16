"""Compare final LSTM and GRU test-set results.

Run from the repository root with::

	python -m lstm_gru.overall_visualization

Only final test-set metrics are used for model selection. Metrics with
different scales are compared by rank, while the generated plots preserve
their original values.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LSTM_GRU_DIR = Path(__file__).resolve().parent
LSTM_RESULTS_PATH = LSTM_GRU_DIR / "lstm" / "results" / "lstm_final_summary.csv"
GRU_RESULTS_PATH = LSTM_GRU_DIR / "gru" / "results" / "results_gru.csv"
OUTPUT_DIR = LSTM_GRU_DIR / "overall_comparison"
PLOTS_DIR = OUTPUT_DIR / "plots"

LOWER_IS_BETTER = [
    "RMSE",
    "MAE",
    "MAPE",
    "Response_Time_ms",
]
HIGHER_IS_BETTER = ["R2"]


def load_comparison_data():
    """Load final test results and map both model schemas to one contract."""
    lstm = pd.read_csv(LSTM_RESULTS_PATH)
    gru = pd.read_csv(GRU_RESULTS_PATH)

    lstm_required = {
        "Zone",
        "Test_RMSE",
        "Test_MAE",
        "Test_MAPE",
        "Test_R2",
        "Average_Response_Time_ms",
        "Median_Response_Time_ms",
    }
    gru_required = {
        "Zone",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "Single_Sample_Latency_ms",
        "Test_Set_Total_Inference_Time_Sec",
        "Avg_Inference_Time_Per_Sample_ms",
        "Throughput_Samples_Per_Sec",
    }
    missing_lstm = lstm_required - set(lstm.columns)
    missing_gru = gru_required - set(gru.columns)
    if missing_lstm or missing_gru:
        raise ValueError(
            "Missing result columns: "
            f"LSTM={sorted(missing_lstm)}, GRU={sorted(missing_gru)}"
        )

    lstm_comparison = pd.DataFrame(
        {
            "Model": "LSTM",
            "Zone": lstm["Zone"],
            "RMSE": lstm["Test_RMSE"],
            "MAE": lstm["Test_MAE"],
            "MAPE": lstm["Test_MAPE"],
            "R2": lstm["Test_R2"],
            "Response_Time_ms": lstm["Average_Response_Time_ms"],
            "Median_Response_Time_ms": lstm["Median_Response_Time_ms"],
            "Total_Inference_Time_Sec": pd.NA,
            "Avg_Inference_Time_Per_Sample_ms": pd.NA,
            "Throughput_Samples_Per_Sec": pd.NA,
            "Source_Test_Size": pd.NA,
        }
    )
    gru_comparison = pd.DataFrame(
        {
            "Model": "GRU",
            "Zone": gru["Zone"].str.replace(
                "_Power_Consumption", "", regex=False),
            "RMSE": gru["RMSE"],
            "MAE": gru["MAE"],
            "MAPE": gru["MAPE"],
            "R2": gru["R2"],
            "Response_Time_ms": gru["Single_Sample_Latency_ms"],
            "Median_Response_Time_ms": pd.NA,
            "Total_Inference_Time_Sec": gru["Test_Set_Total_Inference_Time_Sec"],
            "Avg_Inference_Time_Per_Sample_ms": gru[
                "Avg_Inference_Time_Per_Sample_ms"
            ],
            "Throughput_Samples_Per_Sec": gru["Throughput_Samples_Per_Sec"],
            "Source_Test_Size": gru["Test_Set_Size"],
        }
    )
    comparison = pd.concat(
        [lstm_comparison, gru_comparison], ignore_index=True)
    comparison["Zone_Number"] = comparison["Zone"].str.extract(
        r"(\d+)").astype(int)
    return comparison.sort_values(["Zone_Number", "Model"]).drop(
        columns="Zone_Number"
    )


def rank_models(comparison):
    """Rank models within each zone and calculate equal-weighted rank scores."""
    ranked = comparison.copy()
    rank_columns = []
    for metric in LOWER_IS_BETTER:
        rank_column = f"{metric}_Rank"
        ranked[rank_column] = ranked.groupby("Zone")[metric].rank(
            ascending=True, method="min"
        )
        rank_columns.append(rank_column)
    for metric in HIGHER_IS_BETTER:
        rank_column = f"{metric}_Rank"
        ranked[rank_column] = ranked.groupby("Zone")[metric].rank(
            ascending=False, method="min"
        )
        rank_columns.append(rank_column)

    ranked["Average_Rank"] = ranked[rank_columns].mean(axis=1)
    ranked["Zone_Selection"] = ranked.groupby("Zone")["Average_Rank"].transform(
        lambda values: values == values.min()
    )
    ranked["Overall_Model_Average_Rank"] = ranked.groupby("Model")[
        "Average_Rank"
    ].transform("mean")
    return ranked


def save_results(comparison, ranked):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "comparison_test_metrics.csv", index=False)
    ranked.to_csv(OUTPUT_DIR / "comparison_rankings.csv", index=False)

    selections = ranked[ranked["Zone_Selection"]].copy()
    model_scores = (
        ranked.groupby("Model", as_index=False)["Average_Rank"]
        .mean()
        .rename(columns={"Average_Rank": "Overall_Average_Rank"})
        .sort_values("Overall_Average_Rank")
    )
    overall_model = model_scores.iloc[0]["Model"]
    overall = pd.DataFrame(
        {
            "Decision": ["Overall best model"],
            "Model": [overall_model],
            "Overall_Average_Rank": [
                model_scores.iloc[0]["Overall_Average_Rank"]
            ],
            "Reason": [
                "Lowest equal-weighted average rank across all zones and "
                "test-set comparison metrics"
            ],
        }
    )
    selections = selections[
        ["Zone", "Model", "Average_Rank", "RMSE",
            "MAE", "MAPE", "R2", "Response_Time_ms"]
    ].rename(columns={"Average_Rank": "Average_Rank_Score"})
    selections.to_csv(OUTPUT_DIR / "best_model_per_zone.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "overall_best_model.csv", index=False)
    model_scores.to_csv(OUTPUT_DIR / "overall_model_scores.csv", index=False)
    return selections, model_scores, overall_model


def annotate_bars(axis, bars, values, formatter):
    axis.bar_label(
        bars,
        labels=[formatter(value) for value in values],
        padding=3,
        fontsize=8,
    )


def plot_metric_group(comparison, metrics, filename, title):
    """Plot original values for one metric-direction group."""
    figure, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    colors = {"LSTM": "#20639b", "GRU": "#ed553b"}
    for axis, (column, label, formatter) in zip(axes, metrics):
        pivot = comparison.pivot(index="Zone", columns="Model", values=column)
        bars = pivot.plot(
            kind="bar", ax=axis, color=[colors[model] for model in pivot.columns]
        )
        axis.set_title(label)
        axis.set_xlabel("Zone")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
        for container, model in zip(axis.containers, pivot.columns):
            annotate_bars(axis, container, pivot[model].values, formatter)
        maximum_value = pivot.to_numpy().max()
        axis.set_ylim(0, maximum_value * 1.18)
        if len(metrics) == 1:
            axis.legend(
                title="Model",
                loc="upper right",
                bbox_to_anchor=(1, -0.12),
                ncol=2,
            )
        else:
            axis.legend(title="Model")
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close(figure)


def plot_overall_ranks(ranked):
    scores = (
        ranked.groupby("Model", as_index=False)["Average_Rank"]
        .mean()
        .sort_values("Average_Rank")
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(scores["Model"], scores["Average_Rank"], color=[
                    "#20639b", "#ed553b"])
    annotate_bars(
        axis, bars, scores["Average_Rank"].values, lambda value: f"{value:.2f}")
    axis.set_title("Overall Test-Set Model Ranking")
    axis.set_ylabel("Average rank score (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "overall_model_ranking.png", dpi=180)
    plt.close(figure)


def generate_plots(comparison, ranked):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric_group(
        comparison,
        [
            ("RMSE", "RMSE (W)", lambda value: f"{value:.2f}"),
            ("MAE", "MAE (W)", lambda value: f"{value:.2f}"),
            ("MAPE", "MAPE (%)", lambda value: f"{value:.2f}"),
            ("Response_Time_ms", "Response time (ms)",
             lambda value: f"{value:.2f}"),
        ],
        "lower_is_better_metrics.png",
        "Test-Set Metrics: Lower Is Better",
    )
    plot_metric_group(
        comparison,
        [("R2", "R2 score", lambda value: f"{value:.4f}")],
        "higher_is_better_metrics.png",
        "Test-Set Metrics: Higher Is Better",
    )
    plot_overall_ranks(ranked)


def write_readme(comparison, ranked, selections, model_scores, overall_model):
    def value(model, zone, column, digits=2):
        row = comparison[(comparison["Model"] == model) &
                         (comparison["Zone"] == zone)].iloc[0]
        return f"{row[column]:.{digits}f}"

    lines = [
        "# LSTM vs GRU Test-Set Comparison",
        "",
        "## Scope",
        "",
        "This comparison uses final **test-set results only**. Validation metrics,",
        "training history, and tuning results are excluded from model selection.",
        "The source files are [`lstm_final_summary.csv`](../lstm/results/lstm_final_summary.csv)",
        "and [`results_gru.csv`](../gru/docs/results/results_gru.csv).",
        "",
        "## Comparison Rules",
        "",
        "- Lower is better: RMSE, MAE, MAPE, and response time.",
        "- Higher is better: R2.",
        "- Response time is LSTM `Average_Response_Time_ms` and GRU",
        "  `Single_Sample_Latency_ms`, because both measure one sample at a time.",
        "- GRU `Avg_Inference_Time_Per_Sample_ms` is a separate batched-throughput",
        "  metric and is not used as single-sample response time.",
        "- Models are ranked within each zone for every common test metric.",
        "- The average rank gives equal weight to accuracy and response time.",
        "- The lowest average rank is selected per zone and overall.",
        "- GRU-only throughput and batched inference fields are preserved in the source",
        "  results but are not used in the cross-model ranking because the LSTM summary",
        "  does not report equivalent test-set fields.",
        "",
        "## Test-Set Metrics",
        "",
        "| Zone | Model | RMSE (W) | MAE (W) | MAPE (%) | R2 | Response time (ms) | Average rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in ranked.sort_values(["Zone", "Model"]).iterrows():
        lines.append(
            f"| {row['Zone']} | {row['Model']} | {row['RMSE']:.2f} | "
            f"{row['MAE']:.2f} | {row['MAPE']:.2f} | {row['R2']:.4f} | "
            f"{row['Response_Time_ms']:.2f} | {row['Average_Rank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Final Selections",
            "",
            "### Best Model Per Zone",
            "",
            "| Zone | Selected model | Average rank | Reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for _, row in selections.iterrows():
        competitor = ranked[
            (ranked["Zone"] == row["Zone"])
            & (ranked["Model"] != row["Model"])
        ].iloc[0]
        winning_metrics = [
            metric
            for metric in ["RMSE", "MAE", "MAPE", "R2", "Response_Time_ms"]
            if (
                (metric in LOWER_IS_BETTER and row[metric]
                 < competitor[metric])
                or (
                    metric in HIGHER_IS_BETTER and row[metric] > competitor[metric]
                )
            )
        ]
        metric_text = ", ".join(winning_metrics)
        lines.append(
            f"| {row['Zone']} | **{row['Model']}** | {row['Average_Rank_Score']:.2f} | "
            f"Wins on {metric_text}; best equal-weighted test-set rank. |"
        )
    overall_score = model_scores.iloc[0]["Overall_Average_Rank"]
    lines.extend(
        [
            "",
            "### Overall Best Model",
            "",
            f"**{overall_model}** is the overall best model with an average rank of "
            f"**{overall_score:.2f}** across all zones.",
            "This selection reflects the best combined test-set balance of prediction",
            "accuracy and response time under the stated metric directions.",
            "",
            "## Generated Assets",
            "",
            "- [`comparison_test_metrics.csv`](comparison_test_metrics.csv)",
            "- [`comparison_rankings.csv`](comparison_rankings.csv)",
            "- [`best_model_per_zone.csv`](best_model_per_zone.csv)",
            "- [`overall_model_scores.csv`](overall_model_scores.csv)",
            "- [`overall_best_model.csv`](overall_best_model.csv)",
            "",
            "### Visualizations",
            "",
            "![Lower-is-better test metrics](plots/lower_is_better_metrics.png)",
            "",
            "![Higher-is-better test metrics](plots/higher_is_better_metrics.png)",
            "",
            "![Overall model ranking](plots/overall_model_ranking.png)",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python -m lstm_gru.overall_visualization",
            "```",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    comparison = load_comparison_data()
    ranked = rank_models(comparison)
    selections, model_scores, overall_model = save_results(comparison, ranked)
    generate_plots(comparison, ranked)
    write_readme(comparison, ranked, selections, model_scores, overall_model)
    print(f"Saved comparison assets to: {OUTPUT_DIR}")
    print(selections[["Zone", "Model", "Average_Rank_Score"]
                     ].to_string(index=False))
    print(f"Overall best model: {overall_model}")


if __name__ == "__main__":
    main()
