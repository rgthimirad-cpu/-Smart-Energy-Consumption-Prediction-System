from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, PREDICTIONS_DIR, PUBLIC_RESULTS_DIR, REPORTS_DIR, TARGETS, ensure_output_dirs
from .evaluation import PUBLIC_COLUMNS

RESULT_FILES = [
    PUBLIC_RESULTS_DIR / "results_arima_sarima.csv",
    PUBLIC_RESULTS_DIR / "results_prophet.csv",
    PUBLIC_RESULTS_DIR / "results_regression.csv",
]
REQUIRED_MODELS = {"ARIMA", "SARIMA", "Prophet", "Linear Regression", "Random Forest", "XGBoost"}


def _load_results(require_complete: bool = True) -> pd.DataFrame:
    frames = []
    for path in RESULT_FILES:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            if len(df):
                missing_cols = [c for c in PUBLIC_COLUMNS if c not in df.columns]
                if missing_cols:
                    raise ValueError(f"{path.name} is missing columns: {missing_cols}")
                frames.append(df[PUBLIC_COLUMNS])
    if not frames:
        raise FileNotFoundError("No model result files were found.")
    results = pd.concat(frames, ignore_index=True)
    missing_models = REQUIRED_MODELS - set(results["Model"])
    if require_complete and missing_models:
        raise RuntimeError(
            "The comparison is incomplete. Missing model results: " + ", ".join(sorted(missing_models))
        )
    return results


def _save_metric_chart(summary: pd.DataFrame, metric: str, path: Path) -> None:
    ordered = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["Model"], ordered[metric])
    ax.set_ylabel(f"Average {metric} (Watts)" if metric != "MAPE" else "Average MAPE (%)")
    ax.set_title(f"Model Comparison - Average Test {metric}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prediction_plot(zone: str, available_models: set[str]) -> None:
    frames = []
    for path in PREDICTIONS_DIR.glob("predictions_*.csv"):
        df = pd.read_csv(path, parse_dates=["DateTime"])
        frames.append(df)
    if not frames:
        return
    pred = pd.concat(frames, ignore_index=True)
    pred = pred[(pred["Zone"] == zone) & (pred["Split"] == "test")]
    pred = pred[pred["Model"].isin(available_models)]
    if pred.empty:
        return

    # Plot one representative week to keep the figure readable.
    start = pred["DateTime"].min()
    subset = pred[pred["DateTime"] < start + pd.Timedelta(days=7)]
    actual = subset[["DateTime", "Actual_Watts"]].drop_duplicates("DateTime").sort_values("DateTime")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual["DateTime"], actual["Actual_Watts"], label="Actual", linewidth=1.6)
    for model, part in subset.groupby("Model"):
        part = part.sort_values("DateTime")
        ax.plot(part["DateTime"], part["Predicted_Watts"], label=model, linewidth=1.0, alpha=0.8)
    ax.set_title(f"Actual vs Predicted - {zone} (First Test Week)")
    ax.set_ylabel("Power Consumption (Watts)")
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"actual_vs_predicted_{zone.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_report(results: pd.DataFrame, complete: bool) -> None:
    summary = results.groupby("Model", as_index=False)[["RMSE", "MAE", "MAPE"]].mean()
    summary = summary.sort_values("RMSE").reset_index(drop=True)
    best_overall = summary.iloc[0]

    zone_best = []
    for target in TARGETS.values():
        part = results[results["Zone"] == target].sort_values("RMSE")
        if len(part):
            zone_best.append((target, part.iloc[0]["Model"], float(part.iloc[0]["RMSE"])))

    lines = [
        "# Time-Series Forecasting and Regression Comparison Report",
        "",
        "## Objective",
        "",
        "This work implements and compares ARIMA, SARIMA, Prophet, Linear Regression, Random Forest, and XGBoost for the three power-consumption zones. Model selection uses validation data, while the final comparison uses the held-out test period. RMSE, MAE, and MAPE are reported in original power-consumption units (Watts for RMSE/MAE).",
        "",
        "## Evaluation Protocol",
        "",
        "- Time-series models use the unscaled feature-engineered dataset at the original 10-minute frequency.",
        "- Regression models use the preprocessing team's scaled train/validation/test files and are inverse-transformed to Watts before scoring.",
        "- Current raw Zone 1/2/3 consumption columns are excluded from regression predictors to prevent target leakage; lagged and rolling historical features are retained.",
        "- Hyperparameters are selected using the validation split only. Final models are then fitted on train + validation and evaluated on test.",
        "- ARIMA/SARIMA are evaluated as rolling one-step-ahead forecasts, consistent with the availability of lagged observations used by regression models.",
        "",
        "## Test Results",
        "",
        results.sort_values(["Model", "Zone"])[PUBLIC_COLUMNS].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Average Performance Across Zones",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best-Performing Model",
        "",
        f"{'The lowest average test RMSE is achieved' if complete else 'Among the model results currently available, the lowest average test RMSE is achieved'} by **{best_overall['Model']}** ({best_overall['RMSE']:.3f} W). Its average MAE is {best_overall['MAE']:.3f} W and average MAPE is {best_overall['MAPE']:.3f}%.",
        "",
        "The result indicates that this model provides the strongest overall predictive accuracy under the common evaluation protocol. The reason is assessed from measured test errors rather than model complexity alone: lower RMSE shows fewer/lower large errors, while lower MAE reflects lower typical absolute error. For tree-based regression models, the saved feature-importance table can be used to identify which lag, rolling, weather, and time features contributed most strongly. For SARIMA and Prophet, performance can be interpreted in relation to the strong daily/weekly seasonal structure identified during time-series analysis.",
        "",
        "## Best Model by Zone",
        "",
    ]
    for zone, model, rmse in zone_best:
        lines.append(f"- {zone}: **{model}** (RMSE {rmse:.3f} W)")

    lines.extend([
        "",
        "## Notes on Comparability",
        "",
        "All reported errors are on the original power-consumption scale. The models do not use exactly the same predictor information: ARIMA/SARIMA model the target history directly, Prophet uses calendar/seasonal structure plus the specified external regressors, and regression models use engineered lag/rolling/weather/time predictors. This difference is retained because it is part of the required model families, and it should be considered when interpreting why a model performs better.",
        "",
        "## Completion Status",
        "",
        "All required model families are present in the result table." if complete else "The report is currently partial because one or more required model result files have not yet been generated. Run the full pipeline before final submission.",
        "",
    ])
    (REPORTS_DIR / "forecasting_regression_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_comparison(require_complete: bool = True) -> pd.DataFrame:
    ensure_output_dirs()
    results = _load_results(require_complete=require_complete)
    results.to_csv(PUBLIC_RESULTS_DIR / "results_all_models.csv", index=False)
    summary = results.groupby("Model", as_index=False)[["RMSE", "MAE", "MAPE"]].mean()
    summary.to_csv(PUBLIC_RESULTS_DIR / "model_comparison_summary.csv", index=False)

    _save_metric_chart(summary, "RMSE", FIGURES_DIR / "average_rmse_comparison.png")
    _save_metric_chart(summary, "MAE", FIGURES_DIR / "average_mae_comparison.png")
    for zone in TARGETS.values():
        _prediction_plot(zone, set(results["Model"]))

    complete = REQUIRED_MODELS.issubset(set(results["Model"]))
    _write_report(results, complete)
    return summary.sort_values("RMSE")


if __name__ == "__main__":
    print(build_comparison(require_complete=True).to_string(index=False))
