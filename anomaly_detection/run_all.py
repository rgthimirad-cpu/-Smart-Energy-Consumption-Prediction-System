from __future__ import annotations

# ----------------------------------------------------------------------
# IMPORTANT:
# Use a non-GUI Matplotlib backend.
#
# This prevents Tkinter errors such as:
# RuntimeError: main thread is not in main loop
#
# It must be configured BEFORE importing the visualization module.
# ----------------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from .config import (
    OUTPUT_DIR,
    PREDICTIONS_DIR,
    PUBLIC_RESULTS_DIR,
    TUNING_DIR,
    TARGETS,
    FIGURES_DIR,
    ensure_output_dirs,
)

from .data import (
    load_feature_data,
    split_by_row,
)

from .evaluation import (
    run_synthetic_benchmark,
    tune_all_thresholds,
)

from .pipeline import (
    score_full_dataset,
)

from .visualization import (
    plot_all_zone_outputs,
)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _print_split_summary(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Print chronological train/validation/test split information."""

    print()
    print("Chronological split:")

    for name in (
        "train",
        "validation",
        "test",
    ):
        frame = splits[name]

        start = frame["DateTime"].min()
        end = frame["DateTime"].max()

        print(
            f"  {name:<11}: "
            f"{len(frame):,} rows "
            f"({start} -> {end})"
        )


def _safe_percentage(
    numerator: int,
    denominator: int,
) -> float:
    """Calculate percentage safely."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


def _get_preprocessing_changed_count(
    result: pd.DataFrame,
) -> int:
    """
    Get the number of rows changed by preprocessing.

    Supports either:
        Changed_By_Preprocessing

    or the older possible column name:
        Capped_By_Preprocessing
    """

    if "Changed_By_Preprocessing" in result.columns:
        return int(
            result["Changed_By_Preprocessing"].sum()
        )

    if "Capped_By_Preprocessing" in result.columns:
        return int(
            result["Capped_By_Preprocessing"].sum()
        )

    return 0


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 75)
    print("ANOMALY DETECTION PIPELINE")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------

    ensure_output_dirs()

    # ------------------------------------------------------------------
    # Load feature-engineered dataset
    # ------------------------------------------------------------------

    print()
    print("Loading feature-engineered dataset...")

    df = load_feature_data()

    print(
        f"Loaded {len(df):,} rows "
        f"from {df['DateTime'].min()} "
        f"to {df['DateTime'].max()}"
    )

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------

    splits = split_by_row(df)

    _print_split_summary(splits)

    # ------------------------------------------------------------------
    # Containers for final outputs
    # ------------------------------------------------------------------

    benchmark_frames: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    full_frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Process every zone
    # ------------------------------------------------------------------

    for zone, target in TARGETS.items():

        print()
        print("=" * 75)
        print(f"{zone}: {target}")
        print("=" * 75)

        # ==============================================================
        # 1/3 Tune parameters
        # ==============================================================

        print()
        print(
            "  1/3 Tuning Isolation Forest / "
            "z-score / MAD thresholds..."
        )

        best = tune_all_thresholds(
            train_df=splits["train"],
            validation_df=splits["validation"],
            zone=zone,
        )

        tuning_frames.append(
            best["grid"]
        )

        print()
        print("    Selected parameters:")

        print(
            "      Isolation Forest contamination = "
            f"{best['contamination']}"
        )

        print(
            "      Statistical z-score threshold  = "
            f"{best['zscore_threshold']}"
        )

        print(
            "      MAD z-score threshold          = "
            f"{best['mad_threshold']}"
        )

        # ==============================================================
        # 2/3 Final synthetic benchmark
        # ==============================================================

        print()
        print(
            "  2/3 Running final synthetic-injection "
            "benchmark on TEST data..."
        )

        benchmark = run_synthetic_benchmark(
            train_df=splits["train"],
            evaluation_df=splits["test"],
            zone=zone,
            contamination=best["contamination"],
            zscore_threshold=best["zscore_threshold"],
            mad_threshold=best["mad_threshold"],
        )

        print()
        print("    Benchmark results:")

        print(
            benchmark.to_string(
                index=False
            )
        )

        benchmark_frames.append(
            benchmark
        )

        # ==============================================================
        # 3/3 Score complete dataset
        # ==============================================================

        print()
        print(
            "  3/3 Scoring complete dataset..."
        )

        result = score_full_dataset(
            df=df,
            zone=zone,
            zscore_threshold=best["zscore_threshold"],
            mad_threshold=best["mad_threshold"],
            contamination=best["contamination"],
        )

        full_frames.append(
            result
        )

        # ==============================================================
        # Save per-zone anomaly results
        # ==============================================================

        zone_output = (
            OUTPUT_DIR
            / f"anomalies_{zone.lower()}.csv"
        )

        result.to_csv(
            zone_output,
            index=False,
        )

        print()
        print(
            f"    Saved: {zone_output}"
        )

        # ==============================================================
        # Generate plots
        # ==============================================================

        print()
        print("    Generating plots...")

        try:
            plot_all_zone_outputs(
                result_df=result,
                zone=zone,
            )

            print(
                "    Plots generated."
            )

        except Exception as exc:
            # Do not silently destroy the entire pipeline if a plot
            # fails. The CSV results are still valid.
            print()
            print(
                "    WARNING: Plot generation failed."
            )
            print(
                f"    Reason: {type(exc).__name__}: {exc}"
            )

        # ==============================================================
        # Zone summary
        # ==============================================================

        n_flagged = int(
            result["Is_Anomaly"].sum()
        )

        n_eligible = int(
            result["Eligible_For_Scoring"].sum()
        )

        n_critical = int(
            (
                result["Severity_Level"]
                == "Critical"
            ).sum()
        )

        n_changed = _get_preprocessing_changed_count(
            result
        )

        percentage = _safe_percentage(
            n_flagged,
            n_eligible,
        )

        print()
        print("    Zone summary:")

        print(
            f"      Flagged anomalies       : "
            f"{n_flagged:,}"
        )

        print(
            f"      Eligible rows           : "
            f"{n_eligible:,}"
        )

        print(
            f"      Flagged percentage      : "
            f"{percentage:.2f}%"
        )

        print(
            f"      Critical observations   : "
            f"{n_critical:,}"
        )

        print(
            f"      Preprocessing changed   : "
            f"{n_changed:,}"
        )

    # ------------------------------------------------------------------
    # Save benchmark results
    # ------------------------------------------------------------------

    benchmark_output = (
        PUBLIC_RESULTS_DIR
        / "results_anomaly_detection_benchmark.csv"
    )

    if benchmark_frames:

        pd.concat(
            benchmark_frames,
            ignore_index=True,
        ).to_csv(
            benchmark_output,
            index=False,
        )

        print()
        print(
            "Saved benchmark results:"
        )
        print(
            f"  {benchmark_output}"
        )

    # ------------------------------------------------------------------
    # Save threshold tuning grid
    # ------------------------------------------------------------------

    tuning_output = (
        TUNING_DIR
        / "anomaly_detection_threshold_grid.csv"
    )

    if tuning_frames:

        pd.concat(
            tuning_frames,
            ignore_index=True,
        ).to_csv(
            tuning_output,
            index=False,
        )

        print()
        print(
            "Saved tuning grid:"
        )
        print(
            f"  {tuning_output}"
        )

    # ------------------------------------------------------------------
    # Combine all zones
    # ------------------------------------------------------------------

    if not full_frames:
        raise RuntimeError(
            "No zone results were generated."
        )

    combined = pd.concat(
        full_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Save combined dashboard/project handoff
    # ------------------------------------------------------------------

    combined_output = (
        OUTPUT_DIR
        / "anomalies_all_zones.csv"
    )

    combined.to_csv(
        combined_output,
        index=False,
    )

    print()
    print(
        "Saved combined anomaly results:"
    )
    print(
        f"  {combined_output}"
    )

    # ------------------------------------------------------------------
    # Save internal predictions
    # ------------------------------------------------------------------

    predictions_output = (
        PREDICTIONS_DIR
        / "predictions_anomaly_detection.csv"
    )

    combined.to_csv(
        predictions_output,
        index=False,
    )

    print()
    print(
        "Saved internal prediction results:"
    )
    print(
        f"  {predictions_output}"
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 75)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 75)

    print()
    print("Dashboard / project handoff:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print("Internal predictions:")
    print(
        f"  {PREDICTIONS_DIR}"
    )

    print()
    print("Benchmark results:")
    print(
        f"  {PUBLIC_RESULTS_DIR}"
    )

    print()
    print("Tuning results:")
    print(
        f"  {TUNING_DIR}"
    )

    print()
    print("Review plots:")
    print(
        f"  {FIGURES_DIR}"
    )

    print()
    print("Expected detector-agreement plots:")

    for zone in TARGETS:
        print(
            f"  {FIGURES_DIR / f'detector_agreement_{zone.lower()}.png'}"
        )

    print()
    print("=" * 75)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()

