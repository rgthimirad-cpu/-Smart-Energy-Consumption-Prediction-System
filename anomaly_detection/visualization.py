from __future__ import annotations

# IMPORTANT:
# Use a non-GUI backend because this script runs from the command line.
# This prevents Tkinter "main thread is not in main loop" messages.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR


def _prepare_output_directory() -> None:
    """Ensure the figures directory exists."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def plot_zone_anomalies(
    result_df: pd.DataFrame,
    zone: str,
    n_days: int = 14,
) -> None:
    """
    Plot power consumption and detected anomaly severity.

    Output:
        anomalies_zone_1.png
        anomalies_zone_2.png
        anomalies_zone_3.png
    """

    _prepare_output_directory()

    df = (
        result_df[
            result_df["Eligible_For_Scoring"]
        ]
        .copy()
    )

    if df.empty:
        return

    start = df["DateTime"].min()

    end = (
        start
        + pd.Timedelta(days=n_days)
    )

    window = df[
        df["DateTime"] < end
    ]

    if window.empty:
        return

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        window["DateTime"],
        window["Value_Watts"],
        linewidth=1.0,
        color="#1f77b4",
        label="Power Consumption",
    )

    warning = window[
        window["Severity_Level"] == "Warning"
    ]

    critical = window[
        window["Severity_Level"] == "Critical"
    ]

    if not warning.empty:
        ax.scatter(
            warning["DateTime"],
            warning["Value_Watts"],
            color="orange",
            s=20,
            zorder=5,
            label="Warning",
        )

    if not critical.empty:
        ax.scatter(
            critical["DateTime"],
            critical["Value_Watts"],
            color="red",
            s=30,
            zorder=6,
            label="Critical",
        )

    ax.set_title(
        f"{zone} — Anomaly Detection "
        f"(first {n_days} eligible days)"
    )

    ax.set_ylabel(
        "Power Consumption (Watts)"
    )

    ax.set_xlabel("DateTime")

    ax.legend()

    fig.autofmt_xdate()

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"anomalies_{zone.lower()}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_severity_distribution(
    result_df: pd.DataFrame,
    zone: str,
) -> None:
    """
    Plot distribution of the maximum rolling z-score.

    Output:
        severity_distribution_zone_1.png
        severity_distribution_zone_2.png
        severity_distribution_zone_3.png
    """

    _prepare_output_directory()

    df = result_df[
        result_df["Eligible_For_Scoring"]
    ]

    if df.empty:
        return

    severity = (
        df["Anomaly_Severity"]
        .dropna()
    )

    if severity.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        severity,
        bins=60,
        color="#4c78a8",
        alpha=0.8,
    )

    ax.axvline(
        3.0,
        color="red",
        linestyle="--",
        label="Reference z=3",
    )

    ax.set_title(
        f"{zone} — Statistical Anomaly Severity"
    )

    ax.set_xlabel(
        "Maximum |z-score| "
        "(rolling 1h / 24h)"
    )

    ax.set_ylabel(
        "Number of observations"
    )

    ax.legend()

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"severity_distribution_{zone.lower()}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_detector_agreement(
    result_df: pd.DataFrame,
    zone: str,
) -> None:
    """
    Plot the number of anomaly detectors agreeing for each observation.

    Output:
        detector_agreement_zone_1.png
        detector_agreement_zone_2.png
        detector_agreement_zone_3.png
    """

    _prepare_output_directory()

    df = result_df[
        result_df["Eligible_For_Scoring"]
    ].copy()

    if df.empty:
        return

    if "Signals_Agreeing" not in df.columns:
        raise KeyError(
            "Result data must contain "
            "'Signals_Agreeing'."
        )

    counts = (
        df["Signals_Agreeing"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color="#59a14f",
    )

    ax.set_title(
        f"{zone} — Detector Agreement"
    )

    ax.set_xlabel(
        "Number of agreeing anomaly signals"
    )

    ax.set_ylabel(
        "Observations"
    )

    # Display values above bars.
    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    output_path = (
        FIGURES_DIR
        / f"detector_agreement_{zone.lower()}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_all_zone_outputs(
    result_df: pd.DataFrame,
    zone: str,
    n_days: int = 14,
) -> None:
    """
    Generate every visualization for one zone.
    """

    plot_zone_anomalies(
        result_df=result_df,
        zone=zone,
        n_days=n_days,
    )

    plot_severity_distribution(
        result_df=result_df,
        zone=zone,
    )

    plot_detector_agreement(
        result_df=result_df,
        zone=zone,
    )

