from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from .config import FIGURES_DIR, PUBLIC_RESULTS_DIR, SEASONAL_PERIOD, TARGETS, ensure_output_dirs
from .data import load_unscaled_time_series_splits

warnings.filterwarnings("ignore")


def _adf_row(zone: str, series_name: str, values) -> dict:
    result = adfuller(pd.Series(values).dropna().astype(float).to_numpy(), maxlag=24, autolag="AIC")
    return {
        "Zone": zone,
        "Series": series_name,
        "ADF_Statistic": float(result[0]),
        "p_value": float(result[1]),
        "Used_Lags": int(result[2]),
        "Observations": int(result[3]),
        "Stationary_at_5pct": bool(result[1] < 0.05),
    }


def run_diagnostics() -> pd.DataFrame:
    """Run ADF tests and create ACF/PACF plots for all three zones."""
    ensure_output_dirs()
    train = load_unscaled_time_series_splits()["train"]
    rows = []

    for zone, target in TARGETS.items():
        y = train[target].astype(float)
        diff1 = y.diff().dropna()
        seasonal_diff = y.diff(SEASONAL_PERIOD).dropna()
        rows.extend([
            _adf_row(target, "Original", y),
            _adf_row(target, "First difference", diff1),
            _adf_row(target, f"Seasonal difference (lag {SEASONAL_PERIOD})", seasonal_diff),
        ])

        # Use the most recent 5,000 training observations for correlation plots.
        # This is large enough to show several daily cycles while keeping PACF computation practical.
        plot_y = y.tail(5000)
        plot_seasonal = seasonal_diff.tail(5000)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        plot_acf(plot_y, lags=288, ax=axes[0, 0], fft=True, zero=False)
        axes[0, 0].set_title(f"{target} - ACF (original)")
        plot_pacf(plot_y, lags=48, ax=axes[0, 1], method="ywmle", zero=False)
        axes[0, 1].set_title(f"{target} - PACF (original)")
        plot_acf(plot_seasonal, lags=288, ax=axes[1, 0], fft=True, zero=False)
        axes[1, 0].set_title(f"{target} - ACF (seasonal difference)")
        plot_pacf(plot_seasonal, lags=48, ax=axes[1, 1], method="ywmle", zero=False)
        axes[1, 1].set_title(f"{target} - PACF (seasonal difference)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"acf_pacf_{zone.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    result = pd.DataFrame(rows)
    result.to_csv(PUBLIC_RESULTS_DIR / "stationarity_adf.csv", index=False)
    return result


if __name__ == "__main__":
    print(run_diagnostics().to_string(index=False))
