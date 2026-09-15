from .arima_sarima import run_arima_sarima
from .comparison import build_comparison
from .diagnostics import run_diagnostics
from .prophet_models import run_prophet
from .regression_models import run_regression


def main() -> None:
    print("1/5 Running stationarity diagnostics and ACF/PACF analysis...")
    run_diagnostics()
    print("2/5 Training and evaluating ARIMA/SARIMA models...")
    run_arima_sarima()
    print("3/5 Training and evaluating Prophet models...")
    run_prophet()
    print("4/5 Training and evaluating regression models...")
    run_regression()
    print("5/5 Building final model comparison and report...")
    summary = build_comparison(require_complete=True)
    print("\nFinal average test performance:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
