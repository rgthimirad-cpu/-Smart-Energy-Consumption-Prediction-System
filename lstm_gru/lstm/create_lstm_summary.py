from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = ROOT / "lstm_gru" / "lstm" / "results"

def build_zone_summary(zone):

    # --------------------------------------------------------
    # Load tuning results
    # --------------------------------------------------------

    tuning_path = (
        RESULT_DIR /
        f"{zone}_lstm_tuning_results.csv"
    )

    tuning_df = pd.read_csv(tuning_path)

    best_tuning = tuning_df.loc[
        tuning_df["RMSE"].idxmin()
    ]

    # --------------------------------------------------------
    # Load final test metrics
    # --------------------------------------------------------

    test_path = (
        RESULT_DIR /
        f"{zone}_lstm_test_metrics.csv"
    )

    test_df = pd.read_csv(test_path)

    test_row = test_df.iloc[0]

    # --------------------------------------------------------
    # Load response time
    # --------------------------------------------------------

    response_path = (
        RESULT_DIR /
        f"{zone}_lstm_response_time.csv"
    )

    response_df = pd.read_csv(response_path)

    response_row = response_df.iloc[0]

    # --------------------------------------------------------
    # Build combined row
    # --------------------------------------------------------

    summary = {
        "Zone": zone,

        "Best_Run": best_tuning["Run"],
        "Layers": int(best_tuning["Layers"]),
        "Units": best_tuning["Units"],
        "Dropout": best_tuning["Dropout"],
        "Learning_Rate": best_tuning["Learning_Rate"],
        "Batch_Size": int(best_tuning["Batch_Size"]),
        "Best_Epoch": int(best_tuning["Best_Epoch"]),

        "Validation_RMSE": best_tuning["RMSE"],
        "Validation_MAE": best_tuning["MAE"],
        "Validation_MAPE": best_tuning["MAPE"],
        "Validation_R2": best_tuning["R2"],

        "Test_RMSE": test_row["RMSE"],
        "Test_MAE": test_row["MAE"],
        "Test_MAPE": test_row["MAPE"],
        "Test_R2": test_row["R2"],

        "Average_Response_Time_ms":
            response_row["Average_Response_Time_ms"],

        "Median_Response_Time_ms":
            response_row["Median_Response_Time_ms"],
    }

    return summary


def create_summary():

    zones = [
        "Zone_1",
        "Zone_2",
        "Zone_3"
    ]

    rows = []

    for zone in zones:
        rows.append(
            build_zone_summary(zone)
        )

    summary_df = pd.DataFrame(rows)

    output_path = (
        RESULT_DIR /
        "lstm_final_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False
    )

    print("\nLSTM FINAL SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(index=False)
    )

    print("\nSummary saved to:")
    print(output_path)


if __name__ == "__main__":
    create_summary()