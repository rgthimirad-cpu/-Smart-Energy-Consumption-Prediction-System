# LSTM/GRU Evaluation Metrics

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate RMSE, MAE, MAPE and R2.

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        Dictionary containing the four evaluation metrics.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Root Mean Squared Error
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    # Mean Absolute Error
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # Mean Absolute Percentage Error
    # Ignore zero actual values to avoid division by zero.
    nonzero = np.abs(y_true) > 1e-12

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero])
                / y_true[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    # R-squared
    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }