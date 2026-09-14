# LSTM/GRU Time-Series Windowing

import numpy as np


def create_sequences(
    data,
    feature_columns,
    target_column,
    lookback=144,
    horizon=1
):
    """
    Convert time-series data into sequences for LSTM/GRU models.

    Parameters
    ----------
    data : pandas.DataFrame
        Chronologically ordered dataframe.

    feature_columns : list
        Columns used as model inputs.

    target_column : str
        Column to predict.

    lookback : int
        Number of previous time steps used as input.
        144 steps = 24 hours for 10-minute data.

    horizon : int
        Number of time steps ahead to predict.
        1 step = 10 minutes.

    Returns
    -------
    X : numpy.ndarray
        Shape: (samples, lookback, features)

    y : numpy.ndarray
        Shape: (samples,)
    """

    X_values = data[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_values = data[target_column].to_numpy(
        dtype=np.float32
    )

    X = []
    y = []

    for i in range(
        lookback,
        len(data) - horizon + 1
    ):
        X.append(
            X_values[i - lookback:i]
        )

        y.append(
            y_values[i + horizon - 1]
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )