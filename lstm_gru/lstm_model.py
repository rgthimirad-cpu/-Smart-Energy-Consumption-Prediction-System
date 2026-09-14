from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape,
    units=64,
    dropout=0.2,
    learning_rate=0.001
):
    """
    Build the baseline LSTM forecasting model.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sequence:
        (timesteps, features)

    units : int
        Number of LSTM units.

    dropout : float
        Dropout rate used for regularization.

    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM model.
    """

    model = Sequential([
        Input(shape=input_shape),

        LSTM(units),

        Dropout(dropout),

        Dense(1)
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model


if __name__ == "__main__":

    # Shared data configuration:
    # 144 previous time steps
    # 27 input features
    model = build_lstm_model(
        input_shape=(144, 27)
    )

    model.summary()