from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape,
    units=(64,),
    dropout=0.2,
    learning_rate=0.001
):
    """
    Build an LSTM forecasting model.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sequence:
        (timesteps, features)

    units : tuple, list or int
        Number of units in each LSTM layer.

        Examples:
        64          -> one LSTM layer with 64 units
        (64,)       -> one LSTM layer with 64 units
        (64, 32)    -> two LSTM layers with 64 and 32 units

    dropout : float
        Dropout rate used after each LSTM layer.

    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM model.
    """

    # Allow a single integer for backward compatibility.
    if isinstance(units, int):
        units = (units,)

    model = Sequential()

    model.add(
        Input(shape=input_shape)
    )

    # Add one or more LSTM layers.
    for index, layer_units in enumerate(units):

        # Every LSTM except the final one must return
        # the full sequence to the next LSTM layer.
        return_sequences = index < len(units) - 1

        model.add(
            LSTM(
                layer_units,
                return_sequences=return_sequences
            )
        )

        model.add(
            Dropout(dropout)
        )

    # One output because HORIZON = 1.
    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model


if __name__ == "__main__":

    # Example baseline model
    model = build_lstm_model(
        input_shape=(144, 27),
        units=(64,),
        dropout=0.2,
        learning_rate=0.001
    )

    model.summary()