'''
GRU Model

This file contains the implementation of the Gated Recurrent Unit (GRU) model.
'''

import time
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential, layers


def build_gru_model(architecture, input_shape):
    """
    Build and compile a Keras GRU model from an architecture spec.

    Parameters
    ----------
    architecture : dict
        One entry from gru_config.GRU_ARCHITECTURES. Must contain
        "layers" (list of {"units", "return_sequences"}), "dropout",
        "dense_units" (int or None) and "learning_rate".

    input_shape : tuple
        (lookback, num_features), e.g. (144, 27).

    Returns
    -------
    keras.Model
        Compiled model, ready to fit.
    """

    model = Sequential(name=architecture["name"])
    model.add(layers.Input(shape=input_shape))

    for layer_config in architecture["layers"]:
        model.add(
            layers.GRU(
                layer_config["units"],
                return_sequences=layer_config["return_sequences"]
            )
        )

    if architecture["dropout"] > 0:
        model.add(layers.Dropout(architecture["dropout"]))

    if architecture["dense_units"]:
        model.add(
            layers.Dense(architecture["dense_units"], activation="relu")
        )

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=architecture["learning_rate"]
        ),
        loss="mse"
    )

    return model
