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


def measure_single_sample_latency(model, X, repeats=5):
    """
    Measure single-sample inference latency in milliseconds.

    Uses a direct model call (model(x, training=False)) rather than
    model.predict(), because predict() carries per-call overhead
    (data adapter setup, retracing) that is not representative of a
    single real-time request. A warm-up call is made first so that
    one-time graph-tracing cost does not distort the measurement, and
    the median of several repeats is used to avoid one slow call
    skewing the result.

    Parameters
    ----------
    model : keras.Model
    X : numpy.ndarray
        Full input array; only the first sample is used.
    repeats : int
        Number of timed calls to take the median over.

    Returns
    -------
    float
        Median single-sample latency in milliseconds.
    """

    sample = tf.convert_to_tensor(X[:1])

    # Warm-up call to avoid tracing overhead
    _ = model(sample, training=False)

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()

        _ = model(sample, training=False)

        durations.append(time.perf_counter() - start)

    return float(np.median(durations)) * 1000  # Convert to milliseconds


def measure_full_set_inference(model, X, batch_size=256):
    """
    Run inference over an entire dataset (e.g. the full test set) and
    time the whole operation, returning the predictions alongside
    total wall-clock time and derived throughput figures.

    Parameters
    ----------
    model : keras.Model
    X : numpy.ndarray
        Full input array to run inference over.
    batch_size : int
        Batch size used for the (efficient, vectorised) bulk call.

    Returns
    -------
    dict with keys:
        predictions, total_seconds, avg_ms_per_sample,
        throughput_samples_per_sec
    """

    warmup_n = min(batch_size, len(X))
    _ = model.predict(X[:warmup_n], verbose=0, batch_size=batch_size)

    start = time.perf_counter()
    predictions = model.predict(X, verbose=0, batch_size=batch_size)
    total_seconds = time.perf_counter() - start

    n_samples = len(X)

    return {
        "predictions": predictions.reshape(-1),
        "total_seconds": total_seconds,
        "avg_ms_per_sample": (total_seconds / n_samples) * 1000,
        "throughput_samples_per_sec": n_samples / total_seconds
    }
