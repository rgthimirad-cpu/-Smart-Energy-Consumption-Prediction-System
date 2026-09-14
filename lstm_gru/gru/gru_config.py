'''
GRU Configuration

This file contains the configuration parameters for the Gated Recurrent Unit (GRU) model.
'''

# Training Settings
BATCH_SIZE = 256
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 6

LATENCY_TOLERANCE = 3.0

TIMING_REPEATS = 7


# Architecture Settings
GRU_ARCHITECTURES = [
    {
        "name": "gru_small",
        "layers": [
            {"units": 64, "return_sequences": False},
        ],
        "dropout": 0.0,
        "dense_units": None,
        "learning_rate": 1e-3,
    },
    {
        "name": "gru_medium",
        "layers": [
            {"units": 128, "return_sequences": False},
        ],
        "dropout": 0.2,
        "dense_units": 32,
        "learning_rate": 1e-3,
    },
    {
        "name": "gru_stacked_small",
        "layers": [
            {"units": 64, "return_sequences": True},
            {"units": 32, "return_sequences": False},
        ],
        "dropout": 0.1,
        "dense_units": None,
        "learning_rate": 1e-3,
    },
    {
        "name": "gru_stacked_large",
        "layers": [
            {"units": 128, "return_sequences": True},
            {"units": 64, "return_sequences": False},
        ],
        "dropout": 0.2,
        "dense_units": 32,
        "learning_rate": 5e-4,
    },
]
