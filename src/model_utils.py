"""
IntelliPQ - Model Utilities
Builds, trains, loads, and executes CNN + LSTM hybrid neural networks for time-series classification.
"""

from typing import Optional, Tuple
import numpy as np


FAULT_DICT = {
    0: "Normal",
    1: "Voltage Sag",
    2: "Voltage Swell",
    3: "Harmonic Distortion",
}


def build_cnn_lstm_model(input_shape: Tuple[int, int] = (1000, 1), num_classes: int = 4):
    """
    Constructs a hybrid 1D-CNN + Bidirectional LSTM model for power disturbance classification.
    - 1D-CNN extracts local spatial waveform features (transients, harmonic peaks).
    - Bi-LSTM captures long-range temporal periodicity across multiple AC cycles.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D,
        MaxPooling1D,
        Bidirectional,
        LSTM,
        Dense,
        Dropout,
        BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        # Stage 1: Feature Extraction (CNN)
        Conv1D(filters=32, kernel_size=7, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Stage 2: Temporal Sequence Modeling (Bi-LSTM)
        Bidirectional(LSTM(units=48, return_sequences=False)),
        Dropout(0.3),

        # Stage 3: Classification Head
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
