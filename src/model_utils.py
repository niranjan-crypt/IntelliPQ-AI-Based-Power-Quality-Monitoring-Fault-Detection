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


def load_pq_model(model_path: str = "pq_model.h5"):
    """Safely loads the Keras model if available, returns None otherwise."""
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except Exception:
        return None


def predict_waveform(model, signal: np.ndarray, timesteps: int = 1000) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Runs model inference on raw 1D time-series voltage signal.
    
    Returns:
        (predicted_class_name, confidence_percentage, all_probabilities)
    """
    if model is None:
        return None

    # Reshape signal to (1, timesteps, 1)
    if len(signal) != timesteps:
        # Interpolate or slice if needed
        indices = np.linspace(0, len(signal) - 1, timesteps)
        signal = np.interp(indices, np.arange(len(signal)), signal)

    signal_input = signal.reshape(1, timesteps, 1)
    probabilities = model.predict(signal_input, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx] * 100.0)
    class_name = FAULT_DICT.get(predicted_idx, "Unknown")

    return class_name, confidence, probabilities
