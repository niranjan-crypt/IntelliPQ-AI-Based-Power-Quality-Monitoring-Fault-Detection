"""
IntelliPQ - Deep Learning Model Training Pipeline
Generates IEEE 1159 synthetic disturbance waveforms, trains a CNN+LSTM network, and saves pq_model.h5.
"""

import sys
import numpy as np
from src.signal_generator import SignalGenerator
from src.model_utils import build_cnn_lstm_model, FAULT_DICT


def train(samples_per_class: int = 600):
    print("=" * 60)
    print("⚡ IntelliPQ - CNN + LSTM Model Training Pipeline")
    print("=" * 60)

    # 1. Generate Dataset
    print(f"[*] Generating synthetic dataset: {samples_per_class} samples per class across 4 classes...")
    gen = SignalGenerator(sampling_rate=1000, duration=1.0, base_freq=50.0)
    X, y = gen.generate_balanced_dataset(samples_per_class=samples_per_class)
    print(f"[+] Total samples generated: {len(X)} with shape {X.shape}")

    # 2. Train / Test Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Reshape for Conv1D: (batch_size, timesteps, channels=1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print(f"[+] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
