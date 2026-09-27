"""
IntelliPQ - Deep Learning Model Training Pipeline
Generates IEEE 1159 synthetic disturbance waveforms, trains a CNN+LSTM network, and saves pq_model.h5.
"""

import argparse
import sys
import numpy as np
from src.signal_generator import SignalGenerator
from src.model_utils import build_cnn_lstm_model, FAULT_DICT


def train(samples_per_class: int = 600, epochs: int = 12, batch_size: int = 32, output_path: str = "pq_model.h5"):
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

    # 3. Build Model
    print("[*] Building 1D-CNN + Bidirectional LSTM architecture...")
    model = build_cnn_lstm_model(input_shape=(1000, 1), num_classes=4)
    model.summary()

    # 4. Train
    print(f"[*] Training for {epochs} epochs (batch size: {batch_size})...")
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate
    print("\n[*] Evaluating on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Final Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Predictions breakdown
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nPer-class accuracy:")
    for class_id, class_name in FAULT_DICT.items():
        mask = (y_test == class_id)
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask]) * 100
            print(f"  • {class_name:22s}: {class_acc:.2f}% ({np.sum(mask)} test samples)")

    # 6. Save
    model.save(output_path)
    print(f"\n[✔] Successfully trained and saved model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IntelliPQ CNN+LSTM power quality classifier.")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Number of synthetic samples per class (default: 500)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--output", type=str, default="pq_model.h5", help="Output .h5 file path (default: pq_model.h5)")
    args = parser.parse_args()

    train(
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output,
    )
