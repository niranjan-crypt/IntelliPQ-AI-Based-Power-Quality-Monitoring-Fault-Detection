"""
IntelliPQ - Signal Generator Module
Generates IEEE 1159 compliant synthetic power quality disturbance waveforms.
"""

import numpy as np


class SignalGenerator:
    """Generates synthetic electrical voltage signals with various Power Quality Disturbances (PQDs)."""

    CLASS_LABELS = {
        0: "Normal",
        1: "Voltage Sag",
        2: "Voltage Swell",
        3: "Harmonic Distortion",
    }

    def __init__(self, sampling_rate: int = 1000, duration: float = 1.0, base_freq: float = 50.0):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.base_freq = base_freq
        self.num_samples = int(sampling_rate * duration)
        self.t = np.linspace(0, duration, self.num_samples, endpoint=False)

    def generate_pure_sine(self, amplitude: float = 1.0, freq: float = None, phase: float = 0.0) -> np.ndarray:
        """Generates an ideal sinusoidal voltage signal."""
        f = freq if freq is not None else self.base_freq
        return amplitude * np.sin(2 * np.pi * f * self.t + phase)
