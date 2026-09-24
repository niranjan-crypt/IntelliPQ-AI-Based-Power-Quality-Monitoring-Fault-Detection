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

    def generate_normal(self, amplitude: float = 1.0, freq: float = 50.0, noise_snr: float = None) -> np.ndarray:
        """Generates normal operational voltage signal."""
        signal = amplitude * np.sin(2 * np.pi * freq * self.t)
        if noise_snr is not None:
            signal = self._add_noise(signal, noise_snr)
        return signal

    def generate_voltage_sag(
        self,
        amplitude: float = 0.7,
        freq: float = 50.0,
        start_time: float = 0.0,
        end_time: float = 1.0,
        noise_snr: float = None
    ) -> np.ndarray:
        """Generates voltage sag signal (0.1 pu to 0.89 pu)."""
        nominal = np.sin(2 * np.pi * freq * self.t)
        envelope = np.ones_like(self.t)
        mask = (self.t >= start_time) & (self.t <= end_time)
        envelope[mask] = amplitude
        signal = nominal * envelope
        if noise_snr is not None:
            signal = self._add_noise(signal, noise_snr)
        return signal

    def generate_voltage_swell(
        self,
        amplitude: float = 1.3,
        freq: float = 50.0,
        start_time: float = 0.0,
        end_time: float = 1.0,
        noise_snr: float = None
    ) -> np.ndarray:
        """Generates voltage swell signal (1.11 pu to 1.8 pu)."""
        nominal = np.sin(2 * np.pi * freq * self.t)
        envelope = np.ones_like(self.t)
        mask = (self.t >= start_time) & (self.t <= end_time)
        envelope[mask] = amplitude
        signal = nominal * envelope
        if noise_snr is not None:
            signal = self._add_noise(signal, noise_snr)
        return signal

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Injects Additive White Gaussian Noise (AWGN) at specified SNR (dB)."""
        signal_power = np.mean(signal ** 2)
        if signal_power == 0:
            return signal
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
