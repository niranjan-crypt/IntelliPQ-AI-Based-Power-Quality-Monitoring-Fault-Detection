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
        """
        Initialize signal generator.
        
        Args:
            sampling_rate: Samples per second (default 1000 Hz).
            duration: Duration of signal in seconds (default 1.0s).
            base_freq: Nominal fundamental frequency in Hz (default 50.0 Hz).
        """
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
        """
        Generates normal operational voltage signal.
        IEEE 1159 standard: 0.9 pu <= Amplitude <= 1.1 pu.
        """
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
        """
        Generates voltage sag signal (0.1 pu to 0.89 pu).
        Can be sustained or instantaneous.
        """
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
        """
        Generates voltage swell signal (1.11 pu to 1.8 pu).
        """
        nominal = np.sin(2 * np.pi * freq * self.t)
        envelope = np.ones_like(self.t)
        mask = (self.t >= start_time) & (self.t <= end_time)
        envelope[mask] = amplitude
        signal = nominal * envelope
        if noise_snr is not None:
            signal = self._add_noise(signal, noise_snr)
        return signal

    def generate_harmonic_distortion(
        self,
        amplitude: float = 1.0,
        freq: float = 50.0,
        harmonic_orders: tuple = (3, 5),
        harmonic_amplitudes: tuple = (0.2, 0.1),
        noise_snr: float = None
    ) -> np.ndarray:
        """
        Generates signal with harmonic distortion (Total Harmonic Distortion >= 5%).
        """
        signal = amplitude * np.sin(2 * np.pi * freq * self.t)
        for order, h_amp in zip(harmonic_orders, harmonic_amplitudes):
            signal += h_amp * np.sin(2 * np.pi * order * freq * self.t)
        if noise_snr is not None:
            signal = self._add_noise(signal, noise_snr)
        return signal

    def generate_dataset_sample(self, class_id: int) -> tuple[np.ndarray, int]:
        """
        Generates a randomized single sample belonging to class_id for ML training.
        0: Normal, 1: Sag, 2: Swell, 3: Harmonic
        """
        freq = np.random.uniform(49.5, 50.5)
        phase = np.random.uniform(0, 2 * np.pi)
        noise_snr = np.random.uniform(30, 50)  # Mild realistic sensor noise

        if class_id == 0:  # Normal
            amp = np.random.uniform(0.95, 1.05)
            signal = amp * np.sin(2 * np.pi * freq * self.t + phase)
        elif class_id == 1:  # Voltage Sag
            amp = np.random.uniform(0.3, 0.88)
            t_start = np.random.choice([0.0, np.random.uniform(0.1, 0.3)])
            t_end = np.random.choice([1.0, np.random.uniform(0.7, 0.9)])
            envelope = np.ones_like(self.t)
            mask = (self.t >= t_start) & (self.t <= t_end)
            envelope[mask] = amp
            signal = envelope * np.sin(2 * np.pi * freq * self.t + phase)
        elif class_id == 2:  # Voltage Swell
            amp = np.random.uniform(1.15, 1.6)
            t_start = np.random.choice([0.0, np.random.uniform(0.1, 0.3)])
            t_end = np.random.choice([1.0, np.random.uniform(0.7, 0.9)])
            envelope = np.ones_like(self.t)
            mask = (self.t >= t_start) & (self.t <= t_end)
            envelope[mask] = amp
            signal = envelope * np.sin(2 * np.pi * freq * self.t + phase)
        elif class_id == 3:  # Harmonic Distortion
            amp = np.random.uniform(0.95, 1.05)
            h3_amp = np.random.uniform(0.10, 0.35)
            h5_amp = np.random.uniform(0.05, 0.20)
            signal = (
                amp * np.sin(2 * np.pi * freq * self.t + phase)
                + h3_amp * np.sin(2 * np.pi * 3 * freq * self.t + phase)
                + h5_amp * np.sin(2 * np.pi * 5 * freq * self.t + phase)
            )
        else:
            raise ValueError(f"Unknown class_id: {class_id}")

        signal = self._add_noise(signal, noise_snr)
        return signal, class_id

    def generate_balanced_dataset(self, samples_per_class: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Generates a balanced dataset of synthetic signals across all classes."""
        X = []
        y = []
        for class_id in range(4):
            for _ in range(samples_per_class):
                sig, label = self.generate_dataset_sample(class_id)
                X.append(sig)
                y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        indices = np.random.permutation(len(y))
        return X[indices], y[indices]

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Injects Additive White Gaussian Noise (AWGN) at specified SNR (dB)."""
        signal_power = np.mean(signal ** 2)
        if signal_power == 0:
            return signal
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise
