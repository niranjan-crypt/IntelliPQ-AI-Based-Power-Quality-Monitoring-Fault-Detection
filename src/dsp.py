"""
IntelliPQ - Digital Signal Processing (DSP) Module
Handles Fast Fourier Transforms, THD calculations, and statistical feature extraction.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.fft import fft, fftfreq


def compute_rms(signal: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) of voltage signal."""
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_peak(signal: np.ndarray) -> float:
    """Calculate Peak absolute voltage amplitude."""
    return float(np.max(np.abs(signal)))


def compute_crest_factor(signal: np.ndarray) -> float:
    """Calculate Crest Factor (Peak / RMS). For pure sine wave = sqrt(2) ~ 1.414."""
    rms = compute_rms(signal)
    if rms < 1e-9:
        return 0.0
    return float(compute_peak(signal) / rms)


def compute_fft_spectrum(signal: np.ndarray, sampling_rate: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes single-sided FFT amplitude spectrum.
    
    Returns:
        freqs: Frequency bins (Hz)
        magnitudes: Normalized amplitude spectrum
    """
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    # Single-sided normalized spectrum
    half_n = n // 2
    magnitudes = (fft_vals[:half_n] * 2.0) / n
    magnitudes[0] = magnitudes[0] / 2.0  # DC component
    freqs = fftfreq(n, d=1.0 / sampling_rate)[:half_n]
    return freqs, magnitudes
