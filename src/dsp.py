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


def compute_thd(signal: np.ndarray, sampling_rate: int = 1000, nominal_freq: float = 50.0) -> float:
    """
    Computes Total Harmonic Distortion (THD) as a percentage according to IEEE 519 standard:
    THD (%) = (sqrt(sum(V_h^2)) / V_1) * 100
    where V_1 is fundamental component, V_h are harmonic components (h >= 2).
    """
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    fft_vals = fft_vals[:n // 2]
    
    # Frequency resolution df = sampling_rate / n
    df = sampling_rate / n
    fundamental_bin = int(round(nominal_freq / df))
    
    if fundamental_bin >= len(fft_vals) or fundamental_bin <= 0:
        fundamental_bin = 1
        
    fundamental_mag = fft_vals[fundamental_bin]
    if fundamental_mag < 1e-9:
        return 0.0

    # Search for harmonic bins up to 20th harmonic or Nyquist limit
    max_harmonic_order = min(20, int((sampling_rate / 2) / nominal_freq))
    harmonics_squared = 0.0

    for h in range(2, max_harmonic_order + 1):
        target_bin = int(round(h * nominal_freq / df))
        if target_bin < len(fft_vals):
            # Sum energy in small window around harmonic peak to account for spectral leakage
            window = range(max(0, target_bin - 1), min(len(fft_vals), target_bin + 2))
            harmonics_squared += sum(fft_vals[k] ** 2 for k in window)

    thd_percentage = (np.sqrt(harmonics_squared) / fundamental_mag) * 100.0
    return float(thd_percentage)
