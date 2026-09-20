"""
IntelliPQ - Digital Signal Processing (DSP) Module
Handles Fast Fourier Transforms, THD calculations, and statistical feature extraction.
"""

from typing import Dict, Tuple
import numpy as np


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
