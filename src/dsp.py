"""
IntelliPQ - Digital Signal Processing (DSP) Module
Handles Fast Fourier Transforms, THD calculations, and statistical feature extraction.
"""

from typing import Dict, Tuple
import numpy as np


def compute_rms(signal: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) of voltage signal."""
    return float(np.sqrt(np.mean(signal ** 2)))
