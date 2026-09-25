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
