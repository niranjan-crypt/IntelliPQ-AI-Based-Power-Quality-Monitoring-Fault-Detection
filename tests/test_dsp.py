"""
Unit tests for Digital Signal Processing (DSP) functions.
"""

import pytest
import numpy as np
from src.dsp import (
    compute_rms,
    compute_peak,
    compute_crest_factor,
    compute_thd,
    compute_fft_spectrum,
    extract_features,
)


@pytest.fixture
def pure_sine():
    t = np.linspace(0, 1, 1000, endpoint=False)
    return np.sin(2 * np.pi * 50 * t)


@pytest.fixture
def distorted_signal():
    t = np.linspace(0, 1, 1000, endpoint=False)
    # Fundamental + 20% 3rd Harmonic
    return np.sin(2 * np.pi * 50 * t) + 0.20 * np.sin(2 * np.pi * 150 * t)


def test_pure_sine_rms(pure_sine):
    rms = compute_rms(pure_sine)
    expected = 1.0 / np.sqrt(2)
    assert pytest.approx(rms, rel=1e-2) == expected


def test_pure_sine_peak(pure_sine):
    peak = compute_peak(pure_sine)
    assert pytest.approx(peak, abs=1e-2) == 1.0


def test_pure_sine_crest_factor(pure_sine):
    cf = compute_crest_factor(pure_sine)
    assert pytest.approx(cf, rel=1e-2) == np.sqrt(2)


def test_pure_sine_thd_near_zero(pure_sine):
    thd = compute_thd(pure_sine, sampling_rate=1000, nominal_freq=50.0)
    assert thd < 1.0  # Ideal sine has near-zero harmonic content


def test_distorted_signal_thd(distorted_signal):
    thd = compute_thd(distorted_signal, sampling_rate=1000, nominal_freq=50.0)
    # 20% harmonic should yield ~20% THD
    assert 18.0 <= thd <= 22.0


def test_fft_spectrum_shape(pure_sine):
    freqs, mags = compute_fft_spectrum(pure_sine, sampling_rate=1000)
    assert len(freqs) == 500
    assert len(mags) == 500
    # Peak should occur at 50 Hz
    peak_freq = freqs[np.argmax(mags)]
    assert peak_freq == 50.0


def test_extract_features_keys(pure_sine):
    features = extract_features(pure_sine)
    expected_keys = {"rms", "peak", "crest_factor", "thd", "mean", "std"}
    assert set(features.keys()) == expected_keys
