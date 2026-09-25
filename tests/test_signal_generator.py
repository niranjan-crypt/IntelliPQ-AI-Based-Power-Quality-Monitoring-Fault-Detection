import pytest
import numpy as np
from src.signal_generator import SignalGenerator


def test_generator_output_length():
    gen = SignalGenerator(sampling_rate=1000, duration=1.0)
    sig = gen.generate_normal()
    assert len(sig) == 1000


def test_sag_amplitude():
    gen = SignalGenerator(sampling_rate=1000, duration=1.0)
    sig = gen.generate_voltage_sag(amplitude=0.6, start_time=0.0, end_time=1.0)
    import numpy as np
    assert pytest.approx(np.max(sig), abs=0.05) == 0.6


def test_swell_amplitude():
    gen = SignalGenerator(sampling_rate=1000, duration=1.0)
    sig = gen.generate_voltage_swell(amplitude=1.4, start_time=0.0, end_time=1.0)
    import numpy as np
    assert pytest.approx(np.max(sig), abs=0.05) == 1.4


def test_balanced_dataset_generation():
    gen = SignalGenerator(sampling_rate=1000, duration=1.0)
    X, y = gen.generate_balanced_dataset(samples_per_class=10)
    assert len(X) == 40
    assert len(y) == 40
    import numpy as np
    assert set(np.unique(y)) == {0, 1, 2, 3}
