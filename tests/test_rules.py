"""
Unit tests for deterministic IEEE rule-based engine.
"""

from src.rule_engine import RuleEngine


def test_normal_operation():
    res = RuleEngine.evaluate(amplitude=1.0, thd=2.0)
    assert res["result"] == "Normal"
    assert res["severity"] == "NORMAL"


def test_voltage_sag():
    res = RuleEngine.evaluate(amplitude=0.75, thd=1.5)
    assert res["result"] == "Voltage Sag"
    assert res["severity"] == "WARNING"


def test_voltage_swell():
    res = RuleEngine.evaluate(amplitude=1.25, thd=2.0)
    assert res["result"] == "Voltage Swell"
    assert res["severity"] == "WARNING"


def test_harmonic_distortion():
    res = RuleEngine.evaluate(amplitude=1.0, thd=7.5)
    assert res["result"] == "Harmonic Distortion"
    assert res["severity"] == "CRITICAL"


def test_harmonic_precedence_over_sag():
    # If both THD >= 5% and amplitude < 0.9, IEEE harmonic distortion takes precedence
    res = RuleEngine.evaluate(amplitude=0.8, thd=8.0)
    assert res["result"] == "Harmonic Distortion"
