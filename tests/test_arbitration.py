"""
Unit tests for Neuro-Symbolic Arbitration Engine.
"""

from src.arbitration_engine import ArbitrationEngine


def test_consensus():
    res = ArbitrationEngine.arbitrate(
        rule_result="Voltage Sag",
        ai_result="Voltage Sag",
        ai_confidence=88.5,
        amplitude=0.7,
        thd=1.2,
    )
    assert res["final_decision"] == "Voltage Sag"
    assert "Consensus" in res["decision_method"]
    assert res["disagreement"] is False


def test_critical_safety_override_severe_thd():
    # Even if AI mistakenly says Normal, a critical physical THD breach overrides it
    res = ArbitrationEngine.arbitrate(
        rule_result="Harmonic Distortion",
        ai_result="Normal",
        ai_confidence=91.0,
        amplitude=1.0,
        thd=8.2,  # > 5.0 * 1.5
    )
    assert res["final_decision"] == "Harmonic Distortion"
    assert "Critical Threshold Override" in res["decision_method"]


def test_ai_high_confidence_adoption():
    # Mild boundary case where AI has high confidence (>90%) and no critical threshold breached
    res = ArbitrationEngine.arbitrate(
        rule_result="Normal",
        ai_result="Voltage Sag",
        ai_confidence=94.0,
        amplitude=0.91,  # Near boundary
        thd=2.0,
    )
    assert res["final_decision"] == "Voltage Sag"
    assert "AI High Confidence" in res["decision_method"]


def test_fallback_when_ai_missing():
    res = ArbitrationEngine.arbitrate(
        rule_result="Voltage Swell",
        ai_result=None,
        amplitude=1.25,
        thd=1.0,
    )
    assert res["final_decision"] == "Voltage Swell"
    assert "Rule-Based" in res["decision_method"]
