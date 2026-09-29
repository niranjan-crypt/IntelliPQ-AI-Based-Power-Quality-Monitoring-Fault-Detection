"""
IntelliPQ - Neuro-Symbolic Arbitration Engine
Reconciles statistical/deep learning predictions with deterministic physical IEEE safety rules.
"""

from typing import Dict, Any, List


class ArbitrationEngine:
    """Arbitrates between Deep Learning inferences and deterministic IEEE rule thresholds."""

    @classmethod
    def arbitrate(
        cls,
        rule_result: str,
        ai_result: str = None,
        ai_confidence: float = 0.0,
        amplitude: float = 1.0,
        thd: float = 0.0,
        rms: float = 1.0,
        harmonic_input: float = 0.0,
        frequency: float = 50.0,
        sag_limit: float = 0.90,
        swell_limit: float = 1.10,
        thd_limit: float = 5.0,
    ) -> Dict[str, Any]:
        if not ai_result or ai_result == "N/A":
            return {
                "final_decision": rule_result,
                "decision_method": "Rule-Based (AI Not Active)",
                "confidence_level": 80.0,
                "disagreement": False,
                "evidence_table": [],
            }

        # 1. Consensus
        if rule_result == ai_result:
            return {
                "final_decision": rule_result,
                "decision_method": "Consensus",
                "confidence_level": max(ai_confidence, 85.0),
                "disagreement": False,
                "evidence_table": [],
            }

        # 2. Critical Safety Threshold Overrides (Physical limits take priority)
        if thd >= thd_limit * 1.5:
            final_decision = "Harmonic Distortion"
            method = "Critical Threshold Override (THD Breach)"
            conf = 95.0
        elif amplitude < sag_limit * 0.85:
            final_decision = "Voltage Sag"
            method = "Critical Threshold Override (Severe Sag)"
            conf = 95.0
        elif amplitude > swell_limit * 1.15:
            final_decision = "Voltage Swell"
            method = "Critical Threshold Override (Severe Swell)"
            conf = 95.0
        else:
            final_decision = rule_result
            method = "Rule-Based Fallback"
            conf = 70.0

        return {
            "final_decision": final_decision,
            "decision_method": method,
            "confidence_level": conf,
            "disagreement": True,
            "evidence_table": [],
        }
