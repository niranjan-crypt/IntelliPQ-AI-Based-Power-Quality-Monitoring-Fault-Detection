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
        # 3. High AI Confidence (>90%)
        elif ai_confidence > 90.0:
            final_decision = ai_result
            method = "AI High Confidence (>90%)"
            conf = ai_confidence
        # 4. Moderate AI Confidence (70% - 90%) with Parameter Verification
        elif ai_confidence > 70.0:
            verified = False
            if ai_result == "Harmonic Distortion" and (thd >= thd_limit * 0.8 or harmonic_input > 0.15):
                verified = True
            elif ai_result == "Voltage Sag" and (amplitude < sag_limit * 1.05 or rms < 0.9):
                verified = True
            elif ai_result == "Voltage Swell" and (amplitude > swell_limit * 0.95 or rms > 1.1):
                verified = True
            elif ai_result == "Normal" and (sag_limit * 0.95 <= amplitude <= swell_limit * 1.05 and thd < thd_limit * 0.8):
                verified = True

            if verified:
                final_decision = ai_result
                method = "AI Verified by Physical Parameters"
                conf = ai_confidence
            else:
                final_decision = rule_result
                method = "Rule-Based (AI Not Verified)"
                conf = 75.0
        # 5. Low AI Confidence (<70%) -> Fallback to Rules
        else:
            final_decision = rule_result
            method = "Rule-Based (Low AI Confidence <70%)"
            conf = 70.0

        evidence_table = cls._build_evidence(amplitude, thd, rms, harmonic_input, frequency, sag_limit, swell_limit, thd_limit)

        return {
            "final_decision": final_decision,
            "decision_method": method,
            "confidence_level": conf,
            "disagreement": True,
            "evidence_table": evidence_table,
        }

    @staticmethod
    def _build_evidence(amplitude, thd, rms, harmonic_input, frequency, sag_limit, swell_limit, thd_limit) -> List[Dict[str, str]]:
        rows = [
            {
                "Parameter": "Amplitude",
                "Value": f"{amplitude:.3f} pu",
                "Normal Range": f"{sag_limit:.2f}-{swell_limit:.2f} pu",
                "Status": "Normal" if sag_limit <= amplitude <= swell_limit else "Abnormal",
                "Supports": "Voltage Sag" if amplitude < sag_limit else ("Voltage Swell" if amplitude > swell_limit else "Normal"),
            },
            {
                "Parameter": "THD",
                "Value": f"{thd:.2f}%",
                "Normal Range": f"<{thd_limit:.1f}%",
                "Status": "Normal" if thd < thd_limit else "Abnormal",
                "Supports": "Harmonic Distortion" if thd >= thd_limit else "Normal",
            },
            {
                "Parameter": "RMS Voltage",
                "Value": f"{rms:.3f} pu",
                "Normal Range": "0.90-1.10 pu",
                "Status": "Normal" if 0.9 <= rms <= 1.1 else "Abnormal",
                "Supports": "Voltage Sag" if rms < 0.9 else ("Voltage Swell" if rms > 1.1 else "Normal"),
            },
            {
                "Parameter": "Harmonic Input",
                "Value": f"{harmonic_input:.2f} pu",
                "Normal Range": "<0.10 pu",
                "Status": "Normal" if harmonic_input < 0.1 else "Abnormal",
                "Supports": "Harmonic Distortion" if harmonic_input >= 0.1 else "Normal",
            },
            {
                "Parameter": "Frequency",
                "Value": f"{frequency:.1f} Hz",
                "Normal Range": "49.0-51.0 Hz",
                "Status": "Normal" if 49.0 <= frequency <= 51.0 else "Deviation",
                "Supports": "Normal" if 49.0 <= frequency <= 51.0 else "Distortion/Issue",
            },
        ]
        return rows
