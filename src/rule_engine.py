"""
IntelliPQ - Rule-Based Detection Engine
Compliant with IEEE 1159 (Power Quality Monitoring) and IEEE 519 (Harmonic Control).
"""

from typing import Dict, Any


class RuleEngine:
    """Deterministic rule-based expert system for power quality fault classification."""

    SAG_LIMIT = 0.90       # IEEE 1159: Nominal voltage drop < 0.9 pu
    SWELL_LIMIT = 1.10     # IEEE 1159: Nominal voltage rise > 1.1 pu
    THD_LIMIT = 5.0        # IEEE 519: Total Harmonic Distortion limit 5.0%

    @classmethod
    def evaluate(cls, amplitude: float, thd: float) -> Dict[str, Any]:
        """Classifies power disturbance based on deterministic IEEE thresholds."""
        if thd >= cls.THD_LIMIT:
            classification = "Harmonic Distortion"
        elif amplitude < cls.SAG_LIMIT:
            classification = "Voltage Sag"
        elif amplitude > cls.SWELL_LIMIT:
            classification = "Voltage Swell"
        else:
            classification = "Normal"

        return {"result": classification}
