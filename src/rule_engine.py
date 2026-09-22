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
        """
        Classifies power disturbance based on deterministic IEEE thresholds.
        
        Args:
            amplitude: Peak or nominal per-unit voltage amplitude.
            thd: Total Harmonic Distortion percentage.
            
        Returns:
            Dictionary with diagnosis, status icon, severity, and trigger reason.
        """
        if thd >= cls.THD_LIMIT:
            classification = "Harmonic Distortion"
            reason = f"THD ({thd:.2f}%) exceeds IEEE 519 limit of {cls.THD_LIMIT}%"
            icon = "🔴"
            severity = "CRITICAL"
        elif amplitude < cls.SAG_LIMIT:
            classification = "Voltage Sag"
            reason = f"Amplitude ({amplitude:.3f} pu) dropped below IEEE 1159 limit of {cls.SAG_LIMIT} pu"
            icon = "🟡"
            severity = "WARNING"
        elif amplitude > cls.SWELL_LIMIT:
            classification = "Voltage Swell"
            reason = f"Amplitude ({amplitude:.3f} pu) exceeded IEEE 1159 limit of {cls.SWELL_LIMIT} pu"
            icon = "🟡"
            severity = "WARNING"
        else:
            classification = "Normal"
            reason = "All electrical parameters within nominal IEEE operational limits"
            icon = "🟢"
            severity = "NORMAL"

        return {
            "result": classification,
            "trigger_reason": reason,
            "status_icon": icon,
            "severity": severity,
            "thresholds": {
                "sag_limit": cls.SAG_LIMIT,
                "swell_limit": cls.SWELL_LIMIT,
                "thd_limit": cls.THD_LIMIT,
            },
        }
