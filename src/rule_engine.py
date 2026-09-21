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
