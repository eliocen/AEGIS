"""
AEGIS Cognitive Threat Intelligence levels.

Version: 0.12.0
"""

from enum import Enum


class ThreatSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskBand(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"