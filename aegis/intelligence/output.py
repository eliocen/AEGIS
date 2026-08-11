"""
AEGIS Cognitive Threat Intelligence outputs.

Version: 0.12.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from aegis.classification import (
    CognitiveThreatType,
    IntegrityStatus,
)

from .levels import (
    RiskBand,
    ThreatSeverity,
)


@dataclass
class CognitiveThreatAssessment:
    """
    Structured operational assessment produced by AEGIS.
    """

    sample_id: str

    integrity_status: IntegrityStatus

    threat_type: Optional[CognitiveThreatType]

    severity: ThreatSeverity

    risk_score: float

    risk_band: RiskBand

    classification_confidence: float

    propagation_risk: float
    synthetic_content_indicator: float
    coordination_indicator: float
    context_risk: float

    source_credibility: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )