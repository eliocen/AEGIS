"""
Inputs for the AEGIS Decision Support Layer.

Version: 0.15.0
"""

from dataclasses import dataclass
from typing import Optional

from aegis.attribution import (
    ThreatAttributionAssessment,
)

from aegis.explainability import (
    ExplanationReport,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
)


@dataclass
class DecisionSupportInput:
    """
    Consolidated analytical input for bounded
    decision-support recommendation generation.
    """

    threat_assessment: CognitiveThreatAssessment

    explanation: ExplanationReport

    attribution: Optional[
            ThreatAttributionAssessment
        ] = None