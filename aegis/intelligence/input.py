"""
Inputs for the AEGIS Cognitive Threat Intelligence Engine.

Version: 0.12.0
"""

from dataclasses import dataclass
from typing import Optional

from aegis.classification import (
    HierarchicalClassificationOutput,
)


@dataclass
class ThreatIntelligenceInput:
    """
    Structured input for cognitive cyber threat assessment.
    """

    classification: HierarchicalClassificationOutput

    propagation_score: float = 0.0
    synthetic_content_score: float = 0.0
    coordination_score: float = 0.0
    context_risk_score: float = 0.0

    source_credibility_score: Optional[float] = None