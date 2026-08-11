"""
AEGIS Threat Attribution Outputs.

Version: 0.13.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence import AttributionEvidence
from .hypothesis import AttributionHypothesis
from .source import SourceProfile


@dataclass
class ThreatAttributionAssessment:
    """
    Structured attribution assessment produced by AEGIS.
    """

    sample_id: str

    source_profile: Optional[SourceProfile]

    evidence: List[AttributionEvidence]

    hypotheses: List[AttributionHypothesis]

    overall_confidence: float

    primary_hypothesis: Optional[
        AttributionHypothesis
    ] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )