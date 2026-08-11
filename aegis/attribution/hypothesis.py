"""
AEGIS Attribution Hypothesis Models.

Version: 0.13.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AttributionCategory(Enum):
    UNKNOWN = "unknown"
    ORGANIC = "organic"
    INDIVIDUAL = "individual"
    COORDINATED_NETWORK = "coordinated_network"
    AUTOMATED_ACTIVITY = "automated_activity"
    SYNTHETIC_CONTENT_OPERATION = "synthetic_content_operation"
    ORGANIZATIONAL = "organizational"


@dataclass
class AttributionHypothesis:
    """
    Evidence-supported attribution hypothesis.

    This object represents an analytical hypothesis,
    not a definitive claim of responsibility.
    """

    category: AttributionCategory

    confidence: float

    rationale: str

    supporting_evidence_ids: List[str] = field(
        default_factory=list
    )

    candidate_actor: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )