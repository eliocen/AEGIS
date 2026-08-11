"""
AEGIS Attribution Evidence Models.

Version: 0.13.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EvidenceType(Enum):
    SOURCE_PROVENANCE = "source_provenance"
    TEMPORAL_PATTERN = "temporal_pattern"
    CONTENT_SIMILARITY = "content_similarity"
    COORDINATION_SIGNAL = "coordination_signal"
    NETWORK_RELATIONSHIP = "network_relationship"
    PLATFORM_METADATA = "platform_metadata"
    CONTEXTUAL_INDICATOR = "contextual_indicator"
    SYNTHETIC_CONTENT = "synthetic_content"
    EXTERNAL_VERIFICATION = "external_verification"


@dataclass
class AttributionEvidence:
    """
    One structured evidence item used during
    threat attribution analysis.
    """

    evidence_id: str

    evidence_type: EvidenceType

    description: str

    confidence: float

    source: Optional[str] = None

    value: Optional[Any] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )