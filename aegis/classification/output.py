"""
AEGIS hierarchical classification outputs.

Version: 0.10.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .labels import (
    CognitiveThreatType,
    IntegrityStatus,
)


@dataclass
class HierarchicalClassificationOutput:
    """
    Final output of the AEGIS hierarchical
    Information Integrity classifier.
    """

    sample_id: str

    integrity_status: IntegrityStatus
    integrity_confidence: float

    threat_type: Optional[CognitiveThreatType] = None
    threat_confidence: Optional[float] = None

    integrity_probabilities: Optional[Any] = None
    threat_probabilities: Optional[Any] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_harmful(self) -> bool:
        return (
            self.integrity_status
            == IntegrityStatus.HARMFUL
        )

    @property
    def is_true(self) -> bool:
        return (
            self.integrity_status
            == IntegrityStatus.TRUE
        )