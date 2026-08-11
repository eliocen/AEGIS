"""
AEGIS Explainability Factors.

Version: 0.14.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ExplanationCategory(Enum):
    INTEGRITY = "integrity"
    THREAT_CLASS = "threat_class"
    PROPAGATION = "propagation"
    SYNTHETIC_CONTENT = "synthetic_content"
    COORDINATION = "coordination"
    CONTEXT = "context"
    ATTRIBUTION = "attribution"
    SOURCE = "source"
    UNCERTAINTY = "uncertainty"


class FactorDirection(Enum):
    SUPPORTING = "supporting"
    MITIGATING = "mitigating"
    CONTEXTUAL = "contextual"


def clamp_strength(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


@dataclass
class ExplanationFactor:
    """
    One structured factor contributing to an
    AEGIS explanation.
    """

    factor_id: str

    category: ExplanationCategory

    description: str

    strength: float

    direction: FactorDirection

    source: Optional[str] = None

    evidence_ids: List[str] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.strength = clamp_strength(
            self.strength
        )