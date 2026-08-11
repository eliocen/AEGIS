"""
AEGIS Explainability Outputs.

Version: 0.14.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .factors import (
    ExplanationFactor,
)


@dataclass
class ExplanationReport:
    """
    Structured analyst-facing explanation of
    an AEGIS assessment.
    """

    sample_id: str

    summary: str

    factors: List[
        ExplanationFactor
    ]

    explanation_confidence: float

    uncertainty: float

    evidence_ids: List[str] = field(
        default_factory=list
    )

    caveats: List[str] = field(
        default_factory=list
    )

    reasoning_trace: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )