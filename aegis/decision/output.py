"""
AEGIS Decision Support Outputs.

Version: 0.15.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .actions import (
    RecommendedAction,
    ResponsePriority,
)


@dataclass
class DecisionRecommendation:
    """
    Structured decision-support recommendation.

    AEGIS recommendations are advisory and require
    appropriate human oversight.
    """

    sample_id: str

    priority: ResponsePriority

    actions: List[
        RecommendedAction
    ]

    rationale: List[str]

    human_review_required: bool

    automation_permitted: bool = False

    confidence: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )