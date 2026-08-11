"""
AEGIS Decision Support Action Taxonomy.

Version: 0.15.0
"""

from enum import Enum


class ResponsePriority(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    URGENT = "urgent"


class RecommendedAction(Enum):
    """
    Bounded analyst-facing recommendations.

    These actions support review and coordination.
    They do not autonomously execute high-impact
    interventions.
    """

    NO_ACTION = "no_action"

    MONITOR = "monitor"

    VERIFY = "verify"

    FACT_CHECK = "fact_check"

    PRESERVE_EVIDENCE = "preserve_evidence"

    HUMAN_REVIEW = "human_review"

    ESCALATE_FOR_REVIEW = "escalate_for_review"

    PREPARE_ADVISORY = "prepare_advisory"

    COORDINATE_RESPONSE = "coordinate_response"