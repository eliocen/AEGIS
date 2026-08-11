"""
Transparent decision-support policy for AEGIS.

Version: 0.15.0
"""

from typing import List

from aegis.classification import (
    IntegrityStatus,
)

from aegis.intelligence import (
    RiskBand,
    ThreatSeverity,
)

from .actions import (
    RecommendedAction,
    ResponsePriority,
)


def priority_from_assessment(
    integrity_status,
    severity,
    risk_band,
) -> ResponsePriority:
    """
    Convert AEGIS analytical findings into
    an operational review priority.
    """

    if (
        integrity_status
        == IntegrityStatus.TRUE
    ):
        return ResponsePriority.NONE

    if (
        severity
        == ThreatSeverity.CRITICAL
        or risk_band
        == RiskBand.CRITICAL
    ):
        return ResponsePriority.URGENT

    if (
        severity
        == ThreatSeverity.HIGH
        or risk_band
        == RiskBand.HIGH
    ):
        return ResponsePriority.HIGH

    if (
        severity
        == ThreatSeverity.MODERATE
        or risk_band
        == RiskBand.MODERATE
    ):
        return ResponsePriority.MODERATE

    return ResponsePriority.LOW


def actions_from_assessment(
    integrity_status,
    severity,
    risk_band,
    explanation_confidence: float,
    attribution_confidence: float = 0.0,
) -> List[RecommendedAction]:
    """
    Generate bounded recommended actions.

    AEGIS does not autonomously perform enforcement
    or other high-impact interventions.
    """

    if (
        integrity_status
        == IntegrityStatus.TRUE
    ):
        return [
            RecommendedAction.NO_ACTION
        ]

    actions = [
        RecommendedAction.MONITOR,
        RecommendedAction.PRESERVE_EVIDENCE,
    ]

    if explanation_confidence < 0.60:
        actions.extend(
            [
                RecommendedAction.VERIFY,
                RecommendedAction.HUMAN_REVIEW,
            ]
        )

    if (
        severity
        in {
            ThreatSeverity.MODERATE,
            ThreatSeverity.HIGH,
            ThreatSeverity.CRITICAL,
        }
    ):
        actions.append(
            RecommendedAction.FACT_CHECK
        )

    if (
        risk_band
        in {
            RiskBand.HIGH,
            RiskBand.CRITICAL,
        }
    ):
        actions.extend(
            [
                RecommendedAction.HUMAN_REVIEW,
                RecommendedAction.ESCALATE_FOR_REVIEW,
            ]
        )

    if (
        severity
        == ThreatSeverity.CRITICAL
    ):
        actions.extend(
            [
                RecommendedAction.PREPARE_ADVISORY,
                RecommendedAction.COORDINATE_RESPONSE,
            ]
        )

    if attribution_confidence > 0.0:
        actions.append(
            RecommendedAction.VERIFY
        )

    # Preserve order while eliminating duplicates.
    return list(
        dict.fromkeys(
            actions
        )
    )