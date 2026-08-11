"""
Transparent cognitive threat risk scoring.

Version: 0.12.0
"""

from .levels import (
    RiskBand,
    ThreatSeverity,
)


def clamp_score(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def compute_risk_score(
    classification_confidence: float,
    propagation_score: float,
    synthetic_content_score: float,
    coordination_score: float,
    context_risk_score: float,
    source_credibility_score=None,
) -> float:
    """
    Compute a normalized Cognitive Threat Risk Score.

    Initial transparent weighting:

    classification confidence : 30%
    propagation risk          : 20%
    synthetic-content signal  : 15%
    coordination indicator    : 15%
    contextual risk           : 20%

    Low source credibility can optionally increase risk.
    """

    confidence = clamp_score(
        classification_confidence
    )

    propagation = clamp_score(
        propagation_score
    )

    synthetic = clamp_score(
        synthetic_content_score
    )

    coordination = clamp_score(
        coordination_score
    )

    context = clamp_score(
        context_risk_score
    )

    score = (
        0.30 * confidence
        + 0.20 * propagation
        + 0.15 * synthetic
        + 0.15 * coordination
        + 0.20 * context
    )

    if source_credibility_score is not None:

        credibility = clamp_score(
            source_credibility_score
        )

        credibility_risk = (
            1.0 - credibility
        )

        score = (
            0.90 * score
            + 0.10 * credibility_risk
        )

    return clamp_score(
        score
    )


def risk_band_from_score(
    score: float,
) -> RiskBand:

    score = clamp_score(
        score
    )

    if score < 0.20:
        return RiskBand.MINIMAL

    if score < 0.40:
        return RiskBand.LOW

    if score < 0.60:
        return RiskBand.MODERATE

    if score < 0.80:
        return RiskBand.HIGH

    return RiskBand.CRITICAL


def severity_from_score(
    score: float,
) -> ThreatSeverity:

    score = clamp_score(
        score
    )

    if score == 0.0:
        return ThreatSeverity.NONE

    if score < 0.30:
        return ThreatSeverity.LOW

    if score < 0.55:
        return ThreatSeverity.MODERATE

    if score < 0.80:
        return ThreatSeverity.HIGH

    return ThreatSeverity.CRITICAL