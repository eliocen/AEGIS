"""
AEGIS Attribution Confidence Aggregation.

Version: 0.13.0
"""

from typing import Iterable

from .evidence import AttributionEvidence


def clamp_confidence(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def aggregate_evidence_confidence(
    evidence: Iterable[AttributionEvidence],
) -> float:
    """
    Aggregate evidence confidence using a simple
    weighted-average baseline.

    Future versions may replace this with Bayesian
    or learned evidence fusion.
    """

    evidence = list(evidence)

    if not evidence:
        return 0.0

    values = [
        clamp_confidence(
            item.confidence
        )
        for item in evidence
    ]

    return sum(values) / len(values)