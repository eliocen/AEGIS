"""
AEGIS Hierarchical Uncertainty Evaluation.

Version: 0.23.0
"""

from dataclasses import (
    asdict,
    dataclass,
)

from .uncertainty import (
    prediction_uncertainty,
)


@dataclass
class HierarchicalUncertaintyResult:
    """
    Stage-1 and Stage-2 uncertainty for
    one AEGIS prediction.
    """

    integrity_confidence: float

    integrity_uncertainty: float

    threat_confidence: float | None

    threat_uncertainty: float | None

    overall_confidence: float

    overall_uncertainty: float

    uncertainty_level: str

    def as_dict(self):

        return asdict(
            self
        )


def hierarchical_uncertainty(
    integrity_probabilities,
    threat_probabilities=None,
):
    """
    Combine uncertainty from AEGIS Stage 1
    and Stage 2.

    Overall confidence uses the conservative
    minimum when threat classification exists.
    """

    integrity = (
        prediction_uncertainty(
            integrity_probabilities
        )
    )

    if threat_probabilities is not None:

        threat = (
            prediction_uncertainty(
                threat_probabilities
            )
        )

        overall_confidence = min(
            integrity.confidence,
            threat.confidence,
        )

        overall_uncertainty = max(
            integrity.uncertainty,
            threat.uncertainty,
        )

        threat_confidence = (
            threat.confidence
        )

        threat_uncertainty = (
            threat.uncertainty
        )

    else:

        overall_confidence = (
            integrity.confidence
        )

        overall_uncertainty = (
            integrity.uncertainty
        )

        threat_confidence = None
        threat_uncertainty = None

    if overall_uncertainty < 0.33:

        uncertainty_level = "low"

    elif overall_uncertainty < 0.66:

        uncertainty_level = "moderate"

    else:

        uncertainty_level = "high"

    return (
        HierarchicalUncertaintyResult(
            integrity_confidence=(
                integrity.confidence
            ),

            integrity_uncertainty=(
                integrity.uncertainty
            ),

            threat_confidence=(
                threat_confidence
            ),

            threat_uncertainty=(
                threat_uncertainty
            ),

            overall_confidence=float(
                overall_confidence
            ),

            overall_uncertainty=float(
                overall_uncertainty
            ),

            uncertainty_level=(
                uncertainty_level
            ),
        )
    )