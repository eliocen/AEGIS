"""
AEGIS Cognitive Threat Intelligence Engine.

Version: 0.12.0
"""

from aegis.classification import (
    IntegrityStatus,
)

from .input import (
    ThreatIntelligenceInput,
)

from .output import (
    CognitiveThreatAssessment,
)

from .scoring import (
    compute_risk_score,
    risk_band_from_score,
    severity_from_score,
)


class CognitiveThreatIntelligenceEngine:
    """
    Converts hierarchical classification results and
    supporting cyber-threat indicators into a structured
    Cognitive Threat Assessment.
    """

    def assess(
        self,
        data: ThreatIntelligenceInput,
    ) -> CognitiveThreatAssessment:

        if not isinstance(
            data,
            ThreatIntelligenceInput,
        ):
            raise TypeError(
                "CognitiveThreatIntelligenceEngine "
                "expects ThreatIntelligenceInput."
            )

        classification = (
            data.classification
        )

        if (
            classification.integrity_status
            == IntegrityStatus.TRUE
        ):
            return CognitiveThreatAssessment(
                sample_id=(
                    classification.sample_id
                ),

                integrity_status=(
                    classification.integrity_status
                ),

                threat_type=None,

                severity=severity_from_score(
                    0.0
                ),

                risk_score=0.0,

                risk_band=risk_band_from_score(
                    0.0
                ),

                classification_confidence=(
                    classification
                    .integrity_confidence
                ),

                propagation_risk=0.0,
                synthetic_content_indicator=0.0,
                coordination_indicator=0.0,
                context_risk=0.0,

                source_credibility=(
                    data.source_credibility_score
                ),

                metadata={
                    "assessment": (
                        "no_cognitive_threat_detected"
                    )
                },
            )

        classification_confidence = (
            classification
            .threat_confidence
        )

        if classification_confidence is None:
            classification_confidence = (
                classification
                .integrity_confidence
            )

        risk_score = compute_risk_score(
            classification_confidence=(
                classification_confidence
            ),

            propagation_score=(
                data.propagation_score
            ),

            synthetic_content_score=(
                data.synthetic_content_score
            ),

            coordination_score=(
                data.coordination_score
            ),

            context_risk_score=(
                data.context_risk_score
            ),

            source_credibility_score=(
                data.source_credibility_score
            ),
        )

        return CognitiveThreatAssessment(
            sample_id=(
                classification.sample_id
            ),

            integrity_status=(
                classification.integrity_status
            ),

            threat_type=(
                classification.threat_type
            ),

            severity=severity_from_score(
                risk_score
            ),

            risk_score=(
                risk_score
            ),

            risk_band=risk_band_from_score(
                risk_score
            ),

            classification_confidence=(
                classification_confidence
            ),

            propagation_risk=(
                data.propagation_score
            ),

            synthetic_content_indicator=(
                data.synthetic_content_score
            ),

            coordination_indicator=(
                data.coordination_score
            ),

            context_risk=(
                data.context_risk_score
            ),

            source_credibility=(
                data.source_credibility_score
            ),

            metadata={
                "assessment": (
                    "cognitive_cyber_threat"
                ),

                "scoring_model": (
                    "transparent_weighted_v1"
                ),
            },
        )