"""
AEGIS Threat Attribution Engine.

Version: 0.13.0
"""

from typing import List, Optional

from aegis.intelligence import (
    CognitiveThreatAssessment,
)

from .confidence import (
    aggregate_evidence_confidence,
)

from .evidence import (
    AttributionEvidence,
    EvidenceType,
)

from .hypothesis import (
    AttributionCategory,
    AttributionHypothesis,
)

from .output import (
    ThreatAttributionAssessment,
)

from .source import (
    SourceProfile,
)


class ThreatAttributionEngine:
    """
    Produces cautious evidence-based attribution
    hypotheses from observed indicators.

    The engine does not assert actor responsibility.
    """

    def assess(
        self,
        threat_assessment:
        CognitiveThreatAssessment,

        source_profile:
        Optional[SourceProfile] = None,

        evidence:
        Optional[
            List[AttributionEvidence]
        ] = None,
    ) -> ThreatAttributionAssessment:

        if not isinstance(
            threat_assessment,
            CognitiveThreatAssessment,
        ):
            raise TypeError(
                "ThreatAttributionEngine expects "
                "CognitiveThreatAssessment."
            )

        evidence = list(
            evidence or []
        )

        hypotheses = []

        coordination_evidence = [
            item
            for item in evidence
            if item.evidence_type
            == EvidenceType.COORDINATION_SIGNAL
        ]

        synthetic_evidence = [
            item
            for item in evidence
            if item.evidence_type
            == EvidenceType.SYNTHETIC_CONTENT
        ]

        provenance_evidence = [
            item
            for item in evidence
            if item.evidence_type
            == EvidenceType.SOURCE_PROVENANCE
        ]

        if coordination_evidence:

            confidence = (
                aggregate_evidence_confidence(
                    coordination_evidence
                )
            )

            hypotheses.append(
                AttributionHypothesis(
                    category=(
                        AttributionCategory
                        .COORDINATED_NETWORK
                    ),

                    confidence=confidence,

                    rationale=(
                        "Observed coordination indicators "
                        "support a coordinated-network "
                        "hypothesis."
                    ),

                    supporting_evidence_ids=[
                        item.evidence_id
                        for item
                        in coordination_evidence
                    ],
                )
            )

        if synthetic_evidence:

            confidence = (
                aggregate_evidence_confidence(
                    synthetic_evidence
                )
            )

            hypotheses.append(
                AttributionHypothesis(
                    category=(
                        AttributionCategory
                        .SYNTHETIC_CONTENT_OPERATION
                    ),

                    confidence=confidence,

                    rationale=(
                        "Synthetic-content indicators "
                        "support a synthetic-content "
                        "operation hypothesis."
                    ),

                    supporting_evidence_ids=[
                        item.evidence_id
                        for item
                        in synthetic_evidence
                    ],
                )
            )

        if provenance_evidence:

            confidence = (
                aggregate_evidence_confidence(
                    provenance_evidence
                )
            )

            hypotheses.append(
                AttributionHypothesis(
                    category=(
                        AttributionCategory
                        .ORGANIZATIONAL
                    ),

                    confidence=confidence,

                    rationale=(
                        "Source provenance evidence "
                        "supports an organizational-source "
                        "hypothesis."
                    ),

                    supporting_evidence_ids=[
                        item.evidence_id
                        for item
                        in provenance_evidence
                    ],
                )
            )

        if not hypotheses:

            hypotheses.append(
                AttributionHypothesis(
                    category=(
                        AttributionCategory.UNKNOWN
                    ),

                    confidence=0.0,

                    rationale=(
                        "Insufficient evidence for a "
                        "specific attribution hypothesis."
                    ),
                )
            )

        primary_hypothesis = max(
            hypotheses,
            key=lambda item: item.confidence,
        )

        overall_confidence = (
            aggregate_evidence_confidence(
                evidence
            )
        )

        return ThreatAttributionAssessment(
            sample_id=(
                threat_assessment.sample_id
            ),

            source_profile=(
                source_profile
            ),

            evidence=evidence,

            hypotheses=hypotheses,

            overall_confidence=(
                overall_confidence
            ),

            primary_hypothesis=(
                primary_hypothesis
            ),

            metadata={
                "attribution_mode": (
                    "evidence_based_hypothesis"
                ),

                "definitive_attribution": False,

                "threat_risk_score": (
                    threat_assessment.risk_score
                ),

                "threat_severity": (
                    threat_assessment
                    .severity
                    .value
                ),
            },
        )