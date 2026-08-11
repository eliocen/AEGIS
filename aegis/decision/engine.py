"""
AEGIS Decision Support & Response Engine.

Version: 0.15.0
"""

from aegis.classification import (
    IntegrityStatus,
)

from .actions import (
    RecommendedAction,
)

from .input import (
    DecisionSupportInput,
)

from .output import (
    DecisionRecommendation,
)

from .policy import (
    actions_from_assessment,
    priority_from_assessment,
)


class DecisionSupportEngine:
    """
    Converts AEGIS analytical outputs into bounded,
    explainable recommendations for human analysts.
    """

    def recommend(
        self,
        data: DecisionSupportInput,
    ) -> DecisionRecommendation:

        if not isinstance(
            data,
            DecisionSupportInput,
        ):
            raise TypeError(
                "DecisionSupportEngine expects "
                "DecisionSupportInput."
            )

        threat = (
            data.threat_assessment
        )

        explanation = (
            data.explanation
        )

        if (
            threat.sample_id
            != explanation.sample_id
        ):
            raise ValueError(
                "Threat assessment and explanation "
                "sample_id values must match."
            )

        attribution_confidence = 0.0

        if data.attribution is not None:

            if (
                data.attribution.sample_id
                != threat.sample_id
            ):
                raise ValueError(
                    "Attribution assessment sample_id "
                    "must match threat assessment."
                )

            attribution_confidence = (
                data.attribution
                .overall_confidence
            )

        priority = (
            priority_from_assessment(
                integrity_status=(
                    threat.integrity_status
                ),

                severity=(
                    threat.severity
                ),

                risk_band=(
                    threat.risk_band
                ),
            )
        )

        actions = (
            actions_from_assessment(
                integrity_status=(
                    threat.integrity_status
                ),

                severity=(
                    threat.severity
                ),

                risk_band=(
                    threat.risk_band
                ),

                explanation_confidence=(
                    explanation
                    .explanation_confidence
                ),

                attribution_confidence=(
                    attribution_confidence
                ),
            )
        )

        rationale = []

        if (
            threat.integrity_status
            == IntegrityStatus.TRUE
        ):

            rationale.append(
                "The sample was assessed as "
                "true information."
            )

        else:

            rationale.append(
                "The sample was assessed as harmful."
            )

            if threat.threat_type is not None:

                rationale.append(
                    "The detected Cognitive Cyber "
                    "Threat subtype is "
                    f"{threat.threat_type.value}."
                )

            rationale.append(
                "The Cognitive Threat Intelligence "
                "risk score is "
                f"{threat.risk_score:.2f}."
            )

            rationale.append(
                "The assessed severity is "
                f"{threat.severity.value}."
            )

            rationale.append(
                "The operational risk band is "
                f"{threat.risk_band.value}."
            )

        if (
            explanation.uncertainty
            > 0.40
        ):

            rationale.append(
                "Elevated analytical uncertainty "
                "requires additional verification."
            )

        if attribution_confidence > 0.0:

            rationale.append(
                "Attribution evidence exists but "
                "remains hypothesis-based and requires "
                "human validation."
            )

        human_review_required = any(
            action
            in {
                RecommendedAction.HUMAN_REVIEW,
                RecommendedAction.ESCALATE_FOR_REVIEW,
                RecommendedAction.PREPARE_ADVISORY,
                RecommendedAction.COORDINATE_RESPONSE,
            }
            for action in actions
        )

        if (
            threat.integrity_status
            == IntegrityStatus.TRUE
        ):
            human_review_required = False

        confidence_values = [
            threat.classification_confidence,
            explanation.explanation_confidence,
        ]

        if attribution_confidence > 0.0:
            confidence_values.append(
                attribution_confidence
            )

        confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        return DecisionRecommendation(
            sample_id=(
                threat.sample_id
            ),

            priority=priority,

            actions=actions,

            rationale=rationale,

            human_review_required=(
                human_review_required
            ),

            automation_permitted=False,

            confidence=confidence,

            metadata={
                "decision_mode": (
                    "human_supervised"
                ),

                "policy": (
                    "transparent_rule_based_v1"
                ),

                "autonomous_enforcement": False,

                "risk_score": (
                    threat.risk_score
                ),

                "severity": (
                    threat.severity.value
                ),

                "risk_band": (
                    threat.risk_band.value
                ),
            },
        )