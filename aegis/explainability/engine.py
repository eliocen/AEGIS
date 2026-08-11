"""
AEGIS Explainable AI & Evidence-Based Reasoning Engine.

Version: 0.14.0
"""

from typing import List

from aegis.classification import (
    IntegrityStatus,
)

from .factors import (
    ExplanationCategory,
    ExplanationFactor,
    FactorDirection,
    clamp_strength,
)

from .input import (
    ExplainabilityInput,
)

from .output import (
    ExplanationReport,
)


class ExplainabilityEngine:
    """
    Generates transparent, structured explanations
    from AEGIS analytical outputs.

    This version provides output-level and evidence-level
    explainability rather than neural feature attribution.
    """

    def explain(
        self,
        data: ExplainabilityInput,
    ) -> ExplanationReport:

        if not isinstance(
            data,
            ExplainabilityInput,
        ):
            raise TypeError(
                "ExplainabilityEngine expects "
                "ExplainabilityInput."
            )

        classification = (
            data.classification
        )

        factors: List[
            ExplanationFactor
        ] = []

        caveats = []
        reasoning_trace = []
        evidence_ids = []

        # ---------------------------------
        # Stage 1: Information integrity
        # ---------------------------------

        integrity_confidence = (
            classification
            .integrity_confidence
        )

        if (
            classification.integrity_status
            == IntegrityStatus.TRUE
        ):
            factors.append(
                ExplanationFactor(
                    factor_id="INTEGRITY-TRUE",

                    category=(
                        ExplanationCategory
                        .INTEGRITY
                    ),

                    description=(
                        "The Information Integrity "
                        "classifier assessed the content "
                        "as true information."
                    ),

                    strength=(
                        integrity_confidence
                    ),

                    direction=(
                        FactorDirection
                        .MITIGATING
                    ),

                    source=(
                        "hierarchical_classifier"
                    ),
                )
            )

            reasoning_trace.append(
                "Stage 1 classified the sample "
                "as TRUE."
            )

        else:
            factors.append(
                ExplanationFactor(
                    factor_id="INTEGRITY-HARMFUL",

                    category=(
                        ExplanationCategory
                        .INTEGRITY
                    ),

                    description=(
                        "The Information Integrity "
                        "classifier assessed the content "
                        "as harmful."
                    ),

                    strength=(
                        integrity_confidence
                    ),

                    direction=(
                        FactorDirection
                        .SUPPORTING
                    ),

                    source=(
                        "hierarchical_classifier"
                    ),
                )
            )

            reasoning_trace.append(
                "Stage 1 classified the sample "
                "as HARMFUL."
            )

        # ---------------------------------
        # Stage 2: Threat subtype
        # ---------------------------------

        if (
            classification.integrity_status
            == IntegrityStatus.HARMFUL
            and classification.threat_type
            is not None
        ):

            threat_confidence = (
                classification
                .threat_confidence
                or 0.0
            )

            factors.append(
                ExplanationFactor(
                    factor_id="THREAT-TYPE",

                    category=(
                        ExplanationCategory
                        .THREAT_CLASS
                    ),

                    description=(
                        "The harmful content was "
                        f"classified as "
                        f"{classification.threat_type.value}."
                    ),

                    strength=(
                        threat_confidence
                    ),

                    direction=(
                        FactorDirection
                        .SUPPORTING
                    ),

                    source=(
                        "hierarchical_classifier"
                    ),
                )
            )

            reasoning_trace.append(
                "Stage 2 assigned the Cognitive "
                "Cyber Threat subtype: "
                f"{classification.threat_type.value}."
            )

        # ---------------------------------
        # Threat Intelligence indicators
        # ---------------------------------

        if data.threat_assessment is not None:

            threat = data.threat_assessment

            indicators = [
                (
                    "PROPAGATION",
                    ExplanationCategory.PROPAGATION,
                    threat.propagation_risk,
                    "Propagation risk indicator",
                ),

                (
                    "SYNTHETIC",
                    ExplanationCategory.SYNTHETIC_CONTENT,
                    threat.synthetic_content_indicator,
                    "Synthetic-content indicator",
                ),

                (
                    "COORDINATION",
                    ExplanationCategory.COORDINATION,
                    threat.coordination_indicator,
                    "Coordination indicator",
                ),

                (
                    "CONTEXT-RISK",
                    ExplanationCategory.CONTEXT,
                    threat.context_risk,
                    "Contextual risk indicator",
                ),
            ]

            for (
                factor_id,
                category,
                value,
                description,
            ) in indicators:

                value = clamp_strength(
                    value
                )

                factors.append(
                    ExplanationFactor(
                        factor_id=factor_id,

                        category=category,

                        description=(
                            f"{description}: "
                            f"{value:.2f}"
                        ),

                        strength=value,

                        direction=(
                            FactorDirection
                            .SUPPORTING
                        ),

                        source=(
                            "cognitive_threat_intelligence"
                        ),
                    )
                )

            reasoning_trace.append(
                "Cognitive Threat Intelligence "
                "indicators were incorporated."
            )

        # ---------------------------------
        # Context
        # ---------------------------------

        if data.context:

            domain = data.context.get(
                "domain"
            )

            if domain:

                factors.append(
                    ExplanationFactor(
                        factor_id="CONTEXT-DOMAIN",

                        category=(
                            ExplanationCategory
                            .CONTEXT
                        ),

                        description=(
                            "Operational context domain: "
                            f"{domain}."
                        ),

                        strength=1.0,

                        direction=(
                            FactorDirection
                            .CONTEXTUAL
                        ),

                        source=(
                            "context_intelligence"
                        ),
                    )
                )

                reasoning_trace.append(
                    "Operational context was "
                    f"identified as {domain}."
                )

        # ---------------------------------
        # Attribution evidence
        # ---------------------------------

        attribution_confidence = None

        if data.attribution is not None:

            attribution = data.attribution

            attribution_confidence = (
                attribution.overall_confidence
            )

            if (
                attribution.primary_hypothesis
                is not None
            ):

                hypothesis = (
                    attribution
                    .primary_hypothesis
                )

                factors.append(
                    ExplanationFactor(
                        factor_id="ATTRIBUTION",

                        category=(
                            ExplanationCategory
                            .ATTRIBUTION
                        ),

                        description=(
                            "Primary attribution hypothesis: "
                            f"{hypothesis.category.value}."
                        ),

                        strength=(
                            hypothesis.confidence
                        ),

                        direction=(
                            FactorDirection
                            .SUPPORTING
                        ),

                        source=(
                            "threat_attribution"
                        ),

                        evidence_ids=(
                            hypothesis
                            .supporting_evidence_ids
                        ),
                    )
                )

                reasoning_trace.append(
                    "Attribution evidence supported "
                    f"a {hypothesis.category.value} "
                    "hypothesis."
                )

            for item in attribution.evidence:
                evidence_ids.append(
                    item.evidence_id
                )

            if not attribution.metadata.get(
                "definitive_attribution",
                False,
            ):
                caveats.append(
                    "Attribution findings represent "
                    "analytical hypotheses and do not "
                    "constitute definitive actor attribution."
                )

        # ---------------------------------
        # Modality signals
        # ---------------------------------

        for (
            modality,
            value,
        ) in data.modality_signals.items():

            value = clamp_strength(
                value
            )

            factors.append(
                ExplanationFactor(
                    factor_id=(
                        f"MODALITY-{modality.upper()}"
                    ),

                    category=(
                        ExplanationCategory
                        .INTEGRITY
                    ),

                    description=(
                        f"{modality} modality signal: "
                        f"{value:.2f}"
                    ),

                    strength=value,

                    direction=(
                        FactorDirection
                        .CONTEXTUAL
                    ),

                    source=(
                        "multimodal_analysis"
                    ),
                )
            )

        # ---------------------------------
        # Explanation confidence
        # ---------------------------------

        confidence_values = [
            integrity_confidence
        ]

        if (
            classification.threat_confidence
            is not None
        ):
            confidence_values.append(
                classification
                .threat_confidence
            )

        if (
            attribution_confidence
            is not None
            and attribution_confidence > 0.0
        ):
            confidence_values.append(
                attribution_confidence
            )

        explanation_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

        explanation_confidence = (
            clamp_strength(
                explanation_confidence
            )
        )

        uncertainty = (
            1.0
            - explanation_confidence
        )

        if explanation_confidence < 0.60:
            caveats.append(
                "The explanation is based on "
                "relatively low-confidence analytical "
                "outputs and should be reviewed."
            )

        # ---------------------------------
        # Summary
        # ---------------------------------

        if (
            classification.integrity_status
            == IntegrityStatus.TRUE
        ):

            summary = (
                "AEGIS assessed the sample as "
                "true information and did not identify "
                "a Cognitive Cyber Threat subtype."
            )

        else:

            threat_name = (
                classification
                .threat_type
                .value
                if classification.threat_type
                is not None
                else "unspecified harmful information"
            )

            summary = (
                "AEGIS assessed the sample as harmful "
                f"and classified it as {threat_name}."
            )

            if data.threat_assessment is not None:

                summary += (
                    " The Cognitive Threat Intelligence "
                    "Engine assigned a "
                    f"{data.threat_assessment.severity.value} "
                    "severity level and a risk score of "
                    f"{data.threat_assessment.risk_score:.2f}."
                )

        # Remove duplicate evidence identifiers.
        evidence_ids = list(
            dict.fromkeys(
                evidence_ids
            )
        )

        return ExplanationReport(
            sample_id=(
                classification.sample_id
            ),

            summary=summary,

            factors=factors,

            explanation_confidence=(
                explanation_confidence
            ),

            uncertainty=(
                uncertainty
            ),

            evidence_ids=(
                evidence_ids
            ),

            caveats=caveats,

            reasoning_trace=(
                reasoning_trace
            ),

            metadata={
                "explanation_type": (
                    "structured_evidence_reasoning"
                ),

                "neural_feature_attribution": False,

                "factor_count": len(
                    factors
                ),
            },
        )