"""
Tests for AEGIS v0.14.0:
Explainable AI & Evidence-Based Reasoning Layer.
"""

import unittest

from aegis.attribution import (
    AttributionCategory,
    AttributionEvidence,
    AttributionHypothesis,
    EvidenceType,
    ThreatAttributionAssessment,
)

from aegis.classification import (
    CognitiveThreatType,
    HierarchicalClassificationOutput,
    IntegrityStatus,
)

from aegis.explainability import (
    ExplainabilityEngine,
    ExplainabilityInput,
    ExplainabilityLayer,
    ExplanationCategory,
    FactorDirection,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


def build_harmful_classification():

    return HierarchicalClassificationOutput(
        sample_id="EXP-001",

        integrity_status=(
            IntegrityStatus.HARMFUL
        ),

        integrity_confidence=0.94,

        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),

        threat_confidence=0.91,
    )


def build_threat_assessment():

    return CognitiveThreatAssessment(
        sample_id="EXP-001",

        integrity_status=(
            IntegrityStatus.HARMFUL
        ),

        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),

        severity=(
            ThreatSeverity.HIGH
        ),

        risk_score=0.84,

        risk_band=(
            RiskBand.CRITICAL
        ),

        classification_confidence=0.91,

        propagation_risk=0.88,

        synthetic_content_indicator=0.72,

        coordination_indicator=0.81,

        context_risk=0.90,

        source_credibility=0.30,
    )


def build_attribution():

    evidence = [
        AttributionEvidence(
            evidence_id="E-001",

            evidence_type=(
                EvidenceType
                .COORDINATION_SIGNAL
            ),

            description=(
                "Synchronized dissemination "
                "was observed."
            ),

            confidence=0.88,
        ),

        AttributionEvidence(
            evidence_id="E-002",

            evidence_type=(
                EvidenceType
                .SYNTHETIC_CONTENT
            ),

            description=(
                "Synthetic-content signal "
                "was detected."
            ),

            confidence=0.80,
        ),
    ]

    hypothesis = AttributionHypothesis(
        category=(
            AttributionCategory
            .COORDINATED_NETWORK
        ),

        confidence=0.88,

        rationale=(
            "Coordination indicators support "
            "a coordinated-network hypothesis."
        ),

        supporting_evidence_ids=[
            "E-001"
        ],
    )

    return ThreatAttributionAssessment(
        sample_id="EXP-001",

        source_profile=None,

        evidence=evidence,

        hypotheses=[
            hypothesis
        ],

        overall_confidence=0.84,

        primary_hypothesis=(
            hypothesis
        ),

        metadata={
            "definitive_attribution": False
        },
    )


class TestTrueInformationExplanation(
    unittest.TestCase
):

    def test_true_information_summary(self):

        classification = (
            HierarchicalClassificationOutput(
                sample_id="TRUE-EXP",

                integrity_status=(
                    IntegrityStatus.TRUE
                ),

                integrity_confidence=0.98,
            )
        )

        data = ExplainabilityInput(
            classification=classification
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        self.assertEqual(
            report.sample_id,
            "TRUE-EXP",
        )

        self.assertIn(
            "true information",
            report.summary.lower(),
        )

        threat_factors = [
            factor
            for factor in report.factors
            if factor.category
            == ExplanationCategory.THREAT_CLASS
        ]

        self.assertEqual(
            len(threat_factors),
            0,
        )


class TestHarmfulExplanation(
    unittest.TestCase
):

    def test_harmful_threat_factor(self):

        classification = (
            build_harmful_classification()
        )

        data = ExplainabilityInput(
            classification=classification
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        threat_factors = [
            factor
            for factor in report.factors
            if factor.category
            == ExplanationCategory.THREAT_CLASS
        ]

        self.assertEqual(
            len(threat_factors),
            1,
        )

        self.assertIn(
            "disinformation",
            threat_factors[
                0
            ].description.lower(),
        )

        self.assertEqual(
            threat_factors[
                0
            ].direction,
            FactorDirection.SUPPORTING,
        )


class TestThreatIntelligenceExplanation(
    unittest.TestCase
):

    def test_threat_indicators_added(self):

        data = ExplainabilityInput(
            classification=(
                build_harmful_classification()
            ),

            threat_assessment=(
                build_threat_assessment()
            ),
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        categories = {
            factor.category
            for factor in report.factors
        }

        self.assertIn(
            ExplanationCategory.PROPAGATION,
            categories,
        )

        self.assertIn(
            ExplanationCategory.SYNTHETIC_CONTENT,
            categories,
        )

        self.assertIn(
            ExplanationCategory.COORDINATION,
            categories,
        )

        self.assertIn(
            ExplanationCategory.CONTEXT,
            categories,
        )

        self.assertIn(
            "risk score",
            report.summary.lower(),
        )


class TestAttributionExplanation(
    unittest.TestCase
):

    def test_attribution_evidence_preserved(self):

        data = ExplainabilityInput(
            classification=(
                build_harmful_classification()
            ),

            threat_assessment=(
                build_threat_assessment()
            ),

            attribution=(
                build_attribution()
            ),
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        self.assertIn(
            "E-001",
            report.evidence_ids,
        )

        self.assertIn(
            "E-002",
            report.evidence_ids,
        )

        attribution_factors = [
            factor
            for factor in report.factors
            if factor.category
            == ExplanationCategory.ATTRIBUTION
        ]

        self.assertEqual(
            len(
                attribution_factors
            ),
            1,
        )

        self.assertIn(
            "coordinated_network",
            attribution_factors[
                0
            ].description,
        )

    def test_non_definitive_attribution_caveat(self):

        data = ExplainabilityInput(
            classification=(
                build_harmful_classification()
            ),

            attribution=(
                build_attribution()
            ),
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        caveat_text = " ".join(
            report.caveats
        ).lower()

        self.assertIn(
            "do not constitute definitive",
            caveat_text,
        )


class TestContextExplanation(
    unittest.TestCase
):

    def test_context_domain_added(self):

        data = ExplainabilityInput(
            classification=(
                build_harmful_classification()
            ),

            context={
                "domain": "conflict",
                "country": "Uganda",
                "platform": "X",
            },
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        context_factors = [
            factor
            for factor in report.factors
            if factor.factor_id
            == "CONTEXT-DOMAIN"
        ]

        self.assertEqual(
            len(
                context_factors
            ),
            1,
        )

        self.assertIn(
            "conflict",
            context_factors[
                0
            ].description.lower(),
        )


class TestExplanationConfidence(
    unittest.TestCase
):

    def test_confidence_and_uncertainty(self):

        classification = (
            HierarchicalClassificationOutput(
                sample_id="CONF-001",

                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),

                integrity_confidence=0.80,

                threat_type=(
                    CognitiveThreatType.MISINFORMATION
                ),

                threat_confidence=0.60,
            )
        )

        data = ExplainabilityInput(
            classification=classification
        )

        report = (
            ExplainabilityEngine()
            .explain(
                data
            )
        )

        expected_confidence = 0.70

        self.assertAlmostEqual(
            report.explanation_confidence,
            expected_confidence,
            places=5,
        )

        self.assertAlmostEqual(
            report.uncertainty,
            0.30,
            places=5,
        )


class TestExplainabilityLayer(
    unittest.TestCase
):

    def test_pipeline_layer(self):

        data = ExplainabilityInput(
            classification=(
                build_harmful_classification()
            ),

            threat_assessment=(
                build_threat_assessment()
            ),

            attribution=(
                build_attribution()
            ),

            context={
                "domain": "international_security"
            },
        )

        layer = (
            ExplainabilityLayer()
        )

        result = layer.process(
            data
        )

        self.assertEqual(
            result.sample_id,
            "EXP-001",
        )

        self.assertGreater(
            len(
                result.factors
            ),
            0,
        )

        self.assertGreater(
            len(
                result.reasoning_trace
            ),
            0,
        )

        self.assertEqual(
            result.metadata[
                "explanation_type"
            ],
            "structured_evidence_reasoning",
        )

        self.assertFalse(
            result.metadata[
                "neural_feature_attribution"
            ]
        )


if __name__ == "__main__":
    unittest.main()