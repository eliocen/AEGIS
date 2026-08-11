"""
Tests for AEGIS v0.15.0:
Decision Support & Response Layer.
"""

import unittest

from aegis.attribution import (
    AttributionCategory,
    AttributionHypothesis,
    ThreatAttributionAssessment,
)

from aegis.classification import (
    CognitiveThreatType,
    IntegrityStatus,
)

from aegis.decision import (
    DecisionSupportEngine,
    DecisionSupportInput,
    DecisionSupportLayer,
    RecommendedAction,
    ResponsePriority,
    actions_from_assessment,
    priority_from_assessment,
)

from aegis.explainability import (
    ExplanationReport,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


def build_harmful_threat():

    return CognitiveThreatAssessment(
        sample_id="DEC-001",

        integrity_status=(
            IntegrityStatus.HARMFUL
        ),

        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),

        severity=(
            ThreatSeverity.CRITICAL
        ),

        risk_score=0.88,

        risk_band=(
            RiskBand.CRITICAL
        ),

        classification_confidence=0.93,

        propagation_risk=0.91,

        synthetic_content_indicator=0.77,

        coordination_indicator=0.86,

        context_risk=0.94,

        source_credibility=0.25,
    )


def build_true_threat():

    return CognitiveThreatAssessment(
        sample_id="DEC-TRUE",

        integrity_status=(
            IntegrityStatus.TRUE
        ),

        threat_type=None,

        severity=(
            ThreatSeverity.NONE
        ),

        risk_score=0.0,

        risk_band=(
            RiskBand.MINIMAL
        ),

        classification_confidence=0.98,

        propagation_risk=0.0,

        synthetic_content_indicator=0.0,

        coordination_indicator=0.0,

        context_risk=0.0,

        source_credibility=None,
    )


def build_explanation(
    sample_id="DEC-001",
    confidence=0.90,
):

    return ExplanationReport(
        sample_id=sample_id,

        summary=(
            "Structured AEGIS explanation."
        ),

        factors=[],

        explanation_confidence=(
            confidence
        ),

        uncertainty=(
            1.0 - confidence
        ),

        evidence_ids=[],

        caveats=[],

        reasoning_trace=[],
    )


def build_attribution():

    hypothesis = AttributionHypothesis(
        category=(
            AttributionCategory
            .COORDINATED_NETWORK
        ),

        confidence=0.80,

        rationale=(
            "Observed coordination indicators "
            "support a network hypothesis."
        ),
    )

    return ThreatAttributionAssessment(
        sample_id="DEC-001",

        source_profile=None,

        evidence=[],

        hypotheses=[
            hypothesis
        ],

        overall_confidence=0.80,

        primary_hypothesis=(
            hypothesis
        ),

        metadata={
            "definitive_attribution": False
        },
    )


class TestDecisionPriority(
    unittest.TestCase
):

    def test_true_information_has_no_priority(self):

        priority = (
            priority_from_assessment(
                integrity_status=(
                    IntegrityStatus.TRUE
                ),

                severity=(
                    ThreatSeverity.NONE
                ),

                risk_band=(
                    RiskBand.MINIMAL
                ),
            )
        )

        self.assertEqual(
            priority,
            ResponsePriority.NONE,
        )

    def test_critical_threat_is_urgent(self):

        priority = (
            priority_from_assessment(
                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),

                severity=(
                    ThreatSeverity.CRITICAL
                ),

                risk_band=(
                    RiskBand.CRITICAL
                ),
            )
        )

        self.assertEqual(
            priority,
            ResponsePriority.URGENT,
        )


class TestActionPolicy(
    unittest.TestCase
):

    def test_true_information_returns_no_action(self):

        actions = (
            actions_from_assessment(
                integrity_status=(
                    IntegrityStatus.TRUE
                ),

                severity=(
                    ThreatSeverity.NONE
                ),

                risk_band=(
                    RiskBand.MINIMAL
                ),

                explanation_confidence=0.98,
            )
        )

        self.assertEqual(
            actions,
            [
                RecommendedAction.NO_ACTION
            ],
        )

    def test_critical_threat_generates_review_actions(self):

        actions = (
            actions_from_assessment(
                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),

                severity=(
                    ThreatSeverity.CRITICAL
                ),

                risk_band=(
                    RiskBand.CRITICAL
                ),

                explanation_confidence=0.90,

                attribution_confidence=0.80,
            )
        )

        self.assertIn(
            RecommendedAction.MONITOR,
            actions,
        )

        self.assertIn(
            RecommendedAction.PRESERVE_EVIDENCE,
            actions,
        )

        self.assertIn(
            RecommendedAction.FACT_CHECK,
            actions,
        )

        self.assertIn(
            RecommendedAction.HUMAN_REVIEW,
            actions,
        )

        self.assertIn(
            RecommendedAction.ESCALATE_FOR_REVIEW,
            actions,
        )

        self.assertIn(
            RecommendedAction.PREPARE_ADVISORY,
            actions,
        )

        self.assertIn(
            RecommendedAction.COORDINATE_RESPONSE,
            actions,
        )

    def test_low_confidence_requires_verification(self):

        actions = (
            actions_from_assessment(
                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),

                severity=(
                    ThreatSeverity.MODERATE
                ),

                risk_band=(
                    RiskBand.MODERATE
                ),

                explanation_confidence=0.45,
            )
        )

        self.assertIn(
            RecommendedAction.VERIFY,
            actions,
        )

        self.assertIn(
            RecommendedAction.HUMAN_REVIEW,
            actions,
        )


class TestDecisionSupportEngine(
    unittest.TestCase
):

    def test_true_information_recommendation(self):

        threat = (
            build_true_threat()
        )

        explanation = (
            build_explanation(
                sample_id="DEC-TRUE",
                confidence=0.98,
            )
        )

        data = DecisionSupportInput(
            threat_assessment=threat,
            explanation=explanation,
        )

        result = (
            DecisionSupportEngine()
            .recommend(
                data
            )
        )

        self.assertEqual(
            result.priority,
            ResponsePriority.NONE,
        )

        self.assertEqual(
            result.actions,
            [
                RecommendedAction.NO_ACTION
            ],
        )

        self.assertFalse(
            result.human_review_required
        )

        self.assertFalse(
            result.automation_permitted
        )

    def test_critical_threat_recommendation(self):

        data = DecisionSupportInput(
            threat_assessment=(
                build_harmful_threat()
            ),

            explanation=(
                build_explanation()
            ),

            attribution=(
                build_attribution()
            ),
        )

        result = (
            DecisionSupportEngine()
            .recommend(
                data
            )
        )

        self.assertEqual(
            result.sample_id,
            "DEC-001",
        )

        self.assertEqual(
            result.priority,
            ResponsePriority.URGENT,
        )

        self.assertTrue(
            result.human_review_required
        )

        self.assertFalse(
            result.automation_permitted
        )

        self.assertIn(
            RecommendedAction
            .ESCALATE_FOR_REVIEW,
            result.actions,
        )

        self.assertIn(
            RecommendedAction
            .COORDINATE_RESPONSE,
            result.actions,
        )

        self.assertEqual(
            result.metadata[
                "decision_mode"
            ],
            "human_supervised",
        )

        self.assertFalse(
            result.metadata[
                "autonomous_enforcement"
            ]
        )

    def test_mismatched_sample_ids_fail(self):

        threat = (
            build_harmful_threat()
        )

        explanation = (
            build_explanation(
                sample_id="OTHER-SAMPLE"
            )
        )

        data = DecisionSupportInput(
            threat_assessment=threat,
            explanation=explanation,
        )

        with self.assertRaises(
            ValueError
        ):

            DecisionSupportEngine().recommend(
                data
            )


class TestDecisionSupportLayer(
    unittest.TestCase
):

    def test_pipeline_layer(self):

        data = DecisionSupportInput(
            threat_assessment=(
                build_harmful_threat()
            ),

            explanation=(
                build_explanation()
            ),

            attribution=(
                build_attribution()
            ),
        )

        layer = (
            DecisionSupportLayer()
        )

        result = layer.process(
            data
        )

        self.assertEqual(
            result.sample_id,
            "DEC-001",
        )

        self.assertGreater(
            len(
                result.actions
            ),
            0,
        )

        self.assertGreater(
            len(
                result.rationale
            ),
            0,
        )

        self.assertFalse(
            result.automation_permitted
        )


if __name__ == "__main__":
    unittest.main()