"""
Tests for AEGIS v0.13.0:
Threat Attribution & Evidence Structuring.
"""

import unittest

from aegis.attribution import (
    AttributionCategory,
    AttributionEvidence,
    AttributionInput,
    EvidenceType,
    SourceProfile,
    ThreatAttributionEngine,
    ThreatAttributionLayer,
    aggregate_evidence_confidence,
)

from aegis.classification import (
    CognitiveThreatType,
    IntegrityStatus,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


def build_threat_assessment():
    """
    Create a reusable harmful cognitive threat
    assessment for attribution tests.
    """

    return CognitiveThreatAssessment(
        sample_id="ATTR-001",

        integrity_status=(
            IntegrityStatus.HARMFUL
        ),

        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),

        severity=(
            ThreatSeverity.HIGH
        ),

        risk_score=0.82,

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


class TestEvidenceConfidence(
    unittest.TestCase
):

    def test_confidence_aggregation(self):

        evidence = [
            AttributionEvidence(
                evidence_id="E1",
                evidence_type=(
                    EvidenceType
                    .COORDINATION_SIGNAL
                ),
                description="Signal 1",
                confidence=0.8,
            ),

            AttributionEvidence(
                evidence_id="E2",
                evidence_type=(
                    EvidenceType
                    .CONTENT_SIMILARITY
                ),
                description="Signal 2",
                confidence=0.6,
            ),
        ]

        result = (
            aggregate_evidence_confidence(
                evidence
            )
        )

        self.assertAlmostEqual(
            result,
            0.7,
            places=5,
        )

    def test_empty_evidence_returns_zero(self):

        result = (
            aggregate_evidence_confidence(
                []
            )
        )

        self.assertEqual(
            result,
            0.0,
        )


class TestThreatAttributionEngine(
    unittest.TestCase
):

    def test_unknown_when_no_evidence(self):

        threat = (
            build_threat_assessment()
        )

        engine = (
            ThreatAttributionEngine()
        )

        result = engine.assess(
            threat_assessment=threat,
            evidence=[],
        )

        self.assertEqual(
            result.primary_hypothesis.category,
            AttributionCategory.UNKNOWN,
        )

        self.assertEqual(
            result.overall_confidence,
            0.0,
        )

        self.assertFalse(
            result.metadata[
                "definitive_attribution"
            ]
        )

    def test_coordination_hypothesis(self):

        threat = (
            build_threat_assessment()
        )

        evidence = [
            AttributionEvidence(
                evidence_id="COORD-1",
                evidence_type=(
                    EvidenceType
                    .COORDINATION_SIGNAL
                ),
                description=(
                    "Multiple accounts posted "
                    "near-identical content."
                ),
                confidence=0.90,
            ),

            AttributionEvidence(
                evidence_id="COORD-2",
                evidence_type=(
                    EvidenceType
                    .COORDINATION_SIGNAL
                ),
                description=(
                    "Synchronized posting times."
                ),
                confidence=0.80,
            ),
        ]

        engine = (
            ThreatAttributionEngine()
        )

        result = engine.assess(
            threat_assessment=threat,
            evidence=evidence,
        )

        self.assertEqual(
            result.primary_hypothesis.category,
            AttributionCategory.COORDINATED_NETWORK,
        )

        self.assertGreater(
            result.primary_hypothesis.confidence,
            0.0,
        )

        self.assertEqual(
            len(
                result.primary_hypothesis
                .supporting_evidence_ids
            ),
            2,
        )

    def test_synthetic_hypothesis(self):

        threat = (
            build_threat_assessment()
        )

        evidence = [
            AttributionEvidence(
                evidence_id="SYN-1",
                evidence_type=(
                    EvidenceType
                    .SYNTHETIC_CONTENT
                ),
                description=(
                    "Synthetic media detector "
                    "reported strong signal."
                ),
                confidence=0.92,
            ),
        ]

        result = (
            ThreatAttributionEngine()
            .assess(
                threat_assessment=threat,
                evidence=evidence,
            )
        )

        self.assertEqual(
            result.primary_hypothesis.category,
            (
                AttributionCategory
                .SYNTHETIC_CONTENT_OPERATION
            ),
        )

    def test_multiple_hypotheses_select_highest_confidence(self):

        threat = (
            build_threat_assessment()
        )

        evidence = [
            AttributionEvidence(
                evidence_id="C1",
                evidence_type=(
                    EvidenceType
                    .COORDINATION_SIGNAL
                ),
                description="Coordination",
                confidence=0.70,
            ),

            AttributionEvidence(
                evidence_id="S1",
                evidence_type=(
                    EvidenceType
                    .SYNTHETIC_CONTENT
                ),
                description="Synthetic signal",
                confidence=0.95,
            ),
        ]

        result = (
            ThreatAttributionEngine()
            .assess(
                threat_assessment=threat,
                evidence=evidence,
            )
        )

        self.assertEqual(
            result.primary_hypothesis.category,
            (
                AttributionCategory
                .SYNTHETIC_CONTENT_OPERATION
            ),
        )

        self.assertAlmostEqual(
            result.primary_hypothesis.confidence,
            0.95,
            places=5,
        )


class TestSourceProfile(
    unittest.TestCase
):

    def test_source_profile_preserved(self):

        threat = (
            build_threat_assessment()
        )

        source = SourceProfile(
            source_id="SRC-001",
            platform="X",
            account_name="example_account",
            country="Uganda",
            credibility_score=0.40,
            account_age_days=90,
            follower_count=1500,
        )

        result = (
            ThreatAttributionEngine()
            .assess(
                threat_assessment=threat,
                source_profile=source,
                evidence=[],
            )
        )

        self.assertIsNotNone(
            result.source_profile
        )

        self.assertEqual(
            result.source_profile.source_id,
            "SRC-001",
        )

        self.assertEqual(
            result.source_profile.platform,
            "X",
        )


class TestThreatAttributionLayer(
    unittest.TestCase
):

    def test_pipeline_layer(self):

        threat = (
            build_threat_assessment()
        )

        evidence = [
            AttributionEvidence(
                evidence_id="E-LAYER",
                evidence_type=(
                    EvidenceType
                    .SOURCE_PROVENANCE
                ),
                description=(
                    "Observed provenance indicator."
                ),
                confidence=0.75,
            ),
        ]

        layer = (
            ThreatAttributionLayer()
        )

        data = AttributionInput(
            threat_assessment=threat,
            evidence=evidence,
        )

        result = layer.process(
            data
        )

        self.assertEqual(
            result.sample_id,
            "ATTR-001",
        )

        self.assertEqual(
            result.primary_hypothesis.category,
            AttributionCategory.ORGANIZATIONAL,
        )


if __name__ == "__main__":
    unittest.main()