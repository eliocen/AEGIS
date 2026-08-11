"""
Tests for AEGIS v0.12.0:
Cognitive Threat Intelligence Engine.
"""

import unittest

from aegis.classification import (
    CognitiveThreatType,
    HierarchicalClassificationOutput,
    IntegrityStatus,
)

from aegis.intelligence import (
    CognitiveThreatIntelligenceEngine,
    CognitiveThreatIntelligenceLayer,
    RiskBand,
    ThreatIntelligenceInput,
    ThreatSeverity,
    compute_risk_score,
)


class TestThreatScoring(unittest.TestCase):

    def test_risk_score_range(self):

        score = compute_risk_score(
            classification_confidence=0.9,
            propagation_score=0.7,
            synthetic_content_score=0.6,
            coordination_score=0.8,
            context_risk_score=0.9,
        )

        self.assertGreaterEqual(
            score,
            0.0,
        )

        self.assertLessEqual(
            score,
            1.0,
        )

    def test_higher_indicators_produce_higher_risk(self):

        low = compute_risk_score(
            classification_confidence=0.6,
            propagation_score=0.2,
            synthetic_content_score=0.1,
            coordination_score=0.2,
            context_risk_score=0.3,
        )

        high = compute_risk_score(
            classification_confidence=0.9,
            propagation_score=0.9,
            synthetic_content_score=0.8,
            coordination_score=0.9,
            context_risk_score=0.9,
        )

        self.assertGreater(
            high,
            low,
        )


class TestCognitiveThreatEngine(unittest.TestCase):

    def test_true_information_has_zero_risk(self):

        classification = (
            HierarchicalClassificationOutput(
                sample_id="TRUE-001",
                integrity_status=(
                    IntegrityStatus.TRUE
                ),
                integrity_confidence=0.99,
            )
        )

        data = ThreatIntelligenceInput(
            classification=classification,
            propagation_score=0.9,
            synthetic_content_score=0.9,
            coordination_score=0.9,
            context_risk_score=0.9,
        )

        engine = (
            CognitiveThreatIntelligenceEngine()
        )

        result = engine.assess(
            data
        )

        self.assertEqual(
            result.risk_score,
            0.0,
        )

        self.assertEqual(
            result.severity,
            ThreatSeverity.NONE,
        )

        self.assertIsNone(
            result.threat_type
        )

    def test_harmful_information_produces_risk(self):

        classification = (
            HierarchicalClassificationOutput(
                sample_id="CCT-001",
                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),
                integrity_confidence=0.95,
                threat_type=(
                    CognitiveThreatType.DISINFORMATION
                ),
                threat_confidence=0.90,
            )
        )

        data = ThreatIntelligenceInput(
            classification=classification,
            propagation_score=0.85,
            synthetic_content_score=0.70,
            coordination_score=0.80,
            context_risk_score=0.90,
        )

        engine = (
            CognitiveThreatIntelligenceEngine()
        )

        result = engine.assess(
            data
        )

        self.assertGreater(
            result.risk_score,
            0.0,
        )

        self.assertEqual(
            result.threat_type,
            CognitiveThreatType.DISINFORMATION,
        )

        self.assertIn(
            result.risk_band,
            list(RiskBand),
        )


class TestThreatIntelligenceLayer(unittest.TestCase):

    def test_pipeline_layer(self):

        classification = (
            HierarchicalClassificationOutput(
                sample_id="CCT-LAYER",
                integrity_status=(
                    IntegrityStatus.HARMFUL
                ),
                integrity_confidence=0.90,
                threat_type=(
                    CognitiveThreatType.HATE_SPEECH
                ),
                threat_confidence=0.88,
            )
        )

        data = ThreatIntelligenceInput(
            classification=classification,
            propagation_score=0.75,
            synthetic_content_score=0.20,
            coordination_score=0.65,
            context_risk_score=0.80,
        )

        layer = (
            CognitiveThreatIntelligenceLayer()
        )

        result = layer.process(
            data
        )

        self.assertEqual(
            result.sample_id,
            "CCT-LAYER",
        )

        self.assertEqual(
            result.threat_type,
            CognitiveThreatType.HATE_SPEECH,
        )


if __name__ == "__main__":
    unittest.main()