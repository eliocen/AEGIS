"""
Unit tests for AEGIS Data Models

Author: Eli Haggai Ocen
Version: 0.3.0
"""

import unittest

from aegis.models import (
    ThreatSample,
    ThreatAssessment,
    ContextProfile,
    ThreatType,
    ThreatLevel,
    SecurityDomain,
)


class TestThreatSample(unittest.TestCase):

    def test_sample_creation(self):
        sample = ThreatSample(
            sample_id="001",
            text="Breaking news example",
            image_path=None,
            language="English",
            source="Twitter",
            timestamp="2025-08-10",
            label=ThreatType.MISINFORMATION,
        )

        self.assertEqual(sample.sample_id, "001")
        self.assertEqual(sample.label, ThreatType.MISINFORMATION)


class TestContextProfile(unittest.TestCase):

    def test_context_creation(self):
        context = ContextProfile(
            domain=SecurityDomain.ELECTION,
            country="Uganda",
            platform="Twitter",
            language="English",
            event="2026 General Election",
        )

        self.assertEqual(context.domain, SecurityDomain.ELECTION)


class TestThreatAssessment(unittest.TestCase):

    def test_assessment_creation(self):
        assessment = ThreatAssessment(
            predicted_class=ThreatType.DISINFORMATION,
            confidence=0.95,
            severity=ThreatLevel.HIGH,
            explanation="Potential coordinated disinformation.",
            evidence=["Evidence 1", "Evidence 2"],
        )

        self.assertEqual(assessment.severity, ThreatLevel.HIGH)


if __name__ == "__main__":
    unittest.main()