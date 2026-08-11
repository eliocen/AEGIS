"""
Tests for AEGIS v0.23.0:
Robustness, Calibration & Uncertainty Evaluation.
"""

import tempfile
import unittest

from pathlib import Path

from aegis.robustness import (
    RobustnessEvaluationReport,
    binary_brier_score,
    character_noise_perturbation,
    confidence_degradation,
    confidence_error_analysis,
    expected_calibration_error,
    hierarchical_uncertainty,
    lowercase_perturbation,
    maximum_calibration_error,
    negative_log_likelihood,
    normalized_predictive_entropy,
    performance_degradation,
    prediction_consistency,
    prediction_uncertainty,
    predictive_entropy,
    punctuation_perturbation,
    reliability_bins,
    whitespace_perturbation,
    word_deletion_perturbation,
)


class TestCalibrationMetrics(
    unittest.TestCase
):

    def test_perfect_brier_score(self):

        score = binary_brier_score(
            probabilities=[
                0.0,
                1.0,
                1.0,
                0.0,
            ],

            targets=[
                0,
                1,
                1,
                0,
            ],
        )

        self.assertEqual(
            score,
            0.0,
        )

    def test_negative_log_likelihood(self):

        loss = negative_log_likelihood(
            [
                0.9,
                0.8,
                0.95,
            ]
        )

        self.assertGreater(
            loss,
            0.0,
        )

        self.assertLess(
            loss,
            1.0,
        )

    def test_reliability_bins(self):

        bins = reliability_bins(
            confidences=[
                0.95,
                0.85,
                0.65,
                0.45,
            ],

            correct=[
                1,
                1,
                0,
                0,
            ],

            num_bins=5,
        )

        self.assertEqual(
            len(bins),
            5,
        )

        self.assertEqual(
            sum(
                item[
                    "count"
                ]
                for item in bins
            ),
            4,
        )

    def test_ece_range(self):

        score = expected_calibration_error(
            confidences=[
                0.95,
                0.85,
                0.65,
                0.45,
            ],

            correct=[
                1,
                1,
                0,
                0,
            ],

            num_bins=5,
        )

        self.assertGreaterEqual(
            score,
            0.0,
        )

        self.assertLessEqual(
            score,
            1.0,
        )

    def test_mce_range(self):

        score = maximum_calibration_error(
            confidences=[
                0.95,
                0.85,
                0.65,
                0.45,
            ],

            correct=[
                1,
                1,
                0,
                0,
            ],

            num_bins=5,
        )

        self.assertGreaterEqual(
            score,
            0.0,
        )

        self.assertLessEqual(
            score,
            1.0,
        )


class TestUncertaintyMetrics(
    unittest.TestCase
):

    def test_low_entropy_prediction(self):

        probabilities = [
            0.97,
            0.01,
            0.01,
            0.005,
            0.005,
        ]

        uncertainty = (
            prediction_uncertainty(
                probabilities
            )
        )

        self.assertEqual(
            uncertainty.predicted_class,
            0,
        )

        self.assertAlmostEqual(
            uncertainty.confidence,
            0.97,
            places=6,
        )

        self.assertLess(
            uncertainty
            .normalized_entropy,
            0.2,
        )

    def test_high_entropy_prediction(self):

        probabilities = [
            0.22,
            0.21,
            0.20,
            0.19,
            0.18,
        ]

        normalized = (
            normalized_predictive_entropy(
                probabilities
            )
        )

        self.assertGreater(
            normalized,
            0.95,
        )

    def test_entropy_non_negative(self):

        entropy = predictive_entropy(
            [
                0.5,
                0.5,
            ]
        )

        self.assertGreaterEqual(
            entropy,
            0.0,
        )

    def test_confidence_margin(self):

        uncertainty = (
            prediction_uncertainty(
                [
                    0.70,
                    0.20,
                    0.10,
                ]
            )
        )

        self.assertAlmostEqual(
            uncertainty
            .confidence_margin,
            0.50,
            places=6,
        )


class TestPerturbationFramework(
    unittest.TestCase
):

    def test_whitespace_perturbation(self):

        text = (
            "Information integrity matters"
        )

        perturbed = (
            whitespace_perturbation(
                text
            )
        )

        self.assertNotEqual(
            perturbed,
            text,
        )

        self.assertEqual(
            perturbed.split(),
            text.split(),
        )

    def test_lowercase_perturbation(self):

        self.assertEqual(
            lowercase_perturbation(
                "AEGIS Security"
            ),
            "aegis security",
        )

    def test_punctuation_perturbation(self):

        result = (
            punctuation_perturbation(
                "AEGIS: secure, reliable!"
            )
        )

        self.assertNotIn(
            ":",
            result,
        )

        self.assertNotIn(
            ",",
            result,
        )

        self.assertNotIn(
            "!",
            result,
        )

    def test_word_deletion_deterministic(self):

        text = (
            "This is a controlled robustness "
            "evaluation example"
        )

        first = (
            word_deletion_perturbation(
                text,
                probability=0.30,
                seed=42,
            )
        )

        second = (
            word_deletion_perturbation(
                text,
                probability=0.30,
                seed=42,
            )
        )

        self.assertEqual(
            first,
            second,
        )

    def test_character_noise_deterministic(self):

        text = (
            "InformationIntegrity"
        )

        first = (
            character_noise_perturbation(
                text,
                probability=0.25,
                seed=42,
            )
        )

        second = (
            character_noise_perturbation(
                text,
                probability=0.25,
                seed=42,
            )
        )

        self.assertEqual(
            first,
            second,
        )


class TestRobustnessMetrics(
    unittest.TestCase
):

    def test_performance_degradation(self):

        result = performance_degradation(
            clean_metric=0.80,
            perturbed_metric=0.72,
        )

        self.assertAlmostEqual(
            result[
                "absolute_degradation"
            ],
            -0.08,
            places=6,
        )

        self.assertAlmostEqual(
            result[
                "relative_degradation_percent"
            ],
            -10.0,
            places=6,
        )

    def test_prediction_consistency(self):

        consistency = (
            prediction_consistency(
                clean_predictions=[
                    0,
                    1,
                    2,
                    3,
                ],

                perturbed_predictions=[
                    0,
                    1,
                    4,
                    3,
                ],
            )
        )

        self.assertEqual(
            consistency,
            0.75,
        )

    def test_confidence_degradation(self):

        result = confidence_degradation(
            clean_confidences=[
                0.9,
                0.8,
            ],

            perturbed_confidences=[
                0.7,
                0.6,
            ],
        )

        self.assertAlmostEqual(
            result[
                "clean_mean_confidence"
            ],
            0.85,
        )

        self.assertAlmostEqual(
            result[
                "perturbed_mean_confidence"
            ],
            0.65,
        )

        self.assertAlmostEqual(
            result[
                "absolute_change"
            ],
            -0.20,
        )


class TestConfidenceAwareErrors(
    unittest.TestCase
):

    def test_confidence_error_categories(self):

        result = (
            confidence_error_analysis(
                targets=[
                    0,
                    1,
                    2,
                    3,
                ],

                predictions=[
                    0,
                    1,
                    1,
                    3,
                ],

                confidences=[
                    0.95,
                    0.70,
                    0.91,
                    0.60,
                ],

                high_confidence_threshold=(
                    0.80
                ),
            )
        )

        self.assertEqual(
            result[
                "high_confidence_correct"
            ],
            1,
        )

        self.assertEqual(
            result[
                "low_confidence_correct"
            ],
            2,
        )

        self.assertEqual(
            result[
                "high_confidence_incorrect"
            ],
            1,
        )

        self.assertEqual(
            result[
                "low_confidence_incorrect"
            ],
            0,
        )

        self.assertAlmostEqual(
            result[
                "high_confidence_error_rate"
            ],
            0.25,
        )


class TestHierarchicalUncertainty(
    unittest.TestCase
):

    def test_hierarchical_uncertainty(self):

        result = (
            hierarchical_uncertainty(
                integrity_probabilities=[
                    0.06,
                    0.94,
                ],

                threat_probabilities=[
                    0.10,
                    0.61,
                    0.18,
                    0.11,
                ],
            )
        )

        self.assertAlmostEqual(
            result
            .integrity_confidence,
            0.94,
            places=6,
        )

        self.assertAlmostEqual(
            result
            .threat_confidence,
            0.61,
            places=6,
        )

        self.assertAlmostEqual(
            result
            .overall_confidence,
            0.61,
            places=6,
        )

        self.assertGreaterEqual(
            result
            .overall_uncertainty,
            result
            .integrity_uncertainty,
        )

    def test_true_only_hierarchical_uncertainty(self):

        result = (
            hierarchical_uncertainty(
                integrity_probabilities=[
                    0.96,
                    0.04,
                ]
            )
        )

        self.assertIsNone(
            result.threat_confidence
        )

        self.assertIsNone(
            result.threat_uncertainty
        )

        self.assertAlmostEqual(
            result.overall_confidence,
            0.96,
            places=6,
        )


class TestRobustnessReport(
    unittest.TestCase
):

    def test_report_save_json(self):

        report = (
            RobustnessEvaluationReport(
                experiment_name=(
                    "robustness_test"
                ),

                calibration={
                    "ece": 0.05,
                    "mce": 0.10,
                },

                uncertainty={
                    "mean_uncertainty": (
                        0.25
                    )
                },

                robustness={
                    "prediction_consistency": (
                        0.90
                    )
                },

                confidence_analysis={
                    "high_confidence_error_rate": (
                        0.02
                    )
                },

                metadata={
                    "version": "0.23.0"
                },
            )
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "robustness.json"
            )

            saved = (
                report.save_json(
                    path
                )
            )

            self.assertTrue(
                saved.is_file()
            )


if __name__ == "__main__":
    unittest.main()