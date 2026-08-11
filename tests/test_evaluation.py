"""
Tests for AEGIS v0.20.0:
Research Metrics, Evaluation & Error Analysis.
"""

import tempfile
import unittest
from pathlib import Path

from aegis.evaluation import (
    AEGISEvaluator,
    EvaluationReport,
    FIVE_CLASS_LABELS,
    brier_score_binary,
    classification_metrics,
    collect_errors,
    confusion_matrix,
    expected_calibration_error,
    five_class_index,
    hierarchical_label,
    hierarchical_metrics,
    per_class_metrics,
    stratified_classification_metrics,
)


class TestEvaluationLabels(
    unittest.TestCase
):

    def test_true_label(self):

        self.assertEqual(
            hierarchical_label(
                0,
                -1,
            ),
            "true",
        )

    def test_harmful_label(self):

        self.assertEqual(
            hierarchical_label(
                1,
                1,
            ),
            "disinformation",
        )

    def test_five_class_label_order(self):

        self.assertEqual(
            FIVE_CLASS_LABELS,
            [
                "true",
                "misinformation",
                "disinformation",
                "malinformation",
                "hate_speech",
            ],
        )

        self.assertEqual(
            five_class_index(
                "true"
            ),
            0,
        )

        self.assertEqual(
            five_class_index(
                "hate_speech"
            ),
            4,
        )


class TestConfusionMatrix(
    unittest.TestCase
):

    def test_confusion_matrix_shape_and_values(self):

        targets = [
            0,
            0,
            1,
            1,
        ]

        predictions = [
            0,
            1,
            1,
            1,
        ]

        matrix = confusion_matrix(
            targets,
            predictions,
            num_classes=2,
        )

        self.assertEqual(
            matrix,
            [
                [1, 1],
                [0, 2],
            ],
        )


class TestClassificationMetrics(
    unittest.TestCase
):

    def test_perfect_classification(self):

        targets = [
            0,
            1,
            2,
            3,
        ]

        predictions = [
            0,
            1,
            2,
            3,
        ]

        metrics = classification_metrics(
            targets,
            predictions,
            num_classes=4,
        )

        self.assertEqual(
            metrics[
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "macro_f1"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "weighted_f1"
            ],
            1.0,
        )

    def test_per_class_metrics(self):

        targets = [
            0,
            0,
            1,
            1,
        ]

        predictions = [
            0,
            1,
            1,
            1,
        ]

        metrics = per_class_metrics(
            targets,
            predictions,
            num_classes=2,
        )

        self.assertEqual(
            metrics[
                1
            ][
                "recall"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                0
            ][
                "support"
            ],
            2,
        )


class TestHierarchicalMetrics(
    unittest.TestCase
):

    def test_hierarchical_exact_match(self):

        integrity_targets = [
            0,
            1,
            1,
            1,
        ]

        integrity_predictions = [
            0,
            1,
            1,
            1,
        ]

        threat_targets = [
            -1,
            0,
            1,
            3,
        ]

        threat_predictions = [
            0,
            0,
            1,
            3,
        ]

        metrics = hierarchical_metrics(
            integrity_targets,
            integrity_predictions,
            threat_targets,
            threat_predictions,
        )

        self.assertEqual(
            metrics[
                "integrity"
            ][
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "threat"
            ][
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "hierarchical_exact_accuracy"
            ],
            1.0,
        )

    def test_hierarchical_error_detected(self):

        metrics = hierarchical_metrics(
            integrity_targets=[
                0,
                1,
            ],

            integrity_predictions=[
                0,
                1,
            ],

            threat_targets=[
                -1,
                1,
            ],

            threat_predictions=[
                0,
                0,
            ],
        )

        self.assertEqual(
            metrics[
                "integrity"
            ][
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "threat"
            ][
                "accuracy"
            ],
            0.0,
        )

        self.assertEqual(
            metrics[
                "hierarchical_exact_accuracy"
            ],
            0.5,
        )


class TestFiveClassMetrics(
    unittest.TestCase
):

    def test_five_class_confusion_matrix(self):

        metrics = hierarchical_metrics(
            integrity_targets=[
                0,
                1,
                1,
                1,
                1,
            ],

            integrity_predictions=[
                0,
                1,
                1,
                1,
                0,
            ],

            threat_targets=[
                -1,
                0,
                1,
                2,
                3,
            ],

            threat_predictions=[
                0,
                0,
                1,
                3,
                2,
            ],
        )

        five_class = (
            metrics[
                "five_class"
            ]
        )

        self.assertEqual(
            five_class[
                "labels"
            ],
            FIVE_CLASS_LABELS,
        )

        matrix = (
            five_class[
                "confusion_matrix"
            ]
        )

        self.assertEqual(
            len(matrix),
            5,
        )

        for row in matrix:

            self.assertEqual(
                len(row),
                5,
            )

        expected = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ]

        self.assertEqual(
            matrix,
            expected,
        )

    def test_five_class_named_per_class_metrics(self):

        metrics = hierarchical_metrics(
            integrity_targets=[
                0,
                1,
                1,
                1,
                1,
            ],

            integrity_predictions=[
                0,
                1,
                1,
                1,
                1,
            ],

            threat_targets=[
                -1,
                0,
                1,
                2,
                3,
            ],

            threat_predictions=[
                0,
                0,
                1,
                2,
                3,
            ],
        )

        five_class = (
            metrics[
                "five_class"
            ]
        )

        self.assertEqual(
            five_class[
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            five_class[
                "macro_f1"
            ],
            1.0,
        )

        for label in (
            FIVE_CLASS_LABELS
        ):

            self.assertIn(
                label,
                five_class[
                    "per_class"
                ],
            )

            self.assertEqual(
                five_class[
                    "per_class"
                ][label][
                    "precision"
                ],
                1.0,
            )

            self.assertEqual(
                five_class[
                    "per_class"
                ][label][
                    "recall"
                ],
                1.0,
            )

            self.assertEqual(
                five_class[
                    "per_class"
                ][label][
                    "f1"
                ],
                1.0,
            )

            self.assertEqual(
                five_class[
                    "per_class"
                ][label][
                    "support"
                ],
                1,
            )

    def test_five_class_labeled_matrix_metadata(self):

        metrics = hierarchical_metrics(
            integrity_targets=[
                0,
                1,
            ],

            integrity_predictions=[
                0,
                1,
            ],

            threat_targets=[
                -1,
                3,
            ],

            threat_predictions=[
                0,
                3,
            ],
        )

        labeled = (
            metrics[
                "five_class"
            ][
                "confusion_matrix_labeled"
            ]
        )

        self.assertEqual(
            labeled[
                "rows"
            ],
            FIVE_CLASS_LABELS,
        )

        self.assertEqual(
            labeled[
                "columns"
            ],
            FIVE_CLASS_LABELS,
        )

        self.assertEqual(
            len(
                labeled[
                    "matrix"
                ]
            ),
            5,
        )


class TestCalibrationMetrics(
    unittest.TestCase
):

    def test_perfect_brier_score(self):

        score = brier_score_binary(
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

    def test_expected_calibration_error_range(self):

        score = expected_calibration_error(
            confidences=[
                0.9,
                0.8,
                0.6,
                0.7,
            ],

            correct=[
                1,
                1,
                0,
                1,
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


class TestStratifiedEvaluation(
    unittest.TestCase
):

    def test_language_stratification(self):

        results = (
            stratified_classification_metrics(
                targets=[
                    0,
                    1,
                    0,
                    1,
                ],

                predictions=[
                    0,
                    1,
                    1,
                    1,
                ],

                strata=[
                    "en",
                    "en",
                    "zh",
                    "zh",
                ],

                num_classes=2,
            )
        )

        self.assertIn(
            "en",
            results,
        )

        self.assertIn(
            "zh",
            results,
        )

        self.assertEqual(
            results[
                "en"
            ][
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            results[
                "zh"
            ][
                "accuracy"
            ],
            0.5,
        )


class TestErrorAnalysis(
    unittest.TestCase
):

    def test_collect_errors(self):

        errors = collect_errors(
            sample_ids=[
                "S1",
                "S2",
                "S3",
            ],

            true_labels=[
                "true",
                "disinformation",
                "hate_speech",
            ],

            predicted_labels=[
                "true",
                "misinformation",
                "hate_speech",
            ],

            languages=[
                "en",
                "zh",
                "en",
            ],

            domains=[
                "general",
                "international_security",
                "general",
            ],

            source_datasets=[
                "A",
                "B",
                "A",
            ],

            confidences=[
                0.95,
                0.80,
                0.90,
            ],
        )

        self.assertEqual(
            len(errors),
            1,
        )

        self.assertEqual(
            errors[
                0
            ].sample_id,
            "S2",
        )

        self.assertEqual(
            errors[
                0
            ].true_label,
            "disinformation",
        )

        self.assertEqual(
            errors[
                0
            ].predicted_label,
            "misinformation",
        )


class TestEvaluationReport(
    unittest.TestCase
):

    def test_report_save_json(self):

        report = EvaluationReport(
            experiment_name=(
                "evaluation_test"
            ),

            total_samples=2,

            metrics={
                "accuracy": 0.5
            },
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "evaluation.json"
            )

            saved = (
                report.save_json(
                    path
                )
            )

            self.assertTrue(
                saved.is_file()
            )


class TestAEGISEvaluator(
    unittest.TestCase
):

    def test_complete_evaluation(self):

        evaluator = AEGISEvaluator()

        report = evaluator.evaluate(
            experiment_name=(
                "complete_test"
            ),

            sample_ids=[
                "A",
                "B",
                "C",
                "D",
                "E",
            ],

            integrity_targets=[
                0,
                1,
                1,
                1,
                1,
            ],

            integrity_predictions=[
                0,
                1,
                1,
                1,
                0,
            ],

            threat_targets=[
                -1,
                0,
                1,
                2,
                3,
            ],

            threat_predictions=[
                0,
                0,
                1,
                3,
                2,
            ],

            integrity_confidences=[
                0.95,
                0.90,
                0.85,
                0.75,
                0.60,
            ],

            languages=[
                "en",
                "en",
                "zh",
                "zh",
                "en",
            ],

            domains=[
                "general",
                "international_security",
                "international_security",
                "conflict",
                "conflict",
            ],

            source_datasets=[
                "dataset_a",
                "dataset_a",
                "dataset_b",
                "dataset_b",
                "dataset_b",
            ],
        )

        self.assertEqual(
            report.experiment_name,
            "complete_test",
        )

        self.assertEqual(
            report.total_samples,
            5,
        )

        self.assertIn(
            "hierarchical",
            report.metrics,
        )

        self.assertIn(
            "five_class",
            report.metrics,
        )

        self.assertIn(
            "calibration",
            report.metrics,
        )

        self.assertIn(
            "language_integrity",
            report.stratified_metrics,
        )

        self.assertGreater(
            len(
                report.errors
            ),
            0,
        )

        five_class = (
            report.metrics[
                "five_class"
            ]
        )

        self.assertEqual(
            five_class[
                "labels"
            ],
            FIVE_CLASS_LABELS,
        )

        self.assertEqual(
            len(
                five_class[
                    "confusion_matrix"
                ]
            ),
            5,
        )

        self.assertIn(
            "true",
            five_class[
                "per_class"
            ],
        )

        self.assertIn(
            "misinformation",
            five_class[
                "per_class"
            ],
        )

        self.assertIn(
            "disinformation",
            five_class[
                "per_class"
            ],
        )

        self.assertIn(
            "malinformation",
            five_class[
                "per_class"
            ],
        )

        self.assertIn(
            "hate_speech",
            five_class[
                "per_class"
            ],
        )


if __name__ == "__main__":
    unittest.main()