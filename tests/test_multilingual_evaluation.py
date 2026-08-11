"""
Tests for AEGIS v0.22.0:
Multilingual & Cross-Lingual Evaluation Framework.
"""

import tempfile
import unittest

from pathlib import Path

from aegis.multilingual_evaluation import (
    CrossLingualRunResult,
    LanguagePair,
    MultilingualEvaluationReport,
    MultilingualEvaluator,
    build_transfer_matrix,
    language_performance_gap,
    normalize_language,
    summarize_cross_lingual_transfer,
)


class TestLanguageNormalization(
    unittest.TestCase
):

    def test_english_normalization(self):

        self.assertEqual(
            normalize_language(
                "English"
            ),
            "en",
        )

        self.assertEqual(
            normalize_language(
                "eng"
            ),
            "en",
        )

    def test_chinese_normalization(self):

        self.assertEqual(
            normalize_language(
                "Chinese"
            ),
            "zh",
        )

        self.assertEqual(
            normalize_language(
                "zh-CN"
            ),
            "zh",
        )

    def test_unknown_language_preserved(self):

        self.assertEqual(
            normalize_language(
                "fr"
            ),
            "fr",
        )


class TestLanguagePair(
    unittest.TestCase
):

    def test_in_language_pair(self):

        pair = LanguagePair(
            train_language="English",
            test_language="en",
        )

        self.assertTrue(
            pair.is_in_language
        )

        self.assertFalse(
            pair.is_cross_lingual
        )

        self.assertEqual(
            pair.key,
            "en->en",
        )

    def test_cross_lingual_pair(self):

        pair = LanguagePair(
            train_language="en",
            test_language="Chinese",
        )

        self.assertTrue(
            pair.is_cross_lingual
        )

        self.assertEqual(
            pair.key,
            "en->zh",
        )


class TestMultilingualEvaluator(
    unittest.TestCase
):

    def build_results(self):

        evaluator = (
            MultilingualEvaluator()
        )

        return evaluator.evaluate(
            languages=[
                "en",
                "en",
                "en",
                "en",
                "en",
                "zh",
                "zh",
                "zh",
                "zh",
                "zh",
            ],

            integrity_targets=[
                0,
                1,
                1,
                1,
                1,
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
                0,
                1,
                1,
                0,
                1,
            ],

            threat_targets=[
                -1,
                0,
                1,
                2,
                3,
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
                0,
                0,
                1,
                3,
                3,
            ],
        )

    def test_language_groups_created(self):

        results = (
            self.build_results()
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
            ].sample_count,
            5,
        )

        self.assertEqual(
            results[
                "zh"
            ].sample_count,
            5,
        )

    def test_language_five_class_metrics(self):

        results = (
            self.build_results()
        )

        english = (
            results[
                "en"
            ]
        )

        self.assertIn(
            "five_class",
            english.metrics,
        )

        self.assertEqual(
            english.metrics[
                "five_class"
            ][
                "accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            len(
                english.metrics[
                    "five_class"
                ][
                    "confusion_matrix"
                ]
            ),
            5,
        )

    def test_language_performance_gap(self):

        results = (
            self.build_results()
        )

        gap = (
            language_performance_gap(
                results
            )
        )

        self.assertEqual(
            gap[
                "best_language"
            ],
            "en",
        )

        self.assertEqual(
            gap[
                "worst_language"
            ],
            "zh",
        )

        self.assertGreater(
            gap[
                "absolute_gap"
            ],
            0.0,
        )


class TestCrossLingualTransferMatrix(
    unittest.TestCase
):

    def build_results(self):

        return [
            CrossLingualRunResult(
                train_language="en",
                test_language="en",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.84,
                seed=42,
            ),

            CrossLingualRunResult(
                train_language="en",
                test_language="en",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.82,
                seed=123,
            ),

            CrossLingualRunResult(
                train_language="en",
                test_language="zh",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.71,
                seed=42,
            ),

            CrossLingualRunResult(
                train_language="en",
                test_language="zh",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.69,
                seed=123,
            ),

            CrossLingualRunResult(
                train_language="zh",
                test_language="en",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.73,
                seed=42,
            ),

            CrossLingualRunResult(
                train_language="zh",
                test_language="en",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.71,
                seed=123,
            ),

            CrossLingualRunResult(
                train_language="zh",
                test_language="zh",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.81,
                seed=42,
            ),

            CrossLingualRunResult(
                train_language="zh",
                test_language="zh",
                metric_name=(
                    "five_class_macro_f1"
                ),
                metric_value=0.79,
                seed=123,
            ),
        ]

    def test_transfer_matrix(self):

        matrix = build_transfer_matrix(
            self.build_results(),

            languages=[
                "en",
                "zh",
            ],
        )

        self.assertEqual(
            matrix[
                "languages"
            ],
            [
                "en",
                "zh",
            ],
        )

        self.assertEqual(
            len(
                matrix[
                    "matrix"
                ]
            ),
            2,
        )

        self.assertEqual(
            len(
                matrix[
                    "matrix"
                ][0]
            ),
            2,
        )

        self.assertAlmostEqual(
            matrix[
                "matrix"
            ][0][0],
            0.83,
            places=6,
        )

        self.assertAlmostEqual(
            matrix[
                "matrix"
            ][0][1],
            0.70,
            places=6,
        )

        self.assertAlmostEqual(
            matrix[
                "matrix"
            ][1][0],
            0.72,
            places=6,
        )

        self.assertAlmostEqual(
            matrix[
                "matrix"
            ][1][1],
            0.80,
            places=6,
        )

    def test_transfer_summary(self):

        summary = (
            summarize_cross_lingual_transfer(
                self.build_results()
            )
        )

        self.assertAlmostEqual(
            summary[
                "in_language_mean"
            ],
            0.815,
            places=6,
        )

        self.assertAlmostEqual(
            summary[
                "cross_lingual_mean"
            ],
            0.71,
            places=6,
        )

        self.assertAlmostEqual(
            summary[
                "absolute_transfer_gap"
            ],
            -0.105,
            places=6,
        )

        self.assertEqual(
            summary[
                "in_language_runs"
            ],
            4,
        )

        self.assertEqual(
            summary[
                "cross_lingual_runs"
            ],
            4,
        )


class TestMultilingualEvaluationReport(
    unittest.TestCase
):

    def test_report_save_json(self):

        evaluator = (
            MultilingualEvaluator()
        )

        language_results = (
            evaluator.evaluate(
                languages=[
                    "en",
                    "zh",
                ],

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
                    1,
                ],
            )
        )

        report = (
            MultilingualEvaluationReport(
                experiment_name=(
                    "multilingual_test"
                ),

                language_results=(
                    language_results
                ),

                metadata={
                    "version": "0.22.0"
                },
            )
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "multilingual.json"
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