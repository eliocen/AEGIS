"""
Tests for AEGIS v0.21.0:
Ablation Study & Experimental Comparison Framework.
"""

import unittest

import torch

from aegis.ablation import (
    AblationResult,
    AblationSpec,
    AblationStudyRunner,
    MultiSeedAblationRunner,
    SeedRunResult,
    aggregate_seed_runs,
    apply_ablation_to_batch,
    build_comparison_table,
    build_statistical_comparison_table,
    compare_aggregated_to_baseline,
    compare_to_baseline,
    compute_metric_statistics,
    default_ablation_registry,
    rank_results,
)

from aegis.training import (
    TrainingBatch,
)


def build_batch():

    return TrainingBatch(
        text_embeddings=torch.randn(
            4,
            768,
        ),

        vision_embeddings=torch.randn(
            4,
            512,
        ),

        integrity_targets=torch.tensor(
            [
                0,
                1,
                1,
                1,
            ]
        ),

        threat_targets=torch.tensor(
            [
                -1,
                0,
                1,
                3,
            ]
        ),

        sample_ids=[
            "ABL-001",
            "ABL-002",
            "ABL-003",
            "ABL-004",
        ],
    )


class TestAblationSpec(
    unittest.TestCase
):

    def test_baseline_spec(self):

        spec = AblationSpec(
            name="baseline"
        )

        self.assertTrue(
            spec.use_text
        )

        self.assertTrue(
            spec.use_vision
        )

        self.assertTrue(
            spec.is_multimodal
        )

        self.assertFalse(
            spec.is_text_only
        )

        self.assertFalse(
            spec.is_vision_only
        )

    def test_invalid_no_modality_spec(self):

        with self.assertRaises(
            ValueError
        ):

            AblationSpec(
                name="invalid",

                use_text=False,

                use_vision=False,
            )

    def test_invalid_zero_objectives(self):

        with self.assertRaises(
            ValueError
        ):

            AblationSpec(
                name="invalid_objectives",

                alignment_loss_weight=0.0,

                classification_loss_weight=0.0,
            )


class TestDefaultAblationRegistry(
    unittest.TestCase
):

    def test_default_registry(self):

        registry = (
            default_ablation_registry()
        )

        self.assertIn(
            "baseline",
            registry,
        )

        self.assertIn(
            "text_only",
            registry,
        )

        self.assertIn(
            "vision_only",
            registry,
        )

        self.assertIn(
            "no_alignment_loss",
            registry,
        )

        self.assertTrue(
            registry[
                "text_only"
            ].is_text_only
        )

        self.assertTrue(
            registry[
                "vision_only"
            ].is_vision_only
        )


class TestAblationBatch(
    unittest.TestCase
):

    def test_text_only_zeros_vision(self):

        batch = build_batch()

        spec = AblationSpec(
            name="text_only",
            use_text=True,
            use_vision=False,
        )

        transformed = (
            apply_ablation_to_batch(
                batch,
                spec,
            )
        )

        self.assertTrue(
            torch.equal(
                transformed
                .text_embeddings,

                batch
                .text_embeddings,
            )
        )

        self.assertTrue(
            torch.equal(
                transformed
                .vision_embeddings,

                torch.zeros_like(
                    batch
                    .vision_embeddings
                ),
            )
        )

        self.assertEqual(
            transformed.sample_ids,
            batch.sample_ids,
        )

    def test_vision_only_zeros_text(self):

        batch = build_batch()

        spec = AblationSpec(
            name="vision_only",
            use_text=False,
            use_vision=True,
        )

        transformed = (
            apply_ablation_to_batch(
                batch,
                spec,
            )
        )

        self.assertTrue(
            torch.equal(
                transformed
                .text_embeddings,

                torch.zeros_like(
                    batch
                    .text_embeddings
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                transformed
                .vision_embeddings,

                batch
                .vision_embeddings,
            )
        )


class TestAblationComparison(
    unittest.TestCase
):

    def setUp(self):

        self.baseline = (
            AblationResult(
                spec=(
                    AblationSpec(
                        name="baseline"
                    )
                ),

                metrics={
                    "five_class_macro_f1": 0.80,

                    "hierarchical_exact_accuracy": 0.75,
                },
            )
        )

        self.text_only = (
            AblationResult(
                spec=(
                    AblationSpec(
                        name="text_only",

                        use_text=True,

                        use_vision=False,
                    )
                ),

                metrics={
                    "five_class_macro_f1": 0.70,

                    "hierarchical_exact_accuracy": 0.65,
                },
            )
        )

        self.vision_only = (
            AblationResult(
                spec=(
                    AblationSpec(
                        name="vision_only",

                        use_text=False,

                        use_vision=True,
                    )
                ),

                metrics={
                    "five_class_macro_f1": 0.60,

                    "hierarchical_exact_accuracy": 0.55,
                },
            )
        )

    def test_compare_to_baseline(self):

        comparison = (
            compare_to_baseline(
                self.baseline,
                self.text_only,
                "five_class_macro_f1",
            )
        )

        self.assertAlmostEqual(
            comparison.baseline_value,
            0.80,
        )

        self.assertAlmostEqual(
            comparison.variant_value,
            0.70,
        )

        self.assertAlmostEqual(
            comparison.absolute_delta,
            -0.10,
        )

        self.assertAlmostEqual(
            comparison.relative_delta_percent,
            -12.5,
        )

    def test_rank_results(self):

        ranked = rank_results(
            [
                self.text_only,
                self.vision_only,
                self.baseline,
            ],

            metric=(
                "five_class_macro_f1"
            ),

            higher_is_better=True,
        )

        self.assertEqual(
            ranked[
                0
            ].spec.name,
            "baseline",
        )

        self.assertEqual(
            ranked[
                -1
            ].spec.name,
            "vision_only",
        )

    def test_comparison_table(self):

        table = (
            build_comparison_table(
                [
                    self.baseline,
                    self.text_only,
                    self.vision_only,
                ],

                baseline_name=(
                    "baseline"
                ),

                metrics=[
                    "five_class_macro_f1",
                    "hierarchical_exact_accuracy",
                ],
            )
        )

        self.assertEqual(
            len(table),
            3,
        )

        baseline_row = (
            table[
                0
            ]
        )

        self.assertEqual(
            baseline_row[
                "variant"
            ],
            "baseline",
        )

        text_row = next(
            row
            for row in table
            if row[
                "variant"
            ] == "text_only"
        )

        self.assertAlmostEqual(
            text_row[
                "five_class_macro_f1_delta"
            ],
            -0.10,
        )


class TestAblationStudyRunner(
    unittest.TestCase
):

    def test_runner_executes_variants(self):

        def experiment_fn(
            spec,
        ):

            base = 0.80

            if spec.is_text_only:

                base = 0.70

            elif spec.is_vision_only:

                base = 0.60

            elif (
                spec.alignment_loss_weight
                == 0.0
            ):

                base = 0.68

            return {
                "five_class_macro_f1": (
                    base
                ),

                "hierarchical_exact_accuracy": (
                    base - 0.05
                ),
            }

        runner = (
            AblationStudyRunner(
                experiment_fn
            )
        )

        registry = (
            default_ablation_registry()
        )

        specs = [
            registry[
                "baseline"
            ],

            registry[
                "text_only"
            ],

            registry[
                "vision_only"
            ],

            registry[
                "no_alignment_loss"
            ],
        ]

        results = runner.run(
            specs
        )

        self.assertEqual(
            len(results),
            4,
        )

        self.assertEqual(
            results[
                0
            ].spec.name,
            "baseline",
        )

        self.assertEqual(
            results[
                1
            ].metrics[
                "five_class_macro_f1"
            ],
            0.70,
        )

    def test_duplicate_names_fail(self):

        runner = (
            AblationStudyRunner(
                lambda spec: {
                    "metric": 1.0
                }
            )
        )

        spec = AblationSpec(
            name="duplicate"
        )

        with self.assertRaises(
            ValueError
        ):

            runner.run(
                [
                    spec,
                    spec,
                ]
            )


class TestMetricStatistics(
    unittest.TestCase
):

    def test_metric_statistics(self):

        statistics = (
            compute_metric_statistics(
                [
                    0.80,
                    0.82,
                    0.84,
                ]
            )
        )

        self.assertEqual(
            statistics.count,
            3,
        )

        self.assertAlmostEqual(
            statistics.mean,
            0.82,
            places=6,
        )

        self.assertAlmostEqual(
            statistics
            .standard_deviation,
            0.02,
            places=6,
        )

        self.assertEqual(
            statistics.minimum,
            0.80,
        )

        self.assertEqual(
            statistics.maximum,
            0.84,
        )

    def test_single_observation_std_zero(self):

        statistics = (
            compute_metric_statistics(
                [
                    0.75
                ]
            )
        )

        self.assertEqual(
            statistics
            .standard_deviation,
            0.0,
        )


class TestMultiSeedAblationRunner(
    unittest.TestCase
):

    def test_multi_seed_execution(self):

        registry = (
            default_ablation_registry()
        )

        specs = [
            registry[
                "baseline"
            ],

            registry[
                "text_only"
            ],
        ]

        seeds = [
            42,
            123,
            456,
        ]

        def experiment_fn(
            spec,
            seed,
        ):

            base = (
                0.80
                if spec.name
                == "baseline"
                else 0.70
            )

            seed_adjustment = (
                (
                    seed % 10
                )
                * 0.001
            )

            return {
                "five_class_macro_f1": (
                    base
                    + seed_adjustment
                )
            }

        runner = (
            MultiSeedAblationRunner(
                experiment_fn
            )
        )

        results = runner.run(
            specs,
            seeds,
        )

        self.assertEqual(
            len(results),
            6,
        )

        baseline_results = [
            result
            for result in results
            if result.variant
            == "baseline"
        ]

        self.assertEqual(
            len(
                baseline_results
            ),
            3,
        )

        self.assertEqual(
            {
                result.seed
                for result
                in baseline_results
            },
            {
                42,
                123,
                456,
            },
        )

    def test_duplicate_seeds_fail(self):

        runner = (
            MultiSeedAblationRunner(
                lambda spec, seed: {
                    "metric": 1.0
                }
            )
        )

        with self.assertRaises(
            ValueError
        ):

            runner.run(
                [
                    AblationSpec(
                        name="baseline"
                    )
                ],

                [
                    42,
                    42,
                ],
            )


class TestMultiSeedAggregation(
    unittest.TestCase
):

    def build_seed_results(self):

        return [
            SeedRunResult(
                variant="baseline",
                seed=42,
                metrics={
                    "five_class_macro_f1": 0.80
                },
            ),

            SeedRunResult(
                variant="baseline",
                seed=123,
                metrics={
                    "five_class_macro_f1": 0.82
                },
            ),

            SeedRunResult(
                variant="baseline",
                seed=456,
                metrics={
                    "five_class_macro_f1": 0.84
                },
            ),

            SeedRunResult(
                variant="text_only",
                seed=42,
                metrics={
                    "five_class_macro_f1": 0.70
                },
            ),

            SeedRunResult(
                variant="text_only",
                seed=123,
                metrics={
                    "five_class_macro_f1": 0.72
                },
            ),

            SeedRunResult(
                variant="text_only",
                seed=456,
                metrics={
                    "five_class_macro_f1": 0.74
                },
            ),
        ]

    def test_aggregate_seed_runs(self):

        aggregated = (
            aggregate_seed_runs(
                self.build_seed_results()
            )
        )

        self.assertEqual(
            len(
                aggregated
            ),
            2,
        )

        baseline = next(
            result
            for result in aggregated
            if result.variant
            == "baseline"
        )

        statistics = (
            baseline.metrics[
                "five_class_macro_f1"
            ]
        )

        self.assertAlmostEqual(
            statistics.mean,
            0.82,
            places=6,
        )

        self.assertAlmostEqual(
            statistics
            .standard_deviation,
            0.02,
            places=6,
        )

        self.assertEqual(
            baseline.seeds,
            [
                42,
                123,
                456,
            ],
        )

    def test_compare_aggregated_to_baseline(self):

        aggregated = (
            aggregate_seed_runs(
                self.build_seed_results()
            )
        )

        comparisons = (
            compare_aggregated_to_baseline(
                aggregated,

                baseline_name=(
                    "baseline"
                ),

                metric=(
                    "five_class_macro_f1"
                ),
            )
        )

        text_only = next(
            result
            for result
            in comparisons
            if result[
                "variant"
            ]
            == "text_only"
        )

        self.assertAlmostEqual(
            text_only[
                "mean"
            ],
            0.72,
            places=6,
        )

        self.assertAlmostEqual(
            text_only[
                "absolute_delta"
            ],
            -0.10,
            places=6,
        )

    def test_statistical_comparison_table(self):

        aggregated = (
            aggregate_seed_runs(
                self.build_seed_results()
            )
        )

        table = (
            build_statistical_comparison_table(
                aggregated,

                baseline_name=(
                    "baseline"
                ),

                metrics=[
                    "five_class_macro_f1"
                ],
            )
        )

        self.assertEqual(
            len(table),
            2,
        )

        baseline = next(
            row
            for row in table
            if row[
                "variant"
            ]
            == "baseline"
        )

        text_only = next(
            row
            for row in table
            if row[
                "variant"
            ]
            == "text_only"
        )

        self.assertEqual(
            baseline[
                "run_count"
            ],
            3,
        )

        self.assertAlmostEqual(
            baseline[
                "five_class_macro_f1_mean"
            ],
            0.82,
            places=6,
        )

        self.assertAlmostEqual(
            baseline[
                "five_class_macro_f1_std"
            ],
            0.02,
            places=6,
        )

        self.assertAlmostEqual(
            text_only[
                "five_class_macro_f1_delta"
            ],
            -0.10,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()