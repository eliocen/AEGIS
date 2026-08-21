"""
Tests for deterministic Fakeddit empirical sampling.

Version: 0.24.0
"""

import unittest

from aegis.data.adapters.empirical import (
    EmpiricalSample,
    ImageStatus,
)

from aegis.data.sampling import (
    select_stratified_fakeddit_samples,
    select_streaming_stratified_fakeddit_samples,
)


class TestFakedditSampling(
    unittest.TestCase
):

    def make_sample(
        self,
        index: int,
        label: int,
    ) -> EmpiricalSample:

        return EmpiricalSample(
            sample_id=(
                f"sample-{index}"
            ),

            dataset="fakeddit",

            split="train",

            text=(
                f"sample text {index}"
            ),

            image_path=(
                f"sample-{index}.jpg"
            ),

            image_status=(
                ImageStatus.AVAILABLE
            ),

            native_labels={
                "2_way_label": label,
                "3_way_label": 0,
                "6_way_label": 0,
            },
        )

    def make_population(
        self,
        count: int = 100,
    ):

        return [
            self.make_sample(
                index=index,
                label=(
                    index % 2
                ),
            )
            for index in range(
                count
            )
        ]

    def test_even_balanced_selection(
        self
    ):

        selection = (
            select_stratified_fakeddit_samples(
                self.make_population(),
                sample_count=8,
                seed=42,
            )
        )

        self.assertEqual(
            selection.actual_count,
            8,
        )

        self.assertEqual(
            selection.native_label_counts,
            {
                0: 4,
                1: 4,
            },
        )

    def test_odd_selection(
        self
    ):

        selection = (
            select_stratified_fakeddit_samples(
                self.make_population(),
                sample_count=7,
                seed=42,
            )
        )

        self.assertEqual(
            selection.native_label_counts,
            {
                0: 3,
                1: 4,
            },
        )

    def test_same_seed_reproducible(
        self
    ):

        population = (
            self.make_population()
        )

        first = (
            select_stratified_fakeddit_samples(
                population,
                sample_count=8,
                seed=123,
            )
        )

        second = (
            select_stratified_fakeddit_samples(
                population,
                sample_count=8,
                seed=123,
            )
        )

        self.assertEqual(
            first.sample_ids,
            second.sample_ids,
        )

    def test_different_seed_changes_selection(
        self
    ):

        population = (
            self.make_population()
        )

        first = (
            select_stratified_fakeddit_samples(
                population,
                sample_count=8,
                seed=1,
            )
        )

        second = (
            select_stratified_fakeddit_samples(
                population,
                sample_count=8,
                seed=2,
            )
        )

        self.assertNotEqual(
            first.sample_ids,
            second.sample_ids,
        )

    def test_invalid_sample_count(
        self
    ):

        with self.assertRaises(
            ValueError
        ):

            select_stratified_fakeddit_samples(
                self.make_population(),
                sample_count=0,
            )

    def test_insufficient_class_population(
        self
    ):

        population = [
            self.make_sample(
                index=index,
                label=0,
            )
            for index in range(
                10
            )
        ]

        with self.assertRaises(
            ValueError
        ):

            select_stratified_fakeddit_samples(
                population,
                sample_count=4,
            )

    def test_duplicate_ids_rejected(
        self
    ):

        sample = (
            self.make_sample(
                index=1,
                label=0,
            )
        )

        with self.assertRaises(
            ValueError
        ):

            select_stratified_fakeddit_samples(
                [
                    sample,
                    sample,
                ],
                sample_count=2,
            )

    def test_streaming_even_balanced(
        self
    ):

        selection = (
            select_streaming_stratified_fakeddit_samples(
                self.make_population(),
                sample_count=10,
                seed=42,
            )
        )

        self.assertEqual(
            selection.actual_count,
            10,
        )

        self.assertEqual(
            selection.native_label_counts,
            {
                0: 5,
                1: 5,
            },
        )

    def test_streaming_reproducible(
        self
    ):

        population = (
            self.make_population()
        )

        first = (
            select_streaming_stratified_fakeddit_samples(
                population,
                sample_count=20,
                seed=2026,
            )
        )

        second = (
            select_streaming_stratified_fakeddit_samples(
                population,
                sample_count=20,
                seed=2026,
            )
        )

        self.assertEqual(
            first.sample_ids,
            second.sample_ids,
        )

    def test_streaming_different_seed(
        self
    ):

        population = (
            self.make_population()
        )

        first = (
            select_streaming_stratified_fakeddit_samples(
                population,
                sample_count=20,
                seed=1,
            )
        )

        second = (
            select_streaming_stratified_fakeddit_samples(
                population,
                sample_count=20,
                seed=2,
            )
        )

        self.assertNotEqual(
            first.sample_ids,
            second.sample_ids,
        )

    def test_streaming_population_count(
        self
    ):

        selection = (
            select_streaming_stratified_fakeddit_samples(
                self.make_population(
                    count=100
                ),
                sample_count=8,
                seed=42,
            )
        )

        self.assertEqual(
            selection.population_seen,
            100,
        )

    def test_streaming_strategy_name(
        self
    ):

        selection = (
            select_streaming_stratified_fakeddit_samples(
                self.make_population(),
                sample_count=8,
            )
        )

        self.assertEqual(
            selection.strategy,
            "streaming_stratified_reservoir",
        )


if __name__ == "__main__":
    unittest.main()