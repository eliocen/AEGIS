"""
Tests for deterministic Fakeddit empirical sampling.

Version: 0.24.0
"""

import tempfile
import unittest

from pathlib import Path

from PIL import Image

from aegis.data.adapters.empirical import (
    EmpiricalSample,
    ImageStatus,
)

from aegis.data.sampling import (
    select_stratified_fakeddit_samples,
    select_streaming_decodable_fakeddit_samples,
    select_streaming_stratified_fakeddit_samples,
)


class TestFakedditSampling(
    unittest.TestCase
):

    def make_sample(
        self,
        index: int,
        label: int,
        image_path=None,
    ) -> EmpiricalSample:

        if image_path is None:
            image_path = (
                f"sample-{index}.jpg"
            )

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
                image_path
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


class TestDecodableFakedditSampling(
    unittest.TestCase
):

    def setUp(
        self
    ):

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_directory.name
        )

    def tearDown(
        self
    ):

        self.temp_directory.cleanup()

    def create_valid_image(
        self,
        name: str,
    ) -> Path:

        path = (
            self.root
            / name
        )

        image = Image.new(
            "RGB",
            (
                16,
                16,
            ),
        )

        image.save(
            path,
            format="JPEG",
        )

        return path

    def create_invalid_image(
        self,
        name: str,
    ) -> Path:

        path = (
            self.root
            / name
        )

        path.write_text(
            "<html>invalid image</html>",
            encoding="utf-8",
        )

        return path

    def make_decodable_population(
        self
    ):

        samples = []

        # Four samples per class.
        # One corrupt image per class.
        # Asking for two per class therefore
        # leaves deterministic replacement capacity.

        for index in range(
            8
        ):

            label = (
                index % 2
            )

            if index in {
                0,
                1,
            }:

                image_path = (
                    self.create_invalid_image(
                        f"sample-{index}.jpg"
                    )
                )

            else:

                image_path = (
                    self.create_valid_image(
                        f"sample-{index}.jpg"
                    )
                )

            samples.append(
                EmpiricalSample(
                    sample_id=(
                        f"sample-{index}"
                    ),

                    dataset="fakeddit",

                    split="train",

                    text=(
                        f"text {index}"
                    ),

                    image_path=(
                        image_path
                    ),

                    image_status=(
                        ImageStatus.AVAILABLE
                    ),

                    native_labels={
                        "2_way_label": (
                            label
                        ),
                        "3_way_label": 0,
                        "6_way_label": 0,
                    },
                )
            )

        return samples

    def test_invalid_images_are_excluded(
        self
    ):

        selection = (
            select_streaming_decodable_fakeddit_samples(
                self.make_decodable_population(),
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        self.assertEqual(
            selection.actual_count,
            4,
        )

        self.assertGreaterEqual(
            selection.exclusion_count,
            2,
        )

        selected_ids = set(
            selection.sample_ids
        )

        self.assertNotIn(
            "sample-0",
            selected_ids,
        )

        self.assertNotIn(
            "sample-1",
            selected_ids,
        )

    def test_decodable_selection_balanced(
        self
    ):

        selection = (
            select_streaming_decodable_fakeddit_samples(
                self.make_decodable_population(),
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        self.assertEqual(
            selection.native_label_counts,
            {
                0: 2,
                1: 2,
            },
        )

    def test_decodable_selection_reproducible(
        self
    ):

        population = (
            self.make_decodable_population()
        )

        first = (
            select_streaming_decodable_fakeddit_samples(
                population,
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        second = (
            select_streaming_decodable_fakeddit_samples(
                population,
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        self.assertEqual(
            first.sample_ids,
            second.sample_ids,
        )

    def test_decodable_strategy_name(
        self
    ):

        selection = (
            select_streaming_decodable_fakeddit_samples(
                self.make_decodable_population(),
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        self.assertEqual(
            selection.strategy,
            (
                "streaming_stratified_"
                "reservoir_decodable"
            ),
        )

    def test_exclusion_contains_reason(
        self
    ):

        selection = (
            select_streaming_decodable_fakeddit_samples(
                self.make_decodable_population(),
                sample_count=4,
                seed=42,
                reserve_fraction=0.0,
                min_reserve_per_class=2,
            )
        )

        reasons = {
            exclusion.reason
            for exclusion
            in selection.exclusions
        }

        self.assertIn(
            "unidentified_image",
            reasons,
        )


if __name__ == "__main__":
    unittest.main()