"""
Tests for the AEGIS Fakeddit empirical dataset adapter.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from aegis.data.adapters import (
    EmpiricalSample,
    FakedditAdapter,
    ImageStatus,
)


class FakedditTestEnvironment:
    """
    Temporary miniature Fakeddit installation used for unit tests.

    Real research data is deliberately not required for the
    unit-test suite.
    """

    def __init__(
        self,
        root: Path,
    ) -> None:

        self.root = root

        self.multimodal_root = (
            root
            / "multimodal_only_samples"
        )

        self.image_root = (
            root
            / "images"
            / "public_image_set"
        )

        self.multimodal_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.frame = pd.DataFrame(
            [
                {
                    "author": "researcher",
                    "clean_title": "example multimodal post",
                    "created_utc": 1234567890,
                    "domain": "example.org",
                    "hasImage": True,
                    "id": "sample001",
                    "image_url": "https://example.org/sample001.jpg",
                    "linked_submission_id": None,
                    "num_comments": 3,
                    "score": 10,
                    "subreddit": "example",
                    "title": "Example Multimodal Post",
                    "upvote_ratio": 0.9,
                    "2_way_label": 1,
                    "3_way_label": 0,
                    "6_way_label": 0,
                },
                {
                    "author": "researcher2",
                    "clean_title": "sample with missing local image",
                    "created_utc": 1234567891,
                    "domain": "example.org",
                    "hasImage": True,
                    "id": "sample002",
                    "image_url": "https://example.org/sample002.jpg",
                    "linked_submission_id": None,
                    "num_comments": 1,
                    "score": 5,
                    "subreddit": "example",
                    "title": "Missing Image Sample",
                    "upvote_ratio": 0.8,
                    "2_way_label": 0,
                    "3_way_label": 2,
                    "6_way_label": 2,
                },
            ]
        )

        for filename in (
            "multimodal_train.tsv",
            "multimodal_validate.tsv",
            "multimodal_test_public.tsv",
        ):

            self.frame.to_csv(
                self.multimodal_root
                / filename,
                sep="\t",
                index=False,
            )

        (
            self.image_root
            / "sample001.jpg"
        ).write_bytes(
            b"test-image"
        )


class TestFakedditAdapter(
    unittest.TestCase
):

    def setUp(
        self,
    ) -> None:

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temp_directory.name
        )

        self.environment = (
            FakedditTestEnvironment(
                self.root
            )
        )

        self.adapter = (
            FakedditAdapter(
                self.root
            )
        )

    def tearDown(
        self,
    ) -> None:

        self.temp_directory.cleanup()

    def test_structure_validation(
        self,
    ):

        self.adapter.validate_structure()

    def test_split_paths(
        self,
    ):

        self.assertEqual(
            self.adapter.split_path(
                "train"
            ).name,
            "multimodal_train.tsv",
        )

        self.assertEqual(
            self.adapter.split_path(
                "validation"
            ).name,
            "multimodal_validate.tsv",
        )

        self.assertEqual(
            self.adapter.split_path(
                "test"
            ).name,
            "multimodal_test_public.tsv",
        )

    def test_split_aliases(
        self,
    ):

        self.assertEqual(
            self.adapter.canonical_split(
                "val"
            ),
            "validation",
        )

        self.assertEqual(
            self.adapter.canonical_split(
                "test_public"
            ),
            "test",
        )

    def test_invalid_split_fails(
        self,
    ):

        with self.assertRaises(
            ValueError
        ):
            self.adapter.split_path(
                "unknown"
            )

    def test_schema_validation(
        self,
    ):

        columns = (
            self.adapter.inspect_schema(
                "train"
            )
        )

        self.assertIn(
            "clean_title",
            columns,
        )

        self.assertIn(
            "id",
            columns,
        )

        self.assertIn(
            "6_way_label",
            columns,
        )

    def test_existing_image_resolves(
        self,
    ):

        result = (
            self.adapter.resolve_image(
                "sample001"
            )
        )

        self.assertIsNotNone(
            result
        )

        self.assertTrue(
            result.is_file()
        )

        self.assertEqual(
            result.name,
            "sample001.jpg",
        )

    def test_missing_image_returns_none(
        self,
    ):

        result = (
            self.adapter.resolve_image(
                "sample002"
            )
        )

        self.assertIsNone(
            result
        )

    def test_available_image_status(
        self,
    ):

        status = (
            self.adapter.image_status(
                "sample001",
                has_image=True,
            )
        )

        self.assertEqual(
            status,
            ImageStatus.AVAILABLE,
        )

    def test_missing_image_status(
        self,
    ):

        status = (
            self.adapter.image_status(
                "sample002",
                has_image=True,
            )
        )

        self.assertEqual(
            status,
            ImageStatus.MISSING,
        )

    def test_not_applicable_image_status(
        self,
    ):

        status = (
            self.adapter.image_status(
                "sample002",
                has_image=False,
            )
        )

        self.assertEqual(
            status,
            ImageStatus.NOT_APPLICABLE,
        )

    def test_row_conversion(
        self,
    ):

        row = (
            self.environment
            .frame
            .iloc[0]
        )

        sample = (
            self.adapter.row_to_sample(
                row,
                "train",
            )
        )

        self.assertIsInstance(
            sample,
            EmpiricalSample,
        )

        self.assertEqual(
            sample.sample_id,
            "sample001",
        )

        self.assertEqual(
            sample.dataset,
            "fakeddit",
        )

        self.assertEqual(
            sample.split,
            "train",
        )

        self.assertEqual(
            sample.text,
            "example multimodal post",
        )

        self.assertEqual(
            sample.image_status,
            ImageStatus.AVAILABLE,
        )

    def test_native_labels_preserved(
        self,
    ):

        row = (
            self.environment
            .frame
            .iloc[0]
        )

        sample = (
            self.adapter.row_to_sample(
                row,
                "train",
            )
        )

        self.assertEqual(
            sample.native_labels[
                "2_way_label"
            ],
            1,
        )

        self.assertEqual(
            sample.native_labels[
                "3_way_label"
            ],
            0,
        )

        self.assertEqual(
            sample.native_labels[
                "6_way_label"
            ],
            0,
        )

        self.assertIsNone(
            sample.harmonized_label
        )

        self.assertEqual(
            sample.harmonization_status,
            "not_harmonized",
        )

    def test_missing_local_image_preserved(
        self,
    ):

        row = (
            self.environment
            .frame
            .iloc[1]
        )

        sample = (
            self.adapter.row_to_sample(
                row,
                "train",
            )
        )

        self.assertEqual(
            sample.image_status,
            ImageStatus.MISSING,
        )

        self.assertIsNone(
            sample.image_path
        )

    def test_provenance_preserved(
        self,
    ):

        row = (
            self.environment
            .frame
            .iloc[0]
        )

        sample = (
            self.adapter.row_to_sample(
                row,
                "train",
            )
        )

        self.assertEqual(
            sample.provenance[
                "dataset"
            ],
            "Fakeddit",
        )

        self.assertEqual(
            sample.provenance[
                "subset"
            ],
            "multimodal_only_samples",
        )

        self.assertEqual(
            sample.provenance[
                "text_field"
            ],
            "clean_title",
        )

        self.assertEqual(
            sample.provenance[
                "label_status"
            ],
            "native_unharmonized",
        )

    def test_streaming_iteration(
        self,
    ):

        samples = list(
            self.adapter.iter_samples(
                "train",
                chunksize=1,
                limit=2,
            )
        )

        self.assertEqual(
            len(samples),
            2,
        )

        self.assertEqual(
            samples[0].sample_id,
            "sample001",
        )

        self.assertEqual(
            samples[1].sample_id,
            "sample002",
        )


class TestEmpiricalSample(
    unittest.TestCase
):

    def test_empty_sample_id_fails(
        self,
    ):

        with self.assertRaises(
            ValueError
        ):
            EmpiricalSample(
                sample_id="",
                dataset="test",
                split="train",
                text="valid text",
                image_path=None,
                image_status=(
                    ImageStatus.MISSING
                ),
                native_labels={},
            )

    def test_empty_text_fails(
        self,
    ):

        with self.assertRaises(
            ValueError
        ):
            EmpiricalSample(
                sample_id="sample",
                dataset="test",
                split="train",
                text="",
                image_path=None,
                image_status=(
                    ImageStatus.MISSING
                ),
                native_labels={},
            )


if __name__ == "__main__":
    unittest.main()