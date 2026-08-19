"""
Tests for the complete AEGIS Fakeddit dataset audit
and reproducible manifest generator.
"""

import json
import tempfile
import unittest

from pathlib import Path

import pandas as pd

from aegis.data.audit import (
    FakedditFullAuditor,
    sha256_file,
)


class FakedditAuditTestEnvironment:

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

        self.output_root = (
            root
            / "manifests"
        )

        self.multimodal_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.train = pd.DataFrame(
            [
                {
                    "id": "train001",
                    "clean_title": "valid training sample",
                    "hasImage": True,
                    "domain": "example.org",
                    "subreddit": "news",
                    "image_url": "https://example.org/1.jpg",
                    "2_way_label": 1,
                    "3_way_label": 0,
                    "6_way_label": 0,
                },
                {
                    "id": "train002",
                    "clean_title": "second training sample",
                    "hasImage": True,
                    "domain": "example.org",
                    "subreddit": "news",
                    "image_url": "https://example.org/2.jpg",
                    "2_way_label": 0,
                    "3_way_label": 2,
                    "6_way_label": 4,
                },
            ]
        )

        self.validation = pd.DataFrame(
            [
                {
                    "id": "valid001",
                    "clean_title": "validation sample",
                    "hasImage": True,
                    "domain": "example.net",
                    "subreddit": "world",
                    "image_url": "https://example.org/3.jpg",
                    "2_way_label": 1,
                    "3_way_label": 0,
                    "6_way_label": 0,
                },
            ]
        )

        self.test = pd.DataFrame(
            [
                {
                    "id": "test001",
                    "clean_title": "test sample",
                    "hasImage": True,
                    "domain": "example.com",
                    "subreddit": "science",
                    "image_url": "https://example.org/4.jpg",
                    "2_way_label": 0,
                    "3_way_label": 2,
                    "6_way_label": 2,
                },
            ]
        )

        mapping = {
            "multimodal_train.tsv": (
                self.train
            ),

            "multimodal_validate.tsv": (
                self.validation
            ),

            "multimodal_test_public.tsv": (
                self.test
            ),
        }

        required_extra_columns = {
            "author": "tester",
            "created_utc": 123456,
            "linked_submission_id": None,
            "num_comments": 1,
            "score": 1,
            "title": "Title",
            "upvote_ratio": 0.9,
        }

        for filename, frame in (
            mapping.items()
        ):

            prepared = (
                frame.copy()
            )

            for column, value in (
                required_extra_columns.items()
            ):

                prepared[
                    column
                ] = value

            prepared.to_csv(
                self.multimodal_root
                / filename,
                sep="\t",
                index=False,
            )

        for sample_id in (
            "train001",
            "train002",
            "valid001",
            "test001",
        ):

            (
                self.image_root
                / f"{sample_id}.jpg"
            ).write_bytes(
                b"test-image"
            )


class TestSHA256(
    unittest.TestCase
):

    def test_sha256_is_reproducible(
        self,
    ):

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "test.txt"
            )

            path.write_text(
                "AEGIS",
                encoding="utf-8",
            )

            first = (
                sha256_file(
                    path
                )
            )

            second = (
                sha256_file(
                    path
                )
            )

            self.assertEqual(
                first,
                second,
            )

            self.assertEqual(
                len(first),
                64,
            )


class TestFakedditFullAuditor(
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
            FakedditAuditTestEnvironment(
                self.root
            )
        )

        self.auditor = (
            FakedditFullAuditor(
                dataset_root=(
                    self.root
                ),

                output_root=(
                    self.environment
                    .output_root
                ),

                chunksize=1,

                top_k_categories=5,
            )
        )

    def tearDown(
        self,
    ) -> None:

        self.temp_directory.cleanup()

    def test_train_audit_counts(
        self,
    ):

        result, ids, exclusions = (
            self.auditor.audit_split(
                "train"
            )
        )

        self.assertEqual(
            result.total_rows,
            2,
        )

        self.assertEqual(
            result.unique_ids,
            2,
        )

        self.assertEqual(
            result.image_available,
            2,
        )

        self.assertEqual(
            result.image_missing,
            0,
        )

        self.assertEqual(
            result.eligible_strict_multimodal,
            2,
        )

        self.assertEqual(
            len(ids),
            2,
        )

        self.assertEqual(
            exclusions,
            [],
        )

    def test_native_label_distribution(
        self,
    ):

        result, _, _ = (
            self.auditor.audit_split(
                "train"
            )
        )

        self.assertEqual(
            result
            .native_label_distributions[
                "2_way_label"
            ],
            {
                "0": 1,
                "1": 1,
            },
        )

        self.assertEqual(
            result
            .native_label_distributions[
                "6_way_label"
            ],
            {
                "0": 1,
                "4": 1,
            },
        )

    def test_image_coverage(
        self,
    ):

        result, _, _ = (
            self.auditor.audit_split(
                "train"
            )
        )

        self.assertEqual(
            result.image_coverage,
            1.0,
        )

    def test_no_cross_split_leakage(
        self,
    ):

        result = (
            self.auditor.audit_cross_split(
                train_ids={
                    "train001",
                    "train002",
                },

                validation_ids={
                    "valid001"
                },

                test_ids={
                    "test001"
                },
            )
        )

        self.assertFalse(
            result.any_cross_split_leakage
        )

        self.assertEqual(
            result.train_validation_overlap,
            0,
        )

        self.assertEqual(
            result.train_test_overlap,
            0,
        )

        self.assertEqual(
            result.validation_test_overlap,
            0,
        )

    def test_cross_split_leakage_detected(
        self,
    ):

        result = (
            self.auditor.audit_cross_split(
                train_ids={
                    "shared001",
                    "train002",
                },

                validation_ids={
                    "shared001"
                },

                test_ids={
                    "test001"
                },
            )
        )

        self.assertTrue(
            result.any_cross_split_leakage
        )

        self.assertEqual(
            result.train_validation_overlap,
            1,
        )

        self.assertEqual(
            result.train_validation_ids,
            [
                "shared001"
            ],
        )

    def test_full_audit_passes(
        self,
    ):

        audit = (
            self.auditor.run()
        )

        self.assertEqual(
            audit.integrity_status,
            "passed",
        )

        self.assertEqual(
            audit.dataset_totals[
                "total_rows"
            ],
            4,
        )

        self.assertEqual(
            audit.dataset_totals[
                "strict_multimodal_eligible"
            ],
            4,
        )

        self.assertFalse(
            audit.cross_split[
                "any_cross_split_leakage"
            ]
        )

    def test_manifest_files_created(
        self,
    ):

        self.auditor.run()

        expected_files = [
            "dataset_audit.json",
            "split_statistics.json",
            "provenance.json",
            "excluded_samples.jsonl",
        ]

        for filename in (
            expected_files
        ):

            path = (
                self.environment
                .output_root
                / filename
            )

            self.assertTrue(
                path.is_file()
            )

    def test_dataset_audit_json_valid(
        self,
    ):

        self.auditor.run()

        path = (
            self.environment
            .output_root
            / "dataset_audit.json"
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

        self.assertEqual(
            data[
                "dataset"
            ],
            "Fakeddit",
        )

        self.assertIn(
            "train",
            data[
                "splits"
            ],
        )

        self.assertIn(
            "validation",
            data[
                "splits"
            ],
        )

        self.assertIn(
            "test",
            data[
                "splits"
            ],
        )

    def test_provenance_contains_hashes(
        self,
    ):

        self.auditor.run()

        path = (
            self.environment
            .output_root
            / "provenance.json"
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

        self.assertEqual(
            len(
                data[
                    "source_files"
                ][
                    "train"
                ][
                    "sha256"
                ]
            ),
            64,
        )


if __name__ == "__main__":
    unittest.main()