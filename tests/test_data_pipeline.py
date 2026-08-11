"""
Tests for AEGIS v0.17.0:
Research Dataset & DataLoader Infrastructure.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from aegis.data import (
    AEGISResearchDataset,
    DatasetLabel,
    ResearchSample,
    aegis_collate_fn,
    build_split_manifest,
    compute_dataset_statistics,
    create_dataloader,
    group_aware_split,
    label_to_hierarchical_targets,
    load_split_manifest,
    normalize_dataset_label,
    random_split_samples,
    save_split_manifest,
    validate_research_sample,
)

from aegis.training import (
    TrainingBatch,
)


def build_sample(
    index: int,
    label=DatasetLabel.TRUE,
    group_id=None,
):
    """
    Construct a valid synthetic research sample.
    """

    return ResearchSample(
        sample_id=(
            f"SAMPLE-{index:03d}"
        ),

        text_embedding=torch.randn(
            768
        ),

        vision_embedding=torch.randn(
            512
        ),

        label=label,

        language=(
            "en"
            if index % 2 == 0
            else "zh"
        ),

        domain=(
            "international_security"
        ),

        source_dataset=(
            "unit_test"
        ),

        group_id=(
            group_id
            if group_id is not None
            else f"GROUP-{index}"
        ),

        source_id=(
            f"SOURCE-{index % 4}"
        ),

        event_id=(
            f"EVENT-{index % 5}"
        ),
    )


def build_dataset_samples(
    count=20,
):
    """
    Build a balanced synthetic sample collection
    across all five AEGIS labels.
    """

    labels = [
        DatasetLabel.TRUE,
        DatasetLabel.MISINFORMATION,
        DatasetLabel.DISINFORMATION,
        DatasetLabel.MALINFORMATION,
        DatasetLabel.HATE_SPEECH,
    ]

    samples = []

    for index in range(
        count
    ):

        samples.append(
            build_sample(
                index=index,

                label=(
                    labels[
                        index
                        % len(labels)
                    ]
                ),

                group_id=(
                    f"GROUP-{index // 2}"
                ),
            )
        )

    return samples


class TestDatasetLabels(
    unittest.TestCase
):

    def test_normalize_label(self):

        self.assertEqual(
            normalize_dataset_label(
                "Disinfo"
            ),
            DatasetLabel.DISINFORMATION,
        )

        self.assertEqual(
            normalize_dataset_label(
                "hate speech"
            ),
            DatasetLabel.HATE_SPEECH,
        )

    def test_true_hierarchical_targets(self):

        integrity, threat = (
            label_to_hierarchical_targets(
                DatasetLabel.TRUE
            )
        )

        self.assertEqual(
            integrity,
            0,
        )

        self.assertEqual(
            threat,
            -1,
        )

    def test_harmful_hierarchical_targets(self):

        integrity, threat = (
            label_to_hierarchical_targets(
                DatasetLabel.MALINFORMATION
            )
        )

        self.assertEqual(
            integrity,
            1,
        )

        self.assertEqual(
            threat,
            2,
        )

    def test_unknown_label_fails(self):

        with self.assertRaises(
            ValueError
        ):
            normalize_dataset_label(
                "unsupported-label"
            )


class TestResearchSample(
    unittest.TestCase
):

    def test_sample_validation(self):

        sample = build_sample(
            index=1,
            label=(
                DatasetLabel
                .MISINFORMATION
            ),
        )

        self.assertTrue(
            validate_research_sample(
                sample
            )
        )

    def test_invalid_embedding_dimension(self):

        sample = ResearchSample(
            sample_id="BAD-001",

            text_embedding=torch.randn(
                100
            ),

            vision_embedding=torch.randn(
                512
            ),

            label=DatasetLabel.TRUE,
        )

        with self.assertRaises(
            ValueError
        ):
            validate_research_sample(
                sample
            )


class TestAEGISResearchDataset(
    unittest.TestCase
):

    def test_dataset_creation(self):

        samples = (
            build_dataset_samples(
                count=10
            )
        )

        dataset = (
            AEGISResearchDataset(
                samples
            )
        )

        self.assertEqual(
            len(dataset),
            10,
        )

        self.assertEqual(
            len(
                dataset.sample_ids
            ),
            10,
        )

    def test_duplicate_sample_ids_fail(self):

        sample = build_sample(
            index=1
        )

        with self.assertRaises(
            ValueError
        ):
            AEGISResearchDataset(
                [
                    sample,
                    sample,
                ]
            )


class TestAEGISCollator(
    unittest.TestCase
):

    def test_collate_returns_training_batch(self):

        samples = [
            build_sample(
                index=0,
                label=(
                    DatasetLabel.TRUE
                ),
            ),

            build_sample(
                index=1,
                label=(
                    DatasetLabel.DISINFORMATION
                ),
            ),

            build_sample(
                index=2,
                label=(
                    DatasetLabel.HATE_SPEECH
                ),
            ),
        ]

        batch = aegis_collate_fn(
            samples
        )

        self.assertIsInstance(
            batch,
            TrainingBatch,
        )

        self.assertEqual(
            batch.batch_size,
            3,
        )

        self.assertEqual(
            batch.text_embeddings.shape,
            (3, 768),
        )

        self.assertEqual(
            batch.vision_embeddings.shape,
            (3, 512),
        )

        self.assertTrue(
            torch.equal(
                batch.integrity_targets,
                torch.tensor(
                    [
                        0,
                        1,
                        1,
                    ]
                ),
            )
        )

        self.assertTrue(
            torch.equal(
                batch.threat_targets,
                torch.tensor(
                    [
                        -1,
                        1,
                        3,
                    ]
                ),
            )
        )


class TestAEGISDataLoader(
    unittest.TestCase
):

    def test_dataloader_batch(self):

        dataset = (
            AEGISResearchDataset(
                build_dataset_samples(
                    count=10
                )
            )
        )

        loader = create_dataloader(
            dataset=dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batch = next(
            iter(loader)
        )

        self.assertIsInstance(
            batch,
            TrainingBatch,
        )

        self.assertEqual(
            batch.batch_size,
            4,
        )


class TestRandomSplit(
    unittest.TestCase
):

    def test_random_split_preserves_samples(self):

        samples = (
            build_dataset_samples(
                count=20
            )
        )

        split = (
            random_split_samples(
                samples,
                train_ratio=0.70,
                validation_ratio=0.15,
                test_ratio=0.15,
                seed=42,
            )
        )

        combined = (
            split["train"]
            + split["validation"]
            + split["test"]
        )

        self.assertEqual(
            len(combined),
            20,
        )

        self.assertEqual(
            len(
                {
                    sample.sample_id
                    for sample
                    in combined
                }
            ),
            20,
        )


class TestGroupAwareSplit(
    unittest.TestCase
):

    def test_groups_do_not_cross_splits(self):

        samples = (
            build_dataset_samples(
                count=20
            )
        )

        split = group_aware_split(
            samples,
            group_attribute="group_id",
            train_ratio=0.60,
            validation_ratio=0.20,
            test_ratio=0.20,
            seed=42,
        )

        train_groups = {
            sample.group_id
            for sample
            in split["train"]
        }

        validation_groups = {
            sample.group_id
            for sample
            in split[
                "validation"
            ]
        }

        test_groups = {
            sample.group_id
            for sample
            in split["test"]
        }

        self.assertTrue(
            train_groups.isdisjoint(
                validation_groups
            )
        )

        self.assertTrue(
            train_groups.isdisjoint(
                test_groups
            )
        )

        self.assertTrue(
            validation_groups.isdisjoint(
                test_groups
            )
        )

    def test_group_split_preserves_all_samples(self):

        samples = (
            build_dataset_samples(
                count=20
            )
        )

        split = group_aware_split(
            samples,
            group_attribute="group_id",
            train_ratio=0.60,
            validation_ratio=0.20,
            test_ratio=0.20,
            seed=42,
        )

        combined = (
            split["train"]
            + split["validation"]
            + split["test"]
        )

        original_ids = {
            sample.sample_id
            for sample
            in samples
        }

        split_ids = {
            sample.sample_id
            for sample
            in combined
        }

        self.assertEqual(
            original_ids,
            split_ids,
        )


class TestLoaderTrainerCompatibility(
    unittest.TestCase
):

    def test_dataloader_batch_validates_for_trainer(self):

        dataset = (
            AEGISResearchDataset(
                build_dataset_samples(
                    count=8
                )
            )
        )

        loader = create_dataloader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
        )

        batch = next(
            iter(loader)
        )

        batch.validate(
            text_dim=768,
            vision_dim=512,
        )

        self.assertEqual(
            batch.batch_size,
            8,
        )


class TestDatasetStatistics(
    unittest.TestCase
):

    def test_dataset_statistics_counts(self):

        samples = (
            build_dataset_samples(
                count=20
            )
        )

        statistics = (
            compute_dataset_statistics(
                samples
            )
        )

        self.assertEqual(
            statistics[
                "total_samples"
            ],
            20,
        )

        self.assertEqual(
            statistics[
                "labels"
            ][
                "true"
            ],
            4,
        )

        self.assertEqual(
            statistics[
                "labels"
            ][
                "misinformation"
            ],
            4,
        )

        self.assertEqual(
            statistics[
                "labels"
            ][
                "disinformation"
            ],
            4,
        )

        self.assertEqual(
            statistics[
                "labels"
            ][
                "malinformation"
            ],
            4,
        )

        self.assertEqual(
            statistics[
                "labels"
            ][
                "hate_speech"
            ],
            4,
        )

        self.assertEqual(
            statistics[
                "languages"
            ][
                "en"
            ],
            10,
        )

        self.assertEqual(
            statistics[
                "languages"
            ][
                "zh"
            ],
            10,
        )

        self.assertEqual(
            statistics[
                "unique_groups"
            ],
            10,
        )

        self.assertEqual(
            statistics[
                "unique_events"
            ],
            5,
        )

        self.assertEqual(
            statistics[
                "unique_sources"
            ],
            4,
        )


class TestSplitManifest(
    unittest.TestCase
):

    def setUp(self):

        self.samples = (
            build_dataset_samples(
                count=20
            )
        )

        self.split = (
            group_aware_split(
                self.samples,
                group_attribute="group_id",
                train_ratio=0.60,
                validation_ratio=0.20,
                test_ratio=0.20,
                seed=42,
            )
        )

        self.manifest = (
            build_split_manifest(
                self.split,
                strategy="group_aware",
                seed=42,
                group_attribute="group_id",
            )
        )

    def test_manifest_structure(self):

        self.assertEqual(
            self.manifest[
                "strategy"
            ],
            "group_aware",
        )

        self.assertEqual(
            self.manifest[
                "seed"
            ],
            42,
        )

        self.assertEqual(
            self.manifest[
                "group_attribute"
            ],
            "group_id",
        )

        self.assertEqual(
            set(
                self.manifest[
                    "splits"
                ].keys()
            ),
            {
                "train",
                "validation",
                "test",
            },
        )

    def test_manifest_sample_ids_match_split(self):

        for split_name in (
            "train",
            "validation",
            "test",
        ):

            expected_ids = {
                sample.sample_id
                for sample
                in self.split[
                    split_name
                ]
            }

            manifest_ids = set(
                self.manifest[
                    "splits"
                ][
                    split_name
                ][
                    "sample_ids"
                ]
            )

            self.assertEqual(
                expected_ids,
                manifest_ids,
            )

    def test_manifest_statistics_match_split(self):

        for split_name in (
            "train",
            "validation",
            "test",
        ):

            expected_count = len(
                self.split[
                    split_name
                ]
            )

            manifest_count = (
                self.manifest[
                    "splits"
                ][
                    split_name
                ][
                    "statistics"
                ][
                    "total_samples"
                ]
            )

            self.assertEqual(
                expected_count,
                manifest_count,
            )

    def test_manifest_save_load_round_trip(self):

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "split_manifest.json"
            )

            saved_path = (
                save_split_manifest(
                    self.manifest,
                    path,
                )
            )

            self.assertTrue(
                saved_path.is_file()
            )

            restored = (
                load_split_manifest(
                    saved_path
                )
            )

            self.assertEqual(
                restored,
                self.manifest,
            )


if __name__ == "__main__":
    unittest.main()