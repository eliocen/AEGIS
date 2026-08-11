"""
Tests for AEGIS v0.16.0:
Research Prototype Training Infrastructure.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from aegis.alignment import (
    CrossModalAlignmentModel,
)

from aegis.classification import (
    HierarchicalInformationIntegrityClassifier,
)

from aegis.training import (
    AEGISTrainer,
    TrainingBatch,
    compute_hierarchical_metrics,
    load_checkpoint,
    save_checkpoint,
    set_global_seed,
)


def build_batch(
    batch_size=4,
):
    """
    Construct a valid synthetic AEGIS training batch.
    """

    return TrainingBatch(
        text_embeddings=torch.randn(
            batch_size,
            768,
        ),

        vision_embeddings=torch.randn(
            batch_size,
            512,
        ),

        integrity_targets=torch.tensor(
            [
                0,
                1,
                1,
                1,
            ][:batch_size]
        ),

        threat_targets=torch.tensor(
            [
                -1,
                0,
                1,
                3,
            ][:batch_size]
        ),

        sample_ids=[
            f"SAMPLE-{index}"
            for index in range(
                batch_size
            )
        ],
    )


def build_models():
    """
    Construct lightweight AEGIS trainable models.
    """

    alignment_model = (
        CrossModalAlignmentModel(
            text_dim=768,
            vision_dim=512,
            shared_dim=512,
            dropout=0.0,
            temperature=0.07,
        )
    )

    classification_model = (
        HierarchicalInformationIntegrityClassifier(
            input_dim=512,
            hidden_dim=256,
            dropout=0.0,
        )
    )

    return (
        alignment_model,
        classification_model,
    )


class TestTrainingBatch(
    unittest.TestCase
):

    def test_valid_batch(self):

        batch = build_batch()

        batch.validate()

        self.assertEqual(
            batch.batch_size,
            4,
        )

    def test_invalid_text_dimension(self):

        batch = TrainingBatch(
            text_embeddings=torch.randn(
                4,
                100,
            ),

            vision_embeddings=torch.randn(
                4,
                512,
            ),

            integrity_targets=torch.tensor(
                [0, 1, 1, 1]
            ),

            threat_targets=torch.tensor(
                [-1, 0, 1, 2]
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            batch.validate()

    def test_true_sample_requires_minus_one(self):

        batch = TrainingBatch(
            text_embeddings=torch.randn(
                2,
                768,
            ),

            vision_embeddings=torch.randn(
                2,
                512,
            ),

            integrity_targets=torch.tensor(
                [0, 1]
            ),

            threat_targets=torch.tensor(
                [2, 1]
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            batch.validate()


class TestTrainingMetrics(
    unittest.TestCase
):

    def test_perfect_metrics(self):

        integrity_logits = torch.tensor(
            [
                [10.0, -10.0],
                [-10.0, 10.0],
                [-10.0, 10.0],
            ]
        )

        threat_logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
            ]
        )

        integrity_targets = torch.tensor(
            [0, 1, 1]
        )

        threat_targets = torch.tensor(
            [-1, 0, 1]
        )

        metrics = (
            compute_hierarchical_metrics(
                integrity_logits,
                threat_logits,
                integrity_targets,
                threat_targets,
            )
        )

        self.assertEqual(
            metrics[
                "integrity_accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "threat_accuracy"
            ],
            1.0,
        )

        self.assertEqual(
            metrics[
                "harmful_count"
            ],
            2.0,
        )


class TestAEGISTrainer(
    unittest.TestCase
):

    def setUp(self):

        set_global_seed(42)

        (
            self.alignment_model,
            self.classification_model,
        ) = build_models()

        self.trainer = AEGISTrainer(
            alignment_model=(
                self.alignment_model
            ),

            classification_model=(
                self.classification_model
            ),

            device="cpu",

            alignment_loss_weight=1.0,

            classification_loss_weight=1.0,

            gradient_clip_norm=1.0,
        )

    def test_forward_batch(self):

        batch = build_batch()

        outputs = (
            self.trainer.forward_batch(
                batch
            )
        )

        self.assertTrue(
            torch.isfinite(
                outputs[
                    "loss"
                ]
            )
        )

        self.assertEqual(
            outputs[
                "alignment_outputs"
            ][
                "fused_embedding"
            ].shape,
            (4, 512),
        )

        self.assertEqual(
            outputs[
                "classification_outputs"
            ][
                "integrity_logits"
            ].shape,
            (4, 2),
        )

        self.assertEqual(
            outputs[
                "classification_outputs"
            ][
                "threat_logits"
            ].shape,
            (4, 4),
        )

    def test_train_step_updates_global_step(self):

        batch = build_batch()

        self.assertEqual(
            self.trainer.state.global_step,
            0,
        )

        result = (
            self.trainer.train_step(
                batch
            )
        )

        self.assertEqual(
            self.trainer.state.global_step,
            1,
        )

        self.assertGreater(
            result["loss"],
            0.0,
        )

        self.assertIn(
            "integrity_accuracy",
            result,
        )

        self.assertIn(
            "threat_accuracy",
            result,
        )

    def test_train_step_updates_parameters(self):

        batch = build_batch()

        before = (
            self.classification_model
            .integrity_head
            .weight
            .detach()
            .clone()
        )

        self.trainer.train_step(
            batch
        )

        after = (
            self.classification_model
            .integrity_head
            .weight
            .detach()
            .clone()
        )

        self.assertFalse(
            torch.equal(
                before,
                after,
            )
        )


class TestCheckpointing(
    unittest.TestCase
):

    def test_checkpoint_round_trip(self):

        set_global_seed(42)

        (
            alignment_model,
            classification_model,
        ) = build_models()

        trainer = AEGISTrainer(
            alignment_model=(
                alignment_model
            ),

            classification_model=(
                classification_model
            ),

            device="cpu",
        )

        batch = build_batch()

        trainer.train_step(
            batch
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "test_checkpoint.pt"
            )

            save_checkpoint(
                path=path,

                alignment_model=(
                    alignment_model
                ),

                classification_model=(
                    classification_model
                ),

                optimizer=(
                    trainer.optimizer
                ),

                epoch=2,

                global_step=(
                    trainer.state.global_step
                ),

                metrics={
                    "loss": 1.23
                },

                extra={
                    "experiment": "unit_test"
                },
            )

            self.assertTrue(
                path.is_file()
            )

            (
                restored_alignment,
                restored_classifier,
            ) = build_models()

            restored_trainer = (
                AEGISTrainer(
                    alignment_model=(
                        restored_alignment
                    ),

                    classification_model=(
                        restored_classifier
                    ),

                    device="cpu",
                )
            )

            checkpoint = (
                load_checkpoint(
                    path=path,

                    alignment_model=(
                        restored_alignment
                    ),

                    classification_model=(
                        restored_classifier
                    ),

                    optimizer=(
                        restored_trainer
                        .optimizer
                    ),

                    map_location="cpu",
                )
            )

            self.assertEqual(
                checkpoint["epoch"],
                2,
            )

            self.assertEqual(
                checkpoint[
                    "global_step"
                ],
                1,
            )

            self.assertEqual(
                checkpoint[
                    "extra"
                ][
                    "experiment"
                ],
                "unit_test",
            )

            original_weight = (
                alignment_model
                .text_projection
                .projection
                .weight
                .detach()
            )

            restored_weight = (
                restored_alignment
                .text_projection
                .projection
                .weight
                .detach()
            )

            self.assertTrue(
                torch.allclose(
                    original_weight,
                    restored_weight,
                )
            )


class TestReproducibility(
    unittest.TestCase
):

    def test_seed_reproducibility(self):

        set_global_seed(123)

        first = torch.randn(
            5
        )

        set_global_seed(123)

        second = torch.randn(
            5
        )

        self.assertTrue(
            torch.equal(
                first,
                second,
            )
        )


if __name__ == "__main__":
    unittest.main()