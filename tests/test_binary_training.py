"""
Tests for AEGIS binary empirical training.

Version: 0.24.0
"""

import unittest

import torch

from aegis.alignment import (
    CrossModalAlignmentModel,
)

from aegis.classification import (
    BinaryIntegrityLoss,
    HierarchicalInformationIntegrityClassifier,
)

from aegis.training import (
    BinaryIntegrityBatch,
    BinaryIntegrityTrainer,
    TrainingTask,
    compute_binary_integrity_metrics,
)


class TestTrainingTask(
    unittest.TestCase
):

    def test_binary_task(self):
        self.assertEqual(
            TrainingTask.from_value(
                "binary_integrity"
            ),
            TrainingTask.BINARY_INTEGRITY,
        )

    def test_hierarchical_task(self):
        self.assertEqual(
            TrainingTask.from_value(
                "hierarchical"
            ),
            TrainingTask.HIERARCHICAL,
        )

    def test_invalid_task_fails(self):
        with self.assertRaises(
            ValueError
        ):
            TrainingTask.from_value(
                "unsupported"
            )


class TestBinaryIntegrityBatch(
    unittest.TestCase
):

    def make_batch(self):
        return BinaryIntegrityBatch(
            text_embeddings=torch.randn(
                4,
                768,
            ),
            vision_embeddings=torch.randn(
                4,
                512,
            ),
            integrity_targets=torch.tensor(
                [0, 1, 1, 0],
                dtype=torch.long,
            ),
            sample_ids=[
                "a",
                "b",
                "c",
                "d",
            ],
        )

    def test_valid_batch(self):
        batch = self.make_batch()

        batch.validate()

        self.assertEqual(
            batch.batch_size,
            4,
        )

    def test_no_threat_target(self):
        batch = self.make_batch()

        self.assertFalse(
            hasattr(
                batch,
                "threat_targets",
            )
        )

    def test_invalid_integrity_target(self):
        batch = self.make_batch()

        batch.integrity_targets = (
            torch.tensor(
                [0, 1, 2, 0],
                dtype=torch.long,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            batch.validate()

    def test_invalid_text_dimension(self):
        batch = self.make_batch()

        batch.text_embeddings = (
            torch.randn(
                4,
                10,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            batch.validate()

    def test_device_transfer(self):
        batch = self.make_batch()

        moved = batch.to("cpu")

        self.assertEqual(
            moved.text_embeddings.device.type,
            "cpu",
        )

        self.assertEqual(
            moved.integrity_targets.device.type,
            "cpu",
        )


class TestBinaryIntegrityLoss(
    unittest.TestCase
):

    def test_loss_is_scalar(self):
        loss_fn = (
            BinaryIntegrityLoss()
        )

        outputs = loss_fn(
            torch.randn(
                4,
                2,
            ),
            torch.tensor(
                [0, 1, 1, 0],
                dtype=torch.long,
            ),
        )

        self.assertEqual(
            outputs["loss"].ndim,
            0,
        )

    def test_invalid_logit_shape(self):
        loss_fn = (
            BinaryIntegrityLoss()
        )

        with self.assertRaises(
            ValueError
        ):
            loss_fn(
                torch.randn(
                    4,
                    5,
                ),
                torch.tensor(
                    [0, 1, 1, 0],
                    dtype=torch.long,
                ),
            )


class TestBinaryMetrics(
    unittest.TestCase
):

    def test_perfect_accuracy(self):
        logits = torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 10.0],
                [0.0, 10.0],
                [10.0, 0.0],
            ]
        )

        targets = torch.tensor(
            [0, 1, 1, 0],
            dtype=torch.long,
        )

        metrics = (
            compute_binary_integrity_metrics(
                logits,
                targets,
            )
        )

        self.assertEqual(
            metrics[
                "integrity_accuracy"
            ],
            1.0,
        )


class TestBinaryIntegrityTrainer(
    unittest.TestCase
):

    def make_trainer(self):
        alignment_model = (
            CrossModalAlignmentModel(
                text_dim=768,
                vision_dim=512,
                shared_dim=64,
                temperature=0.07,
            )
        )

        classifier = (
            HierarchicalInformationIntegrityClassifier(
                input_dim=64,
                hidden_dim=32,
                dropout=0.0,
            )
        )

        return BinaryIntegrityTrainer(
            alignment_model=(
                alignment_model
            ),
            classification_model=(
                classifier
            ),
            device="cpu",
        )

    def make_batch(self):
        return BinaryIntegrityBatch(
            text_embeddings=torch.randn(
                8,
                768,
            ),
            vision_embeddings=torch.randn(
                8,
                512,
            ),
            integrity_targets=torch.tensor(
                [
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                ],
                dtype=torch.long,
            ),
            sample_ids=[
                f"sample-{i}"
                for i in range(8)
            ],
        )

    def test_forward_batch(self):
        trainer = (
            self.make_trainer()
        )

        outputs = trainer.forward_batch(
            self.make_batch()
        )

        self.assertIn(
            "loss",
            outputs,
        )

        self.assertIn(
            "integrity_loss",
            outputs,
        )

        self.assertIsNone(
            outputs[
                "threat_loss"
            ]
        )

    def test_train_step(self):
        trainer = (
            self.make_trainer()
        )

        result = trainer.train_step(
            self.make_batch()
        )

        self.assertEqual(
            trainer.state.global_step,
            1,
        )

        self.assertIn(
            "loss",
            result,
        )

        self.assertIn(
            "integrity_accuracy",
            result,
        )

    def test_parameters_update(self):
        trainer = (
            self.make_trainer()
        )

        before = [
            parameter.detach()
            .clone()
            for parameter
            in trainer.classification_model.parameters()
        ]

        trainer.train_step(
            self.make_batch()
        )

        after = list(
            trainer.classification_model.parameters()
        )

        changed = any(
            not torch.equal(
                old,
                new.detach(),
            )
            for old, new
            in zip(
                before,
                after,
            )
        )

        self.assertTrue(
            changed
        )


if __name__ == "__main__":
    unittest.main()