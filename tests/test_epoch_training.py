"""
Tests for AEGIS v0.18.0:
Epoch Training, Validation & Early Stopping Engine.
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

from aegis.data import (
    AEGISResearchDataset,
    DatasetLabel,
    ResearchSample,
    create_dataloader,
)

from aegis.training import (
    AEGISExperimentRunner,
    AEGISTrainer,
    EarlyStoppingState,
    ExperimentHistory,
    run_training_epoch,
    run_validation_epoch,
    set_global_seed,
)


def build_samples(
    count=20,
):

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
            ResearchSample(
                sample_id=(
                    f"EPOCH-{index:03d}"
                ),

                text_embedding=torch.randn(
                    768
                ),

                vision_embedding=torch.randn(
                    512
                ),

                label=(
                    labels[
                        index
                        % len(labels)
                    ]
                ),

                language="en",

                domain=(
                    "international_security"
                ),

                source_dataset=(
                    "epoch_unit_test"
                ),

                group_id=(
                    f"GROUP-{index // 2}"
                ),
            )
        )

    return samples


def build_loaders():

    samples = build_samples(
        20
    )

    train_dataset = (
        AEGISResearchDataset(
            samples[:16]
        )
    )

    validation_dataset = (
        AEGISResearchDataset(
            samples[16:]
        )
    )

    train_loader = (
        create_dataloader(
            dataset=train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
    )

    validation_loader = (
        create_dataloader(
            dataset=validation_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )
    )

    return (
        train_loader,
        validation_loader,
    )


def build_trainer():

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

    return AEGISTrainer(
        alignment_model=(
            alignment_model
        ),

        classification_model=(
            classification_model
        ),

        device="cpu",

        alignment_loss_weight=1.0,

        classification_loss_weight=1.0,

        gradient_clip_norm=1.0,
    )


class TestEarlyStopping(
    unittest.TestCase
):

    def test_initial_improvement(self):

        state = EarlyStoppingState(
            patience=3,
            min_delta=0.0,
        )

        improved = state.update(
            1.0
        )

        self.assertTrue(
            improved
        )

        self.assertEqual(
            state.best_loss,
            1.0,
        )

        self.assertEqual(
            state.bad_epochs,
            0,
        )

        self.assertFalse(
            state.stopped
        )

    def test_patience_triggers_stop(self):

        state = EarlyStoppingState(
            patience=2,
            min_delta=0.0,
        )

        state.update(
            1.0
        )

        state.update(
            1.1
        )

        self.assertFalse(
            state.stopped
        )

        state.update(
            1.2
        )

        self.assertTrue(
            state.stopped
        )

        self.assertEqual(
            state.bad_epochs,
            2,
        )

    def test_improvement_resets_patience(self):

        state = EarlyStoppingState(
            patience=3,
            min_delta=0.0,
        )

        state.update(
            1.0
        )

        state.update(
            1.1
        )

        self.assertEqual(
            state.bad_epochs,
            1,
        )

        state.update(
            0.9
        )

        self.assertEqual(
            state.bad_epochs,
            0,
        )

        self.assertEqual(
            state.best_loss,
            0.9,
        )


class TestEpochExecution(
    unittest.TestCase
):

    def setUp(self):

        set_global_seed(
            42
        )

        self.trainer = (
            build_trainer()
        )

        (
            self.train_loader,
            self.validation_loader,
        ) = build_loaders()

    def test_training_epoch(self):

        metrics = (
            run_training_epoch(
                trainer=(
                    self.trainer
                ),

                dataloader=(
                    self.train_loader
                ),
            )
        )

        self.assertIn(
            "loss",
            metrics,
        )

        self.assertIn(
            "integrity_accuracy",
            metrics,
        )

        self.assertIn(
            "threat_accuracy",
            metrics,
        )

        self.assertGreater(
            metrics["loss"],
            0.0,
        )

        self.assertEqual(
            self.trainer
            .state
            .global_step,
            4,
        )

    def test_validation_does_not_update_global_step(self):

        before = (
            self.trainer
            .state
            .global_step
        )

        metrics = (
            run_validation_epoch(
                trainer=(
                    self.trainer
                ),

                dataloader=(
                    self.validation_loader
                ),
            )
        )

        after = (
            self.trainer
            .state
            .global_step
        )

        self.assertEqual(
            before,
            after,
        )

        self.assertIn(
            "loss",
            metrics,
        )

        self.assertGreater(
            metrics["loss"],
            0.0,
        )

    def test_validation_does_not_update_parameters(self):

        before = (
            self.trainer
            .classification_model
            .integrity_head
            .weight
            .detach()
            .clone()
        )

        run_validation_epoch(
            trainer=self.trainer,
            dataloader=(
                self.validation_loader
            ),
        )

        after = (
            self.trainer
            .classification_model
            .integrity_head
            .weight
            .detach()
            .clone()
        )

        self.assertTrue(
            torch.equal(
                before,
                after,
            )
        )


class TestExperimentHistory(
    unittest.TestCase
):

    def test_history_initialization(self):

        history = ExperimentHistory(
            experiment_name=(
                "unit_test"
            )
        )

        self.assertEqual(
            history.experiment_name,
            "unit_test",
        )

        self.assertEqual(
            len(
                history.records
            ),
            0,
        )

        self.assertEqual(
            history.best_epoch,
            0,
        )


class TestAEGISExperimentRunner(
    unittest.TestCase
):

    def setUp(self):

        set_global_seed(
            42
        )

        self.trainer = (
            build_trainer()
        )

        (
            self.train_loader,
            self.validation_loader,
        ) = build_loaders()

    def test_experiment_runs_multiple_epochs(self):

        with tempfile.TemporaryDirectory() as directory:

            runner = (
                AEGISExperimentRunner(
                    trainer=(
                        self.trainer
                    ),

                    experiment_name=(
                        "multi_epoch_test"
                    ),

                    max_epochs=2,

                    patience=5,

                    checkpoint_directory=(
                        directory
                    ),

                    save_best=True,

                    save_last=True,
                )
            )

            history = runner.run(
                train_loader=(
                    self.train_loader
                ),

                validation_loader=(
                    self.validation_loader
                ),
            )

            self.assertEqual(
                len(
                    history.records
                ),
                2,
            )

            self.assertEqual(
                self.trainer
                .state
                .epoch,
                2,
            )

            self.assertTrue(
                runner
                .best_checkpoint_path
                .is_file()
            )

            self.assertTrue(
                runner
                .last_checkpoint_path
                .is_file()
            )

            self.assertTrue(
                runner
                .history_path
                .is_file()
            )

    def test_best_epoch_recorded(self):

        with tempfile.TemporaryDirectory() as directory:

            runner = (
                AEGISExperimentRunner(
                    trainer=(
                        self.trainer
                    ),

                    experiment_name=(
                        "best_epoch_test"
                    ),

                    max_epochs=2,

                    patience=5,

                    checkpoint_directory=(
                        directory
                    ),
                )
            )

            history = runner.run(
                train_loader=(
                    self.train_loader
                ),

                validation_loader=(
                    self.validation_loader
                ),
            )

            self.assertGreaterEqual(
                history.best_epoch,
                1,
            )

            self.assertLessEqual(
                history.best_epoch,
                2,
            )

            self.assertTrue(
                torch.isfinite(
                    torch.tensor(
                        history
                        .best_validation_loss
                    )
                )
            )

    def test_history_contains_train_and_validation_metrics(self):

        with tempfile.TemporaryDirectory() as directory:

            runner = (
                AEGISExperimentRunner(
                    trainer=(
                        self.trainer
                    ),

                    experiment_name=(
                        "history_metrics_test"
                    ),

                    max_epochs=1,

                    patience=5,

                    checkpoint_directory=(
                        directory
                    ),
                )
            )

            history = runner.run(
                train_loader=(
                    self.train_loader
                ),

                validation_loader=(
                    self.validation_loader
                ),
            )

            record = (
                history.records[
                    0
                ]
            )

            self.assertIn(
                "loss",
                record.train_metrics,
            )

            self.assertIn(
                "loss",
                record.validation_metrics,
            )

            self.assertIn(
                "integrity_accuracy",
                record.train_metrics,
            )

            self.assertIn(
                "threat_accuracy",
                record.validation_metrics,
            )


if __name__ == "__main__":
    unittest.main()