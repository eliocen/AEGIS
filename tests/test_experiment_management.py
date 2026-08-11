"""
Tests for AEGIS v0.19.0:
Experiment Management, Reproducibility & Resume.
"""

import json
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

from aegis.experiment import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentMetadata,
    load_experiment_checkpoint,
    resume_experiment,
    save_experiment_checkpoint,
)

from aegis.training import (
    AEGISTrainer,
    EarlyStoppingState,
    TrainingBatch,
    set_global_seed,
)


def build_models():

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


def build_trainer():

    (
        alignment_model,
        classification_model,
    ) = build_models()

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
            "EXP-0",
            "EXP-1",
            "EXP-2",
            "EXP-3",
        ],
    )


class TestExperimentConfig(
    unittest.TestCase
):

    def test_config_creation(self):

        config = ExperimentConfig(
            experiment_name=(
                "research_test"
            ),

            seed=42,

            max_epochs=10,
        )

        self.assertEqual(
            config.experiment_name,
            "research_test",
        )

        self.assertEqual(
            config.framework_version,
            "0.19.0",
        )

        self.assertEqual(
            config.seed,
            42,
        )

        self.assertEqual(
            config.max_epochs,
            10,
        )

    def test_invalid_config_fails(self):

        with self.assertRaises(
            ValueError
        ):
            ExperimentConfig(
                experiment_name="",
            )


class TestExperimentMetadata(
    unittest.TestCase
):

    def test_metadata_capture(self):

        metadata = (
            ExperimentMetadata.capture(
                "metadata-test"
            )
        )

        self.assertEqual(
            metadata.experiment_id,
            "metadata-test",
        )

        self.assertTrue(
            metadata.python_version
        )

        self.assertTrue(
            metadata.platform
        )

        self.assertTrue(
            metadata.pytorch_version
        )


class TestExperimentManager(
    unittest.TestCase
):

    def test_initialize_experiment(self):

        with tempfile.TemporaryDirectory() as directory:

            config = ExperimentConfig(
                experiment_name=(
                    "manager_test"
                )
            )

            manager = ExperimentManager(
                config=config,

                root_directory=(
                    directory
                ),

                experiment_id=(
                    "manager_test_run"
                ),
            )

            manager.initialize()

            self.assertTrue(
                manager
                .run_directory
                .is_dir()
            )

            self.assertTrue(
                manager
                .checkpoint_directory
                .is_dir()
            )

            self.assertTrue(
                manager
                .artifact_directory
                .is_dir()
            )

            self.assertTrue(
                manager
                .config_path
                .is_file()
            )

            self.assertTrue(
                manager
                .metadata_path
                .is_file()
            )

    def test_config_snapshot_round_trip(self):

        with tempfile.TemporaryDirectory() as directory:

            config = ExperimentConfig(
                experiment_name=(
                    "snapshot_test"
                ),

                seed=123,
            )

            manager = ExperimentManager(
                config=config,

                root_directory=(
                    directory
                ),

                experiment_id=(
                    "snapshot_run"
                ),
            )

            manager.initialize()

            snapshot = (
                manager
                .load_config_snapshot()
            )

            self.assertEqual(
                snapshot[
                    "experiment_name"
                ],
                "snapshot_test",
            )

            self.assertEqual(
                snapshot[
                    "seed"
                ],
                123,
            )

            self.assertEqual(
                snapshot[
                    "framework_version"
                ],
                "0.19.0",
            )

    def test_metadata_snapshot_valid_json(self):

        with tempfile.TemporaryDirectory() as directory:

            config = ExperimentConfig(
                experiment_name=(
                    "metadata_snapshot"
                )
            )

            manager = ExperimentManager(
                config=config,

                root_directory=(
                    directory
                ),

                experiment_id=(
                    "metadata_snapshot_run"
                ),
            )

            manager.initialize()

            with manager.metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            self.assertEqual(
                metadata[
                    "experiment_id"
                ],
                "metadata_snapshot_run",
            )

            self.assertIn(
                "python_version",
                metadata,
            )

            self.assertIn(
                "pytorch_version",
                metadata,
            )


class TestExperimentCheckpoint(
    unittest.TestCase
):

    def setUp(self):

        set_global_seed(
            42
        )

        self.trainer = (
            build_trainer()
        )

        self.batch = (
            build_batch()
        )

        self.config = (
            ExperimentConfig(
                experiment_name=(
                    "checkpoint_test"
                )
            )
        )

    def test_experiment_checkpoint_round_trip(self):

        self.trainer.train_step(
            self.batch
        )

        early_stopping = (
            EarlyStoppingState(
                patience=3,
                min_delta=0.001,
            )
        )

        early_stopping.update(
            1.25
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "resume.pt"
            )

            save_experiment_checkpoint(
                path=path,

                alignment_model=(
                    self.trainer
                    .alignment_model
                ),

                classification_model=(
                    self.trainer
                    .classification_model
                ),

                optimizer=(
                    self.trainer
                    .optimizer
                ),

                epoch=3,

                global_step=(
                    self.trainer
                    .state
                    .global_step
                ),

                best_validation_loss=(
                    1.25
                ),

                early_stopping_state=(
                    early_stopping
                ),

                experiment_config=(
                    self.config
                ),

                checkpoint_type=(
                    "resume"
                ),

                metrics={
                    "loss": 1.25
                },
            )

            checkpoint = (
                load_experiment_checkpoint(
                    path
                )
            )

            self.assertEqual(
                checkpoint[
                    "checkpoint_format"
                ],
                "aegis_experiment_v1",
            )

            self.assertEqual(
                checkpoint[
                    "checkpoint_type"
                ],
                "resume",
            )

            self.assertEqual(
                checkpoint[
                    "epoch"
                ],
                3,
            )

            self.assertEqual(
                checkpoint[
                    "global_step"
                ],
                1,
            )

            self.assertEqual(
                checkpoint[
                    "experiment_config"
                ][
                    "framework_version"
                ],
                "0.19.0",
            )


class TestExperimentResume(
    unittest.TestCase
):

    def test_resume_restores_training_state(self):

        set_global_seed(
            42
        )

        original_trainer = (
            build_trainer()
        )

        batch = build_batch()

        original_trainer.train_step(
            batch
        )

        original_trainer.state.epoch = (
            4
        )

        original_trainer.state.best_validation_loss = (
            0.875
        )

        early_stopping = (
            EarlyStoppingState(
                patience=5,
                min_delta=0.001,
            )
        )

        early_stopping.best_loss = (
            0.875
        )

        early_stopping.bad_epochs = (
            2
        )

        config = ExperimentConfig(
            experiment_name=(
                "resume_test"
            )
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "resume.pt"
            )

            save_experiment_checkpoint(
                path=path,

                alignment_model=(
                    original_trainer
                    .alignment_model
                ),

                classification_model=(
                    original_trainer
                    .classification_model
                ),

                optimizer=(
                    original_trainer
                    .optimizer
                ),

                epoch=(
                    original_trainer
                    .state
                    .epoch
                ),

                global_step=(
                    original_trainer
                    .state
                    .global_step
                ),

                best_validation_loss=(
                    original_trainer
                    .state
                    .best_validation_loss
                ),

                early_stopping_state=(
                    early_stopping
                ),

                experiment_config=(
                    config
                ),

                checkpoint_type=(
                    "resume"
                ),

                metrics={
                    "validation_loss": (
                        0.875
                    )
                },
            )

            restored_trainer = (
                build_trainer()
            )

            (
                resume_state,
                restored_early_stopping,
            ) = resume_experiment(
                path=path,

                trainer=(
                    restored_trainer
                ),

                map_location="cpu",
            )

            self.assertEqual(
                resume_state.epoch,
                4,
            )

            self.assertEqual(
                resume_state.global_step,
                1,
            )

            self.assertAlmostEqual(
                resume_state
                .best_validation_loss,
                0.875,
                places=6,
            )

            self.assertEqual(
                restored_trainer
                .state
                .epoch,
                4,
            )

            self.assertEqual(
                restored_trainer
                .state
                .global_step,
                1,
            )

            self.assertEqual(
                restored_early_stopping
                .bad_epochs,
                2,
            )

            self.assertAlmostEqual(
                restored_early_stopping
                .best_loss,
                0.875,
                places=6,
            )

            original_weight = (
                original_trainer
                .classification_model
                .integrity_head
                .weight
                .detach()
            )

            restored_weight = (
                restored_trainer
                .classification_model
                .integrity_head
                .weight
                .detach()
            )

            self.assertTrue(
                torch.allclose(
                    original_weight,
                    restored_weight,
                )
            )

    def test_resume_can_continue_training(self):

        set_global_seed(
            42
        )

        original_trainer = (
            build_trainer()
        )

        batch = (
            build_batch()
        )

        original_trainer.train_step(
            batch
        )

        early_stopping = (
            EarlyStoppingState(
                patience=3
            )
        )

        config = ExperimentConfig(
            experiment_name=(
                "continue_test"
            )
        )

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "resume.pt"
            )

            save_experiment_checkpoint(
                path=path,

                alignment_model=(
                    original_trainer
                    .alignment_model
                ),

                classification_model=(
                    original_trainer
                    .classification_model
                ),

                optimizer=(
                    original_trainer
                    .optimizer
                ),

                epoch=2,

                global_step=(
                    original_trainer
                    .state
                    .global_step
                ),

                best_validation_loss=(
                    1.0
                ),

                early_stopping_state=(
                    early_stopping
                ),

                experiment_config=(
                    config
                ),

                checkpoint_type=(
                    "resume"
                ),
            )

            restored_trainer = (
                build_trainer()
            )

            resume_experiment(
                path=path,

                trainer=(
                    restored_trainer
                ),

                map_location="cpu",
            )

            before_step = (
                restored_trainer
                .state
                .global_step
            )

            result = (
                restored_trainer
                .train_step(
                    batch
                )
            )

            self.assertEqual(
                restored_trainer
                .state
                .global_step,
                before_step + 1,
            )

            self.assertTrue(
                torch.isfinite(
                    torch.tensor(
                        result["loss"]
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()