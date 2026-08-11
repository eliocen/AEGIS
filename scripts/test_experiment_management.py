"""
AEGIS v0.19.0
Experiment Management, Reproducibility & Resume
sanity experiment.
"""

import tempfile
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
    resume_experiment,
    save_experiment_checkpoint,
)

from aegis.training import (
    AEGISTrainer,
    EarlyStoppingState,
    TrainingBatch,
    set_global_seed,
)


set_global_seed(
    42
)


config = ExperimentConfig(
    experiment_name=(
        "aegis_resume_sanity"
    ),

    framework_version=(
        "0.19.0"
    ),

    seed=42,

    max_epochs=20,

    batch_size=4,

    device="cpu",
)


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


trainer = AEGISTrainer(
    alignment_model=(
        alignment_model
    ),

    classification_model=(
        classification_model
    ),

    device="cpu",
)


batch = TrainingBatch(
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
        "RESUME-0",
        "RESUME-1",
        "RESUME-2",
        "RESUME-3",
    ],
)


print(
    "AEGIS Experiment Management Sanity Test"
)

print(
    "=" * 60
)


with tempfile.TemporaryDirectory() as directory:

    manager = ExperimentManager(
        config=config,

        root_directory=(
            directory
        ),

        experiment_id=(
            "resume_sanity_run"
        ),
    )

    manager.initialize()


    print(
        "Experiment ID:",
        manager.experiment_id,
    )

    print(
        "Run directory exists:",
        manager.run_directory.is_dir(),
    )

    print(
        "Config snapshot exists:",
        manager.config_path.is_file(),
    )

    print(
        "Metadata snapshot exists:",
        manager.metadata_path.is_file(),
    )


    first_result = trainer.train_step(
        batch
    )


    trainer.state.epoch = (
        3
    )

    trainer.state.best_validation_loss = (
        1.2345
    )


    early_stopping = (
        EarlyStoppingState(
            patience=5,
            min_delta=0.0001,
        )
    )

    early_stopping.best_loss = (
        1.2345
    )

    early_stopping.bad_epochs = (
        2
    )


    checkpoint_path = (
        manager
        .checkpoint_directory
        / "resume.pt"
    )


    save_experiment_checkpoint(
        path=(
            checkpoint_path
        ),

        alignment_model=(
            trainer
            .alignment_model
        ),

        classification_model=(
            trainer
            .classification_model
        ),

        optimizer=(
            trainer
            .optimizer
        ),

        epoch=(
            trainer
            .state
            .epoch
        ),

        global_step=(
            trainer
            .state
            .global_step
        ),

        best_validation_loss=(
            trainer
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
            "loss": (
                first_result[
                    "loss"
                ]
            )
        },
    )


    print(
        "Checkpoint exists:",
        checkpoint_path.is_file(),
    )


    restored_alignment = (
        CrossModalAlignmentModel(
            text_dim=768,
            vision_dim=512,
            shared_dim=512,
            dropout=0.0,
            temperature=0.07,
        )
    )


    restored_classifier = (
        HierarchicalInformationIntegrityClassifier(
            input_dim=512,
            hidden_dim=256,
            dropout=0.0,
        )
    )


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


    (
        resume_state,
        restored_early_stopping,
    ) = resume_experiment(
        path=(
            checkpoint_path
        ),

        trainer=(
            restored_trainer
        ),

        map_location="cpu",
    )


    print(
        "\nRestored epoch:",
        resume_state.epoch,
    )

    print(
        "Restored global step:",
        resume_state.global_step,
    )

    print(
        "Restored best validation loss:",
        resume_state.best_validation_loss,
    )

    print(
        "Restored bad epochs:",
        restored_early_stopping.bad_epochs,
    )

    print(
        "Restored checkpoint type:",
        resume_state.checkpoint_type,
    )


    continued_result = (
        restored_trainer
        .train_step(
            batch
        )
    )


    print(
        "\nContinued training loss:",
        round(
            continued_result[
                "loss"
            ],
            4,
        ),
    )

    print(
        "Global step after resume:",
        restored_trainer
        .state
        .global_step,
    )

    print(
        "\nAEGIS experiment resume is operational."
    )