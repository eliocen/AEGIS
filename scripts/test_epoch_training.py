"""
AEGIS v0.18.0
Epoch Training, Validation & Early Stopping
sanity experiment.
"""

import tempfile

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
    set_global_seed,
)


set_global_seed(
    42
)


labels = [
    DatasetLabel.TRUE,
    DatasetLabel.MISINFORMATION,
    DatasetLabel.DISINFORMATION,
    DatasetLabel.MALINFORMATION,
    DatasetLabel.HATE_SPEECH,
]


samples = []


for index in range(
    40
):

    samples.append(
        ResearchSample(
            sample_id=(
                f"EPOCH-SANITY-{index:03d}"
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

            language=(
                "en"
                if index % 2 == 0
                else "zh"
            ),

            domain=(
                "international_security"
            ),

            source_dataset=(
                "epoch_sanity"
            ),

            group_id=(
                f"GROUP-{index // 2}"
            ),
        )
    )


train_dataset = (
    AEGISResearchDataset(
        samples[:32]
    )
)


validation_dataset = (
    AEGISResearchDataset(
        samples[32:]
    )
)


train_loader = (
    create_dataloader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
)


validation_loader = (
    create_dataloader(
        dataset=validation_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
)


alignment_model = (
    CrossModalAlignmentModel(
        text_dim=768,
        vision_dim=512,
        shared_dim=512,
        dropout=0.1,
        temperature=0.07,
    )
)


classification_model = (
    HierarchicalInformationIntegrityClassifier(
        input_dim=512,
        hidden_dim=256,
        dropout=0.2,
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

    alignment_loss_weight=1.0,

    classification_loss_weight=1.0,

    gradient_clip_norm=1.0,
)


with tempfile.TemporaryDirectory() as directory:

    runner = (
        AEGISExperimentRunner(
            trainer=trainer,

            experiment_name=(
                "epoch_sanity"
            ),

            max_epochs=5,

            patience=3,

            min_delta=0.0001,

            checkpoint_directory=(
                directory
            ),

            save_best=True,

            save_last=True,
        )
    )


    print(
        "AEGIS Epoch Training Experiment"
    )

    print(
        "=" * 60
    )


    history = runner.run(
        train_loader=(
            train_loader
        ),

        validation_loader=(
            validation_loader
        ),
    )


    for record in history.records:

        print(
            f"Epoch {record.epoch}"
        )

        print(
            "Train loss:",
            round(
                record
                .train_metrics[
                    "loss"
                ],
                4,
            ),
        )

        print(
            "Validation loss:",
            round(
                record
                .validation_metrics[
                    "loss"
                ],
                4,
            ),
        )

        print(
            "Train integrity accuracy:",
            round(
                record
                .train_metrics[
                    "integrity_accuracy"
                ],
                4,
            ),
        )

        print(
            "Validation integrity accuracy:",
            round(
                record
                .validation_metrics[
                    "integrity_accuracy"
                ],
                4,
            ),
        )

        print(
            "Train threat accuracy:",
            round(
                record
                .train_metrics[
                    "threat_accuracy"
                ],
                4,
            ),
        )

        print(
            "Validation threat accuracy:",
            round(
                record
                .validation_metrics[
                    "threat_accuracy"
                ],
                4,
            ),
        )

        print(
            "Improved:",
            record.improved,
        )

        print(
            "-" * 60
        )


    print(
        "Epochs completed:",
        len(
            history.records
        ),
    )

    print(
        "Best epoch:",
        history.best_epoch,
    )

    print(
        "Best validation loss:",
        round(
            history.best_validation_loss,
            4,
        ),
    )

    print(
        "Stopped early:",
        history.stopped_early,
    )

    print(
        "Global steps:",
        trainer.state.global_step,
    )

    print(
        "Best checkpoint exists:",
        runner
        .best_checkpoint_path
        .is_file(),
    )

    print(
        "Last checkpoint exists:",
        runner
        .last_checkpoint_path
        .is_file(),
    )

    print(
        "History exists:",
        runner
        .history_path
        .is_file(),
    )