"""
AEGIS v0.16.0
Research Prototype Training Infrastructure
sanity experiment.
"""

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
    set_global_seed,
)


set_global_seed(
    42
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
    alignment_model=alignment_model,

    classification_model=(
        classification_model
    ),

    device="cpu",

    alignment_loss_weight=1.0,

    classification_loss_weight=1.0,

    gradient_clip_norm=1.0,
)


batch = TrainingBatch(
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
            1,
            1,
            0,
            1,
            1,
            0,
        ]
    ),

    threat_targets=torch.tensor(
        [
            -1,
            0,
            1,
            2,
            -1,
            3,
            1,
            -1,
        ]
    ),

    sample_ids=[
        f"TRAIN-{index}"
        for index in range(8)
    ],
)


print(
    "AEGIS Training Sanity Experiment"
)

print(
    "=" * 50
)


for step in range(
    1,
    6,
):

    metrics = trainer.train_step(
        batch
    )

    print(
        f"Step {step}"
    )

    print(
        "Total loss:",
        round(
            metrics["loss"],
            4,
        ),
    )

    print(
        "Alignment loss:",
        round(
            metrics[
                "alignment_loss"
            ],
            4,
        ),
    )

    print(
        "Classification loss:",
        round(
            metrics[
                "classification_loss"
            ],
            4,
        ),
    )

    print(
        "Integrity accuracy:",
        round(
            metrics[
                "integrity_accuracy"
            ],
            4,
        ),
    )

    print(
        "Threat accuracy:",
        round(
            metrics[
                "threat_accuracy"
            ],
            4,
        ),
    )

    print(
        "-" * 50
    )


print(
    "Global steps:",
    trainer.state.global_step,
)