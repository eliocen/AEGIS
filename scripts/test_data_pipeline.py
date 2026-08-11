"""
AEGIS v0.17.0
Research Dataset & DataLoader Infrastructure
sanity experiment.
"""

import torch

from aegis.data import (
    AEGISResearchDataset,
    DatasetLabel,
    ResearchSample,
    create_dataloader,
    group_aware_split,
)


torch.manual_seed(
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
    30
):

    samples.append(
        ResearchSample(
            sample_id=(
                f"RESEARCH-{index:03d}"
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
                "synthetic_sanity"
            ),

            group_id=(
                f"STORY-{index // 3}"
            ),

            source_id=(
                f"SOURCE-{index % 5}"
            ),

            event_id=(
                f"EVENT-{index % 4}"
            ),
        )
    )


print(
    "AEGIS Dataset Sanity Experiment"
)

print(
    "=" * 60
)

print(
    "Total samples:",
    len(samples),
)


split = group_aware_split(
    samples,
    group_attribute="group_id",
    train_ratio=0.60,
    validation_ratio=0.20,
    test_ratio=0.20,
    seed=42,
)


print(
    "Train samples:",
    len(
        split["train"]
    ),
)

print(
    "Validation samples:",
    len(
        split["validation"]
    ),
)

print(
    "Test samples:",
    len(
        split["test"]
    ),
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


print(
    "Group leakage train/validation:",
    bool(
        train_groups
        & validation_groups
    ),
)

print(
    "Group leakage train/test:",
    bool(
        train_groups
        & test_groups
    ),
)

print(
    "Group leakage validation/test:",
    bool(
        validation_groups
        & test_groups
    ),
)


train_dataset = (
    AEGISResearchDataset(
        split["train"]
    )
)


train_loader = create_dataloader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)


first_batch = next(
    iter(
        train_loader
    )
)


print(
    "\nFirst training batch:"
)

print(
    "Batch size:",
    first_batch.batch_size,
)

print(
    "Text shape:",
    first_batch
    .text_embeddings
    .shape,
)

print(
    "Vision shape:",
    first_batch
    .vision_embeddings
    .shape,
)

print(
    "Integrity targets:",
    first_batch
    .integrity_targets
    .tolist(),
)

print(
    "Threat targets:",
    first_batch
    .threat_targets
    .tolist(),
)

print(
    "Sample IDs:",
    first_batch.sample_ids,
)


print(
    "\nAEGIS dataset pipeline is operational."
)