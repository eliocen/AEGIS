"""
AEGIS DataLoader Collation.

Version: 0.17.0
"""

import torch

from aegis.training import (
    TrainingBatch,
)

from .record import (
    ResearchSample,
)


def aegis_collate_fn(
    samples,
):
    """
    Collate ResearchSample objects into
    an AEGIS TrainingBatch.
    """

    if not samples:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    for sample in samples:

        if not isinstance(
            sample,
            ResearchSample,
        ):
            raise TypeError(
                "AEGIS collator expects "
                "ResearchSample objects."
            )

    text_embeddings = torch.stack(
        [
            sample.text_embedding
            for sample
            in samples
        ]
    ).float()

    vision_embeddings = torch.stack(
        [
            sample.vision_embedding
            for sample
            in samples
        ]
    ).float()

    integrity_targets = torch.tensor(
        [
            sample.integrity_target
            for sample
            in samples
        ],
        dtype=torch.long,
    )

    threat_targets = torch.tensor(
        [
            sample.threat_target
            for sample
            in samples
        ],
        dtype=torch.long,
    )

    batch = TrainingBatch(
        text_embeddings=(
            text_embeddings
        ),

        vision_embeddings=(
            vision_embeddings
        ),

        integrity_targets=(
            integrity_targets
        ),

        threat_targets=(
            threat_targets
        ),

        sample_ids=[
            sample.sample_id
            for sample
            in samples
        ],
    )

    batch.validate()

    return batch