"""
AEGIS Ablation Batch Transformation.

Version: 0.21.0
"""

import torch

from aegis.training import (
    TrainingBatch,
)

from .spec import (
    AblationSpec,
)


def apply_ablation_to_batch(
    batch: TrainingBatch,
    spec: AblationSpec,
):
    """
    Apply a controlled modality ablation to an
    existing TrainingBatch.

    Disabled modalities are replaced by zero tensors
    with identical dimensions.

    This preserves:
        sample order
        batch size
        tensor shapes
        labels
        experiment comparability
    """

    if not isinstance(
        batch,
        TrainingBatch,
    ):
        raise TypeError(
            "batch must be TrainingBatch."
        )

    if not isinstance(
        spec,
        AblationSpec,
    ):
        raise TypeError(
            "spec must be AblationSpec."
        )

    text_embeddings = (
        batch.text_embeddings
    )

    vision_embeddings = (
        batch.vision_embeddings
    )

    if not spec.use_text:

        text_embeddings = (
            torch.zeros_like(
                text_embeddings
            )
        )

    if not spec.use_vision:

        vision_embeddings = (
            torch.zeros_like(
                vision_embeddings
            )
        )

    transformed = TrainingBatch(
        text_embeddings=(
            text_embeddings
        ),

        vision_embeddings=(
            vision_embeddings
        ),

        integrity_targets=(
            batch.integrity_targets
        ),

        threat_targets=(
            batch.threat_targets
        ),

        sample_ids=(
            list(
                batch.sample_ids
            )
            if batch.sample_ids
            is not None
            else None
        ),
    )

    transformed.validate(
        text_dim=(
            batch.text_embeddings.shape[-1]
        ),

        vision_dim=(
            batch.vision_embeddings.shape[-1]
        ),
    )

    return transformed