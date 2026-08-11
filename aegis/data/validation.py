"""
AEGIS Dataset Validation.

Version: 0.17.0
"""

import torch

from .record import (
    ResearchSample,
)


def validate_research_sample(
    sample: ResearchSample,
    text_dim: int = 768,
    vision_dim: int = 512,
):
    """
    Validate one model-ready AEGIS research sample.
    """

    if not isinstance(
        sample,
        ResearchSample,
    ):
        raise TypeError(
            "Expected ResearchSample."
        )

    if not sample.sample_id:
        raise ValueError(
            "sample_id cannot be empty."
        )

    if not isinstance(
        sample.text_embedding,
        torch.Tensor,
    ):
        raise TypeError(
            "text_embedding must be a torch.Tensor."
        )

    if not isinstance(
        sample.vision_embedding,
        torch.Tensor,
    ):
        raise TypeError(
            "vision_embedding must be a torch.Tensor."
        )

    if sample.text_embedding.ndim != 1:
        raise ValueError(
            "text_embedding must have shape "
            "[text_dimension]."
        )

    if sample.vision_embedding.ndim != 1:
        raise ValueError(
            "vision_embedding must have shape "
            "[vision_dimension]."
        )

    if (
        sample.text_embedding.shape[0]
        != text_dim
    ):
        raise ValueError(
            f"Expected text dimension {text_dim}, "
            f"received "
            f"{sample.text_embedding.shape[0]}."
        )

    if (
        sample.vision_embedding.shape[0]
        != vision_dim
    ):
        raise ValueError(
            f"Expected vision dimension {vision_dim}, "
            f"received "
            f"{sample.vision_embedding.shape[0]}."
        )

    if not torch.isfinite(
        sample.text_embedding
    ).all():
        raise ValueError(
            "text_embedding contains "
            "non-finite values."
        )

    if not torch.isfinite(
        sample.vision_embedding
    ).all():
        raise ValueError(
            "vision_embedding contains "
            "non-finite values."
        )

    return True