"""
Contrastive alignment objectives for AEGIS.

Version: 0.9.0
"""

import torch
from torch import nn
from torch.nn import functional as F


class SymmetricContrastiveLoss(nn.Module):
    """
    Symmetric text-image contrastive objective.

    Positive pairs occupy matching batch indices.
    All other within-batch combinations act as negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        self.temperature = temperature

    def forward(
        self,
        text_embeddings,
        vision_embeddings,
    ):
        if text_embeddings.ndim != 2:
            raise ValueError(
                "text_embeddings must have shape "
                "[batch_size, dimension]."
            )

        if vision_embeddings.ndim != 2:
            raise ValueError(
                "vision_embeddings must have shape "
                "[batch_size, dimension]."
            )

        if text_embeddings.shape != vision_embeddings.shape:
            raise ValueError(
                "Text and vision embeddings must "
                "have identical aligned shapes."
            )

        text_embeddings = F.normalize(
            text_embeddings,
            p=2,
            dim=-1,
        )

        vision_embeddings = F.normalize(
            vision_embeddings,
            p=2,
            dim=-1,
        )

        logits = (
            text_embeddings
            @ vision_embeddings.T
        ) / self.temperature

        batch_size = logits.shape[0]

        targets = torch.arange(
            batch_size,
            device=logits.device,
        )

        text_to_image = F.cross_entropy(
            logits,
            targets,
        )

        image_to_text = F.cross_entropy(
            logits.T,
            targets,
        )

        return (
            text_to_image
            + image_to_text
        ) / 2