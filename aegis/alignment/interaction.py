"""
AEGIS cross-modal evidence interaction.

Version: 0.25.0

This module explicitly models relationships between aligned modality
representations. It preserves agreement and contradiction signals that
may otherwise be obscured by direct multimodal fusion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CrossModalEvidenceInteraction(nn.Module):
    """
    Learn explicit interaction evidence between two aligned modalities.

    For aligned representations h_a and h_b, the interaction features are:

        h_a
        h_b
        |h_a - h_b|
        h_a * h_b
        cosine_similarity(h_a, h_b)

    The combined interaction vector is projected back into the shared
    AEGIS evidence space.
    """

    def __init__(
        self,
        dimension: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        self.dimension = dimension

        interaction_dim = (
            dimension * 4
            + 1
        )

        self.interaction_projection = nn.Sequential(
            nn.Linear(
                interaction_dim,
                dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dimension),
        )

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute explicit cross-modal interaction evidence.

        Both inputs must have shape:

            [batch_size, dimension]
        """

        if first_embedding.ndim != 2:
            raise ValueError(
                "first_embedding must have shape "
                "[batch_size, dimension]."
            )

        if second_embedding.ndim != 2:
            raise ValueError(
                "second_embedding must have shape "
                "[batch_size, dimension]."
            )

        if first_embedding.shape != second_embedding.shape:
            raise ValueError(
                "Cross-modal evidence representations must "
                "have identical shapes."
            )

        if first_embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected evidence dimension {self.dimension}, "
                f"received {first_embedding.shape[-1]}."
            )

        difference = torch.abs(
            first_embedding
            - second_embedding
        )

        product = (
            first_embedding
            * second_embedding
        )

        cosine_similarity = F.cosine_similarity(
            first_embedding,
            second_embedding,
            dim=-1,
        ).unsqueeze(-1)

        interaction_features = torch.cat(
            [
                first_embedding,
                second_embedding,
                difference,
                product,
                cosine_similarity,
            ],
            dim=-1,
        )

        interaction_embedding = (
            self.interaction_projection(
                interaction_features
            )
        )

        return {
            "difference": difference,
            "product": product,
            "cosine_similarity": cosine_similarity,
            "interaction_embedding": interaction_embedding,
        }