"""
AEGIS interaction-only evidence fusion.

Version: 0.25.2

This module implements the M1 mechanism-isolation condition for AEGIS.

M1 deliberately retains explicit cross-modal evidence interaction while
removing modality reliability estimation and reliability-aware adaptive
weighting.

For aligned modality representations h_a and h_b:

    c_ab = CrossModalEvidenceInteraction(h_a, h_b)

    z_base = 0.5 * (h_a + h_b)

    z_M1 = LayerNorm(
        z_base + c_ab / sqrt(d)
    )

The 1/sqrt(d) factor is deterministic and parameter-free. It compensates
for the scale difference between L2-normalized aligned modality vectors
and the LayerNorm-produced interaction embedding, avoiding an arbitrary
learned reliability/gating mechanism in this ablation.

This module is an experimental ablation component. It is not intended to
replace the validated legacy gated fusion or the complete evidence-aware
M3 pathway.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .interaction import CrossModalEvidenceInteraction


class InteractionOnlyEvidenceFusion(nn.Module):
    """
    M1: explicit interaction plus deterministic residual fusion.
    """

    def __init__(
        self,
        dimension: int = 512,
        interaction_dropout: float = 0.1,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if not 0.0 <= interaction_dropout < 1.0:
            raise ValueError(
                "interaction_dropout must satisfy "
                "0.0 <= interaction_dropout < 1.0."
            )

        self.dimension = dimension

        self.interaction = CrossModalEvidenceInteraction(
            dimension=dimension,
            dropout=interaction_dropout,
        )

        self.output_normalization = nn.LayerNorm(
            dimension
        )

        self.interaction_scale = (
            1.0 / math.sqrt(float(dimension))
        )

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Fuse two aligned modality representations through M1.
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
                "Interaction-only evidence representations "
                "must have identical shapes."
            )

        if first_embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected evidence dimension {self.dimension}, "
                f"received {first_embedding.shape[-1]}."
            )

        interaction_output = self.interaction(
            first_embedding,
            second_embedding,
        )

        base_fusion = 0.5 * (
            first_embedding
            + second_embedding
        )

        scaled_interaction = (
            interaction_output[
                "interaction_embedding"
            ]
            * self.interaction_scale
        )

        fused_embedding = (
            self.output_normalization(
                base_fusion
                + scaled_interaction
            )
        )

        return {
            "first_embedding": first_embedding,
            "second_embedding": second_embedding,
            "difference": interaction_output[
                "difference"
            ],
            "product": interaction_output[
                "product"
            ],
            "cosine_similarity": interaction_output[
                "cosine_similarity"
            ],
            "interaction_embedding": (
                interaction_output[
                    "interaction_embedding"
                ]
            ),
            "base_fusion": base_fusion,
            "scaled_interaction": scaled_interaction,
            "interaction_scale": torch.tensor(
                self.interaction_scale,
                dtype=first_embedding.dtype,
                device=first_embedding.device,
            ),
            "fused_embedding": fused_embedding,
        }
