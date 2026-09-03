"""
AEGIS gated interaction evidence fusion.

Version: 0.25.3

M1b preserves the validated legacy gated multimodal fusion and adds an
explicit cross-modal interaction residual:

    z_gate = G(h_text, h_vision)
    c_ti   = phi([h_text, h_vision, |h_text-h_vision|,
                  h_text*h_vision, cosine(h_text,h_vision)])
    z      = LayerNorm(z_gate + c_ti / sqrt(d))

The interaction scaling is deterministic so this ablation introduces no
additional learned reliability or residual-gating mechanism.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .fusion import GatedMultimodalFusion
from .interaction import CrossModalEvidenceInteraction


class GatedInteractionEvidenceFusion(nn.Module):
    """Legacy gated fusion augmented with explicit interaction evidence."""

    def __init__(
        self,
        dimension: int = 512,
        interaction_dropout: float = 0.1,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        if not 0.0 <= interaction_dropout < 1.0:
            raise ValueError(
                "interaction_dropout must satisfy 0.0 <= "
                "interaction_dropout < 1.0."
            )

        self.dimension = dimension
        self.interaction_scale = 1.0 / math.sqrt(dimension)

        self.gated_fusion = GatedMultimodalFusion(
            dimension=dimension,
        )

        self.interaction = CrossModalEvidenceInteraction(
            dimension=dimension,
            dropout=interaction_dropout,
        )

        self.output_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse two aligned modality embeddings using M1b."""

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
                "Gated-interaction evidence representations must "
                "have identical shapes."
            )

        if first_embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected evidence dimension {self.dimension}, "
                f"received {first_embedding.shape[-1]}."
            )

        gated_fusion = self.gated_fusion(
            first_embedding,
            second_embedding,
        )

        interaction_output = self.interaction(
            first_embedding,
            second_embedding,
        )

        scaled_interaction = (
            interaction_output["interaction_embedding"]
            * self.interaction_scale
        )

        fused_embedding = self.output_norm(
            gated_fusion + scaled_interaction
        )

        return {
            "difference": interaction_output["difference"],
            "product": interaction_output["product"],
            "cosine_similarity": interaction_output[
                "cosine_similarity"
            ],
            "interaction_embedding": interaction_output[
                "interaction_embedding"
            ],
            "gated_base_fusion": gated_fusion,
            "scaled_interaction": scaled_interaction,
            "interaction_scale": torch.as_tensor(
                self.interaction_scale,
                dtype=fused_embedding.dtype,
                device=fused_embedding.device,
            ),
            "fused_embedding": fused_embedding,
        }
