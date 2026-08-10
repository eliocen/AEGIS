"""
Multimodal fusion components for AEGIS.

Version: 0.9.0
"""

import torch
from torch import nn


class GatedMultimodalFusion(nn.Module):
    """
    Learnable gated fusion between aligned textual
    and visual representations.

    The gate learns how strongly each feature dimension
    should rely on textual versus visual evidence.
    """

    def __init__(
        self,
        dimension: int = 512,
    ):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                dimension * 2,
                dimension,
            ),
            nn.Sigmoid(),
        )

        self.output_norm = nn.LayerNorm(
            dimension
        )

    def forward(
        self,
        text_embedding,
        vision_embedding,
    ):
        if text_embedding.shape != vision_embedding.shape:
            raise ValueError(
                "Aligned text and vision embeddings "
                "must have identical shapes."
            )

        combined = torch.cat(
            [
                text_embedding,
                vision_embedding,
            ],
            dim=-1,
        )

        gate = self.gate(
            combined
        )

        fused = (
            gate * text_embedding
            + (1.0 - gate) * vision_embedding
        )

        return self.output_norm(
            fused
        )