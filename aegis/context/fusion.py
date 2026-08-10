"""
Context-aware multimodal fusion for AEGIS.

Version: 0.11.0
"""

import torch
from torch import nn


class ContextAwareFusion(nn.Module):
    """
    Fuses the multimodal content representation
    with operational context.
    """

    def __init__(
        self,
        content_dim: int = 512,
        context_dim: int = 128,
        output_dim: int = 512,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.context_dim = context_dim
        self.output_dim = output_dim

        self.context_projection = nn.Linear(
            context_dim,
            content_dim,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                content_dim * 2,
                content_dim,
            ),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                content_dim,
                output_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                output_dim
            ),
        )

    def forward(
        self,
        content_embedding,
        context_embedding,
    ):
        if content_embedding.ndim == 1:
            content_embedding = (
                content_embedding.unsqueeze(0)
            )

        if context_embedding.ndim == 1:
            context_embedding = (
                context_embedding.unsqueeze(0)
            )

        if (
            content_embedding.shape[-1]
            != self.content_dim
        ):
            raise ValueError(
                f"Expected content dimension "
                f"{self.content_dim}, received "
                f"{content_embedding.shape[-1]}."
            )

        if (
            context_embedding.shape[-1]
            != self.context_dim
        ):
            raise ValueError(
                f"Expected context dimension "
                f"{self.context_dim}, received "
                f"{context_embedding.shape[-1]}."
            )

        projected_context = (
            self.context_projection(
                context_embedding
            )
        )

        combined = torch.cat(
            [
                content_embedding,
                projected_context,
            ],
            dim=-1,
        )

        gate = self.gate(
            combined
        )

        fused = (
            gate * content_embedding
            +
            (1.0 - gate)
            * projected_context
        )

        return self.output_projection(
            fused
        )