"""
Trainable projection heads for the AEGIS shared semantic space.

Version: 0.9.0
"""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """
    Projects modality-specific representations into
    a common semantic embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            output_dim,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.norm = nn.LayerNorm(
            output_dim
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)

        return F.normalize(
            x,
            p=2,
            dim=-1,
        )