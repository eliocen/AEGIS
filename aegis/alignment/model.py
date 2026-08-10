"""
AEGIS cross-modal semantic alignment model.

Version: 0.9.0
"""

import torch
from torch import nn

from .fusion import GatedMultimodalFusion
from .loss import SymmetricContrastiveLoss
from .projection import ProjectionHead


class CrossModalAlignmentModel(nn.Module):
    """
    Trainable AEGIS cross-modal semantic alignment subsystem.

    Maps heterogeneous text and visual representations
    into a shared semantic embedding space.
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        shared_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.shared_dim = shared_dim

        self.text_projection = ProjectionHead(
            input_dim=text_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )

        self.vision_projection = ProjectionHead(
            input_dim=vision_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )

        self.fusion = GatedMultimodalFusion(
            dimension=shared_dim
        )

        self.contrastive_loss = (
            SymmetricContrastiveLoss(
                temperature=temperature
            )
        )

    def align(
        self,
        text_embedding,
        vision_embedding,
    ):
        aligned_text = self.text_projection(
            text_embedding
        )

        aligned_vision = self.vision_projection(
            vision_embedding
        )

        return (
            aligned_text,
            aligned_vision,
        )

    def forward(
        self,
        text_embedding,
        vision_embedding,
        compute_loss: bool = False,
    ):
        aligned_text, aligned_vision = (
            self.align(
                text_embedding,
                vision_embedding,
            )
        )

        fused = self.fusion(
            aligned_text,
            aligned_vision,
        )

        result = {
            "aligned_text": aligned_text,
            "aligned_vision": aligned_vision,
            "fused_embedding": fused,
        }

        if compute_loss:
            result["alignment_loss"] = (
                self.contrastive_loss(
                    aligned_text,
                    aligned_vision,
                )
            )

        return result