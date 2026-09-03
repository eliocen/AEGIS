"""
AEGIS cross-modal semantic alignment model.

Version: 0.25.0

This module preserves the validated v0.24 cross-modal alignment
path while adding an optional evidence-aware integration pathway.

Legacy pathway:
    text / vision projections
        -> gated multimodal fusion

Evidence-aware pathway:
    text / vision projections
        -> explicit cross-modal interaction
        -> modality reliability estimation
        -> reliability-aware adaptive fusion

The contrastive alignment objective remains defined over the
aligned modality representations and is independent of the
selected fusion pathway.
"""

from __future__ import annotations

import torch
from torch import nn

from .evidence_integration import (
    AEGISEvidenceIntegrationBlock,
)
from .fusion import GatedMultimodalFusion
from .loss import SymmetricContrastiveLoss
from .projection import ProjectionHead


class CrossModalAlignmentModel(nn.Module):
    """
    Trainable AEGIS cross-modal semantic alignment subsystem.

    Maps heterogeneous text and visual representations into a
    shared semantic embedding space.

    Parameters
    ----------
    text_dim:
        Dimensionality of the upstream text representation.

    vision_dim:
        Dimensionality of the upstream vision representation.

    shared_dim:
        Shared AEGIS evidence-space dimensionality.

    dropout:
        Dropout used by the modality projection heads.

    temperature:
        Temperature used by the symmetric contrastive
        alignment objective.

    evidence_aware:
        If False, preserve the validated v0.24 gated-fusion
        pathway.

        If True, use the v0.25 AEGIS evidence integration
        pathway consisting of explicit cross-modal interaction,
        modality reliability estimation, and reliability-aware
        adaptive fusion.

    evidence_reliability_hidden_dim:
        Hidden dimensionality of each modality reliability
        estimator when evidence-aware integration is enabled.

    evidence_interaction_dropout:
        Dropout used by the learned cross-modal interaction
        projection.

    evidence_reliability_dropout:
        Dropout used by the modality reliability estimators.

    evidence_fusion_temperature:
        Temperature controlling conversion of modality
        reliability scores into normalized adaptive fusion
        weights.
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        shared_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
        evidence_aware: bool = False,
        evidence_reliability_hidden_dim: int = 256,
        evidence_interaction_dropout: float = 0.1,
        evidence_reliability_dropout: float = 0.1,
        evidence_fusion_temperature: float = 1.0,
    ):
        super().__init__()

        if not isinstance(evidence_aware, bool):
            raise TypeError(
                "evidence_aware must be a bool."
            )

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.shared_dim = shared_dim

        self.evidence_aware = evidence_aware

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

        if self.evidence_aware:
            self.evidence_integration = (
                AEGISEvidenceIntegrationBlock(
                    dimension=shared_dim,
                    reliability_hidden_dim=(
                        evidence_reliability_hidden_dim
                    ),
                    interaction_dropout=(
                        evidence_interaction_dropout
                    ),
                    reliability_dropout=(
                        evidence_reliability_dropout
                    ),
                    fusion_temperature=(
                        evidence_fusion_temperature
                    ),
                )
            )
        else:
            self.evidence_integration = None

        self.contrastive_loss = (
            SymmetricContrastiveLoss(
                temperature=temperature
            )
        )

    def align(
        self,
        text_embedding: torch.Tensor,
        vision_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project text and vision representations into the
        shared AEGIS semantic evidence space.
        """

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
        text_embedding: torch.Tensor,
        vision_embedding: torch.Tensor,
        compute_loss: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Align and integrate text and visual evidence.
        """

        aligned_text, aligned_vision = (
            self.align(
                text_embedding,
                vision_embedding,
            )
        )

        if self.evidence_aware:
            if self.evidence_integration is None:
                raise RuntimeError(
                    "Evidence-aware mode is enabled but "
                    "the evidence integration block is "
                    "not initialized."
                )

            evidence_output = (
                self.evidence_integration(
                    aligned_text,
                    aligned_vision,
                )
            )

            fused = evidence_output[
                "fused_embedding"
            ]

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": fused,
                "evidence_difference": (
                    evidence_output["difference"]
                ),
                "evidence_product": (
                    evidence_output["product"]
                ),
                "cosine_similarity": (
                    evidence_output[
                        "cosine_similarity"
                    ]
                ),
                "interaction_embedding": (
                    evidence_output[
                        "interaction_embedding"
                    ]
                ),
                "text_reliability": (
                    evidence_output[
                        "first_reliability"
                    ]
                ),
                "vision_reliability": (
                    evidence_output[
                        "second_reliability"
                    ]
                ),
                "text_weight": (
                    evidence_output[
                        "first_weight"
                    ]
                ),
                "vision_weight": (
                    evidence_output[
                        "second_weight"
                    ]
                ),
                "evidence_weights": (
                    evidence_output[
                        "weights"
                    ]
                ),
            }

        else:
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