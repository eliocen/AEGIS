"""
AEGIS evidence integration block.

Version: 0.25.0

This module integrates explicit cross-modal interaction, modality-level
evidence reliability estimation, and reliability-aware adaptive fusion
into a unified trainable AEGIS evidence integration architecture.

The block operates on already aligned modality representations. It does
not replace modality encoders or projection heads.
"""

from __future__ import annotations

import torch
from torch import nn

from aegis.alignment.adaptive_fusion import (
    ReliabilityAwareAdaptiveFusion,
)
from aegis.alignment.interaction import (
    CrossModalEvidenceInteraction,
)
from aegis.alignment.reliability import (
    EvidenceReliabilityEstimator,
)


class AEGISEvidenceIntegrationBlock(nn.Module):
    """
    Integrate aligned evidence from two modalities.

    Processing sequence:

        aligned modality evidence
                |
                v
        cross-modal interaction
                |
                v
        modality reliability estimation
                |
                v
        reliability-aware adaptive fusion
                |
                v
        integrated evidence state

    The block preserves intermediate evidence signals for downstream
    classification, uncertainty estimation, explanation, and ablation.

    Parameters
    ----------
    dimension:
        Shared dimensionality of aligned modality representations.

    reliability_hidden_dim:
        Hidden dimensionality used by each reliability estimator.

    interaction_dropout:
        Dropout used by the interaction projection.

    reliability_dropout:
        Dropout used by the reliability estimators.

    fusion_temperature:
        Temperature controlling normalization of modality reliability
        scores into adaptive fusion weights.
    """

    def __init__(
        self,
        dimension: int = 512,
        reliability_hidden_dim: int = 256,
        interaction_dropout: float = 0.1,
        reliability_dropout: float = 0.1,
        fusion_temperature: float = 1.0,
    ):
        super().__init__()

        self.dimension = dimension

        self.interaction = CrossModalEvidenceInteraction(
            dimension=dimension,
            dropout=interaction_dropout,
        )

        # Separate estimators are deliberate. Text and image evidence
        # need not learn identical reliability functions.
        self.first_reliability_estimator = (
            EvidenceReliabilityEstimator(
                dimension=dimension,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
            )
        )

        self.second_reliability_estimator = (
            EvidenceReliabilityEstimator(
                dimension=dimension,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
            )
        )

        self.adaptive_fusion = (
            ReliabilityAwareAdaptiveFusion(
                dimension=dimension,
                temperature=fusion_temperature,
            )
        )

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Construct an integrated AEGIS evidence state.

        Parameters
        ----------
        first_embedding:
            First aligned modality representation with shape
            [batch_size, dimension].

        second_embedding:
            Second aligned modality representation with shape
            [batch_size, dimension].

        Returns
        -------
        dict
            first_embedding:
                Original first aligned evidence representation.

            second_embedding:
                Original second aligned evidence representation.

            difference:
                Absolute cross-modal difference.

            product:
                Element-wise cross-modal product.

            cosine_similarity:
                Scalar cross-modal agreement.

            interaction_embedding:
                Learned explicit interaction evidence.

            first_reliability:
                Estimated reliability of the first modality.

            second_reliability:
                Estimated reliability of the second modality.

            first_weight:
                Normalized adaptive contribution of the first modality.

            second_weight:
                Normalized adaptive contribution of the second modality.

            weights:
                Both adaptive modality weights.

            fused_embedding:
                Reliability-aware integrated modality evidence.
        """

        interaction_output = self.interaction(
            first_embedding,
            second_embedding,
        )

        interaction_embedding = (
            interaction_output[
                "interaction_embedding"
            ]
        )

        cosine_similarity = (
            interaction_output[
                "cosine_similarity"
            ]
        )

        first_reliability = (
            self.first_reliability_estimator(
                first_embedding,
                interaction_embedding,
                cosine_similarity,
            )
        )

        second_reliability = (
            self.second_reliability_estimator(
                second_embedding,
                interaction_embedding,
                cosine_similarity,
            )
        )

        fusion_output = self.adaptive_fusion(
            first_embedding,
            second_embedding,
            first_reliability,
            second_reliability,
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
            "cosine_similarity": cosine_similarity,
            "interaction_embedding": (
                interaction_embedding
            ),
            "first_reliability": first_reliability,
            "second_reliability": second_reliability,
            "first_weight": fusion_output[
                "first_weight"
            ],
            "second_weight": fusion_output[
                "second_weight"
            ],
            "weights": fusion_output[
                "weights"
            ],
            "fused_embedding": fusion_output[
                "fused_embedding"
            ],
        }