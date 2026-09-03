"""Estimate reliability for aligned modality interaction evidence.

Reliability is distinct from feature-wise fusion. It represents how
strongly AEGIS should trust a modality as evidence for the current
sample before adaptive evidence integration.
"""

from __future__ import annotations

import torch
from torch import nn


class EvidenceReliabilityEstimator(nn.Module):
    """
    Estimate scalar reliability for aligned modality evidence.

    The estimator conditions reliability on:

        1. the modality-specific aligned representation,
        2. the learned cross-modal interaction representation,
        3. the scalar cross-modal cosine agreement.

    For modality m:

        r_m = sigmoid(
            g_m([h_m, c_ab, s_ab])
        )

    where:

        h_m  = aligned modality representation,
        c_ab = learned cross-modal interaction representation,
        s_ab = cosine agreement between the modality pair.

    The returned reliability score has shape:

        [batch_size, 1]

    and lies in the interval [0, 1].
    """

    def __init__(
        self,
        dimension: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        self.dimension = dimension
        self.hidden_dim = hidden_dim

        reliability_input_dim = (
            dimension * 2
            + 1
        )

        self.estimator = nn.Sequential(
            nn.Linear(
                reliability_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        modality_embedding: torch.Tensor,
        interaction_embedding: torch.Tensor,
        cosine_similarity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate modality reliability.

        Parameters
        ----------
        modality_embedding:
            Aligned modality representation with shape
            [batch_size, dimension].

        interaction_embedding:
            Learned cross-modal interaction representation with shape
            [batch_size, dimension].

        cosine_similarity:
            Cross-modal cosine agreement with shape
            [batch_size, 1].

        Returns
        -------
        torch.Tensor
            Reliability score with shape [batch_size, 1].
        """

        self._validate_embedding(
            modality_embedding,
            "modality_embedding",
        )

        self._validate_embedding(
            interaction_embedding,
            "interaction_embedding",
        )

        if (
            modality_embedding.shape
            != interaction_embedding.shape
        ):
            raise ValueError(
                "modality_embedding and interaction_embedding "
                "must have identical shapes."
            )

        if cosine_similarity.ndim != 2:
            raise ValueError(
                "cosine_similarity must have shape "
                "[batch_size, 1]."
            )

        if cosine_similarity.shape[-1] != 1:
            raise ValueError(
                "cosine_similarity must have exactly one "
                "feature per sample."
            )

        if (
            cosine_similarity.shape[0]
            != modality_embedding.shape[0]
        ):
            raise ValueError(
                "cosine_similarity batch size must match "
                "the modality embedding batch size."
            )

        if (
            modality_embedding.device
            != interaction_embedding.device
            or modality_embedding.device
            != cosine_similarity.device
        ):
            raise ValueError(
                "All reliability inputs must be on the same device."
            )

        reliability_features = torch.cat(
            [
                modality_embedding,
                interaction_embedding,
                cosine_similarity,
            ],
            dim=-1,
        )

        return self.estimator(
            reliability_features
        )

    def _validate_embedding(
        self,
        embedding: torch.Tensor,
        name: str,
    ) -> None:
        if embedding.ndim != 2:
            raise ValueError(
                f"{name} must have shape "
                "[batch_size, dimension]."
            )

        if embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected {name} dimension "
                f"{self.dimension}, received "
                f"{embedding.shape[-1]}."
            )
