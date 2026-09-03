"""
AEGIS M2 reliability-only residual fusion.

Version: 0.25.4

Controlled mechanism-isolation pathway
--------------------------------------
M2 preserves the validated M0 gated fusion and adds a modality-only
reliability residual:

    z_g = G(h_T, h_V)

    r_T = R_T(h_T)
    r_V = R_V(h_V)

    [alpha_T, alpha_V] =
        ReliabilityAwareAdaptiveFusion(r_T, r_V)

    z_r = alpha_T h_T + alpha_V h_V

    z_M2 = LayerNorm(z_g + z_r / sqrt(d))

The reliability estimators deliberately depend only on their own modality.
There is no explicit cross-modal interaction embedding, difference/product
feature, or cosine-conditioned reliability in M2.

The gated base is supplied by CrossModalAlignmentModel.fusion. This means M0
and M2 use the exact same legacy gate object/initialization mechanism, avoiding
a duplicate-gate initialization confound.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .adaptive_fusion import ReliabilityAwareAdaptiveFusion


class ModalityOnlyReliabilityEstimator(nn.Module):
    """Estimate scalar reliability from one aligned modality only."""

    def __init__(
        self,
        dimension: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        self.dimension = dimension
        self.hidden_dim = hidden_dim

        self.estimator = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        modality_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if modality_embedding.ndim != 2:
            raise ValueError(
                "modality_embedding must have shape "
                "[batch_size, dimension]."
            )

        if modality_embedding.shape[-1] != self.dimension:
            raise ValueError(
                "Expected modality_embedding dimension "
                f"{self.dimension}, received "
                f"{modality_embedding.shape[-1]}."
            )

        return self.estimator(modality_embedding)


class ReliabilityResidualFusion(nn.Module):
    """
    M2 modality-only reliability residual layered on a supplied gated base.

    This block does not own a gated fusion layer. The base gate is computed by
    CrossModalAlignmentModel.fusion so the M0/M2 gate is literally the same
    historical component.
    """

    def __init__(
        self,
        dimension: int = 512,
        reliability_hidden_dim: int = 256,
        reliability_dropout: float = 0.1,
        fusion_temperature: float = 1.0,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.dimension = dimension
        self.reliability_scale = 1.0 / math.sqrt(dimension)

        self.first_reliability_estimator = (
            ModalityOnlyReliabilityEstimator(
                dimension=dimension,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
            )
        )

        self.second_reliability_estimator = (
            ModalityOnlyReliabilityEstimator(
                dimension=dimension,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
            )
        )

        self.adaptive_fusion = ReliabilityAwareAdaptiveFusion(
            dimension=dimension,
            temperature=fusion_temperature,
        )

        self.output_normalization = nn.LayerNorm(dimension)

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        gated_base_fusion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            first_embedding,
            second_embedding,
            gated_base_fusion,
        )

        first_reliability = self.first_reliability_estimator(
            first_embedding
        )
        second_reliability = self.second_reliability_estimator(
            second_embedding
        )

        adaptive_output = self.adaptive_fusion(
            first_embedding,
            second_embedding,
            first_reliability,
            second_reliability,
        )

        reliability_adaptive_fusion = adaptive_output[
            "fused_embedding"
        ]

        scaled_reliability_fusion = (
            reliability_adaptive_fusion
            * self.reliability_scale
        )

        fused_embedding = self.output_normalization(
            gated_base_fusion
            + scaled_reliability_fusion
        )

        return {
            "first_embedding": first_embedding,
            "second_embedding": second_embedding,
            "gated_base_fusion": gated_base_fusion,
            "first_reliability": first_reliability,
            "second_reliability": second_reliability,
            "first_weight": adaptive_output["first_weight"],
            "second_weight": adaptive_output["second_weight"],
            "weights": adaptive_output["weights"],
            "reliability_adaptive_fusion": (
                reliability_adaptive_fusion
            ),
            "scaled_reliability_fusion": (
                scaled_reliability_fusion
            ),
            "reliability_scale": torch.tensor(
                self.reliability_scale,
                dtype=first_embedding.dtype,
                device=first_embedding.device,
            ),
            "fused_embedding": fused_embedding,
        }

    def _validate_inputs(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        gated_base_fusion: torch.Tensor,
    ) -> None:
        for name, embedding in (
            ("first_embedding", first_embedding),
            ("second_embedding", second_embedding),
            ("gated_base_fusion", gated_base_fusion),
        ):
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

        if first_embedding.shape != second_embedding.shape:
            raise ValueError(
                "first_embedding and second_embedding must have "
                "identical shapes."
            )

        if gated_base_fusion.shape != first_embedding.shape:
            raise ValueError(
                "gated_base_fusion must have the same shape as "
                "the modality embeddings."
            )

        if not (
            first_embedding.device
            == second_embedding.device
            == gated_base_fusion.device
        ):
            raise ValueError(
                "All M2 fusion inputs must be on the same device."
            )
