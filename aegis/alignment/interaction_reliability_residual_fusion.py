"""
AEGIS M2b interaction-conditioned reliability residual fusion.

Version: 0.25.5

Controlled mechanism-isolation pathway
--------------------------------------
M2b preserves the validated M0 gated fusion and adds an interaction-conditioned
reliability residual:

    z_g = G(h_T, h_V)
    c_TV = phi([h_T; h_V; |h_T-h_V|; h_T*h_V; cos(h_T,h_V)])
    r_T = R_T(h_T, c_TV, s_TV)
    r_V = R_V(h_V, c_TV, s_TV)
    z_r = alpha_T*h_T + alpha_V*h_V
    z_M2b = LayerNorm(z_g + z_r/sqrt(d))

The gated base is supplied by CrossModalAlignmentModel.fusion. This block does
not own a duplicate gate, so M0, M2 and M2b share the same historical parent
gate initialization mechanism under the paired experiment seed.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .adaptive_fusion import ReliabilityAwareAdaptiveFusion
from .interaction import CrossModalEvidenceInteraction
from .reliability import EvidenceReliabilityEstimator


class InteractionConditionedReliabilityResidualFusion(nn.Module):
    """M2b interaction-conditioned reliability residual on a supplied gate."""

    def __init__(
        self,
        dimension: int = 512,
        reliability_hidden_dim: int = 256,
        interaction_dropout: float = 0.1,
        reliability_dropout: float = 0.1,
        fusion_temperature: float = 1.0,
    ):
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.dimension = dimension
        self.reliability_scale = 1.0 / math.sqrt(dimension)
        self.interaction = CrossModalEvidenceInteraction(
            dimension=dimension,
            dropout=interaction_dropout,
        )
        self.first_reliability_estimator = EvidenceReliabilityEstimator(
            dimension=dimension,
            hidden_dim=reliability_hidden_dim,
            dropout=reliability_dropout,
        )
        self.second_reliability_estimator = EvidenceReliabilityEstimator(
            dimension=dimension,
            hidden_dim=reliability_hidden_dim,
            dropout=reliability_dropout,
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
        self._validate_inputs(first_embedding, second_embedding, gated_base_fusion)

        interaction_output = self.interaction(first_embedding, second_embedding)
        interaction_embedding = interaction_output["interaction_embedding"]
        cosine_similarity = interaction_output["cosine_similarity"]

        first_reliability = self.first_reliability_estimator(
            first_embedding, interaction_embedding, cosine_similarity
        )
        second_reliability = self.second_reliability_estimator(
            second_embedding, interaction_embedding, cosine_similarity
        )
        adaptive_output = self.adaptive_fusion(
            first_embedding,
            second_embedding,
            first_reliability,
            second_reliability,
        )
        adaptive_fusion = adaptive_output["fused_embedding"]
        scaled_fusion = adaptive_fusion * self.reliability_scale
        fused_embedding = self.output_normalization(gated_base_fusion + scaled_fusion)

        return {
            "first_embedding": first_embedding,
            "second_embedding": second_embedding,
            "gated_base_fusion": gated_base_fusion,
            "difference": interaction_output["difference"],
            "product": interaction_output["product"],
            "cosine_similarity": cosine_similarity,
            "interaction_embedding": interaction_embedding,
            "first_reliability": first_reliability,
            "second_reliability": second_reliability,
            "first_weight": adaptive_output["first_weight"],
            "second_weight": adaptive_output["second_weight"],
            "weights": adaptive_output["weights"],
            "reliability_adaptive_fusion": adaptive_fusion,
            "scaled_reliability_fusion": scaled_fusion,
            "reliability_scale": torch.tensor(
                self.reliability_scale,
                dtype=first_embedding.dtype,
                device=first_embedding.device,
            ),
            "fused_embedding": fused_embedding,
        }

    def _validate_inputs(self, first_embedding, second_embedding, gated_base_fusion):
        for name, embedding in (
            ("first_embedding", first_embedding),
            ("second_embedding", second_embedding),
            ("gated_base_fusion", gated_base_fusion),
        ):
            if embedding.ndim != 2:
                raise ValueError(f"{name} must have shape [batch_size, dimension].")
            if embedding.shape[-1] != self.dimension:
                raise ValueError(
                    f"Expected {name} dimension {self.dimension}, "
                    f"received {embedding.shape[-1]}."
                )
        if first_embedding.shape != second_embedding.shape:
            raise ValueError("first_embedding and second_embedding must have identical shapes.")
        if gated_base_fusion.shape != first_embedding.shape:
            raise ValueError(
                "gated_base_fusion must have the same shape as the modality embeddings."
            )
        if not (
            first_embedding.device == second_embedding.device == gated_base_fusion.device
        ):
            raise ValueError("All M2b fusion inputs must be on the same device.")
