"""
AEGIS reliability-aware adaptive evidence fusion.

Version: 0.25.0

This module combines aligned modality evidence using independently
estimated sample-level reliability scores.

Unlike feature-wise gated fusion, reliability-aware fusion assigns
explicit modality-level evidence weights that are observable and can
be exposed for analysis and explanation.
"""

from __future__ import annotations

import torch
from torch import nn


class ReliabilityAwareAdaptiveFusion(nn.Module):
    """
    Fuse aligned modality representations using reliability weights.

    Given aligned modality evidence h_a and h_b with corresponding
    reliability estimates r_a and r_b:

        alpha = softmax([r_a, r_b] / temperature)

        z = alpha_a * h_a + alpha_b * h_b

    Reliability scores are expected to lie in [0, 1].

    The fusion weights are returned explicitly so that AEGIS can
    inspect and report the contribution of each modality.
    """

    def __init__(
        self,
        dimension: int = 512,
        temperature: float = 1.0,
    ):
        super().__init__()

        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero."
            )

        if temperature <= 0.0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        self.dimension = dimension
        self.temperature = temperature

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        first_reliability: torch.Tensor,
        second_reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Perform reliability-aware adaptive fusion.

        Parameters
        ----------
        first_embedding:
            First aligned modality representation with shape
            [batch_size, dimension].

        second_embedding:
            Second aligned modality representation with shape
            [batch_size, dimension].

        first_reliability:
            Reliability of the first modality with shape
            [batch_size, 1].

        second_reliability:
            Reliability of the second modality with shape
            [batch_size, 1].

        Returns
        -------
        dict
            fused_embedding:
                Reliability-weighted multimodal evidence embedding.

            first_weight:
                Normalized contribution weight of the first modality.

            second_weight:
                Normalized contribution weight of the second modality.

            weights:
                Both normalized modality weights with shape
                [batch_size, 2].
        """

        self._validate_embedding(
            first_embedding,
            "first_embedding",
        )

        self._validate_embedding(
            second_embedding,
            "second_embedding",
        )

        if first_embedding.shape != second_embedding.shape:
            raise ValueError(
                "Aligned modality embeddings must have "
                "identical shapes."
            )

        self._validate_reliability(
            first_reliability,
            "first_reliability",
            first_embedding.shape[0],
        )

        self._validate_reliability(
            second_reliability,
            "second_reliability",
            second_embedding.shape[0],
        )

        devices = {
            first_embedding.device,
            second_embedding.device,
            first_reliability.device,
            second_reliability.device,
        }

        if len(devices) != 1:
            raise ValueError(
                "All adaptive fusion inputs must be "
                "on the same device."
            )

        reliability_scores = torch.cat(
            [
                first_reliability,
                second_reliability,
            ],
            dim=-1,
        )

        weights = torch.softmax(
            reliability_scores / self.temperature,
            dim=-1,
        )

        first_weight = weights[:, 0:1]
        second_weight = weights[:, 1:2]

        fused_embedding = (
            first_weight * first_embedding
            + second_weight * second_embedding
        )

        return {
            "fused_embedding": fused_embedding,
            "first_weight": first_weight,
            "second_weight": second_weight,
            "weights": weights,
        }

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

    @staticmethod
    def _validate_reliability(
        reliability: torch.Tensor,
        name: str,
        expected_batch_size: int,
    ) -> None:
        if reliability.ndim != 2:
            raise ValueError(
                f"{name} must have shape "
                "[batch_size, 1]."
            )

        if reliability.shape[-1] != 1:
            raise ValueError(
                f"{name} must contain exactly one "
                "reliability value per sample."
            )

        if reliability.shape[0] != expected_batch_size:
            raise ValueError(
                f"{name} batch size must match its "
                "modality embedding batch size."
            )

        if torch.any(reliability < 0.0):
            raise ValueError(
                f"{name} values must lie in [0, 1]."
            )

        if torch.any(reliability > 1.0):
            raise ValueError(
                f"{name} values must lie in [0, 1]."
            )