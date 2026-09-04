"""
AEGIS v0.27 — Cross-Modal Compatibility Estimator.

This module implements the standalone M4qc compatibility estimator frozen in
the Step 11A protocol.

Scientific scope
----------------
The estimator predicts pairwise text-image compatibility from projected AEGIS
representations:

    c_TV = C(h_T, h_V),    c_TV in [0, 1]

It intentionally does NOT:
    - estimate intrinsic modality quality;
    - alter multimodal fusion;
    - perform factual verification;
    - infer truthfulness, source credibility, or intent.

A clean text representation and a clean vision representation may each have
high intrinsic quality while being mutually incompatible. Compatibility is
therefore represented by a separate learned head rather than by the M4q
quality estimators.

Frozen Step 11A pair representation
-----------------------------------
For h_T, h_V in R^d:

    z_C = [
        h_T;
        h_V;
        |h_T - h_V|;
        h_T * h_V;
        cosine(h_T, h_V)
    ]

so:

    dim(z_C) = 4d + 1

Frozen estimator architecture
-----------------------------
    (4d + 1) -> 256 -> 64 -> 1
    GELU
    dropout = 0.1
    sigmoid output

M4qc integration is intentionally outside this module and belongs to the
subsequent protocol step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CompatibilityEstimatorConfig:
    """Serializable standalone compatibility-estimator configuration."""

    shared_dim: int
    hidden_dims: Tuple[int, int] = (256, 64)
    dropout: float = 0.1
    cosine_eps: float = 1e-8


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _validate_hidden_dims(hidden_dims: Sequence[int]) -> Tuple[int, int]:
    if isinstance(hidden_dims, (str, bytes)) or not isinstance(
        hidden_dims, Sequence
    ):
        raise TypeError("hidden_dims must be a sequence of exactly two integers.")

    values = tuple(hidden_dims)
    if len(values) != 2:
        raise ValueError(
            "hidden_dims must contain exactly two dimensions; "
            f"got {len(values)}."
        )

    first = _validate_positive_int(values[0], name="hidden_dims[0]")
    second = _validate_positive_int(values[1], name="hidden_dims[1]")
    return first, second


def _validate_dropout(dropout: float) -> float:
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be a real number in [0, 1).")
    value = float(dropout)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"dropout must be in [0, 1); got {value}.")
    return value


def _validate_positive_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive real number.")
    result = float(value)
    if not result > 0.0:
        raise ValueError(f"{name} must be > 0; got {result}.")
    return result


class CrossModalCompatibilityEstimator(nn.Module):
    """
    Estimate text-image compatibility from two projected representations.

    Parameters
    ----------
    shared_dim:
        Common projected dimensionality d of h_T and h_V.
    hidden_dims:
        Frozen Step 11A default: (256, 64).
    dropout:
        Frozen Step 11A default: 0.1.
    cosine_eps:
        Numerical-stability epsilon for cosine similarity.

    Forward
    -------
    text_embedding:
        Floating tensor with shape [batch, shared_dim].
    vision_embedding:
        Floating tensor with shape [batch, shared_dim].

    Returns
    -------
    Tensor:
        Compatibility score with shape [batch, 1], bounded to [0, 1] by
        sigmoid.
    """

    def __init__(
        self,
        shared_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.1,
        cosine_eps: float = 1e-8,
    ) -> None:
        super().__init__()

        shared_dim = _validate_positive_int(shared_dim, name="shared_dim")
        hidden_1, hidden_2 = _validate_hidden_dims(hidden_dims)
        dropout = _validate_dropout(dropout)
        cosine_eps = _validate_positive_real(cosine_eps, name="cosine_eps")

        self.config = CompatibilityEstimatorConfig(
            shared_dim=shared_dim,
            hidden_dims=(hidden_1, hidden_2),
            dropout=dropout,
            cosine_eps=cosine_eps,
        )

        self.network = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

    @property
    def shared_dim(self) -> int:
        return self.config.shared_dim

    @property
    def hidden_dims(self) -> Tuple[int, int]:
        return self.config.hidden_dims

    @property
    def dropout_probability(self) -> float:
        return self.config.dropout

    @property
    def cosine_eps(self) -> float:
        return self.config.cosine_eps

    @property
    def feature_dim(self) -> int:
        return 4 * self.shared_dim + 1

    def _validate_pair(
        self,
        text_embedding: Tensor,
        vision_embedding: Tensor,
    ) -> None:
        if not isinstance(text_embedding, Tensor):
            raise TypeError("text_embedding must be a torch.Tensor.")
        if not isinstance(vision_embedding, Tensor):
            raise TypeError("vision_embedding must be a torch.Tensor.")

        if text_embedding.ndim != 2:
            raise ValueError(
                "text_embedding must have shape [batch, shared_dim]; "
                f"got {tuple(text_embedding.shape)}."
            )
        if vision_embedding.ndim != 2:
            raise ValueError(
                "vision_embedding must have shape [batch, shared_dim]; "
                f"got {tuple(vision_embedding.shape)}."
            )

        if text_embedding.shape[0] < 1:
            raise ValueError("embeddings must contain at least one sample.")
        if text_embedding.shape[0] != vision_embedding.shape[0]:
            raise ValueError(
                "text and vision batch sizes must match: "
                f"{text_embedding.shape[0]} != {vision_embedding.shape[0]}."
            )

        if text_embedding.shape[1] != self.shared_dim:
            raise ValueError(
                "text_embedding feature dimension does not match shared_dim: "
                f"{text_embedding.shape[1]} != {self.shared_dim}."
            )
        if vision_embedding.shape[1] != self.shared_dim:
            raise ValueError(
                "vision_embedding feature dimension does not match shared_dim: "
                f"{vision_embedding.shape[1]} != {self.shared_dim}."
            )

        if not torch.is_floating_point(text_embedding):
            raise TypeError(
                "text_embedding must have a floating-point dtype; "
                f"got {text_embedding.dtype}."
            )
        if not torch.is_floating_point(vision_embedding):
            raise TypeError(
                "vision_embedding must have a floating-point dtype; "
                f"got {vision_embedding.dtype}."
            )

        if text_embedding.dtype != vision_embedding.dtype:
            raise TypeError(
                "text and vision embeddings must have the same dtype: "
                f"{text_embedding.dtype} != {vision_embedding.dtype}."
            )
        if text_embedding.device != vision_embedding.device:
            raise ValueError(
                "text and vision embeddings must be on the same device: "
                f"{text_embedding.device} != {vision_embedding.device}."
            )

        if not torch.isfinite(text_embedding).all():
            raise ValueError(
                "text_embedding contains NaN or infinite values."
            )
        if not torch.isfinite(vision_embedding).all():
            raise ValueError(
                "vision_embedding contains NaN or infinite values."
            )

    def build_pair_features(
        self,
        text_embedding: Tensor,
        vision_embedding: Tensor,
    ) -> Tensor:
        """
        Construct the frozen Step 11A ordered pair feature z_C.

        The operation is non-mutating. Text and vision roles remain ordered;
        the estimator is not required to be invariant to swapping modalities.
        """
        self._validate_pair(text_embedding, vision_embedding)

        absolute_difference = torch.abs(text_embedding - vision_embedding)
        elementwise_product = text_embedding * vision_embedding
        cosine_similarity = F.cosine_similarity(
            text_embedding,
            vision_embedding,
            dim=1,
            eps=self.cosine_eps,
        ).unsqueeze(1)

        pair_features = torch.cat(
            (
                text_embedding,
                vision_embedding,
                absolute_difference,
                elementwise_product,
                cosine_similarity,
            ),
            dim=1,
        )

        if pair_features.shape != (
            text_embedding.shape[0],
            self.feature_dim,
        ):
            raise RuntimeError(
                "Internal compatibility feature shape invariant failed: "
                f"got {tuple(pair_features.shape)}, expected "
                f"({text_embedding.shape[0]}, {self.feature_dim})."
            )
        if not torch.isfinite(pair_features).all():
            raise RuntimeError(
                "Constructed compatibility features contain NaN or infinite values."
            )

        return pair_features

    def forward(
        self,
        text_embedding: Tensor,
        vision_embedding: Tensor,
    ) -> Tensor:
        pair_features = self.build_pair_features(
            text_embedding,
            vision_embedding,
        )
        return self.network(pair_features)

    def architecture_metadata(self) -> dict[str, object]:
        """Return stable, JSON-serializable architecture metadata."""
        return {
            "module": self.__class__.__name__,
            "shared_dim": self.shared_dim,
            "pair_feature_dim": self.feature_dim,
            "pair_features": [
                "text",
                "vision",
                "absolute_difference",
                "elementwise_product",
                "cosine_similarity",
            ],
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_probability,
            "cosine_eps": self.cosine_eps,
            "output_dim": 1,
            "hidden_activation": "GELU",
            "output_activation": "Sigmoid",
            "semantic_role": "cross_modal_compatibility",
            "cross_modal_input": True,
            "ordered_modalities": ["text", "vision"],
            "affects_primary_fusion": False,
        }
