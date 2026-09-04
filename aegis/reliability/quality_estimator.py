"""
AEGIS v0.27 — Intrinsic Modality Quality Estimator.

This module implements the first learned component of the frozen v0.27
corruption-aware supervision protocol.

Scientific scope
----------------
The estimator predicts intrinsic quality for ONE modality representation:

    q_m = Q_m(h_m),    q_m in [0, 1]

It intentionally does NOT:
    - inspect the other modality;
    - estimate cross-modal compatibility;
    - alter multimodal fusion;
    - perform factual verification;
    - infer truthfulness or intent.

M4q will instantiate separate estimators for text and vision. Keeping the
estimators separate prevents parameter sharing from silently imposing a common
quality function across heterogeneous modalities.

Frozen architecture
-------------------
    d -> 256 -> 64 -> 1
    GELU
    dropout = 0.1
    sigmoid output

The estimator operates on learned AEGIS projected embeddings h_m, not raw
encoder representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QualityEstimatorConfig:
    """Serializable architecture configuration."""

    input_dim: int
    hidden_dims: Tuple[int, int] = (256, 64)
    dropout: float = 0.1


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


class ModalityQualityEstimator(nn.Module):
    """
    Estimate intrinsic quality from one modality embedding.

    Parameters
    ----------
    input_dim:
        Dimensionality d of the projected modality representation.
    hidden_dims:
        Frozen v0.27 default: (256, 64).
    dropout:
        Frozen v0.27 default: 0.1.

    Forward
    -------
    embedding:
        Floating tensor with shape [batch, input_dim].

    Returns
    -------
    Tensor:
        Quality score with shape [batch, 1], bounded to [0, 1] by sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        input_dim = _validate_positive_int(input_dim, name="input_dim")
        hidden_1, hidden_2 = _validate_hidden_dims(hidden_dims)
        dropout = _validate_dropout(dropout)

        self.config = QualityEstimatorConfig(
            input_dim=input_dim,
            hidden_dims=(hidden_1, hidden_2),
            dropout=dropout,
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 1),
            nn.Sigmoid(),
        )

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def hidden_dims(self) -> Tuple[int, int]:
        return self.config.hidden_dims

    @property
    def dropout_probability(self) -> float:
        return self.config.dropout

    def forward(self, embedding: Tensor) -> Tensor:
        if not isinstance(embedding, Tensor):
            raise TypeError("embedding must be a torch.Tensor.")

        if embedding.ndim != 2:
            raise ValueError(
                "embedding must have shape [batch, input_dim]; "
                f"got {tuple(embedding.shape)}."
            )

        if embedding.shape[0] < 1:
            raise ValueError("embedding must contain at least one sample.")

        if embedding.shape[1] != self.input_dim:
            raise ValueError(
                "embedding feature dimension does not match estimator "
                f"input_dim: {embedding.shape[1]} != {self.input_dim}."
            )

        if not torch.is_floating_point(embedding):
            raise TypeError(
                "embedding must have a floating-point dtype; "
                f"got {embedding.dtype}."
            )

        if not torch.isfinite(embedding).all():
            raise ValueError("embedding contains NaN or infinite values.")

        return self.network(embedding)

    def architecture_metadata(self) -> dict[str, object]:
        """
        Return stable, JSON-serializable architecture metadata.
        """
        return {
            "module": self.__class__.__name__,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout_probability,
            "output_dim": 1,
            "hidden_activation": "GELU",
            "output_activation": "Sigmoid",
            "semantic_role": "intrinsic_modality_quality",
            "cross_modal_input": False,
            "affects_primary_fusion": False,
        }
