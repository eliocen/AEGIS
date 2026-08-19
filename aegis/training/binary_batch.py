"""
AEGIS Binary Integrity Training Batch.

Version: 0.24.0

Used for datasets that support only Stage-1 integrity supervision:

    0 = True
    1 = Harmful / non-authentic

No threat subtype target is stored or manufactured.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class BinaryIntegrityBatch:
    """
    Multimodal batch for binary AEGIS integrity training.

    Expected dimensions:
        text_embeddings   -> [B, text_dim]
        vision_embeddings -> [B, vision_dim]
        integrity_targets -> [B]

    There is intentionally no threat_targets field.
    """

    text_embeddings: torch.Tensor
    vision_embeddings: torch.Tensor
    integrity_targets: torch.Tensor

    sample_ids: Optional[List[str]] = field(
        default=None
    )

    def validate(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
    ) -> None:

        if self.text_embeddings.ndim != 2:
            raise ValueError(
                "text_embeddings must have shape "
                "[batch_size, text_dimension]."
            )

        if self.vision_embeddings.ndim != 2:
            raise ValueError(
                "vision_embeddings must have shape "
                "[batch_size, vision_dimension]."
            )

        if (
            self.text_embeddings.shape[-1]
            != text_dim
        ):
            raise ValueError(
                f"Expected text dimension {text_dim}, "
                f"received "
                f"{self.text_embeddings.shape[-1]}."
            )

        if (
            self.vision_embeddings.shape[-1]
            != vision_dim
        ):
            raise ValueError(
                f"Expected vision dimension {vision_dim}, "
                f"received "
                f"{self.vision_embeddings.shape[-1]}."
            )

        batch_size = (
            self.text_embeddings.shape[0]
        )

        if (
            self.vision_embeddings.shape[0]
            != batch_size
        ):
            raise ValueError(
                "Text and vision batch sizes must match."
            )

        if (
            self.integrity_targets.ndim != 1
            or self.integrity_targets.shape[0]
            != batch_size
        ):
            raise ValueError(
                "integrity_targets must have shape "
                "[batch_size]."
            )

        if self.sample_ids is not None:
            if len(self.sample_ids) != batch_size:
                raise ValueError(
                    "sample_ids length must equal "
                    "batch_size."
                )

        if not torch.all(
            (self.integrity_targets >= 0)
            & (self.integrity_targets <= 1)
        ):
            raise ValueError(
                "integrity_targets must contain "
                "only 0 or 1."
            )

    @property
    def batch_size(self) -> int:
        return int(
            self.text_embeddings.shape[0]
        )

    def to(
        self,
        device,
    ) -> "BinaryIntegrityBatch":

        return BinaryIntegrityBatch(
            text_embeddings=(
                self.text_embeddings.to(device)
            ),

            vision_embeddings=(
                self.vision_embeddings.to(device)
            ),

            integrity_targets=(
                self.integrity_targets.to(device)
            ),

            sample_ids=self.sample_ids,
        )