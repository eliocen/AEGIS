"""
AEGIS Training Batch Models.

Version: 0.16.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TrainingBatch:
    """
    One multimodal training batch for the AEGIS
    alignment and hierarchical classification models.

    Expected dimensions:
        text_embeddings   -> [B, 768]
        vision_embeddings -> [B, 512]
        integrity_targets -> [B]
        threat_targets    -> [B]

    Threat targets use:
        -1 for TRUE information
         0 misinformation
         1 disinformation
         2 malinformation
         3 hate speech
    """

    text_embeddings: torch.Tensor
    vision_embeddings: torch.Tensor

    integrity_targets: torch.Tensor
    threat_targets: torch.Tensor

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

        if (
            self.threat_targets.ndim != 1
            or self.threat_targets.shape[0]
            != batch_size
        ):
            raise ValueError(
                "threat_targets must have shape "
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

        harmful_mask = (
            self.integrity_targets == 1
        )

        true_mask = (
            self.integrity_targets == 0
        )

        if harmful_mask.any():

            harmful_targets = (
                self.threat_targets[
                    harmful_mask
                ]
            )

            if not torch.all(
                (harmful_targets >= 0)
                & (harmful_targets <= 3)
            ):
                raise ValueError(
                    "Harmful samples must have threat "
                    "targets in the range 0..3."
                )

        if true_mask.any():

            true_targets = (
                self.threat_targets[
                    true_mask
                ]
            )

            if not torch.all(
                true_targets == -1
            ):
                raise ValueError(
                    "TRUE samples must use -1 as "
                    "their threat target."
                )

    @property
    def batch_size(self) -> int:
        return int(
            self.text_embeddings.shape[0]
        )

    def to(
        self,
        device,
    ) -> "TrainingBatch":

        return TrainingBatch(
            text_embeddings=(
                self.text_embeddings.to(device)
            ),

            vision_embeddings=(
                self.vision_embeddings.to(device)
            ),

            integrity_targets=(
                self.integrity_targets.to(device)
            ),

            threat_targets=(
                self.threat_targets.to(device)
            ),

            sample_ids=(
                self.sample_ids
            ),
        )