"""
AEGIS Binary Integrity Classification Loss.

Version: 0.24.0

Provides Stage-1-only supervision for datasets that do not contain
valid AEGIS threat subtype annotations.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class BinaryIntegrityLoss(nn.Module):
    """
    Cross-entropy loss over the AEGIS integrity head.

    Targets:
        0 = True
        1 = Harmful / non-authentic

    This loss never accesses or infers threat subtype labels.
    """

    def __init__(
        self,
        class_weights: Optional[
            torch.Tensor
        ] = None,
    ):
        super().__init__()

        if class_weights is not None:
            if class_weights.numel() != 2:
                raise ValueError(
                    "Binary integrity class weights "
                    "must contain exactly two values."
                )

            self.register_buffer(
                "class_weights",
                class_weights.float(),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        integrity_logits,
        integrity_targets,
    ):
        if (
            integrity_logits.ndim != 2
            or integrity_logits.shape[-1] != 2
        ):
            raise ValueError(
                "integrity_logits must have shape "
                "[batch_size, 2]."
            )

        if integrity_targets.ndim != 1:
            raise ValueError(
                "integrity_targets must have shape "
                "[batch_size]."
            )

        if (
            integrity_targets.shape[0]
            != integrity_logits.shape[0]
        ):
            raise ValueError(
                "Logit and target batch sizes "
                "must match."
            )

        loss = F.cross_entropy(
            integrity_logits,
            integrity_targets,
            weight=self.class_weights,
        )

        return {
            "loss": loss,
            "integrity_loss": loss,
        }