"""
AEGIS Weighted Hierarchical Classification Loss.

Version: 0.10.0
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class WeightedHierarchicalLoss(
    nn.Module
):
    """
    Joint hierarchical objective.

    Total loss:

        L = alpha * L_integrity
            + beta * L_threat

    Threat loss is calculated only for samples
    labelled as harmful.
    """

    def __init__(
        self,
        integrity_weight: float = 1.0,
        threat_weight: float = 1.0,
        threat_class_weights: Optional[
            torch.Tensor
        ] = None,
    ):
        super().__init__()

        if integrity_weight < 0:
            raise ValueError(
                "integrity_weight must be >= 0."
            )

        if threat_weight < 0:
            raise ValueError(
                "threat_weight must be >= 0."
            )

        self.integrity_weight = (
            integrity_weight
        )

        self.threat_weight = (
            threat_weight
        )

        if threat_class_weights is not None:
            self.register_buffer(
                "threat_class_weights",
                threat_class_weights.float(),
            )
        else:
            self.threat_class_weights = None

    def forward(
        self,
        integrity_logits,
        threat_logits,
        integrity_targets,
        threat_targets,
    ):
        """
        integrity_targets:
            0 = True
            1 = Harmful

        threat_targets:
            0 = Misinformation
            1 = Disinformation
            2 = Malinformation
            3 = Hate Speech

        For True samples, threat_targets may be -1.
        """

        integrity_loss = F.cross_entropy(
            integrity_logits,
            integrity_targets,
        )

        harmful_mask = (
            integrity_targets == 1
        )

        harmful_count = int(
            harmful_mask.sum().item()
        )

        if harmful_count > 0:

            harmful_logits = (
                threat_logits[
                    harmful_mask
                ]
            )

            harmful_targets = (
                threat_targets[
                    harmful_mask
                ]
            )

            threat_loss = F.cross_entropy(
                harmful_logits,
                harmful_targets,
                weight=(
                    self.threat_class_weights
                ),
            )

        else:
            # Keep graph/device compatibility
            # while contributing zero threat loss.
            threat_loss = (
                threat_logits.sum()
                * 0.0
            )

        total_loss = (
            self.integrity_weight
            * integrity_loss
            +
            self.threat_weight
            * threat_loss
        )

        return {
            "loss": total_loss,
            "integrity_loss": (
                integrity_loss
            ),
            "threat_loss": (
                threat_loss
            ),
            "harmful_count": (
                harmful_count
            ),
        }