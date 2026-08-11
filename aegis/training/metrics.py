"""
AEGIS Training Metrics.

Version: 0.16.0
"""

from typing import Dict

import torch


def compute_hierarchical_metrics(
    integrity_logits: torch.Tensor,
    threat_logits: torch.Tensor,
    integrity_targets: torch.Tensor,
    threat_targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute basic hierarchical classification metrics.
    """

    integrity_predictions = (
        integrity_logits.argmax(
            dim=-1
        )
    )

    integrity_accuracy = (
        (
            integrity_predictions
            == integrity_targets
        )
        .float()
        .mean()
        .item()
    )

    harmful_mask = (
        integrity_targets == 1
    )

    harmful_count = int(
        harmful_mask.sum().item()
    )

    if harmful_count > 0:

        threat_predictions = (
            threat_logits[
                harmful_mask
            ]
            .argmax(
                dim=-1
            )
        )

        harmful_targets = (
            threat_targets[
                harmful_mask
            ]
        )

        threat_accuracy = (
            (
                threat_predictions
                == harmful_targets
            )
            .float()
            .mean()
            .item()
        )

    else:
        threat_accuracy = 0.0

    return {
        "integrity_accuracy": (
            float(
                integrity_accuracy
            )
        ),

        "threat_accuracy": (
            float(
                threat_accuracy
            )
        ),

        "harmful_count": (
            float(
                harmful_count
            )
        ),
    }