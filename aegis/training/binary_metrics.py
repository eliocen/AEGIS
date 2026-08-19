"""
AEGIS Binary Integrity Training Metrics.

Version: 0.24.0
"""

from typing import Dict

import torch


def compute_binary_integrity_metrics(
    integrity_logits: torch.Tensor,
    integrity_targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute lightweight batch-level binary metrics.

    Full dataset-level precision, recall, F1, confusion matrices,
    calibration and uncertainty are handled by the evaluation layer.
    """

    predictions = torch.argmax(
        integrity_logits,
        dim=-1,
    )

    correct = (
        predictions
        == integrity_targets
    ).sum()

    total = integrity_targets.numel()

    accuracy = (
        float(correct.item()) / float(total)
        if total > 0
        else 0.0
    )

    return {
        "integrity_accuracy": accuracy,
    }