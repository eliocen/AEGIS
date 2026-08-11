"""
AEGIS Epoch Training and Validation Utilities.

Version: 0.18.0
"""

from collections import defaultdict
from typing import Dict

import torch

from .trainer import (
    AEGISTrainer,
)


def _average_metrics(
    totals,
    batches: int,
) -> Dict[str, float]:
    """
    Convert accumulated metric totals into
    epoch averages.
    """

    if batches <= 0:
        raise ValueError(
            "Cannot average metrics over zero batches."
        )

    return {
        key: float(
            value / batches
        )
        for key, value
        in totals.items()
    }


def run_training_epoch(
    trainer: AEGISTrainer,
    dataloader,
) -> Dict[str, float]:
    """
    Train AEGIS for one complete epoch.
    """

    totals = defaultdict(
        float
    )

    batch_count = 0

    for batch in dataloader:

        metrics = trainer.train_step(
            batch
        )

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float),
            ):
                totals[key] += float(
                    value
                )

        batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "Training DataLoader produced no batches."
        )

    return _average_metrics(
        totals,
        batch_count,
    )


def run_validation_epoch(
    trainer: AEGISTrainer,
    dataloader,
) -> Dict[str, float]:
    """
    Evaluate AEGIS for one complete validation epoch.

    No gradients or optimizer updates are performed.
    """

    trainer.alignment_model.eval()
    trainer.classification_model.eval()

    totals = defaultdict(
        float
    )

    batch_count = 0

    with torch.no_grad():

        for batch in dataloader:

            outputs = trainer.forward_batch(
                batch,
                compute_alignment_loss=True,
            )

            metrics = {
                "loss": float(
                    outputs[
                        "loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),

                "alignment_loss": float(
                    outputs[
                        "alignment_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),

                "classification_loss": float(
                    outputs[
                        "classification_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),

                "integrity_loss": float(
                    outputs[
                        "integrity_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),

                "threat_loss": float(
                    outputs[
                        "threat_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                ),
            }

            metrics.update(
                outputs[
                    "metrics"
                ]
            )

            for (
                key,
                value,
            ) in metrics.items():

                if isinstance(
                    value,
                    (int, float),
                ):
                    totals[key] += float(
                        value
                    )

            batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "Validation DataLoader produced no batches."
        )

    return _average_metrics(
        totals,
        batch_count,
    )