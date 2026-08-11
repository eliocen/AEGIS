"""
AEGIS Training Checkpoint Utilities.

Version: 0.16.0
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path,
    alignment_model,
    classification_model,
    optimizer,
    epoch: int,
    global_step: int,
    metrics: Optional[
        Dict[str, float]
    ] = None,
    extra: Optional[
        Dict[str, Any]
    ] = None,
) -> Path:
    """
    Save a reproducible AEGIS training checkpoint.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "epoch": int(epoch),

        "global_step": int(
            global_step
        ),

        "alignment_model": (
            alignment_model
            .state_dict()
        ),

        "classification_model": (
            classification_model
            .state_dict()
        ),

        "optimizer": (
            optimizer
            .state_dict()
        ),

        "metrics": (
            metrics or {}
        ),

        "extra": (
            extra or {}
        ),
    }

    torch.save(
        payload,
        path,
    )

    return path


def load_checkpoint(
    path,
    alignment_model,
    classification_model,
    optimizer=None,
    map_location="cpu",
):
    """
    Restore an AEGIS training checkpoint.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    alignment_model.load_state_dict(
        checkpoint[
            "alignment_model"
        ]
    )

    classification_model.load_state_dict(
        checkpoint[
            "classification_model"
        ]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint[
                "optimizer"
            ]
        )

    return checkpoint