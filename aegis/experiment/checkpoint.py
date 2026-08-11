"""
AEGIS Experiment Checkpoint Manager.

Version: 0.19.0
"""

from pathlib import Path

import torch


def save_experiment_checkpoint(
    path,
    alignment_model,
    classification_model,
    optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    early_stopping_state,
    experiment_config,
    checkpoint_type: str,
    metrics=None,
):
    """
    Save all state required to resume an AEGIS
    experiment.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "checkpoint_format": (
            "aegis_experiment_v1"
        ),

        "checkpoint_type": (
            checkpoint_type
        ),

        "epoch": int(
            epoch
        ),

        "global_step": int(
            global_step
        ),

        "best_validation_loss": float(
            best_validation_loss
        ),

        "alignment_model": (
            alignment_model.state_dict()
        ),

        "classification_model": (
            classification_model.state_dict()
        ),

        "optimizer": (
            optimizer.state_dict()
        ),

        "early_stopping": {
            "patience": (
                early_stopping_state.patience
            ),

            "min_delta": (
                early_stopping_state.min_delta
            ),

            "best_loss": (
                early_stopping_state.best_loss
            ),

            "bad_epochs": (
                early_stopping_state.bad_epochs
            ),

            "stopped": (
                early_stopping_state.stopped
            ),
        },

        "experiment_config": (
            experiment_config.as_dict()
            if hasattr(
                experiment_config,
                "as_dict"
            )
            else dict(
                experiment_config
            )
        ),

        "metrics": (
            metrics or {}
        ),
    }

    torch.save(
        payload,
        path,
    )

    return path


def load_experiment_checkpoint(
    path,
    map_location="cpu",
):
    """
    Load an AEGIS experiment checkpoint payload.
    """

    path = Path(
        path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=(
            map_location
        ),
        weights_only=False,
    )

    if (
        checkpoint.get(
            "checkpoint_format"
        )
        != "aegis_experiment_v1"
    ):
        raise ValueError(
            "Unsupported AEGIS experiment "
            "checkpoint format."
        )

    return checkpoint