"""
AEGIS Experiment Resume Utilities.

Version: 0.19.0
"""

from dataclasses import (
    dataclass,
)

from aegis.training import (
    EarlyStoppingState,
)

from .checkpoint import (
    load_experiment_checkpoint,
)


@dataclass
class ResumeState:
    """
    Restored AEGIS experiment state.
    """

    epoch: int

    global_step: int

    best_validation_loss: float

    checkpoint_type: str

    metrics: dict


def resume_experiment(
    path,
    trainer,
    early_stopping_state=None,
    map_location="cpu",
):
    """
    Restore model, optimizer, trainer state, and
    early-stopping state from an AEGIS checkpoint.
    """

    checkpoint = (
        load_experiment_checkpoint(
            path,
            map_location=(
                map_location
            ),
        )
    )

    trainer.alignment_model.load_state_dict(
        checkpoint[
            "alignment_model"
        ]
    )

    trainer.classification_model.load_state_dict(
        checkpoint[
            "classification_model"
        ]
    )

    trainer.optimizer.load_state_dict(
        checkpoint[
            "optimizer"
        ]
    )

    trainer.state.epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    trainer.state.global_step = int(
        checkpoint[
            "global_step"
        ]
    )

    trainer.state.best_validation_loss = float(
        checkpoint[
            "best_validation_loss"
        ]
    )

    saved_early = checkpoint[
        "early_stopping"
    ]

    if early_stopping_state is None:

        early_stopping_state = (
            EarlyStoppingState(
                patience=(
                    saved_early[
                        "patience"
                    ]
                ),

                min_delta=(
                    saved_early[
                        "min_delta"
                    ]
                ),
            )
        )

    early_stopping_state.best_loss = float(
        saved_early[
            "best_loss"
        ]
    )

    early_stopping_state.bad_epochs = int(
        saved_early[
            "bad_epochs"
        ]
    )

    early_stopping_state.stopped = bool(
        saved_early[
            "stopped"
        ]
    )

    return (
        ResumeState(
            epoch=(
                trainer.state.epoch
            ),

            global_step=(
                trainer.state.global_step
            ),

            best_validation_loss=(
                trainer.state
                .best_validation_loss
            ),

            checkpoint_type=(
                checkpoint[
                    "checkpoint_type"
                ]
            ),

            metrics=(
                checkpoint.get(
                    "metrics",
                    {},
                )
            ),
        ),

        early_stopping_state,
    )