"""
AEGIS Early Stopping.

Version: 0.18.0
"""

from dataclasses import dataclass


@dataclass
class EarlyStoppingState:
    """
    Tracks validation-loss improvement across epochs.
    """

    patience: int = 5
    min_delta: float = 0.0

    best_loss: float = float("inf")
    bad_epochs: int = 0

    stopped: bool = False

    def __post_init__(self):

        if self.patience < 1:
            raise ValueError(
                "patience must be at least 1."
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative."
            )

    def update(
        self,
        validation_loss: float,
    ) -> bool:
        """
        Update early-stopping state.

        Returns True when validation loss improved.
        """

        validation_loss = float(
            validation_loss
        )

        improved = (
            validation_loss
            < (
                self.best_loss
                - self.min_delta
            )
        )

        if improved:

            self.best_loss = (
                validation_loss
            )

            self.bad_epochs = 0

        else:

            self.bad_epochs += 1

            if (
                self.bad_epochs
                >= self.patience
            ):
                self.stopped = True

        return improved