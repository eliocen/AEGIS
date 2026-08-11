"""
AEGIS Experiment Configuration.

Version: 0.19.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """
    Reproducible configuration snapshot for one
    AEGIS research experiment.
    """

    experiment_name: str

    framework_version: str = "0.19.0"

    seed: int = 42

    max_epochs: int = 20

    batch_size: int = 16

    learning_rate: float = 0.0001

    weight_decay: float = 0.0001

    alignment_loss_weight: float = 1.0

    classification_loss_weight: float = 1.0

    patience: int = 5

    min_delta: float = 0.0001

    device: str = "cpu"

    model: Dict[
        str,
        Any
    ] = field(
        default_factory=lambda: {
            "text_dimension": 768,
            "vision_dimension": 512,
            "shared_dimension": 512,
            "classifier_hidden_dimension": 256,
        }
    )

    extra: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def __post_init__(self):

        if not self.experiment_name:
            raise ValueError(
                "experiment_name cannot be empty."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )

        if self.max_epochs < 1:
            raise ValueError(
                "max_epochs must be at least 1."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.patience < 1:
            raise ValueError(
                "patience must be at least 1."
            )

    def as_dict(self):

        return asdict(
            self
        )