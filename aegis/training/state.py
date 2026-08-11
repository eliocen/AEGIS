"""
AEGIS Training State.

Version: 0.16.0
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainingState:
    """
    Tracks the progress of an AEGIS experiment.
    """

    epoch: int = 0

    global_step: int = 0

    best_validation_loss: float = float(
        "inf"
    )

    metrics: Dict[
        str,
        float
    ] = field(
        default_factory=dict
    )