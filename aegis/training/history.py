"""
AEGIS Experiment History.

Version: 0.18.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

import json
from pathlib import Path
from typing import Dict, List


@dataclass
class EpochRecord:
    """
    Metrics recorded for one training epoch.
    """

    epoch: int

    train_metrics: Dict[
        str,
        float
    ]

    validation_metrics: Dict[
        str,
        float
    ]

    improved: bool = False


@dataclass
class ExperimentHistory:
    """
    Complete training history for one AEGIS experiment.
    """

    experiment_name: str

    records: List[
        EpochRecord
    ] = field(
        default_factory=list
    )

    best_epoch: int = 0

    best_validation_loss: float = float(
        "inf"
    )

    stopped_early: bool = False

    def add(
        self,
        record: EpochRecord,
    ) -> None:

        self.records.append(
            record
        )

        validation_loss = (
            record
            .validation_metrics
            .get(
                "loss"
            )
        )

        if (
            validation_loss
            is not None
            and validation_loss
            < self.best_validation_loss
        ):

            self.best_validation_loss = float(
                validation_loss
            )

            self.best_epoch = (
                record.epoch
            )

    def as_dict(self):

        return {
            "experiment_name": (
                self.experiment_name
            ),

            "records": [
                asdict(record)
                for record
                in self.records
            ],

            "best_epoch": (
                self.best_epoch
            ),

            "best_validation_loss": (
                self.best_validation_loss
            ),

            "stopped_early": (
                self.stopped_early
            ),
        }

    def save(
        self,
        path,
    ) -> Path:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.as_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return path