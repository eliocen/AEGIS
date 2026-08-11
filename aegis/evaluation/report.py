"""
AEGIS Evaluation Report.

Version: 0.20.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

import json
from pathlib import Path

from typing import Any, Dict, List

from .errors import (
    EvaluationError,
)


@dataclass
class EvaluationReport:
    """
    Reproducible AEGIS research evaluation report.
    """

    experiment_name: str

    total_samples: int

    metrics: Dict[
        str,
        Any
    ]

    stratified_metrics: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    errors: List[
        EvaluationError
    ] = field(
        default_factory=list
    )

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def as_dict(self):

        return {
            "experiment_name": (
                self.experiment_name
            ),

            "total_samples": (
                self.total_samples
            ),

            "metrics": (
                self.metrics
            ),

            "stratified_metrics": (
                self.stratified_metrics
            ),

            "errors": [
                error.as_dict()
                for error
                in self.errors
            ],

            "metadata": (
                self.metadata
            ),
        }

    def save_json(
        self,
        path,
    ):

        path = Path(
            path
        )

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