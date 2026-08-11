"""
AEGIS Robustness Evaluation Report.

Version: 0.23.0
"""

import json

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

from typing import (
    Any,
    Dict,
)


@dataclass
class RobustnessEvaluationReport:
    """
    Serializable calibration, uncertainty and
    robustness evaluation report.
    """

    experiment_name: str

    calibration: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    uncertainty: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    robustness: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    confidence_analysis: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
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

            "calibration": (
                self.calibration
            ),

            "uncertainty": (
                self.uncertainty
            ),

            "robustness": (
                self.robustness
            ),

            "confidence_analysis": (
                self.confidence_analysis
            ),

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