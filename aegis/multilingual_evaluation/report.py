"""
AEGIS Multilingual Evaluation Report.

Version: 0.22.0
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
class MultilingualEvaluationReport:
    """
    Serializable multilingual and cross-lingual
    research evaluation report.
    """

    experiment_name: str

    language_results: Dict[
        str,
        Any
    ]

    language_gap: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    transfer_matrix: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    transfer_summary: Dict[
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

            "language_results": {
                language: (
                    result.as_dict()
                    if hasattr(
                        result,
                        "as_dict"
                    )
                    else result
                )
                for (
                    language,
                    result,
                ) in self.language_results.items()
            },

            "language_gap": (
                self.language_gap
            ),

            "transfer_matrix": (
                self.transfer_matrix
            ),

            "transfer_summary": (
                self.transfer_summary
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