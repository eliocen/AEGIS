"""
AEGIS Ablation Result Models.

Version: 0.21.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from typing import Any, Dict

from .spec import (
    AblationSpec,
)


@dataclass
class AblationResult:
    """
    Result from one controlled ablation run.
    """

    spec: AblationSpec

    metrics: Dict[
        str,
        float
    ]

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def as_dict(self):

        return {
            "spec": (
                self.spec.as_dict()
            ),

            "metrics": dict(
                self.metrics
            ),

            "metadata": dict(
                self.metadata
            ),
        }