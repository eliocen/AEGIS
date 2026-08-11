"""
AEGIS Multilingual Evaluation Result Models.

Version: 0.22.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from typing import (
    Any,
    Dict,
)


@dataclass
class LanguageEvaluationResult:
    """
    Evaluation results for one language subset.
    """

    language: str

    sample_count: int

    metrics: Dict[
        str,
        Any
    ]

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def as_dict(self):

        return {
            "language": (
                self.language
            ),

            "sample_count": (
                self.sample_count
            ),

            "metrics": (
                self.metrics
            ),

            "metadata": (
                self.metadata
            ),
        }


@dataclass
class CrossLingualRunResult:
    """
    One train-language/test-language evaluation.
    """

    train_language: str

    test_language: str

    metric_name: str

    metric_value: float

    seed: int | None = None

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def as_dict(self):

        return asdict(
            self
        )