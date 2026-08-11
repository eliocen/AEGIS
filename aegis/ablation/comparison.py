"""
AEGIS Ablation Comparison Utilities.

Version: 0.21.0
"""

from dataclasses import (
    asdict,
    dataclass,
)

from typing import Dict, Iterable, List

from .result import (
    AblationResult,
)


@dataclass
class MetricComparison:
    """
    Comparison between one ablation and baseline.
    """

    variant: str

    metric: str

    baseline_value: float

    variant_value: float

    absolute_delta: float

    relative_delta_percent: float

    def as_dict(self):

        return asdict(
            self
        )


def compare_to_baseline(
    baseline: AblationResult,
    variant: AblationResult,
    metric: str,
) -> MetricComparison:
    """
    Compare one ablation against the baseline.
    """

    if metric not in baseline.metrics:

        raise KeyError(
            f"Baseline does not contain "
            f"metric: {metric}"
        )

    if metric not in variant.metrics:

        raise KeyError(
            f"Variant does not contain "
            f"metric: {metric}"
        )

    baseline_value = float(
        baseline.metrics[
            metric
        ]
    )

    variant_value = float(
        variant.metrics[
            metric
        ]
    )

    delta = (
        variant_value
        - baseline_value
    )

    if baseline_value != 0:

        relative_delta = (
            delta
            / abs(
                baseline_value
            )
            * 100.0
        )

    else:

        relative_delta = 0.0

    return MetricComparison(
        variant=(
            variant.spec.name
        ),

        metric=metric,

        baseline_value=(
            baseline_value
        ),

        variant_value=(
            variant_value
        ),

        absolute_delta=float(
            delta
        ),

        relative_delta_percent=float(
            relative_delta
        ),
    )


def rank_results(
    results: Iterable[
        AblationResult
    ],
    metric: str,
    higher_is_better: bool = True,
) -> List[AblationResult]:
    """
    Rank ablation results by one metric.
    """

    results = list(
        results
    )

    for result in results:

        if metric not in result.metrics:

            raise KeyError(
                f"{result.spec.name} does "
                f"not contain metric: {metric}"
            )

    return sorted(
        results,

        key=lambda result: (
            result.metrics[
                metric
            ]
        ),

        reverse=(
            higher_is_better
        ),
    )


def build_comparison_table(
    results: Iterable[
        AblationResult
    ],
    baseline_name: str = "baseline",
    metrics=None,
):
    """
    Build a serializable research comparison table.
    """

    results = list(
        results
    )

    if not results:
        raise ValueError(
            "No ablation results were supplied."
        )

    result_map = {
        result.spec.name: result
        for result in results
    }

    if (
        baseline_name
        not in result_map
    ):
        raise ValueError(
            "Baseline result was not found."
        )

    baseline = (
        result_map[
            baseline_name
        ]
    )

    if metrics is None:

        metrics = list(
            baseline.metrics.keys()
        )

    table = []

    for result in results:

        row = {
            "variant": (
                result.spec.name
            )
        }

        for metric in metrics:

            if metric not in result.metrics:

                continue

            value = float(
                result.metrics[
                    metric
                ]
            )

            row[
                metric
            ] = value

            if (
                result.spec.name
                != baseline_name
                and metric
                in baseline.metrics
            ):

                comparison = (
                    compare_to_baseline(
                        baseline,
                        result,
                        metric,
                    )
                )

                row[
                    f"{metric}_delta"
                ] = (
                    comparison
                    .absolute_delta
                )

        table.append(
            row
        )

    return table