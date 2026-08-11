"""
AEGIS Multi-Seed Ablation Aggregation.

Version: 0.21.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from collections import (
    defaultdict,
)

from typing import (
    Any,
    Dict,
    Iterable,
    List,
)

from .multiseed import (
    SeedRunResult,
)

from .statistics import (
    MetricStatistics,
    compute_metric_statistics,
)


@dataclass
class AggregatedAblationResult:
    """
    Aggregated metrics for one ablation variant
    across repeated seeds.
    """

    variant: str

    seeds: List[
        int
    ]

    metrics: Dict[
        str,
        MetricStatistics
    ]

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def as_dict(self):

        return {
            "variant": (
                self.variant
            ),

            "seeds": list(
                self.seeds
            ),

            "metrics": {
                metric: statistics.as_dict()
                for (
                    metric,
                    statistics,
                ) in self.metrics.items()
            },

            "metadata": dict(
                self.metadata
            ),
        }


def aggregate_seed_runs(
    results: Iterable[
        SeedRunResult
    ],
) -> List[
    AggregatedAblationResult
]:
    """
    Aggregate repeated seed runs by ablation variant.
    """

    results = list(
        results
    )

    if not results:
        raise ValueError(
            "No seed-run results supplied."
        )

    grouped = defaultdict(
        list
    )

    for result in results:

        if not isinstance(
            result,
            SeedRunResult,
        ):
            raise TypeError(
                "All entries must be "
                "SeedRunResult."
            )

        grouped[
            result.variant
        ].append(
            result
        )

    aggregated = []

    for (
        variant,
        variant_results,
    ) in grouped.items():

        seeds = [
            result.seed
            for result in variant_results
        ]

        if len(
            set(seeds)
        ) != len(seeds):

            raise ValueError(
                "Duplicate seeds detected for "
                f"variant: {variant}"
            )

        metric_names = set(
            variant_results[
                0
            ].metrics.keys()
        )

        for result in variant_results[1:]:

            if (
                set(
                    result.metrics.keys()
                )
                != metric_names
            ):
                raise ValueError(
                    "Metric sets must be identical "
                    "across repeated runs of "
                    f"variant: {variant}"
                )

        metric_statistics = {}

        for metric in sorted(
            metric_names
        ):

            values = [
                result.metrics[
                    metric
                ]
                for result
                in variant_results
            ]

            metric_statistics[
                metric
            ] = (
                compute_metric_statistics(
                    values
                )
            )

        aggregated.append(
            AggregatedAblationResult(
                variant=variant,

                seeds=sorted(
                    seeds
                ),

                metrics=(
                    metric_statistics
                ),
            )
        )

    return aggregated


def compare_aggregated_to_baseline(
    results: Iterable[
        AggregatedAblationResult
    ],
    baseline_name: str,
    metric: str,
):
    """
    Compare mean metric values across aggregated
    ablation variants.
    """

    results = list(
        results
    )

    result_map = {
        result.variant: result
        for result in results
    }

    if (
        baseline_name
        not in result_map
    ):
        raise ValueError(
            "Baseline was not found in "
            "aggregated results."
        )

    baseline = (
        result_map[
            baseline_name
        ]
    )

    if metric not in baseline.metrics:

        raise KeyError(
            f"Baseline does not contain "
            f"metric: {metric}"
        )

    baseline_mean = (
        baseline
        .metrics[
            metric
        ]
        .mean
    )

    comparisons = []

    for result in results:

        if metric not in result.metrics:

            raise KeyError(
                f"{result.variant} does not "
                f"contain metric: {metric}"
            )

        variant_mean = (
            result
            .metrics[
                metric
            ]
            .mean
        )

        absolute_delta = (
            variant_mean
            - baseline_mean
        )

        if baseline_mean != 0:

            relative_delta_percent = (
                absolute_delta
                / abs(
                    baseline_mean
                )
                * 100.0
            )

        else:

            relative_delta_percent = 0.0

        comparisons.append(
            {
                "variant": (
                    result.variant
                ),

                "metric": (
                    metric
                ),

                "mean": float(
                    variant_mean
                ),

                "standard_deviation": float(
                    result
                    .metrics[
                        metric
                    ]
                    .standard_deviation
                ),

                "baseline_mean": float(
                    baseline_mean
                ),

                "absolute_delta": float(
                    absolute_delta
                ),

                "relative_delta_percent": float(
                    relative_delta_percent
                ),
            }
        )

    return comparisons


def build_statistical_comparison_table(
    results: Iterable[
        AggregatedAblationResult
    ],
    baseline_name: str = "baseline",
    metrics=None,
):
    """
    Build a publication-oriented table containing
    mean, standard deviation and delta-to-baseline.
    """

    results = list(
        results
    )

    if not results:
        raise ValueError(
            "No aggregated results supplied."
        )

    result_map = {
        result.variant: result
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
                result.variant
            ),

            "seeds": list(
                result.seeds
            ),

            "run_count": len(
                result.seeds
            ),
        }

        for metric in metrics:

            if metric not in result.metrics:
                continue

            statistics = (
                result.metrics[
                    metric
                ]
            )

            row[
                f"{metric}_mean"
            ] = (
                statistics.mean
            )

            row[
                f"{metric}_std"
            ] = (
                statistics
                .standard_deviation
            )

            row[
                f"{metric}_min"
            ] = (
                statistics.minimum
            )

            row[
                f"{metric}_max"
            ] = (
                statistics.maximum
            )

            if (
                metric
                in baseline.metrics
            ):

                baseline_mean = (
                    baseline
                    .metrics[
                        metric
                    ]
                    .mean
                )

                row[
                    f"{metric}_delta"
                ] = (
                    statistics.mean
                    - baseline_mean
                )

        table.append(
            row
        )

    return table