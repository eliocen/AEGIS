"""
AEGIS Ablation Statistical Aggregation.

Version: 0.21.0
"""

import math
from dataclasses import (
    asdict,
    dataclass,
)
from typing import (
    Iterable,
    List,
)


@dataclass
class MetricStatistics:
    """
    Descriptive statistics for one metric
    across repeated experimental seeds.
    """

    count: int

    mean: float

    standard_deviation: float

    minimum: float

    maximum: float

    def as_dict(self):

        return asdict(
            self
        )


def compute_metric_statistics(
    values: Iterable[
        float
    ],
) -> MetricStatistics:
    """
    Compute descriptive statistics for repeated
    experimental measurements.

    Standard deviation uses the sample standard
    deviation when at least two observations exist.
    """

    values: List[
        float
    ] = [
        float(value)
        for value in values
    ]

    if not values:
        raise ValueError(
            "At least one value is required "
            "to compute metric statistics."
        )

    count = len(
        values
    )

    mean = (
        sum(values)
        / count
    )

    if count > 1:

        variance = (
            sum(
                (
                    value
                    - mean
                ) ** 2
                for value in values
            )
            / (
                count - 1
            )
        )

        standard_deviation = (
            math.sqrt(
                variance
            )
        )

    else:

        standard_deviation = 0.0

    return MetricStatistics(
        count=count,

        mean=float(
            mean
        ),

        standard_deviation=float(
            standard_deviation
        ),

        minimum=float(
            min(values)
        ),

        maximum=float(
            max(values)
        ),
    )