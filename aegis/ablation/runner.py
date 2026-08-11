"""
AEGIS Ablation Study Runner.

Version: 0.21.0
"""

from typing import (
    Callable,
    Iterable,
    List,
)

from .result import (
    AblationResult,
)

from .spec import (
    AblationSpec,
)


class AblationStudyRunner:
    """
    Executes a controlled collection of
    AEGIS ablation experiments.

    experiment_fn must accept one AblationSpec
    and return a dictionary of scalar metrics.
    """

    def __init__(
        self,
        experiment_fn: Callable,
    ):

        if not callable(
            experiment_fn
        ):
            raise TypeError(
                "experiment_fn must be callable."
            )

        self.experiment_fn = (
            experiment_fn
        )

    def run(
        self,
        specs: Iterable[
            AblationSpec
        ],
    ) -> List[
        AblationResult
    ]:

        specs = list(
            specs
        )

        if not specs:
            raise ValueError(
                "At least one ablation "
                "specification is required."
            )

        names = [
            spec.name
            for spec in specs
        ]

        if len(
            set(names)
        ) != len(names):

            raise ValueError(
                "Ablation names must be unique."
            )

        results = []

        for spec in specs:

            if not isinstance(
                spec,
                AblationSpec,
            ):
                raise TypeError(
                    "All study entries must "
                    "be AblationSpec."
                )

            metrics = (
                self.experiment_fn(
                    spec
                )
            )

            if not isinstance(
                metrics,
                dict,
            ):
                raise TypeError(
                    "experiment_fn must "
                    "return a metrics dictionary."
                )

            normalized_metrics = {}

            for (
                key,
                value,
            ) in metrics.items():

                if not isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                ):
                    raise TypeError(
                        "Ablation metrics must "
                        "be scalar numeric values."
                    )

                normalized_metrics[
                    str(key)
                ] = float(
                    value
                )

            results.append(
                AblationResult(
                    spec=spec,

                    metrics=(
                        normalized_metrics
                    ),
                )
            )

        return results