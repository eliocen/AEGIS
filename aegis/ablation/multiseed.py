"""
AEGIS Multi-Seed Ablation Execution.

Version: 0.21.0
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
)

from .spec import (
    AblationSpec,
)


@dataclass
class SeedRunResult:
    """
    Result for one ablation variant under one seed.
    """

    variant: str

    seed: int

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

        return asdict(
            self
        )


class MultiSeedAblationRunner:
    """
    Execute every AblationSpec across the same
    controlled collection of random seeds.

    experiment_fn signature:

        experiment_fn(spec, seed) -> dict
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
        seeds: Iterable[
            int
        ],
    ) -> List[
        SeedRunResult
    ]:

        specs = list(
            specs
        )

        seeds = [
            int(seed)
            for seed in seeds
        ]

        if not specs:
            raise ValueError(
                "At least one ablation "
                "specification is required."
            )

        if not seeds:
            raise ValueError(
                "At least one seed is required."
            )

        if len(
            set(seeds)
        ) != len(seeds):

            raise ValueError(
                "Seeds must be unique."
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

            for seed in seeds:

                if seed < 0:
                    raise ValueError(
                        "Seeds must be "
                        "non-negative."
                    )

                metrics = (
                    self.experiment_fn(
                        spec,
                        seed,
                    )
                )

                if not isinstance(
                    metrics,
                    dict,
                ):
                    raise TypeError(
                        "experiment_fn must return "
                        "a metrics dictionary."
                    )

                normalized = {}

                for (
                    metric,
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
                            "Multi-seed metrics must "
                            "be numeric scalars."
                        )

                    normalized[
                        str(metric)
                    ] = float(
                        value
                    )

                results.append(
                    SeedRunResult(
                        variant=(
                            spec.name
                        ),

                        seed=seed,

                        metrics=(
                            normalized
                        ),
                    )
                )

        return results