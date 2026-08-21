"""
AEGIS Deterministic Fakeddit Sampling
=====================================

Version: 0.24.0

Provides reproducible class-stratified sample selection for empirical
Fakeddit experiments.

Two sampling strategies are provided:

1. In-memory deterministic stratified sampling
2. Streaming deterministic reservoir sampling

The streaming strategy is intended for large empirical splits such as
the 564,000-sample Fakeddit training partition.

Sampling uses only the native Fakeddit 2-way label.
"""

from __future__ import annotations

import random

from collections import Counter
from dataclasses import dataclass

from typing import (
    Dict,
    Iterable,
    List,
)

from aegis.data.adapters.empirical import (
    EmpiricalSample,
)


@dataclass(frozen=True)
class FakedditSampleSelection:
    """
    Reproducible selected empirical subset.
    """

    samples: List[EmpiricalSample]

    requested_count: int
    actual_count: int

    seed: int

    native_label_counts: Dict[
        int,
        int,
    ]

    strategy: str

    population_seen: int = 0

    @property
    def sample_ids(
        self,
    ) -> List[str]:

        return [
            sample.sample_id
            for sample in self.samples
        ]


def _native_binary_label(
    sample: EmpiricalSample,
) -> int:
    """
    Extract and validate one native Fakeddit 2-way label.
    """

    if not isinstance(
        sample,
        EmpiricalSample,
    ):
        raise TypeError(
            "Expected EmpiricalSample."
        )

    if (
        "2_way_label"
        not in sample.native_labels
    ):
        raise KeyError(
            f"Sample {sample.sample_id} "
            "does not contain 2_way_label."
        )

    value = sample.native_labels[
        "2_way_label"
    ]

    if value is None:
        raise ValueError(
            f"Sample {sample.sample_id} "
            "has missing 2_way_label."
        )

    label = int(
        value
    )

    if label not in {
        0,
        1,
    }:
        raise ValueError(
            f"Invalid Fakeddit 2_way_label "
            f"{label} for sample "
            f"{sample.sample_id}."
        )

    return label


def _target_counts(
    sample_count: int,
) -> Dict[int, int]:
    """
    Calculate approximately balanced binary targets.
    """

    sample_count = int(
        sample_count
    )

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be greater than zero."
        )

    return {
        0: sample_count // 2,

        1: (
            sample_count
            - sample_count // 2
        ),
    }


def select_stratified_fakeddit_samples(
    samples: Iterable[
        EmpiricalSample
    ],

    sample_count: int,

    seed: int = 42,

) -> FakedditSampleSelection:
    """
    Select a deterministic approximately balanced binary subset.

    This implementation collects the supplied population in memory.

    For large Fakeddit splits, prefer:

        select_streaming_stratified_fakeddit_samples()
    """

    sample_count = int(
        sample_count
    )

    seed = int(
        seed
    )

    targets = _target_counts(
        sample_count
    )

    buckets = {
        0: [],
        1: [],
    }

    seen_ids = set()

    population_seen = 0

    for sample in samples:

        population_seen += 1

        if sample.sample_id in seen_ids:
            raise ValueError(
                "Duplicate sample ID encountered "
                f"during sampling: "
                f"{sample.sample_id}"
            )

        seen_ids.add(
            sample.sample_id
        )

        label = (
            _native_binary_label(
                sample
            )
        )

        buckets[
            label
        ].append(
            sample
        )

    for label, required in (
        targets.items()
    ):

        available = len(
            buckets[
                label
            ]
        )

        if available < required:
            raise ValueError(
                f"Not enough samples for "
                f"native label {label}: "
                f"required {required}, "
                f"available {available}."
            )

    rng = random.Random(
        seed
    )

    selected = []

    for label in (
        0,
        1,
    ):

        candidates = list(
            buckets[
                label
            ]
        )

        rng.shuffle(
            candidates
        )

        selected.extend(
            candidates[
                : targets[
                    label
                ]
            ]
        )

    rng.shuffle(
        selected
    )

    counts = Counter(
        _native_binary_label(
            sample
        )
        for sample in selected
    )

    return (
        FakedditSampleSelection(
            samples=selected,

            requested_count=(
                sample_count
            ),

            actual_count=len(
                selected
            ),

            seed=seed,

            native_label_counts={
                int(label): int(
                    count
                )
                for label, count
                in sorted(
                    counts.items()
                )
            },

            strategy=(
                "stratified_binary"
            ),

            population_seen=(
                population_seen
            ),
        )
    )


def select_streaming_stratified_fakeddit_samples(
    samples: Iterable[
        EmpiricalSample
    ],

    sample_count: int,

    seed: int = 42,

) -> FakedditSampleSelection:
    """
    Deterministic class-stratified reservoir sampling.

    Memory complexity is O(sample_count), rather than O(dataset_size).

    Separate reservoirs are maintained for Fakeddit native labels
    0 and 1.

    For a fixed:
        - source sample order
        - sample_count
        - seed

    the result is deterministic.
    """

    sample_count = int(
        sample_count
    )

    seed = int(
        seed
    )

    targets = _target_counts(
        sample_count
    )

    reservoirs = {
        0: [],
        1: [],
    }

    class_seen = {
        0: 0,
        1: 0,
    }

    population_seen = 0

    seen_ids = set()

    rng = random.Random(
        seed
    )

    for sample in samples:

        population_seen += 1

        if sample.sample_id in seen_ids:
            raise ValueError(
                "Duplicate sample ID encountered "
                f"during streaming sampling: "
                f"{sample.sample_id}"
            )

        seen_ids.add(
            sample.sample_id
        )

        label = (
            _native_binary_label(
                sample
            )
        )

        class_seen[
            label
        ] += 1

        required = targets[
            label
        ]

        if required == 0:
            continue

        reservoir = reservoirs[
            label
        ]

        if len(
            reservoir
        ) < required:

            reservoir.append(
                sample
            )

            continue

        # Reservoir sampling:
        #
        # Given the n-th item in a class stream,
        # replace one existing reservoir item
        # with probability required / n.

        candidate_index = (
            rng.randrange(
                class_seen[
                    label
                ]
            )
        )

        if candidate_index < required:

            reservoir[
                candidate_index
            ] = sample

    for label, required in (
        targets.items()
    ):

        available = class_seen[
            label
        ]

        if available < required:

            raise ValueError(
                f"Not enough samples for "
                f"native label {label}: "
                f"required {required}, "
                f"available {available}."
            )

    selected = (
        list(
            reservoirs[0]
        )
        +
        list(
            reservoirs[1]
        )
    )

    # Final deterministic shuffle so the returned
    # subset is not grouped by class.

    rng.shuffle(
        selected
    )

    counts = Counter(
        _native_binary_label(
            sample
        )
        for sample in selected
    )

    return (
        FakedditSampleSelection(
            samples=selected,

            requested_count=(
                sample_count
            ),

            actual_count=len(
                selected
            ),

            seed=seed,

            native_label_counts={
                int(label): int(
                    count
                )
                for label, count
                in sorted(
                    counts.items()
                )
            },

            strategy=(
                "streaming_stratified_reservoir"
            ),

            population_seen=(
                population_seen
            ),
        )
    )