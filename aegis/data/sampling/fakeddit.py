"""
AEGIS Deterministic Fakeddit Sampling
=====================================

Version: 0.24.0

Provides reproducible class-stratified sampling for empirical Fakeddit
experiments.

Supported modes
---------------
1. In-memory stratified sampling.
2. Streaming stratified reservoir sampling.
3. Streaming stratified reservoir sampling with image-decodability
   eligibility filtering and deterministic same-class replacement.

The decodability-aware sampler is intended for empirical cache
generation.

Important
---------
Sampling operates on Fakeddit's native 2-way labels.

No Fakeddit label is interpreted as an AEGIS threat subtype.
"""

from __future__ import annotations

import random

from collections import Counter
from dataclasses import dataclass, field

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from aegis.data.adapters.empirical import (
    EmpiricalSample,
)

from aegis.data.image_validation import (
    validate_decodable_image,
)


@dataclass(frozen=True)
class FakedditSampleExclusion:
    """
    One sample rejected during empirical eligibility validation.
    """

    sample_id: str
    native_label: int
    image_path: Optional[str]

    reason: str
    message: Optional[str]

    replacement_required: bool = True

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "native_label": self.native_label,
            "image_path": self.image_path,
            "reason": self.reason,
            "message": self.message,
            "replacement_required": (
                self.replacement_required
            ),
        }


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

    candidate_count: int = 0

    valid_candidate_count: int = 0

    exclusions: Tuple[
        FakedditSampleExclusion,
        ...,
    ] = field(
        default_factory=tuple
    )

    @property
    def sample_ids(
        self,
    ) -> List[str]:
        return [
            sample.sample_id
            for sample in self.samples
        ]

    @property
    def exclusion_count(
        self,
    ) -> int:
        return len(
            self.exclusions
        )


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

    try:
        label = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Invalid Fakeddit 2_way_label "
            f"{value!r} for sample "
            f"{sample.sample_id}."
        ) from exc

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

    Even:
        half label 0
        half label 1

    Odd:
        floor(N/2) label 0
        ceil(N/2) label 1
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


def _reserve_counts(
    targets: Dict[int, int],
    reserve_fraction: float,
    min_reserve_per_class: int,
) -> Dict[int, int]:
    """
    Calculate additional deterministic candidate capacity per class.
    """

    reserve_fraction = float(
        reserve_fraction
    )

    min_reserve_per_class = int(
        min_reserve_per_class
    )

    if reserve_fraction < 0:
        raise ValueError(
            "reserve_fraction must be >= 0."
        )

    if min_reserve_per_class < 0:
        raise ValueError(
            "min_reserve_per_class must be >= 0."
        )

    result = {}

    for label, target in (
        targets.items()
    ):

        proportional = int(
            round(
                target
                * reserve_fraction
            )
        )

        reserve = max(
            proportional,
            min_reserve_per_class,
        )

        result[
            label
        ] = reserve

    return result


def select_stratified_fakeddit_samples(
    samples: Iterable[
        EmpiricalSample
    ],
    sample_count: int,
    seed: int = 42,
) -> FakedditSampleSelection:
    """
    Deterministic approximately balanced sampling.

    This implementation stores the full supplied population in memory.

    For large Fakeddit partitions prefer one of the streaming
    alternatives.
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

    return FakedditSampleSelection(
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

        candidate_count=len(
            selected
        ),

        valid_candidate_count=len(
            selected
        ),
    )


def _stream_reservoir(
    samples: Iterable[
        EmpiricalSample
    ],
    reservoir_targets: Dict[
        int,
        int,
    ],
    seed: int,
):
    """
    Build deterministic class-specific reservoirs.

    Returns:
        reservoirs,
        class_seen,
        population_seen
    """

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
                "during streaming sampling: "
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

        required = (
            reservoir_targets[
                label
            ]
        )

        if required <= 0:
            continue

        reservoir = (
            reservoirs[
                label
            ]
        )

        if len(
            reservoir
        ) < required:

            reservoir.append(
                sample
            )

            continue

        candidate_index = (
            rng.randrange(
                class_seen[
                    label
                ]
            )
        )

        if (
            candidate_index
            < required
        ):

            reservoir[
                candidate_index
            ] = sample

    return (
        reservoirs,
        class_seen,
        population_seen,
    )


def select_streaming_stratified_fakeddit_samples(
    samples: Iterable[
        EmpiricalSample
    ],
    sample_count: int,
    seed: int = 42,
) -> FakedditSampleSelection:
    """
    Deterministic binary class-stratified reservoir sampling.

    Memory complexity:
        O(sample_count)
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

    (
        reservoirs,
        class_seen,
        population_seen,
    ) = _stream_reservoir(
        samples=samples,
        reservoir_targets=targets,
        seed=seed,
    )

    for label, required in (
        targets.items()
    ):

        available = (
            class_seen[
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

    selected = (
        list(
            reservoirs[
                0
            ]
        )
        +
        list(
            reservoirs[
                1
            ]
        )
    )

    rng = random.Random(
        seed
        + 1
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

    return FakedditSampleSelection(
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

        candidate_count=len(
            selected
        ),

        valid_candidate_count=len(
            selected
        ),
    )


def select_streaming_decodable_fakeddit_samples(
    samples: Iterable[
        EmpiricalSample
    ],
    sample_count: int,
    seed: int = 42,
    reserve_fraction: float = 0.10,
    min_reserve_per_class: int = 32,
) -> FakedditSampleSelection:
    """
    Deterministic stratified reservoir sampling with image eligibility.

    Algorithm
    ---------
    1. Determine target count for each native binary class.
    2. Build a bounded-memory reservoir containing:
           target + deterministic reserve
       candidates for each class.
    3. Validate image decodability only for those candidates.
    4. Reject invalid images.
    5. Fill the final class quota from valid candidates.
    6. Deterministically shuffle the final balanced selection.

    This prevents expensive XLM-R/CLIP representation generation from
    starting before all selected empirical images are known to be
    decodable.

    Source dataset files are never modified.
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

    reserves = _reserve_counts(
        targets=targets,
        reserve_fraction=(
            reserve_fraction
        ),
        min_reserve_per_class=(
            min_reserve_per_class
        ),
    )

    candidate_targets = {
        label: (
            targets[
                label
            ]
            +
            reserves[
                label
            ]
        )
        for label in (
            0,
            1,
        )
    }

    (
        reservoirs,
        class_seen,
        population_seen,
    ) = _stream_reservoir(
        samples=samples,
        reservoir_targets=(
            candidate_targets
        ),
        seed=seed,
    )

    for label in (
        0,
        1,
    ):

        required = (
            targets[
                label
            ]
        )

        population_available = (
            class_seen[
                label
            ]
        )

        if (
            population_available
            < required
        ):
            raise ValueError(
                f"Not enough source samples for "
                f"native label {label}: "
                f"required {required}, "
                f"available "
                f"{population_available}."
            )

    exclusions: List[
        FakedditSampleExclusion
    ] = []

    valid_by_class = {
        0: [],
        1: [],
    }

    candidate_count = 0

    for label in (
        0,
        1,
    ):

        for sample in (
            reservoirs[
                label
            ]
        ):

            candidate_count += 1

            if (
                sample.image_path
                is None
            ):

                exclusions.append(
                    FakedditSampleExclusion(
                        sample_id=(
                            sample.sample_id
                        ),
                        native_label=(
                            label
                        ),
                        image_path=None,
                        reason=(
                            "missing_image_path"
                        ),
                        message=(
                            "Empirical sample does "
                            "not contain a local "
                            "image path."
                        ),
                    )
                )

                continue

            result = (
                validate_decodable_image(
                    sample.image_path
                )
            )

            if not result.valid:

                exclusions.append(
                    FakedditSampleExclusion(
                        sample_id=(
                            sample.sample_id
                        ),
                        native_label=(
                            label
                        ),
                        image_path=(
                            str(
                                sample.image_path
                            )
                        ),
                        reason=(
                            result.error_type
                            or
                            "image_validation_failed"
                        ),
                        message=(
                            result.error_message
                        ),
                    )
                )

                continue

            valid_by_class[
                label
            ].append(
                sample
            )

    for label in (
        0,
        1,
    ):

        required = (
            targets[
                label
            ]
        )

        valid_available = len(
            valid_by_class[
                label
            ]
        )

        if (
            valid_available
            < required
        ):

            raise RuntimeError(
                "Decodability-aware candidate pool "
                "was insufficient for native label "
                f"{label}: required {required}, "
                f"valid candidates "
                f"{valid_available}. "
                "Increase reserve_fraction or "
                "min_reserve_per_class."
            )

    selected = []

    for label in (
        0,
        1,
    ):

        # Reservoir positions are already
        # deterministically generated.
        # Preserve that candidate order and
        # take the first eligible quota.
        selected.extend(
            valid_by_class[
                label
            ][
                : targets[
                    label
                ]
            ]
        )

    final_rng = random.Random(
        seed
        + 2
    )

    final_rng.shuffle(
        selected
    )

    ids = [
        sample.sample_id
        for sample in selected
    ]

    if (
        len(
            ids
        )
        != len(
            set(
                ids
            )
        )
    ):
        raise RuntimeError(
            "Duplicate sample IDs detected "
            "after eligibility filtering."
        )

    counts = Counter(
        _native_binary_label(
            sample
        )
        for sample in selected
    )

    expected = {
        int(label): int(
            count
        )
        for label, count
        in sorted(
            targets.items()
        )
    }

    actual = {
        int(label): int(
            count
        )
        for label, count
        in sorted(
            counts.items()
        )
    }

    if actual != expected:
        raise RuntimeError(
            "Final decodable Fakeddit selection "
            "does not preserve requested "
            f"class balance. Expected {expected}, "
            f"received {actual}."
        )

    valid_candidate_count = sum(
        len(
            valid_by_class[
                label
            ]
        )
        for label in (
            0,
            1,
        )
    )

    return FakedditSampleSelection(
        samples=selected,

        requested_count=(
            sample_count
        ),

        actual_count=len(
            selected
        ),

        seed=seed,

        native_label_counts=(
            actual
        ),

        strategy=(
            "streaming_stratified_reservoir_"
            "decodable"
        ),

        population_seen=(
            population_seen
        ),

        candidate_count=(
            candidate_count
        ),

        valid_candidate_count=(
            valid_candidate_count
        ),

        exclusions=tuple(
            exclusions
        ),
    )