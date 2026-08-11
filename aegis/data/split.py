"""
AEGIS Research Dataset Splitting.

Version: 0.17.0
"""

import random
from collections import defaultdict
from typing import Dict, Iterable, List

from .record import (
    ResearchSample,
)


def _validate_ratios(
    train_ratio,
    validation_ratio,
    test_ratio,
):
    total = (
        train_ratio
        + validation_ratio
        + test_ratio
    )

    if abs(
        total - 1.0
    ) > 1e-8:
        raise ValueError(
            "Train, validation and test ratios "
            "must sum to 1.0."
        )

    if (
        train_ratio <= 0
        or validation_ratio < 0
        or test_ratio < 0
    ):
        raise ValueError(
            "Split ratios are invalid."
        )


def random_split_samples(
    samples: Iterable[
        ResearchSample
    ],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[
    str,
    List[ResearchSample]
]:

    _validate_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    samples = list(
        samples
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        samples
    )

    total = len(
        samples
    )

    train_end = int(
        total * train_ratio
    )

    validation_end = (
        train_end
        + int(
            total
            * validation_ratio
        )
    )

    return {
        "train": (
            samples[
                :train_end
            ]
        ),

        "validation": (
            samples[
                train_end:
                validation_end
            ]
        ),

        "test": (
            samples[
                validation_end:
            ]
        ),
    }


def group_aware_split(
    samples: Iterable[
        ResearchSample
    ],
    group_attribute: str = "group_id",
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Split complete groups rather than individual samples.

    Useful for limiting leakage across event/story/source
    clusters.
    """

    _validate_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    groups = defaultdict(
        list
    )

    for sample in samples:

        group_value = getattr(
            sample,
            group_attribute,
            None,
        )

        if group_value is None:

            group_value = (
                f"__sample__"
                f"{sample.sample_id}"
            )

        groups[
            str(group_value)
        ].append(
            sample
        )

    group_keys = list(
        groups.keys()
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        group_keys
    )

    total_groups = len(
        group_keys
    )

    train_end = int(
        total_groups
        * train_ratio
    )

    validation_end = (
        train_end
        + int(
            total_groups
            * validation_ratio
        )
    )

    train_keys = set(
        group_keys[
            :train_end
        ]
    )

    validation_keys = set(
        group_keys[
            train_end:
            validation_end
        ]
    )

    test_keys = set(
        group_keys[
            validation_end:
        ]
    )

    return {
        "train": [
            sample
            for key in train_keys
            for sample in groups[
                key
            ]
        ],

        "validation": [
            sample
            for key in validation_keys
            for sample in groups[
                key
            ]
        ],

        "test": [
            sample
            for key in test_keys
            for sample in groups[
                key
            ]
        ],
    }