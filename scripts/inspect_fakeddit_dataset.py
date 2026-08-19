"""
AEGIS Empirical Research Phase 1.

Real Fakeddit dataset inspection and acquisition audit.

This script streams the official multimodal splits and reports
dataset integrity without loading images into memory.
"""

from collections import Counter
from pathlib import Path

from aegis.data.adapters import (
    FakedditAdapter,
    ImageStatus,
)


DATASET_ROOT = Path(
    "data/raw/fakeddit"
)


AUDIT_LIMIT = 10_000


def inspect_split(
    adapter: FakedditAdapter,
    split: str,
    limit: int = AUDIT_LIMIT,
) -> None:

    print()
    print("=" * 72)
    print(
        "FAKEDDIT SPLIT:",
        split.upper(),
    )
    print("=" * 72)

    total = 0
    image_counts = Counter()
    two_way = Counter()
    three_way = Counter()
    six_way = Counter()

    first_samples = []

    for sample in adapter.iter_samples(
        split,
        chunksize=2_000,
        limit=limit,
    ):

        total += 1

        image_counts[
            sample.image_status.value
        ] += 1

        two_way[
            sample.native_labels[
                "2_way_label"
            ]
        ] += 1

        three_way[
            sample.native_labels[
                "3_way_label"
            ]
        ] += 1

        six_way[
            sample.native_labels[
                "6_way_label"
            ]
        ] += 1

        if len(
            first_samples
        ) < 5:

            first_samples.append(
                sample
            )

    print(
        "Samples audited:",
        total,
    )

    print()
    print(
        "IMAGE STATUS"
    )

    for key, value in sorted(
        image_counts.items()
    ):
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "2-WAY LABELS:",
        dict(
            sorted(
                two_way.items()
            )
        ),
    )

    print(
        "3-WAY LABELS:",
        dict(
            sorted(
                three_way.items()
            )
        ),
    )

    print(
        "6-WAY LABELS:",
        dict(
            sorted(
                six_way.items()
            )
        ),
    )

    print()
    print(
        "FIRST FIVE SAMPLES"
    )

    for sample in first_samples:

        print(
            "-",
            sample.sample_id,
            "|",
            sample.image_status.value,
            "|",
            sample.native_labels,
        )


def main() -> None:

    print(
        "AEGIS Fakeddit Empirical Dataset Audit"
    )

    print(
        "=" * 72
    )

    adapter = FakedditAdapter(
        DATASET_ROOT
    )

    adapter.validate_structure()

    print(
        "Dataset root:",
        adapter.dataset_root.resolve(),
    )

    print(
        "Multimodal root:",
        adapter.multimodal_root.resolve(),
    )

    print(
        "Image root:",
        adapter.image_root.resolve(),
    )

    print()
    print(
        "Structure validation: PASSED"
    )

    for split in (
        "train",
        "validation",
        "test",
    ):

        columns = (
            adapter.inspect_schema(
                split
            )
        )

        print()
        print(
            split,
            "schema columns:",
            len(columns),
        )

    for split in (
        "train",
        "validation",
        "test",
    ):

        inspect_split(
            adapter,
            split,
        )

    print()
    print("=" * 72)
    print(
        "AEGIS Fakeddit empirical adapter is operational."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()