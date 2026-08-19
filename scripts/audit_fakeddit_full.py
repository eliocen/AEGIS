"""
AEGIS Empirical Research Phase 1.

Complete Fakeddit Dataset Integrity Audit
and Reproducible Manifest Generator.

This script scans the full official multimodal
train, validation and public-test splits.

The process can take several minutes because
hundreds of thousands of image paths are checked.
"""

from pathlib import Path

from aegis.data.audit import (
    FakedditFullAuditor,
)


DATASET_ROOT = Path(
    "data/raw/fakeddit"
)

OUTPUT_ROOT = Path(
    "data/manifests/fakeddit"
)


def print_split(
    split: str,
    statistics,
) -> None:

    print()
    print(
        "=" * 72
    )

    print(
        split.upper()
    )

    print(
        "=" * 72
    )

    print(
        "Total rows:",
        statistics[
            "total_rows"
        ],
    )

    print(
        "Unique IDs:",
        statistics[
            "unique_ids"
        ],
    )

    print(
        "Duplicate rows:",
        statistics[
            "duplicate_rows"
        ],
    )

    print(
        "Missing IDs:",
        statistics[
            "missing_ids"
        ],
    )

    print(
        "Missing text:",
        statistics[
            "missing_text"
        ],
    )

    print(
        "Empty text:",
        statistics[
            "empty_text"
        ],
    )

    print(
        "Images available:",
        statistics[
            "image_available"
        ],
    )

    print(
        "Images missing:",
        statistics[
            "image_missing"
        ],
    )

    print(
        "Image coverage:",
        (
            f"{statistics['image_coverage'] * 100:.4f}%"
        ),
    )

    print(
        "Strict multimodal eligible:",
        statistics[
            "eligible_strict_multimodal"
        ],
    )

    print(
        "Strict multimodal excluded:",
        statistics[
            "excluded_strict_multimodal"
        ],
    )

    print()
    print(
        "2-way labels:",
        statistics[
            "native_label_distributions"
        ][
            "2_way_label"
        ],
    )

    print(
        "3-way labels:",
        statistics[
            "native_label_distributions"
        ][
            "3_way_label"
        ],
    )

    print(
        "6-way labels:",
        statistics[
            "native_label_distributions"
        ][
            "6_way_label"
        ],
    )

    print()
    print(
        "Top domains:"
    )

    for domain, count in (
        statistics[
            "top_domains"
        ].items()
    ):

        print(
            f"  {domain}: {count}"
        )

    print()
    print(
        "Top subreddits:"
    )

    for subreddit, count in (
        statistics[
            "top_subreddits"
        ].items()
    ):

        print(
            f"  {subreddit}: {count}"
        )


def main() -> None:

    print(
        "AEGIS Full Fakeddit Dataset Audit"
    )

    print(
        "=" * 72
    )

    print(
        "Dataset:",
        DATASET_ROOT.resolve(),
    )

    print(
        "Manifest output:",
        OUTPUT_ROOT.resolve(),
    )

    print()
    print(
        "NOTE: This is a complete scan of all official "
        "multimodal splits."
    )

    print(
        "Image files are checked using direct ID-based "
        "path resolution."
    )

    print()

    auditor = (
        FakedditFullAuditor(
            dataset_root=(
                DATASET_ROOT
            ),

            output_root=(
                OUTPUT_ROOT
            ),

            chunksize=25_000,

            top_k_categories=20,
        )
    )

    audit = (
        auditor.run()
    )

    result = (
        audit.as_dict()
    )

    for split in (
        "train",
        "validation",
        "test",
    ):

        print_split(
            split,
            result[
                "splits"
            ][
                split
            ],
        )

    print()
    print(
        "=" * 72
    )

    print(
        "CROSS-SPLIT LEAKAGE"
    )

    print(
        "=" * 72
    )

    cross = result[
        "cross_split"
    ]

    print(
        "Train ↔ Validation:",
        cross[
            "train_validation_overlap"
        ],
    )

    print(
        "Train ↔ Test:",
        cross[
            "train_test_overlap"
        ],
    )

    print(
        "Validation ↔ Test:",
        cross[
            "validation_test_overlap"
        ],
    )

    print(
        "Any leakage:",
        cross[
            "any_cross_split_leakage"
        ],
    )

    print()
    print(
        "=" * 72
    )

    print(
        "DATASET TOTALS"
    )

    print(
        "=" * 72
    )

    totals = result[
        "dataset_totals"
    ]

    print(
        "Total records:",
        totals[
            "total_rows"
        ],
    )

    print(
        "Strict multimodal eligible:",
        totals[
            "strict_multimodal_eligible"
        ],
    )

    print(
        "Strict multimodal excluded:",
        totals[
            "strict_multimodal_excluded"
        ],
    )

    print(
        "Images available:",
        totals[
            "images_available"
        ],
    )

    print(
        "Images missing:",
        totals[
            "images_missing"
        ],
    )

    print(
        "Overall image coverage:",
        (
            f"{totals['overall_image_coverage'] * 100:.4f}%"
        ),
    )

    print(
        "Within-split duplicate rows:",
        totals[
            "duplicate_rows_within_splits"
        ],
    )

    print(
        "Missing/empty text:",
        totals[
            "missing_or_empty_text"
        ],
    )

    print(
        "Invalid native labels:",
        totals[
            "invalid_native_labels"
        ],
    )

    print()
    print(
        "Integrity status:",
        result[
            "integrity_status"
        ].upper(),
    )

    print()
    print(
        "Generated manifests:"
    )

    print(
        "  data/manifests/fakeddit/dataset_audit.json"
    )

    print(
        "  data/manifests/fakeddit/split_statistics.json"
    )

    print(
        "  data/manifests/fakeddit/provenance.json"
    )

    print(
        "  data/manifests/fakeddit/excluded_samples.jsonl"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "Fakeddit full empirical audit completed."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()