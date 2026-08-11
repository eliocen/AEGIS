"""
AEGIS Dataset Statistics.

Version: 0.17.0
"""

from collections import Counter


def compute_dataset_statistics(
    samples,
):
    """
    Compute reproducible descriptive statistics
    for an AEGIS research dataset.
    """

    samples = list(samples)

    total = len(samples)

    label_counts = Counter(
        sample.label.value
        for sample in samples
    )

    language_counts = Counter(
        sample.language
        for sample in samples
    )

    domain_counts = Counter(
        sample.domain
        for sample in samples
    )

    source_dataset_counts = Counter(
        sample.source_dataset
        for sample in samples
    )

    group_counts = Counter(
        sample.group_id
        for sample in samples
        if sample.group_id is not None
    )

    event_counts = Counter(
        sample.event_id
        for sample in samples
        if sample.event_id is not None
    )

    source_counts = Counter(
        sample.source_id
        for sample in samples
        if sample.source_id is not None
    )

    return {
        "total_samples": total,

        "labels": dict(
            label_counts
        ),

        "languages": dict(
            language_counts
        ),

        "domains": dict(
            domain_counts
        ),

        "source_datasets": dict(
            source_dataset_counts
        ),

        "unique_groups": len(
            group_counts
        ),

        "unique_events": len(
            event_counts
        ),

        "unique_sources": len(
            source_counts
        ),
    }