"""
AEGIS Error Analysis.

Version: 0.20.0
"""

from dataclasses import (
    asdict,
    dataclass,
)

from typing import Any, Dict, Optional


@dataclass
class EvaluationError:
    """
    One incorrectly classified research sample.
    """

    sample_id: str

    true_label: str

    predicted_label: str

    language: Optional[str] = None

    domain: Optional[str] = None

    source_dataset: Optional[str] = None

    confidence: Optional[float] = None

    metadata: Optional[
        Dict[str, Any]
    ] = None

    def as_dict(self):

        return asdict(
            self
        )


def collect_errors(
    sample_ids,
    true_labels,
    predicted_labels,
    languages=None,
    domains=None,
    source_datasets=None,
    confidences=None,
):
    """
    Collect misclassified samples for subsequent
    qualitative error analysis.
    """

    sample_ids = list(
        sample_ids
    )

    true_labels = list(
        true_labels
    )

    predicted_labels = list(
        predicted_labels
    )

    count = len(
        sample_ids
    )

    if not (
        len(true_labels)
        == len(predicted_labels)
        == count
    ):
        raise ValueError(
            "Core error-analysis arrays "
            "must have equal length."
        )

    languages = (
        list(languages)
        if languages is not None
        else [None] * count
    )

    domains = (
        list(domains)
        if domains is not None
        else [None] * count
    )

    source_datasets = (
        list(source_datasets)
        if source_datasets is not None
        else [None] * count
    )

    confidences = (
        list(confidences)
        if confidences is not None
        else [None] * count
    )

    if not (
        len(languages)
        == len(domains)
        == len(source_datasets)
        == len(confidences)
        == count
    ):
        raise ValueError(
            "Optional error-analysis arrays "
            "must match sample count."
        )

    errors = []

    for index in range(
        count
    ):

        if (
            true_labels[index]
            == predicted_labels[index]
        ):
            continue

        errors.append(
            EvaluationError(
                sample_id=(
                    sample_ids[index]
                ),

                true_label=(
                    true_labels[index]
                ),

                predicted_label=(
                    predicted_labels[index]
                ),

                language=(
                    languages[index]
                ),

                domain=(
                    domains[index]
                ),

                source_dataset=(
                    source_datasets[index]
                ),

                confidence=(
                    confidences[index]
                ),
            )
        )

    return errors