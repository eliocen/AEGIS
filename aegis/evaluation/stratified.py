"""
AEGIS Stratified Evaluation.

Version: 0.20.0
"""

from collections import defaultdict

from .classification import (
    classification_metrics,
)


def stratified_classification_metrics(
    targets,
    predictions,
    strata,
    num_classes,
):
    """
    Calculate classification metrics independently
    for each language/domain/source stratum.
    """

    targets = list(
        targets
    )

    predictions = list(
        predictions
    )

    strata = list(
        strata
    )

    if not (
        len(targets)
        == len(predictions)
        == len(strata)
    ):
        raise ValueError(
            "targets, predictions and strata "
            "must have equal length."
        )

    grouped = defaultdict(
        lambda: {
            "targets": [],
            "predictions": [],
        }
    )

    for (
        target,
        prediction,
        stratum,
    ) in zip(
        targets,
        predictions,
        strata,
    ):

        key = (
            str(stratum)
            if stratum is not None
            else "unknown"
        )

        grouped[key][
            "targets"
        ].append(
            target
        )

        grouped[key][
            "predictions"
        ].append(
            prediction
        )

    return {
        key: classification_metrics(
            values["targets"],
            values["predictions"],
            num_classes,
        )
        for key, values
        in grouped.items()
    }