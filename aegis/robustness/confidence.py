"""
AEGIS Confidence-Aware Error Analysis.

Version: 0.23.0
"""


def confidence_error_analysis(
    targets,
    predictions,
    confidences,
    high_confidence_threshold: float = 0.80,
):
    """
    Categorize predictions by correctness and
    confidence.
    """

    targets = list(
        targets
    )

    predictions = list(
        predictions
    )

    confidences = [
        float(value)
        for value in confidences
    ]

    if not (
        len(targets)
        == len(predictions)
        == len(confidences)
    ):
        raise ValueError(
            "targets, predictions and confidences "
            "must have equal length."
        )

    if not targets:
        raise ValueError(
            "Cannot analyze an empty "
            "prediction collection."
        )

    if not (
        0.0
        <= high_confidence_threshold
        <= 1.0
    ):
        raise ValueError(
            "high_confidence_threshold must "
            "lie within [0, 1]."
        )

    categories = {
        "high_confidence_correct": 0,
        "low_confidence_correct": 0,
        "high_confidence_incorrect": 0,
        "low_confidence_incorrect": 0,
    }

    correct_confidences = []
    incorrect_confidences = []

    for (
        target,
        prediction,
        confidence,
    ) in zip(
        targets,
        predictions,
        confidences,
    ):

        is_correct = (
            target
            == prediction
        )

        is_high_confidence = (
            confidence
            >= high_confidence_threshold
        )

        if is_correct:

            correct_confidences.append(
                confidence
            )

            if is_high_confidence:

                categories[
                    "high_confidence_correct"
                ] += 1

            else:

                categories[
                    "low_confidence_correct"
                ] += 1

        else:

            incorrect_confidences.append(
                confidence
            )

            if is_high_confidence:

                categories[
                    "high_confidence_incorrect"
                ] += 1

            else:

                categories[
                    "low_confidence_incorrect"
                ] += 1

    total = len(
        targets
    )

    high_confidence_errors = (
        categories[
            "high_confidence_incorrect"
        ]
    )

    mean_correct = (
        sum(
            correct_confidences
        )
        / len(
            correct_confidences
        )
        if correct_confidences
        else 0.0
    )

    mean_incorrect = (
        sum(
            incorrect_confidences
        )
        / len(
            incorrect_confidences
        )
        if incorrect_confidences
        else 0.0
    )

    return {
        **categories,

        "total": (
            total
        ),

        "high_confidence_error_rate": float(
            high_confidence_errors
            / total
        ),

        "mean_confidence_correct": float(
            mean_correct
        ),

        "mean_confidence_incorrect": float(
            mean_incorrect
        ),

        "threshold": float(
            high_confidence_threshold
        ),
    }