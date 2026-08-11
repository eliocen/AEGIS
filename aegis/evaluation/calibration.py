"""
AEGIS Confidence Calibration Metrics.

Version: 0.20.0
"""

def brier_score_binary(
    probabilities,
    targets,
):
    """
    Binary Brier score.
    """

    probabilities = list(
        probabilities
    )

    targets = list(
        targets
    )

    if len(probabilities) != len(
        targets
    ):
        raise ValueError(
            "probabilities and targets "
            "must have equal length."
        )

    if not probabilities:
        raise ValueError(
            "Cannot calculate Brier score "
            "for empty inputs."
        )

    return float(
        sum(
            (
                float(probability)
                - float(target)
            ) ** 2
            for probability, target
            in zip(
                probabilities,
                targets,
            )
        )
        / len(probabilities)
    )


def expected_calibration_error(
    confidences,
    correct,
    num_bins=10,
):
    """
    Expected Calibration Error.

    confidences:
        predicted confidence values in [0, 1]

    correct:
        1 if prediction was correct, otherwise 0
    """

    confidences = list(
        confidences
    )

    correct = list(
        correct
    )

    if len(confidences) != len(
        correct
    ):
        raise ValueError(
            "confidences and correctness arrays "
            "must have equal length."
        )

    if not confidences:
        raise ValueError(
            "Cannot calculate calibration "
            "for empty inputs."
        )

    if num_bins < 1:
        raise ValueError(
            "num_bins must be at least 1."
        )

    total = len(
        confidences
    )

    ece = 0.0

    for bin_index in range(
        num_bins
    ):

        lower = (
            bin_index
            / num_bins
        )

        upper = (
            (
                bin_index + 1
            )
            / num_bins
        )

        if (
            bin_index
            == num_bins - 1
        ):

            indices = [
                index
                for index, confidence
                in enumerate(
                    confidences
                )
                if (
                    confidence >= lower
                    and confidence <= upper
                )
            ]

        else:

            indices = [
                index
                for index, confidence
                in enumerate(
                    confidences
                )
                if (
                    confidence >= lower
                    and confidence < upper
                )
            ]

        if not indices:
            continue

        average_confidence = (
            sum(
                confidences[index]
                for index
                in indices
            )
            / len(indices)
        )

        accuracy = (
            sum(
                correct[index]
                for index
                in indices
            )
            / len(indices)
        )

        ece += (
            len(indices)
            / total
        ) * abs(
            average_confidence
            - accuracy
        )

    return float(
        ece
    )