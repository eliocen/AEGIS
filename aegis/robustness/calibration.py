"""
AEGIS Robustness Calibration Metrics.

Version: 0.23.0
"""

import math


def _validate_probabilities(
    probabilities,
):
    probabilities = [
        float(value)
        for value in probabilities
    ]

    if not probabilities:
        raise ValueError(
            "Probability collection cannot be empty."
        )

    for probability in probabilities:

        if (
            probability < 0.0
            or probability > 1.0
        ):
            raise ValueError(
                "Probabilities must lie "
                "within [0, 1]."
            )

    return probabilities


def binary_brier_score(
    probabilities,
    targets,
):
    """
    Binary Brier score.
    """

    probabilities = (
        _validate_probabilities(
            probabilities
        )
    )

    targets = [
        int(target)
        for target in targets
    ]

    if len(probabilities) != len(
        targets
    ):
        raise ValueError(
            "probabilities and targets "
            "must have equal length."
        )

    for target in targets:

        if target not in (
            0,
            1,
        ):
            raise ValueError(
                "Binary targets must "
                "be 0 or 1."
            )

    return float(
        sum(
            (
                probability
                - target
            ) ** 2
            for probability, target
            in zip(
                probabilities,
                targets,
            )
        )
        / len(probabilities)
    )


def negative_log_likelihood(
    true_class_probabilities,
    epsilon: float = 1e-12,
):
    """
    Mean negative log likelihood using
    probability assigned to the true class.
    """

    probabilities = (
        _validate_probabilities(
            true_class_probabilities
        )
    )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    losses = []

    for probability in probabilities:

        probability = max(
            probability,
            epsilon,
        )

        losses.append(
            -math.log(
                probability
            )
        )

    return float(
        sum(losses)
        / len(losses)
    )


def reliability_bins(
    confidences,
    correct,
    num_bins: int = 10,
):
    """
    Construct reliability-diagram bins.

    Returns:
        count
        mean_confidence
        accuracy
        calibration_gap
    """

    confidences = (
        _validate_probabilities(
            confidences
        )
    )

    correct = [
        int(value)
        for value in correct
    ]

    if len(confidences) != len(
        correct
    ):
        raise ValueError(
            "confidences and correctness values "
            "must have equal length."
        )

    if num_bins < 1:
        raise ValueError(
            "num_bins must be at least 1."
        )

    for value in correct:

        if value not in (
            0,
            1,
        ):
            raise ValueError(
                "correct values must "
                "be 0 or 1."
            )

    bins = []

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

        indices = []

        for index, confidence in enumerate(
            confidences
        ):

            if bin_index == (
                num_bins - 1
            ):

                inside = (
                    confidence >= lower
                    and confidence <= upper
                )

            else:

                inside = (
                    confidence >= lower
                    and confidence < upper
                )

            if inside:

                indices.append(
                    index
                )

        if indices:

            mean_confidence = (
                sum(
                    confidences[index]
                    for index in indices
                )
                / len(indices)
            )

            accuracy = (
                sum(
                    correct[index]
                    for index in indices
                )
                / len(indices)
            )

        else:

            mean_confidence = 0.0
            accuracy = 0.0

        bins.append(
            {
                "bin_index": (
                    bin_index
                ),

                "lower": float(
                    lower
                ),

                "upper": float(
                    upper
                ),

                "count": len(
                    indices
                ),

                "mean_confidence": float(
                    mean_confidence
                ),

                "accuracy": float(
                    accuracy
                ),

                "calibration_gap": float(
                    abs(
                        mean_confidence
                        - accuracy
                    )
                ),
            }
        )

    return bins


def expected_calibration_error(
    confidences,
    correct,
    num_bins: int = 10,
):
    """
    Expected Calibration Error.
    """

    bins = reliability_bins(
        confidences,
        correct,
        num_bins=num_bins,
    )

    total = sum(
        item[
            "count"
        ]
        for item in bins
    )

    if total == 0:
        raise ValueError(
            "Cannot calculate ECE "
            "without observations."
        )

    return float(
        sum(
            (
                item[
                    "count"
                ]
                / total
            )
            * item[
                "calibration_gap"
            ]
            for item in bins
        )
    )


def maximum_calibration_error(
    confidences,
    correct,
    num_bins: int = 10,
):
    """
    Maximum Calibration Error.
    """

    bins = reliability_bins(
        confidences,
        correct,
        num_bins=num_bins,
    )

    populated = [
        item[
            "calibration_gap"
        ]
        for item in bins
        if item[
            "count"
        ] > 0
    ]

    if not populated:
        raise ValueError(
            "Cannot calculate MCE "
            "without observations."
        )

    return float(
        max(
            populated
        )
    )