"""
AEGIS Robustness Evaluation Metrics.

Version: 0.23.0
"""


def performance_degradation(
    clean_metric,
    perturbed_metric,
):
    """
    Measure absolute and relative degradation.
    """

    clean_metric = float(
        clean_metric
    )

    perturbed_metric = float(
        perturbed_metric
    )

    absolute = (
        perturbed_metric
        - clean_metric
    )

    if clean_metric != 0:

        relative_percent = (
            absolute
            / abs(
                clean_metric
            )
            * 100.0
        )

    else:

        relative_percent = 0.0

    return {
        "clean": (
            clean_metric
        ),

        "perturbed": (
            perturbed_metric
        ),

        "absolute_degradation": float(
            absolute
        ),

        "relative_degradation_percent": float(
            relative_percent
        ),
    }


def prediction_consistency(
    clean_predictions,
    perturbed_predictions,
):
    """
    Fraction of predictions unchanged after
    perturbation.
    """

    clean_predictions = list(
        clean_predictions
    )

    perturbed_predictions = list(
        perturbed_predictions
    )

    if len(clean_predictions) != len(
        perturbed_predictions
    ):
        raise ValueError(
            "Prediction arrays must "
            "have equal length."
        )

    if not clean_predictions:
        raise ValueError(
            "Prediction arrays cannot be empty."
        )

    unchanged = sum(
        int(
            clean
            == perturbed
        )
        for clean, perturbed
        in zip(
            clean_predictions,
            perturbed_predictions,
        )
    )

    return float(
        unchanged
        / len(
            clean_predictions
        )
    )


def confidence_degradation(
    clean_confidences,
    perturbed_confidences,
):
    """
    Mean confidence change following perturbation.
    """

    clean_confidences = [
        float(value)
        for value in clean_confidences
    ]

    perturbed_confidences = [
        float(value)
        for value in perturbed_confidences
    ]

    if len(clean_confidences) != len(
        perturbed_confidences
    ):
        raise ValueError(
            "Confidence arrays must "
            "have equal length."
        )

    if not clean_confidences:
        raise ValueError(
            "Confidence arrays cannot be empty."
        )

    clean_mean = (
        sum(
            clean_confidences
        )
        / len(
            clean_confidences
        )
    )

    perturbed_mean = (
        sum(
            perturbed_confidences
        )
        / len(
            perturbed_confidences
        )
    )

    return {
        "clean_mean_confidence": float(
            clean_mean
        ),

        "perturbed_mean_confidence": float(
            perturbed_mean
        ),

        "absolute_change": float(
            perturbed_mean
            - clean_mean
        ),
    }