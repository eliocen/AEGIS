"""
AEGIS Prediction Uncertainty Metrics.

Version: 0.23.0
"""

import math

from dataclasses import (
    asdict,
    dataclass,
)


@dataclass
class UncertaintyResult:
    """
    Uncertainty diagnostics for one
    categorical prediction.
    """

    predicted_class: int

    confidence: float

    predictive_entropy: float

    normalized_entropy: float

    confidence_margin: float

    uncertainty: float

    def as_dict(self):

        return asdict(
            self
        )


def predictive_entropy(
    probabilities,
    epsilon: float = 1e-12,
):
    """
    Shannon entropy for a categorical distribution.
    """

    probabilities = [
        float(value)
        for value in probabilities
    ]

    if not probabilities:
        raise ValueError(
            "probabilities cannot be empty."
        )

    total = sum(
        probabilities
    )

    if abs(
        total - 1.0
    ) > 1e-5:
        raise ValueError(
            "Probabilities must sum to 1."
        )

    entropy = 0.0

    for probability in probabilities:

        if probability < 0:
            raise ValueError(
                "Probabilities cannot be negative."
            )

        if probability > 0:

            entropy -= (
                probability
                * math.log(
                    max(
                        probability,
                        epsilon,
                    )
                )
            )

    return float(
        entropy
    )


def normalized_predictive_entropy(
    probabilities,
):
    """
    Predictive entropy normalized to [0, 1].
    """

    probabilities = list(
        probabilities
    )

    class_count = len(
        probabilities
    )

    if class_count < 2:
        raise ValueError(
            "At least two classes "
            "are required."
        )

    entropy = (
        predictive_entropy(
            probabilities
        )
    )

    maximum_entropy = math.log(
        class_count
    )

    return float(
        entropy
        / maximum_entropy
    )


def prediction_uncertainty(
    probabilities,
):
    """
    Compute confidence, entropy and prediction
    margin for one categorical prediction.
    """

    probabilities = [
        float(value)
        for value in probabilities
    ]

    if len(probabilities) < 2:
        raise ValueError(
            "At least two class probabilities "
            "are required."
        )

    total = sum(
        probabilities
    )

    if abs(
        total - 1.0
    ) > 1e-5:
        raise ValueError(
            "Probabilities must sum to 1."
        )

    ranked = sorted(
        enumerate(
            probabilities
        ),
        key=lambda item: (
            item[1]
        ),
        reverse=True,
    )

    predicted_class = (
        ranked[
            0
        ][0]
    )

    confidence = (
        ranked[
            0
        ][1]
    )

    second_confidence = (
        ranked[
            1
        ][1]
    )

    margin = (
        confidence
        - second_confidence
    )

    entropy = predictive_entropy(
        probabilities
    )

    normalized_entropy = (
        normalized_predictive_entropy(
            probabilities
        )
    )

    return UncertaintyResult(
        predicted_class=(
            predicted_class
        ),

        confidence=float(
            confidence
        ),

        predictive_entropy=float(
            entropy
        ),

        normalized_entropy=float(
            normalized_entropy
        ),

        confidence_margin=float(
            margin
        ),

        uncertainty=float(
            normalized_entropy
        ),
    )