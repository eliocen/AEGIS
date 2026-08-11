"""
AEGIS Classification Metrics.

Version: 0.20.0
"""

from collections import defaultdict


def confusion_matrix(
    targets,
    predictions,
    num_classes,
):
    """
    Build a confusion matrix as a nested list.

    Rows = true classes
    Columns = predicted classes
    """

    matrix = [
        [
            0
            for _ in range(num_classes)
        ]
        for _ in range(num_classes)
    ]

    for target, prediction in zip(
        targets,
        predictions,
    ):
        matrix[
            int(target)
        ][
            int(prediction)
        ] += 1

    return matrix


def per_class_metrics(
    targets,
    predictions,
    num_classes,
):
    """
    Compute precision, recall, F1 and support
    independently for every class.
    """

    matrix = confusion_matrix(
        targets,
        predictions,
        num_classes,
    )

    results = {}

    for class_index in range(
        num_classes
    ):

        true_positive = (
            matrix[
                class_index
            ][
                class_index
            ]
        )

        false_positive = sum(
            matrix[row][class_index]
            for row in range(
                num_classes
            )
            if row != class_index
        )

        false_negative = sum(
            matrix[
                class_index
            ][column]
            for column in range(
                num_classes
            )
            if column != class_index
        )

        support = sum(
            matrix[
                class_index
            ]
        )

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
            if (
                true_positive
                + false_positive
            ) > 0
            else 0.0
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        results[
            class_index
        ] = {
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "support": int(
                support
            ),
        }

    return results


def classification_metrics(
    targets,
    predictions,
    num_classes,
):
    """
    Compute accuracy, macro metrics and weighted F1.
    """

    targets = list(
        targets
    )

    predictions = list(
        predictions
    )

    if len(targets) != len(
        predictions
    ):
        raise ValueError(
            "targets and predictions must "
            "have equal length."
        )

    if not targets:
        raise ValueError(
            "Cannot evaluate an empty prediction set."
        )

    correct = sum(
        int(
            target == prediction
        )
        for target, prediction
        in zip(
            targets,
            predictions,
        )
    )

    accuracy = (
        correct
        / len(targets)
    )

    class_metrics = (
        per_class_metrics(
            targets,
            predictions,
            num_classes,
        )
    )

    precisions = [
        class_metrics[index][
            "precision"
        ]
        for index in range(
            num_classes
        )
    ]

    recalls = [
        class_metrics[index][
            "recall"
        ]
        for index in range(
            num_classes
        )
    ]

    f1_scores = [
        class_metrics[index][
            "f1"
        ]
        for index in range(
            num_classes
        )
    ]

    supports = [
        class_metrics[index][
            "support"
        ]
        for index in range(
            num_classes
        )
    ]

    total_support = sum(
        supports
    )

    macro_precision = (
        sum(precisions)
        / num_classes
    )

    macro_recall = (
        sum(recalls)
        / num_classes
    )

    macro_f1 = (
        sum(f1_scores)
        / num_classes
    )

    weighted_f1 = (
        sum(
            f1_scores[index]
            * supports[index]
            for index in range(
                num_classes
            )
        )
        / total_support
        if total_support > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "per_class": (
            class_metrics
        ),

        "confusion_matrix": (
            confusion_matrix(
                targets,
                predictions,
                num_classes,
            )
        ),
    }