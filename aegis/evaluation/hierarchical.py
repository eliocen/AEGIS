"""
AEGIS Hierarchical Evaluation Metrics.

Version: 0.20.0
"""

from .classification import (
    classification_metrics,
)

from .labels import (
    FIVE_CLASS_LABELS,
    five_class_index,
    hierarchical_label,
)


def _empty_threat_metrics():
    """
    Return an empty Stage-2 metric structure when
    the evaluated subset contains no harmful samples.
    """

    return {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "per_class": {},
        "confusion_matrix": [
            [0, 0, 0, 0]
            for _ in range(4)
        ],
    }


def _build_named_five_class_metrics(
    raw_metrics,
):
    """
    Convert numeric five-class metric indices into
    analyst- and publication-friendly class names.
    """

    named_per_class = {}

    for index, label in enumerate(
        FIVE_CLASS_LABELS
    ):

        metrics = (
            raw_metrics[
                "per_class"
            ][index]
        )

        named_per_class[
            label
        ] = {
            "precision": float(
                metrics[
                    "precision"
                ]
            ),

            "recall": float(
                metrics[
                    "recall"
                ]
            ),

            "f1": float(
                metrics[
                    "f1"
                ]
            ),

            "support": int(
                metrics[
                    "support"
                ]
            ),
        }

    return {
        "labels": list(
            FIVE_CLASS_LABELS
        ),

        "accuracy": float(
            raw_metrics[
                "accuracy"
            ]
        ),

        "macro_precision": float(
            raw_metrics[
                "macro_precision"
            ]
        ),

        "macro_recall": float(
            raw_metrics[
                "macro_recall"
            ]
        ),

        "macro_f1": float(
            raw_metrics[
                "macro_f1"
            ]
        ),

        "weighted_f1": float(
            raw_metrics[
                "weighted_f1"
            ]
        ),

        "per_class": (
            named_per_class
        ),

        "confusion_matrix": (
            raw_metrics[
                "confusion_matrix"
            ]
        ),

        "confusion_matrix_labeled": {
            "rows": list(
                FIVE_CLASS_LABELS
            ),

            "columns": list(
                FIVE_CLASS_LABELS
            ),

            "matrix": (
                raw_metrics[
                    "confusion_matrix"
                ]
            ),
        },
    }


def hierarchical_metrics(
    integrity_targets,
    integrity_predictions,
    threat_targets,
    threat_predictions,
):
    """
    Evaluate the complete AEGIS hierarchical
    classification architecture.

    Returns:

    1. Stage-1 integrity metrics
    2. Stage-2 threat metrics
    3. Hierarchical exact-match accuracy
    4. Canonical five-class labels
    5. Explicit five-class classification metrics
    6. Five-class confusion matrix
    """

    integrity_targets = list(
        integrity_targets
    )

    integrity_predictions = list(
        integrity_predictions
    )

    threat_targets = list(
        threat_targets
    )

    threat_predictions = list(
        threat_predictions
    )

    if not (
        len(integrity_targets)
        == len(integrity_predictions)
        == len(threat_targets)
        == len(threat_predictions)
    ):
        raise ValueError(
            "All hierarchical prediction arrays "
            "must have equal length."
        )

    if not integrity_targets:
        raise ValueError(
            "Cannot evaluate an empty "
            "hierarchical prediction set."
        )

    # ---------------------------------
    # Stage 1:
    # TRUE versus HARMFUL
    # ---------------------------------

    stage1 = classification_metrics(
        integrity_targets,
        integrity_predictions,
        num_classes=2,
    )

    # ---------------------------------
    # Stage 2:
    # Threat subtype among ground-truth
    # harmful samples
    # ---------------------------------

    harmful_indices = [
        index
        for index, target
        in enumerate(
            integrity_targets
        )
        if target == 1
    ]

    if harmful_indices:

        stage2_targets = [
            threat_targets[index]
            for index
            in harmful_indices
        ]

        stage2_predictions = [
            threat_predictions[index]
            for index
            in harmful_indices
        ]

        stage2 = (
            classification_metrics(
                stage2_targets,
                stage2_predictions,
                num_classes=4,
            )
        )

    else:

        stage2 = (
            _empty_threat_metrics()
        )

    # ---------------------------------
    # Canonical five-class evaluation
    # ---------------------------------

    five_class_targets = []
    five_class_predictions = []

    five_class_target_indices = []
    five_class_prediction_indices = []

    exact_correct = 0

    for (
        integrity_target,
        integrity_prediction,
        threat_target,
        threat_prediction,
    ) in zip(
        integrity_targets,
        integrity_predictions,
        threat_targets,
        threat_predictions,
    ):

        target_label = (
            hierarchical_label(
                integrity_target,
                threat_target,
            )
        )

        if integrity_prediction == 0:

            predicted_label = (
                "true"
            )

        elif integrity_prediction == 1:

            predicted_label = (
                hierarchical_label(
                    1,
                    threat_prediction,
                )
            )

        else:

            raise ValueError(
                "integrity_prediction must "
                "be 0 or 1."
            )

        five_class_targets.append(
            target_label
        )

        five_class_predictions.append(
            predicted_label
        )

        five_class_target_indices.append(
            five_class_index(
                target_label
            )
        )

        five_class_prediction_indices.append(
            five_class_index(
                predicted_label
            )
        )

        if (
            target_label
            == predicted_label
        ):
            exact_correct += 1

    hierarchical_exact_accuracy = (
        exact_correct
        / len(
            integrity_targets
        )
    )

    raw_five_class_metrics = (
        classification_metrics(
            five_class_target_indices,
            five_class_prediction_indices,
            num_classes=5,
        )
    )

    five_class_metrics = (
        _build_named_five_class_metrics(
            raw_five_class_metrics
        )
    )

    return {
        "integrity": (
            stage1
        ),

        "threat": (
            stage2
        ),

        "hierarchical_exact_accuracy": float(
            hierarchical_exact_accuracy
        ),

        "five_class": (
            five_class_metrics
        ),

        # Retained for compatibility with
        # the v0.20.0 error-analysis system.
        "five_class_targets": (
            five_class_targets
        ),

        "five_class_predictions": (
            five_class_predictions
        ),
    }