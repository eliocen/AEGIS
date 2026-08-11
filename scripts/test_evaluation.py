"""
AEGIS v0.20.0
Research Metrics, Evaluation & Error Analysis
sanity experiment.
"""

from aegis.evaluation import (
    AEGISEvaluator,
)


sample_ids = [
    "EVAL-001",
    "EVAL-002",
    "EVAL-003",
    "EVAL-004",
    "EVAL-005",
    "EVAL-006",
    "EVAL-007",
    "EVAL-008",
    "EVAL-009",
    "EVAL-010",
]


integrity_targets = [
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
]


integrity_predictions = [
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
]


threat_targets = [
    -1,
    0,
    1,
    2,
    3,
    -1,
    0,
    1,
    3,
    -1,
]


threat_predictions = [
    0,
    0,
    1,
    3,
    2,
    0,
    1,
    1,
    3,
    0,
]


integrity_confidences = [
    0.95,
    0.91,
    0.87,
    0.78,
    0.62,
    0.89,
    0.71,
    0.93,
    0.58,
    0.96,
]


languages = [
    "en",
    "en",
    "zh",
    "zh",
    "en",
    "en",
    "zh",
    "en",
    "zh",
    "en",
]


domains = [
    "general",
    "international_security",
    "international_security",
    "conflict",
    "conflict",
    "general",
    "international_security",
    "international_security",
    "conflict",
    "general",
]


source_datasets = [
    "dataset_a",
    "dataset_a",
    "dataset_b",
    "dataset_b",
    "dataset_a",
    "dataset_a",
    "dataset_b",
    "dataset_a",
    "dataset_b",
    "dataset_a",
]


evaluator = (
    AEGISEvaluator()
)


report = evaluator.evaluate(
    experiment_name=(
        "evaluation_sanity"
    ),

    sample_ids=(
        sample_ids
    ),

    integrity_targets=(
        integrity_targets
    ),

    integrity_predictions=(
        integrity_predictions
    ),

    threat_targets=(
        threat_targets
    ),

    threat_predictions=(
        threat_predictions
    ),

    integrity_confidences=(
        integrity_confidences
    ),

    languages=(
        languages
    ),

    domains=(
        domains
    ),

    source_datasets=(
        source_datasets
    ),
)


hierarchical = (
    report.metrics[
        "hierarchical"
    ]
)


five_class = (
    report.metrics[
        "five_class"
    ]
)


print(
    "AEGIS Research Evaluation"
)

print(
    "=" * 70
)


print(
    "Total samples:",
    report.total_samples,
)


print(
    "\nSTAGE 1 — INTEGRITY CLASSIFICATION"
)

print(
    "Accuracy:",
    round(
        hierarchical[
            "integrity"
        ][
            "accuracy"
        ],
        4,
    ),
)

print(
    "Macro F1:",
    round(
        hierarchical[
            "integrity"
        ][
            "macro_f1"
        ],
        4,
    ),
)


print(
    "\nSTAGE 2 — THREAT CLASSIFICATION"
)

print(
    "Accuracy:",
    round(
        hierarchical[
            "threat"
        ][
            "accuracy"
        ],
        4,
    ),
)

print(
    "Macro F1:",
    round(
        hierarchical[
            "threat"
        ][
            "macro_f1"
        ],
        4,
    ),
)


print(
    "\nHIERARCHICAL EXACT ACCURACY:",
    round(
        hierarchical[
            "hierarchical_exact_accuracy"
        ],
        4,
    ),
)


print(
    "\nFIVE-CLASS EVALUATION"
)

print(
    "-" * 70
)

print(
    "Accuracy:",
    round(
        five_class[
            "accuracy"
        ],
        4,
    ),
)

print(
    "Macro Precision:",
    round(
        five_class[
            "macro_precision"
        ],
        4,
    ),
)

print(
    "Macro Recall:",
    round(
        five_class[
            "macro_recall"
        ],
        4,
    ),
)

print(
    "Macro F1:",
    round(
        five_class[
            "macro_f1"
        ],
        4,
    ),
)

print(
    "Weighted F1:",
    round(
        five_class[
            "weighted_f1"
        ],
        4,
    ),
)


print(
    "\nPER-CLASS METRICS"
)

print(
    "-" * 70
)

for label in five_class[
    "labels"
]:

    metrics = (
        five_class[
            "per_class"
        ][label]
    )

    print(
        label
    )

    print(
        "  Precision:",
        round(
            metrics[
                "precision"
            ],
            4,
        ),
    )

    print(
        "  Recall:",
        round(
            metrics[
                "recall"
            ],
            4,
        ),
    )

    print(
        "  F1:",
        round(
            metrics[
                "f1"
            ],
            4,
        ),
    )

    print(
        "  Support:",
        metrics[
            "support"
        ],
    )


print(
    "\nFIVE-CLASS CONFUSION MATRIX"
)

print(
    "Rows    = Ground Truth"
)

print(
    "Columns = Prediction"
)

print(
    "-" * 70
)


labels = (
    five_class[
        "labels"
    ]
)


matrix = (
    five_class[
        "confusion_matrix"
    ]
)


print(
    "Class order:"
)

for index, label in enumerate(
    labels
):

    print(
        f"{index}: {label}"
    )


print(
    "\nMatrix:"
)


for row in matrix:

    print(
        row
    )


print(
    "\nCALIBRATION"
)

print(
    "Brier score:",
    round(
        report.metrics[
            "calibration"
        ][
            "integrity_brier_score"
        ],
        4,
    ),
)

print(
    "ECE:",
    round(
        report.metrics[
            "calibration"
        ][
            "integrity_ece"
        ],
        4,
    ),
)


print(
    "\nLANGUAGE-STRATIFIED RESULTS"
)

for (
    language,
    metrics,
) in report.stratified_metrics[
    "language_integrity"
].items():

    print(
        language,
        "| accuracy:",
        round(
            metrics[
                "accuracy"
            ],
            4,
        ),
        "| macro F1:",
        round(
            metrics[
                "macro_f1"
            ],
            4,
        ),
    )


print(
    "\nERROR ANALYSIS"
)

print(
    "Errors:",
    len(
        report.errors
    ),
)


for error in report.errors:

    print(
        "-",
        error.sample_id,
        "|",
        error.true_label,
        "->",
        error.predicted_label,
        "| language:",
        error.language,
        "| domain:",
        error.domain,
    )