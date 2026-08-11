"""
AEGIS v0.22.0
Multilingual & Cross-Lingual Evaluation
sanity experiment.

All cross-lingual transfer measurements in this
script are synthetic and exist only to validate
the evaluation framework.
"""

from aegis.multilingual_evaluation import (
    CrossLingualRunResult,
    MultilingualEvaluationReport,
    MultilingualEvaluator,
    build_transfer_matrix,
    language_performance_gap,
    summarize_cross_lingual_transfer,
)


evaluator = (
    MultilingualEvaluator()
)


language_results = evaluator.evaluate(
    languages=[
        "en",
        "en",
        "en",
        "en",
        "en",
        "zh",
        "zh",
        "zh",
        "zh",
        "zh",
    ],

    integrity_targets=[
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
    ],

    integrity_predictions=[
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
    ],

    threat_targets=[
        -1,
        0,
        1,
        2,
        3,
        -1,
        0,
        1,
        2,
        3,
    ],

    threat_predictions=[
        0,
        0,
        1,
        2,
        3,
        0,
        0,
        1,
        3,
        3,
    ],
)


print(
    "AEGIS Multilingual Evaluation"
)

print(
    "=" * 80
)


print(
    "\nMONOLINGUAL PERFORMANCE"
)

print(
    "-" * 80
)


for (
    language,
    result,
) in language_results.items():

    five_class = (
        result.metrics[
            "five_class"
        ]
    )

    print(
        "Language:",
        language,
    )

    print(
        "  Samples:",
        result.sample_count,
    )

    print(
        "  Five-Class Accuracy:",
        round(
            five_class[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "  Five-Class Macro F1:",
        round(
            five_class[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "  Hierarchical Accuracy:",
        round(
            result.metrics[
                "hierarchical_exact_accuracy"
            ],
            4,
        ),
    )

    print(
        "-" * 80
    )


gap = (
    language_performance_gap(
        language_results
    )
)


print(
    "\nLANGUAGE PERFORMANCE GAP"
)

print(
    "-" * 80
)

print(
    "Best language:",
    gap[
        "best_language"
    ],
)

print(
    "Best Macro F1:",
    round(
        gap[
            "best_value"
        ],
        4,
    ),
)

print(
    "Worst language:",
    gap[
        "worst_language"
    ],
)

print(
    "Worst Macro F1:",
    round(
        gap[
            "worst_value"
        ],
        4,
    ),
)

print(
    "Absolute gap:",
    round(
        gap[
            "absolute_gap"
        ],
        4,
    ),
)


transfer_results = [
    CrossLingualRunResult(
        train_language="en",
        test_language="en",
        metric_name="five_class_macro_f1",
        metric_value=0.84,
        seed=42,
    ),

    CrossLingualRunResult(
        train_language="en",
        test_language="en",
        metric_name="five_class_macro_f1",
        metric_value=0.82,
        seed=123,
    ),

    CrossLingualRunResult(
        train_language="en",
        test_language="zh",
        metric_name="five_class_macro_f1",
        metric_value=0.71,
        seed=42,
    ),

    CrossLingualRunResult(
        train_language="en",
        test_language="zh",
        metric_name="five_class_macro_f1",
        metric_value=0.69,
        seed=123,
    ),

    CrossLingualRunResult(
        train_language="zh",
        test_language="en",
        metric_name="five_class_macro_f1",
        metric_value=0.73,
        seed=42,
    ),

    CrossLingualRunResult(
        train_language="zh",
        test_language="en",
        metric_name="five_class_macro_f1",
        metric_value=0.71,
        seed=123,
    ),

    CrossLingualRunResult(
        train_language="zh",
        test_language="zh",
        metric_name="five_class_macro_f1",
        metric_value=0.81,
        seed=42,
    ),

    CrossLingualRunResult(
        train_language="zh",
        test_language="zh",
        metric_name="five_class_macro_f1",
        metric_value=0.79,
        seed=123,
    ),
]


transfer_matrix = (
    build_transfer_matrix(
        transfer_results,

        languages=[
            "en",
            "zh",
        ],
    )
)


transfer_summary = (
    summarize_cross_lingual_transfer(
        transfer_results
    )
)


print(
    "\nCROSS-LINGUAL TRANSFER MATRIX"
)

print(
    "Rows    = Training language"
)

print(
    "Columns = Evaluation language"
)

print(
    "-" * 80
)


print(
    "Languages:",
    transfer_matrix[
        "languages"
    ],
)


for row in transfer_matrix[
    "matrix"
]:

    print(
        [
            round(
                value,
                4,
            )
            if value
            is not None
            else None
            for value in row
        ]
    )


print(
    "\nTRANSFER SUMMARY"
)

print(
    "-" * 80
)


print(
    "In-language mean:",
    round(
        transfer_summary[
            "in_language_mean"
        ],
        4,
    ),
)

print(
    "Cross-lingual mean:",
    round(
        transfer_summary[
            "cross_lingual_mean"
        ],
        4,
    ),
)

print(
    "Transfer gap:",
    round(
        transfer_summary[
            "absolute_transfer_gap"
        ],
        4,
    ),
)

print(
    "Relative transfer gap:",
    (
        f"{transfer_summary['relative_transfer_gap_percent']:.2f}%"
    ),
)


report = (
    MultilingualEvaluationReport(
        experiment_name=(
            "multilingual_sanity"
        ),

        language_results=(
            language_results
        ),

        language_gap=(
            gap
        ),

        transfer_matrix=(
            transfer_matrix
        ),

        transfer_summary=(
            transfer_summary
        ),

        metadata={
            "evaluation_version": (
                "0.22.0"
            ),

            "transfer_metrics_synthetic": (
                True
            ),
        },
    )
)


print(
    "\nREPORT LANGUAGES:",
    list(
        report.language_results.keys()
    ),
)


print(
    "\nIMPORTANT: Cross-lingual transfer "
    "values above are synthetic."
)


print(
    "\nAEGIS multilingual and cross-lingual "
    "evaluation framework is operational."
)