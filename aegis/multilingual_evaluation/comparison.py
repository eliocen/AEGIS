"""
AEGIS Cross-Lingual Performance Comparison.

Version: 0.22.0
"""

from .language import (
    normalize_language,
)

from .result import (
    CrossLingualRunResult,
)


def summarize_cross_lingual_transfer(
    results,
):
    """
    Compare in-language and cross-language
    performance using mean metric values.
    """

    results = list(
        results
    )

    if not results:
        raise ValueError(
            "No transfer results supplied."
        )

    metric_names = {
        result.metric_name
        for result in results
    }

    if len(
        metric_names
    ) != 1:
        raise ValueError(
            "Transfer summary supports "
            "one metric at a time."
        )

    in_language_values = []

    cross_lingual_values = []

    for result in results:

        train_language = (
            normalize_language(
                result.train_language
            )
        )

        test_language = (
            normalize_language(
                result.test_language
            )
        )

        value = float(
            result.metric_value
        )

        if (
            train_language
            == test_language
        ):

            in_language_values.append(
                value
            )

        else:

            cross_lingual_values.append(
                value
            )

    if not in_language_values:
        raise ValueError(
            "At least one in-language result "
            "is required."
        )

    if not cross_lingual_values:
        raise ValueError(
            "At least one cross-lingual result "
            "is required."
        )

    in_language_mean = (
        sum(
            in_language_values
        )
        / len(
            in_language_values
        )
    )

    cross_lingual_mean = (
        sum(
            cross_lingual_values
        )
        / len(
            cross_lingual_values
        )
    )

    transfer_gap = (
        cross_lingual_mean
        - in_language_mean
    )

    relative_gap_percent = (
        (
            transfer_gap
            / abs(
                in_language_mean
            )
        )
        * 100.0
        if in_language_mean != 0
        else 0.0
    )

    return {
        "metric": next(
            iter(
                metric_names
            )
        ),

        "in_language_mean": float(
            in_language_mean
        ),

        "cross_lingual_mean": float(
            cross_lingual_mean
        ),

        "absolute_transfer_gap": float(
            transfer_gap
        ),

        "relative_transfer_gap_percent": float(
            relative_gap_percent
        ),

        "in_language_runs": len(
            in_language_values
        ),

        "cross_lingual_runs": len(
            cross_lingual_values
        ),
    }


def language_performance_gap(
    language_results,
    metric_path=(
        "five_class",
        "macro_f1",
    ),
):
    """
    Measure the performance spread across
    language-specific evaluations.

    metric_path defaults to:
        five_class -> macro_f1
    """

    if not language_results:
        raise ValueError(
            "No language results supplied."
        )

    values = {}

    for (
        language,
        result,
    ) in language_results.items():

        metric_value = (
            result.metrics
        )

        for key in metric_path:

            metric_value = (
                metric_value[
                    key
                ]
            )

        values[
            normalize_language(
                language
            )
        ] = float(
            metric_value
        )

    best_language = max(
        values,
        key=values.get,
    )

    worst_language = min(
        values,
        key=values.get,
    )

    gap = (
        values[
            best_language
        ]
        - values[
            worst_language
        ]
    )

    return {
        "metric_path": list(
            metric_path
        ),

        "languages": (
            values
        ),

        "best_language": (
            best_language
        ),

        "best_value": (
            values[
                best_language
            ]
        ),

        "worst_language": (
            worst_language
        ),

        "worst_value": (
            values[
                worst_language
            ]
        ),

        "absolute_gap": float(
            gap
        ),
    }