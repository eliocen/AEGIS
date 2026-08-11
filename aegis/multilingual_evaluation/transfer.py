"""
AEGIS Cross-Lingual Transfer Evaluation.

Version: 0.22.0
"""

from collections import (
    defaultdict,
)

from .language import (
    normalize_language,
)

from .result import (
    CrossLingualRunResult,
)


def build_transfer_matrix(
    results,
    languages=None,
):
    """
    Build a train-language × test-language matrix.

    Each train/test pair may contain multiple seeded
    runs. The matrix stores their mean metric value.
    """

    results = list(
        results
    )

    if not results:
        raise ValueError(
            "No cross-lingual results supplied."
        )

    for result in results:

        if not isinstance(
            result,
            CrossLingualRunResult,
        ):
            raise TypeError(
                "All transfer results must be "
                "CrossLingualRunResult."
            )

    metric_names = {
        result.metric_name
        for result in results
    }

    if len(
        metric_names
    ) != 1:
        raise ValueError(
            "A transfer matrix must represent "
            "one metric at a time."
        )

    grouped = defaultdict(
        list
    )

    observed_languages = set()

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

        observed_languages.add(
            train_language
        )

        observed_languages.add(
            test_language
        )

        grouped[
            (
                train_language,
                test_language,
            )
        ].append(
            float(
                result.metric_value
            )
        )

    if languages is None:

        language_order = sorted(
            observed_languages
        )

    else:

        language_order = [
            normalize_language(
                language
            )
            for language
            in languages
        ]

    matrix = []

    for train_language in language_order:

        row = []

        for test_language in language_order:

            values = grouped.get(
                (
                    train_language,
                    test_language,
                ),
                [],
            )

            if values:

                value = (
                    sum(values)
                    / len(values)
                )

            else:

                value = None

            row.append(
                value
            )

        matrix.append(
            row
        )

    return {
        "metric": next(
            iter(
                metric_names
            )
        ),

        "languages": (
            language_order
        ),

        "rows": (
            language_order
        ),

        "columns": (
            language_order
        ),

        "matrix": (
            matrix
        ),
    }