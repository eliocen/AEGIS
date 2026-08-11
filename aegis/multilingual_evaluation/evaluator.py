"""
AEGIS Language-Stratified Evaluator.

Version: 0.22.0
"""

from collections import (
    defaultdict,
)

from aegis.evaluation import (
    hierarchical_metrics,
)

from .language import (
    normalize_language,
)

from .result import (
    LanguageEvaluationResult,
)


class MultilingualEvaluator:
    """
    Evaluate full AEGIS hierarchical performance
    independently for each language.
    """

    def evaluate(
        self,
        languages,
        integrity_targets,
        integrity_predictions,
        threat_targets,
        threat_predictions,
    ):

        languages = [
            normalize_language(
                language
            )
            for language
            in languages
        ]

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

        sample_count = len(
            languages
        )

        if not (
            len(integrity_targets)
            == len(integrity_predictions)
            == len(threat_targets)
            == len(threat_predictions)
            == sample_count
        ):
            raise ValueError(
                "All multilingual evaluation arrays "
                "must have equal length."
            )

        if sample_count == 0:
            raise ValueError(
                "Cannot evaluate an empty "
                "multilingual dataset."
            )

        grouped = defaultdict(
            lambda: {
                "integrity_targets": [],
                "integrity_predictions": [],
                "threat_targets": [],
                "threat_predictions": [],
            }
        )

        for index, language in enumerate(
            languages
        ):

            grouped[
                language
            ][
                "integrity_targets"
            ].append(
                integrity_targets[
                    index
                ]
            )

            grouped[
                language
            ][
                "integrity_predictions"
            ].append(
                integrity_predictions[
                    index
                ]
            )

            grouped[
                language
            ][
                "threat_targets"
            ].append(
                threat_targets[
                    index
                ]
            )

            grouped[
                language
            ][
                "threat_predictions"
            ].append(
                threat_predictions[
                    index
                ]
            )

        results = {}

        for (
            language,
            values,
        ) in grouped.items():

            metrics = (
                hierarchical_metrics(
                    values[
                        "integrity_targets"
                    ],

                    values[
                        "integrity_predictions"
                    ],

                    values[
                        "threat_targets"
                    ],

                    values[
                        "threat_predictions"
                    ],
                )
            )

            results[
                language
            ] = (
                LanguageEvaluationResult(
                    language=language,

                    sample_count=len(
                        values[
                            "integrity_targets"
                        ]
                    ),

                    metrics=metrics,

                    metadata={
                        "evaluation_scope": (
                            "language"
                        )
                    },
                )
            )

        return results