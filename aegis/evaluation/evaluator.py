"""
AEGIS Research Evaluator.

Version: 0.20.0
"""

from .calibration import (
    brier_score_binary,
    expected_calibration_error,
)

from .errors import (
    collect_errors,
)

from .hierarchical import (
    hierarchical_metrics,
)

from .report import (
    EvaluationReport,
)

from .stratified import (
    stratified_classification_metrics,
)


class AEGISEvaluator:
    """
    High-level evaluator for AEGIS research runs.
    """

    def evaluate(
        self,
        experiment_name,
        sample_ids,
        integrity_targets,
        integrity_predictions,
        threat_targets,
        threat_predictions,
        integrity_confidences=None,
        languages=None,
        domains=None,
        source_datasets=None,
    ):
        """
        Produce a structured AEGIS research
        evaluation report.
        """

        # ---------------------------------
        # Materialize inputs
        # ---------------------------------

        sample_ids = list(
            sample_ids
        )

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
            sample_ids
        )

        if not (
            len(integrity_targets)
            == len(integrity_predictions)
            == len(threat_targets)
            == len(threat_predictions)
            == sample_count
        ):
            raise ValueError(
                "All core evaluation arrays "
                "must have equal length."
            )

        if sample_count == 0:
            raise ValueError(
                "Cannot evaluate an empty dataset."
            )

        if integrity_confidences is not None:

            integrity_confidences = list(
                integrity_confidences
            )

            if (
                len(
                    integrity_confidences
                )
                != sample_count
            ):
                raise ValueError(
                    "integrity_confidences must "
                    "match sample count."
                )

        if languages is not None:

            languages = list(
                languages
            )

            if (
                len(languages)
                != sample_count
            ):
                raise ValueError(
                    "languages must match "
                    "sample count."
                )

        if domains is not None:

            domains = list(
                domains
            )

            if (
                len(domains)
                != sample_count
            ):
                raise ValueError(
                    "domains must match "
                    "sample count."
                )

        if source_datasets is not None:

            source_datasets = list(
                source_datasets
            )

            if (
                len(source_datasets)
                != sample_count
            ):
                raise ValueError(
                    "source_datasets must match "
                    "sample count."
                )

        # ---------------------------------
        # Hierarchical + five-class metrics
        # ---------------------------------

        hierarchical = (
            hierarchical_metrics(
                integrity_targets,
                integrity_predictions,
                threat_targets,
                threat_predictions,
            )
        )

        five_class_targets = (
            hierarchical[
                "five_class_targets"
            ]
        )

        five_class_predictions = (
            hierarchical[
                "five_class_predictions"
            ]
        )

        # ---------------------------------
        # Calibration
        # ---------------------------------

        calibration = {}

        if (
            integrity_confidences
            is not None
        ):

            correctness = [
                int(
                    target
                    == prediction
                )
                for target, prediction
                in zip(
                    integrity_targets,
                    integrity_predictions,
                )
            ]

            harmful_probabilities = []

            for (
                prediction,
                confidence,
            ) in zip(
                integrity_predictions,
                integrity_confidences,
            ):

                confidence = float(
                    confidence
                )

                if prediction == 1:

                    harmful_probabilities.append(
                        confidence
                    )

                else:

                    harmful_probabilities.append(
                        1.0
                        - confidence
                    )

            calibration = {
                "integrity_brier_score": (
                    brier_score_binary(
                        harmful_probabilities,
                        integrity_targets,
                    )
                ),

                "integrity_ece": (
                    expected_calibration_error(
                        integrity_confidences,
                        correctness,
                    )
                ),
            }

        # ---------------------------------
        # Stratified evaluation
        # ---------------------------------

        stratified = {}

        if languages is not None:

            stratified[
                "language_integrity"
            ] = (
                stratified_classification_metrics(
                    integrity_targets,
                    integrity_predictions,
                    languages,
                    num_classes=2,
                )
            )

        if domains is not None:

            stratified[
                "domain_integrity"
            ] = (
                stratified_classification_metrics(
                    integrity_targets,
                    integrity_predictions,
                    domains,
                    num_classes=2,
                )
            )

        if source_datasets is not None:

            stratified[
                "source_dataset_integrity"
            ] = (
                stratified_classification_metrics(
                    integrity_targets,
                    integrity_predictions,
                    source_datasets,
                    num_classes=2,
                )
            )

        # ---------------------------------
        # Error analysis
        # ---------------------------------

        errors = collect_errors(
            sample_ids=(
                sample_ids
            ),

            true_labels=(
                five_class_targets
            ),

            predicted_labels=(
                five_class_predictions
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

            confidences=(
                integrity_confidences
            ),
        )

        # ---------------------------------
        # Report
        # ---------------------------------

        return EvaluationReport(
            experiment_name=(
                experiment_name
            ),

            total_samples=(
                sample_count
            ),

            metrics={
                "hierarchical": (
                    hierarchical
                ),

                "five_class": (
                    hierarchical[
                        "five_class"
                    ]
                ),

                "calibration": (
                    calibration
                ),
            },

            stratified_metrics=(
                stratified
            ),

            errors=(
                errors
            ),

            metadata={
                "evaluation_version": (
                    "0.20.0"
                ),

                "five_class_evaluation": True,
            },
        )