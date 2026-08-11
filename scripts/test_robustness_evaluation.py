"""
AEGIS v0.23.0
Robustness, Calibration & Uncertainty Evaluation
sanity experiment.

All performance measurements in this script
are synthetic framework-validation values.
"""

from aegis.robustness import (
    RobustnessEvaluationReport,
    binary_brier_score,
    confidence_degradation,
    confidence_error_analysis,
    expected_calibration_error,
    hierarchical_uncertainty,
    maximum_calibration_error,
    negative_log_likelihood,
    performance_degradation,
    prediction_consistency,
    prediction_uncertainty,
    reliability_bins,
)


print(
    "AEGIS Robustness, Calibration & "
    "Uncertainty Evaluation"
)

print(
    "=" * 80
)

print(
    "IMPORTANT: Metrics in this sanity experiment "
    "are synthetic framework-validation values."
)


targets = [
    0,
    1,
    1,
    0,
    1,
]


predictions = [
    0,
    1,
    0,
    0,
    1,
]


confidences = [
    0.95,
    0.88,
    0.83,
    0.79,
    0.91,
]


correct = [
    int(
        target
        == prediction
    )
    for target, prediction
    in zip(
        targets,
        predictions,
    )
]


harmful_probabilities = [
    0.05,
    0.88,
    0.17,
    0.21,
    0.91,
]


true_class_probabilities = [
    0.95,
    0.88,
    0.17,
    0.79,
    0.91,
]


ece = expected_calibration_error(
    confidences,
    correct,
    num_bins=5,
)


mce = maximum_calibration_error(
    confidences,
    correct,
    num_bins=5,
)


brier = binary_brier_score(
    harmful_probabilities,
    targets,
)


nll = negative_log_likelihood(
    true_class_probabilities
)


print(
    "\nCALIBRATION"
)

print(
    "-" * 80
)

print(
    "ECE:",
    round(
        ece,
        4,
    ),
)

print(
    "MCE:",
    round(
        mce,
        4,
    ),
)

print(
    "Brier score:",
    round(
        brier,
        4,
    ),
)

print(
    "Negative log likelihood:",
    round(
        nll,
        4,
    ),
)


print(
    "\nRELIABILITY BINS"
)

print(
    "-" * 80
)


for item in reliability_bins(
    confidences,
    correct,
    num_bins=5,
):

    if item[
        "count"
    ] == 0:
        continue

    print(
        (
            f"Bin {item['bin_index']} "
            f"| count: {item['count']} "
            f"| confidence: "
            f"{item['mean_confidence']:.4f} "
            f"| accuracy: "
            f"{item['accuracy']:.4f} "
            f"| gap: "
            f"{item['calibration_gap']:.4f}"
        )
    )


low_uncertainty = (
    prediction_uncertainty(
        [
            0.97,
            0.01,
            0.01,
            0.005,
            0.005,
        ]
    )
)


high_uncertainty = (
    prediction_uncertainty(
        [
            0.22,
            0.21,
            0.20,
            0.19,
            0.18,
        ]
    )
)


print(
    "\nUNCERTAINTY"
)

print(
    "-" * 80
)

print(
    "Confident prediction:"
)

print(
    "  Confidence:",
    round(
        low_uncertainty.confidence,
        4,
    ),
)

print(
    "  Normalized entropy:",
    round(
        low_uncertainty
        .normalized_entropy,
        4,
    ),
)

print(
    "  Margin:",
    round(
        low_uncertainty
        .confidence_margin,
        4,
    ),
)


print(
    "\nAmbiguous prediction:"
)

print(
    "  Confidence:",
    round(
        high_uncertainty.confidence,
        4,
    ),
)

print(
    "  Normalized entropy:",
    round(
        high_uncertainty
        .normalized_entropy,
        4,
    ),
)

print(
    "  Margin:",
    round(
        high_uncertainty
        .confidence_margin,
        4,
    ),
)


hierarchical = (
    hierarchical_uncertainty(
        integrity_probabilities=[
            0.06,
            0.94,
        ],

        threat_probabilities=[
            0.10,
            0.61,
            0.18,
            0.11,
        ],
    )
)


print(
    "\nHIERARCHICAL UNCERTAINTY"
)

print(
    "-" * 80
)

print(
    "Integrity confidence:",
    round(
        hierarchical
        .integrity_confidence,
        4,
    ),
)

print(
    "Threat confidence:",
    round(
        hierarchical
        .threat_confidence,
        4,
    ),
)

print(
    "Overall confidence:",
    round(
        hierarchical
        .overall_confidence,
        4,
    ),
)

print(
    "Overall uncertainty:",
    round(
        hierarchical
        .overall_uncertainty,
        4,
    ),
)

print(
    "Uncertainty level:",
    hierarchical
    .uncertainty_level,
)


robustness = (
    performance_degradation(
        clean_metric=0.82,
        perturbed_metric=0.76,
    )
)


consistency = (
    prediction_consistency(
        clean_predictions=[
            0,
            1,
            2,
            3,
            4,
        ],

        perturbed_predictions=[
            0,
            1,
            2,
            0,
            4,
        ],
    )
)


confidence_change = (
    confidence_degradation(
        clean_confidences=[
            0.95,
            0.88,
            0.84,
            0.90,
            0.92,
        ],

        perturbed_confidences=[
            0.89,
            0.79,
            0.76,
            0.71,
            0.86,
        ],
    )
)


print(
    "\nROBUSTNESS"
)

print(
    "-" * 80
)

print(
    "Clean Macro F1:",
    robustness[
        "clean"
    ],
)

print(
    "Perturbed Macro F1:",
    robustness[
        "perturbed"
    ],
)

print(
    "Absolute degradation:",
    round(
        robustness[
            "absolute_degradation"
        ],
        4,
    ),
)

print(
    "Relative degradation:",
    (
        f"{robustness['relative_degradation_percent']:.2f}%"
    ),
)

print(
    "Prediction consistency:",
    round(
        consistency,
        4,
    ),
)

print(
    "Confidence change:",
    round(
        confidence_change[
            "absolute_change"
        ],
        4,
    ),
)


confidence_analysis = (
    confidence_error_analysis(
        targets=[
            0,
            1,
            2,
            3,
            4,
        ],

        predictions=[
            0,
            1,
            1,
            3,
            4,
        ],

        confidences=[
            0.95,
            0.72,
            0.93,
            0.65,
            0.91,
        ],

        high_confidence_threshold=(
            0.80
        ),
    )
)


print(
    "\nCONFIDENCE-AWARE ERROR ANALYSIS"
)

print(
    "-" * 80
)


for key, value in (
    confidence_analysis.items()
):

    print(
        key,
        ":",
        (
            round(
                value,
                4,
            )
            if isinstance(
                value,
                float,
            )
            else value
        ),
    )


report = (
    RobustnessEvaluationReport(
        experiment_name=(
            "robustness_sanity"
        ),

        calibration={
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
            "negative_log_likelihood": (
                nll
            ),
        },

        uncertainty={
            "confident": (
                low_uncertainty
                .as_dict()
            ),

            "ambiguous": (
                high_uncertainty
                .as_dict()
            ),

            "hierarchical": (
                hierarchical
                .as_dict()
            ),
        },

        robustness={
            "performance": (
                robustness
            ),

            "prediction_consistency": (
                consistency
            ),

            "confidence": (
                confidence_change
            ),
        },

        confidence_analysis=(
            confidence_analysis
        ),

        metadata={
            "version": "0.23.0",

            "synthetic_validation": True,
        },
    )
)


print(
    "\nREPORT CREATED:",
    report.experiment_name,
)


print(
    "\nAEGIS robustness, calibration and uncertainty "
    "framework is operational."
)