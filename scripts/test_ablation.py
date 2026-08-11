"""
AEGIS v0.21.0
Multi-Seed Ablation Study &
Experimental Comparison sanity experiment.

All metrics in this script are synthetic and are
used exclusively to validate framework behavior.
"""

from aegis.ablation import (
    MultiSeedAblationRunner,
    aggregate_seed_runs,
    build_statistical_comparison_table,
    default_ablation_registry,
)


registry = (
    default_ablation_registry()
)


specs = [
    registry[
        "baseline"
    ],

    registry[
        "text_only"
    ],

    registry[
        "vision_only"
    ],

    registry[
        "no_alignment_loss"
    ],

    registry[
        "reduced_alignment"
    ],

    registry[
        "reduced_classification"
    ],
]


seeds = [
    42,
    123,
    456,
    789,
    2026,
]


BASE_VALUES = {
    "baseline": 0.82,

    "text_only": 0.74,

    "vision_only": 0.65,

    "no_alignment_loss": 0.71,

    "reduced_alignment": 0.78,

    "reduced_classification": 0.76,
}


def synthetic_multi_seed_experiment(
    spec,
    seed,
):
    """
    Deterministic synthetic measurements for
    framework validation.

    These are NOT empirical AEGIS findings.
    """

    seed_effects = {
        42: -0.010,

        123: 0.005,

        456: 0.000,

        789: 0.012,

        2026: -0.007,
    }

    value = (
        BASE_VALUES[
            spec.name
        ]
        + seed_effects[
            seed
        ]
    )

    return {
        "five_class_macro_f1": (
            value
        ),

        "five_class_weighted_f1": (
            min(
                1.0,
                value + 0.02,
            )
        ),

        "hierarchical_exact_accuracy": (
            max(
                0.0,
                value - 0.03,
            )
        ),

        "integrity_macro_f1": (
            min(
                1.0,
                value + 0.08,
            )
        ),

        "threat_macro_f1": (
            max(
                0.0,
                value - 0.05,
            )
        ),
    }


runner = (
    MultiSeedAblationRunner(
        synthetic_multi_seed_experiment
    )
)


seed_results = runner.run(
    specs=specs,
    seeds=seeds,
)


aggregated = (
    aggregate_seed_runs(
        seed_results
    )
)


table = (
    build_statistical_comparison_table(
        aggregated,

        baseline_name=(
            "baseline"
        ),

        metrics=[
            "five_class_macro_f1",
            "hierarchical_exact_accuracy",
            "integrity_macro_f1",
            "threat_macro_f1",
        ],
    )
)


print(
    "AEGIS Multi-Seed Ablation Study"
)

print(
    "=" * 80
)

print(
    "IMPORTANT: Synthetic metrics are used "
    "for framework validation only."
)

print(
    "Seeds:",
    seeds,
)


print(
    "\nSTATISTICAL ABLATION RESULTS"
)

print(
    "-" * 80
)


for row in table:

    print(
        "Variant:",
        row[
            "variant"
        ],
    )

    print(
        "  Runs:",
        row[
            "run_count"
        ],
    )

    print(
        "  Five-Class Macro F1:",
        (
            f"{row['five_class_macro_f1_mean']:.4f}"
            " ± "
            f"{row['five_class_macro_f1_std']:.4f}"
        ),
    )

    print(
        "  Minimum:",
        round(
            row[
                "five_class_macro_f1_min"
            ],
            4,
        ),
    )

    print(
        "  Maximum:",
        round(
            row[
                "five_class_macro_f1_max"
            ],
            4,
        ),
    )

    print(
        "  Δ vs baseline:",
        round(
            row[
                "five_class_macro_f1_delta"
            ],
            4,
        ),
    )

    print(
        "  Hierarchical Accuracy:",
        (
            f"{row['hierarchical_exact_accuracy_mean']:.4f}"
            " ± "
            f"{row['hierarchical_exact_accuracy_std']:.4f}"
        ),
    )

    print(
        "  Integrity Macro F1:",
        (
            f"{row['integrity_macro_f1_mean']:.4f}"
            " ± "
            f"{row['integrity_macro_f1_std']:.4f}"
        ),
    )

    print(
        "  Threat Macro F1:",
        (
            f"{row['threat_macro_f1_mean']:.4f}"
            " ± "
            f"{row['threat_macro_f1_std']:.4f}"
        ),
    )

    print(
        "-" * 80
    )


ranked = sorted(
    table,

    key=lambda row: (
        row[
            "five_class_macro_f1_mean"
        ]
    ),

    reverse=True,
)


print(
    "\nRANKING — MEAN FIVE-CLASS MACRO F1"
)

print(
    "-" * 80
)


for rank, row in enumerate(
    ranked,
    start=1,
):

    print(
        rank,
        row[
            "variant"
        ],
        (
            f"{row['five_class_macro_f1_mean']:.4f}"
            " ± "
            f"{row['five_class_macro_f1_std']:.4f}"
        ),
    )


print(
    "\nMULTI-SEED EXPERIMENT SUMMARY"
)

print(
    "-" * 80
)

print(
    "Variants:",
    len(
        specs
    ),
)

print(
    "Seeds per variant:",
    len(
        seeds
    ),
)

print(
    "Total controlled runs:",
    len(
        seed_results
    ),
)


print(
    "\nAEGIS multi-seed ablation "
    "framework is operational."
)