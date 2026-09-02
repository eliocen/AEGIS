"""
AEGIS v0.24.0
Fakeddit Alignment-Loss Ablation Runner
=======================================

Runs a controlled alignment-weight ablation by invoking the existing
Fakeddit generalization runner under identical settings except for
alignment loss weight.

The official Fakeddit test split remains sealed.

Default alignment weights:
    0.0
    0.1
    0.5
    1.0

Primary comparison metric:
    validation Macro-F1

Tie/context metrics:
    validation classification loss
    accuracy
    precision
    recall
    positive-class F1
    confusion matrix
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from pathlib import Path
from statistics import mean
from typing import Dict, List


DEFAULT_TRAIN_CACHE = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings/"
    "train_n800_seed42"
)

DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings/"
    "validation_n100_seed42"
)

DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/"
    "alignment_ablation_seed42"
)

DEFAULT_WEIGHTS = [
    0.0,
    0.1,
    0.5,
    1.0,
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run controlled Fakeddit alignment-loss "
            "ablation experiments."
        )
    )

    parser.add_argument(
        "--train-cache",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
    )

    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=DEFAULT_WEIGHTS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--shared-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=5,
    )

    return parser.parse_args()


def weight_name(
    weight: float,
) -> str:

    text = (
        f"{weight:g}"
        .replace(
            ".",
            "p"
        )
    )

    return (
        f"align_{text}"
    )


def load_summary(
    experiment_root: Path,
) -> Dict:

    path = (
        experiment_root
        / "summary.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing summary file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def extract_result(
    weight: float,
    experiment_root: Path,
    summary: Dict,
) -> Dict:

    best = (
        summary[
            "best_validation"
        ]
    )

    confusion = (
        best[
            "confusion_matrix"
        ]
    )

    return {
        "alignment_weight": (
            weight
        ),

        "experiment_root": str(
            experiment_root
        ),

        "best_epoch": (
            summary[
                "best_epoch"
            ]
        ),

        "epochs_completed": (
            summary[
                "epochs_completed"
            ]
        ),

        "stop_reason": (
            summary[
                "stop_reason"
            ]
        ),

        "accuracy": (
            best[
                "accuracy"
            ]
        ),

        "precision": (
            best[
                "precision"
            ]
        ),

        "recall": (
            best[
                "recall"
            ]
        ),

        "f1": (
            best[
                "f1"
            ]
        ),

        "macro_f1": (
            best[
                "macro_f1"
            ]
        ),

        "classification_loss": (
            best[
                "classification_loss"
            ]
        ),

        "alignment_loss": (
            best[
                "alignment_loss"
            ]
        ),

        "tn": (
            confusion[
                "tn"
            ]
        ),

        "fp": (
            confusion[
                "fp"
            ]
        ),

        "fn": (
            confusion[
                "fn"
            ]
        ),

        "tp": (
            confusion[
                "tp"
            ]
        ),
    }


def run_experiment(
    args,
    weight: float,
) -> Dict:

    experiment_root = (
        args.output_root
        / weight_name(
            weight
        )
    )

    command = [
        sys.executable,
        "-m",
        "scripts.run_fakeddit_generalization",

        "--train-cache",
        str(
            args.train_cache
        ),

        "--validation-cache",
        str(
            args.validation_cache
        ),

        "--experiment-root",
        str(
            experiment_root
        ),

        "--epochs",
        str(
            args.epochs
        ),

        "--batch-size",
        str(
            args.batch_size
        ),

        "--seed",
        str(
            args.seed
        ),

        "--learning-rate",
        str(
            args.learning_rate
        ),

        "--weight-decay",
        str(
            args.weight_decay
        ),

        "--shared-dim",
        str(
            args.shared_dim
        ),

        "--hidden-dim",
        str(
            args.hidden_dim
        ),

        "--dropout",
        str(
            args.dropout
        ),

        "--alignment-weight",
        str(
            weight
        ),

        "--classification-weight",
        str(
            args.classification_weight
        ),

        "--patience",
        str(
            args.patience
        ),

        "--min-epochs",
        str(
            args.min_epochs
        ),
    ]

    print()
    print(
        "=" * 78
    )

    print(
        f"RUNNING ALIGNMENT WEIGHT: {weight}"
    )

    print(
        "=" * 78
    )

    print(
        "Experiment root:",
        experiment_root
    )

    print()

    subprocess.run(
        command,
        check=True,
    )

    summary = (
        load_summary(
            experiment_root
        )
    )

    return (
        extract_result(
            weight=weight,
            experiment_root=(
                experiment_root
            ),
            summary=summary,
        )
    )


def choose_best(
    results: List[
        Dict
    ],
) -> Dict:

    if not results:

        raise ValueError(
            "No ablation results available."
        )

    return sorted(
        results,
        key=lambda row: (
            -row[
                "macro_f1"
            ],
            row[
                "classification_loss"
            ],
        ),
    )[0]


def save_json(
    path: Path,
    payload,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def save_csv(
    path: Path,
    rows: List[
        Dict
    ],
):

    import csv

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = (
            csv.DictWriter(
                handle,
                fieldnames=(
                    fieldnames
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_table(
    results: List[
        Dict
    ],
):

    print()

    print(
        "=" * 110
    )

    print(
        "AEGIS ALIGNMENT ABLATION RESULTS"
    )

    print(
        "=" * 110
    )

    header = (
        f"{'λ_align':>8} "
        f"{'Acc':>8} "
        f"{'Prec':>8} "
        f"{'Recall':>8} "
        f"{'F1':>8} "
        f"{'MacroF1':>10} "
        f"{'ValCls':>10} "
        f"{'Epoch':>7}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for result in sorted(
        results,
        key=lambda row: (
            row[
                "alignment_weight"
            ]
        ),
    ):

        print(
            f"{result['alignment_weight']:>8.2f} "
            f"{result['accuracy']:>8.4f} "
            f"{result['precision']:>8.4f} "
            f"{result['recall']:>8.4f} "
            f"{result['f1']:>8.4f} "
            f"{result['macro_f1']:>10.4f} "
            f"{result['classification_loss']:>10.4f} "
            f"{result['best_epoch']:>7}"
        )

    print()


def main():

    args = parse_args()

    if not args.train_cache.is_dir():

        raise FileNotFoundError(
            f"Train cache missing: "
            f"{args.train_cache.resolve()}"
        )

    if not args.validation_cache.is_dir():

        raise FileNotFoundError(
            f"Validation cache missing: "
            f"{args.validation_cache.resolve()}"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for weight in (
        args.weights
    ):

        result = (
            run_experiment(
                args=args,
                weight=weight,
            )
        )

        results.append(
            result
        )

    best = (
        choose_best(
            results
        )
    )

    print_table(
        results
    )

    output_json = (
        args.output_root
        / "alignment_ablation_summary.json"
    )

    output_csv = (
        args.output_root
        / "alignment_ablation_summary.csv"
    )

    payload = {
        "experiment": (
            "AEGIS Fakeddit "
            "alignment-loss ablation"
        ),

        "dataset": (
            "Fakeddit"
        ),

        "protocol": (
            "train800-validation100"
        ),

        "test_split_status": (
            "sealed_not_accessed"
        ),

        "selection_metric": (
            "validation_macro_f1"
        ),

        "selection_tie_breaker": (
            "validation_classification_loss"
        ),

        "seed": (
            args.seed
        ),

        "alignment_weights": (
            args.weights
        ),

        "results": (
            results
        ),

        "best": (
            best
        ),
    }

    save_json(
        output_json,
        payload,
    )

    save_csv(
        output_csv,
        results,
    )

    print(
        "=" * 78
    )

    print(
        "BEST ALIGNMENT CONFIGURATION"
    )

    print(
        "=" * 78
    )

    print(
        "Alignment weight:",
        best[
            "alignment_weight"
        ],
    )

    print(
        "Validation Macro-F1:",
        round(
            best[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Validation accuracy:",
        round(
            best[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Validation F1:",
        round(
            best[
                "f1"
            ],
            4,
        ),
    )

    print(
        "Validation classification loss:",
        round(
            best[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Best epoch:",
        best[
            "best_epoch"
        ],
    )

    print()

    print(
        "Official Fakeddit test split:",
        "SEALED / NOT ACCESSED"
    )

    print()

    print(
        "Summary JSON:",
        output_json
    )

    print(
        "Summary CSV:",
        output_csv
    )


if __name__ == "__main__":
    main()