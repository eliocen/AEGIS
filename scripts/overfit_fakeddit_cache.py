"""
AEGIS v0.24.0
Real Fakeddit FD-32 Overfit Sanity Experiment
=============================================

Purpose
-------
Verify that the trainable AEGIS multimodal pathway can learn a very
small real Fakeddit dataset from frozen XLM-R and CLIP representations.

This is an optimization sanity experiment, NOT a generalization result.

The same cached examples are used for training and evaluation.

Pipeline
--------
Frozen Fakeddit representations
    -> CrossModalAlignmentModel
    -> GatedMultimodalFusion
    -> AEGIS integrity head
    -> BinaryIntegrityLoss
    -> optimization

Important
---------
The Fakeddit binary task provides Stage-1 integrity supervision only.

No misinformation, disinformation, malinformation, or hate-speech
subtype targets are manufactured.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from aegis.alignment import (
    CrossModalAlignmentModel,
)

from aegis.classification import (
    HierarchicalInformationIntegrityClassifier,
)

from aegis.data.cache import (
    FakedditRepresentationCache,
)

from aegis.training import (
    BinaryIntegrityTrainer,
    set_global_seed,
)


DEFAULT_CACHE_ROOT = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings/"
    "train_n32_seed42"
)

DEFAULT_EXPERIMENT_ROOT = Path(
    "experiments/fakeddit/"
    "fd32_overfit_seed42"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run an AEGIS real-data overfit sanity "
            "experiment using cached Fakeddit representations."
        )
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
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
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
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
        default=0.0,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    return parser.parse_args()


def separator():

    print(
        "=" * 78
    )


def validate_args(
    args,
):

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay must be >= 0."
        )

    if args.shared_dim <= 0:
        raise ValueError(
            "--shared-dim must be greater than zero."
        )

    if args.hidden_dim <= 0:
        raise ValueError(
            "--hidden-dim must be greater than zero."
        )

    if not (
        0.0
        <= args.dropout
        < 1.0
    ):
        raise ValueError(
            "--dropout must be in [0, 1)."
        )

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be greater than zero."
        )

    if args.alignment_weight < 0:
        raise ValueError(
            "--alignment-weight must be >= 0."
        )

    if args.classification_weight <= 0:
        raise ValueError(
            "--classification-weight must be > 0."
        )

    if not (
        0.0
        < args.target_accuracy
        <= 1.0
    ):
        raise ValueError(
            "--target-accuracy must be in (0, 1]."
        )

    if args.log_every <= 0:
        raise ValueError(
            "--log-every must be greater than zero."
        )

    if not args.cache_root.is_dir():
        raise FileNotFoundError(
            f"Cache root does not exist: "
            f"{args.cache_root.resolve()}"
        )


def resolve_device():

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def count_parameters(
    module,
):

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


def load_all_batches(
    cache,
    batch_size,
):

    batches = list(
        cache.iter_batches(
            batch_size=batch_size
        )
    )

    if not batches:
        raise RuntimeError(
            "No batches were loaded from the cache."
        )

    return batches


def collect_predictions(
    trainer,
    batches,
):

    trainer.alignment_model.eval()
    trainer.classification_model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in batches:

            outputs = trainer.forward_batch(
                batch,
                compute_alignment_loss=False,
            )

            logits = (
                outputs[
                    "classification_outputs"
                ][
                    "integrity_logits"
                ]
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            targets = (
                batch.integrity_targets
                .to(
                    logits.device
                )
            )

            batch_size = int(
                targets.numel()
            )

            loss_value = float(
                outputs[
                    "classification_loss"
                ]
                .detach()
                .cpu()
                .item()
            )

            total_loss += (
                loss_value
                * batch_size
            )

            total_samples += (
                batch_size
            )

            all_targets.extend(
                targets
                .detach()
                .cpu()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .detach()
                .cpu()
                .tolist()
            )

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation contained zero samples."
        )

    return {
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "classification_loss": (
            total_loss
            / total_samples
        ),
    }


def binary_metrics(
    targets,
    predictions,
):

    if len(targets) != len(
        predictions
    ):
        raise ValueError(
            "targets and predictions "
            "must have equal lengths."
        )

    if not targets:
        raise ValueError(
            "Cannot compute metrics "
            "from zero observations."
        )

    tp = sum(
        1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
        if (
            target == 1
            and prediction == 1
        )
    )

    tn = sum(
        1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
        if (
            target == 0
            and prediction == 0
        )
    )

    fp = sum(
        1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
        if (
            target == 0
            and prediction == 1
        )
    )

    fn = sum(
        1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
        if (
            target == 1
            and prediction == 0
        )
    )

    total = len(
        targets
    )

    accuracy = (
        tp + tn
    ) / total

    precision = (
        tp
        / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def save_json(
    path,
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


def save_checkpoint(
    path,
    trainer,
    epoch,
    metrics,
    args,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,

            "alignment_model_state_dict": (
                trainer
                .alignment_model
                .state_dict()
            ),

            "classification_model_state_dict": (
                trainer
                .classification_model
                .state_dict()
            ),

            "optimizer_state_dict": (
                trainer
                .optimizer
                .state_dict()
            ),

            "training_state": {
                "global_step": (
                    trainer
                    .state
                    .global_step
                ),
            },

            "metrics": metrics,

            "configuration": {
                "seed": args.seed,
                "learning_rate": (
                    args.learning_rate
                ),
                "weight_decay": (
                    args.weight_decay
                ),
                "shared_dim": (
                    args.shared_dim
                ),
                "hidden_dim": (
                    args.hidden_dim
                ),
                "dropout": (
                    args.dropout
                ),
                "temperature": (
                    args.temperature
                ),
                "alignment_weight": (
                    args.alignment_weight
                ),
                "classification_weight": (
                    args.classification_weight
                ),
            },
        },
        path,
    )


def main():

    args = parse_args()

    validate_args(
        args
    )

    set_global_seed(
        args.seed
    )

    device = (
        resolve_device()
    )

    args.experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        args.experiment_root
        / "metrics.jsonl"
    )

    best_model_path = (
        args.experiment_root
        / "best_model.pt"
    )

    final_model_path = (
        args.experiment_root
        / "final_model.pt"
    )

    summary_path = (
        args.experiment_root
        / "summary.json"
    )

    experiment_path = (
        args.experiment_root
        / "experiment.json"
    )

    separator()

    print(
        "AEGIS FD-32 REAL OVERFIT "
        "SANITY EXPERIMENT"
    )

    separator()

    print(
        "Cache root:",
        args.cache_root.resolve(),
    )

    print(
        "Experiment root:",
        args.experiment_root.resolve(),
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Device:",
        device,
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA build:",
        torch.version.cuda,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "GPU memory:",
            (
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            ),
        )

    print()

    separator()

    print(
        "LOADING FROZEN "
        "REPRESENTATION CACHE"
    )

    separator()

    cache = (
        FakedditRepresentationCache(
            args.cache_root
        )
    )

    batches = (
        load_all_batches(
            cache=cache,
            batch_size=(
                args.batch_size
            ),
        )
    )

    print(
        "Cached samples:",
        cache.sample_count,
    )

    print(
        "Text dimension:",
        cache.text_dimension,
    )

    print(
        "Vision dimension:",
        cache.vision_dimension,
    )

    print(
        "Training batches:",
        len(
            batches
        ),
    )

    target_counts = Counter()

    for batch in batches:

        target_counts.update(
            batch.integrity_targets.tolist()
        )

    print(
        "Integrity target counts:",
        dict(
            sorted(
                target_counts.items()
            )
        ),
    )

    print()

    separator()

    print(
        "INITIALIZING TRAINABLE AEGIS"
    )

    separator()

    alignment_model = (
        CrossModalAlignmentModel(
            text_dim=(
                cache.text_dimension
            ),
            vision_dim=(
                cache.vision_dimension
            ),
            shared_dim=(
                args.shared_dim
            ),
            temperature=(
                args.temperature
            ),
        )
    )

    classification_model = (
        HierarchicalInformationIntegrityClassifier(
            input_dim=(
                args.shared_dim
            ),
            hidden_dim=(
                args.hidden_dim
            ),
            dropout=(
                args.dropout
            ),
        )
    )

    trainable_parameters = (
        list(
            alignment_model.parameters()
        )
        +
        list(
            classification_model.parameters()
        )
    )

    optimizer = (
        torch.optim.AdamW(
            trainable_parameters,
            lr=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    trainer = (
        BinaryIntegrityTrainer(
            alignment_model=(
                alignment_model
            ),
            classification_model=(
                classification_model
            ),
            optimizer=(
                optimizer
            ),
            device=(
                device
            ),
            alignment_loss_weight=(
                args.alignment_weight
            ),
            classification_loss_weight=(
                args.classification_weight
            ),
            gradient_clip_norm=1.0,
        )
    )

    alignment_parameters = (
        count_parameters(
            alignment_model
        )
    )

    classifier_parameters = (
        count_parameters(
            classification_model
        )
    )

    total_parameters = (
        alignment_parameters
        +
        classifier_parameters
    )

    print(
        "Alignment parameters:",
        alignment_parameters,
    )

    print(
        "Classifier parameters:",
        classifier_parameters,
    )

    print(
        "Total trainable parameters:",
        total_parameters,
    )

    print(
        "Alignment loss weight:",
        args.alignment_weight,
    )

    print(
        "Classification loss weight:",
        args.classification_weight,
    )

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print()

    experiment_metadata = {
        "experiment": (
            "AEGIS-FD-32-overfit"
        ),

        "purpose": (
            "real-data optimization sanity test"
        ),

        "is_generalization_result": False,

        "dataset": (
            "Fakeddit"
        ),

        "task": (
            "binary_integrity"
        ),

        "cache_root": str(
            args.cache_root.resolve()
        ),

        "sample_count": (
            cache.sample_count
        ),

        "cache_manifest": (
            cache.manifest
        ),

        "seed": (
            args.seed
        ),

        "epochs": (
            args.epochs
        ),

        "batch_size": (
            args.batch_size
        ),

        "learning_rate": (
            args.learning_rate
        ),

        "weight_decay": (
            args.weight_decay
        ),

        "shared_dim": (
            args.shared_dim
        ),

        "hidden_dim": (
            args.hidden_dim
        ),

        "dropout": (
            args.dropout
        ),

        "temperature": (
            args.temperature
        ),

        "alignment_weight": (
            args.alignment_weight
        ),

        "classification_weight": (
            args.classification_weight
        ),

        "target_accuracy": (
            args.target_accuracy
        ),

        "trainable_parameters": {
            "alignment": (
                alignment_parameters
            ),

            "classification": (
                classifier_parameters
            ),

            "total": (
                total_parameters
            ),
        },

        "environment": {
            "python": (
                sys.version
            ),

            "platform": (
                platform.platform()
            ),

            "pytorch": (
                torch.__version__
            ),

            "cuda_build": (
                torch.version.cuda
            ),

            "cuda_available": (
                torch.cuda.is_available()
            ),

            "device": (
                device
            ),

            "gpu": (
                torch.cuda.get_device_name(
                    0
                )
                if torch.cuda.is_available()
                else None
            ),
        },

        "started_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    save_json(
        experiment_path,
        experiment_metadata,
    )

    separator()

    print(
        "BASELINE BEFORE TRAINING"
    )

    separator()

    initial_outputs = (
        collect_predictions(
            trainer,
            batches,
        )
    )

    initial_metrics = (
        binary_metrics(
            initial_outputs[
                "targets"
            ],
            initial_outputs[
                "predictions"
            ],
        )
    )

    initial_metrics[
        "classification_loss"
    ] = (
        initial_outputs[
            "classification_loss"
        ]
    )

    print(
        "Initial loss:",
        round(
            initial_metrics[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Initial accuracy:",
        round(
            initial_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Initial F1:",
        round(
            initial_metrics[
                "f1"
            ],
            4,
        ),
    )

    print()

    separator()

    print(
        "TRAINING"
    )

    separator()

    best_accuracy = (
        initial_metrics[
            "accuracy"
        ]
    )

    best_loss = (
        initial_metrics[
            "classification_loss"
        ]
    )

    best_epoch = 0

    reached_target_epoch = None

    start_time = (
        time.perf_counter()
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_handle:

        for epoch in range(
            1,
            args.epochs + 1,
        ):

            epoch_losses = []
            epoch_alignment_losses = []
            epoch_classification_losses = []

            for batch in batches:

                result = (
                    trainer.train_step(
                        batch
                    )
                )

                epoch_losses.append(
                    result[
                        "loss"
                    ]
                )

                epoch_alignment_losses.append(
                    result[
                        "alignment_loss"
                    ]
                )

                epoch_classification_losses.append(
                    result[
                        "classification_loss"
                    ]
                )

            evaluation = (
                collect_predictions(
                    trainer,
                    batches,
                )
            )

            metrics = (
                binary_metrics(
                    evaluation[
                        "targets"
                    ],
                    evaluation[
                        "predictions"
                    ],
                )
            )

            metrics.update(
                {
                    "epoch": (
                        epoch
                    ),

                    "global_step": (
                        trainer
                        .state
                        .global_step
                    ),

                    "mean_training_loss": (
                        sum(
                            epoch_losses
                        )
                        / len(
                            epoch_losses
                        )
                    ),

                    "mean_alignment_loss": (
                        sum(
                            epoch_alignment_losses
                        )
                        / len(
                            epoch_alignment_losses
                        )
                    ),

                    "mean_classification_loss": (
                        sum(
                            epoch_classification_losses
                        )
                        / len(
                            epoch_classification_losses
                        )
                    ),

                    "evaluation_classification_loss": (
                        evaluation[
                            "classification_loss"
                        ]
                    ),
                }
            )

            metrics_handle.write(
                json.dumps(
                    metrics,
                    ensure_ascii=False,
                )
                + "\n"
            )

            metrics_handle.flush()

            accuracy = (
                metrics[
                    "accuracy"
                ]
            )

            current_loss = (
                metrics[
                    "evaluation_classification_loss"
                ]
            )

            improved = (
                accuracy
                > best_accuracy
                or (
                    accuracy
                    == best_accuracy
                    and current_loss
                    < best_loss
                )
            )

            if improved:

                best_accuracy = (
                    accuracy
                )

                best_loss = (
                    current_loss
                )

                best_epoch = (
                    epoch
                )

                save_checkpoint(
                    path=(
                        best_model_path
                    ),
                    trainer=(
                        trainer
                    ),
                    epoch=(
                        epoch
                    ),
                    metrics=(
                        metrics
                    ),
                    args=(
                        args
                    ),
                )

            if (
                reached_target_epoch
                is None
                and accuracy
                >= args.target_accuracy
            ):

                reached_target_epoch = (
                    epoch
                )

            if (
                epoch == 1
                or epoch
                % args.log_every
                == 0
                or accuracy
                >= args.target_accuracy
                or epoch
                == args.epochs
            ):

                print(
                    f"Epoch {epoch:03d} | "
                    f"loss={current_loss:.6f} | "
                    f"acc={accuracy:.4f} | "
                    f"f1={metrics['f1']:.4f} | "
                    f"best={best_accuracy:.4f}"
                )

            if (
                accuracy >= 1.0
                and current_loss < 0.01
            ):

                print()
                print(
                    "Perfect training classification "
                    "with very low loss reached."
                )

                break

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    separator()

    print(
        "FINAL EVALUATION"
    )

    separator()

    final_outputs = (
        collect_predictions(
            trainer,
            batches,
        )
    )

    final_metrics = (
        binary_metrics(
            final_outputs[
                "targets"
            ],
            final_outputs[
                "predictions"
            ],
        )
    )

    final_metrics[
        "classification_loss"
    ] = (
        final_outputs[
            "classification_loss"
        ]
    )

    final_epoch = (
        epoch
    )

    save_checkpoint(
        path=(
            final_model_path
        ),
        trainer=(
            trainer
        ),
        epoch=(
            final_epoch
        ),
        metrics=(
            final_metrics
        ),
        args=(
            args
        ),
    )

    passed = (
        best_accuracy
        >= args.target_accuracy
    )

    summary = {
        "experiment": (
            "AEGIS-FD-32-overfit"
        ),

        "status": (
            "PASSED"
            if passed
            else "FAILED"
        ),

        "is_generalization_result": False,

        "sample_count": (
            cache.sample_count
        ),

        "seed": (
            args.seed
        ),

        "epochs_completed": (
            final_epoch
        ),

        "global_steps": (
            trainer
            .state
            .global_step
        ),

        "initial": (
            initial_metrics
        ),

        "final": (
            final_metrics
        ),

        "best_accuracy": (
            best_accuracy
        ),

        "best_loss": (
            best_loss
        ),

        "best_epoch": (
            best_epoch
        ),

        "target_accuracy": (
            args.target_accuracy
        ),

        "target_reached_epoch": (
            reached_target_epoch
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),

        "best_checkpoint": str(
            best_model_path.resolve()
        )
        if best_model_path.exists()
        else None,

        "final_checkpoint": str(
            final_model_path.resolve()
        ),

        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    print(
        "Initial accuracy:",
        round(
            initial_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Final accuracy:",
        round(
            final_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Best accuracy:",
        round(
            best_accuracy,
            4,
        ),
    )

    print(
        "Initial loss:",
        round(
            initial_metrics[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Final loss:",
        round(
            final_metrics[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Target reached epoch:",
        reached_target_epoch,
    )

    print(
        "Elapsed seconds:",
        round(
            elapsed_seconds,
            2,
        ),
    )

    print(
        "Confusion matrix:",
        final_metrics[
            "confusion_matrix"
        ],
    )

    print()

    separator()

    if passed:

        print(
            "AEGIS FD-32 OVERFIT SANITY: PASSED"
        )

    else:

        print(
            "AEGIS FD-32 OVERFIT SANITY: FAILED"
        )

    separator()

    print()

    print(
        "This result measures optimization "
        "on the training samples only."
    )

    print(
        "It must NOT be reported as "
        "Fakeddit generalization performance."
    )

    print()

    print(
        "Experiment metadata:",
        experiment_path,
    )

    print(
        "Epoch metrics:",
        metrics_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Best checkpoint:",
        best_model_path,
    )

    print(
        "Final checkpoint:",
        final_model_path,
    )


if __name__ == "__main__":
    main()