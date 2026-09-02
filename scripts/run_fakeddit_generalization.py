"""
AEGIS v0.24.0
Fakeddit Train/Validation Generalization Experiment
===================================================

Purpose
-------
Run the first genuine AEGIS empirical generalization experiment:

    official Fakeddit train split
        -> deterministic cached sample
        -> AEGIS training

    official Fakeddit validation split
        -> deterministic cached sample
        -> validation only

The official Fakeddit test split remains SEALED.

This experiment uses frozen XLM-R and CLIP representations and trains
only the downstream AEGIS alignment/fusion/classification components.

Task
----
Binary Stage-1 Information Integrity:

    0 = True / authentic
    1 = Harmful / non-authentic

No misinformation, disinformation, malinformation, or hate-speech
subtype labels are fabricated from Fakeddit.

Checkpoint selection
--------------------
Primary:
    highest validation Macro-F1

Tie-break:
    lowest validation classification loss

Early stopping
--------------
Stop after N consecutive validation epochs without improvement.

Important
---------
Unlike the earlier FD-32 overfit sanity experiment, training and
validation samples are completely separate here.

This therefore measures genuine held-out validation generalization,
although it is still a small balanced development experiment and is
NOT the final Fakeddit benchmark evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    BinaryIntegrityBatch,
    BinaryIntegrityTrainer,
    set_global_seed,
)


# ---------------------------------------------------------------------
# Default empirical protocol
# ---------------------------------------------------------------------

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

DEFAULT_EXPERIMENT_ROOT = Path(
    "experiments/fakeddit/"
    "fd1k_dev_train800_val100_seed42"
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the AEGIS Fakeddit 800-train / "
            "100-validation generalization experiment."
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
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
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
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gradient-clip",
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

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def separator():

    print(
        "=" * 78
    )


def validate_args(
    args,
):

    if not args.train_cache.is_dir():

        raise FileNotFoundError(
            "Training cache does not exist: "
            f"{args.train_cache.resolve()}"
        )

    if not args.validation_cache.is_dir():

        raise FileNotFoundError(
            "Validation cache does not exist: "
            f"{args.validation_cache.resolve()}"
        )

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

    if args.gradient_clip <= 0:

        raise ValueError(
            "--gradient-clip must be greater than zero."
        )

    if args.patience <= 0:

        raise ValueError(
            "--patience must be greater than zero."
        )

    if args.min_epochs < 0:

        raise ValueError(
            "--min-epochs must be >= 0."
        )

    if args.min_delta < 0:

        raise ValueError(
            "--min-delta must be >= 0."
        )

    if args.log_every <= 0:

        raise ValueError(
            "--log-every must be greater than zero."
        )


def resolve_device() -> str:

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def save_json(
    path: Path,
    payload: Dict,
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


def count_parameters(
    module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


# ---------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------

def load_cache_as_single_batch(
    cache: FakedditRepresentationCache,
) -> BinaryIntegrityBatch:
    """
    Load a complete small empirical cache into one CPU batch.

    This is safe for the current 800/100 development experiment and
    greatly simplifies deterministic epoch-level shuffling.

    It does NOT move the full cache to CUDA.
    """

    text_parts = []
    vision_parts = []
    target_parts = []
    sample_ids = []

    for shard in cache.iter_shards():

        text_parts.append(
            shard[
                "text_embeddings"
            ]
            .detach()
            .cpu()
            .float()
        )

        vision_parts.append(
            shard[
                "vision_embeddings"
            ]
            .detach()
            .cpu()
            .float()
        )

        target_parts.append(
            shard[
                "integrity_targets"
            ]
            .detach()
            .cpu()
            .long()
        )

        sample_ids.extend(
            list(
                shard[
                    "sample_ids"
                ]
            )
        )

    if not text_parts:

        raise RuntimeError(
            "Cache contains no representation shards."
        )

    batch = BinaryIntegrityBatch(
        text_embeddings=torch.cat(
            text_parts,
            dim=0,
        ),

        vision_embeddings=torch.cat(
            vision_parts,
            dim=0,
        ),

        integrity_targets=torch.cat(
            target_parts,
            dim=0,
        ),

        sample_ids=sample_ids,
    )

    batch.validate(
        text_dim=cache.text_dimension,
        vision_dim=cache.vision_dimension,
    )

    if (
        batch.batch_size
        != cache.sample_count
    ):

        raise RuntimeError(
            "Cache sample count does not match "
            "loaded representation count."
        )

    return batch


def validate_cache_pair(
    train_cache,
    validation_cache,
    train_data,
    validation_data,
):
    """
    Validate dimensional compatibility and sample separation.
    """

    if (
        train_cache.text_dimension
        != validation_cache.text_dimension
    ):

        raise ValueError(
            "Train and validation text dimensions differ."
        )

    if (
        train_cache.vision_dimension
        != validation_cache.vision_dimension
    ):

        raise ValueError(
            "Train and validation vision dimensions differ."
        )

    train_ids = set(
        train_data.sample_ids
    )

    validation_ids = set(
        validation_data.sample_ids
    )

    overlap = (
        train_ids
        & validation_ids
    )

    if overlap:

        preview = sorted(
            overlap
        )[
            :10
        ]

        raise RuntimeError(
            "Train/validation leakage detected. "
            f"Overlapping sample IDs: {preview}"
        )

    if (
        len(
            train_ids
        )
        != train_data.batch_size
    ):

        raise RuntimeError(
            "Duplicate sample IDs detected "
            "inside training cache."
        )

    if (
        len(
            validation_ids
        )
        != validation_data.batch_size
    ):

        raise RuntimeError(
            "Duplicate sample IDs detected "
            "inside validation cache."
        )


# ---------------------------------------------------------------------
# Mini-batch construction
# ---------------------------------------------------------------------

def iter_training_batches(
    full_batch: BinaryIntegrityBatch,
    batch_size: int,
    epoch: int,
    seed: int,
):
    """
    Yield a deterministic shuffled training order for one epoch.

    Epoch-specific seeding ensures:
        - deterministic reproducibility
        - different sample order each epoch
    """

    generator = (
        torch.Generator(
            device="cpu"
        )
    )

    generator.manual_seed(
        seed
        + epoch
    )

    permutation = (
        torch.randperm(
            full_batch.batch_size,
            generator=generator,
        )
    )

    for start in range(
        0,
        full_batch.batch_size,
        batch_size,
    ):

        indices = (
            permutation[
                start:
                start
                + batch_size
            ]
        )

        yield BinaryIntegrityBatch(
            text_embeddings=(
                full_batch
                .text_embeddings[
                    indices
                ]
            ),

            vision_embeddings=(
                full_batch
                .vision_embeddings[
                    indices
                ]
            ),

            integrity_targets=(
                full_batch
                .integrity_targets[
                    indices
                ]
            ),

            sample_ids=[
                full_batch.sample_ids[
                    int(
                        index
                    )
                ]
                for index
                in indices.tolist()
            ],
        )


def iter_sequential_batches(
    full_batch: BinaryIntegrityBatch,
    batch_size: int,
):
    """
    Deterministic non-shuffled evaluation batching.
    """

    for start in range(
        0,
        full_batch.batch_size,
        batch_size,
    ):

        end = min(
            start
            + batch_size,
            full_batch.batch_size,
        )

        yield BinaryIntegrityBatch(
            text_embeddings=(
                full_batch
                .text_embeddings[
                    start:end
                ]
            ),

            vision_embeddings=(
                full_batch
                .vision_embeddings[
                    start:end
                ]
            ),

            integrity_targets=(
                full_batch
                .integrity_targets[
                    start:end
                ]
            ),

            sample_ids=(
                full_batch.sample_ids[
                    start:end
                ]
            ),
        )


# ---------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:

        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_binary_metrics(
    targets: List[int],
    predictions: List[int],
) -> Dict:
    """
    Compute binary and macro classification metrics.

    Positive class:
        AEGIS integrity target 1
        harmful / non-authentic
    """

    if (
        len(
            targets
        )
        != len(
            predictions
        )
    ):

        raise ValueError(
            "targets and predictions "
            "must have equal lengths."
        )

    if not targets:

        raise ValueError(
            "Cannot calculate metrics "
            "from zero samples."
        )

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for target, prediction in zip(
        targets,
        predictions,
    ):

        if (
            target == 1
            and prediction == 1
        ):
            tp += 1

        elif (
            target == 0
            and prediction == 0
        ):
            tn += 1

        elif (
            target == 0
            and prediction == 1
        ):
            fp += 1

        elif (
            target == 1
            and prediction == 0
        ):
            fn += 1

        else:

            raise ValueError(
                "Binary metrics received "
                "non-binary target/prediction."
            )

    total = len(
        targets
    )

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision_positive = safe_divide(
        tp,
        tp + fp,
    )

    recall_positive = safe_divide(
        tp,
        tp + fn,
    )

    f1_positive = safe_divide(
        2.0
        * precision_positive
        * recall_positive,
        precision_positive
        + recall_positive,
    )

    # Treat class 0 as the positive class
    # to compute symmetric macro metrics.
    precision_negative = safe_divide(
        tn,
        tn + fn,
    )

    recall_negative = safe_divide(
        tn,
        tn + fp,
    )

    f1_negative = safe_divide(
        2.0
        * precision_negative
        * recall_negative,
        precision_negative
        + recall_negative,
    )

    macro_precision = (
        precision_positive
        + precision_negative
    ) / 2.0

    macro_recall = (
        recall_positive
        + recall_negative
    ) / 2.0

    macro_f1 = (
        f1_positive
        + f1_negative
    ) / 2.0

    return {
        "accuracy": (
            accuracy
        ),

        "precision": (
            precision_positive
        ),

        "recall": (
            recall_positive
        ),

        "f1": (
            f1_positive
        ),

        "macro_precision": (
            macro_precision
        ),

        "macro_recall": (
            macro_recall
        ),

        "macro_f1": (
            macro_f1
        ),

        "class_0": {
            "precision": (
                precision_negative
            ),

            "recall": (
                recall_negative
            ),

            "f1": (
                f1_negative
            ),
        },

        "class_1": {
            "precision": (
                precision_positive
            ),

            "recall": (
                recall_positive
            ),

            "f1": (
                f1_positive
            ),
        },

        "confusion_matrix": {
            "tn": (
                tn
            ),

            "fp": (
                fp
            ),

            "fn": (
                fn
            ),

            "tp": (
                tp
            ),
        },
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def evaluate(
    trainer: BinaryIntegrityTrainer,
    full_batch: BinaryIntegrityBatch,
    batch_size: int,
) -> Dict:
    """
    Evaluate without model updates.

    Reports:
        total loss
        alignment loss
        classification loss
        binary metrics
    """

    trainer.alignment_model.eval()

    trainer.classification_model.eval()

    total_samples = 0

    weighted_total_loss = 0.0
    weighted_alignment_loss = 0.0
    weighted_classification_loss = 0.0

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_sample_ids = []

    with torch.no_grad():

        for batch in (
            iter_sequential_batches(
                full_batch=full_batch,
                batch_size=batch_size,
            )
        ):

            outputs = (
                trainer.forward_batch(
                    batch,
                    compute_alignment_loss=True,
                )
            )

            classification_outputs = (
                outputs[
                    "classification_outputs"
                ]
            )

            logits = (
                classification_outputs[
                    "integrity_logits"
                ]
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
            )

            predictions = (
                torch.argmax(
                    probabilities,
                    dim=-1,
                )
            )

            targets = (
                batch
                .integrity_targets
                .to(
                    logits.device
                )
            )

            current_batch_size = (
                int(
                    targets.numel()
                )
            )

            total_samples += (
                current_batch_size
            )

            weighted_total_loss += (
                float(
                    outputs[
                        "loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                )
                * current_batch_size
            )

            weighted_alignment_loss += (
                float(
                    outputs[
                        "alignment_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                )
                * current_batch_size
            )

            weighted_classification_loss += (
                float(
                    outputs[
                        "classification_loss"
                    ]
                    .detach()
                    .cpu()
                    .item()
                )
                * current_batch_size
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

            all_sample_ids.extend(
                batch.sample_ids
            )

    if total_samples == 0:

        raise RuntimeError(
            "Validation contained zero samples."
        )

    metrics = (
        calculate_binary_metrics(
            targets=all_targets,
            predictions=(
                all_predictions
            ),
        )
    )

    metrics.update(
        {
            "sample_count": (
                total_samples
            ),

            "total_loss": (
                weighted_total_loss
                / total_samples
            ),

            "alignment_loss": (
                weighted_alignment_loss
                / total_samples
            ),

            "classification_loss": (
                weighted_classification_loss
                / total_samples
            ),

            "targets": (
                all_targets
            ),

            "predictions": (
                all_predictions
            ),

            "probabilities": (
                all_probabilities
            ),

            "sample_ids": (
                all_sample_ids
            ),
        }
    )

    return metrics


# ---------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------

def train_one_epoch(
    trainer: BinaryIntegrityTrainer,
    full_batch: BinaryIntegrityBatch,
    batch_size: int,
    epoch: int,
    seed: int,
) -> Dict:

    weighted_total_loss = 0.0
    weighted_alignment_loss = 0.0
    weighted_classification_loss = 0.0

    total_samples = 0

    batch_count = 0

    for batch in iter_training_batches(
        full_batch=full_batch,
        batch_size=batch_size,
        epoch=epoch,
        seed=seed,
    ):

        result = (
            trainer.train_step(
                batch
            )
        )

        current_batch_size = (
            batch.batch_size
        )

        total_samples += (
            current_batch_size
        )

        batch_count += 1

        weighted_total_loss += (
            result[
                "loss"
            ]
            * current_batch_size
        )

        weighted_alignment_loss += (
            result[
                "alignment_loss"
            ]
            * current_batch_size
        )

        weighted_classification_loss += (
            result[
                "classification_loss"
            ]
            * current_batch_size
        )

    if total_samples == 0:

        raise RuntimeError(
            "Training epoch contained zero samples."
        )

    return {
        "sample_count": (
            total_samples
        ),

        "batch_count": (
            batch_count
        ),

        "total_loss": (
            weighted_total_loss
            / total_samples
        ),

        "alignment_loss": (
            weighted_alignment_loss
            / total_samples
        ),

        "classification_loss": (
            weighted_classification_loss
            / total_samples
        ),
    }


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    trainer: BinaryIntegrityTrainer,
    epoch: int,
    validation_metrics: Dict,
    configuration: Dict,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": (
                epoch
            ),

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

            "validation_metrics": (
                validation_metrics
            ),

            "configuration": (
                configuration
            ),
        },
        path,
    )


# ---------------------------------------------------------------------
# Best-checkpoint logic
# ---------------------------------------------------------------------

def is_better_checkpoint(
    current_macro_f1: float,
    current_loss: float,
    best_macro_f1: Optional[float],
    best_loss: Optional[float],
    min_delta: float,
) -> bool:
    """
    Primary:
        higher validation Macro-F1

    Tie:
        lower validation classification loss
    """

    if best_macro_f1 is None:

        return True

    if (
        current_macro_f1
        > best_macro_f1
        + min_delta
    ):

        return True

    if (
        abs(
            current_macro_f1
            - best_macro_f1
        )
        <= min_delta
    ):

        if best_loss is None:

            return True

        if (
            current_loss
            < best_loss
            - min_delta
        ):

            return True

    return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    validate_args(
        args
    )

    set_global_seed(
        args.seed
    )

    random.seed(
        args.seed
    )

    device = (
        resolve_device()
    )

    args.experiment_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_path = (
        args.experiment_root
        / "experiment.json"
    )

    metrics_path = (
        args.experiment_root
        / "metrics.jsonl"
    )

    best_checkpoint_path = (
        args.experiment_root
        / "best_model.pt"
    )

    final_checkpoint_path = (
        args.experiment_root
        / "final_model.pt"
    )

    summary_path = (
        args.experiment_root
        / "summary.json"
    )

    predictions_path = (
        args.experiment_root
        / "best_validation_predictions.json"
    )

    # -------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------

    separator()

    print(
        "AEGIS FAKEDDIT GENERALIZATION EXPERIMENT"
    )

    separator()

    print(
        "Train cache:",
        args.train_cache.resolve(),
    )

    print(
        "Validation cache:",
        args.validation_cache.resolve(),
    )

    print(
        "Experiment root:",
        args.experiment_root.resolve(),
    )

    print()

    print(
        "Official test split:",
        "SEALED / NOT ACCESSED",
    )

    print()

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

    # -------------------------------------------------------------
    # Load caches
    # -------------------------------------------------------------

    separator()

    print(
        "LOADING EMPIRICAL CACHES"
    )

    separator()

    train_cache = (
        FakedditRepresentationCache(
            args.train_cache
        )
    )

    validation_cache = (
        FakedditRepresentationCache(
            args.validation_cache
        )
    )

    train_data = (
        load_cache_as_single_batch(
            train_cache
        )
    )

    validation_data = (
        load_cache_as_single_batch(
            validation_cache
        )
    )

    validate_cache_pair(
        train_cache=(
            train_cache
        ),
        validation_cache=(
            validation_cache
        ),
        train_data=(
            train_data
        ),
        validation_data=(
            validation_data
        ),
    )

    train_counts = dict(
        sorted(
            Counter(
                train_data
                .integrity_targets
                .tolist()
            ).items()
        )
    )

    validation_counts = dict(
        sorted(
            Counter(
                validation_data
                .integrity_targets
                .tolist()
            ).items()
        )
    )

    print(
        "Training samples:",
        train_data.batch_size,
    )

    print(
        "Training target counts:",
        train_counts,
    )

    print(
        "Validation samples:",
        validation_data.batch_size,
    )

    print(
        "Validation target counts:",
        validation_counts,
    )

    print(
        "Text dimension:",
        train_cache.text_dimension,
    )

    print(
        "Vision dimension:",
        train_cache.vision_dimension,
    )

    print(
        "Train/validation overlap:",
        0,
    )

    print()

    # -------------------------------------------------------------
    # Models
    # -------------------------------------------------------------

    separator()

    print(
        "INITIALIZING TRAINABLE AEGIS"
    )

    separator()

    alignment_model = (
        CrossModalAlignmentModel(
            text_dim=(
                train_cache.text_dimension
            ),

            vision_dim=(
                train_cache.vision_dimension
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

            gradient_clip_norm=(
                args.gradient_clip
            ),
        )
    )

    alignment_parameter_count = (
        count_parameters(
            alignment_model
        )
    )

    classification_parameter_count = (
        count_parameters(
            classification_model
        )
    )

    total_parameter_count = (
        alignment_parameter_count
        +
        classification_parameter_count
    )

    print(
        "Alignment parameters:",
        alignment_parameter_count,
    )

    print(
        "Classification parameters:",
        classification_parameter_count,
    )

    print(
        "Total trainable parameters:",
        total_parameter_count,
    )

    print()

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Weight decay:",
        args.weight_decay,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Dropout:",
        args.dropout,
    )

    print(
        "Alignment weight:",
        args.alignment_weight,
    )

    print(
        "Classification weight:",
        args.classification_weight,
    )

    print(
        "Maximum epochs:",
        args.epochs,
    )

    print(
        "Early stopping patience:",
        args.patience,
    )

    print(
        "Checkpoint criterion:",
        "validation Macro-F1, then validation classification loss",
    )

    print()

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    configuration = {
        "seed": (
            args.seed
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

        "gradient_clip": (
            args.gradient_clip
        ),

        "max_epochs": (
            args.epochs
        ),

        "patience": (
            args.patience
        ),

        "min_epochs": (
            args.min_epochs
        ),

        "min_delta": (
            args.min_delta
        ),
    }

    experiment_metadata = {
        "experiment": (
            "AEGIS-FD-1K-development"
        ),

        "protocol": (
            "train800-validation100"
        ),

        "task": (
            "binary_integrity"
        ),

        "dataset": (
            "Fakeddit"
        ),

        "is_overfit_sanity_test": False,

        "is_final_test_result": False,

        "test_split_status": (
            "sealed_not_accessed"
        ),

        "checkpoint_selection": {
            "primary_metric": (
                "validation_macro_f1"
            ),

            "tie_breaker": (
                "validation_classification_loss"
            ),
        },

        "train_cache": {
            "path": str(
                args.train_cache.resolve()
            ),

            "manifest": (
                train_cache.manifest
            ),

            "sample_count": (
                train_data.batch_size
            ),

            "target_counts": (
                train_counts
            ),
        },

        "validation_cache": {
            "path": str(
                args.validation_cache.resolve()
            ),

            "manifest": (
                validation_cache.manifest
            ),

            "sample_count": (
                validation_data.batch_size
            ),

            "target_counts": (
                validation_counts
            ),
        },

        "configuration": (
            configuration
        ),

        "trainable_parameters": {
            "alignment": (
                alignment_parameter_count
            ),

            "classification": (
                classification_parameter_count
            ),

            "total": (
                total_parameter_count
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

    # -------------------------------------------------------------
    # Baseline before training
    # -------------------------------------------------------------

    separator()

    print(
        "INITIAL VALIDATION BASELINE"
    )

    separator()

    initial_validation = (
        evaluate(
            trainer=(
                trainer
            ),

            full_batch=(
                validation_data
            ),

            batch_size=(
                args.batch_size
            ),
        )
    )

    print(
        "Validation total loss:",
        round(
            initial_validation[
                "total_loss"
            ],
            6,
        ),
    )

    print(
        "Validation alignment loss:",
        round(
            initial_validation[
                "alignment_loss"
            ],
            6,
        ),
    )

    print(
        "Validation classification loss:",
        round(
            initial_validation[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Validation accuracy:",
        round(
            initial_validation[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Validation F1:",
        round(
            initial_validation[
                "f1"
            ],
            4,
        ),
    )

    print(
        "Validation Macro-F1:",
        round(
            initial_validation[
                "macro_f1"
            ],
            4,
        ),
    )

    print()

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    separator()

    print(
        "TRAINING WITH HELD-OUT VALIDATION"
    )

    separator()

    best_macro_f1 = None
    best_validation_loss = None
    best_epoch = None

    best_validation_metrics = None

    bad_epochs = 0

    stop_reason = (
        "maximum_epochs_reached"
    )

    start_time = (
        time.perf_counter()
    )

    final_epoch = 0

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as metrics_handle:

        for epoch in range(
            1,
            args.epochs + 1,
        ):

            final_epoch = (
                epoch
            )

            epoch_start = (
                time.perf_counter()
            )

            training_metrics = (
                train_one_epoch(
                    trainer=(
                        trainer
                    ),

                    full_batch=(
                        train_data
                    ),

                    batch_size=(
                        args.batch_size
                    ),

                    epoch=(
                        epoch
                    ),

                    seed=(
                        args.seed
                    ),
                )
            )

            validation_metrics = (
                evaluate(
                    trainer=(
                        trainer
                    ),

                    full_batch=(
                        validation_data
                    ),

                    batch_size=(
                        args.batch_size
                    ),
                )
            )

            epoch_seconds = (
                time.perf_counter()
                - epoch_start
            )

            current_macro_f1 = (
                validation_metrics[
                    "macro_f1"
                ]
            )

            current_validation_loss = (
                validation_metrics[
                    "classification_loss"
                ]
            )

            improved = (
                is_better_checkpoint(
                    current_macro_f1=(
                        current_macro_f1
                    ),

                    current_loss=(
                        current_validation_loss
                    ),

                    best_macro_f1=(
                        best_macro_f1
                    ),

                    best_loss=(
                        best_validation_loss
                    ),

                    min_delta=(
                        args.min_delta
                    ),
                )
            )

            if improved:

                best_macro_f1 = (
                    current_macro_f1
                )

                best_validation_loss = (
                    current_validation_loss
                )

                best_epoch = (
                    epoch
                )

                best_validation_metrics = (
                    validation_metrics.copy()
                )

                bad_epochs = 0

                checkpoint_metrics = {
                    key: value
                    for key, value
                    in validation_metrics.items()
                    if key not in {
                        "targets",
                        "predictions",
                        "probabilities",
                        "sample_ids",
                    }
                }

                save_checkpoint(
                    path=(
                        best_checkpoint_path
                    ),

                    trainer=(
                        trainer
                    ),

                    epoch=(
                        epoch
                    ),

                    validation_metrics=(
                        checkpoint_metrics
                    ),

                    configuration=(
                        configuration
                    ),
                )

                save_json(
                    predictions_path,
                    {
                        "epoch": (
                            epoch
                        ),

                        "sample_ids": (
                            validation_metrics[
                                "sample_ids"
                            ]
                        ),

                        "targets": (
                            validation_metrics[
                                "targets"
                            ]
                        ),

                        "predictions": (
                            validation_metrics[
                                "predictions"
                            ]
                        ),

                        "probabilities": (
                            validation_metrics[
                                "probabilities"
                            ]
                        ),
                    },
                )

            else:

                bad_epochs += 1

            epoch_record = {
                "epoch": (
                    epoch
                ),

                "global_step": (
                    trainer
                    .state
                    .global_step
                ),

                "train": {
                    "sample_count": (
                        training_metrics[
                            "sample_count"
                        ]
                    ),

                    "batch_count": (
                        training_metrics[
                            "batch_count"
                        ]
                    ),

                    "total_loss": (
                        training_metrics[
                            "total_loss"
                        ]
                    ),

                    "alignment_loss": (
                        training_metrics[
                            "alignment_loss"
                        ]
                    ),

                    "classification_loss": (
                        training_metrics[
                            "classification_loss"
                        ]
                    ),
                },

                "validation": {
                    key: value
                    for key, value
                    in validation_metrics.items()
                    if key not in {
                        "targets",
                        "predictions",
                        "probabilities",
                        "sample_ids",
                    }
                },

                "checkpoint_improved": (
                    improved
                ),

                "best_validation_macro_f1": (
                    best_macro_f1
                ),

                "best_validation_loss": (
                    best_validation_loss
                ),

                "bad_epochs": (
                    bad_epochs
                ),

                "epoch_seconds": (
                    epoch_seconds
                ),
            }

            metrics_handle.write(
                json.dumps(
                    epoch_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            metrics_handle.flush()

            if (
                epoch == 1
                or epoch
                % args.log_every
                == 0
                or improved
            ):

                marker = (
                    " *BEST*"
                    if improved
                    else ""
                )

                print(
                    f"Epoch {epoch:03d} | "
                    f"train_total="
                    f"{training_metrics['total_loss']:.4f} | "
                    f"train_align="
                    f"{training_metrics['alignment_loss']:.4f} | "
                    f"train_cls="
                    f"{training_metrics['classification_loss']:.4f} | "
                    f"val_total="
                    f"{validation_metrics['total_loss']:.4f} | "
                    f"val_cls="
                    f"{validation_metrics['classification_loss']:.4f} | "
                    f"acc="
                    f"{validation_metrics['accuracy']:.4f} | "
                    f"f1="
                    f"{validation_metrics['f1']:.4f} | "
                    f"macro_f1="
                    f"{validation_metrics['macro_f1']:.4f}"
                    f"{marker}"
                )

            if (
                epoch
                >= args.min_epochs
                and bad_epochs
                >= args.patience
            ):

                stop_reason = (
                    "early_stopping"
                )

                print()

                print(
                    "Early stopping triggered."
                )

                print(
                    "No validation improvement "
                    f"for {bad_epochs} epochs."
                )

                break

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    # -------------------------------------------------------------
    # Final current-model validation
    # -------------------------------------------------------------

    separator()

    print(
        "FINAL CURRENT-MODEL VALIDATION"
    )

    separator()

    final_validation = (
        evaluate(
            trainer=(
                trainer
            ),

            full_batch=(
                validation_data
            ),

            batch_size=(
                args.batch_size
            ),
        )
    )

    final_checkpoint_metrics = {
        key: value
        for key, value
        in final_validation.items()
        if key not in {
            "targets",
            "predictions",
            "probabilities",
            "sample_ids",
        }
    }

    save_checkpoint(
        path=(
            final_checkpoint_path
        ),

        trainer=(
            trainer
        ),

        epoch=(
            final_epoch
        ),

        validation_metrics=(
            final_checkpoint_metrics
        ),

        configuration=(
            configuration
        ),
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    if best_validation_metrics is None:

        raise RuntimeError(
            "No best validation checkpoint was created."
        )

    best_metrics_clean = {
        key: value
        for key, value
        in best_validation_metrics.items()
        if key not in {
            "targets",
            "predictions",
            "probabilities",
            "sample_ids",
        }
    }

    initial_metrics_clean = {
        key: value
        for key, value
        in initial_validation.items()
        if key not in {
            "targets",
            "predictions",
            "probabilities",
            "sample_ids",
        }
    }

    final_metrics_clean = {
        key: value
        for key, value
        in final_validation.items()
        if key not in {
            "targets",
            "predictions",
            "probabilities",
            "sample_ids",
        }
    }

    summary = {
        "experiment": (
            "AEGIS-FD-1K-development"
        ),

        "protocol": (
            "train800-validation100"
        ),

        "status": (
            "completed"
        ),

        "test_split_status": (
            "sealed_not_accessed"
        ),

        "epochs_completed": (
            final_epoch
        ),

        "global_steps": (
            trainer
            .state
            .global_step
        ),

        "stop_reason": (
            stop_reason
        ),

        "initial_validation": (
            initial_metrics_clean
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_validation": (
            best_metrics_clean
        ),

        "final_current_model_validation": (
            final_metrics_clean
        ),

        "selection_metric": (
            "validation_macro_f1"
        ),

        "selection_tie_breaker": (
            "validation_classification_loss"
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),

        "best_checkpoint": str(
            best_checkpoint_path.resolve()
        ),

        "final_checkpoint": str(
            final_checkpoint_path.resolve()
        ),

        "best_validation_predictions": str(
            predictions_path.resolve()
        ),

        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "important_interpretation": (
            "This is a held-out validation generalization "
            "experiment on balanced development subsets. "
            "It is not the final Fakeddit benchmark result "
            "and the official test split was not accessed."
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    # -------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------

    print(
        "Final validation accuracy:",
        round(
            final_validation[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Final validation F1:",
        round(
            final_validation[
                "f1"
            ],
            4,
        ),
    )

    print(
        "Final validation Macro-F1:",
        round(
            final_validation[
                "macro_f1"
            ],
            4,
        ),
    )

    print()

    separator()

    print(
        "BEST VALIDATION CHECKPOINT"
    )

    separator()

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation accuracy:",
        round(
            best_validation_metrics[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Best validation precision:",
        round(
            best_validation_metrics[
                "precision"
            ],
            4,
        ),
    )

    print(
        "Best validation recall:",
        round(
            best_validation_metrics[
                "recall"
            ],
            4,
        ),
    )

    print(
        "Best validation F1:",
        round(
            best_validation_metrics[
                "f1"
            ],
            4,
        ),
    )

    print(
        "Best validation Macro-F1:",
        round(
            best_validation_metrics[
                "macro_f1"
            ],
            4,
        ),
    )

    print(
        "Best validation classification loss:",
        round(
            best_validation_metrics[
                "classification_loss"
            ],
            6,
        ),
    )

    print(
        "Best validation alignment loss:",
        round(
            best_validation_metrics[
                "alignment_loss"
            ],
            6,
        ),
    )

    print(
        "Best validation confusion matrix:",
        best_validation_metrics[
            "confusion_matrix"
        ],
    )

    print()

    print(
        "Epochs completed:",
        final_epoch,
    )

    print(
        "Stop reason:",
        stop_reason,
    )

    print(
        "Elapsed seconds:",
        round(
            elapsed_seconds,
            2,
        ),
    )

    print()

    separator()

    print(
        "AEGIS TRAIN/VALIDATION "
        "GENERALIZATION EXPERIMENT: COMPLETED"
    )

    separator()

    print()

    print(
        "TRAINING SAMPLES:",
        train_data.batch_size,
    )

    print(
        "VALIDATION SAMPLES:",
        validation_data.batch_size,
    )

    print(
        "TEST SAMPLES ACCESSED:",
        0,
    )

    print()

    print(
        "The official Fakeddit test split "
        "remains SEALED."
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
        "Best checkpoint:",
        best_checkpoint_path,
    )

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )

    print(
        "Best validation predictions:",
        predictions_path,
    )

    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
