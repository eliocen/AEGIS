"""
AEGIS v0.24.0
Fakeddit Controlled Modality Ablation Runner
============================================

Purpose
-------
Run controlled modality and alignment ablations using the same
empirical protocol as scripts.run_fakeddit_generalization.

Supported modes
---------------
1. text_only
       Frozen XLM-R representation
       -> trainable text projection
       -> AEGIS integrity classifier

2. vision_only
       Frozen CLIP representation
       -> trainable vision projection
       -> AEGIS integrity classifier

3. multimodal
       Frozen XLM-R + CLIP representations
       -> trainable modality projections
       -> gated multimodal fusion
       -> AEGIS integrity classifier

       Optional symmetric contrastive alignment loss is controlled
       through --alignment-weight.

Scientific constraints
----------------------
* text_only and vision_only MUST use alignment_weight = 0.0.
* Unimodal modes bypass multimodal fusion completely.
* No zero/dummy modality is injected.
* Fakeddit Stage-1 binary integrity supervision only is used.
* Threat subtype labels are not fabricated.
* The official Fakeddit test split remains sealed.

Recommended controlled conditions
---------------------------------
Text only:
    --mode text_only
    --alignment-weight 0.0

Vision only:
    --mode vision_only
    --alignment-weight 0.0

Multimodal fusion without contrastive alignment:
    --mode multimodal
    --alignment-weight 0.0

Full AEGIS:
    --mode multimodal
    --alignment-weight 0.5

Checkpoint selection
--------------------
Primary:
    highest validation Macro-F1

Tie-break:
    lowest validation classification loss
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

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
    compute_binary_integrity_metrics,
    set_global_seed,
)

# Reuse the validated experiment infrastructure.
from scripts.run_fakeddit_generalization import (
    calculate_binary_metrics,
    count_parameters,
    is_better_checkpoint,
    iter_sequential_batches,
    iter_training_batches,
    load_cache_as_single_batch,
    resolve_device,
    save_json,
    validate_cache_pair,
)


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_TRAIN_CACHE = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings/"
    "train_n5000_seed42"
)

DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/"
    "frozen_embeddings/"
    "validation_n1000_seed42"
)

DEFAULT_EXPERIMENT_ROOT = Path(
    "experiments/fakeddit/"
    "ablation"
)


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run controlled AEGIS Fakeddit modality "
            "and alignment ablations."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "text_only",
            "vision_only",
            "multimodal",
        ],
        help=(
            "Ablation representation pathway. "
            "text_only bypasses visual projection and fusion; "
            "vision_only bypasses textual projection and fusion; "
            "multimodal uses standard AEGIS gated fusion."
        ),
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


# =====================================================================
# Validation
# =====================================================================

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

    if (
        args.mode
        in {
            "text_only",
            "vision_only",
        }
        and args.alignment_weight != 0.0
    ):

        raise ValueError(
            f"{args.mode} requires "
            "--alignment-weight 0.0 because "
            "cross-modal alignment is undefined "
            "for a unimodal ablation."
        )


# =====================================================================
# Console helpers
# =====================================================================

def separator():

    print(
        "=" * 78
    )


def mode_description(
    mode: str,
) -> str:

    descriptions = {
        "text_only": (
            "XLM-R -> text projection -> classifier"
        ),

        "vision_only": (
            "CLIP -> vision projection -> classifier"
        ),

        "multimodal": (
            "XLM-R + CLIP -> projections -> "
            "gated fusion -> classifier"
        ),
    }

    return descriptions[
        mode
    ]


# =====================================================================
# Ablation-aware trainer
# =====================================================================

class FakedditAblationTrainer(
    BinaryIntegrityTrainer
):
    """
    Binary integrity trainer with explicit modality pathways.

    This class intentionally reuses the existing
    CrossModalAlignmentModel projection heads so that all conditions
    share the same projected representation dimensionality.

    Unimodal modes bypass the gated fusion layer entirely.
    """

    VALID_MODES = {
        "text_only",
        "vision_only",
        "multimodal",
    }

    def __init__(
        self,
        *args,
        mode: str = "multimodal",
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        if mode not in self.VALID_MODES:

            raise ValueError(
                f"Unsupported ablation mode: "
                f"{mode!r}. "
                f"Expected one of "
                f"{sorted(self.VALID_MODES)}."
            )

        self.mode = mode

        if (
            self.mode
            != "multimodal"
            and self.alignment_loss_weight
            != 0.0
        ):

            raise ValueError(
                "alignment_loss_weight must be "
                "0.0 for unimodal ablations."
            )

    def _representation_forward(
        self,
        batch: BinaryIntegrityBatch,
        compute_alignment_loss: bool,
    ) -> Dict:
        """
        Construct the representation consumed by the classifier.
        """

        # ---------------------------------------------------------
        # TEXT ONLY
        # ---------------------------------------------------------

        if self.mode == "text_only":

            aligned_text = (
                self.alignment_model
                .text_projection(
                    batch.text_embeddings
                )
            )

            zero_alignment_loss = (
                aligned_text.sum()
                * 0.0
            )

            return {
                "classifier_embedding": (
                    aligned_text
                ),

                "alignment_loss": (
                    zero_alignment_loss
                ),

                "aligned_text": (
                    aligned_text
                ),

                "aligned_vision": None,

                "fused_embedding": None,
            }

        # ---------------------------------------------------------
        # VISION ONLY
        # ---------------------------------------------------------

        if self.mode == "vision_only":

            aligned_vision = (
                self.alignment_model
                .vision_projection(
                    batch.vision_embeddings
                )
            )

            zero_alignment_loss = (
                aligned_vision.sum()
                * 0.0
            )

            return {
                "classifier_embedding": (
                    aligned_vision
                ),

                "alignment_loss": (
                    zero_alignment_loss
                ),

                "aligned_text": None,

                "aligned_vision": (
                    aligned_vision
                ),

                "fused_embedding": None,
            }

        # ---------------------------------------------------------
        # MULTIMODAL
        # ---------------------------------------------------------

        alignment_outputs = (
            self.alignment_model(
                batch.text_embeddings,
                batch.vision_embeddings,
                compute_loss=(
                    compute_alignment_loss
                ),
            )
        )

        if compute_alignment_loss:

            alignment_loss = (
                alignment_outputs[
                    "alignment_loss"
                ]
            )

        else:

            alignment_loss = (
                alignment_outputs[
                    "fused_embedding"
                ].sum()
                * 0.0
            )

        return {
            "classifier_embedding": (
                alignment_outputs[
                    "fused_embedding"
                ]
            ),

            "alignment_loss": (
                alignment_loss
            ),

            "aligned_text": (
                alignment_outputs[
                    "aligned_text"
                ]
            ),

            "aligned_vision": (
                alignment_outputs[
                    "aligned_vision"
                ]
            ),

            "fused_embedding": (
                alignment_outputs[
                    "fused_embedding"
                ]
            ),
        }

    def forward_batch(
        self,
        batch: BinaryIntegrityBatch,
        compute_alignment_loss: bool = True,
    ) -> Dict:
        """
        Forward one binary integrity batch through the selected
        experimental pathway.
        """

        batch.validate(
            text_dim=(
                self.alignment_model.text_dim
            ),

            vision_dim=(
                self.alignment_model.vision_dim
            ),
        )

        batch = batch.to(
            self.device
        )

        use_alignment_loss = (
            compute_alignment_loss
            and self.mode
            == "multimodal"
            and self.alignment_loss_weight
            > 0.0
        )

        representation_outputs = (
            self._representation_forward(
                batch=batch,
                compute_alignment_loss=(
                    use_alignment_loss
                ),
            )
        )

        classification_outputs = (
            self.classification_model(
                representation_outputs[
                    "classifier_embedding"
                ]
            )
        )

        classification_losses = (
            self.binary_loss(
                classification_outputs[
                    "integrity_logits"
                ],
                batch.integrity_targets,
            )
        )

        alignment_loss = (
            representation_outputs[
                "alignment_loss"
            ]
        )

        total_loss = (
            self.alignment_loss_weight
            * alignment_loss
            +
            self.classification_loss_weight
            * classification_losses[
                "loss"
            ]
        )

        metrics = (
            compute_binary_integrity_metrics(
                classification_outputs[
                    "integrity_logits"
                ],
                batch.integrity_targets,
            )
        )

        alignment_outputs = {
            "mode": (
                self.mode
            ),

            "aligned_text": (
                representation_outputs[
                    "aligned_text"
                ]
            ),

            "aligned_vision": (
                representation_outputs[
                    "aligned_vision"
                ]
            ),

            "fused_embedding": (
                representation_outputs[
                    "fused_embedding"
                ]
            ),

            "alignment_loss": (
                alignment_loss
            ),
        }

        return {
            "loss": (
                total_loss
            ),

            "alignment_loss": (
                alignment_loss
            ),

            "classification_loss": (
                classification_losses[
                    "loss"
                ]
            ),

            "integrity_loss": (
                classification_losses[
                    "integrity_loss"
                ]
            ),

            "threat_loss": None,

            "alignment_outputs": (
                alignment_outputs
            ),

            "classification_outputs": (
                classification_outputs
            ),

            "metrics": metrics,
        }


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(
    trainer: FakedditAblationTrainer,
    full_batch: BinaryIntegrityBatch,
    batch_size: int,
) -> Dict:
    """
    Evaluate the selected ablation pathway without gradient updates.
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

        for batch in iter_sequential_batches(
            full_batch=full_batch,
            batch_size=batch_size,
        ):

            outputs = (
                trainer.forward_batch(
                    batch,
                    compute_alignment_loss=(
                        trainer.mode
                        == "multimodal"
                        and trainer
                        .alignment_loss_weight
                        > 0.0
                    ),
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

            current_batch_size = int(
                targets.numel()
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


# =====================================================================
# Training
# =====================================================================

def train_one_epoch(
    trainer: FakedditAblationTrainer,
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


# =====================================================================
# Checkpointing
# =====================================================================

def save_checkpoint(
    path: Path,
    trainer: FakedditAblationTrainer,
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

            "ablation_mode": (
                trainer.mode
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


# =====================================================================
# Parameter accounting
# =====================================================================

def effective_parameter_counts(
    alignment_model,
    classification_model,
    mode: str,
) -> Dict[str, int]:
    """
    Count parameters actually participating in the selected pathway.

    The full alignment module is retained for checkpoint compatibility,
    but unused projection/fusion parameters are excluded from the
    effective count.
    """

    classifier_count = (
        count_parameters(
            classification_model
        )
    )

    if mode == "text_only":

        representation_count = (
            count_parameters(
                alignment_model
                .text_projection
            )
        )

    elif mode == "vision_only":

        representation_count = (
            count_parameters(
                alignment_model
                .vision_projection
            )
        )

    else:

        representation_count = (
            count_parameters(
                alignment_model
            )
        )

    return {
        "representation": (
            representation_count
        ),

        "classification": (
            classifier_count
        ),

        "effective_total": (
            representation_count
            + classifier_count
        ),

        "full_alignment_module": (
            count_parameters(
                alignment_model
            )
        ),
    }


# =====================================================================
# Main experiment
# =====================================================================

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

    # =============================================================
    # Header
    # =============================================================

    separator()

    print(
        "AEGIS FAKEDDIT CONTROLLED ABLATION EXPERIMENT"
    )

    separator()

    print(
        "Ablation mode:",
        args.mode,
    )

    print(
        "Architecture:",
        mode_description(
            args.mode
        ),
    )

    print(
        "Text pathway:",
        (
            "ENABLED"
            if args.mode
            in {
                "text_only",
                "multimodal",
            }
            else "DISABLED"
        ),
    )

    print(
        "Vision pathway:",
        (
            "ENABLED"
            if args.mode
            in {
                "vision_only",
                "multimodal",
            }
            else "DISABLED"
        ),
    )

    print(
        "Gated multimodal fusion:",
        (
            "ENABLED"
            if args.mode
            == "multimodal"
            else "BYPASSED"
        ),
    )

    print(
        "Contrastive alignment objective:",
        (
            "ENABLED"
            if (
                args.mode
                == "multimodal"
                and args.alignment_weight
                > 0.0
            )
            else "DISABLED"
        ),
    )

    print()

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

    # =============================================================
    # Load caches
    # =============================================================

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

    # =============================================================
    # Models
    # =============================================================

    separator()

    print(
        "INITIALIZING ABLATION MODEL"
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

            dropout=(
                args.dropout
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

    # -------------------------------------------------------------
    # Optimizer parameter selection
    # -------------------------------------------------------------

    if args.mode == "text_only":

        representation_parameters = list(
            alignment_model
            .text_projection
            .parameters()
        )

    elif args.mode == "vision_only":

        representation_parameters = list(
            alignment_model
            .vision_projection
            .parameters()
        )

    else:

        representation_parameters = list(
            alignment_model
            .parameters()
        )

    trainable_parameters = (
        representation_parameters
        +
        list(
            classification_model
            .parameters()
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
        FakedditAblationTrainer(
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

            mode=(
                args.mode
            ),
        )
    )

    parameter_counts = (
        effective_parameter_counts(
            alignment_model=(
                alignment_model
            ),

            classification_model=(
                classification_model
            ),

            mode=(
                args.mode
            ),
        )
    )

    print(
        "Representation parameters:",
        parameter_counts[
            "representation"
        ],
    )

    print(
        "Classification parameters:",
        parameter_counts[
            "classification"
        ],
    )

    print(
        "Effective trainable parameters:",
        parameter_counts[
            "effective_total"
        ],
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
        (
            "validation Macro-F1, then "
            "validation classification loss"
        ),
    )

    print()

    # =============================================================
    # Metadata
    # =============================================================

    configuration = {
        "ablation_mode": (
            args.mode
        ),

        "uses_text": (
            args.mode
            in {
                "text_only",
                "multimodal",
            }
        ),

        "uses_vision": (
            args.mode
            in {
                "vision_only",
                "multimodal",
            }
        ),

        "uses_fusion": (
            args.mode
            == "multimodal"
        ),

        "uses_cross_modal_alignment": (
            args.mode
            == "multimodal"
            and args.alignment_weight
            > 0.0
        ),

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
            "AEGIS-Fakeddit-modality-ablation"
        ),

        "protocol": (
            "train5000-validation1000-ablation"
        ),

        "ablation_mode": (
            args.mode
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

        "trainable_parameters": (
            parameter_counts
        ),

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

    # =============================================================
    # Initial validation baseline
    # =============================================================

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

    # =============================================================
    # Training
    # =============================================================

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

            final_epoch = epoch

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

                best_epoch = epoch

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

                        "ablation_mode": (
                            args.mode
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

                "ablation_mode": (
                    args.mode
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
                    f"train_cls="
                    f"{training_metrics['classification_loss']:.4f} | "
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

    # =============================================================
    # Final current-model validation
    # =============================================================

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

    if best_validation_metrics is None:

        raise RuntimeError(
            "No best validation checkpoint was created."
        )

    # =============================================================
    # Summary
    # =============================================================

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
            "AEGIS-Fakeddit-modality-ablation"
        ),

        "protocol": (
            "train5000-validation1000-ablation"
        ),

        "ablation_mode": (
            args.mode
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

        "configuration": (
            configuration
        ),

        "trainable_parameters": (
            parameter_counts
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
            "This is a held-out development-set "
            "modality ablation. It is not the "
            "final Fakeddit benchmark result. "
            "The official test split was not accessed."
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    # =============================================================
    # Final terminal report
    # =============================================================

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
        "Ablation mode:",
        args.mode,
    )

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
        "AEGIS FAKEDDIT ABLATION: COMPLETED"
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