"""
AEGIS v0.27.7-dev
Fakeddit Controlled Modality, Fusion, and M4q Quality-Supervision Runner
============================================

Purpose
-------
Run controlled modality, alignment, and fusion-architecture ablations using
the same empirical protocol as scripts.run_fakeddit_generalization.

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
       -> selected fusion architecture:
          legacy gated fusion OR interaction-only fusion OR gated-interaction fusion OR reliability-only residual fusion OR interaction-conditioned reliability residual fusion OR evidence-aware adaptive fusion
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

Fusion architectures
--------------------
legacy:
    Validated v0.24 gated multimodal fusion.

interaction_only:
    v0.25.2 explicit cross-modal evidence interaction
    -> interaction-conditioned fusion without reliability estimation.

    This architecture is valid only for --mode multimodal.

gated_interaction:
    v0.25.3 validated gated multimodal fusion
    -> explicit cross-modal interaction residual
    -> deterministic 1/sqrt(d) interaction scaling.

    This architecture is valid only for --mode multimodal.

reliability_only:
    v0.25.4 legacy gated fusion
    -> modality-only reliability estimation
    -> reliability-aware adaptive residual fusion.

    This architecture is valid only for --mode multimodal.

interaction_reliability:
    v0.25.5 legacy gated fusion
    -> interaction-conditioned reliability estimation
    -> reliability-aware adaptive residual fusion.

    This architecture is valid only for --mode multimodal.

quality_supervised:
    v0.27 M4q exact M1b fusion plus separate intrinsic-quality heads.
    Quality heads are auxiliary/diagnostic and never feed the fusion equation.

quality_compatibility_supervised:
    v0.27 M4qc exact M1b fusion plus separate intrinsic-quality heads and
    a separate text-image compatibility head. All three diagnostic scores are
    auxiliary and never feed the fusion equation.

quality_compatibility_fusion:
    v0.27 M4qcf reliability-informed fusion. Intrinsic quality and compatibility
    are explicitly supervised as in M4qc, while detached q_T/q_V/c_TV signals
    deterministically control modality weighting and interaction suppression.

evidence_aware:
    v0.25 cross-modal evidence interaction
    -> modality reliability estimation
    -> reliability-aware adaptive fusion.

    This architecture is valid only for --mode multimodal.

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

Legacy full AEGIS v0.24 (M0):
    --mode multimodal
    --fusion-architecture legacy
    --alignment-weight 0.5

Interaction-only AEGIS v0.25.2 (M1):
    --mode multimodal
    --fusion-architecture interaction_only
    --alignment-weight 0.5

Gated-interaction AEGIS v0.25.3 (M1b):
    --mode multimodal
    --fusion-architecture gated_interaction
    --alignment-weight 0.5

Reliability-only AEGIS v0.25.4 (M2):
    --mode multimodal
    --fusion-architecture reliability_only
    --alignment-weight 0.5

Interaction-conditioned reliability residual AEGIS v0.25.5 (M2b):
    --mode multimodal
    --fusion-architecture interaction_reliability
    --alignment-weight 0.5

Quality-supervised AEGIS v0.27 (M4q):
    --mode multimodal
    --fusion-architecture quality_supervised
    --alignment-weight 0.5
    --quality-weight 1.0

Quality + compatibility-supervised AEGIS v0.27 (M4qc):
    --mode multimodal
    --fusion-architecture quality_compatibility_supervised
    --alignment-weight 0.5
    --quality-weight 1.0
    --compatibility-weight 1.0

Evidence-aware AEGIS v0.25 (M3):
    --mode multimodal
    --fusion-architecture evidence_aware
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
import copy
import hashlib
import json
import platform
import random
import sys
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import nn

from aegis.alignment import (
    CrossModalAlignmentModel,
)

from aegis.classification import (
    HierarchicalInformationIntegrityClassifier,
)

from aegis.data.cache import (
    FakedditRepresentationCache,
)

from aegis.reliability.corruption import (
    attenuation,
    compute_feature_std,
    deterministic_derangement_indices,
    gaussian_noise,
    zero_dropout,
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

V030_SELECTIVE_INTERVENTION_ARCHITECTURE_MODES = {
    "quality_compatibility_selective_intervention_weights": "weights_only",
    "quality_compatibility_selective_intervention_transition": "transition_only",
    "quality_compatibility_selective_intervention": "combined",
}
V030_SELECTIVE_INTERVENTION_ARCHITECTURES = frozenset(
    V030_SELECTIVE_INTERVENTION_ARCHITECTURE_MODES
)


def v030_selective_intervention_mode(
    fusion_architecture: str,
) -> Optional[str]:
    """Map a v0.30 runner architecture to the frozen Step2 model mode."""
    return V030_SELECTIVE_INTERVENTION_ARCHITECTURE_MODES.get(
        fusion_architecture
    )


def is_v030_selective_intervention_architecture(
    fusion_architecture: str,
) -> bool:
    """Return whether the runner architecture belongs to frozen v0.30."""
    return fusion_architecture in V030_SELECTIVE_INTERVENTION_ARCHITECTURES


V031_CALIBRATED_INTERVENTION_ARCHITECTURE_MODES = {
    "quality_compatibility_calibrated_intervention_calibration": "calibration_only",
    "quality_compatibility_calibrated_intervention_utility": "utility_only",
    "quality_compatibility_calibrated_intervention": "combined",
}
V031_CALIBRATED_INTERVENTION_ARCHITECTURES = frozenset(
    V031_CALIBRATED_INTERVENTION_ARCHITECTURE_MODES
)
V031_CALIBRATION_LOSS_WEIGHT = 0.25
V031_UTILITY_LOSS_WEIGHT = 0.50
V031_UTILITY_TARGET_TEMPERATURE = 0.05

def v031_calibrated_intervention_mode(fusion_architecture: str) -> Optional[str]:
    return V031_CALIBRATED_INTERVENTION_ARCHITECTURE_MODES.get(fusion_architecture)

def is_v031_calibrated_intervention_architecture(fusion_architecture: str) -> bool:
    return fusion_architecture in V031_CALIBRATED_INTERVENTION_ARCHITECTURES

V033_UTILITY_SUPERVISED_ARCHITECTURE = (
    "quality_compatibility_utility_supervised_intervention"
)
V033_UTILITY_SUPERVISED_ARCHITECTURES = frozenset({
    V033_UTILITY_SUPERVISED_ARCHITECTURE,
})
V033_UTILITY_LOSS_WEIGHT = 0.50

def is_v033_utility_supervised_architecture(fusion_architecture: str) -> bool:
    return fusion_architecture in V033_UTILITY_SUPERVISED_ARCHITECTURES

def compute_v033_utility_loss(
    selector_logit: torch.Tensor,
    utility_target_outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    u_target = utility_target_outputs["u_target"]
    delta_u = utility_target_outputs["delta_u"]
    if selector_logit.ndim != 2 or selector_logit.shape[1] != 1:
        raise ValueError("selector_logit must have shape [B,1].")
    if u_target.ndim != 1 or delta_u.ndim != 1:
        raise ValueError("v0.33 utility targets must have shape [B].")
    if selector_logit.shape[0] != u_target.shape[0]:
        raise ValueError("v0.33 utility batch size mismatch.")
    weights = 1.0 + 4.0 * torch.clamp(
        delta_u.detach().abs() / 0.10,
        min=0.0,
        max=1.0,
    )
    per_sample = nn.functional.binary_cross_entropy_with_logits(
        selector_logit.view(-1),
        u_target.detach(),
        reduction="none",
    )
    return (weights * per_sample).mean()

QUALITY_TRAINING_ARCHITECTURES = frozenset({
    "quality_supervised",
    "quality_compatibility_supervised",
    "quality_compatibility_fusion",
    "quality_compatibility_selective_weights",
    "quality_compatibility_selective_interaction",
    "quality_compatibility_selective_fusion",
    "quality_compatibility_graded_weights",
    "quality_compatibility_transition_control",
    "quality_compatibility_graded_transition_fusion",
}) | V030_SELECTIVE_INTERVENTION_ARCHITECTURES | V031_CALIBRATED_INTERVENTION_ARCHITECTURES | V033_UTILITY_SUPERVISED_ARCHITECTURES


def requires_quality_feature_statistics(fusion_architecture: str) -> bool:
    """Return whether training needs frozen-cache Gaussian scale statistics."""
    return fusion_architecture in QUALITY_TRAINING_ARCHITECTURES


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
            "multimodal uses the architecture selected by "
            "--fusion-architecture."
        ),
    )

    parser.add_argument(
        "--fusion-architecture",
        type=str,
        default="legacy",
        choices=[
            "legacy",
            "interaction_only",
            "gated_interaction",
            "reliability_only",
            "interaction_reliability",
            "quality_supervised",
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            "evidence_aware",
        ],
        help=(
            "Multimodal fusion architecture. legacy preserves the "
            "validated v0.24 gated-fusion pathway; interaction_only "
            "enables explicit cross-modal interaction with "
            "interaction-conditioned fusion but no reliability "
            "estimation; gated_interaction preserves the legacy gate "
            "and adds a deterministic scaled interaction residual; "
            "reliability_only preserves the legacy gate and adds "
            "modality-only reliability with a deterministic scaled "
            "adaptive-fusion residual; interaction_reliability preserves "
            "the legacy gate and adds interaction-conditioned reliability "
            "with a deterministic scaled adaptive-fusion residual; "
            "quality_supervised enables v0.27 M4q: the exact M1b fusion "
            "path plus diagnostic intrinsic-quality heads trained with "
            "explicit corruption-aware supervision; "
            "quality_compatibility_supervised enables v0.27 M4qc: the exact "
            "M1b path plus diagnostic quality and compatibility heads; "
            "quality_compatibility_fusion enables v0.27 M4qcf: supervised "
            "quality and compatibility signals control reliability-informed fusion; "
            "evidence_aware enables "
            "v0.25 interaction, reliability "
            "estimation, and reliability-aware adaptive fusion. "
            "Non-legacy architectures are valid only with "
            "--mode multimodal."
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
        "--quality-weight",
        type=float,
        default=1.0,
        help=(
            "M4q intrinsic-quality auxiliary loss coefficient. Frozen "
            "v0.27 default: 1.0. Ignored outside quality_supervised."
        ),
    )

    parser.add_argument(
        "--compatibility-weight",
        type=float,
        default=1.0,
        help=(
            "M4qc cross-modal compatibility auxiliary BCE coefficient. "
            "Frozen Step 11A/Step 13A default: 1.0. Ignored outside "
            "M4qc/M4qcf architectures."
        ),
    )

    parser.add_argument(
        "--transition-objective-weight",
        type=float,
        default=0.25,
        help=(
            "Frozen v0.29 training-only harmful-transition objective "
            "coefficient. Required to equal 0.25 for M4qgr, M4qtc, and "
            "M4qgrt. The objective is active only for M4qtc and M4qgrt."
        ),
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

    if args.quality_weight < 0:

        raise ValueError(
            "--quality-weight must be >= 0."
        )

    if args.fusion_architecture == "quality_supervised" and args.quality_weight <= 0:

        raise ValueError(
            "quality_supervised requires --quality-weight > 0."
        )

    if args.compatibility_weight < 0:

        raise ValueError(
            "--compatibility-weight must be >= 0."
        )

    if args.transition_objective_weight < 0:

        raise ValueError(
            "--transition-objective-weight must be >= 0."
        )

    if (
        is_v031_calibrated_intervention_architecture(args.fusion_architecture)
        and args.transition_objective_weight != 0.0
    ):
        raise ValueError(
            "v0.31 M4qcesi architectures require "
            "--transition-objective-weight 0.0."
        )

    if (
        is_v033_utility_supervised_architecture(args.fusion_architecture)
        and args.transition_objective_weight != 0.0
    ):
        raise ValueError(
            "v0.33 M4qusli requires --transition-objective-weight 0.0."
        )

    if (
        args.fusion_architecture in {
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
        }
        and args.transition_objective_weight != 0.25
    ):

        raise ValueError(
            "The frozen v0.29 architectures require "
            "--transition-objective-weight 0.25."
        )

    if (
        args.fusion_architecture in {
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }
        and args.quality_weight <= 0
    ):

        raise ValueError(
            "M4qc/M4qcf requires --quality-weight > 0."
        )

    if (
        args.fusion_architecture in {
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }
        and args.compatibility_weight <= 0
    ):

        raise ValueError(
            "M4qc/M4qcf requires --compatibility-weight > 0."
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
        args.fusion_architecture
        in {
            "interaction_only",
            "gated_interaction",
            "reliability_only",
            "interaction_reliability",
            "quality_supervised",
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            "evidence_aware",
        }
        and args.mode != "multimodal"
    ):

        raise ValueError(
            "--fusion-architecture "
            f"{args.fusion_architecture} requires "
            "--mode multimodal. Unimodal ablations bypass fusion."
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
    fusion_architecture: str,
) -> str:

    if mode == "text_only":
        return "XLM-R -> text projection -> classifier"

    if mode == "vision_only":
        return "CLIP -> vision projection -> classifier"

    if fusion_architecture == "interaction_only":
        return (
            "XLM-R + CLIP -> projections -> evidence interaction -> "
            "interaction-conditioned fusion -> classifier"
        )

    if fusion_architecture == "gated_interaction":
        return (
            "XLM-R + CLIP -> projections -> legacy gated fusion + "
            "explicit evidence interaction residual -> classifier"
        )

    if fusion_architecture == "reliability_only":
        return (
            "XLM-R + CLIP -> projections -> legacy gated fusion + "
            "modality-only reliability adaptive residual -> classifier"
        )

    if fusion_architecture == "interaction_reliability":
        return (
            "XLM-R + CLIP -> projections -> legacy gated fusion + "
            "interaction-conditioned reliability adaptive residual -> classifier"
        )

    if fusion_architecture == "quality_supervised":
        return (
            "XLM-R + CLIP -> projections -> M1b gated-interaction fusion + "
            "diagnostic intrinsic-quality heads -> classifier"
        )

    if fusion_architecture == "quality_compatibility_supervised":
        return (
            "XLM-R + CLIP -> projections -> M1b gated-interaction fusion + "
            "diagnostic intrinsic-quality and compatibility heads -> classifier"
        )

    if fusion_architecture == "quality_compatibility_fusion":
        return (
            "XLM-R + CLIP -> projections -> supervised q/c reliability controller + "
            "reliability-weighted evidence + compatibility-scaled interaction -> classifier"
        )

    if fusion_architecture == "quality_compatibility_selective_weights":
        return "M4qcs-w selective quality allocation with unsuppressed interaction"

    if fusion_architecture == "quality_compatibility_selective_interaction":
        return "M4qcs-i equal allocation with selective interaction suppression"

    if fusion_architecture == "quality_compatibility_selective_fusion":
        return "M4qcs combined selective allocation and interaction suppression"
    if fusion_architecture == "quality_compatibility_graded_weights":
        return "M4qgr graded reliability allocation with unsuppressed interaction"
    if fusion_architecture == "quality_compatibility_transition_control":
        return "M4qtc transition-aware interaction control with equal allocation"
    if fusion_architecture == "quality_compatibility_graded_transition_fusion":
        return "M4qgrt combined graded reliability and transition control"
    if fusion_architecture == "quality_compatibility_selective_intervention_weights":
        return "M4qesri-w evidence-conditioned selective modality weighting"
    if fusion_architecture == "quality_compatibility_selective_intervention_transition":
        return "M4qesri-t evidence-conditioned selective transition intervention"
    if fusion_architecture == "quality_compatibility_selective_intervention":
        return "M4qesri combined evidence-conditioned selective intervention"

    if fusion_architecture == "evidence_aware":
        return (
            "XLM-R + CLIP -> projections -> evidence interaction -> "
            "reliability estimation -> adaptive fusion -> classifier"
        )

    return (
        "XLM-R + CLIP -> projections -> "
        "legacy gated fusion -> classifier"
    )


# =====================================================================
# Ablation-aware trainer
# =====================================================================

def harmful_transition_loss(
    matched_logits: torch.Tensor,
    mismatched_logits: torch.Tensor,
    integrity_targets: torch.Tensor,
    mismatch_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize only mismatch-induced reductions in true-class probability.

    The binary integrity classifier emits two class logits per sample and
    integrity_targets contains one integer class index per sample.

    The matched probability is a detached reference. Consequently the model
    cannot reduce this loss by degrading its matched prediction. A beneficial
    transition, where mismatch increases or preserves true-class probability,
    contributes exactly zero.
    """
    if matched_logits.shape != mismatched_logits.shape:
        raise ValueError("Matched and mismatched logit shapes must agree.")

    if mismatched_logits.ndim != 2 or mismatched_logits.shape[1] != 2:
        raise ValueError(
            "Transition-objective logits must have shape [batch_size, 2]."
        )

    if integrity_targets.ndim != 1:
        raise ValueError(
            "Transition-objective integrity targets must have shape [batch_size]."
        )

    if integrity_targets.shape[0] != mismatched_logits.shape[0]:
        raise ValueError(
            "Transition-objective target and logit batch sizes must agree."
        )

    targets = integrity_targets.to(
        device=mismatched_logits.device,
        dtype=torch.long,
    )

    if bool(((targets < 0) | (targets >= 2)).any().item()):
        raise ValueError(
            "Transition-objective integrity targets must be binary class "
            "indices 0 or 1."
        )

    mask = mismatch_mask.to(
        device=mismatched_logits.device,
        dtype=torch.bool,
    )

    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask.reshape(-1)

    if mask.ndim != 1 or mask.shape[0] != mismatched_logits.shape[0]:
        raise ValueError(
            "Transition-objective mismatch mask must have shape "
            "[batch_size] or [batch_size, 1]."
        )

    matched_probability = torch.softmax(
        matched_logits.detach().to(mismatched_logits.device),
        dim=-1,
    )

    mismatched_probability = torch.softmax(
        mismatched_logits,
        dim=-1,
    )

    gather_index = targets.reshape(-1, 1)

    matched_true_probability = matched_probability.gather(
        1,
        gather_index,
    ).squeeze(1)

    mismatched_true_probability = mismatched_probability.gather(
        1,
        gather_index,
    ).squeeze(1)

    per_sample_harm = torch.relu(
        matched_true_probability - mismatched_true_probability
    )

    if not bool(mask.any().item()):
        return mismatched_logits.sum() * 0.0

    return per_sample_harm.masked_select(mask).mean()

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
        fusion_architecture: str = "legacy",
        **kwargs,
    ):

        # Consume ablation-runner-specific arguments here instead of
        # forwarding them to BinaryIntegrityTrainer, whose constructor
        # intentionally knows nothing about fusion-architecture choices.
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

        if fusion_architecture not in {
            "legacy",
            "interaction_only",
            "gated_interaction",
            "reliability_only",
            "interaction_reliability",
            "quality_supervised",
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            "evidence_aware",
        }:
            raise ValueError(
                "Unsupported fusion architecture: "
                f"{fusion_architecture!r}. Expected one of "
                "['evidence_aware', 'gated_interaction', "
                "'interaction_only', 'interaction_reliability', 'legacy', "
                "'quality_compatibility_fusion', 'quality_compatibility_supervised', 'quality_supervised', "
                "'quality_compatibility_selective_intervention_weights', "
                "'quality_compatibility_selective_intervention_transition', "
                "'quality_compatibility_selective_intervention', 'reliability_only']."
            )

        if (
            self.mode != "multimodal"
            and fusion_architecture != "legacy"
        ):
            raise ValueError(
                "Non-legacy fusion architectures are valid only for "
                "multimodal ablations."
            )

        expected_evidence_aware = (
            self.mode == "multimodal"
            and fusion_architecture == "evidence_aware"
        )

        expected_interaction_only = (
            self.mode == "multimodal"
            and fusion_architecture == "interaction_only"
        )

        expected_gated_interaction = (
            self.mode == "multimodal"
            and fusion_architecture == "gated_interaction"
        )

        expected_reliability_only = (
            self.mode == "multimodal"
            and fusion_architecture == "reliability_only"
        )

        expected_interaction_reliability = (
            self.mode == "multimodal"
            and fusion_architecture == "interaction_reliability"
        )

        if bool(self.alignment_model.evidence_aware) != expected_evidence_aware:
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.evidence_aware."
            )

        if (
            bool(self.alignment_model.interaction_only)
            != expected_interaction_only
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.interaction_only."
            )

        if (
            bool(self.alignment_model.gated_interaction)
            != expected_gated_interaction
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.gated_interaction."
            )

        if (
            bool(self.alignment_model.reliability_only)
            != expected_reliability_only
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.reliability_only."
            )

        if (
            bool(self.alignment_model.interaction_reliability)
            != expected_interaction_reliability
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.interaction_reliability."
            )

        expected_quality_supervised = (
            self.mode == "multimodal"
            and fusion_architecture == "quality_supervised"
        )

        if (
            bool(self.alignment_model.quality_supervised)
            != expected_quality_supervised
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.quality_supervised."
            )

        expected_quality_compatibility_supervised = (
            self.mode == "multimodal"
            and fusion_architecture == "quality_compatibility_supervised"
        )

        if (
            bool(self.alignment_model.quality_compatibility_supervised)
            != expected_quality_compatibility_supervised
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.quality_compatibility_supervised."
            )

        expected_quality_compatibility_fusion = (
            self.mode == "multimodal"
            and fusion_architecture == "quality_compatibility_fusion"
        )

        if (
            bool(self.alignment_model.quality_compatibility_fusion)
            != expected_quality_compatibility_fusion
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with "
                "alignment_model.quality_compatibility_fusion."
            )

        expected_selective_mode = {
            "quality_compatibility_selective_weights": "weights_only",
            "quality_compatibility_selective_interaction": "interaction_only",
            "quality_compatibility_selective_fusion": "combined",
        }.get(fusion_architecture)
        if (
            self.alignment_model.quality_compatibility_selective_fusion
            != expected_selective_mode
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with alignment_model."
                "quality_compatibility_selective_fusion."
            )

        expected_graded_transition_mode = {
            "quality_compatibility_graded_weights": "graded_weights_only",
            "quality_compatibility_transition_control": "transition_only",
            "quality_compatibility_graded_transition_fusion": "combined",
        }.get(fusion_architecture)
        if (
            self.alignment_model.quality_compatibility_graded_transition_fusion
            != expected_graded_transition_mode
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with alignment_model."
                "quality_compatibility_graded_transition_fusion."
            )

        expected_v030_selective_intervention_mode = (
            v030_selective_intervention_mode(fusion_architecture)
        )
        if (
            self.alignment_model.quality_compatibility_selective_intervention
            != expected_v030_selective_intervention_mode
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with alignment_model."
                "quality_compatibility_selective_intervention."
            )

        expected_v031_mode = v031_calibrated_intervention_mode(
            fusion_architecture
        )
        if (
            self.alignment_model.quality_compatibility_calibrated_intervention
            != expected_v031_mode
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with alignment_model."
                "quality_compatibility_calibrated_intervention."
            )

        expected_v033 = (
            self.mode == "multimodal"
            and is_v033_utility_supervised_architecture(fusion_architecture)
        )
        if (
            bool(
                self.alignment_model
                .quality_compatibility_utility_supervised_intervention
            )
            != expected_v033
        ):
            raise ValueError(
                "fusion_architecture is inconsistent with alignment_model."
                "quality_compatibility_utility_supervised_intervention."
            )

        self.fusion_architecture = fusion_architecture

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
        """Construct the representation consumed by the classifier."""

        if self.mode == "text_only":
            aligned_text = self.alignment_model.text_projection(
                batch.text_embeddings
            )
            zero_alignment_loss = aligned_text.sum() * 0.0
            return {
                "classifier_embedding": aligned_text,
                "alignment_loss": zero_alignment_loss,
                "aligned_text": aligned_text,
                "aligned_vision": None,
                "fused_embedding": None,
                "text_quality": None,
                "vision_quality": None,
                "compatibility_score": None,
            }

        if self.mode == "vision_only":
            aligned_vision = self.alignment_model.vision_projection(
                batch.vision_embeddings
            )
            zero_alignment_loss = aligned_vision.sum() * 0.0
            return {
                "classifier_embedding": aligned_vision,
                "alignment_loss": zero_alignment_loss,
                "aligned_text": None,
                "aligned_vision": aligned_vision,
                "fused_embedding": None,
                "text_quality": None,
                "vision_quality": None,
                "compatibility_score": None,
            }

        alignment_outputs = self.alignment_model(
            batch.text_embeddings,
            batch.vision_embeddings,
            compute_loss=compute_alignment_loss,
        )

        if compute_alignment_loss:
            alignment_loss = alignment_outputs["alignment_loss"]
        else:
            alignment_loss = alignment_outputs["fused_embedding"].sum() * 0.0

        return {
            "classifier_embedding": alignment_outputs["fused_embedding"],
            "alignment_loss": alignment_loss,
            "aligned_text": alignment_outputs["aligned_text"],
            "aligned_vision": alignment_outputs["aligned_vision"],
            "fused_embedding": alignment_outputs["fused_embedding"],
            "text_quality": alignment_outputs.get("text_quality"),
            "vision_quality": alignment_outputs.get("vision_quality"),
            "compatibility_score": alignment_outputs.get("compatibility_score"),
            "v031_control": alignment_outputs.get("v031_control"),
            "v033_control": alignment_outputs.get("v033_control"),
            "stable_reference_fused_embedding": alignment_outputs.get(
                "stable_reference_fused_embedding"
            ),
            "candidate_fused_embedding": alignment_outputs.get(
                "candidate_fused_embedding"
            ),
        }

    def forward_batch(
        self,
        batch: BinaryIntegrityBatch,
        compute_alignment_loss: bool = True,
        quality_targets: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        quality_loss_weight: float = 0.0,
        quality_sample_weights: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        compatibility_targets: Optional[torch.Tensor] = None,
        compatibility_loss_weight: float = 0.0,
        calibration_loss_weight: float = 0.0,
        utility_loss_weight: float = 0.0,
        transition_reference_batch: Optional[BinaryIntegrityBatch] = None,
        transition_loss_weight: float = 0.0,
    ) -> Dict:
        """
        Forward one binary-integrity batch.

        For M4q/M4qc, ``quality_targets`` contains [B,1] intrinsic
        text/vision quality supervision. For M4qc, ``compatibility_targets``
        contains [B,1] matched/mismatched supervision. None of q_T, q_V, or
        c_TV enters primary fusion; only auxiliary loss terms are added.
        """
        batch.validate(
            text_dim=self.alignment_model.text_dim,
            vision_dim=self.alignment_model.vision_dim,
        )
        batch = batch.to(self.device)

        use_alignment_loss = (
            compute_alignment_loss
            and self.mode == "multimodal"
            and self.alignment_loss_weight > 0.0
        )

        representation_outputs = self._representation_forward(
            batch=batch,
            compute_alignment_loss=use_alignment_loss,
        )

        classification_outputs = self.classification_model(
            representation_outputs["classifier_embedding"]
        )
        classification_losses = self.binary_loss(
            classification_outputs["integrity_logits"],
            batch.integrity_targets,
        )
        alignment_loss = representation_outputs["alignment_loss"]

        quality_loss = classification_losses["loss"] * 0.0
        text_quality_loss = classification_losses["loss"] * 0.0
        vision_quality_loss = classification_losses["loss"] * 0.0
        compatibility_loss = classification_losses["loss"] * 0.0
        calibration_loss = classification_losses["loss"] * 0.0
        utility_loss = classification_losses["loss"] * 0.0
        utility_target_outputs = None
        transition_loss = classification_losses["loss"] * 0.0

        if quality_targets is not None:
            if self.fusion_architecture not in {
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }:
                raise ValueError(
                    "quality_targets are valid only for M4q/M4qc quality "
                    "supervision architectures."
                )
            text_quality = representation_outputs["text_quality"]
            vision_quality = representation_outputs["vision_quality"]
            if text_quality is None or vision_quality is None:
                raise RuntimeError("M4q quality predictions are unavailable.")

            text_target, vision_target = quality_targets
            text_target = text_target.to(
                device=text_quality.device, dtype=text_quality.dtype
            )
            vision_target = vision_target.to(
                device=vision_quality.device, dtype=vision_quality.dtype
            )
            if text_target.shape != text_quality.shape:
                raise ValueError(
                    "text quality target shape mismatch: "
                    f"{tuple(text_target.shape)} != {tuple(text_quality.shape)}."
                )
            if vision_target.shape != vision_quality.shape:
                raise ValueError(
                    "vision quality target shape mismatch: "
                    f"{tuple(vision_target.shape)} != {tuple(vision_quality.shape)}."
                )
            if quality_sample_weights is None:
                text_quality_loss = nn.functional.mse_loss(
                    text_quality, text_target
                )
                vision_quality_loss = nn.functional.mse_loss(
                    vision_quality, vision_target
                )
            else:
                text_sample_weight, vision_sample_weight = (
                    quality_sample_weights
                )
                text_sample_weight = text_sample_weight.to(
                    device=text_quality.device, dtype=text_quality.dtype
                )
                vision_sample_weight = vision_sample_weight.to(
                    device=vision_quality.device, dtype=vision_quality.dtype
                )
                if text_sample_weight.shape != text_quality.shape:
                    raise ValueError("text quality sample-weight shape mismatch.")
                if vision_sample_weight.shape != vision_quality.shape:
                    raise ValueError("vision quality sample-weight shape mismatch.")
                if torch.any(text_sample_weight <= 0.0) or torch.any(
                    vision_sample_weight <= 0.0
                ):
                    raise ValueError("quality sample weights must be positive.")
                text_quality_loss = (
                    (text_quality - text_target).pow(2) * text_sample_weight
                ).mean()
                vision_quality_loss = (
                    (vision_quality - vision_target).pow(2)
                    * vision_sample_weight
                ).mean()
            quality_loss = 0.5 * (text_quality_loss + vision_quality_loss)

        if compatibility_targets is not None:
            if self.fusion_architecture not in {
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }:
                raise ValueError(
                    "compatibility_targets are valid only for M4qc/M4qcf."
                )
            compatibility_score = representation_outputs["compatibility_score"]
            if compatibility_score is None:
                raise RuntimeError("M4qc compatibility prediction is unavailable.")
            compatibility_target = compatibility_targets.to(
                device=compatibility_score.device,
                dtype=compatibility_score.dtype,
            )
            if compatibility_target.shape != compatibility_score.shape:
                raise ValueError(
                    "compatibility target shape mismatch: "
                    f"{tuple(compatibility_target.shape)} != "
                    f"{tuple(compatibility_score.shape)}."
                )
            compatibility_loss = nn.functional.binary_cross_entropy(
                compatibility_score,
                compatibility_target,
            )

        if is_v031_calibrated_intervention_architecture(
            self.fusion_architecture
        ):
            control = representation_outputs["v031_control"]
            if control is None:
                raise RuntimeError("M4qcesi controller output is unavailable.")
            if quality_targets is None or compatibility_targets is None:
                if calibration_loss_weight > 0.0 or utility_loss_weight > 0.0:
                    raise ValueError(
                        "v0.31 auxiliary training requires training-only "
                        "quality and compatibility targets."
                    )
            else:
                text_target, vision_target = quality_targets
                text_target = text_target.to(
                    device=control.q_text_calibrated.device,
                    dtype=control.q_text_calibrated.dtype,
                )
                vision_target = vision_target.to(
                    device=control.q_vision_calibrated.device,
                    dtype=control.q_vision_calibrated.dtype,
                )
                compatibility_target = compatibility_targets.to(
                    device=control.compatibility_calibrated.device,
                    dtype=control.compatibility_calibrated.dtype,
                )
                if self.alignment_model.calibrated_utility_controller.calibration_enabled:
                    calibration_loss = (
                        nn.functional.binary_cross_entropy(
                            control.q_text_calibrated, text_target
                        )
                        + nn.functional.binary_cross_entropy(
                            control.q_vision_calibrated, vision_target
                        )
                        + nn.functional.binary_cross_entropy(
                            control.compatibility_calibrated, compatibility_target
                        )
                    ) / 3.0
                elif calibration_loss_weight != 0.0:
                    raise ValueError("Calibration loss must be zero when disabled.")

                if self.alignment_model.calibrated_utility_controller.utility_selector_enabled:
                    ref = representation_outputs["stable_reference_fused_embedding"]
                    cand = representation_outputs["candidate_fused_embedding"]
                    if ref is None or cand is None:
                        raise RuntimeError("M4qcesi counterfactual embeddings unavailable.")
                    ref_logits = self.classification_model(ref)["integrity_logits"]
                    cand_logits = self.classification_model(cand)["integrity_logits"]
                    utility_target_outputs = (
                        self.alignment_model.calibrated_utility_controller
                        .utility_target_from_logits(
                            ref_logits, cand_logits, batch.integrity_targets
                        )
                    )
                    utility_loss = nn.functional.binary_cross_entropy(
                        control.utility_probability,
                        utility_target_outputs["u_target"],
                    )
                elif utility_loss_weight != 0.0:
                    raise ValueError("Utility loss must be zero when disabled.")

        if is_v033_utility_supervised_architecture(self.fusion_architecture):
            control = representation_outputs["v033_control"]
            if control is None:
                raise RuntimeError("M4qusli controller output is unavailable.")
            if quality_targets is None or compatibility_targets is None:
                if utility_loss_weight > 0.0:
                    raise ValueError(
                        "v0.33 utility supervision requires training-only "
                        "quality and compatibility targets."
                    )
            else:
                ref = representation_outputs["stable_reference_fused_embedding"]
                cand = representation_outputs["candidate_fused_embedding"]
                if ref is None or cand is None:
                    raise RuntimeError(
                        "M4qusli counterfactual embeddings unavailable."
                    )
                ref_logits = self.classification_model(ref)["integrity_logits"]
                cand_logits = self.classification_model(cand)["integrity_logits"]
                utility_target_outputs = (
                    self.alignment_model
                    .utility_supervised_intervention_controller
                    .utility_target_from_logits(
                        ref_logits, cand_logits, batch.integrity_targets
                    )
                )
                utility_loss = compute_v033_utility_loss(
                    control.selector_logit,
                    utility_target_outputs,
                )
                if float(utility_loss_weight) != V033_UTILITY_LOSS_WEIGHT:
                    raise ValueError(
                        "M4qusli effective utility loss weight must equal 0.50."
                    )

        if transition_reference_batch is not None:
            if self.fusion_architecture not in {
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
            }:
                raise ValueError(
                    "A transition reference is valid only for M4qtc/M4qgrt."
                )
            if transition_loss_weight <= 0.0:
                raise ValueError(
                    "A transition reference requires transition_loss_weight > 0."
                )
            if compatibility_targets is None:
                raise ValueError(
                    "Transition supervision requires compatibility targets."
                )

            transition_reference_batch.validate(
                text_dim=self.alignment_model.text_dim,
                vision_dim=self.alignment_model.vision_dim,
            )
            reference_batch = transition_reference_batch.to(self.device)
            if reference_batch.batch_size != batch.batch_size:
                raise ValueError("Transition-reference batch size mismatch.")
            if not torch.equal(
                reference_batch.integrity_targets,
                batch.integrity_targets,
            ):
                raise ValueError("Transition-reference targets changed.")

            alignment_was_training = self.alignment_model.training
            classifier_was_training = self.classification_model.training
            self.alignment_model.eval()
            self.classification_model.eval()
            try:
                with torch.no_grad():
                    matched_representation = self._representation_forward(
                        batch=reference_batch,
                        compute_alignment_loss=False,
                    )
                    matched_logits = self.classification_model(
                        matched_representation["classifier_embedding"]
                    )["integrity_logits"]

                deterministic_mismatch_representation = (
                    self._representation_forward(
                        batch=batch,
                        compute_alignment_loss=False,
                    )
                )
                deterministic_mismatch_logits = self.classification_model(
                    deterministic_mismatch_representation["classifier_embedding"]
                )["integrity_logits"]
            finally:
                self.alignment_model.train(alignment_was_training)
                self.classification_model.train(classifier_was_training)

            mismatch_mask = compatibility_targets.to(
                device=deterministic_mismatch_logits.device
            ) < 0.5
            transition_loss = harmful_transition_loss(
                matched_logits=matched_logits,
                mismatched_logits=deterministic_mismatch_logits,
                integrity_targets=batch.integrity_targets,
                mismatch_mask=mismatch_mask,
            )

        total_loss = (
            self.alignment_loss_weight * alignment_loss
            + self.classification_loss_weight * classification_losses["loss"]
            + float(quality_loss_weight) * quality_loss
            + float(compatibility_loss_weight) * compatibility_loss
            + float(calibration_loss_weight) * calibration_loss
            + float(utility_loss_weight) * utility_loss
            + float(transition_loss_weight) * transition_loss
        )

        metrics = compute_binary_integrity_metrics(
            classification_outputs["integrity_logits"],
            batch.integrity_targets,
        )

        alignment_outputs = {
            "mode": self.mode,
            "aligned_text": representation_outputs["aligned_text"],
            "aligned_vision": representation_outputs["aligned_vision"],
            "fused_embedding": representation_outputs["fused_embedding"],
            "alignment_loss": alignment_loss,
            "text_quality": representation_outputs["text_quality"],
            "vision_quality": representation_outputs["vision_quality"],
            "compatibility_score": representation_outputs["compatibility_score"],
        }

        return {
            "loss": total_loss,
            "alignment_loss": alignment_loss,
            "classification_loss": classification_losses["loss"],
            "quality_loss": quality_loss,
            "text_quality_loss": text_quality_loss,
            "vision_quality_loss": vision_quality_loss,
            "compatibility_loss": compatibility_loss,
            "calibration_loss": calibration_loss,
            "utility_loss": utility_loss,
            "utility_target_outputs": utility_target_outputs,
            "v031_control": representation_outputs["v031_control"],
            "v033_control": representation_outputs["v033_control"],
            "transition_loss": transition_loss,
            "integrity_loss": classification_losses["integrity_loss"],
            "threat_loss": None,
            "alignment_outputs": alignment_outputs,
            "classification_outputs": classification_outputs,
            "metrics": metrics,
        }

    def train_step(
        self,
        batch: BinaryIntegrityBatch,
        *,
        quality_targets: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        quality_loss_weight: float = 0.0,
        quality_sample_weights: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        compatibility_targets: Optional[torch.Tensor] = None,
        compatibility_loss_weight: float = 0.0,
        calibration_loss_weight: float = 0.0,
        utility_loss_weight: float = 0.0,
        transition_reference_batch: Optional[BinaryIntegrityBatch] = None,
        transition_loss_weight: float = 0.0,
    ) -> Dict[str, float]:
        """Execute one optimization step, including M4q/M4qc auxiliaries."""
        self.alignment_model.train()
        self.classification_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        v033_selector_before = None
        v033_selector_parameters = []
        if is_v033_utility_supervised_architecture(self.fusion_architecture):
            controller = (
                self.alignment_model.utility_supervised_intervention_controller
            )
            if controller is None:
                raise RuntimeError("M4qusli controller is unavailable.")
            v033_selector_parameters = list(
                controller.utility_selector.parameters()
            )
            v033_selector_before = [
                parameter.detach().clone()
                for parameter in v033_selector_parameters
            ]

        outputs = self.forward_batch(
            batch,
            compute_alignment_loss=True,
            quality_targets=quality_targets,
            quality_loss_weight=quality_loss_weight,
            quality_sample_weights=quality_sample_weights,
            compatibility_targets=compatibility_targets,
            compatibility_loss_weight=compatibility_loss_weight,
            calibration_loss_weight=calibration_loss_weight,
            utility_loss_weight=utility_loss_weight,
            transition_reference_batch=transition_reference_batch,
            transition_loss_weight=transition_loss_weight,
        )
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss detected.")
        loss.backward()

        v033_selector_gradient_norm = 0.0
        if v033_selector_parameters:
            grad_sq = 0.0
            for parameter in v033_selector_parameters:
                if parameter.grad is not None:
                    grad_sq += float(
                        parameter.grad.detach().pow(2).sum().cpu().item()
                    )
            v033_selector_gradient_norm = grad_sq ** 0.5

        if self.gradient_clip_norm is not None:
            parameters = list(self.alignment_model.parameters()) + list(
                self.classification_model.parameters()
            )
            nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self.gradient_clip_norm,
            )

        self.optimizer.step()
        self.state.global_step += 1

        v033_selector_parameter_l2_movement = 0.0
        if v033_selector_before is not None:
            movement_sq = 0.0
            for before, after in zip(
                v033_selector_before,
                v033_selector_parameters,
            ):
                movement_sq += float(
                    (after.detach() - before).pow(2).sum().cpu().item()
                )
            v033_selector_parameter_l2_movement = movement_sq ** 0.5

        result = {
            "loss": float(loss.detach().cpu().item()),
            "alignment_loss": float(outputs["alignment_loss"].detach().cpu().item()),
            "classification_loss": float(
                outputs["classification_loss"].detach().cpu().item()
            ),
            "quality_loss": float(outputs["quality_loss"].detach().cpu().item()),
            "text_quality_loss": float(
                outputs["text_quality_loss"].detach().cpu().item()
            ),
            "vision_quality_loss": float(
                outputs["vision_quality_loss"].detach().cpu().item()
            ),
            "compatibility_loss": float(
                outputs["compatibility_loss"].detach().cpu().item()
            ),
            "calibration_loss": float(
                outputs["calibration_loss"].detach().cpu().item()
            ),
            "utility_loss": float(
                outputs["utility_loss"].detach().cpu().item()
            ),
            "effective_utility_loss_weight": (
                float(utility_loss_weight)
                if is_v033_utility_supervised_architecture(
                    self.fusion_architecture
                )
                else 0.0
            ),
            "selector_gradient_norm": v033_selector_gradient_norm,
            "selector_parameter_l2_movement": (
                v033_selector_parameter_l2_movement
            ),
            "utility_probability_mean": (
                float(
                    outputs["v033_control"].utility_probability
                    .detach().mean().cpu().item()
                )
                if outputs.get("v033_control") is not None
                else 0.0
            ),
            "active_intervention_rate": (
                float(
                    outputs["v033_control"].active_intervention_indicator
                    .detach().mean().cpu().item()
                )
                if outputs.get("v033_control") is not None
                else 0.0
            ),
            "delta_u_mean": (
                float(
                    outputs["utility_target_outputs"]["delta_u"]
                    .detach().mean().cpu().item()
                )
                if outputs.get("utility_target_outputs") is not None
                else 0.0
            ),
            "delta_u_std": (
                float(
                    outputs["utility_target_outputs"]["delta_u"]
                    .detach().float().std(unbiased=False).cpu().item()
                )
                if outputs.get("utility_target_outputs") is not None
                else 0.0
            ),
            "u_target_mean": (
                float(
                    outputs["utility_target_outputs"]["u_target"]
                    .detach().mean().cpu().item()
                )
                if outputs.get("utility_target_outputs") is not None
                else 0.0
            ),
            "transition_loss": float(
                outputs["transition_loss"].detach().cpu().item()
            ),
            "integrity_loss": float(outputs["integrity_loss"].detach().cpu().item()),
        }
        result.update(outputs["metrics"])
        self.state.metrics = result.copy()
        return result


# =====================================================================
# v0.27 M4q corruption-aware supervision
# =====================================================================

M4Q_SEVERITIES = (0.25, 0.50, 0.75, 1.00)


def _local_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def prepare_m4q_training_batch(
    batch: BinaryIntegrityBatch,
    *,
    text_feature_std: torch.Tensor,
    vision_feature_std: torch.Tensor,
    seed: int,
) -> tuple[BinaryIntegrityBatch, tuple[torch.Tensor, torch.Tensor], Dict[str, int]]:
    """
    Apply the frozen v0.27 training corruption mixture per sample.

    Mixture probabilities:
        clean 0.40, text-quality 0.20, vision-quality 0.20, mismatch 0.20.

    Within text/vision quality corruption:
        Gaussian 0.40, attenuation 0.40, zero 0.20.

    Gaussian/attenuation severities are sampled uniformly from
    {0.25, 0.50, 0.75, 1.00}. Permutation mismatch changes compatibility but
    leaves both intrinsic-quality targets at 1.0. No simultaneous bimodal
    intrinsic-quality corruption is generated.
    """
    if batch.batch_size < 1:
        raise ValueError("M4q training batch must contain at least one sample.")

    text = batch.text_embeddings.clone()
    vision = batch.vision_embeddings.clone()
    batch_size = batch.batch_size

    text_targets = torch.ones((batch_size, 1), dtype=text.dtype, device=text.device)
    vision_targets = torch.ones(
        (batch_size, 1), dtype=vision.dtype, device=vision.device
    )

    generator = _local_generator(seed)
    condition_draw = torch.rand(batch_size, generator=generator)
    family_draw = torch.rand(batch_size, generator=generator)
    modality_draw = torch.rand(batch_size, generator=generator)
    severity_index = torch.randint(
        low=0,
        high=len(M4Q_SEVERITIES),
        size=(batch_size,),
        generator=generator,
    )

    clean_mask = condition_draw < 0.40
    text_mask = (condition_draw >= 0.40) & (condition_draw < 0.60)
    vision_mask = (condition_draw >= 0.60) & (condition_draw < 0.80)
    mismatch_mask = condition_draw >= 0.80

    counts = {
        "clean": int(clean_mask.sum().item()),
        "text_quality": int(text_mask.sum().item()),
        "vision_quality": int(vision_mask.sum().item()),
        "mismatch": int(mismatch_mask.sum().item()),
        "gaussian_noise": 0,
        "attenuation": 0,
        "zero_dropout": 0,
    }

    def apply_quality_corruption(
        representation: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        feature_std: torch.Tensor,
        seed_offset: int,
    ) -> None:
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        for local_order, index_cpu in enumerate(indices.tolist()):
            index = int(index_cpu)
            row = representation[index:index + 1]
            family_value = float(family_draw[index].item())
            severity = M4Q_SEVERITIES[int(severity_index[index].item())]

            if family_value < 0.40:
                result = gaussian_noise(
                    row,
                    feature_std,
                    severity=severity,
                    seed=int(seed) + seed_offset + local_order,
                )
                target[index, 0] = 1.0 - severity
                counts["gaussian_noise"] += 1
            elif family_value < 0.80:
                result = attenuation(row, severity=severity)
                target[index, 0] = 1.0 - severity
                counts["attenuation"] += 1
            else:
                result = zero_dropout(row)
                target[index, 0] = 0.0
                counts["zero_dropout"] += 1

            representation[index:index + 1] = result.corrupted

    apply_quality_corruption(
        text,
        text_targets,
        text_mask,
        text_feature_std,
        10_000,
    )
    apply_quality_corruption(
        vision,
        vision_targets,
        vision_mask,
        vision_feature_std,
        20_000,
    )

    if mismatch_mask.any():
        if batch_size < 2:
            raise ValueError(
                "M4q permutation mismatch requires batch_size >= 2 whenever "
                "the sampled batch contains a mismatch condition."
            )
        permutation = deterministic_derangement_indices(
            batch_size,
            seed=int(seed) + 30_000,
        )
        mismatch_indices = torch.nonzero(
            mismatch_mask, as_tuple=False
        ).reshape(-1)
        text_mismatch = mismatch_indices[
            modality_draw[mismatch_indices] < 0.5
        ]
        vision_mismatch = mismatch_indices[
            modality_draw[mismatch_indices] >= 0.5
        ]
        if text_mismatch.numel() > 0:
            donor = permutation[text_mismatch].to(text.device)
            receiver = text_mismatch.to(text.device)
            text[receiver] = batch.text_embeddings.index_select(0, donor)
        if vision_mismatch.numel() > 0:
            donor = permutation[vision_mismatch].to(vision.device)
            receiver = vision_mismatch.to(vision.device)
            vision[receiver] = batch.vision_embeddings.index_select(0, donor)

    corrupted_batch = copy.copy(batch)
    corrupted_batch.text_embeddings = text
    corrupted_batch.vision_embeddings = vision

    return corrupted_batch, (text_targets, vision_targets), counts


def prepare_m4qc_training_batch(
    batch: BinaryIntegrityBatch,
    *,
    text_feature_std: torch.Tensor,
    vision_feature_std: torch.Tensor,
    seed: int,
) -> tuple[
    BinaryIntegrityBatch,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    Dict[str, int],
]:
    """
    Apply the frozen Step 11A M4qc training mixture per sample.

    The primary condition probabilities and intrinsic-quality corruption
    families are identical to M4q. Compatibility supervision is defined for
    every sample:

        clean / text-quality / vision-quality -> c* = 1
        mismatch                            -> c* = 0

    Mismatch construction follows Step 11A: text remains in original order,
    while mismatched rows receive vision donors from one deterministic cyclic
    batch derangement using a single non-zero offset k.
    """
    if batch.batch_size < 1:
        raise ValueError("M4qc training batch must contain at least one sample.")

    text = batch.text_embeddings.clone()
    vision = batch.vision_embeddings.clone()
    batch_size = batch.batch_size

    text_targets = torch.ones(
        (batch_size, 1),
        dtype=text.dtype,
        device=text.device,
    )
    vision_targets = torch.ones(
        (batch_size, 1),
        dtype=vision.dtype,
        device=vision.device,
    )
    compatibility_targets = torch.ones(
        (batch_size, 1),
        dtype=text.dtype,
        device=text.device,
    )

    generator = _local_generator(seed)
    condition_draw = torch.rand(batch_size, generator=generator)
    family_draw = torch.rand(batch_size, generator=generator)
    severity_index = torch.randint(
        low=0,
        high=len(M4Q_SEVERITIES),
        size=(batch_size,),
        generator=generator,
    )

    clean_mask = condition_draw < 0.40
    text_mask = (condition_draw >= 0.40) & (condition_draw < 0.60)
    vision_mask = (condition_draw >= 0.60) & (condition_draw < 0.80)
    mismatch_mask = condition_draw >= 0.80

    counts = {
        "clean": int(clean_mask.sum().item()),
        "text_quality": int(text_mask.sum().item()),
        "vision_quality": int(vision_mask.sum().item()),
        "mismatch": int(mismatch_mask.sum().item()),
        "gaussian_noise": 0,
        "attenuation": 0,
        "zero_dropout": 0,
    }

    def apply_quality_corruption(
        representation: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        feature_std: torch.Tensor,
        seed_offset: int,
    ) -> None:
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        for local_order, index_cpu in enumerate(indices.tolist()):
            index = int(index_cpu)
            row = representation[index:index + 1]
            family_value = float(family_draw[index].item())
            severity = M4Q_SEVERITIES[int(severity_index[index].item())]

            if family_value < 0.40:
                result = gaussian_noise(
                    row,
                    feature_std,
                    severity=severity,
                    seed=int(seed) + seed_offset + local_order,
                )
                target[index, 0] = 1.0 - severity
                counts["gaussian_noise"] += 1
            elif family_value < 0.80:
                result = attenuation(row, severity=severity)
                target[index, 0] = 1.0 - severity
                counts["attenuation"] += 1
            else:
                result = zero_dropout(row)
                target[index, 0] = 0.0
                counts["zero_dropout"] += 1

            representation[index:index + 1] = result.corrupted

    apply_quality_corruption(
        text,
        text_targets,
        text_mask,
        text_feature_std,
        10_000,
    )
    apply_quality_corruption(
        vision,
        vision_targets,
        vision_mask,
        vision_feature_std,
        20_000,
    )

    if mismatch_mask.any():
        if batch_size < 2:
            raise ValueError(
                "M4qc cyclic mismatch requires batch_size >= 2 whenever "
                "the sampled batch contains a mismatch condition."
            )

        offset = int(
            torch.randint(
                low=1,
                high=batch_size,
                size=(1,),
                generator=generator,
            ).item()
        )
        all_indices = torch.arange(batch_size, dtype=torch.long)
        donors = (all_indices + offset) % batch_size
        receivers_cpu = torch.nonzero(
            mismatch_mask,
            as_tuple=False,
        ).reshape(-1)

        receivers = receivers_cpu.to(vision.device)
        donor_indices = donors[receivers_cpu].to(vision.device)
        vision[receivers] = batch.vision_embeddings.index_select(
            0,
            donor_indices,
        )
        compatibility_targets[
            receivers.to(compatibility_targets.device),
            0,
        ] = 0.0

    corrupted_batch = copy.copy(batch)
    corrupted_batch.text_embeddings = text
    corrupted_batch.vision_embeddings = vision

    return (
        corrupted_batch,
        (text_targets, vision_targets),
        compatibility_targets,
        counts,
    )


def prepare_m4qcs_training_batch(
    batch: BinaryIntegrityBatch,
    *,
    text_feature_std: torch.Tensor,
    vision_feature_std: torch.Tensor,
    seed: int,
) -> tuple[
    BinaryIntegrityBatch,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    Dict[str, int],
]:
    """Prepare the frozen v0.28 mixture and text-Gaussian loss weights."""
    prepared, quality_targets, compatibility_targets, counts = (
        prepare_m4qc_training_batch(
            batch,
            text_feature_std=text_feature_std,
            vision_feature_std=vision_feature_std,
            seed=seed,
        )
    )
    generator = _local_generator(seed)
    condition_draw = torch.rand(batch.batch_size, generator=generator)
    family_draw = torch.rand(batch.batch_size, generator=generator)
    # Consume the same severity draw used by prepare_m4qc_training_batch.
    torch.randint(
        low=0,
        high=len(M4Q_SEVERITIES),
        size=(batch.batch_size,),
        generator=generator,
    )
    text_gaussian = (
        (condition_draw >= 0.40)
        & (condition_draw < 0.60)
        & (family_draw < 0.40)
    )
    text_weights = torch.ones_like(quality_targets[0])
    text_weights[text_gaussian.to(text_weights.device), 0] = 2.0
    vision_weights = torch.ones_like(quality_targets[1])
    counts = dict(counts)
    counts["text_gaussian_weight_2"] = int(text_gaussian.sum().item())
    return (
        prepared,
        quality_targets,
        compatibility_targets,
        (text_weights, vision_weights),
        counts,
    )


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
    *,
    quality_loss_weight: float = 0.0,
    compatibility_loss_weight: float = 0.0,
    transition_loss_weight: float = 0.0,
    text_feature_std: Optional[torch.Tensor] = None,
    vision_feature_std: Optional[torch.Tensor] = None,
) -> Dict:
    weighted_total_loss = 0.0
    weighted_alignment_loss = 0.0
    weighted_classification_loss = 0.0
    weighted_quality_loss = 0.0
    weighted_text_quality_loss = 0.0
    weighted_vision_quality_loss = 0.0
    weighted_compatibility_loss = 0.0
    weighted_utility_loss = 0.0
    weighted_selector_gradient_norm = 0.0
    weighted_selector_parameter_l2_movement = 0.0
    weighted_utility_probability_mean = 0.0
    weighted_active_intervention_rate = 0.0
    weighted_delta_u_mean = 0.0
    weighted_delta_u_std = 0.0
    weighted_u_target_mean = 0.0
    weighted_transition_loss = 0.0
    total_samples = 0
    batch_count = 0
    corruption_counts = Counter()

    for batch_index, batch in enumerate(iter_training_batches(
        full_batch=full_batch,
        batch_size=batch_size,
        epoch=epoch,
        seed=seed,
    )):
        quality_targets = None
        quality_sample_weights = None
        compatibility_targets = None
        training_batch = batch

        if trainer.fusion_architecture in {
            "quality_supervised",
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }:
            if text_feature_std is None or vision_feature_std is None:
                raise RuntimeError(
                    "M4q/M4qc training requires training-cache feature "
                    "statistics."
                )
            corruption_seed = (
                int(seed)
                + 27_000_000
                + int(epoch) * 100_000
                + int(batch_index)
            )

            if trainer.fusion_architecture == "quality_supervised":
                (
                    training_batch,
                    quality_targets,
                    counts,
                ) = prepare_m4q_training_batch(
                    batch,
                    text_feature_std=text_feature_std,
                    vision_feature_std=vision_feature_std,
                    seed=corruption_seed,
                )
            elif trainer.fusion_architecture in {
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }:
                (
                    training_batch,
                    quality_targets,
                    compatibility_targets,
                    quality_sample_weights,
                    counts,
                ) = prepare_m4qcs_training_batch(
                    batch,
                    text_feature_std=text_feature_std,
                    vision_feature_std=vision_feature_std,
                    seed=corruption_seed,
                )
            else:
                (
                    training_batch,
                    quality_targets,
                    compatibility_targets,
                    counts,
                ) = prepare_m4qc_training_batch(
                    batch,
                    text_feature_std=text_feature_std,
                    vision_feature_std=vision_feature_std,
                    seed=corruption_seed,
                )
            corruption_counts.update(counts)

        result = trainer.train_step(
            training_batch,
            quality_targets=quality_targets,
            quality_loss_weight=(
                quality_loss_weight
                if trainer.fusion_architecture in {
                    "quality_supervised",
                    "quality_compatibility_supervised",
                    "quality_compatibility_fusion",
                    "quality_compatibility_selective_weights",
                    "quality_compatibility_selective_interaction",
                    "quality_compatibility_selective_fusion",
                    "quality_compatibility_graded_weights",
                    "quality_compatibility_transition_control",
                    "quality_compatibility_graded_transition_fusion",
                    "quality_compatibility_selective_intervention_weights",
                    "quality_compatibility_selective_intervention_transition",
                    "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
                }
                else 0.0
            ),
            quality_sample_weights=quality_sample_weights,
            compatibility_targets=compatibility_targets,
            compatibility_loss_weight=(
                compatibility_loss_weight
                if trainer.fusion_architecture
                in {
                    "quality_compatibility_supervised",
                    "quality_compatibility_fusion",
                    "quality_compatibility_selective_weights",
                    "quality_compatibility_selective_interaction",
                    "quality_compatibility_selective_fusion",
                    "quality_compatibility_graded_weights",
                    "quality_compatibility_transition_control",
                    "quality_compatibility_graded_transition_fusion",
                    "quality_compatibility_selective_intervention_weights",
                    "quality_compatibility_selective_intervention_transition",
                    "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
                }
                else 0.0
            ),
            calibration_loss_weight=(
                V031_CALIBRATION_LOSS_WEIGHT
                if (
                    is_v031_calibrated_intervention_architecture(
                        trainer.fusion_architecture
                    )
                    and v031_calibrated_intervention_mode(
                        trainer.fusion_architecture
                    ) in {"calibration_only", "combined"}
                )
                else 0.0
            ),
            utility_loss_weight=(
                V031_UTILITY_LOSS_WEIGHT
                if (
                    is_v031_calibrated_intervention_architecture(
                        trainer.fusion_architecture
                    )
                    and v031_calibrated_intervention_mode(
                        trainer.fusion_architecture
                    ) in {"utility_only", "combined"}
                )
                else (
                    V033_UTILITY_LOSS_WEIGHT
                    if is_v033_utility_supervised_architecture(
                        trainer.fusion_architecture
                    )
                    else 0.0
                )
            ),
            transition_reference_batch=(
                batch
                if trainer.fusion_architecture in {
                    "quality_compatibility_transition_control",
                    "quality_compatibility_graded_transition_fusion",
                }
                and transition_loss_weight > 0.0
                else None
            ),
            transition_loss_weight=(
                transition_loss_weight
                if trainer.fusion_architecture in {
                    "quality_compatibility_transition_control",
                    "quality_compatibility_graded_transition_fusion",
                }
                else 0.0
            ),
        )

        current_batch_size = batch.batch_size
        total_samples += current_batch_size
        batch_count += 1
        weighted_total_loss += result["loss"] * current_batch_size
        weighted_alignment_loss += result["alignment_loss"] * current_batch_size
        weighted_classification_loss += (
            result["classification_loss"] * current_batch_size
        )
        weighted_quality_loss += result["quality_loss"] * current_batch_size
        weighted_text_quality_loss += (
            result["text_quality_loss"] * current_batch_size
        )
        weighted_vision_quality_loss += (
            result["vision_quality_loss"] * current_batch_size
        )
        weighted_compatibility_loss += (
            result["compatibility_loss"] * current_batch_size
        )
        weighted_utility_loss += result["utility_loss"] * current_batch_size
        weighted_selector_gradient_norm += (
            result["selector_gradient_norm"] * current_batch_size
        )
        weighted_selector_parameter_l2_movement += (
            result["selector_parameter_l2_movement"] * current_batch_size
        )
        weighted_utility_probability_mean += (
            result["utility_probability_mean"] * current_batch_size
        )
        weighted_active_intervention_rate += (
            result["active_intervention_rate"] * current_batch_size
        )
        weighted_delta_u_mean += result["delta_u_mean"] * current_batch_size
        weighted_delta_u_std += result["delta_u_std"] * current_batch_size
        weighted_u_target_mean += result["u_target_mean"] * current_batch_size
        weighted_transition_loss += result["transition_loss"] * current_batch_size

    if total_samples == 0:
        raise RuntimeError("Training epoch contained zero samples.")

    return {
        "sample_count": total_samples,
        "batch_count": batch_count,
        "total_loss": weighted_total_loss / total_samples,
        "alignment_loss": weighted_alignment_loss / total_samples,
        "classification_loss": weighted_classification_loss / total_samples,
        "quality_loss": weighted_quality_loss / total_samples,
        "text_quality_loss": weighted_text_quality_loss / total_samples,
        "vision_quality_loss": weighted_vision_quality_loss / total_samples,
        "compatibility_loss": weighted_compatibility_loss / total_samples,
        "utility_loss": weighted_utility_loss / total_samples,
        "selector_gradient_norm": (
            weighted_selector_gradient_norm / total_samples
        ),
        "selector_parameter_l2_movement": (
            weighted_selector_parameter_l2_movement / total_samples
        ),
        "utility_probability_mean": (
            weighted_utility_probability_mean / total_samples
        ),
        "active_intervention_rate": (
            weighted_active_intervention_rate / total_samples
        ),
        "delta_u_mean": weighted_delta_u_mean / total_samples,
        "delta_u_std": weighted_delta_u_std / total_samples,
        "u_target_mean": weighted_u_target_mean / total_samples,
        "effective_utility_loss_weight": (
            V033_UTILITY_LOSS_WEIGHT
            if is_v033_utility_supervised_architecture(
                trainer.fusion_architecture
            )
            else 0.0
        ),
        "transition_loss": weighted_transition_loss / total_samples,
        "corruption_counts": dict(sorted(corruption_counts.items())),
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

            "fusion_architecture": (
                configuration.get(
                    "fusion_architecture",
                    "legacy",
                )
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
# Reproducibility / initialization audit helpers
# =====================================================================

def module_state_fingerprint(module) -> str:
    """Return a deterministic SHA-256 fingerprint of a module state_dict."""

    digest = hashlib.sha256()

    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))

        value = (
            tensor
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            value.numpy().tobytes()
        )

    return digest.hexdigest()


def collect_component_fingerprints(
    alignment_model,
    classification_model,
    mode: str,
    fusion_architecture: str,
) -> Dict[str, str]:
    """Fingerprint common and architecture-specific trainable components."""

    fingerprints = {
        "text_projection": module_state_fingerprint(
            alignment_model.text_projection
        ),
        "vision_projection": module_state_fingerprint(
            alignment_model.vision_projection
        ),
        "classifier": module_state_fingerprint(
            classification_model
        ),
    }

    if mode == "multimodal":

        if fusion_architecture == "legacy":
            fingerprints["legacy_fusion"] = (
                module_state_fingerprint(
                    alignment_model.fusion
                )
            )

        elif fusion_architecture == "interaction_only":

            if alignment_model.interaction_fusion is None:
                raise RuntimeError(
                    "Interaction-only fingerprint requested, but "
                    "the interaction fusion block is unavailable."
                )

            fingerprints["interaction_fusion"] = (
                module_state_fingerprint(
                    alignment_model.interaction_fusion
                )
            )

        elif fusion_architecture == "gated_interaction":

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "Gated-interaction fingerprint requested, but "
                    "the gated interaction fusion block is unavailable."
                )

            fingerprints["gated_interaction_fusion"] = (
                module_state_fingerprint(
                    alignment_model.gated_interaction_fusion
                )
            )

            fingerprints["gated_base_fusion"] = (
                module_state_fingerprint(
                    alignment_model
                    .gated_interaction_fusion
                    .gated_fusion
                )
            )

        elif fusion_architecture == "reliability_only":

            if alignment_model.reliability_only_fusion is None:
                raise RuntimeError(
                    "Reliability-only fingerprint requested, but "
                    "the reliability-only fusion block is unavailable."
                )

            fingerprints["reliability_only_fusion"] = (
                module_state_fingerprint(
                    alignment_model.reliability_only_fusion
                )
            )

            fingerprints["legacy_fusion"] = (
                module_state_fingerprint(
                    alignment_model.fusion
                )
            )

        elif fusion_architecture == "interaction_reliability":

            if alignment_model.interaction_reliability_fusion is None:
                raise RuntimeError(
                    "Interaction-reliability fingerprint requested, but "
                    "the interaction reliability fusion block is unavailable."
                )

            fingerprints["interaction_reliability_fusion"] = (
                module_state_fingerprint(
                    alignment_model.interaction_reliability_fusion
                )
            )
            fingerprints["legacy_fusion"] = (
                module_state_fingerprint(alignment_model.fusion)
            )

        elif fusion_architecture == "quality_supervised":

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "M4q fingerprint requested, but the gated-interaction "
                    "fusion block is unavailable."
                )
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError("M4q text quality estimator is unavailable.")
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError("M4q vision quality estimator is unavailable.")

            fingerprints["gated_interaction_fusion"] = module_state_fingerprint(
                alignment_model.gated_interaction_fusion
            )
            fingerprints["gated_base_fusion"] = module_state_fingerprint(
                alignment_model.gated_interaction_fusion.gated_fusion
            )
            fingerprints["text_quality_estimator"] = module_state_fingerprint(
                alignment_model.text_quality_estimator
            )
            fingerprints["vision_quality_estimator"] = module_state_fingerprint(
                alignment_model.vision_quality_estimator
            )

        elif fusion_architecture in {
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }:

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "M4qc fingerprint requested, but the gated-interaction "
                    "fusion block is unavailable."
                )
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError("M4qc text quality estimator is unavailable.")
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError("M4qc vision quality estimator is unavailable.")
            if alignment_model.compatibility_estimator is None:
                raise RuntimeError("M4qc compatibility estimator is unavailable.")

            fingerprints["gated_interaction_fusion"] = module_state_fingerprint(
                alignment_model.gated_interaction_fusion
            )
            fingerprints["gated_base_fusion"] = module_state_fingerprint(
                alignment_model.gated_interaction_fusion.gated_fusion
            )
            fingerprints["text_quality_estimator"] = module_state_fingerprint(
                alignment_model.text_quality_estimator
            )
            fingerprints["vision_quality_estimator"] = module_state_fingerprint(
                alignment_model.vision_quality_estimator
            )
            fingerprints["compatibility_estimator"] = module_state_fingerprint(
                alignment_model.compatibility_estimator
            )
            if is_v033_utility_supervised_architecture(fusion_architecture):
                if (
                    alignment_model
                    .utility_supervised_intervention_controller is None
                ):
                    raise RuntimeError(
                        "M4qusli controller unavailable for fingerprinting."
                    )
                fingerprints["v033_utility_selector"] = (
                    module_state_fingerprint(
                        alignment_model
                        .utility_supervised_intervention_controller
                        .utility_selector
                    )
                )

        elif fusion_architecture == "evidence_aware":

            if alignment_model.evidence_integration is None:
                raise RuntimeError(
                    "Evidence-aware fingerprint requested, but "
                    "the evidence integration block is unavailable."
                )

            fingerprints["evidence_integration"] = (
                module_state_fingerprint(
                    alignment_model.evidence_integration
                )
            )

    return fingerprints


def compare_fingerprints(
    initial: Dict[str, str],
    final: Dict[str, str],
) -> Dict[str, bool]:
    """Report whether each tracked component changed during training."""

    return {
        key: initial.get(key) != final.get(key)
        for key in sorted(
            set(initial) | set(final)
        )
    }


# =====================================================================
# Parameter accounting
# =====================================================================

def effective_parameter_counts(
    alignment_model,
    classification_model,
    mode: str,
    fusion_architecture: str,
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
                alignment_model.text_projection
            )
            + count_parameters(
                alignment_model.vision_projection
            )
        )

        if fusion_architecture == "legacy":

            representation_count += count_parameters(
                alignment_model.fusion
            )

        elif fusion_architecture == "interaction_only":

            if alignment_model.interaction_fusion is None:
                raise RuntimeError(
                    "Interaction-only parameter accounting requires "
                    "alignment_model.interaction_fusion."
                )

            representation_count += count_parameters(
                alignment_model.interaction_fusion
            )

        elif fusion_architecture == "gated_interaction":

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "Gated-interaction parameter accounting requires "
                    "alignment_model.gated_interaction_fusion."
                )

            representation_count += count_parameters(
                alignment_model.gated_interaction_fusion
            )

        elif fusion_architecture == "reliability_only":

            if alignment_model.reliability_only_fusion is None:
                raise RuntimeError(
                    "Reliability-only parameter accounting requires "
                    "alignment_model.reliability_only_fusion."
                )

            representation_count += count_parameters(
                alignment_model.fusion
            )

            representation_count += count_parameters(
                alignment_model.reliability_only_fusion
            )

        elif fusion_architecture == "interaction_reliability":

            if alignment_model.interaction_reliability_fusion is None:
                raise RuntimeError(
                    "Interaction-reliability parameter accounting requires "
                    "alignment_model.interaction_reliability_fusion."
                )

            representation_count += count_parameters(alignment_model.fusion)
            representation_count += count_parameters(
                alignment_model.interaction_reliability_fusion
            )

        elif fusion_architecture == "quality_supervised":

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "M4q parameter accounting requires "
                    "alignment_model.gated_interaction_fusion."
                )
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError(
                    "M4q parameter accounting requires text_quality_estimator."
                )
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError(
                    "M4q parameter accounting requires vision_quality_estimator."
                )
            representation_count += count_parameters(
                alignment_model.gated_interaction_fusion
            )
            representation_count += count_parameters(
                alignment_model.text_quality_estimator
            )
            representation_count += count_parameters(
                alignment_model.vision_quality_estimator
            )

        elif fusion_architecture in {
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }:

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError(
                    "M4qc parameter accounting requires "
                    "alignment_model.gated_interaction_fusion."
                )
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError(
                    "M4qc parameter accounting requires text_quality_estimator."
                )
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError(
                    "M4qc parameter accounting requires vision_quality_estimator."
                )
            if alignment_model.compatibility_estimator is None:
                raise RuntimeError(
                    "M4qc parameter accounting requires compatibility_estimator."
                )
            representation_count += count_parameters(
                alignment_model.gated_interaction_fusion
            )
            representation_count += count_parameters(
                alignment_model.text_quality_estimator
            )
            representation_count += count_parameters(
                alignment_model.vision_quality_estimator
            )
            representation_count += count_parameters(
                alignment_model.compatibility_estimator
            )
            if is_v033_utility_supervised_architecture(fusion_architecture):
                if (
                    alignment_model
                    .utility_supervised_intervention_controller is None
                ):
                    raise RuntimeError(
                        "M4qusli parameter accounting requires controller."
                    )
                representation_count += count_parameters(
                    alignment_model
                    .utility_supervised_intervention_controller
                    .utility_selector
                )

        else:

            if alignment_model.evidence_integration is None:

                raise RuntimeError(
                    "Evidence-aware parameter accounting requires "
                    "alignment_model.evidence_integration."
                )

            representation_count += count_parameters(
                alignment_model.evidence_integration
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
            args.mode,
            args.fusion_architecture,
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
        "Fusion architecture:",
        (
            args.fusion_architecture
            if args.mode == "multimodal"
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

            evidence_aware=(
                args.fusion_architecture
                == "evidence_aware"
            ),

            interaction_only=(
                args.fusion_architecture
                == "interaction_only"
            ),

            gated_interaction=(
                args.fusion_architecture
                == "gated_interaction"
            ),

            reliability_only=(
                args.fusion_architecture
                == "reliability_only"
            ),

            interaction_reliability=(
                args.fusion_architecture
                == "interaction_reliability"
            ),

            quality_supervised=(
                args.fusion_architecture
                == "quality_supervised"
            ),

            quality_compatibility_supervised=(
                args.fusion_architecture == "quality_compatibility_supervised"
            ),

            quality_compatibility_fusion=(
                args.fusion_architecture == "quality_compatibility_fusion"
            ),

            quality_compatibility_selective_fusion={
                "quality_compatibility_selective_weights": "weights_only",
                "quality_compatibility_selective_interaction": "interaction_only",
                "quality_compatibility_selective_fusion": "combined",
            }.get(args.fusion_architecture),

            quality_compatibility_graded_transition_fusion={
                "quality_compatibility_graded_weights": "graded_weights_only",
                "quality_compatibility_transition_control": "transition_only",
                "quality_compatibility_graded_transition_fusion": "combined",
            }.get(args.fusion_architecture),

            quality_compatibility_selective_intervention=(
                v030_selective_intervention_mode(args.fusion_architecture)
            ),
            quality_compatibility_calibrated_intervention=(
                v031_calibrated_intervention_mode(args.fusion_architecture)
            ),
            quality_compatibility_utility_supervised_intervention=(
                is_v033_utility_supervised_architecture(
                    args.fusion_architecture
                )
            ),
            v031_utility_initialization_seed=args.seed,
            v033_utility_initialization_seed=args.seed,
        )
    )

    # M1b gate-initialization control:
    # copy the historical M0 gate state into M1b's active gated base so
    # M0 and M1b begin with exactly the same gated-fusion parameters.
    if args.fusion_architecture in {
        "gated_interaction",
        "quality_supervised",
        "quality_compatibility_supervised",
        "quality_compatibility_fusion",
        "quality_compatibility_selective_weights",
        "quality_compatibility_selective_interaction",
        "quality_compatibility_selective_fusion",
        "quality_compatibility_graded_weights",
        "quality_compatibility_transition_control",
        "quality_compatibility_graded_transition_fusion",
        "quality_compatibility_selective_intervention_weights",
        "quality_compatibility_selective_intervention_transition",
        "quality_compatibility_selective_intervention",
    }:

        if alignment_model.gated_interaction_fusion is None:
            raise RuntimeError(
                "Gated-interaction architecture was requested but "
                "alignment_model.gated_interaction_fusion is unavailable."
            )

        alignment_model.gated_interaction_fusion.gated_fusion.load_state_dict(
            alignment_model.fusion.state_dict()
        )

    # M2 gate-initialization control:
    # M2 directly reuses alignment_model.fusion as its active legacy gated
    # base. No copied gate is created, so M0 and M2 share the exact same
    # parent-gate initialization under the paired experiment seed.

    # M2b gate-initialization control:
    # M2b also directly reuses alignment_model.fusion as its active legacy
    # gated base; no duplicate gate is created.

    # Paired-initialization control:
    # non-legacy alignment models contain additional modules that consume
    # RNG state. Re-seeding immediately before classifier creation guarantees
    # identical classifier initialization across M0/M1/M1b/M2/M2b/M3 for the same
    # experiment seed.
    classifier_initialization_seed = (
        args.seed + 1_000_003
    )

    set_global_seed(
        classifier_initialization_seed
    )

    random.seed(
        classifier_initialization_seed
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

    # Reset the stochastic training protocol to the experiment seed so that
    # data-order/randomized training behaviour is anchored to --seed rather
    # than to architecture-dependent model-construction RNG consumption.
    set_global_seed(
        args.seed
    )

    random.seed(
        args.seed
    )

    text_feature_std = None
    vision_feature_std = None

    if requires_quality_feature_statistics(args.fusion_architecture):
        text_feature_std = compute_feature_std(
            train_data.text_embeddings,
            unbiased=False,
        )
        vision_feature_std = compute_feature_std(
            train_data.vision_embeddings,
            unbiased=False,
        )

        if args.fusion_architecture == "quality_supervised":
            print("M4q corruption-aware quality supervision: ENABLED")
            print("M4q quality loss weight:", args.quality_weight)
        elif args.fusion_architecture == "quality_compatibility_supervised":
            print("M4qc quality + compatibility supervision: ENABLED")
            print("M4qc quality loss weight:", args.quality_weight)
            print("M4qc compatibility loss weight:", args.compatibility_weight)
        elif args.fusion_architecture == "quality_compatibility_fusion":
            print("M4qcf reliability-informed fusion supervision: ENABLED")
            print("M4qcf quality loss weight:", args.quality_weight)
            print("M4qcf compatibility loss weight:", args.compatibility_weight)
        elif is_v030_selective_intervention_architecture(
            args.fusion_architecture
        ):
            print("M4qesri selective intervention supervision: ENABLED")
            print("M4qesri architecture:", args.fusion_architecture)
            print(
                "M4qesri mode:",
                v030_selective_intervention_mode(args.fusion_architecture),
            )
            print("M4qesri quality loss weight:", args.quality_weight)
            print("M4qesri compatibility loss weight:", args.compatibility_weight)
            print("M4qesri v0.29 transition objective: DISABLED")
        elif is_v033_utility_supervised_architecture(
            args.fusion_architecture
        ):
            print("M4qusli utility-supervised selector learning: ENABLED")
            print("M4qusli quality loss weight:", args.quality_weight)
            print("M4qusli compatibility loss weight:", args.compatibility_weight)
            print("M4qusli utility loss weight:", V033_UTILITY_LOSS_WEIGHT)
            print("M4qusli transition objective: DISABLED")
        else:
            print("M4qcs selective reliability supervision: ENABLED")
            print("M4qcs architecture:", args.fusion_architecture)
            print("M4qcs quality loss weight:", args.quality_weight)
            print("M4qcs compatibility loss weight:", args.compatibility_weight)
        print("v0.27 Gaussian feature scale: training-cache population std")
        print()

    initial_state_fingerprints = (
        collect_component_fingerprints(
            alignment_model=(
                alignment_model
            ),
            classification_model=(
                classification_model
            ),
            mode=(
                args.mode
            ),
            fusion_architecture=(
                args.fusion_architecture
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

        representation_parameters = (
            list(
                alignment_model
                .text_projection
                .parameters()
            )
            + list(
                alignment_model
                .vision_projection
                .parameters()
            )
        )

        if args.fusion_architecture == "legacy":

            representation_parameters += list(
                alignment_model
                .fusion
                .parameters()
            )

        elif args.fusion_architecture == "interaction_only":

            if alignment_model.interaction_fusion is None:

                raise RuntimeError(
                    "Interaction-only fusion was requested but the "
                    "AEGIS interaction fusion block is unavailable."
                )

            representation_parameters += list(
                alignment_model
                .interaction_fusion
                .parameters()
            )

        elif args.fusion_architecture == "gated_interaction":

            if alignment_model.gated_interaction_fusion is None:

                raise RuntimeError(
                    "Gated-interaction fusion was requested but the "
                    "AEGIS gated interaction fusion block is unavailable."
                )

            representation_parameters += list(
                alignment_model
                .gated_interaction_fusion
                .parameters()
            )

        elif args.fusion_architecture == "reliability_only":

            if alignment_model.reliability_only_fusion is None:

                raise RuntimeError(
                    "Reliability-only fusion was requested but the "
                    "AEGIS reliability-only fusion block is unavailable."
                )

            representation_parameters += list(
                alignment_model
                .fusion
                .parameters()
            )

            representation_parameters += list(
                alignment_model
                .reliability_only_fusion
                .parameters()
            )

        elif args.fusion_architecture == "quality_supervised":

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError("M4q gated-interaction fusion is unavailable.")
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError("M4q text quality estimator is unavailable.")
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError("M4q vision quality estimator is unavailable.")

            representation_parameters += list(
                alignment_model.gated_interaction_fusion.parameters()
            )
            representation_parameters += list(
                alignment_model.text_quality_estimator.parameters()
            )
            representation_parameters += list(
                alignment_model.vision_quality_estimator.parameters()
            )

        elif args.fusion_architecture in {
            "quality_compatibility_supervised",
            "quality_compatibility_fusion",
            "quality_compatibility_selective_weights",
            "quality_compatibility_selective_interaction",
            "quality_compatibility_selective_fusion",
            "quality_compatibility_graded_weights",
            "quality_compatibility_transition_control",
            "quality_compatibility_graded_transition_fusion",
            "quality_compatibility_selective_intervention_weights",
            "quality_compatibility_selective_intervention_transition",
            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
        }:

            if alignment_model.gated_interaction_fusion is None:
                raise RuntimeError("M4qc gated-interaction fusion is unavailable.")
            if alignment_model.text_quality_estimator is None:
                raise RuntimeError("M4qc text quality estimator is unavailable.")
            if alignment_model.vision_quality_estimator is None:
                raise RuntimeError("M4qc vision quality estimator is unavailable.")
            if alignment_model.compatibility_estimator is None:
                raise RuntimeError("M4qc compatibility estimator is unavailable.")

            representation_parameters += list(
                alignment_model.gated_interaction_fusion.parameters()
            )
            representation_parameters += list(
                alignment_model.text_quality_estimator.parameters()
            )
            representation_parameters += list(
                alignment_model.vision_quality_estimator.parameters()
            )
            representation_parameters += list(
                alignment_model.compatibility_estimator.parameters()
            )
            if is_v031_calibrated_intervention_architecture(
                args.fusion_architecture
            ):
                if alignment_model.calibrated_utility_controller is None:
                    raise RuntimeError("M4qcesi controller unavailable.")
                representation_parameters += list(
                    alignment_model.calibrated_utility_controller.parameters()
                )
            if is_v033_utility_supervised_architecture(
                args.fusion_architecture
            ):
                if (
                    alignment_model
                    .utility_supervised_intervention_controller is None
                ):
                    raise RuntimeError("M4qusli controller unavailable.")
                representation_parameters += list(
                    alignment_model
                    .utility_supervised_intervention_controller.parameters()
                )

        elif args.fusion_architecture == "interaction_reliability":

            if alignment_model.interaction_reliability_fusion is None:
                raise RuntimeError(
                    "Interaction-reliability fusion was requested but the "
                    "AEGIS interaction reliability residual block is unavailable."
                )

            representation_parameters += list(
                alignment_model.fusion.parameters()
            )
            representation_parameters += list(
                alignment_model.interaction_reliability_fusion.parameters()
            )

        else:

            if alignment_model.evidence_integration is None:

                raise RuntimeError(
                    "Evidence-aware fusion was requested but the "
                    "AEGIS evidence integration block is unavailable."
                )

            representation_parameters += list(
                alignment_model
                .evidence_integration
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

            fusion_architecture=(
                args.fusion_architecture
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

            fusion_architecture=(
                args.fusion_architecture
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
        "Quality weight:",
        (
            args.quality_weight
            if args.fusion_architecture in {
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else 0.0
        ),
    )

    print(
        "Compatibility weight:",
        (
            args.compatibility_weight
            if args.fusion_architecture in {
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else 0.0
        ),
    )

    print(
        "Transition objective weight:",
        (
            args.transition_objective_weight
            if args.fusion_architecture in {
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
            }
            else 0.0
        ),
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

        "fusion_architecture": (
            args.fusion_architecture
            if args.mode == "multimodal"
            else "bypassed"
        ),

        "v033_utility_supervised_intervention": (
            {
                "architecture_label": "M4qusli",
                "protocol_version": "0.33.0-step2",
                "implementation_version": "0.33.0-step5",
                "utility_target": "clip(0.5 + delta_u / 0.20, 0, 1)",
                "effective_utility_loss_weight": V033_UTILITY_LOSS_WEIGHT,
                "active_threshold": 0.60,
                "max_intervention_strength": 0.15,
                "transition_objective_weight": 0.0,
                "official_test_accessed": False,
            }
            if is_v033_utility_supervised_architecture(
                args.fusion_architecture
            )
            else None
        ),

        "evidence_aware": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "evidence_aware"
        ),

        "interaction_only": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "interaction_only"
        ),

        "gated_interaction": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "gated_interaction"
        ),

        "reliability_only": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "reliability_only"
        ),

        "interaction_reliability": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "interaction_reliability"
        ),

        "quality_supervised": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "quality_supervised"
        ),

        "quality_compatibility_supervised": (
            args.mode == "multimodal"
            and args.fusion_architecture
            == "quality_compatibility_supervised"
        ),

        "quality_compatibility_fusion": (
            args.mode == "multimodal"
            and args.fusion_architecture == "quality_compatibility_fusion"
        ),

        "quality_compatibility_selective_intervention": (
            v030_selective_intervention_mode(args.fusion_architecture)
            if args.mode == "multimodal"
            else None
        ),

        "v030_selective_intervention_protocol": (
            {
                "protocol_version": "0.30.0-step2a1",
                "architecture_label": {
                    "quality_compatibility_selective_intervention_weights": "M4qesri-w",
                    "quality_compatibility_selective_intervention_transition": "M4qesri-t",
                    "quality_compatibility_selective_intervention": "M4qesri",
                }[args.fusion_architecture],
                "mode": v030_selective_intervention_mode(
                    args.fusion_architecture
                ),
                "stable_reference": "M4qcs-w",
                "training_mixture": "M4qcs-compatible corruption/mismatch mixture",
                "v029_transition_objective_active": False,
                "controller_parameter_free": True,
                "controller_inputs_stop_gradient": True,
                "formal_hypotheses_computed": False,
                "official_test_accessed": False,
            }
            if is_v030_selective_intervention_architecture(
                args.fusion_architecture
            )
            else None
        ),

        "quality_supervision_enabled": (
            args.mode == "multimodal"
            and args.fusion_architecture in {
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
        ),

        "quality_weight": (
            args.quality_weight
            if args.fusion_architecture in {
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else 0.0
        ),

        "compatibility_weight": (
            args.compatibility_weight
            if args.fusion_architecture in {
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else 0.0
        ),

        "transition_objective": {
            "enabled": args.fusion_architecture in {
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
            },
            "requested_weight": args.transition_objective_weight,
            "effective_weight": (
                args.transition_objective_weight
                if args.fusion_architecture in {
                    "quality_compatibility_transition_control",
                    "quality_compatibility_graded_transition_fusion",
                }
                else 0.0
            ),
            "training_split_only": True,
            "matched_reference_detached": True,
            "beneficial_transition_penalty": 0.0,
            "definition": (
                "mean_relu(matched_true_class_probability_detached_minus_"
                "mismatched_true_class_probability)_over_mismatch_samples"
            ),
            "validation_feedback_used": False,
        },

        "m4q_corruption_protocol": (
            {
                "sampling_level": "per_sample",
                "clean_probability": 0.40,
                "text_quality_probability": 0.20,
                "vision_quality_probability": 0.20,
                "mismatch_probability": 0.20,
                "quality_family_probabilities": {
                    "gaussian_noise": 0.40,
                    "attenuation": 0.40,
                    "zero_dropout": 0.20,
                },
                "continuous_severities": list(M4Q_SEVERITIES),
                "gaussian_scale_source": "training_cache_per_feature_population_std",
                "quality_target_rule": "1_minus_severity",
                "permutation_quality_targets": [1.0, 1.0],
            }
            if args.fusion_architecture in {
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_graded_weights",
                "quality_compatibility_transition_control",
                "quality_compatibility_graded_transition_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else None
        ),

        "m4qc_compatibility_protocol": (
            {
                "protocol_version": "0.27.0-step11a",
                "diagnostic_only": True,
                "matched_target": 1.0,
                "mismatch_target": 0.0,
                "quality_corruption_compatibility_target": 1.0,
                "training_mismatch": "deterministic_cyclic_vision_derangement",
                "lambda_c": args.compatibility_weight,
                "affects_primary_fusion": False,
                "official_test_accessed": False,
            }
            if args.fusion_architecture == "quality_compatibility_supervised"
            else None
        ),

        "m4qcf_reliability_fusion_protocol": (
            {
                "protocol_version": "0.27.0-step13a",
                "epsilon": 0.10,
                "gamma": 1.0,
                "controller_parameter_free": True,
                "stop_gradient_reliability_signals": True,
                "affects_primary_fusion": True,
                "training_mixture": "same_as_m4qc_step11a",
                "training_mismatch": "deterministic_cyclic_vision_derangement",
                "lambda_q": args.quality_weight,
                "lambda_c": args.compatibility_weight,
                "formal_hypotheses_computed": False,
                "official_test_accessed": False,
            }
            if args.fusion_architecture == "quality_compatibility_fusion"
            else None
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

        "initialization_protocol": (
            "paired_common_components_v1"
        ),

        "m1b_gate_initialization_control": (
            "copy_legacy_gate_state_v1"
            if args.fusion_architecture
            in {
                "gated_interaction",
                "quality_supervised",
                "quality_compatibility_supervised",
                "quality_compatibility_fusion",
                "quality_compatibility_selective_weights",
                "quality_compatibility_selective_interaction",
                "quality_compatibility_selective_fusion",
                "quality_compatibility_selective_intervention_weights",
                "quality_compatibility_selective_intervention_transition",
                "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
            }
            else None
        ),

        "m2_gate_initialization_control": (
            "shared_parent_legacy_gate_v1"
            if args.fusion_architecture
            == "reliability_only"
            else None
        ),

        "m2b_gate_initialization_control": (
            "shared_parent_legacy_gate_v1"
            if args.fusion_architecture
            == "interaction_reliability"
            else None
        ),

        "classifier_initialization_seed": (
            classifier_initialization_seed
        ),

        "training_rng_reset_seed": (
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

        "fusion_architecture": (
            args.fusion_architecture
            if args.mode == "multimodal"
            else "bypassed"
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

        "initial_state_fingerprints": (
            initial_state_fingerprints
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

                    quality_loss_weight=(
                        args.quality_weight
                        if args.fusion_architecture in {
                            "quality_supervised",
                            "quality_compatibility_supervised",
                            "quality_compatibility_fusion",
                            "quality_compatibility_selective_weights",
                            "quality_compatibility_selective_interaction",
                            "quality_compatibility_selective_fusion",
                            "quality_compatibility_graded_weights",
                            "quality_compatibility_transition_control",
                            "quality_compatibility_graded_transition_fusion",
                            "quality_compatibility_selective_intervention_weights",
                            "quality_compatibility_selective_intervention_transition",
                            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
                        }
                        else 0.0
                    ),

                    compatibility_loss_weight=(
                        args.compatibility_weight
                        if args.fusion_architecture in {
                            "quality_compatibility_supervised",
                            "quality_compatibility_fusion",
                            "quality_compatibility_selective_weights",
                            "quality_compatibility_selective_interaction",
                            "quality_compatibility_selective_fusion",
                            "quality_compatibility_graded_weights",
                            "quality_compatibility_transition_control",
                            "quality_compatibility_graded_transition_fusion",
                            "quality_compatibility_selective_intervention_weights",
                            "quality_compatibility_selective_intervention_transition",
                            "quality_compatibility_selective_intervention",
            "quality_compatibility_calibrated_intervention_calibration",
            "quality_compatibility_calibrated_intervention_utility",
            "quality_compatibility_calibrated_intervention",
            "quality_compatibility_utility_supervised_intervention",
                        }
                        else 0.0
                    ),

                    transition_loss_weight=(
                        args.transition_objective_weight
                        if args.fusion_architecture in {
                            "quality_compatibility_transition_control",
                            "quality_compatibility_graded_transition_fusion",
                        }
                        else 0.0
                    ),

                    text_feature_std=(
                        text_feature_std
                    ),

                    vision_feature_std=(
                        vision_feature_std
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

                        "fusion_architecture": (
                            args.fusion_architecture
                            if args.mode == "multimodal"
                            else "bypassed"
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

                "fusion_architecture": (
                    args.fusion_architecture
                    if args.mode == "multimodal"
                    else "bypassed"
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

                    "quality_loss": training_metrics["quality_loss"],
                    "text_quality_loss": training_metrics[
                        "text_quality_loss"
                    ],
                    "vision_quality_loss": training_metrics[
                        "vision_quality_loss"
                    ],
                    "compatibility_loss": training_metrics[
                        "compatibility_loss"
                    ],
                    "utility_loss": training_metrics["utility_loss"],
                    "effective_utility_loss_weight": training_metrics[
                        "effective_utility_loss_weight"
                    ],
                    "selector_gradient_norm": training_metrics[
                        "selector_gradient_norm"
                    ],
                    "selector_parameter_l2_movement": training_metrics[
                        "selector_parameter_l2_movement"
                    ],
                    "utility_probability_mean": training_metrics[
                        "utility_probability_mean"
                    ],
                    "active_intervention_rate": training_metrics[
                        "active_intervention_rate"
                    ],
                    "delta_u_mean": training_metrics["delta_u_mean"],
                    "delta_u_std": training_metrics["delta_u_std"],
                    "u_target_mean": training_metrics["u_target_mean"],
                    "transition_loss": training_metrics[
                        "transition_loss"
                    ],
                    "corruption_counts": training_metrics[
                        "corruption_counts"
                    ],
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

    final_state_fingerprints = (
        collect_component_fingerprints(
            alignment_model=(
                alignment_model
            ),
            classification_model=(
                classification_model
            ),
            mode=(
                args.mode
            ),
            fusion_architecture=(
                args.fusion_architecture
            ),
        )
    )

    parameter_update_audit = (
        compare_fingerprints(
            initial=(
                initial_state_fingerprints
            ),
            final=(
                final_state_fingerprints
            ),
        )
    )

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

        "fusion_architecture": (
            args.fusion_architecture
            if args.mode == "multimodal"
            else "bypassed"
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

        "initial_state_fingerprints": (
            initial_state_fingerprints
        ),

        "final_state_fingerprints": (
            final_state_fingerprints
        ),

        "parameter_update_audit": (
            parameter_update_audit
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
        "Fusion architecture:",
        (
            args.fusion_architecture
            if args.mode == "multimodal"
            else "BYPASSED"
        ),
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

    print(
        "Parameter update audit:",
        parameter_update_audit,
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
