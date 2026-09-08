"""
AEGIS v0.27 - Step14A M4qcf Formal Robustness Diagnostics

Protocol
--------
0.27.0-step14a
Parent: 0.27.0-step13a

Purpose
-------
Generate deterministic validation-only robustness evidence for the frozen
M1b, M4qc, and M4qcf checkpoints.

This runner:
- performs NO training;
- performs NO checkpoint reselection;
- performs NO threshold tuning;
- accesses NO official Fakeddit test samples;
- reuses the frozen v0.27 Step9 representation-corruption machinery;
- reuses the frozen Step12B class-preserving mismatch construction;
- evaluates the frozen Stage-1 classifier;
- records M4qcf reliability-controller outputs;
- DOES NOT decide H8-F, H9-F, or H10-F.

Formal hypothesis decisions are deferred to Step14B.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sys

import torch
from torch import nn

# Allow direct execution from the repository root via:
# python scripts/run_fakeddit_m4qcf_robustness_diagnostics_v027.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from aegis.reliability.corruption import compute_feature_std

from scripts.run_fakeddit_quality_diagnostics_v027 import (
    assert_safe_input_path,
    build_conditions,
    condition_slug,
    get_sample_ids,
    load_representation_cache,
    module_state_fingerprint,
    precompute_validation_conditions,
    tensor_sha256,
)

from scripts.run_fakeddit_m4qc_compatibility_diagnostics_v027 import (
    build_class_preserving_derangement,
    validate_class_preserving_derangement,
)



VERSION = "0.27.0-step14a"
PARENT_PROTOCOL_VERSION = "0.27.0-step13a"
EXPERIMENT_NAME = "fakeddit_m4qcf_robustness_diagnostics_v027"

DEFAULT_SEEDS = (42, 43, 44)

DEFAULT_TRAIN_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/train_n5000_seed42"
)
DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
)
DEFAULT_EXPERIMENTS_ROOT = Path("experiments/fakeddit")
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v027_m4qcf_robustness_diagnostics"
)

EXPECTED_TRAIN_SAMPLES = 5000
EXPECTED_VALIDATION_SAMPLES = 1000

ARCHITECTURES = ("M1b", "M4qc", "M4qcf")

ARCHITECTURE_SPECS = {
    "M1b": {
        "fusion_architecture": "gated_interaction",
        "directory_pattern": "v025_m1b_gated_interaction_seed{seed}",
    },
    "M4qc": {
        "fusion_architecture": "quality_compatibility_supervised",
        "directory_pattern": "v027_m4qc_seed{seed}",
    },
    "M4qcf": {
        "fusion_architecture": "quality_compatibility_fusion",
        "directory_pattern": "v027_m4qcf_seed{seed}",
    },
}

M4QC_PROTOCOL_VERSION = "0.27.0-step11a"
M4QCF_PROTOCOL_VERSION = "0.27.0-step13a"

WEIGHT_SUM_TOLERANCE = 1e-5


def resolve_runtime_device(requested: Optional[str]) -> torch.device:
    """Resolve the Step14A inference device without changing experiment semantics."""
    if requested is None or str(requested).lower() == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    normalized = str(requested).lower()
    device = torch.device(normalized)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested for Step14A, but PyTorch "
            "reports that CUDA is unavailable."
        )

    return device


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.27 Step14A deterministic M4qcf robustness diagnostics."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
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
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expected-train-samples",
        type=int,
        default=EXPECTED_TRAIN_SAMPLES,
    )
    parser.add_argument(
        "--expected-validation-samples",
        type=int,
        default=EXPECTED_VALIDATION_SAMPLES,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Engineering smoke only: seed42, all three architectures, "
            "clean + text Gaussian severity .25, plus mismatch."
        ),
    )

    args = parser.parse_args()

    if not args.seeds:
        raise ValueError("--seeds requires at least one seed.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.expected_train_samples <= 0:
        raise ValueError("--expected-train-samples must be greater than zero.")
    if args.expected_validation_samples <= 0:
        raise ValueError("--expected-validation-samples must be greater than zero.")

    if args.smoke:
        args.seeds = [42]

    return args


# ============================================================================
# Generic helpers
# ============================================================================


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def save_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
            count += 1

    if count == 0:
        raise RuntimeError(f"Refusing to write empty JSONL artifact: {path}")


def sha256_jsonable(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is non-finite: {result}")
    return result


def optional_scalar(
    outputs: Mapping[str, Any],
    name: str,
    expected_rows: int,
) -> Optional[torch.Tensor]:
    value = outputs.get(name)

    if value is None:
        return None

    tensor = value.detach().cpu().reshape(-1).float()

    if tensor.numel() != expected_rows:
        raise RuntimeError(
            f"{name} produced {tensor.numel()} values; "
            f"expected {expected_rows}."
        )

    if not bool(torch.isfinite(tensor).all()):
        raise RuntimeError(f"{name} contains non-finite values.")

    return tensor


def bounded_unit_interval(
    tensor: Optional[torch.Tensor],
    name: str,
) -> None:
    if tensor is None:
        return
    if bool(torch.any(tensor < 0.0)) or bool(torch.any(tensor > 1.0)):
        raise RuntimeError(f"{name} contains values outside [0,1].")


def population_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("population_stats requires at least one value.")
    numeric = [finite_float(v, "statistic input") for v in values]
    return {
        "mean": float(mean(numeric)),
        "std_population": float(pstdev(numeric)),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
    }


def binary_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
) -> Dict[str, float]:
    if len(targets) != len(predictions):
        raise RuntimeError("Target/prediction lengths differ.")
    if not targets:
        raise RuntimeError("Cannot calculate metrics on zero samples.")

    t = [int(v) for v in targets]
    p = [int(v) for v in predictions]

    if any(v not in (0, 1) for v in t):
        raise RuntimeError("Stage-1 targets must be binary.")
    if any(v not in (0, 1) for v in p):
        raise RuntimeError("Stage-1 predictions must be binary.")

    def class_stats(cls: int) -> Tuple[float, float, float]:
        tp = sum(1 for y, yh in zip(t, p) if y == cls and yh == cls)
        fp = sum(1 for y, yh in zip(t, p) if y != cls and yh == cls)
        fn = sum(1 for y, yh in zip(t, p) if y == cls and yh != cls)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return precision, recall, f1

    p0, r0, f0 = class_stats(0)
    p1, r1, f1 = class_stats(1)

    accuracy = sum(int(y == yh) for y, yh in zip(t, p)) / len(t)

    return {
        "accuracy": float(accuracy),
        "precision": float(p1),
        "recall": float(r1),
        "f1": float(f1),
        "macro_f1": float(0.5 * (f0 + f1)),
        "class0_precision": float(p0),
        "class0_recall": float(r0),
        "class0_f1": float(f0),
        "class1_precision": float(p1),
        "class1_recall": float(r1),
        "class1_f1": float(f1),
    }


# ============================================================================
# Frozen checkpoint reconstruction
# ============================================================================


def checkpoint_path(
    experiments_root: Path,
    architecture: str,
    seed: int,
) -> Path:
    if architecture not in ARCHITECTURE_SPECS:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    directory = ARCHITECTURE_SPECS[architecture][
        "directory_pattern"
    ].format(seed=int(seed))

    return experiments_root / directory / "best_model.pt"


def load_checkpoint(
    path: Path,
    *,
    architecture: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    assert_safe_input_path(path, f"{architecture} checkpoint")

    require(
        path.name == "best_model.pt",
        "Step14A is restricted to frozen best_model.pt checkpoints.",
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen {architecture} checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    required = {
        "epoch",
        "ablation_mode",
        "fusion_architecture",
        "alignment_model_state_dict",
        "classification_model_state_dict",
        "configuration",
        "validation_metrics",
    }

    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            f"{architecture} checkpoint is missing keys: {sorted(missing)}"
        )

    require(
        checkpoint["ablation_mode"] == "multimodal",
        f"{architecture}: checkpoint is not multimodal.",
    )

    expected_fusion = ARCHITECTURE_SPECS[architecture][
        "fusion_architecture"
    ]

    require(
        checkpoint["fusion_architecture"] == expected_fusion,
        f"{architecture}: expected fusion_architecture={expected_fusion!r}, "
        f"found {checkpoint['fusion_architecture']!r}.",
    )

    config = checkpoint["configuration"]
    require(
        isinstance(config, Mapping),
        f"{architecture}: malformed checkpoint configuration.",
    )

    require(
        int(config.get("seed")) == int(seed),
        f"{architecture}: checkpoint seed mismatch.",
    )

    config_fusion = config.get("fusion_architecture")
    require(
        config_fusion in (None, expected_fusion),
        f"{architecture}: configuration fusion architecture mismatch.",
    )

    if architecture == "M1b":
        require(
            bool(config.get("gated_interaction")) is True,
            "M1b checkpoint does not confirm gated_interaction=True.",
        )

    elif architecture == "M4qc":
        require(
            bool(config.get("quality_compatibility_supervised")) is True,
            "M4qc checkpoint flag is not enabled.",
        )
        protocol = config.get("m4qc_compatibility_protocol")
        require(
            isinstance(protocol, Mapping),
            "M4qc checkpoint is missing compatibility protocol metadata.",
        )
        require(
            protocol.get("protocol_version") == M4QC_PROTOCOL_VERSION,
            "M4qc protocol version mismatch.",
        )
        require(
            protocol.get("diagnostic_only") is True,
            "M4qc checkpoint does not preserve diagnostic-only compatibility.",
        )
        require(
            protocol.get("affects_primary_fusion") is False,
            "M4qc compatibility unexpectedly affects primary fusion.",
        )
        require(
            protocol.get("official_test_accessed") is False,
            "M4qc metadata reports official test access.",
        )

    elif architecture == "M4qcf":
        require(
            bool(config.get("quality_compatibility_fusion")) is True,
            "M4qcf checkpoint flag is not enabled.",
        )
        protocol = config.get("m4qcf_reliability_fusion_protocol")
        require(
            isinstance(protocol, Mapping),
            "M4qcf checkpoint is missing Step13A protocol metadata.",
        )
        require(
            protocol.get("protocol_version") == M4QCF_PROTOCOL_VERSION,
            "M4qcf protocol version mismatch.",
        )
        require(
            protocol.get("affects_primary_fusion") is True,
            "M4qcf checkpoint does not confirm reliability-informed fusion.",
        )
        require(
            protocol.get("stop_gradient_reliability_signals") is True,
            "M4qcf checkpoint violates stop-gradient protocol.",
        )
        require(
            protocol.get("official_test_accessed") is False,
            "M4qcf metadata reports official test access.",
        )

    return dict(checkpoint)


def build_models(
    checkpoint: Mapping[str, Any],
    *,
    architecture: str,
    text_dim: int,
    vision_dim: int,
    device: torch.device,
) -> Tuple[
    CrossModalAlignmentModel,
    HierarchicalInformationIntegrityClassifier,
    Dict[str, Any],
]:
    config = checkpoint["configuration"]

    shared_dim = int(config["shared_dim"])
    hidden_dim = int(config["hidden_dim"])
    dropout = float(config["dropout"])
    temperature = float(config["temperature"])

    kwargs: Dict[str, Any] = {
        "text_dim": int(text_dim),
        "vision_dim": int(vision_dim),
        "shared_dim": shared_dim,
        "dropout": dropout,
        "temperature": temperature,
    }

    if architecture == "M1b":
        kwargs["gated_interaction"] = True
    elif architecture == "M4qc":
        kwargs["quality_compatibility_supervised"] = True
    elif architecture == "M4qcf":
        kwargs["quality_compatibility_fusion"] = True
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    alignment_model = CrossModalAlignmentModel(**kwargs)
    alignment_model.load_state_dict(
        checkpoint["alignment_model_state_dict"],
        strict=True,
    )

    classification_model = HierarchicalInformationIntegrityClassifier(
        input_dim=shared_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    classification_model.load_state_dict(
        checkpoint["classification_model_state_dict"],
        strict=True,
    )

    alignment_model.to(device).eval()
    classification_model.to(device).eval()

    if architecture == "M4qc":
        require(
            alignment_model.text_quality_estimator is not None,
            "M4qc text quality estimator unavailable.",
        )
        require(
            alignment_model.vision_quality_estimator is not None,
            "M4qc vision quality estimator unavailable.",
        )
        require(
            alignment_model.compatibility_estimator is not None,
            "M4qc compatibility estimator unavailable.",
        )

    if architecture == "M4qcf":
        require(
            alignment_model.text_quality_estimator is not None,
            "M4qcf text quality estimator unavailable.",
        )
        require(
            alignment_model.vision_quality_estimator is not None,
            "M4qcf vision quality estimator unavailable.",
        )
        require(
            alignment_model.compatibility_estimator is not None,
            "M4qcf compatibility estimator unavailable.",
        )

    resolved = {
        "shared_dim": shared_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "temperature": temperature,
        "fusion_architecture": ARCHITECTURE_SPECS[architecture][
            "fusion_architecture"
        ],
        "constructor_kwargs": kwargs,
    }

    return alignment_model, classification_model, resolved


# ============================================================================
# Forward/evaluation
# ============================================================================


def evaluate_embeddings(
    *,
    architecture: str,
    seed: int,
    condition_name: str,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    targets: torch.Tensor,
    sample_ids: Sequence[str],
    alignment_model: CrossModalAlignmentModel,
    classification_model: HierarchicalInformationIntegrityClassifier,
    batch_size: int,
    device: torch.device,
    donor_indices: Optional[torch.Tensor] = None,
    mismatch_state: Optional[str] = None,
    corruption_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n = int(text_embeddings.shape[0])

    require(
        int(vision_embeddings.shape[0]) == n,
        "Text/vision evaluation sample counts differ.",
    )
    require(
        int(targets.reshape(-1).numel()) == n,
        "Target count does not match evaluation sample count.",
    )
    require(
        len(sample_ids) == n,
        "Sample identity count does not match evaluation sample count.",
    )

    target_cpu = targets.detach().cpu().reshape(-1).long()

    if donor_indices is not None:
        donor_indices = donor_indices.detach().cpu().reshape(-1).long()
        require(
            donor_indices.numel() == n,
            "Mismatch donor mapping has incorrect cardinality.",
        )

    rows: List[Dict[str, Any]] = []
    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_classification_losses: List[float] = []

    reliability_accumulator: Dict[str, List[float]] = {
        "text_quality": [],
        "vision_quality": [],
        "compatibility_score": [],
        "effective_text_reliability": [],
        "effective_vision_reliability": [],
        "text_weight": [],
        "vision_weight": [],
        "interaction_multiplier": [],
    }

    alignment_model.eval()
    classification_model.eval()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            current_n = end - start

            text_batch = text_embeddings[start:end].to(device)
            vision_batch = vision_embeddings[start:end].to(device)
            target_batch = target_cpu[start:end].to(device)

            outputs = alignment_model(
                text_batch,
                vision_batch,
                compute_loss=False,
            )

            fused = outputs.get("fused_embedding")
            if fused is None:
                raise RuntimeError(
                    f"{architecture}: fused_embedding unavailable."
                )
            if not bool(torch.isfinite(fused).all()):
                raise RuntimeError(
                    f"{architecture}: non-finite fused representation."
                )

            classification = classification_model(fused)
            logits = classification.get("integrity_logits")

            if logits is None:
                raise RuntimeError("Classifier integrity_logits unavailable.")
            if logits.ndim != 2 or logits.shape[0] != current_n:
                raise RuntimeError(
                    f"Unexpected classifier logit shape: {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Classifier logits contain non-finite values.")

            probabilities = torch.softmax(logits, dim=-1)
            if not bool(torch.isfinite(probabilities).all()):
                raise RuntimeError("Classifier probabilities are non-finite.")

            probability_sums = probabilities.sum(dim=-1)
            if not torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise RuntimeError("Classifier probabilities do not sum to one.")

            predictions = torch.argmax(probabilities, dim=-1)

            losses = nn.functional.cross_entropy(
                logits,
                target_batch,
                reduction="none",
            )

            diagnostic_tensors: Dict[str, Optional[torch.Tensor]] = {}

            for name in reliability_accumulator:
                diagnostic_tensors[name] = optional_scalar(
                    outputs,
                    name,
                    current_n,
                )

            for name in (
                "text_quality",
                "vision_quality",
                "compatibility_score",
                "text_weight",
                "vision_weight",
                "interaction_multiplier",
            ):
                bounded_unit_interval(diagnostic_tensors[name], name)

            if architecture == "M1b":
                for name, tensor in diagnostic_tensors.items():
                    if tensor is not None:
                        raise RuntimeError(
                            f"M1b unexpectedly emitted {name}."
                        )

            if architecture == "M4qc":
                for name in (
                    "text_quality",
                    "vision_quality",
                    "compatibility_score",
                ):
                    if diagnostic_tensors[name] is None:
                        raise RuntimeError(
                            f"M4qc required diagnostic {name} unavailable."
                        )

                for name in (
                    "effective_text_reliability",
                    "effective_vision_reliability",
                    "text_weight",
                    "vision_weight",
                    "interaction_multiplier",
                ):
                    if diagnostic_tensors[name] is not None:
                        raise RuntimeError(
                            f"M4qc unexpectedly emitted fusion diagnostic {name}."
                        )

            if architecture == "M4qcf":
                for name in reliability_accumulator:
                    if diagnostic_tensors[name] is None:
                        raise RuntimeError(
                            f"M4qcf required diagnostic {name} unavailable."
                        )

                weight_sum = (
                    diagnostic_tensors["text_weight"]
                    + diagnostic_tensors["vision_weight"]
                )
                if not torch.allclose(
                    weight_sum,
                    torch.ones_like(weight_sum),
                    atol=WEIGHT_SUM_TOLERANCE,
                    rtol=0.0,
                ):
                    max_error = float(
                        torch.max(torch.abs(weight_sum - 1.0)).item()
                    )
                    raise RuntimeError(
                        "M4qcf evidence weights do not sum to one; "
                        f"max error={max_error}"
                    )

                if bool(torch.any(diagnostic_tensors["text_weight"] <= 0.0)):
                    raise RuntimeError("M4qcf text_weight must be > 0.")
                if bool(torch.any(diagnostic_tensors["text_weight"] >= 1.0)):
                    raise RuntimeError("M4qcf text_weight must be < 1.")
                if bool(torch.any(diagnostic_tensors["vision_weight"] <= 0.0)):
                    raise RuntimeError("M4qcf vision_weight must be > 0.")
                if bool(torch.any(diagnostic_tensors["vision_weight"] >= 1.0)):
                    raise RuntimeError("M4qcf vision_weight must be < 1.")

            pred_cpu = predictions.detach().cpu().reshape(-1)
            prob_cpu = probabilities.detach().cpu()
            loss_cpu = losses.detach().cpu().reshape(-1)

            for local_index in range(current_n):
                global_index = start + local_index
                target = int(target_cpu[global_index].item())
                prediction = int(pred_cpu[local_index].item())
                cls_loss = finite_float(
                    loss_cpu[local_index].item(),
                    "classification loss",
                )

                all_targets.append(target)
                all_predictions.append(prediction)
                all_classification_losses.append(cls_loss)

                row: Dict[str, Any] = {
                    "protocol_version": VERSION,
                    "architecture": architecture,
                    "model_seed": int(seed),
                    "condition": condition_name,
                    "row_index": global_index,
                    "sample_id": str(sample_ids[global_index]),
                    "target": target,
                    "prediction": prediction,
                    "correct": bool(target == prediction),
                    "probability_class0": finite_float(
                        prob_cpu[local_index, 0].item(),
                        "class0 probability",
                    ),
                    "probability_class1": finite_float(
                        prob_cpu[local_index, 1].item(),
                        "class1 probability",
                    ),
                    "classification_loss": cls_loss,
                    "mismatch_state": mismatch_state,
                    "receiver_sample_id": (
                        str(sample_ids[global_index])
                        if mismatch_state is not None
                        else None
                    ),
                    "vision_donor_index": None,
                    "vision_donor_sample_id": None,
                    "receiver_class": (
                        target if mismatch_state is not None else None
                    ),
                    "donor_class": None,
                    "official_test_accessed": False,
                }

                if corruption_metadata is not None:
                    row.update(
                        {
                            "corruption_type": corruption_metadata.get(
                                "corruption_type"
                            ),
                            "corrupted_modality": corruption_metadata.get(
                                "corrupted_modality"
                            ),
                            "severity": corruption_metadata.get("severity"),
                            "corruption_seed": corruption_metadata.get(
                                "corruption_seed"
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "corruption_type": None,
                            "corrupted_modality": None,
                            "severity": None,
                            "corruption_seed": None,
                        }
                    )

                if donor_indices is not None:
                    donor = int(donor_indices[global_index].item())
                    row["vision_donor_index"] = donor
                    row["vision_donor_sample_id"] = str(sample_ids[donor])
                    row["donor_class"] = int(target_cpu[donor].item())

                for name, tensor in diagnostic_tensors.items():
                    value = (
                        None
                        if tensor is None
                        else finite_float(
                            tensor[local_index].item(),
                            name,
                        )
                    )
                    row[name] = value
                    if value is not None:
                        reliability_accumulator[name].append(value)

                rows.append(row)

    metrics = binary_metrics(all_targets, all_predictions)
    metrics["classification_loss"] = float(
        mean(all_classification_losses)
    )
    metrics["sample_count"] = n

    diagnostic_summary: Dict[str, Any] = {}
    for name, values in reliability_accumulator.items():
        diagnostic_summary[name] = (
            None if not values else population_stats(values)
        )

    summary = {
        "protocol_version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "architecture": architecture,
        "model_seed": int(seed),
        "condition": condition_name,
        "mismatch_state": mismatch_state,
        "metrics": metrics,
        "diagnostics": diagnostic_summary,
        "official_test_accessed": False,
        "formal_hypothesis_decisions": {
            "H8_F": "NOT_COMPUTED",
            "H9_F": "NOT_COMPUTED",
            "H10_F": "NOT_COMPUTED",
        },
    }

    return rows, summary


# ============================================================================
# Mismatch construction
# ============================================================================


def build_frozen_mismatch_mapping(
    targets: torch.Tensor,
) -> torch.Tensor:
    labels = targets.detach().cpu().reshape(-1).long()

    mapping, class_metadata = build_class_preserving_derangement(labels)

    require(
        isinstance(mapping, torch.Tensor),
        "Frozen Step12B derangement did not return donor indices as a tensor.",
    )
    require(
        isinstance(class_metadata, Mapping),
        "Frozen Step12B derangement did not return class metadata.",
    )

    mapping = mapping.detach().cpu().reshape(-1).long()

    # Reuse the exact frozen Step12B validator contract.
    validate_class_preserving_derangement(
        targets=labels,
        donor_indices=mapping,
    )

    n = int(labels.numel())

    require(
        mapping.numel() == n,
        "Mismatch mapping cardinality violation.",
    )
    require(
        bool(torch.all(mapping >= 0)) and bool(torch.all(mapping < n)),
        "Mismatch mapping contains invalid donor indices.",
    )

    indices = torch.arange(n, dtype=torch.long)
    require(
        not bool(torch.any(mapping == indices)),
        "Mismatch mapping contains self-pairs.",
    )

    require(
        bool(torch.equal(labels, labels.index_select(0, mapping))),
        "Mismatch mapping changes Stage-1 class.",
    )

    for cls in sorted(set(labels.tolist())):
        receivers = torch.nonzero(
            labels == int(cls),
            as_tuple=False,
        ).reshape(-1)

        donors = mapping.index_select(0, receivers)

        require(
            sorted(donors.tolist()) == sorted(receivers.tolist()),
            f"Class {cls}: donor mapping is not a within-class permutation.",
        )

    return mapping


def mapping_records(
    mapping: torch.Tensor,
    *,
    sample_ids: Sequence[str],
    targets: torch.Tensor,
) -> List[Dict[str, Any]]:
    labels = targets.detach().cpu().reshape(-1).long()
    records: List[Dict[str, Any]] = []

    for receiver in range(mapping.numel()):
        donor = int(mapping[receiver].item())
        records.append(
            {
                "receiver_index": receiver,
                "receiver_sample_id": str(sample_ids[receiver]),
                "receiver_class": int(labels[receiver].item()),
                "vision_donor_index": donor,
                "vision_donor_sample_id": str(sample_ids[donor]),
                "donor_class": int(labels[donor].item()),
                "self_pair": bool(receiver == donor),
                "class_preserved": bool(
                    int(labels[receiver].item())
                    == int(labels[donor].item())
                ),
            }
        )

    return records


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device(args.device)

    assert_safe_input_path(args.train_cache, "training cache")
    assert_safe_input_path(args.validation_cache, "validation cache")
    assert_safe_input_path(args.experiments_root, "experiments root")

    _, train_batch = load_representation_cache(args.train_cache)
    _, validation_batch = load_representation_cache(args.validation_cache)

    train_n = int(train_batch.text_embeddings.shape[0])
    validation_n = int(validation_batch.text_embeddings.shape[0])

    require(
        train_n == args.expected_train_samples,
        f"Training cache count mismatch: "
        f"{train_n} != {args.expected_train_samples}",
    )
    require(
        validation_n == args.expected_validation_samples,
        f"Validation cache count mismatch: "
        f"{validation_n} != {args.expected_validation_samples}",
    )

    require(
        int(train_batch.vision_embeddings.shape[0]) == train_n,
        "Training text/vision sample counts differ.",
    )
    require(
        int(validation_batch.vision_embeddings.shape[0]) == validation_n,
        "Validation text/vision sample counts differ.",
    )

    text_dim = int(validation_batch.text_embeddings.shape[1])
    vision_dim = int(validation_batch.vision_embeddings.shape[1])

    require(
        int(train_batch.text_embeddings.shape[1]) == text_dim,
        "Training/validation text dimensions differ.",
    )
    require(
        int(train_batch.vision_embeddings.shape[1]) == vision_dim,
        "Training/validation vision dimensions differ.",
    )

    targets = validation_batch.integrity_targets.detach().cpu().reshape(-1).long()

    require(
        targets.numel() == validation_n,
        "Validation target count mismatch.",
    )
    require(
        set(targets.tolist()).issubset({0, 1}),
        "Validation Stage-1 targets are not binary.",
    )

    sample_ids = get_sample_ids(validation_batch, validation_n)

    require(
        len(set(sample_ids)) == validation_n,
        "Validation sample identities contain duplicates.",
    )

    text_feature_std = compute_feature_std(
        train_batch.text_embeddings,
        unbiased=False,
    )
    vision_feature_std = compute_feature_std(
        train_batch.vision_embeddings,
        unbiased=False,
    )

    conditions = build_conditions(smoke=args.smoke)

    realized_conditions = precompute_validation_conditions(
        text_embeddings=validation_batch.text_embeddings,
        vision_embeddings=validation_batch.vision_embeddings,
        text_feature_std=text_feature_std,
        vision_feature_std=vision_feature_std,
        conditions=conditions,
    )

    expected_condition_count = 2 if args.smoke else 19
    require(
        len(realized_conditions) == expected_condition_count,
        f"Unexpected condition count: {len(realized_conditions)}",
    )

    mismatch_mapping = build_frozen_mismatch_mapping(targets)

    mapping_rows = mapping_records(
        mismatch_mapping,
        sample_ids=sample_ids,
        targets=targets,
    )

    mapping_hash = sha256_jsonable(mapping_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)

    save_json(
        args.output_root / "validation_derangement_mapping.json",
        {
            "protocol_version": VERSION,
            "construction": (
                "frozen_step12b_deterministic_class_preserving_derangement"
            ),
            "sample_count": validation_n,
            "mapping_sha256": mapping_hash,
            "rows": mapping_rows,
            "official_test_accessed": False,
        },
    )

    quality_condition_records = []
    for slug, record in realized_conditions.items():
        quality_condition_records.append(
            {
                "slug": slug,
                **dict(record["condition"]),
                "text_sha256": record["text_sha256"],
                "vision_sha256": record["vision_sha256"],
            }
        )

    save_json(
        args.output_root / "quality_conditions.json",
        {
            "protocol_version": VERSION,
            "conditions": quality_condition_records,
            "conditions_per_architecture_seed": len(realized_conditions),
            "corruption_realizations_shared_across_architectures": True,
            "corruption_realizations_shared_across_model_seeds": True,
            "official_test_accessed": False,
        },
    )

    manifest: Dict[str, Any] = {
        "protocol_version": VERSION,
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "scientific_status": (
            "validation-only diagnostic evidence; "
            "formal H8-F/H9-F/H10-F decisions deferred to Step14B"
        ),
        "smoke": bool(args.smoke),
        "formal_seeds": [int(seed) for seed in args.seeds],
        "architectures": list(ARCHITECTURES),
        "training_cache": str(args.train_cache),
        "validation_cache": str(args.validation_cache),
        "training_samples": train_n,
        "validation_samples": validation_n,
        "text_dimension": text_dim,
        "vision_dimension": vision_dim,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "quality_conditions_per_architecture_seed": len(realized_conditions),
        "quality_condition_level_evaluations": (
            len(realized_conditions)
            * len(ARCHITECTURES)
            * len(args.seeds)
        ),
        "formal_quality_condition_level_evaluations": (
            171 if not args.smoke else None
        ),
        "text_feature_std_sha256": tensor_sha256(text_feature_std),
        "vision_feature_std_sha256": tensor_sha256(vision_feature_std),
        "mismatch_mapping_sha256": mapping_hash,
        "mismatch_mapping_shared_across_architectures": True,
        "mismatch_mapping_shared_across_model_seeds": True,
        "checkpoint_reselection_performed": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "official_test_accessed": False,
        "official_test_samples_accessed": 0,
        "hypotheses": {
            "H8_F": "NOT_COMPUTED",
            "H9_F": "NOT_COMPUTED",
            "H10_F": "NOT_COMPUTED",
        },
        "checkpoints": {},
    }

    all_quality_summaries: List[Dict[str, Any]] = []
    all_mismatch_summaries: List[Dict[str, Any]] = []

    print("=" * 78)
    print("AEGIS v0.27 STEP14A - M4qcf ROBUSTNESS DIAGNOSTICS")
    print("=" * 78)
    print("Device:", device)
    print("Seeds:", args.seeds)
    print("Architectures:", ARCHITECTURES)
    print("Validation samples:", validation_n)
    print("Quality conditions:", len(realized_conditions))
    print("Official test: SEALED / NOT ACCESSED")
    print("H8-F/H9-F/H10-F: NOT COMPUTED")
    print()

    for seed in args.seeds:
        manifest["checkpoints"][str(seed)] = {}

        for architecture in ARCHITECTURES:
            path = checkpoint_path(
                args.experiments_root,
                architecture,
                seed,
            )

            print(
                f"Loading {architecture} seed{seed}: {path}"
            )

            checkpoint = load_checkpoint(
                path,
                architecture=architecture,
                seed=seed,
                device=device,
            )

            alignment_model, classification_model, resolved = build_models(
                checkpoint,
                architecture=architecture,
                text_dim=text_dim,
                vision_dim=vision_dim,
                device=device,
            )

            alignment_before = module_state_fingerprint(alignment_model)
            classifier_before = module_state_fingerprint(classification_model)

            manifest["checkpoints"][str(seed)][architecture] = {
                "path": str(path),
                "epoch": int(checkpoint["epoch"]),
                "fusion_architecture": checkpoint["fusion_architecture"],
                "validation_metrics": checkpoint["validation_metrics"],
                "resolved_model_configuration": resolved,
                "alignment_fingerprint_before": alignment_before,
                "classifier_fingerprint_before": classifier_before,
            }

            architecture_dir = (
                args.output_root
                / f"seed{seed}"
                / architecture.lower()
            )

            # ------------------------------------------------------------
            # Quality-corruption track
            # ------------------------------------------------------------

            for slug, condition_record in realized_conditions.items():
                condition = condition_record["condition"]

                rows, summary = evaluate_embeddings(
                    architecture=architecture,
                    seed=seed,
                    condition_name=slug,
                    text_embeddings=condition_record["text_embeddings"],
                    vision_embeddings=condition_record["vision_embeddings"],
                    targets=targets,
                    sample_ids=sample_ids,
                    alignment_model=alignment_model,
                    classification_model=classification_model,
                    batch_size=args.batch_size,
                    device=device,
                    corruption_metadata=condition,
                )

                summary.update(
                    {
                        "checkpoint_path": str(path),
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "text_corruption_tensor_sha256": (
                            condition_record["text_sha256"]
                        ),
                        "vision_corruption_tensor_sha256": (
                            condition_record["vision_sha256"]
                        ),
                    }
                )

                save_jsonl(
                    architecture_dir
                    / "quality"
                    / f"{slug}_samples.jsonl",
                    rows,
                )
                save_json(
                    architecture_dir
                    / "quality"
                    / f"{slug}_summary.json",
                    summary,
                )

                all_quality_summaries.append(summary)

                print(
                    f"  quality {slug:32s} "
                    f"MacroF1={summary['metrics']['macro_f1']:.6f}"
                )

            # ------------------------------------------------------------
            # Frozen matched/mismatched track
            # ------------------------------------------------------------

            identity_mapping = torch.arange(
                validation_n,
                dtype=torch.long,
            )

            matched_rows, matched_summary = evaluate_embeddings(
                architecture=architecture,
                seed=seed,
                condition_name="matched",
                text_embeddings=validation_batch.text_embeddings,
                vision_embeddings=validation_batch.vision_embeddings,
                targets=targets,
                sample_ids=sample_ids,
                alignment_model=alignment_model,
                classification_model=classification_model,
                batch_size=args.batch_size,
                device=device,
                donor_indices=identity_mapping,
                mismatch_state="matched",
            )

            mismatched_vision = validation_batch.vision_embeddings.index_select(
                0,
                mismatch_mapping.to(
                    validation_batch.vision_embeddings.device
                ),
            )

            mismatched_rows, mismatched_summary = evaluate_embeddings(
                architecture=architecture,
                seed=seed,
                condition_name="class_preserving_mismatched",
                text_embeddings=validation_batch.text_embeddings,
                vision_embeddings=mismatched_vision,
                targets=targets,
                sample_ids=sample_ids,
                alignment_model=alignment_model,
                classification_model=classification_model,
                batch_size=args.batch_size,
                device=device,
                donor_indices=mismatch_mapping,
                mismatch_state="mismatched",
            )

            for summary in (matched_summary, mismatched_summary):
                summary.update(
                    {
                        "checkpoint_path": str(path),
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "mapping_sha256": mapping_hash,
                    }
                )
                all_mismatch_summaries.append(summary)

            save_jsonl(
                architecture_dir / "mismatch" / "matched_samples.jsonl",
                matched_rows,
            )
            save_jsonl(
                architecture_dir / "mismatch" / "mismatched_samples.jsonl",
                mismatched_rows,
            )
            save_json(
                architecture_dir / "mismatch" / "matched_summary.json",
                matched_summary,
            )
            save_json(
                architecture_dir / "mismatch" / "mismatched_summary.json",
                mismatched_summary,
            )

            alignment_after = module_state_fingerprint(alignment_model)
            classifier_after = module_state_fingerprint(classification_model)

            require(
                alignment_before == alignment_after,
                f"{architecture} seed{seed}: alignment model mutated "
                "during Step14A evaluation.",
            )
            require(
                classifier_before == classifier_after,
                f"{architecture} seed{seed}: classifier mutated "
                "during Step14A evaluation.",
            )

            manifest["checkpoints"][str(seed)][architecture].update(
                {
                    "alignment_fingerprint_after": alignment_after,
                    "classifier_fingerprint_after": classifier_after,
                    "model_state_unchanged": True,
                }
            )

            print(
                f"  mismatch matched={matched_summary['metrics']['macro_f1']:.6f} "
                f"mismatched="
                f"{mismatched_summary['metrics']['macro_f1']:.6f}"
            )
            print()

            del alignment_model
            del classification_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_quality_summaries = (
        len(args.seeds)
        * len(ARCHITECTURES)
        * len(realized_conditions)
    )
    expected_mismatch_summaries = (
        len(args.seeds)
        * len(ARCHITECTURES)
        * 2
    )

    require(
        len(all_quality_summaries) == expected_quality_summaries,
        "Incomplete quality-condition summary matrix.",
    )
    require(
        len(all_mismatch_summaries) == expected_mismatch_summaries,
        "Incomplete mismatch summary matrix.",
    )

    save_json(
        args.output_root / "quality_condition_summaries.json",
        {
            "protocol_version": VERSION,
            "rows": all_quality_summaries,
            "row_count": len(all_quality_summaries),
            "H8_F": "NOT_COMPUTED",
            "H9_F": "NOT_COMPUTED",
            "official_test_accessed": False,
        },
    )

    save_json(
        args.output_root / "mismatch_summaries.json",
        {
            "protocol_version": VERSION,
            "rows": all_mismatch_summaries,
            "row_count": len(all_mismatch_summaries),
            "H10_F": "NOT_COMPUTED",
            "official_test_accessed": False,
        },
    )

    summary = {
        "protocol_version": VERSION,
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": (
            "STEP14A_SMOKE_COMPLETE"
            if args.smoke
            else "STEP14A_DIAGNOSTIC_EVIDENCE_COMPLETE"
        ),
        "smoke": bool(args.smoke),
        "quality_summary_rows": len(all_quality_summaries),
        "mismatch_summary_rows": len(all_mismatch_summaries),
        "quality_conditions_per_architecture_seed": len(
            realized_conditions
        ),
        "architectures": list(ARCHITECTURES),
        "seeds": [int(seed) for seed in args.seeds],
        "validation_samples": validation_n,
        "mismatch_mapping_sha256": mapping_hash,
        "model_state_unchanged": True,
        "training_performed": False,
        "checkpoint_reselection_performed": False,
        "threshold_tuning_performed": False,
        "official_test_accessed": False,
        "official_test_samples_accessed": 0,
        "formal_hypothesis_decisions": {
            "H8_F": "NOT_COMPUTED",
            "H9_F": "NOT_COMPUTED",
            "H10_F": "NOT_COMPUTED",
        },
        "next_step": (
            "Step14B formal analysis after Step14A artifact audit/freeze"
        ),
    }

    save_json(args.output_root / "summary.json", summary)
    save_json(args.output_root / "manifest.json", manifest)

    print("=" * 78)
    print("STEP14A COMPLETE")
    print("=" * 78)
    print("Quality summary rows:", len(all_quality_summaries))
    print("Mismatch summary rows:", len(all_mismatch_summaries))
    print("Mapping SHA256:", mapping_hash)
    print("Model state unchanged: YES")
    print("Training performed: NO")
    print("Checkpoint reselection: NO")
    print("Official test accessed: NO")
    print("H8-F: NOT_COMPUTED")
    print("H9-F: NOT_COMPUTED")
    print("H10-F: NOT_COMPUTED")
    print("Output:", args.output_root)


if __name__ == "__main__":
    main()
