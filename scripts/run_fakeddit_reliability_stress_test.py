"""
AEGIS v0.26.0 — Fakeddit Reliability Stress Test

Purpose
-------
Evaluate whether learned modality reliability responds meaningfully when one
cached modality representation is deliberately degraded, without retraining,
changing checkpoint weights, or accessing the official Fakeddit test split.

Architectures
-------------
M1b:
    legacy gated fusion + explicit interaction residual
    (robustness control; no scalar reliability diagnostics)

M2b:
    legacy gated fusion + interaction-conditioned reliability residual

M3:
    interaction-conditioned reliability -> scalar adaptive fusion

Scientific constraints
----------------------
1. Fixed validation cache only (default: validation_n1000_seed42).
2. Best frozen checkpoints only.
3. No gradient updates.
4. No official test-cache argument exists.
5. Exactly 1000 validation samples are required by default.
6. Corruption is applied to cached encoder representations before AEGIS
   projection/alignment.
7. Text and vision are corrupted separately.
8. Gaussian noise and attenuation use severities 0, .25, .50, .75, 1.
9. Permutation and zero-dropout are discrete conditions.
10. Corruption RNG is deterministic and recorded.
11. Clean evaluation is performed before corrupted conditions.
12. Model state fingerprints are checked before/after evaluation.

Important interpretation
------------------------
This is a diagnostic robustness/reliability experiment on the same fixed
validation cache used during model development. It is not an independent
generalization estimate and must not be reported as official test performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from aegis.data.cache import FakedditRepresentationCache
from scripts.run_fakeddit_generalization import (
    load_cache_as_single_batch,
    resolve_device,
)


VERSION = "0.26.0"

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_ARCHITECTURES = ("m1b", "m2b", "m3")
DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/reliability_stress"
)

ARCHITECTURES: Dict[str, Dict[str, str]] = {
    "m1b": {
        "label": "M1b",
        "fusion_architecture": "gated_interaction",
        "experiment_template": (
            "experiments/fakeddit/"
            "v025_m1b_gated_interaction_seed{seed}"
        ),
    },
    "m2b": {
        "label": "M2b",
        "fusion_architecture": "interaction_reliability",
        "experiment_template": (
            "experiments/fakeddit/ablation/m2b_seed{seed}"
        ),
    },
    "m3": {
        "label": "M3",
        "fusion_architecture": "evidence_aware",
        "experiment_template": (
            "experiments/fakeddit/v025_m3_evidence_seed{seed}"
        ),
    },
}

CONTINUOUS_CORRUPTIONS = ("gaussian_noise", "attenuation")
DISCRETE_CORRUPTIONS = ("permutation", "zero_dropout")
MODALITIES = ("text", "vision")
DEFAULT_SEVERITIES = (0.0, 0.25, 0.50, 0.75, 1.0)

CORRUPTION_IDS = {
    "clean": 0,
    "gaussian_noise": 1,
    "attenuation": 2,
    "permutation": 3,
    "zero_dropout": 4,
}

MODALITY_IDS = {
    "none": 0,
    "text": 1,
    "vision": 2,
}


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.26 Fakeddit reliability stress test on the fixed "
            "validation representation cache."
        )
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=list(DEFAULT_ARCHITECTURES),
        help="Architectures to evaluate. Default: m1b m2b m3.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Model seeds. Default: 42 43 44.",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
        help="Frozen validation representation cache.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename inside each experiment directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Stress-test output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size. Default: 64.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--severities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SEVERITIES),
        help=(
            "Continuous corruption severities. "
            "Default: 0 0.25 0.5 0.75 1."
        ),
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=list(CONTINUOUS_CORRUPTIONS + DISCRETE_CORRUPTIONS),
        default=list(CONTINUOUS_CORRUPTIONS + DISCRETE_CORRUPTIONS),
        help="Corruption families to evaluate.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=list(MODALITIES),
        default=list(MODALITIES),
        help="Modalities to corrupt separately. Default: text vision.",
    )
    parser.add_argument(
        "--expected-validation-samples",
        type=int,
        default=1000,
        help="Hard expected validation sample count. Default: 1000.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run only M2b seed42, text Gaussian noise severity .25, "
            "plus the mandatory clean control."
        ),
    )

    args = parser.parse_args()

    if not args.seeds:
        parser.error("--seeds must contain at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.expected_validation_samples <= 0:
        parser.error("--expected-validation-samples must be positive.")
    if not args.severities:
        parser.error("--severities must not be empty.")
    if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in args.severities):
        parser.error("Every severity must be finite and in [0, 1].")
    if len(set(args.severities)) != len(args.severities):
        parser.error("--severities must not contain duplicates.")

    if args.smoke:
        args.architectures = ["m2b"]
        args.seeds = [42]
        args.corruptions = ["gaussian_noise"]
        args.modalities = ["text"]
        args.severities = [0.25]

    return args


# =====================================================================
# Generic helpers
# =====================================================================

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_present(
    configuration: Dict[str, Any],
    names: Sequence[str],
    default: Any,
) -> Any:
    for name in names:
        if name in configuration and configuration[name] is not None:
            return configuration[name]
    return default


def checkpoint_configuration(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    configuration = checkpoint.get("configuration", {})
    if not isinstance(configuration, dict):
        raise RuntimeError("Checkpoint configuration is not a dictionary.")
    return configuration


def assert_safe_input_path(path: Path, purpose: str) -> None:
    normalized_parts = [part.lower() for part in path.parts]
    forbidden = {
        "test",
        "testing",
        "test_cache",
        "official_test",
        "official-test",
    }
    if any(part in forbidden for part in normalized_parts):
        raise RuntimeError(
            f"Refusing {purpose} path because it appears to reference "
            f"the test split: {path}"
        )


def get_sample_ids(full_batch: Any, expected: int) -> List[str]:
    for attribute in ("sample_ids", "sample_id"):
        if hasattr(full_batch, attribute):
            value = getattr(full_batch, attribute)
            if isinstance(value, torch.Tensor):
                values = value.detach().cpu().reshape(-1).tolist()
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]
            if len(values) == expected:
                return [str(v) for v in values]

    return [f"validation_row_{i}" for i in range(expected)]


def population_stats(
    values: Sequence[Optional[float]],
) -> Dict[str, Any]:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": pstdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def state_fingerprint(
    alignment_model: torch.nn.Module,
    classification_model: torch.nn.Module,
) -> str:
    digest = hashlib.sha256()
    for prefix, model in (
        ("alignment", alignment_model),
        ("classification", classification_model),
    ):
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(prefix.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            detached = tensor.detach().contiguous().cpu()
            digest.update(str(detached.dtype).encode("utf-8"))
            digest.update(str(tuple(detached.shape)).encode("utf-8"))
            digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def binary_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
) -> Dict[str, Any]:
    if len(targets) != len(predictions) or not targets:
        raise RuntimeError("Targets/predictions must be non-empty and aligned.")

    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))

    accuracy = (tp + tn) / len(targets)
    precision_1 = tp / (tp + fp) if (tp + fp) else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1 = (
        2.0 * precision_1 * recall_1 / (precision_1 + recall_1)
        if (precision_1 + recall_1)
        else 0.0
    )

    precision_0 = tn / (tn + fn) if (tn + fn) else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) else 0.0
    f1_0 = (
        2.0 * precision_0 * recall_0 / (precision_0 + recall_0)
        if (precision_0 + recall_0)
        else 0.0
    )
    macro_f1 = (f1_0 + f1_1) / 2.0

    return {
        "accuracy": accuracy,
        "precision_class_1": precision_1,
        "recall_class_1": recall_1,
        "f1_class_1": f1_1,
        "f1_class_0": f1_0,
        "macro_f1": macro_f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


# =====================================================================
# Model reconstruction
# =====================================================================

def build_models_from_checkpoint(
    checkpoint: Dict[str, Any],
    expected_architecture: str,
    device: torch.device,
) -> Tuple[
    CrossModalAlignmentModel,
    HierarchicalInformationIntegrityClassifier,
    Dict[str, Any],
]:
    configuration = checkpoint_configuration(checkpoint)

    fusion_architecture = checkpoint.get(
        "fusion_architecture",
        configuration.get("fusion_architecture", "legacy"),
    )
    if fusion_architecture != expected_architecture:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Expected "
            f"{expected_architecture!r}, found {fusion_architecture!r}."
        )

    text_dim = int(first_present(configuration, ("text_dim",), 768))
    vision_dim = int(first_present(configuration, ("vision_dim",), 512))
    shared_dim = int(first_present(configuration, ("shared_dim",), 512))
    dropout = float(first_present(configuration, ("dropout",), 0.1))
    contrastive_temperature = float(
        first_present(
            configuration,
            ("temperature", "alignment_temperature"),
            0.07,
        )
    )
    reliability_hidden_dim = int(
        first_present(
            configuration,
            ("evidence_reliability_hidden_dim", "reliability_hidden_dim"),
            256,
        )
    )
    interaction_dropout = float(
        first_present(
            configuration,
            ("evidence_interaction_dropout",),
            0.1,
        )
    )
    reliability_dropout = float(
        first_present(
            configuration,
            ("evidence_reliability_dropout",),
            0.1,
        )
    )
    fusion_temperature = float(
        first_present(
            configuration,
            ("evidence_fusion_temperature", "fusion_temperature"),
            1.0,
        )
    )
    hidden_dim = int(
        first_present(
            configuration,
            ("hidden_dim", "classification_hidden_dim"),
            256,
        )
    )

    alignment_model = CrossModalAlignmentModel(
        text_dim=text_dim,
        vision_dim=vision_dim,
        shared_dim=shared_dim,
        dropout=dropout,
        temperature=contrastive_temperature,
        evidence_aware=(expected_architecture == "evidence_aware"),
        interaction_only=False,
        gated_interaction=(expected_architecture == "gated_interaction"),
        reliability_only=False,
        interaction_reliability=(
            expected_architecture == "interaction_reliability"
        ),
        evidence_reliability_hidden_dim=reliability_hidden_dim,
        evidence_interaction_dropout=interaction_dropout,
        evidence_reliability_dropout=reliability_dropout,
        evidence_fusion_temperature=fusion_temperature,
    ).to(device)

    classification_model = HierarchicalInformationIntegrityClassifier(
        input_dim=shared_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    alignment_state = checkpoint.get("alignment_model_state_dict")
    classification_state = checkpoint.get("classification_model_state_dict")

    if alignment_state is None:
        raise RuntimeError(
            "Checkpoint is missing alignment_model_state_dict."
        )
    if classification_state is None:
        raise RuntimeError(
            "Checkpoint is missing classification_model_state_dict."
        )

    alignment_model.load_state_dict(alignment_state, strict=True)
    classification_model.load_state_dict(classification_state, strict=True)
    alignment_model.eval()
    classification_model.eval()

    resolved = {
        "fusion_architecture": fusion_architecture,
        "text_dim": text_dim,
        "vision_dim": vision_dim,
        "shared_dim": shared_dim,
        "dropout": dropout,
        "temperature": contrastive_temperature,
        "hidden_dim": hidden_dim,
        "evidence_reliability_hidden_dim": reliability_hidden_dim,
        "evidence_interaction_dropout": interaction_dropout,
        "evidence_reliability_dropout": reliability_dropout,
        "evidence_fusion_temperature": fusion_temperature,
    }
    return alignment_model, classification_model, resolved


# =====================================================================
# Corruption protocol
# =====================================================================

def corruption_seed(
    model_seed: int,
    corruption_type: str,
    modality: str,
    severity: Optional[float],
) -> int:
    severity_code = (
        0 if severity is None else int(round(float(severity) * 1000.0))
    )
    return (
        int(model_seed)
        + 10_000
        + 10_000 * CORRUPTION_IDS[corruption_type]
        + 1_000 * MODALITY_IDS[modality]
        + severity_code
    )


def make_derangement_indices(
    n: int,
    seed: int,
) -> torch.Tensor:
    if n < 2:
        raise RuntimeError("Permutation corruption requires at least 2 samples.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shift = int(
        torch.randint(
            low=1,
            high=n,
            size=(1,),
            generator=generator,
        ).item()
    )
    return (torch.arange(n, dtype=torch.long) + shift) % n


def corrupt_tensor(
    original: torch.Tensor,
    corruption_type: str,
    severity: Optional[float],
    seed: int,
) -> torch.Tensor:
    x = original.detach().clone()

    if corruption_type == "clean":
        return x

    if corruption_type == "gaussian_noise":
        if severity is None:
            raise ValueError("Gaussian noise requires a severity.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(
            x.shape,
            generator=generator,
            dtype=x.dtype,
            device="cpu",
        ).to(x.device)
        # Per-feature empirical scale; zero-variance dimensions remain stable.
        feature_std = x.float().std(dim=0, unbiased=False).to(x.dtype)
        return x + float(severity) * feature_std.unsqueeze(0) * noise

    if corruption_type == "attenuation":
        if severity is None:
            raise ValueError("Attenuation requires a severity.")
        return (1.0 - float(severity)) * x

    if corruption_type == "permutation":
        indices = make_derangement_indices(x.shape[0], seed).to(x.device)
        return x.index_select(0, indices)

    if corruption_type == "zero_dropout":
        return torch.zeros_like(x)

    raise ValueError(f"Unsupported corruption type: {corruption_type}")


def build_conditions(
    corruptions: Sequence[str],
    modalities: Sequence[str],
    severities: Sequence[float],
) -> List[Dict[str, Any]]:
    # Exactly one clean control per model/seed.
    conditions: List[Dict[str, Any]] = [
        {
            "corruption_type": "clean",
            "corrupted_modality": "none",
            "severity": 0.0,
        }
    ]

    for corruption_type in corruptions:
        for modality in modalities:
            if corruption_type in CONTINUOUS_CORRUPTIONS:
                for severity in sorted(set(float(x) for x in severities)):
                    # Severity zero is scientifically identical to clean and
                    # is represented by the single clean control above.
                    if severity == 0.0:
                        continue
                    conditions.append(
                        {
                            "corruption_type": corruption_type,
                            "corrupted_modality": modality,
                            "severity": severity,
                        }
                    )
            else:
                conditions.append(
                    {
                        "corruption_type": corruption_type,
                        "corrupted_modality": modality,
                        "severity": None,
                    }
                )
    return conditions


# =====================================================================
# Inference
# =====================================================================

def optional_tensor_values(
    outputs: Dict[str, torch.Tensor],
    key: str,
    size: int,
) -> List[Optional[float]]:
    if key not in outputs:
        return [None] * size
    values = outputs[key].detach().float().cpu().reshape(-1)
    if values.numel() != size:
        raise RuntimeError(
            f"{key} has {values.numel()} values for batch size {size}."
        )
    return [float(x) for x in values.tolist()]


def evaluate_condition(
    *,
    architecture_key: str,
    seed: int,
    checkpoint_path: Path,
    alignment_model: CrossModalAlignmentModel,
    classification_model: HierarchicalInformationIntegrityClassifier,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    targets: torch.Tensor,
    sample_ids: Sequence[str],
    condition: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    architecture = ARCHITECTURES[architecture_key]
    corruption_type = str(condition["corruption_type"])
    modality = str(condition["corrupted_modality"])
    severity = condition["severity"]

    cseed = corruption_seed(
        seed,
        corruption_type,
        modality,
        severity,
    )

    corrupted_text = text_embeddings
    corrupted_vision = vision_embeddings

    if corruption_type != "clean":
        if modality == "text":
            corrupted_text = corrupt_tensor(
                text_embeddings,
                corruption_type,
                severity,
                cseed,
            )
        elif modality == "vision":
            corrupted_vision = corrupt_tensor(
                vision_embeddings,
                corruption_type,
                severity,
                cseed,
            )
        else:
            raise RuntimeError(
                f"Corrupted condition has invalid modality: {modality}"
            )

    rows: List[Dict[str, Any]] = []
    all_targets: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for start in range(0, targets.shape[0], batch_size):
            end = min(start + batch_size, targets.shape[0])

            text_batch = corrupted_text[start:end].to(device)
            vision_batch = corrupted_vision[start:end].to(device)
            target_batch = targets[start:end].to(device).reshape(-1)

            outputs = alignment_model(
                text_batch,
                vision_batch,
                compute_loss=False,
            )

            required = ("aligned_text", "aligned_vision", "fused_embedding")
            missing = [key for key in required if key not in outputs]
            if missing:
                raise RuntimeError(
                    f"{architecture['label']} output missing: "
                    + ", ".join(missing)
                )

            computed_cosine = F.cosine_similarity(
                outputs["aligned_text"],
                outputs["aligned_vision"],
                dim=-1,
            )

            classification_outputs = classification_model(
                outputs["fused_embedding"]
            )
            logits = classification_outputs["integrity_logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            current_size = int(target_batch.numel())
            target_values = target_batch.detach().cpu().tolist()
            prediction_values = predictions.detach().cpu().tolist()
            probability_values = probabilities[:, 1].detach().cpu().tolist()
            cosine_values = computed_cosine.detach().cpu().tolist()

            if architecture_key == "m1b":
                text_rel = [None] * current_size
                vision_rel = [None] * current_size
                text_weight = [None] * current_size
                vision_weight = [None] * current_size
            else:
                diagnostic_keys = (
                    "text_reliability",
                    "vision_reliability",
                    "text_weight",
                    "vision_weight",
                )
                diagnostic_missing = [
                    key for key in diagnostic_keys if key not in outputs
                ]
                if diagnostic_missing:
                    raise RuntimeError(
                        f"{architecture['label']} output missing reliability "
                        "diagnostics: " + ", ".join(diagnostic_missing)
                    )
                text_rel = optional_tensor_values(
                    outputs, "text_reliability", current_size
                )
                vision_rel = optional_tensor_values(
                    outputs, "vision_reliability", current_size
                )
                text_weight = optional_tensor_values(
                    outputs, "text_weight", current_size
                )
                vision_weight = optional_tensor_values(
                    outputs, "vision_weight", current_size
                )

            for local_index in range(current_size):
                global_index = start + local_index
                target = int(target_values[local_index])
                prediction = int(prediction_values[local_index])

                rows.append(
                    {
                        "architecture": architecture["label"],
                        "architecture_key": architecture_key,
                        "fusion_architecture": (
                            architecture["fusion_architecture"]
                        ),
                        "model_seed": seed,
                        "checkpoint_path": str(checkpoint_path),
                        "corruption_type": corruption_type,
                        "corrupted_modality": modality,
                        "severity": severity,
                        "corruption_seed": cseed,
                        "row_index": global_index,
                        "sample_id": str(sample_ids[global_index]),
                        "target": target,
                        "prediction": prediction,
                        "probability_class_1": float(
                            probability_values[local_index]
                        ),
                        "correct": int(target == prediction),
                        "text_reliability": text_rel[local_index],
                        "vision_reliability": vision_rel[local_index],
                        "text_weight": text_weight[local_index],
                        "vision_weight": vision_weight[local_index],
                        "cosine_similarity": float(
                            cosine_values[local_index]
                        ),
                    }
                )

            all_targets.extend(int(x) for x in target_values)
            all_predictions.extend(int(x) for x in prediction_values)

    metrics = binary_metrics(all_targets, all_predictions)
    summary: Dict[str, Any] = {
        "aegis_version": VERSION,
        "architecture": architecture["label"],
        "architecture_key": architecture_key,
        "fusion_architecture": architecture["fusion_architecture"],
        "model_seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "corruption_type": corruption_type,
        "corrupted_modality": modality,
        "severity": severity,
        "corruption_seed": cseed,
        "sample_count": len(rows),
        **metrics,
        "mean_cosine_similarity": population_stats(
            [row["cosine_similarity"] for row in rows]
        )["mean"],
        "mean_text_reliability": population_stats(
            [row["text_reliability"] for row in rows]
        )["mean"],
        "mean_vision_reliability": population_stats(
            [row["vision_reliability"] for row in rows]
        )["mean"],
        "mean_text_weight": population_stats(
            [row["text_weight"] for row in rows]
        )["mean"],
        "mean_vision_weight": population_stats(
            [row["vision_weight"] for row in rows]
        )["mean"],
        "test_split_accessed": False,
    }
    return rows, summary


# =====================================================================
# Derived clean-relative diagnostics
# =====================================================================

def add_clean_relative_deltas(
    summaries: List[Dict[str, Any]],
) -> None:
    clean_by_model: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for summary in summaries:
        if summary["corruption_type"] == "clean":
            key = (
                str(summary["architecture_key"]),
                int(summary["model_seed"]),
            )
            clean_by_model[key] = summary

    fields = (
        "accuracy",
        "macro_f1",
        "mean_cosine_similarity",
        "mean_text_reliability",
        "mean_vision_reliability",
        "mean_text_weight",
        "mean_vision_weight",
    )

    for summary in summaries:
        key = (
            str(summary["architecture_key"]),
            int(summary["model_seed"]),
        )
        clean = clean_by_model.get(key)
        if clean is None:
            raise RuntimeError(f"Missing clean control for {key}.")

        for field in fields:
            current = summary.get(field)
            baseline = clean.get(field)
            delta_key = f"delta_{field}_from_clean"
            if current is None or baseline is None:
                summary[delta_key] = None
            else:
                summary[delta_key] = float(current) - float(baseline)

        if summary["corrupted_modality"] == "text":
            d_corrupt = summary.get(
                "delta_mean_text_reliability_from_clean"
            )
            d_intact = summary.get(
                "delta_mean_vision_reliability_from_clean"
            )
        elif summary["corrupted_modality"] == "vision":
            d_corrupt = summary.get(
                "delta_mean_vision_reliability_from_clean"
            )
            d_intact = summary.get(
                "delta_mean_text_reliability_from_clean"
            )
        else:
            d_corrupt = None
            d_intact = None

        summary["reliability_selectivity"] = (
            None
            if d_corrupt is None or d_intact is None
            else float(d_intact) - float(d_corrupt)
        )


def condition_slug(summary: Dict[str, Any]) -> str:
    corruption = str(summary["corruption_type"])
    modality = str(summary["corrupted_modality"])
    severity = summary["severity"]
    if corruption == "clean":
        return "clean"
    if severity is None:
        return f"{modality}_{corruption}"
    rendered = str(severity).replace(".", "p")
    return f"{modality}_{corruption}_s{rendered}"


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()

    device = (
        resolve_device()
        if args.device is None
        else torch.device(args.device)
    )

    assert_safe_input_path(
        args.validation_cache,
        "validation-cache",
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    validation_cache = FakedditRepresentationCache(
        args.validation_cache
    )
    validation_batch = load_cache_as_single_batch(validation_cache)

    sample_count = int(validation_batch.batch_size)
    if sample_count != args.expected_validation_samples:
        raise RuntimeError(
            "Validation sample-count guard failed. Expected "
            f"{args.expected_validation_samples}, found {sample_count}."
        )

    text_embeddings = validation_batch.text_embeddings.detach().cpu()
    vision_embeddings = validation_batch.vision_embeddings.detach().cpu()
    targets = validation_batch.integrity_targets.detach().cpu().reshape(-1)

    if text_embeddings.shape[0] != sample_count:
        raise RuntimeError("Text representation count mismatch.")
    if vision_embeddings.shape[0] != sample_count:
        raise RuntimeError("Vision representation count mismatch.")
    if targets.numel() != sample_count:
        raise RuntimeError("Target count mismatch.")

    sample_ids = get_sample_ids(validation_batch, sample_count)

    conditions = build_conditions(
        args.corruptions,
        args.modalities,
        args.severities,
    )

    protocol = {
        "aegis_version": VERSION,
        "experiment": "fakeddit_reliability_stress",
        "scientific_status": (
            "diagnostic validation experiment; not independent test "
            "generalization"
        ),
        "architectures": args.architectures,
        "seeds": args.seeds,
        "validation_cache": str(args.validation_cache),
        "validation_samples": sample_count,
        "text_dimension": int(validation_cache.text_dimension),
        "vision_dimension": int(validation_cache.vision_dimension),
        "checkpoint_name": args.checkpoint_name,
        "batch_size": args.batch_size,
        "device": str(device),
        "corruptions": args.corruptions,
        "modalities": args.modalities,
        "continuous_severities": args.severities,
        "conditions": conditions,
        "gaussian_scaling": (
            "per-feature empirical population standard deviation of the "
            "fixed validation representations"
        ),
        "permutation": (
            "deterministic cyclic derangement; labels and intact modality "
            "remain fixed"
        ),
        "hypotheses": {
            "H1_sensitivity": (
                "Increasing corruption of modality m should reduce its "
                "learned reliability/weight."
            ),
            "H2_selectivity": (
                "Reliability reduction should be larger for the corrupted "
                "modality than for the intact modality."
            ),
            "H3_utility": (
                "Architectures with appropriate reliability response should "
                "show smaller classification degradation."
            ),
        },
        "official_test_split_accessed": False,
        "smoke_mode": bool(args.smoke),
    }
    save_json(args.output_root / "protocol.json", protocol)

    print("=" * 78)
    print("AEGIS v0.26.0 — FAKEDDIT RELIABILITY STRESS TEST")
    print("=" * 78)
    print("Device:", device)
    print("Validation cache:", args.validation_cache)
    print("Validation samples:", sample_count)
    print("Architectures:", args.architectures)
    print("Seeds:", args.seeds)
    print("Conditions per checkpoint:", len(conditions))
    print("Official test split: SEALED / NOT ACCESSED")
    print()

    all_summaries: List[Dict[str, Any]] = []

    for architecture_key in args.architectures:
        architecture = ARCHITECTURES[architecture_key]

        for seed in args.seeds:
            experiment_dir = Path(
                architecture["experiment_template"].format(seed=seed)
            )
            checkpoint_path = experiment_dir / args.checkpoint_name

            assert_safe_input_path(
                checkpoint_path,
                f"{architecture['label']} checkpoint",
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_path}"
                )

            print(
                f"Loading {architecture['label']} seed {seed}: "
                f"{checkpoint_path}"
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            (
                alignment_model,
                classification_model,
                resolved,
            ) = build_models_from_checkpoint(
                checkpoint,
                architecture["fusion_architecture"],
                device,
            )

            before_fingerprint = state_fingerprint(
                alignment_model,
                classification_model,
            )

            seed_dir = (
                args.output_root
                / architecture_key
                / f"seed{seed}"
            )
            seed_dir.mkdir(parents=True, exist_ok=True)

            seed_summaries: List[Dict[str, Any]] = []

            for condition in conditions:
                rows, summary = evaluate_condition(
                    architecture_key=architecture_key,
                    seed=seed,
                    checkpoint_path=checkpoint_path,
                    alignment_model=alignment_model,
                    classification_model=classification_model,
                    text_embeddings=text_embeddings,
                    vision_embeddings=vision_embeddings,
                    targets=targets,
                    sample_ids=sample_ids,
                    condition=condition,
                    batch_size=args.batch_size,
                    device=device,
                )

                summary["checkpoint_epoch"] = checkpoint.get("epoch")
                summary["checkpoint_validation_metrics"] = checkpoint.get(
                    "validation_metrics"
                )
                summary["resolved_model_configuration"] = resolved

                slug = condition_slug(summary)
                save_csv(seed_dir / f"{slug}_samples.csv", rows)
                seed_summaries.append(summary)

                print(
                    f"  {slug:38s} "
                    f"Macro-F1={summary['macro_f1']:.4f} "
                    f"Acc={summary['accuracy']:.4f}"
                )

            after_fingerprint = state_fingerprint(
                alignment_model,
                classification_model,
            )
            if before_fingerprint != after_fingerprint:
                raise RuntimeError(
                    f"Model parameters changed during evaluation for "
                    f"{architecture['label']} seed {seed}."
                )

            add_clean_relative_deltas(seed_summaries)

            for summary in seed_summaries:
                summary["model_state_fingerprint_before"] = (
                    before_fingerprint
                )
                summary["model_state_fingerprint_after"] = (
                    after_fingerprint
                )
                summary["parameters_unchanged"] = True
                save_json(
                    seed_dir / f"{condition_slug(summary)}_summary.json",
                    summary,
                )

            save_csv(seed_dir / "condition_summary.csv", seed_summaries)
            save_json(
                seed_dir / "run_manifest.json",
                {
                    "architecture": architecture["label"],
                    "architecture_key": architecture_key,
                    "fusion_architecture": (
                        architecture["fusion_architecture"]
                    ),
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "resolved_model_configuration": resolved,
                    "conditions_evaluated": len(seed_summaries),
                    "model_state_fingerprint_before": before_fingerprint,
                    "model_state_fingerprint_after": after_fingerprint,
                    "parameters_unchanged": True,
                    "test_split_accessed": False,
                },
            )

            all_summaries.extend(seed_summaries)

    # Clean-relative deltas have already been added per seed. Save one
    # machine-readable aggregate table without treating repeated seed rows as
    # independent dataset samples.
    aggregate_dir = args.output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    save_csv(aggregate_dir / "all_condition_summaries.csv", all_summaries)

    aggregate_payload: Dict[str, Any] = {
        "aegis_version": VERSION,
        "number_of_condition_evaluations": len(all_summaries),
        "architectures": args.architectures,
        "seeds": args.seeds,
        "test_split_accessed": False,
        "scientific_note": (
            "Each seed evaluates the same fixed validation examples. Seed "
            "runs are repeated model evaluations, not independent dataset "
            "samples. Corruption results are diagnostic and validation-only."
        ),
        "per_architecture": {},
    }

    for architecture_key in args.architectures:
        architecture_rows = [
            row
            for row in all_summaries
            if row["architecture_key"] == architecture_key
        ]
        clean_rows = [
            row
            for row in architecture_rows
            if row["corruption_type"] == "clean"
        ]
        aggregate_payload["per_architecture"][architecture_key] = {
            "label": ARCHITECTURES[architecture_key]["label"],
            "number_of_condition_rows": len(architecture_rows),
            "clean_macro_f1_seed_distribution": population_stats(
                [row["macro_f1"] for row in clean_rows]
            ),
            "clean_accuracy_seed_distribution": population_stats(
                [row["accuracy"] for row in clean_rows]
            ),
        }

    save_json(
        aggregate_dir / "aggregate_summary.json",
        aggregate_payload,
    )

    print()
    print("=" * 78)
    print("RELIABILITY STRESS TEST COMPLETE")
    print("=" * 78)
    print("Condition evaluations:", len(all_summaries))
    print("Output root:", args.output_root)
    print("Model parameters: UNCHANGED")
    print("Official test split: SEALED / NOT ACCESSED")


if __name__ == "__main__":
    main()
