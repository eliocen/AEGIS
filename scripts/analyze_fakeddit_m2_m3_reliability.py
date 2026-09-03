"""
AEGIS v0.25.5 — Targeted M2 vs M3 Reliability Behaviour Diagnostics

Purpose
-------
Compare the trained M2 reliability-residual pathway against the M3
interaction-conditioned reliability/adaptive-fusion pathway without changing
weights, tuning hyperparameters, or accessing the official Fakeddit test split.

M2:
    legacy gated fusion
    + modality-only reliability adaptive residual

M3:
    explicit cross-modal interaction
    -> interaction-conditioned reliability
    -> adaptive fusion

The script evaluates the same fixed validation cache for seeds 42/43/44 and
records per-sample evidence behaviour. It then performs paired M2-vs-M3
comparisons using sample_id within each seed.

Primary diagnostic questions
----------------------------
1. Do the reliability estimators saturate?
2. How much sample-to-sample variation exists in reliability and fusion weight?
3. Does one modality dominate systematically?
4. How strongly are M3 reliabilities/weights associated with cross-modal cosine?
5. Does reliability behaviour differ by target class or correctness?
6. For the same validation example and seed, how do M2 and M3 differ?
7. When M2 is correct and M3 is wrong (or vice versa), what happens to weights,
   reliability scores, and prediction confidence?

Scientific constraints
----------------------
1. Validation cache only.
2. Best checkpoints only by default.
3. No gradient updates.
4. No official test-cache argument exists.
5. No architecture/hyperparameter modification or tuning.
6. Pooled rows across seeds are repeated evaluations of the same validation
   examples and must not be interpreted as independent dataset samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

import torch

from aegis.alignment import CrossModalAlignmentModel
from aegis.classification import HierarchicalInformationIntegrityClassifier
from aegis.data.cache import FakedditRepresentationCache
from scripts.run_fakeddit_generalization import (
    iter_sequential_batches,
    load_cache_as_single_batch,
    resolve_device,
)


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
)
DEFAULT_M2_TEMPLATE = (
    "experiments/fakeddit/v025_m2_reliability_only_seed{seed}"
)
DEFAULT_M3_TEMPLATE = (
    "experiments/fakeddit/v025_m3_evidence_seed{seed}"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v025_m2_m3_reliability_diagnostics"
)


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS targeted M2-vs-M3 reliability diagnostics on the "
            "fixed Fakeddit validation cache."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Experiment seeds. Default: 42 43 44.",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
        help="Frozen validation representation cache.",
    )
    parser.add_argument(
        "--m2-experiment-template",
        type=str,
        default=DEFAULT_M2_TEMPLATE,
        help=(
            "M2 experiment directory template containing '{seed}'. "
            "Default: experiments/fakeddit/"
            "v025_m2_reliability_only_seed{seed}"
        ),
    )
    parser.add_argument(
        "--m3-experiment-template",
        type=str,
        default=DEFAULT_M3_TEMPLATE,
        help=(
            "M3 experiment directory template containing '{seed}'. "
            "Default: experiments/fakeddit/v025_m3_evidence_seed{seed}"
        ),
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename within each experiment directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for diagnostic outputs.",
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

    args = parser.parse_args()

    if not args.seeds:
        parser.error("--seeds must contain at least one seed.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if "{seed}" not in args.m2_experiment_template:
        parser.error(
            "--m2-experiment-template must contain '{seed}'."
        )
    if "{seed}" not in args.m3_experiment_template:
        parser.error(
            "--m3-experiment-template must contain '{seed}'."
        )

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
        raise RuntimeError("Refusing to write an empty CSV.")

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


def checkpoint_configuration(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    configuration = checkpoint.get("configuration", {})
    if not isinstance(configuration, dict):
        raise RuntimeError(
            "Checkpoint configuration is not a dictionary."
        )
    return configuration


def get_batch_sample_ids(
    batch: Any,
    batch_size: int,
    global_offset: int,
) -> List[str]:
    for attribute in ("sample_ids", "sample_id"):
        if hasattr(batch, attribute):
            value = getattr(batch, attribute)

            if isinstance(value, torch.Tensor):
                values = value.detach().cpu().reshape(-1).tolist()
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]

            if len(values) == batch_size:
                return [str(v) for v in values]

    return [
        f"diagnostic_row_{global_offset + i}"
        for i in range(batch_size)
    ]


def tensor_1d(
    tensor: torch.Tensor,
    expected: int,
    name: str,
) -> List[float]:
    values = tensor.detach().float().cpu().reshape(-1)
    if values.numel() != expected:
        raise RuntimeError(
            f"{name} has {values.numel()} values for a batch "
            f"of {expected}."
        )
    return values.tolist()


def population_stats(
    values: Sequence[float],
) -> Dict[str, float | int | None]:
    clean = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
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
        "std": (
            pstdev(clean)
            if len(clean) > 1
            else 0.0
        ),
        "min": min(clean),
        "max": max(clean),
    }


def pearson_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float | None:
    pairs = [
        (float(a), float(b))
        for a, b in zip(first, second)
        if (
            a is not None
            and b is not None
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        )
    ]

    if len(pairs) < 2:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]

    mean_x = mean(xs)
    mean_y = mean(ys)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in pairs
    )

    denominator_x = sum(
        (x - mean_x) ** 2
        for x in xs
    )
    denominator_y = sum(
        (y - mean_y) ** 2
        for y in ys
    )

    denominator = math.sqrt(
        denominator_x * denominator_y
    )

    if denominator == 0.0:
        return None

    return numerator / denominator


def percentage(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def confusion_category(
    target: int,
    prediction: int,
) -> str:
    if target == 1 and prediction == 1:
        return "TP"
    if target == 0 and prediction == 0:
        return "TN"
    if target == 0 and prediction == 1:
        return "FP"
    if target == 1 and prediction == 0:
        return "FN"

    raise ValueError(
        "Expected binary target/prediction, got "
        f"target={target}, prediction={prediction}."
    )


def binary_entropy(
    first_weight: float,
    second_weight: float,
) -> float:
    entropy = 0.0

    for weight in (
        float(first_weight),
        float(second_weight),
    ):
        if weight > 0.0:
            entropy -= weight * math.log(
                weight,
                2,
            )

    return entropy


def saturation_summary(
    values: Sequence[float],
) -> Dict[str, Any]:
    clean = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    n = len(clean)

    def count_if(predicate) -> int:
        return sum(
            1
            for value in clean
            if predicate(value)
        )

    return {
        "n": n,
        "lt_0_01_percent": percentage(
            count_if(lambda x: x < 0.01),
            n,
        ),
        "lt_0_05_percent": percentage(
            count_if(lambda x: x < 0.05),
            n,
        ),
        "lt_0_10_percent": percentage(
            count_if(lambda x: x < 0.10),
            n,
        ),
        "gt_0_90_percent": percentage(
            count_if(lambda x: x > 0.90),
            n,
        ),
        "gt_0_95_percent": percentage(
            count_if(lambda x: x > 0.95),
            n,
        ),
        "gt_0_99_percent": percentage(
            count_if(lambda x: x > 0.99),
            n,
        ),
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
    configuration = checkpoint_configuration(
        checkpoint
    )

    fusion_architecture = checkpoint.get(
        "fusion_architecture",
        configuration.get(
            "fusion_architecture",
            "legacy",
        ),
    )

    if fusion_architecture != expected_architecture:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Expected "
            f"{expected_architecture!r}, found "
            f"{fusion_architecture!r}."
        )

    text_dim = int(
        first_present(
            configuration,
            ("text_dim",),
            768,
        )
    )
    vision_dim = int(
        first_present(
            configuration,
            ("vision_dim",),
            512,
        )
    )
    shared_dim = int(
        first_present(
            configuration,
            ("shared_dim",),
            512,
        )
    )
    dropout = float(
        first_present(
            configuration,
            ("dropout",),
            0.1,
        )
    )
    contrastive_temperature = float(
        first_present(
            configuration,
            (
                "temperature",
                "alignment_temperature",
            ),
            0.07,
        )
    )
    reliability_hidden_dim = int(
        first_present(
            configuration,
            (
                "evidence_reliability_hidden_dim",
                "reliability_hidden_dim",
            ),
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
            (
                "evidence_fusion_temperature",
                "fusion_temperature",
            ),
            1.0,
        )
    )
    hidden_dim = int(
        first_present(
            configuration,
            (
                "hidden_dim",
                "classification_hidden_dim",
            ),
            256,
        )
    )

    alignment_model = CrossModalAlignmentModel(
        text_dim=text_dim,
        vision_dim=vision_dim,
        shared_dim=shared_dim,
        dropout=dropout,
        temperature=contrastive_temperature,
        evidence_aware=(
            expected_architecture
            == "evidence_aware"
        ),
        reliability_only=(
            expected_architecture
            == "reliability_only"
        ),
        evidence_reliability_hidden_dim=(
            reliability_hidden_dim
        ),
        evidence_interaction_dropout=(
            interaction_dropout
        ),
        evidence_reliability_dropout=(
            reliability_dropout
        ),
        evidence_fusion_temperature=(
            fusion_temperature
        ),
    ).to(device)

    classification_model = (
        HierarchicalInformationIntegrityClassifier(
            input_dim=shared_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
    )

    alignment_state = checkpoint.get(
        "alignment_model_state_dict"
    )
    classification_state = checkpoint.get(
        "classification_model_state_dict"
    )

    if alignment_state is None:
        raise RuntimeError(
            "Checkpoint is missing "
            "alignment_model_state_dict."
        )

    if classification_state is None:
        raise RuntimeError(
            "Checkpoint is missing "
            "classification_model_state_dict."
        )

    # Strict loading is intentional. If the currently installed model
    # definition cannot faithfully reconstruct the saved checkpoint,
    # diagnostics must stop rather than silently ignore parameters.
    alignment_model.load_state_dict(
        alignment_state,
        strict=True,
    )
    classification_model.load_state_dict(
        classification_state,
        strict=True,
    )

    alignment_model.eval()
    classification_model.eval()

    resolved = {
        "fusion_architecture": (
            fusion_architecture
        ),
        "text_dim": text_dim,
        "vision_dim": vision_dim,
        "shared_dim": shared_dim,
        "dropout": dropout,
        "temperature": (
            contrastive_temperature
        ),
        "hidden_dim": hidden_dim,
        "evidence_reliability_hidden_dim": (
            reliability_hidden_dim
        ),
        "evidence_interaction_dropout": (
            interaction_dropout
        ),
        "evidence_reliability_dropout": (
            reliability_dropout
        ),
        "evidence_fusion_temperature": (
            fusion_temperature
        ),
    }

    return (
        alignment_model,
        classification_model,
        resolved,
    )


# =====================================================================
# Diagnostic inference
# =====================================================================

def analyse_checkpoint(
    *,
    architecture_label: str,
    expected_architecture: str,
    seed: int,
    checkpoint_path: Path,
    validation_batch: Any,
    batch_size: int,
    device: torch.device,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
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
        checkpoint=checkpoint,
        expected_architecture=(
            expected_architecture
        ),
        device=device,
    )

    rows: List[Dict[str, Any]] = []
    global_index = 0

    with torch.no_grad():
        for batch in iter_sequential_batches(
            full_batch=validation_batch,
            batch_size=batch_size,
        ):
            text_embeddings = (
                batch.text_embeddings.to(device)
            )
            vision_embeddings = (
                batch.vision_embeddings.to(device)
            )
            targets_tensor = (
                batch.integrity_targets
                .to(device)
                .reshape(-1)
            )

            alignment_outputs = alignment_model(
                text_embeddings,
                vision_embeddings,
                compute_loss=False,
            )

            required = (
                "aligned_text",
                "aligned_vision",
                "fused_embedding",
                "text_reliability",
                "vision_reliability",
                "text_weight",
                "vision_weight",
            )

            missing = [
                name
                for name in required
                if name not in alignment_outputs
            ]

            if missing:
                raise RuntimeError(
                    f"{architecture_label} model output is "
                    "missing diagnostic keys: "
                    + ", ".join(missing)
                )

            # Compute cosine from aligned modality embeddings for BOTH
            # architectures so M2 and M3 use exactly the same diagnostic
            # definition. For M3 this can also be checked against the
            # model-exposed cosine value.
            computed_cosine = (
                torch.nn.functional.cosine_similarity(
                    alignment_outputs[
                        "aligned_text"
                    ],
                    alignment_outputs[
                        "aligned_vision"
                    ],
                    dim=-1,
                )
                .unsqueeze(-1)
            )

            if (
                expected_architecture
                == "evidence_aware"
                and "cosine_similarity"
                in alignment_outputs
            ):
                exposed_cosine = (
                    alignment_outputs[
                        "cosine_similarity"
                    ]
                )

                if not torch.allclose(
                    computed_cosine,
                    exposed_cosine,
                    atol=1e-6,
                    rtol=1e-5,
                ):
                    raise RuntimeError(
                        "M3 exposed cosine_similarity does not "
                        "match cosine computed from aligned "
                        "embeddings."
                    )

            classification_outputs = (
                classification_model(
                    alignment_outputs[
                        "fused_embedding"
                    ]
                )
            )

            logits = classification_outputs[
                "integrity_logits"
            ]

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions_tensor = torch.argmax(
                probabilities,
                dim=-1,
            )

            current_size = int(
                targets_tensor.numel()
            )

            sample_ids = get_batch_sample_ids(
                batch,
                current_size,
                global_index,
            )

            targets = (
                targets_tensor
                .detach()
                .cpu()
                .tolist()
            )
            predictions = (
                predictions_tensor
                .detach()
                .cpu()
                .tolist()
            )
            positive_probabilities = (
                probabilities[:, 1]
                .detach()
                .cpu()
                .tolist()
            )

            text_reliability = tensor_1d(
                alignment_outputs[
                    "text_reliability"
                ],
                current_size,
                "text_reliability",
            )
            vision_reliability = tensor_1d(
                alignment_outputs[
                    "vision_reliability"
                ],
                current_size,
                "vision_reliability",
            )
            text_weight = tensor_1d(
                alignment_outputs[
                    "text_weight"
                ],
                current_size,
                "text_weight",
            )
            vision_weight = tensor_1d(
                alignment_outputs[
                    "vision_weight"
                ],
                current_size,
                "vision_weight",
            )
            cosine_similarity = tensor_1d(
                computed_cosine,
                current_size,
                "cosine_similarity",
            )

            for local_index in range(
                current_size
            ):
                target = int(
                    targets[local_index]
                )
                prediction = int(
                    predictions[local_index]
                )

                text_rel = float(
                    text_reliability[
                        local_index
                    ]
                )
                vision_rel = float(
                    vision_reliability[
                        local_index
                    ]
                )
                text_w = float(
                    text_weight[
                        local_index
                    ]
                )
                vision_w = float(
                    vision_weight[
                        local_index
                    ]
                )

                row = {
                    "architecture": (
                        architecture_label
                    ),
                    "fusion_architecture": (
                        expected_architecture
                    ),
                    "seed": seed,
                    "row_index": (
                        global_index
                    ),
                    "sample_id": (
                        sample_ids[
                            local_index
                        ]
                    ),
                    "target": target,
                    "prediction": prediction,
                    "probability_class_1": float(
                        positive_probabilities[
                            local_index
                        ]
                    ),
                    "correct": int(
                        target == prediction
                    ),
                    "confusion_category": (
                        confusion_category(
                            target,
                            prediction,
                        )
                    ),
                    "text_reliability": (
                        text_rel
                    ),
                    "vision_reliability": (
                        vision_rel
                    ),
                    "reliability_gap_text_minus_vision": (
                        text_rel
                        - vision_rel
                    ),
                    "absolute_reliability_gap": abs(
                        text_rel
                        - vision_rel
                    ),
                    "text_weight": text_w,
                    "vision_weight": vision_w,
                    "weight_gap_text_minus_vision": (
                        text_w
                        - vision_w
                    ),
                    "absolute_weight_gap": abs(
                        text_w
                        - vision_w
                    ),
                    "weight_entropy_bits": (
                        binary_entropy(
                            text_w,
                            vision_w,
                        )
                    ),
                    "cosine_similarity": float(
                        cosine_similarity[
                            local_index
                        ]
                    ),
                }

                rows.append(row)
                global_index += 1

    if not rows:
        raise RuntimeError(
            "Validation diagnostics produced zero rows."
        )

    summary = summarize_rows(rows)
    summary.update(
        {
            "architecture": (
                architecture_label
            ),
            "fusion_architecture": (
                expected_architecture
            ),
            "seed": seed,
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_epoch": (
                checkpoint.get("epoch")
            ),
            "checkpoint_validation_metrics": (
                checkpoint.get(
                    "validation_metrics"
                )
            ),
            "resolved_model_configuration": (
                resolved
            ),
            "test_split_accessed": False,
        }
    )

    return rows, summary


# =====================================================================
# Statistical summaries
# =====================================================================

DIAGNOSTIC_FIELDS = (
    "text_reliability",
    "vision_reliability",
    "reliability_gap_text_minus_vision",
    "absolute_reliability_gap",
    "text_weight",
    "vision_weight",
    "weight_gap_text_minus_vision",
    "absolute_weight_gap",
    "weight_entropy_bits",
    "cosine_similarity",
    "probability_class_1",
)


def summarize_subset(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": len(rows),
    }

    for field in DIAGNOSTIC_FIELDS:
        result[field] = population_stats(
            [
                float(row[field])
                for row in rows
            ]
        )

    if not rows:
        result["weight_preference"] = {}
        result["reliability_preference"] = {}
        result["reliability_saturation"] = {}
        result["weight_extremity"] = {}
        return result

    n = len(rows)

    text_weight_greater = sum(
        row["text_weight"]
        > row["vision_weight"]
        for row in rows
    )
    vision_weight_greater = sum(
        row["vision_weight"]
        > row["text_weight"]
        for row in rows
    )
    exact_weight_equal = (
        n
        - text_weight_greater
        - vision_weight_greater
    )

    text_rel_greater = sum(
        row["text_reliability"]
        > row["vision_reliability"]
        for row in rows
    )
    vision_rel_greater = sum(
        row["vision_reliability"]
        > row["text_reliability"]
        for row in rows
    )
    exact_rel_equal = (
        n
        - text_rel_greater
        - vision_rel_greater
    )

    result["weight_preference"] = {
        "text_greater_count": (
            text_weight_greater
        ),
        "text_greater_percent": percentage(
            text_weight_greater,
            n,
        ),
        "vision_greater_count": (
            vision_weight_greater
        ),
        "vision_greater_percent": percentage(
            vision_weight_greater,
            n,
        ),
        "exact_equal_count": (
            exact_weight_equal
        ),
        "exact_equal_percent": percentage(
            exact_weight_equal,
            n,
        ),
        "abs_gap_lt_0_05_percent": percentage(
            sum(
                row["absolute_weight_gap"]
                < 0.05
                for row in rows
            ),
            n,
        ),
        "abs_gap_lt_0_10_percent": percentage(
            sum(
                row["absolute_weight_gap"]
                < 0.10
                for row in rows
            ),
            n,
        ),
    }

    result["reliability_preference"] = {
        "text_greater_count": (
            text_rel_greater
        ),
        "text_greater_percent": percentage(
            text_rel_greater,
            n,
        ),
        "vision_greater_count": (
            vision_rel_greater
        ),
        "vision_greater_percent": percentage(
            vision_rel_greater,
            n,
        ),
        "exact_equal_count": (
            exact_rel_equal
        ),
        "exact_equal_percent": percentage(
            exact_rel_equal,
            n,
        ),
    }

    result["reliability_saturation"] = {
        "text": saturation_summary(
            [
                row["text_reliability"]
                for row in rows
            ]
        ),
        "vision": saturation_summary(
            [
                row["vision_reliability"]
                for row in rows
            ]
        ),
    }

    result["weight_extremity"] = {
        "text_weight_lt_0_30_percent": percentage(
            sum(
                row["text_weight"] < 0.30
                for row in rows
            ),
            n,
        ),
        "text_weight_gt_0_70_percent": percentage(
            sum(
                row["text_weight"] > 0.70
                for row in rows
            ),
            n,
        ),
        "vision_weight_lt_0_30_percent": percentage(
            sum(
                row["vision_weight"] < 0.30
                for row in rows
            ),
            n,
        ),
        "vision_weight_gt_0_70_percent": percentage(
            sum(
                row["vision_weight"] > 0.70
                for row in rows
            ),
            n,
        ),
        "entropy_lt_0_90_bits_percent": percentage(
            sum(
                row["weight_entropy_bits"] < 0.90
                for row in rows
            ),
            n,
        ),
        "entropy_lt_0_80_bits_percent": percentage(
            sum(
                row["weight_entropy_bits"] < 0.80
                for row in rows
            ),
            n,
        ),
    }

    return result


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n": len(rows),
        "overall": summarize_subset(rows),
    }

    summary["by_target"] = {
        str(target): summarize_subset(
            [
                row
                for row in rows
                if int(row["target"]) == target
            ]
        )
        for target in (0, 1)
    }

    summary["by_correctness"] = {
        "correct": summarize_subset(
            [
                row
                for row in rows
                if int(row["correct"]) == 1
            ]
        ),
        "incorrect": summarize_subset(
            [
                row
                for row in rows
                if int(row["correct"]) == 0
            ]
        ),
    }

    summary["by_confusion_category"] = {
        category: summarize_subset(
            [
                row
                for row in rows
                if row[
                    "confusion_category"
                ] == category
            ]
        )
        for category in (
            "TP",
            "TN",
            "FP",
            "FN",
        )
    }

    summary["correlations"] = {
        "cosine_vs_text_reliability": (
            pearson_correlation(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "text_reliability"
                    ]
                    for row in rows
                ],
            )
        ),
        "cosine_vs_vision_reliability": (
            pearson_correlation(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "vision_reliability"
                    ]
                    for row in rows
                ],
            )
        ),
        "cosine_vs_text_weight": (
            pearson_correlation(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "text_weight"
                    ]
                    for row in rows
                ],
            )
        ),
        "cosine_vs_vision_weight": (
            pearson_correlation(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "vision_weight"
                    ]
                    for row in rows
                ],
            )
        ),
        "cosine_vs_weight_gap": (
            pearson_correlation(
                [
                    row[
                        "cosine_similarity"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "weight_gap_text_minus_vision"
                    ]
                    for row in rows
                ],
            )
        ),
        "text_reliability_vs_text_weight": (
            pearson_correlation(
                [
                    row[
                        "text_reliability"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "text_weight"
                    ]
                    for row in rows
                ],
            )
        ),
        "vision_reliability_vs_vision_weight": (
            pearson_correlation(
                [
                    row[
                        "vision_reliability"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "vision_weight"
                    ]
                    for row in rows
                ],
            )
        ),
        "reliability_gap_vs_weight_gap": (
            pearson_correlation(
                [
                    row[
                        "reliability_gap_text_minus_vision"
                    ]
                    for row in rows
                ],
                [
                    row[
                        "weight_gap_text_minus_vision"
                    ]
                    for row in rows
                ],
            )
        ),
        "absolute_weight_gap_vs_correct": (
            pearson_correlation(
                [
                    row[
                        "absolute_weight_gap"
                    ]
                    for row in rows
                ],
                [
                    row["correct"]
                    for row in rows
                ],
            )
        ),
        "weight_entropy_vs_correct": (
            pearson_correlation(
                [
                    row[
                        "weight_entropy_bits"
                    ]
                    for row in rows
                ],
                [
                    row["correct"]
                    for row in rows
                ],
            )
        ),
    }

    targets = [
        int(row["target"])
        for row in rows
    ]
    predictions = [
        int(row["prediction"])
        for row in rows
    ]

    tp = sum(
        target == 1
        and prediction == 1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
    )
    tn = sum(
        target == 0
        and prediction == 0
        for target, prediction
        in zip(
            targets,
            predictions,
        )
    )
    fp = sum(
        target == 0
        and prediction == 1
        for target, prediction
        in zip(
            targets,
            predictions,
        )
    )
    fn = sum(
        target == 1
        and prediction == 0
        for target, prediction
        in zip(
            targets,
            predictions,
        )
    )

    accuracy = (
        (tp + tn) / len(rows)
        if rows
        else None
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )
    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )
    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    summary["classification_check"] = {
        "accuracy": accuracy,
        "precision_class_1": precision,
        "recall_class_1": recall,
        "f1_class_1": f1,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    return summary


def aggregate_architecture(
    rows: Sequence[Dict[str, Any]],
    seed_summaries: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    aggregate = summarize_rows(rows)

    aggregate["seeds"] = [
        int(summary["seed"])
        for summary in seed_summaries
    ]
    aggregate["number_of_seed_runs"] = (
        len(seed_summaries)
    )
    aggregate["total_sample_evaluations"] = (
        len(rows)
    )
    aggregate["test_split_accessed"] = False

    per_seed_means: Dict[str, Any] = {}

    for field in DIAGNOSTIC_FIELDS:
        means = [
            summary["overall"][field]["mean"]
            for summary in seed_summaries
            if summary["overall"][field][
                "mean"
            ] is not None
        ]

        per_seed_means[
            field
            + "_seed_mean_distribution"
        ] = population_stats(means)

    aggregate[
        "per_seed_mean_distributions"
    ] = per_seed_means

    aggregate["scientific_note"] = (
        "Rows pooled across seeds are repeated "
        "evaluations of the same fixed validation "
        "examples under different trained seeds; "
        "the pooled sample count is not an "
        "independent dataset size."
    )

    return aggregate


# =====================================================================
# Paired M2-vs-M3 analysis
# =====================================================================

def pair_rows(
    m2_rows: Sequence[Dict[str, Any]],
    m3_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    m2_index = {
        (
            int(row["seed"]),
            str(row["sample_id"]),
        ): row
        for row in m2_rows
    }

    m3_index = {
        (
            int(row["seed"]),
            str(row["sample_id"]),
        ): row
        for row in m3_rows
    }

    if set(m2_index) != set(m3_index):
        only_m2 = (
            set(m2_index)
            - set(m3_index)
        )
        only_m3 = (
            set(m3_index)
            - set(m2_index)
        )

        raise RuntimeError(
            "M2/M3 diagnostic rows do not pair "
            "exactly by (seed, sample_id). "
            f"Only M2: {len(only_m2)}; "
            f"only M3: {len(only_m3)}."
        )

    paired: List[Dict[str, Any]] = []

    for key in sorted(
        m2_index,
        key=lambda item: (
            item[0],
            item[1],
        ),
    ):
        m2 = m2_index[key]
        m3 = m3_index[key]

        if int(m2["target"]) != int(
            m3["target"]
        ):
            raise RuntimeError(
                "Paired target mismatch for "
                f"{key}."
            )

        if (
            int(m2["correct"]) == 1
            and int(m3["correct"]) == 1
        ):
            transition = "both_correct"
        elif (
            int(m2["correct"]) == 1
            and int(m3["correct"]) == 0
        ):
            transition = "m2_correct_m3_wrong"
        elif (
            int(m2["correct"]) == 0
            and int(m3["correct"]) == 1
        ):
            transition = "m2_wrong_m3_correct"
        else:
            transition = "both_wrong"

        paired.append(
            {
                "seed": int(key[0]),
                "sample_id": str(
                    key[1]
                ),
                "target": int(
                    m2["target"]
                ),
                "m2_prediction": int(
                    m2["prediction"]
                ),
                "m3_prediction": int(
                    m3["prediction"]
                ),
                "m2_correct": int(
                    m2["correct"]
                ),
                "m3_correct": int(
                    m3["correct"]
                ),
                "correctness_transition": (
                    transition
                ),
                "m2_probability_class_1": (
                    float(
                        m2[
                            "probability_class_1"
                        ]
                    )
                ),
                "m3_probability_class_1": (
                    float(
                        m3[
                            "probability_class_1"
                        ]
                    )
                ),
                "delta_probability_m3_minus_m2": (
                    float(
                        m3[
                            "probability_class_1"
                        ]
                    )
                    - float(
                        m2[
                            "probability_class_1"
                        ]
                    )
                ),
                "m2_text_reliability": (
                    float(
                        m2[
                            "text_reliability"
                        ]
                    )
                ),
                "m3_text_reliability": (
                    float(
                        m3[
                            "text_reliability"
                        ]
                    )
                ),
                "delta_text_reliability_m3_minus_m2": (
                    float(
                        m3[
                            "text_reliability"
                        ]
                    )
                    - float(
                        m2[
                            "text_reliability"
                        ]
                    )
                ),
                "m2_vision_reliability": (
                    float(
                        m2[
                            "vision_reliability"
                        ]
                    )
                ),
                "m3_vision_reliability": (
                    float(
                        m3[
                            "vision_reliability"
                        ]
                    )
                ),
                "delta_vision_reliability_m3_minus_m2": (
                    float(
                        m3[
                            "vision_reliability"
                        ]
                    )
                    - float(
                        m2[
                            "vision_reliability"
                        ]
                    )
                ),
                "m2_text_weight": (
                    float(
                        m2[
                            "text_weight"
                        ]
                    )
                ),
                "m3_text_weight": (
                    float(
                        m3[
                            "text_weight"
                        ]
                    )
                ),
                "delta_text_weight_m3_minus_m2": (
                    float(
                        m3[
                            "text_weight"
                        ]
                    )
                    - float(
                        m2[
                            "text_weight"
                        ]
                    )
                ),
                "m2_vision_weight": (
                    float(
                        m2[
                            "vision_weight"
                        ]
                    )
                ),
                "m3_vision_weight": (
                    float(
                        m3[
                            "vision_weight"
                        ]
                    )
                ),
                "delta_vision_weight_m3_minus_m2": (
                    float(
                        m3[
                            "vision_weight"
                        ]
                    )
                    - float(
                        m2[
                            "vision_weight"
                        ]
                    )
                ),
                "m2_absolute_weight_gap": (
                    float(
                        m2[
                            "absolute_weight_gap"
                        ]
                    )
                ),
                "m3_absolute_weight_gap": (
                    float(
                        m3[
                            "absolute_weight_gap"
                        ]
                    )
                ),
                "delta_absolute_weight_gap_m3_minus_m2": (
                    float(
                        m3[
                            "absolute_weight_gap"
                        ]
                    )
                    - float(
                        m2[
                            "absolute_weight_gap"
                        ]
                    )
                ),
                "m2_weight_entropy_bits": (
                    float(
                        m2[
                            "weight_entropy_bits"
                        ]
                    )
                ),
                "m3_weight_entropy_bits": (
                    float(
                        m3[
                            "weight_entropy_bits"
                        ]
                    )
                ),
                "delta_weight_entropy_m3_minus_m2": (
                    float(
                        m3[
                            "weight_entropy_bits"
                        ]
                    )
                    - float(
                        m2[
                            "weight_entropy_bits"
                        ]
                    )
                ),
                "m2_cosine_similarity": (
                    float(
                        m2[
                            "cosine_similarity"
                        ]
                    )
                ),
                "m3_cosine_similarity": (
                    float(
                        m3[
                            "cosine_similarity"
                        ]
                    )
                ),
                "delta_cosine_m3_minus_m2": (
                    float(
                        m3[
                            "cosine_similarity"
                        ]
                    )
                    - float(
                        m2[
                            "cosine_similarity"
                        ]
                    )
                ),
            }
        )

    return paired


PAIRED_DELTA_FIELDS = (
    "delta_probability_m3_minus_m2",
    "delta_text_reliability_m3_minus_m2",
    "delta_vision_reliability_m3_minus_m2",
    "delta_text_weight_m3_minus_m2",
    "delta_vision_weight_m3_minus_m2",
    "delta_absolute_weight_gap_m3_minus_m2",
    "delta_weight_entropy_m3_minus_m2",
    "delta_cosine_m3_minus_m2",
)


def summarize_paired_subset(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": len(rows),
    }

    for field in PAIRED_DELTA_FIELDS:
        result[field] = population_stats(
            [
                float(row[field])
                for row in rows
            ]
        )

    return result


def summarize_paired_rows(
    paired_rows: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    transitions = (
        "both_correct",
        "m2_correct_m3_wrong",
        "m2_wrong_m3_correct",
        "both_wrong",
    )

    transition_counts = {
        transition: sum(
            row[
                "correctness_transition"
            ] == transition
            for row in paired_rows
        )
        for transition in transitions
    }

    summary: Dict[str, Any] = {
        "n": len(paired_rows),
        "overall_deltas": (
            summarize_paired_subset(
                paired_rows
            )
        ),
        "correctness_transitions": {
            transition: {
                "count": (
                    transition_counts[
                        transition
                    ]
                ),
                "percent": percentage(
                    transition_counts[
                        transition
                    ],
                    len(paired_rows),
                ),
            }
            for transition
            in transitions
        },
        "by_correctness_transition": {
            transition: (
                summarize_paired_subset(
                    [
                        row
                        for row
                        in paired_rows
                        if row[
                            "correctness_transition"
                        ] == transition
                    ]
                )
            )
            for transition
            in transitions
        },
        "by_target": {
            str(target): (
                summarize_paired_subset(
                    [
                        row
                        for row
                        in paired_rows
                        if int(
                            row["target"]
                        ) == target
                    ]
                )
            )
            for target
            in (0, 1)
        },
        "test_split_accessed": False,
    }

    return summary


# =====================================================================
# Console output
# =====================================================================

def print_architecture_seed_report(
    summary: Dict[str, Any],
) -> None:
    overall = summary["overall"]
    classification = (
        summary[
            "classification_check"
        ]
    )

    print()
    print("=" * 78)
    print(
        f"{summary['architecture']} "
        f"RELIABILITY DIAGNOSTICS — "
        f"SEED {summary['seed']}"
    )
    print("=" * 78)
    print(
        "Checkpoint epoch:",
        summary.get(
            "checkpoint_epoch"
        ),
    )
    print(
        "Validation accuracy:",
        f"{classification['accuracy']:.4f}",
    )
    print(
        "Text reliability:",
        f"mean={overall['text_reliability']['mean']:.6f}",
        f"std={overall['text_reliability']['std']:.6f}",
    )
    print(
        "Vision reliability:",
        f"mean={overall['vision_reliability']['mean']:.6f}",
        f"std={overall['vision_reliability']['std']:.6f}",
    )
    print(
        "Text weight:",
        f"mean={overall['text_weight']['mean']:.6f}",
        f"std={overall['text_weight']['std']:.6f}",
    )
    print(
        "Vision weight:",
        f"mean={overall['vision_weight']['mean']:.6f}",
        f"std={overall['vision_weight']['std']:.6f}",
    )
    print(
        "Weight entropy:",
        f"mean={overall['weight_entropy_bits']['mean']:.6f} bits",
    )
    print(
        "Cosine similarity:",
        f"mean={overall['cosine_similarity']['mean']:.6f}",
        f"std={overall['cosine_similarity']['std']:.6f}",
    )

    pref = overall[
        "weight_preference"
    ]

    print(
        "Text > vision weight:",
        f"{pref['text_greater_percent']:.2f}%",
    )
    print(
        "Vision > text weight:",
        f"{pref['vision_greater_percent']:.2f}%",
    )
    print(
        "|weight gap| < 0.05:",
        f"{pref['abs_gap_lt_0_05_percent']:.2f}%",
    )

    text_sat = (
        overall[
            "reliability_saturation"
        ]["text"]
    )
    vision_sat = (
        overall[
            "reliability_saturation"
        ]["vision"]
    )

    print(
        "Text reliability <0.05 / >0.95:",
        f"{text_sat['lt_0_05_percent']:.2f}% / "
        f"{text_sat['gt_0_95_percent']:.2f}%",
    )
    print(
        "Vision reliability <0.05 / >0.95:",
        f"{vision_sat['lt_0_05_percent']:.2f}% / "
        f"{vision_sat['gt_0_95_percent']:.2f}%",
    )

    print("Correlations:")
    for key in (
        "cosine_vs_text_reliability",
        "cosine_vs_vision_reliability",
        "cosine_vs_text_weight",
        "cosine_vs_vision_weight",
    ):
        value = summary[
            "correlations"
        ][key]
        rendered = (
            "undefined"
            if value is None
            else f"{value:.6f}"
        )
        print(
            f"  {key}: {rendered}"
        )

    print(
        "TEST SPLIT: SEALED / NOT ACCESSED"
    )


def print_paired_report(
    summary: Dict[str, Any],
) -> None:
    print()
    print("=" * 78)
    print(
        "PAIRED M2-vs-M3 DIAGNOSTIC SUMMARY"
    )
    print("=" * 78)

    print(
        "Paired sample evaluations:",
        summary["n"],
    )

    print("Correctness transitions:")

    for name, payload in (
        summary[
            "correctness_transitions"
        ].items()
    ):
        print(
            f"  {name:24s} "
            f"{payload['count']:4d} "
            f"({payload['percent']:.2f}%)"
        )

    deltas = summary[
        "overall_deltas"
    ]

    for field in (
        "delta_text_reliability_m3_minus_m2",
        "delta_vision_reliability_m3_minus_m2",
        "delta_text_weight_m3_minus_m2",
        "delta_vision_weight_m3_minus_m2",
        "delta_absolute_weight_gap_m3_minus_m2",
        "delta_weight_entropy_m3_minus_m2",
        "delta_cosine_m3_minus_m2",
    ):
        stats = deltas[field]
        print(
            f"{field}: "
            f"mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f}"
        )

    print(
        "TEST SPLIT: SEALED / NOT ACCESSED"
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    args = parse_args()

    if args.device is None:
        device = resolve_device()
    else:
        device = torch.device(
            args.device
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "AEGIS v0.25.5 — TARGETED "
        "M2-vs-M3 RELIABILITY DIAGNOSTICS"
    )
    print("=" * 78)
    print("Device:", device)
    print(
        "Validation cache:",
        args.validation_cache,
    )
    print("Seeds:", args.seeds)
    print(
        "M2 template:",
        args.m2_experiment_template,
    )
    print(
        "M3 template:",
        args.m3_experiment_template,
    )
    print(
        "Output root:",
        args.output_root,
    )
    print(
        "Official test split: "
        "SEALED / NOT ACCESSED"
    )
    print()

    validation_cache = (
        FakedditRepresentationCache(
            args.validation_cache
        )
    )

    validation_batch = (
        load_cache_as_single_batch(
            validation_cache
        )
    )

    print(
        "Validation samples loaded:",
        validation_batch.batch_size,
    )
    print(
        "Validation text dimension:",
        validation_cache.text_dimension,
    )
    print(
        "Validation vision dimension:",
        validation_cache.vision_dimension,
    )
    print()

    all_m2_rows: List[
        Dict[str, Any]
    ] = []
    all_m3_rows: List[
        Dict[str, Any]
    ] = []

    m2_seed_summaries: List[
        Dict[str, Any]
    ] = []
    m3_seed_summaries: List[
        Dict[str, Any]
    ] = []

    for seed in args.seeds:
        m2_checkpoint = (
            Path(
                args
                .m2_experiment_template
                .format(seed=seed)
            )
            / args.checkpoint_name
        )

        m3_checkpoint = (
            Path(
                args
                .m3_experiment_template
                .format(seed=seed)
            )
            / args.checkpoint_name
        )

        if not m2_checkpoint.exists():
            raise FileNotFoundError(
                "M2 checkpoint not found for "
                f"seed {seed}: "
                f"{m2_checkpoint}"
            )

        if not m3_checkpoint.exists():
            raise FileNotFoundError(
                "M3 checkpoint not found for "
                f"seed {seed}: "
                f"{m3_checkpoint}"
            )

        print(
            f"Analysing M2 seed {seed}: "
            f"{m2_checkpoint}"
        )

        (
            m2_rows,
            m2_summary,
        ) = analyse_checkpoint(
            architecture_label="M2",
            expected_architecture=(
                "reliability_only"
            ),
            seed=seed,
            checkpoint_path=m2_checkpoint,
            validation_batch=(
                validation_batch
            ),
            batch_size=args.batch_size,
            device=device,
        )

        print_architecture_seed_report(
            m2_summary
        )

        print(
            f"Analysing M3 seed {seed}: "
            f"{m3_checkpoint}"
        )

        (
            m3_rows,
            m3_summary,
        ) = analyse_checkpoint(
            architecture_label="M3",
            expected_architecture=(
                "evidence_aware"
            ),
            seed=seed,
            checkpoint_path=m3_checkpoint,
            validation_batch=(
                validation_batch
            ),
            batch_size=args.batch_size,
            device=device,
        )

        print_architecture_seed_report(
            m3_summary
        )

        save_csv(
            args.output_root
            / f"m2_seed{seed}_samples.csv",
            m2_rows,
        )
        save_json(
            args.output_root
            / f"m2_seed{seed}_summary.json",
            m2_summary,
        )
        save_csv(
            args.output_root
            / f"m3_seed{seed}_samples.csv",
            m3_rows,
        )
        save_json(
            args.output_root
            / f"m3_seed{seed}_summary.json",
            m3_summary,
        )

        all_m2_rows.extend(
            m2_rows
        )
        all_m3_rows.extend(
            m3_rows
        )
        m2_seed_summaries.append(
            m2_summary
        )
        m3_seed_summaries.append(
            m3_summary
        )

    m2_aggregate = (
        aggregate_architecture(
            all_m2_rows,
            m2_seed_summaries,
        )
    )

    m3_aggregate = (
        aggregate_architecture(
            all_m3_rows,
            m3_seed_summaries,
        )
    )

    paired_rows = pair_rows(
        all_m2_rows,
        all_m3_rows,
    )

    paired_summary = (
        summarize_paired_rows(
            paired_rows
        )
    )

    save_json(
        args.output_root
        / "m2_aggregate_summary.json",
        m2_aggregate,
    )
    save_json(
        args.output_root
        / "m3_aggregate_summary.json",
        m3_aggregate,
    )
    save_csv(
        args.output_root
        / "m2_all_seeds_samples.csv",
        all_m2_rows,
    )
    save_csv(
        args.output_root
        / "m3_all_seeds_samples.csv",
        all_m3_rows,
    )
    save_csv(
        args.output_root
        / "m2_vs_m3_paired_samples.csv",
        paired_rows,
    )
    save_json(
        args.output_root
        / "m2_vs_m3_paired_summary.json",
        paired_summary,
    )

    print_paired_report(
        paired_summary
    )

    print()
    print("=" * 78)
    print(
        "TARGETED RELIABILITY "
        "DIAGNOSTICS COMPLETE"
    )
    print("=" * 78)
    print(
        "M2 sample evaluations:",
        len(all_m2_rows),
    )
    print(
        "M3 sample evaluations:",
        len(all_m3_rows),
    )
    print(
        "Paired evaluations:",
        len(paired_rows),
    )
    print(
        "Saved output root:",
        args.output_root,
    )
    print(
        "Official test split: "
        "SEALED / NOT ACCESSED"
    )


if __name__ == "__main__":
    main()
