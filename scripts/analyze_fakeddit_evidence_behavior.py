"""
AEGIS v0.25.1 — Fakeddit Evidence Behaviour Diagnostics

Purpose
-------
Interrogate trained evidence-aware AEGIS checkpoints without changing model
weights or accessing the official test split.

For each validation sample, records:
    - text reliability
    - vision reliability
    - text fusion weight
    - vision fusion weight
    - text/vision cosine similarity
    - target, prediction, class-1 probability
    - correctness and confusion category (TP/TN/FP/FN)

It produces per-seed CSV/JSON files and an aggregate JSON summary.

Scientific constraints
----------------------
1. Validation cache only.
2. Best checkpoints only by default.
3. No gradient updates.
4. No test-cache arguments or test-set access.
5. No architecture or hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

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
DEFAULT_EXPERIMENT_TEMPLATE = (
    "experiments/fakeddit/v025_m3_evidence_seed{seed}"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v025_evidence_diagnostics"
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.25.1 evidence-behaviour diagnostics on the "
            "sealed-development validation cache."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Experiment seeds to analyse. Default: 42 43 44.",
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
        help="Frozen validation representation cache.",
    )
    parser.add_argument(
        "--experiment-template",
        type=str,
        default=DEFAULT_EXPERIMENT_TEMPLATE,
        help=(
            "Experiment directory template. Must contain '{seed}'. "
            "Default: experiments/fakeddit/v025_m3_evidence_seed{seed}"
        ),
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
        help="Directory for diagnostic CSV/JSON outputs.",
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
    if "{seed}" not in args.experiment_template:
        parser.error("--experiment-template must contain '{seed}'.")

    return args


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor.")
        return value.detach().cpu().item()
    return value


def safe_float(value: Any) -> float:
    return float(to_python_scalar(value))


def safe_int(value: Any) -> int:
    return int(to_python_scalar(value))


def tensor_1d(tensor: torch.Tensor, expected: int, name: str) -> List[float]:
    values = tensor.detach().float().cpu().reshape(-1)
    if values.numel() != expected:
        raise RuntimeError(
            f"{name} has {values.numel()} values for a batch of {expected}."
        )
    return values.tolist()


def get_batch_sample_ids(batch: Any, batch_size: int) -> List[str]:
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

    # Diagnostic integrity is still preserved if the cache batch class does
    # not expose IDs. The generated IDs are explicitly marked as diagnostic.
    return [f"diagnostic_row_{i}" for i in range(batch_size)]


def population_stats(values: Sequence[float]) -> Dict[str, float | int | None]:
    clean = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
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

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)

    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    x_term = sum((x - mx) ** 2 for x in xs)
    y_term = sum((y - my) ** 2 for y in ys)

    denominator = math.sqrt(x_term * y_term)
    if denominator == 0.0:
        return None
    return numerator / denominator


def percentage(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def confusion_category(target: int, prediction: int) -> str:
    if target == 1 and prediction == 1:
        return "TP"
    if target == 0 and prediction == 0:
        return "TN"
    if target == 0 and prediction == 1:
        return "FP"
    if target == 1 and prediction == 0:
        return "FN"
    raise ValueError(
        f"Binary target/prediction expected, got target={target}, "
        f"prediction={prediction}."
    )


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("Refusing to write an empty diagnostic CSV.")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Checkpoint/model reconstruction
# ---------------------------------------------------------------------

def checkpoint_configuration(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    configuration = checkpoint.get("configuration", {})
    if not isinstance(configuration, dict):
        raise RuntimeError("Checkpoint configuration is not a dictionary.")
    return configuration


def first_present(
    configuration: Dict[str, Any],
    names: Sequence[str],
    default: Any,
) -> Any:
    for name in names:
        if name in configuration and configuration[name] is not None:
            return configuration[name]
    return default


def build_models_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> tuple[
    CrossModalAlignmentModel,
    HierarchicalInformationIntegrityClassifier,
    Dict[str, Any],
]:
    configuration = checkpoint_configuration(checkpoint)

    fusion_architecture = checkpoint.get(
        "fusion_architecture",
        configuration.get("fusion_architecture", "legacy"),
    )
    if fusion_architecture != "evidence_aware":
        raise RuntimeError(
            "Diagnostics require an evidence-aware checkpoint, but "
            f"fusion_architecture={fusion_architecture!r}."
        )

    text_dim = int(
        first_present(configuration, ("text_dim",), 768)
    )
    vision_dim = int(
        first_present(configuration, ("vision_dim",), 512)
    )
    shared_dim = int(
        first_present(configuration, ("shared_dim",), 512)
    )
    dropout = float(
        first_present(configuration, ("dropout",), 0.1)
    )
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
        evidence_aware=True,
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
    classification_state = checkpoint.get(
        "classification_model_state_dict"
    )
    if alignment_state is None or classification_state is None:
        raise RuntimeError(
            "Checkpoint is missing alignment/classification state dictionaries."
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


# ---------------------------------------------------------------------
# Diagnostic inference
# ---------------------------------------------------------------------

def analyse_checkpoint(
    *,
    seed: int,
    checkpoint_path: Path,
    validation_batch: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    alignment_model, classification_model, resolved = (
        build_models_from_checkpoint(checkpoint, device)
    )

    rows: List[Dict[str, Any]] = []
    global_index = 0

    with torch.no_grad():
        for batch in iter_sequential_batches(
            full_batch=validation_batch,
            batch_size=batch_size,
        ):
            text_embeddings = batch.text_embeddings.to(device)
            vision_embeddings = batch.vision_embeddings.to(device)
            targets_tensor = batch.integrity_targets.to(device).reshape(-1)

            alignment_outputs = alignment_model(
                text_embeddings,
                vision_embeddings,
                compute_loss=False,
            )

            required = (
                "fused_embedding",
                "cosine_similarity",
                "text_reliability",
                "vision_reliability",
                "text_weight",
                "vision_weight",
            )
            missing = [
                name for name in required
                if name not in alignment_outputs
            ]
            if missing:
                raise RuntimeError(
                    "Evidence-aware model output is missing diagnostic keys: "
                    + ", ".join(missing)
                )

            classification_outputs = classification_model(
                alignment_outputs["fused_embedding"]
            )
            logits = classification_outputs["integrity_logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions_tensor = torch.argmax(probabilities, dim=-1)

            current_size = int(targets_tensor.numel())
            sample_ids = get_batch_sample_ids(batch, current_size)

            targets = targets_tensor.detach().cpu().tolist()
            predictions = predictions_tensor.detach().cpu().tolist()
            positive_probabilities = (
                probabilities[:, 1].detach().cpu().tolist()
            )

            text_reliability = tensor_1d(
                alignment_outputs["text_reliability"],
                current_size,
                "text_reliability",
            )
            vision_reliability = tensor_1d(
                alignment_outputs["vision_reliability"],
                current_size,
                "vision_reliability",
            )
            text_weight = tensor_1d(
                alignment_outputs["text_weight"],
                current_size,
                "text_weight",
            )
            vision_weight = tensor_1d(
                alignment_outputs["vision_weight"],
                current_size,
                "vision_weight",
            )
            cosine_similarity = tensor_1d(
                alignment_outputs["cosine_similarity"],
                current_size,
                "cosine_similarity",
            )

            for local_index in range(current_size):
                target = int(targets[local_index])
                prediction = int(predictions[local_index])
                category = confusion_category(target, prediction)

                row = {
                    "seed": seed,
                    "row_index": global_index,
                    "sample_id": sample_ids[local_index],
                    "target": target,
                    "prediction": prediction,
                    "probability_class_1": float(
                        positive_probabilities[local_index]
                    ),
                    "correct": int(target == prediction),
                    "confusion_category": category,
                    "text_reliability": float(
                        text_reliability[local_index]
                    ),
                    "vision_reliability": float(
                        vision_reliability[local_index]
                    ),
                    "text_weight": float(
                        text_weight[local_index]
                    ),
                    "vision_weight": float(
                        vision_weight[local_index]
                    ),
                    "absolute_weight_gap": abs(
                        float(text_weight[local_index])
                        - float(vision_weight[local_index])
                    ),
                    "cosine_similarity": float(
                        cosine_similarity[local_index]
                    ),
                }
                rows.append(row)
                global_index += 1

    if not rows:
        raise RuntimeError("Validation diagnostics produced zero rows.")

    summary = summarize_rows(rows)
    summary["seed"] = seed
    summary["checkpoint"] = str(checkpoint_path)
    summary["checkpoint_epoch"] = checkpoint.get("epoch")
    summary["checkpoint_validation_metrics"] = checkpoint.get(
        "validation_metrics"
    )
    summary["resolved_model_configuration"] = resolved
    summary["test_split_accessed"] = False

    return rows, summary


# ---------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------

DIAGNOSTIC_FIELDS = (
    "text_reliability",
    "vision_reliability",
    "text_weight",
    "vision_weight",
    "absolute_weight_gap",
    "cosine_similarity",
)


def summarize_subset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "n": len(rows),
    }

    for field in DIAGNOSTIC_FIELDS:
        result[field] = population_stats(
            [float(row[field]) for row in rows]
        )

    if rows:
        text_greater = sum(
            row["text_weight"] > row["vision_weight"]
            for row in rows
        )
        vision_greater = sum(
            row["vision_weight"] > row["text_weight"]
            for row in rows
        )
        equal = len(rows) - text_greater - vision_greater
        near_equal_005 = sum(
            row["absolute_weight_gap"] < 0.05
            for row in rows
        )
        near_equal_010 = sum(
            row["absolute_weight_gap"] < 0.10
            for row in rows
        )

        result["weight_preference"] = {
            "text_greater_count": text_greater,
            "text_greater_percent": percentage(
                text_greater, len(rows)
            ),
            "vision_greater_count": vision_greater,
            "vision_greater_percent": percentage(
                vision_greater, len(rows)
            ),
            "exact_equal_count": equal,
            "exact_equal_percent": percentage(equal, len(rows)),
            "abs_gap_lt_0_05_count": near_equal_005,
            "abs_gap_lt_0_05_percent": percentage(
                near_equal_005, len(rows)
            ),
            "abs_gap_lt_0_10_count": near_equal_010,
            "abs_gap_lt_0_10_percent": percentage(
                near_equal_010, len(rows)
            ),
        }
    else:
        result["weight_preference"] = {}

    return result


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n": len(rows),
        "overall": summarize_subset(rows),
    }

    correct_rows = [row for row in rows if row["correct"] == 1]
    incorrect_rows = [row for row in rows if row["correct"] == 0]

    summary["by_correctness"] = {
        "correct": summarize_subset(correct_rows),
        "incorrect": summarize_subset(incorrect_rows),
    }

    summary["by_confusion_category"] = {
        category: summarize_subset(
            [
                row for row in rows
                if row["confusion_category"] == category
            ]
        )
        for category in ("TP", "TN", "FP", "FN")
    }

    summary["correlations"] = {
        "cosine_vs_text_reliability": pearson_correlation(
            [row["cosine_similarity"] for row in rows],
            [row["text_reliability"] for row in rows],
        ),
        "cosine_vs_vision_reliability": pearson_correlation(
            [row["cosine_similarity"] for row in rows],
            [row["vision_reliability"] for row in rows],
        ),
        "cosine_vs_text_weight": pearson_correlation(
            [row["cosine_similarity"] for row in rows],
            [row["text_weight"] for row in rows],
        ),
        "cosine_vs_vision_weight": pearson_correlation(
            [row["cosine_similarity"] for row in rows],
            [row["vision_weight"] for row in rows],
        ),
        "text_reliability_vs_text_weight": pearson_correlation(
            [row["text_reliability"] for row in rows],
            [row["text_weight"] for row in rows],
        ),
        "vision_reliability_vs_vision_weight": pearson_correlation(
            [row["vision_reliability"] for row in rows],
            [row["vision_weight"] for row in rows],
        ),
        "cosine_vs_absolute_weight_gap": pearson_correlation(
            [row["cosine_similarity"] for row in rows],
            [row["absolute_weight_gap"] for row in rows],
        ),
    }

    targets = [int(row["target"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    tn = sum(t == 0 and p == 0 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))

    accuracy = (tp + tn) / len(rows) if rows else None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
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


def aggregate_seed_summaries(
    all_rows: Sequence[Dict[str, Any]],
    seed_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    aggregate = summarize_rows(all_rows)
    aggregate["seeds"] = [
        int(summary["seed"]) for summary in seed_summaries
    ]
    aggregate["number_of_seed_runs"] = len(seed_summaries)
    aggregate["total_sample_evaluations"] = len(all_rows)
    aggregate["test_split_accessed"] = False

    per_seed_metrics: Dict[str, Any] = {}
    for field in DIAGNOSTIC_FIELDS:
        means = [
            summary["overall"][field]["mean"]
            for summary in seed_summaries
            if summary["overall"][field]["mean"] is not None
        ]
        per_seed_metrics[field + "_seed_mean_distribution"] = (
            population_stats(means)
        )

    aggregate["per_seed_mean_distributions"] = per_seed_metrics
    aggregate["scientific_note"] = (
        "Rows pooled across seeds are repeated evaluations of the same "
        "fixed validation examples under different trained seeds; pooled "
        "sample count must not be interpreted as independent dataset size."
    )
    return aggregate


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------

def print_seed_report(summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    classification = summary["classification_check"]

    print()
    print("=" * 72)
    print(f"AEGIS v0.25.1 EVIDENCE DIAGNOSTICS — SEED {summary['seed']}")
    print("=" * 72)
    print(f"Checkpoint epoch: {summary.get('checkpoint_epoch')}")
    print(f"Validation samples: {summary['n']}")
    print(
        "Accuracy check: "
        f"{classification['accuracy']:.4f}"
    )

    for name in (
        "text_reliability",
        "vision_reliability",
        "text_weight",
        "vision_weight",
        "cosine_similarity",
    ):
        stats = overall[name]
        print(
            f"{name:24s} "
            f"mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f} "
            f"min={stats['min']:.6f} "
            f"max={stats['max']:.6f}"
        )

    pref = overall["weight_preference"]
    print(
        "Text > vision weight:     "
        f"{pref['text_greater_percent']:.2f}%"
    )
    print(
        "Vision > text weight:     "
        f"{pref['vision_greater_percent']:.2f}%"
    )
    print(
        "|text - vision| < 0.05:   "
        f"{pref['abs_gap_lt_0_05_percent']:.2f}%"
    )

    print("Correlations:")
    for key, value in summary["correlations"].items():
        rendered = "undefined" if value is None else f"{value:.6f}"
        print(f"  {key}: {rendered}")

    print("Confusion matrix:")
    print(
        "  "
        + json.dumps(
            classification["confusion_matrix"],
            sort_keys=True,
        )
    )
    print("TEST SPLIT: SEALED / NOT ACCESSED")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.device is None:
        device = resolve_device()
    else:
        device = torch.device(args.device)

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AEGIS v0.25.1 — EVIDENCE BEHAVIOUR DIAGNOSTICS")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Validation cache: {args.validation_cache}")
    print(f"Seeds: {args.seeds}")
    print(f"Output root: {args.output_root}")
    print("Official test split: SEALED / NOT ACCESSED")
    print()

    validation_cache = FakedditRepresentationCache(
        args.validation_cache
    )
    validation_batch = load_cache_as_single_batch(
        validation_cache
    )

    all_rows: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []

    for seed in args.seeds:
        experiment_dir = Path(
            args.experiment_template.format(seed=seed)
        )
        checkpoint_path = experiment_dir / args.checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for seed {seed}: "
                f"{checkpoint_path}"
            )

        print(f"Analysing seed {seed}: {checkpoint_path}")

        rows, summary = analyse_checkpoint(
            seed=seed,
            checkpoint_path=checkpoint_path,
            validation_batch=validation_batch,
            batch_size=args.batch_size,
            device=device,
        )

        seed_csv = args.output_root / f"seed{seed}_samples.csv"
        seed_json = args.output_root / f"seed{seed}_summary.json"

        save_csv(seed_csv, rows)
        save_json(seed_json, summary)

        all_rows.extend(rows)
        seed_summaries.append(summary)

        print_seed_report(summary)
        print(f"Saved: {seed_csv}")
        print(f"Saved: {seed_json}")

    aggregate = aggregate_seed_summaries(
        all_rows,
        seed_summaries,
    )

    aggregate_path = args.output_root / "aggregate_summary.json"
    pooled_csv_path = args.output_root / "all_seeds_samples.csv"

    save_json(aggregate_path, aggregate)
    save_csv(pooled_csv_path, all_rows)

    print()
    print("=" * 72)
    print("AGGREGATE DIAGNOSTICS COMPLETE")
    print("=" * 72)
    print(f"Seed runs: {len(seed_summaries)}")
    print(f"Sample evaluations: {len(all_rows)}")
    print(
        "Mean text weight: "
        f"{aggregate['overall']['text_weight']['mean']:.6f}"
    )
    print(
        "Mean vision weight: "
        f"{aggregate['overall']['vision_weight']['mean']:.6f}"
    )
    print(
        "Near-equal weights (gap < 0.05): "
        f"{aggregate['overall']['weight_preference']['abs_gap_lt_0_05_percent']:.2f}%"
    )
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {pooled_csv_path}")
    print("Official test split: SEALED / NOT ACCESSED")


if __name__ == "__main__":
    main()
