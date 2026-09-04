"""
AEGIS v0.27 Step12B deterministic M4qc compatibility diagnostics.

This runner is evaluation-only. It loads the frozen best M4qc checkpoints for
seeds 42/43/44, constructs the frozen deterministic class-preserving validation
derangement from Step11A, and persists matched/mismatched q_T, q_V and c_TV
scores together with representation/source identity.

It deliberately does NOT compute ROC-AUC, H4-C, H5-C, the classification-
preservation decision, threshold tuning, checkpoint reselection, or M4qcf.
Those decisions remain reserved for Step12C.

Official Fakeddit test access: forbidden / zero samples.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

from aegis.alignment import CrossModalAlignmentModel
from aegis.data.cache import FakedditRepresentationCache
from scripts.run_fakeddit_generalization import (
    load_cache_as_single_batch,
    resolve_device,
)


PROTOCOL_VERSION = "0.27.0-step11a"
RUNNER_VERSION = "0.27.0-step12b"
FORMAL_SEEDS = (42, 43, 44)
EXPECTED_VALIDATION_SAMPLES = 1000
DERANGEMENT_SEED_BASE = 27_110_000

DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
)

DEFAULT_CHECKPOINT_ROOT = Path("experiments/fakeddit")
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v027_m4qc_compatibility_diagnostics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Step12B deterministic class-preserving M4qc "
            "compatibility diagnostics on Fakeddit validation only."
        )
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional torch device override, e.g. cpu or cuda. "
            "Default uses the repository device resolver."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing Step12B output directory. Use only to rerun "
            "the same frozen diagnostic implementation, never to tune from results."
        ),
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is non-finite: {value!r}")
    if value < 0.0 or value > 1.0:
        raise RuntimeError(f"{name} is outside [0,1]: {value!r}")
    return value


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors can be converted to JSON scalars.")
        return value.detach().cpu().item()
    return str(value)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_class_preserving_derangement(
    targets: torch.Tensor,
) -> tuple[torch.Tensor, Dict[int, Dict[str, Any]]]:
    """
    Build the frozen Step11A primary validation mismatch mapping.

    For each Stage-1 class independently:
      * collect indices in ascending validation order;
      * seed a CPU torch.Generator with 27110000 + class_id;
      * draw randperm repeatedly until a full derangement is obtained;
      * map each text row to exactly one different vision row in the same class.

    Returns:
      donor_indices: LongTensor [N], where donor_indices[i] = pi(i)
      class_metadata: deterministic construction metadata per class
    """
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor.")
    if targets.ndim != 1:
        raise ValueError(
            f"targets must be 1-D, got shape {tuple(targets.shape)}."
        )
    if targets.numel() == 0:
        raise ValueError("targets must contain at least one validation sample.")

    labels = targets.detach().cpu().to(torch.long)
    donor_indices = torch.full(
        (labels.numel(),),
        fill_value=-1,
        dtype=torch.long,
    )
    class_metadata: Dict[int, Dict[str, Any]] = {}

    unique_classes = sorted(int(x) for x in torch.unique(labels).tolist())

    for class_id in unique_classes:
        class_indices = torch.nonzero(
            labels == class_id,
            as_tuple=False,
        ).reshape(-1)
        class_indices, _ = torch.sort(class_indices)
        class_size = int(class_indices.numel())

        if class_size < 2:
            raise ValueError(
                "Step11A class-preserving derangement requires at least two "
                f"validation samples in class {class_id}; got {class_size}."
            )

        generator_seed = DERANGEMENT_SEED_BASE + class_id
        generator = torch.Generator(device="cpu")
        generator.manual_seed(generator_seed)
        positions = torch.arange(class_size, dtype=torch.long)

        attempts = 0
        while True:
            attempts += 1
            permutation = torch.randperm(
                class_size,
                generator=generator,
            )
            if bool(torch.all(permutation != positions)):
                break
            if attempts >= 1_000_000:
                raise RuntimeError(
                    "Failed to construct a full deterministic derangement for "
                    f"class {class_id} after {attempts} attempts."
                )

        donors = class_indices.index_select(0, permutation)
        donor_indices[class_indices] = donors

        class_metadata[class_id] = {
            "class_id": class_id,
            "class_size": class_size,
            "rng_engine": "torch.Generator(cpu)+torch.randperm",
            "rng_seed": generator_seed,
            "attempts_until_full_derangement": attempts,
        }

    validate_class_preserving_derangement(
        targets=labels,
        donor_indices=donor_indices,
    )
    return donor_indices, class_metadata


def validate_class_preserving_derangement(
    targets: torch.Tensor,
    donor_indices: torch.Tensor,
) -> None:
    if targets.ndim != 1 or donor_indices.ndim != 1:
        raise ValueError("targets and donor_indices must both be 1-D.")
    if targets.numel() != donor_indices.numel():
        raise ValueError(
            "targets and donor_indices must contain the same number of rows."
        )

    labels = targets.detach().cpu().to(torch.long)
    donors = donor_indices.detach().cpu().to(torch.long)
    n = int(labels.numel())

    if n == 0:
        raise ValueError("Cannot validate an empty derangement.")
    if bool(torch.any(donors < 0)) or bool(torch.any(donors >= n)):
        raise ValueError("Derangement contains out-of-range donor indices.")

    row_indices = torch.arange(n, dtype=torch.long)
    if bool(torch.any(donors == row_indices)):
        raise ValueError("Derangement contains one or more self-pairs.")

    donor_labels = labels.index_select(0, donors)
    if not torch.equal(donor_labels, labels):
        raise ValueError("Derangement changes one or more Stage-1 class labels.")

    for class_id in sorted(int(x) for x in torch.unique(labels).tolist()):
        receivers = torch.nonzero(
            labels == class_id,
            as_tuple=False,
        ).reshape(-1)
        class_donors = donors.index_select(0, receivers)
        expected = torch.sort(receivers).values
        observed = torch.sort(class_donors).values
        if not torch.equal(observed, expected):
            raise ValueError(
                "Each validation vision must be used exactly once within class; "
                f"class {class_id} violates this invariant."
            )


def _checkpoint_path(checkpoint_root: Path, seed: int) -> Path:
    return (
        checkpoint_root
        / f"v027_m4qc_seed{seed}"
        / "best_model.pt"
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen M4qc checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    required = {
        "epoch",
        "ablation_mode",
        "fusion_architecture",
        "alignment_model_state_dict",
        "configuration",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} is missing required keys: {sorted(missing)}"
        )
    return checkpoint


def _build_frozen_m4qc_model(
    checkpoint: Dict[str, Any],
    *,
    text_dim: int,
    vision_dim: int,
    expected_seed: int,
    device: torch.device,
) -> tuple[CrossModalAlignmentModel, Dict[str, Any]]:
    config = checkpoint["configuration"]

    _require(
        checkpoint["ablation_mode"] == "multimodal",
        f"Seed {expected_seed}: checkpoint ablation mode is not multimodal.",
    )
    _require(
        checkpoint["fusion_architecture"]
        == "quality_compatibility_supervised",
        f"Seed {expected_seed}: checkpoint is not M4qc.",
    )
    _require(
        int(config["seed"]) == expected_seed,
        f"Seed {expected_seed}: checkpoint configuration seed mismatch.",
    )
    _require(
        bool(config.get("quality_compatibility_supervised")) is True,
        f"Seed {expected_seed}: M4qc flag is not enabled.",
    )

    protocol = config.get("m4qc_compatibility_protocol")
    _require(
        isinstance(protocol, dict),
        f"Seed {expected_seed}: missing M4qc protocol metadata.",
    )
    _require(
        protocol.get("protocol_version") == PROTOCOL_VERSION,
        f"Seed {expected_seed}: protocol version mismatch.",
    )
    _require(
        protocol.get("diagnostic_only") is True,
        f"Seed {expected_seed}: compatibility is not diagnostic-only.",
    )
    _require(
        protocol.get("affects_primary_fusion") is False,
        f"Seed {expected_seed}: compatibility affects primary fusion.",
    )
    _require(
        protocol.get("official_test_accessed") is False,
        f"Seed {expected_seed}: checkpoint metadata reports official test access.",
    )

    model = CrossModalAlignmentModel(
        text_dim=text_dim,
        vision_dim=vision_dim,
        shared_dim=int(config["shared_dim"]),
        dropout=float(config["dropout"]),
        temperature=float(config["temperature"]),
        quality_compatibility_supervised=True,
    )
    model.load_state_dict(
        checkpoint["alignment_model_state_dict"],
        strict=True,
    )
    model.to(device)
    model.eval()

    _require(
        model.text_quality_estimator is not None,
        f"Seed {expected_seed}: text quality estimator unavailable.",
    )
    _require(
        model.vision_quality_estimator is not None,
        f"Seed {expected_seed}: vision quality estimator unavailable.",
    )
    _require(
        model.compatibility_estimator is not None,
        f"Seed {expected_seed}: compatibility estimator unavailable.",
    )

    return model, config


def _extract_scores(outputs: Dict[str, torch.Tensor], expected_rows: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    names = (
        "text_quality",
        "vision_quality",
        "compatibility_score",
    )
    for name in names:
        if name not in outputs or outputs[name] is None:
            raise RuntimeError(f"M4qc output {name!r} is unavailable.")

    text_q = outputs["text_quality"].detach().cpu().reshape(-1)
    vision_q = outputs["vision_quality"].detach().cpu().reshape(-1)
    compatibility = outputs["compatibility_score"].detach().cpu().reshape(-1)

    for name, tensor in (
        ("text_quality", text_q),
        ("vision_quality", vision_q),
        ("compatibility_score", compatibility),
    ):
        if tensor.numel() != expected_rows:
            raise RuntimeError(
                f"{name} produced {tensor.numel()} rows; expected {expected_rows}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{name} contains non-finite values.")
        if bool(torch.any(tensor < 0.0)) or bool(torch.any(tensor > 1.0)):
            raise RuntimeError(f"{name} contains values outside [0,1].")

    return text_q, vision_q, compatibility


def evaluate_seed_diagnostics(
    *,
    model: CrossModalAlignmentModel,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    donor_indices: torch.Tensor,
    sample_ids: Sequence[Any],
    targets: torch.Tensor,
    seed: int,
    checkpoint_epoch: int,
    batch_size: int,
    device: torch.device,
) -> list[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    n = int(targets.numel())
    _require(text_embeddings.shape[0] == n, "Text row count mismatch.")
    _require(vision_embeddings.shape[0] == n, "Vision row count mismatch.")
    _require(len(sample_ids) == n, "sample_ids row count mismatch.")

    validate_class_preserving_derangement(targets, donor_indices)

    matched_text_q = torch.empty(n, dtype=torch.float64)
    matched_vision_q = torch.empty(n, dtype=torch.float64)
    matched_c = torch.empty(n, dtype=torch.float64)
    mismatch_text_q = torch.empty(n, dtype=torch.float64)
    mismatch_vision_q = torch.empty(n, dtype=torch.float64)
    mismatch_c = torch.empty(n, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            rows = torch.arange(start, stop, dtype=torch.long)
            donors = donor_indices.index_select(0, rows)

            matched_outputs = model(
                text_embeddings.index_select(0, rows).to(device),
                vision_embeddings.index_select(0, rows).to(device),
                compute_loss=False,
            )
            mtq, mvq, mc = _extract_scores(
                matched_outputs,
                expected_rows=stop - start,
            )

            mismatch_outputs = model(
                text_embeddings.index_select(0, rows).to(device),
                vision_embeddings.index_select(0, donors).to(device),
                compute_loss=False,
            )
            xtq, xvq, xc = _extract_scores(
                mismatch_outputs,
                expected_rows=stop - start,
            )

            matched_text_q[start:stop] = mtq.to(torch.float64)
            matched_vision_q[start:stop] = mvq.to(torch.float64)
            matched_c[start:stop] = mc.to(torch.float64)
            mismatch_text_q[start:stop] = xtq.to(torch.float64)
            mismatch_vision_q[start:stop] = xvq.to(torch.float64)
            mismatch_c[start:stop] = xc.to(torch.float64)

    rows_out: list[Dict[str, Any]] = []
    labels = targets.detach().cpu().to(torch.long)

    for index in range(n):
        donor = int(donor_indices[index].item())
        class_id = int(labels[index].item())

        row = {
            "protocol_version": PROTOCOL_VERSION,
            "diagnostic_runner_version": RUNNER_VERSION,
            "seed": seed,
            "checkpoint_epoch": checkpoint_epoch,
            "validation_row_index": index,
            "stage1_label": class_id,
            "text_source_index": index,
            "text_source_sample_id": _json_scalar(sample_ids[index]),
            "matched": {
                "vision_source_index": index,
                "vision_source_sample_id": _json_scalar(sample_ids[index]),
                "text_quality": _finite_unit_interval(
                    matched_text_q[index].item(),
                    f"seed {seed} row {index} matched text_quality",
                ),
                "vision_quality": _finite_unit_interval(
                    matched_vision_q[index].item(),
                    f"seed {seed} row {index} matched vision_quality",
                ),
                "compatibility": _finite_unit_interval(
                    matched_c[index].item(),
                    f"seed {seed} row {index} matched compatibility",
                ),
                "compatibility_target": 1,
            },
            "mismatched": {
                "vision_source_index": donor,
                "vision_source_sample_id": _json_scalar(sample_ids[donor]),
                "vision_source_stage1_label": int(labels[donor].item()),
                "text_quality": _finite_unit_interval(
                    mismatch_text_q[index].item(),
                    f"seed {seed} row {index} mismatch text_quality",
                ),
                "vision_quality": _finite_unit_interval(
                    mismatch_vision_q[index].item(),
                    f"seed {seed} row {index} mismatch vision_quality",
                ),
                "compatibility": _finite_unit_interval(
                    mismatch_c[index].item(),
                    f"seed {seed} row {index} mismatch compatibility",
                ),
                "compatibility_target": 0,
            },
            "official_test_accessed": False,
        }
        rows_out.append(row)

    _require(len(rows_out) == n, f"Seed {seed}: diagnostic row count mismatch.")
    return rows_out


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. "
                "Use --overwrite only for an intentional rerun of the same "
                "frozen Step12B implementation."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if not args.validation_cache.is_dir():
        raise FileNotFoundError(
            f"Frozen validation cache does not exist: {args.validation_cache}"
        )

    if args.device is None:
        device = resolve_device()
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

    _prepare_output_root(args.output_root, args.overwrite)

    print("=" * 78)
    print("AEGIS v0.27 STEP12B M4qc DETERMINISTIC COMPATIBILITY DIAGNOSTICS")
    print("=" * 78)
    print("Protocol:", PROTOCOL_VERSION)
    print("Runner:", RUNNER_VERSION)
    print("Validation cache:", args.validation_cache.resolve())
    print("Checkpoint root:", args.checkpoint_root.resolve())
    print("Output root:", args.output_root.resolve())
    print("Seeds:", FORMAL_SEEDS)
    print("Device:", device)
    print("Official test split: SEALED / NOT ACCESSED")
    print("Formal H4-C/H5-C decisions: NOT COMPUTED IN STEP12B")
    print()

    validation_cache = FakedditRepresentationCache(args.validation_cache)
    validation_data = load_cache_as_single_batch(validation_cache)

    n = int(validation_data.batch_size)
    _require(
        n == EXPECTED_VALIDATION_SAMPLES,
        f"Expected {EXPECTED_VALIDATION_SAMPLES} validation samples; got {n}.",
    )

    targets = validation_data.integrity_targets.detach().cpu().to(torch.long)
    sample_ids = list(validation_data.sample_ids)
    _require(len(sample_ids) == n, "Validation sample_ids length mismatch.")
    _require(
        len(set(map(str, sample_ids))) == n,
        "Validation sample_ids must be unique for Step12B identity tracking.",
    )

    donor_indices, class_metadata = build_class_preserving_derangement(targets)
    validate_class_preserving_derangement(targets, donor_indices)

    mapping_rows = []
    for index in range(n):
        donor = int(donor_indices[index].item())
        mapping_rows.append(
            {
                "validation_row_index": index,
                "text_sample_id": _json_scalar(sample_ids[index]),
                "stage1_label": int(targets[index].item()),
                "donor_vision_index": donor,
                "donor_vision_sample_id": _json_scalar(sample_ids[donor]),
                "donor_stage1_label": int(targets[donor].item()),
                "self_pair": bool(index == donor),
            }
        )

    mapping_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_runner_version": RUNNER_VERSION,
        "construction": "deterministic_class_preserving_derangement",
        "rng_implementation": "torch.Generator(cpu)+torch.randperm until full derangement",
        "seed_rule": "27110000 + class_id",
        "validation_sample_count": n,
        "class_counts": {
            str(k): int(v)
            for k, v in sorted(Counter(targets.tolist()).items())
        },
        "class_metadata": {
            str(k): v
            for k, v in sorted(class_metadata.items())
        },
        "invariants": {
            "no_self_pairs": True,
            "class_preserving": True,
            "one_mismatch_partner_per_text": True,
            "each_vision_used_exactly_once_within_class": True,
            "mapping_shared_across_model_seeds": True,
            "mapping_independent_of_training_rng": True,
        },
        "official_test_accessed": False,
        "mapping": mapping_rows,
    }
    _save_json(
        args.output_root / "validation_derangement_mapping.json",
        mapping_payload,
    )

    seed_summaries: list[Dict[str, Any]] = []

    for seed in FORMAL_SEEDS:
        checkpoint_path = _checkpoint_path(args.checkpoint_root, seed)
        checkpoint = _load_checkpoint(checkpoint_path, device)
        model, config = _build_frozen_m4qc_model(
            checkpoint,
            text_dim=int(validation_cache.text_dimension),
            vision_dim=int(validation_cache.vision_dimension),
            expected_seed=seed,
            device=device,
        )

        print(
            f"Seed {seed}: checkpoint epoch {int(checkpoint['epoch'])} -> "
            "running matched + class-preserving mismatched diagnostics"
        )

        diagnostic_rows = evaluate_seed_diagnostics(
            model=model,
            text_embeddings=validation_data.text_embeddings,
            vision_embeddings=validation_data.vision_embeddings,
            donor_indices=donor_indices,
            sample_ids=sample_ids,
            targets=targets,
            seed=seed,
            checkpoint_epoch=int(checkpoint["epoch"]),
            batch_size=args.batch_size,
            device=device,
        )

        output_path = args.output_root / f"seed{seed}_diagnostics.jsonl"
        written = _write_jsonl(output_path, diagnostic_rows)
        _require(
            written == EXPECTED_VALIDATION_SAMPLES,
            f"Seed {seed}: wrote {written} rows, expected 1000.",
        )

        seed_summary = {
            "seed": seed,
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "diagnostic_rows": written,
            "matched_compatibility_observations": written,
            "mismatched_compatibility_observations": written,
            "total_compatibility_observations": written * 2,
            "quality_pairs_recorded": {
                "matched_text": written,
                "matched_vision": written,
                "mismatched_text": written,
                "mismatched_vision": written,
            },
            "checkpoint_selection_metric": "validation_macro_f1",
            "checkpoint_reselected": False,
            "training_performed": False,
            "formal_h4_computed": False,
            "formal_h5_computed": False,
            "official_test_accessed": False,
            "diagnostic_file": output_path.name,
            "configuration_seed": int(config["seed"]),
        }
        seed_summaries.append(seed_summary)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_runner_version": RUNNER_VERSION,
        "stage": "Step12B",
        "status": "DIAGNOSTICS_GENERATED_FORMAL_ANALYSIS_PENDING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_cache": str(args.validation_cache).replace("\\", "/"),
        "validation_sample_count": n,
        "formal_seeds": list(FORMAL_SEEDS),
        "mapping_artifact": "validation_derangement_mapping.json",
        "mapping_identical_across_seeds": True,
        "matched_observations_per_seed": n,
        "mismatched_observations_per_seed": n,
        "total_compatibility_observations_per_seed": 2 * n,
        "seed_summaries": seed_summaries,
        "analysis_boundary": {
            "roc_auc_computed": False,
            "h4_c_decision_computed": False,
            "h5_c_decision_computed": False,
            "classification_preservation_recomputed": False,
            "threshold_tuning_performed": False,
            "checkpoint_reselection_performed": False,
            "training_performed": False,
            "m4qcf_computed": False,
        },
        "official_test_split_accessed": False,
        "next_stage": "Step12C formal H4-C/H5-C analysis",
    }
    _save_json(args.output_root / "manifest.json", manifest)

    summary = {
        "stage": "Step12B",
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_runner_version": RUNNER_VERSION,
        "validation_samples": n,
        "class_counts": {
            str(k): int(v)
            for k, v in sorted(Counter(targets.tolist()).items())
        },
        "seeds": list(FORMAL_SEEDS),
        "diagnostic_rows_per_seed": n,
        "compatibility_observations_per_seed": 2 * n,
        "mapping_invariants_passed": True,
        "formal_h4_computed": False,
        "formal_h5_computed": False,
        "official_test_accessed": False,
        "seed_summaries": seed_summaries,
    }
    _save_json(args.output_root / "summary.json", summary)

    print()
    print("=" * 78)
    print("STEP12B COMPLETE")
    print("=" * 78)
    print("Validation samples:", n)
    print("Class counts:", dict(sorted(Counter(targets.tolist()).items())))
    print("Matched observations per seed:", n)
    print("Mismatched observations per seed:", n)
    print("Total compatibility observations per seed:", 2 * n)
    print("Mapping invariants: PASS")
    print("Formal H4-C: NOT COMPUTED")
    print("Formal H5-C: NOT COMPUTED")
    print("Official test accessed: FALSE")
    print("Artifacts:")
    print(" -", args.output_root / "manifest.json")
    print(" -", args.output_root / "validation_derangement_mapping.json")
    for seed in FORMAL_SEEDS:
        print(" -", args.output_root / f"seed{seed}_diagnostics.jsonl")
    print(" -", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
