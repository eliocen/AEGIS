"""
AEGIS v0.27 - Fakeddit M4q Intrinsic-Quality Diagnostics

Purpose
-------
Evaluate whether explicitly supervised M4q modality-quality scores respond
meaningfully to controlled single-modality degradation on the fixed Fakeddit
validation representation cache.

This runner is diagnostic only. It:
- loads only frozen M4q best checkpoints;
- performs no training and creates no optimizer;
- never accesses the official Fakeddit test split;
- computes Gaussian feature scale from the frozen TRAINING cache only;
- uses validation corruption realizations that are independent of model seed;
- reuses aegis.reliability.corruption rather than reimplementing corruption;
- fingerprints the alignment model before and after evaluation.

Scientific scope
----------------
Step 9 produces the raw/derived quality diagnostics needed for later Step 10
analysis of H1-Q, H2-Q, and H3-Q. It does not establish factual verification,
source credibility, real-world calibration, or raw-input corruption
generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from aegis.alignment import CrossModalAlignmentModel
from aegis.data.cache import FakedditRepresentationCache
from aegis.reliability.corruption import (
    compute_feature_std,
    corrupt_representation,
)
from scripts.run_fakeddit_generalization import (
    load_cache_as_single_batch,
    resolve_device,
)


VERSION = "0.27.4-dev"
EXPERIMENT_NAME = "fakeddit_m4q_quality_diagnostics_v027"

DEFAULT_MODEL_SEEDS = (42, 43, 44)
DEFAULT_TRAIN_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/train_n5000_seed42"
)
DEFAULT_VALIDATION_CACHE = Path(
    "data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42"
)
DEFAULT_EXPERIMENTS_ROOT = Path("experiments/fakeddit")
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v027_m4q_quality_diagnostics"
)

EXPECTED_TRAIN_SAMPLES = 5000
EXPECTED_VALIDATION_SAMPLES = 1000

CONTINUOUS_CORRUPTIONS = ("gaussian_noise", "attenuation")
DISCRETE_CORRUPTIONS = ("zero_dropout",)
MODALITIES = ("text", "vision")
NONZERO_SEVERITIES = (0.25, 0.50, 0.75, 1.0)

# The frozen v0.27 protocol requires validation corruption randomness to be
# independent of model-training seed. The exact formula was not preregistered;
# this deterministic namespace is therefore an implementation detail recorded
# in every Step 9 manifest.
VALIDATION_CORRUPTION_BASE_SEED = 27_100_000

CORRUPTION_IDS = {
    "gaussian_noise": 1,
    "attenuation": 2,
    "zero_dropout": 3,
}

MODALITY_IDS = {
    "text": 1,
    "vision": 2,
}


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.27 M4q intrinsic-quality diagnostics on the fixed "
            "Fakeddit validation representation cache."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_MODEL_SEEDS),
        help="Frozen M4q model seeds. Default: 42 43 44.",
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
        help=(
            "Frozen training representation cache. Used ONLY to compute "
            "Gaussian per-feature population standard deviations."
        ),
    )
    parser.add_argument(
        "--validation-cache",
        type=Path,
        default=DEFAULT_VALIDATION_CACHE,
        help="Frozen validation representation cache.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help=(
            "Root containing v027_m4q_seed{seed}/best_model.pt directories."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Step 9 diagnostic output directory.",
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
        "--expected-train-samples",
        type=int,
        default=EXPECTED_TRAIN_SAMPLES,
        help="Hard expected training-cache sample count. Default: 5000.",
    )
    parser.add_argument(
        "--expected-validation-samples",
        type=int,
        default=EXPECTED_VALIDATION_SAMPLES,
        help="Hard expected validation-cache sample count. Default: 1000.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Evaluate seed42 only on clean plus text Gaussian severity .25. "
            "The smoke run remains validation-only and test-sealed."
        ),
    )

    args = parser.parse_args()

    if not args.seeds:
        raise ValueError("--seeds must contain at least one model seed.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("All model seeds must be >= 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.expected_train_samples <= 0:
        raise ValueError("--expected-train-samples must be > 0.")
    if args.expected_validation_samples <= 0:
        raise ValueError("--expected-validation-samples must be > 0.")

    if args.smoke:
        args.seeds = [42]

    return args


# ============================================================================
# Generic helpers
# ============================================================================


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assert_safe_input_path(path: Path, purpose: str) -> None:
    """
    Hard guard against accidentally pointing Step 9 at an official test path.

    This intentionally checks path components rather than arbitrary substrings
    so names such as "stress_test" in historical source files do not matter.
    """
    forbidden = {"test", "testing", "official_test", "test_split"}
    normalized = {part.lower() for part in path.parts}
    intersection = normalized & forbidden
    if intersection:
        raise RuntimeError(
            f"Unsafe {purpose} path references a forbidden test component "
            f"{sorted(intersection)}: {path}"
        )


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def module_state_fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def population_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("population_stats requires at least one value.")
    numeric = [float(v) for v in values]
    return {
        "mean": float(mean(numeric)),
        "std_population": float(pstdev(numeric)),
        "min": float(min(numeric)),
        "max": float(max(numeric)),
    }


def resolve_runtime_device(requested: Optional[str]) -> torch.device:
    """
    Resolve the Step 9 inference device.

    The historical scripts.run_fakeddit_generalization.resolve_device helper
    takes no arguments, so it is used only for the automatic/default path.
    Explicit --device values are handled locally.
    """
    if requested is None:
        return torch.device(resolve_device())

    value = str(requested).strip()
    if not value:
        raise ValueError("--device must not be empty.")

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested but CUDA is not available."
        )

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported Step 9 device {value!r}; expected cpu or cuda."
        )

    return device


def get_sample_ids(full_batch: Any, expected: int) -> List[str]:
    raw = getattr(full_batch, "sample_ids", None)
    if raw is None:
        return [str(index) for index in range(expected)]

    if isinstance(raw, torch.Tensor):
        values = raw.detach().cpu().reshape(-1).tolist()
    else:
        values = list(raw)

    if len(values) != expected:
        raise RuntimeError(
            "sample_ids length does not match validation sample count: "
            f"{len(values)} != {expected}"
        )
    return [str(value) for value in values]


def load_representation_cache(path: Path) -> Tuple[Any, Any]:
    """
    Load a Fakeddit cache and collapse it into the repository-standard
    single-batch representation.

    Current AEGIS cache code exposes FakedditRepresentationCache.load(...).
    The constructor fallback keeps this helper compatible with older local
    cache revisions without changing scientific semantics.
    """
    assert_safe_input_path(path, "representation cache")
    if not path.exists():
        raise FileNotFoundError(f"Representation cache not found: {path}")

    loader = getattr(FakedditRepresentationCache, "load", None)
    if callable(loader):
        cache = loader(path)
    else:
        cache = FakedditRepresentationCache(path)

    batch = load_cache_as_single_batch(cache)
    return cache, batch


# ============================================================================
# Frozen Step 9 corruption protocol
# ============================================================================


def validation_corruption_seed(
    corruption_type: str,
    modality: str,
    severity: Optional[float],
) -> int:
    """
    Deterministic validation corruption seed independent of model seed.

    No model/training seed is accepted by this function by design.
    """
    if corruption_type not in CORRUPTION_IDS:
        raise ValueError(f"Unsupported corruption_type: {corruption_type!r}")
    if modality not in MODALITY_IDS:
        raise ValueError(f"Unsupported modality: {modality!r}")

    if severity is None:
        severity_code = 0
    else:
        severity_value = float(severity)
        if not 0.0 <= severity_value <= 1.0:
            raise ValueError("severity must be in [0, 1].")
        severity_code = int(round(severity_value * 1000.0))

    return (
        VALIDATION_CORRUPTION_BASE_SEED
        + 10_000 * CORRUPTION_IDS[corruption_type]
        + 1_000 * MODALITY_IDS[modality]
        + severity_code
    )


def build_conditions(smoke: bool = False) -> List[Dict[str, Any]]:
    clean = {
        "corruption_type": "clean",
        "corrupted_modality": "none",
        "severity": None,
        "corruption_seed": None,
    }

    if smoke:
        return [
            clean,
            {
                "corruption_type": "gaussian_noise",
                "corrupted_modality": "text",
                "severity": 0.25,
                "corruption_seed": validation_corruption_seed(
                    "gaussian_noise", "text", 0.25
                ),
            },
        ]

    conditions: List[Dict[str, Any]] = [clean]

    for corruption_type in CONTINUOUS_CORRUPTIONS:
        for modality in MODALITIES:
            for severity in NONZERO_SEVERITIES:
                conditions.append(
                    {
                        "corruption_type": corruption_type,
                        "corrupted_modality": modality,
                        "severity": float(severity),
                        "corruption_seed": validation_corruption_seed(
                            corruption_type,
                            modality,
                            float(severity),
                        ),
                    }
                )

    for modality in MODALITIES:
        conditions.append(
            {
                "corruption_type": "zero_dropout",
                "corrupted_modality": modality,
                "severity": 1.0,
                "corruption_seed": validation_corruption_seed(
                    "zero_dropout",
                    modality,
                    1.0,
                ),
            }
        )

    return conditions


def condition_slug(condition: Mapping[str, Any]) -> str:
    corruption = str(condition["corruption_type"])
    modality = str(condition["corrupted_modality"])
    severity = condition.get("severity")

    if corruption == "clean":
        return "clean"
    if severity is None:
        return f"{modality}_{corruption}"

    rendered = (
        f"{float(severity):.2f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )
    return f"{modality}_{corruption}_s{rendered}"


def apply_condition(
    *,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    text_feature_std: torch.Tensor,
    vision_feature_std: torch.Tensor,
    condition: Mapping[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply one frozen Step 9 condition at encoder-representation level.
    """
    corruption_type = str(condition["corruption_type"])
    modality = str(condition["corrupted_modality"])
    severity = condition.get("severity")
    cseed = condition.get("corruption_seed")

    if corruption_type == "clean":
        return text_embeddings.clone(), vision_embeddings.clone()

    if modality == "text":
        kwargs: Dict[str, Any] = {
            "family": corruption_type,
            "severity": severity,
            "seed": cseed,
        }
        if corruption_type == "gaussian_noise":
            kwargs["feature_std"] = text_feature_std

        corrupted = corrupt_representation(
            text_embeddings,
            **kwargs,
        ).corrupted
        return corrupted, vision_embeddings.clone()

    if modality == "vision":
        kwargs = {
            "family": corruption_type,
            "severity": severity,
            "seed": cseed,
        }
        if corruption_type == "gaussian_noise":
            kwargs["feature_std"] = vision_feature_std

        corrupted = corrupt_representation(
            vision_embeddings,
            **kwargs,
        ).corrupted
        return text_embeddings.clone(), corrupted

    raise RuntimeError(
        f"Corrupted condition has invalid modality {modality!r}."
    )


def precompute_validation_conditions(
    *,
    text_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    text_feature_std: torch.Tensor,
    vision_feature_std: torch.Tensor,
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Precompute each validation corruption exactly once.

    All model seeds subsequently consume these same tensors, which makes the
    "identical corruption across model seeds" invariant stronger than merely
    reusing the same seed formula.
    """
    realized: Dict[str, Dict[str, Any]] = {}

    for condition in conditions:
        slug = condition_slug(condition)
        if slug in realized:
            raise RuntimeError(f"Duplicate condition slug: {slug}")

        text_value, vision_value = apply_condition(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            text_feature_std=text_feature_std,
            vision_feature_std=vision_feature_std,
            condition=condition,
        )

        realized[slug] = {
            "condition": dict(condition),
            "text_embeddings": text_value,
            "vision_embeddings": vision_value,
            "text_sha256": tensor_sha256(text_value),
            "vision_sha256": tensor_sha256(vision_value),
        }

    return realized


# ============================================================================
# M4q checkpoint reconstruction
# ============================================================================


def validate_m4q_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must be a dictionary-like mapping.")

    if checkpoint.get("ablation_mode") != "multimodal":
        raise RuntimeError(
            "Step 9 requires ablation_mode='multimodal'; found "
            f"{checkpoint.get('ablation_mode')!r}."
        )

    if checkpoint.get("fusion_architecture") != "quality_supervised":
        raise RuntimeError(
            "Step 9 requires fusion_architecture='quality_supervised'; found "
            f"{checkpoint.get('fusion_architecture')!r}."
        )

    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("Checkpoint configuration is not a dictionary.")

    if configuration.get("quality_supervised") is not True:
        raise RuntimeError(
            "Checkpoint configuration does not confirm quality_supervised=True."
        )

    if configuration.get("fusion_architecture") not in (
        None,
        "quality_supervised",
    ):
        raise RuntimeError(
            "Checkpoint configuration fusion_architecture is inconsistent."
        )

    if expected_seed is not None:
        recorded_seed = configuration.get("seed")
        if recorded_seed is not None and int(recorded_seed) != int(expected_seed):
            raise RuntimeError(
                "Checkpoint seed mismatch: expected "
                f"{expected_seed}, found {recorded_seed}."
            )

    if checkpoint.get("alignment_model_state_dict") is None:
        raise RuntimeError(
            "Checkpoint is missing alignment_model_state_dict."
        )

    return dict(configuration)


def _filtered_constructor_kwargs(
    cls: type,
    candidates: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Filter candidate kwargs against the installed constructor signature.

    This avoids hard-coding historical optional flags while still refusing to
    guess required arguments that are absent from the checkpoint/cache.
    """
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for name, param in parameters.items()
        if name != "self"
    ):
        return dict(candidates)

    return {
        name: value
        for name, value in candidates.items()
        if name in parameters
    }


def build_m4q_alignment_from_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    text_dim: int,
    vision_dim: int,
    device: torch.device,
    expected_seed: Optional[int] = None,
) -> Tuple[CrossModalAlignmentModel, Dict[str, Any]]:
    configuration = validate_m4q_checkpoint_metadata(
        checkpoint,
        expected_seed=expected_seed,
    )

    shared_dim = int(configuration.get("shared_dim", 512))
    dropout = float(configuration.get("dropout", 0.1))
    temperature = float(configuration.get("temperature", 0.07))

    candidates = {
        "text_dim": int(text_dim),
        "vision_dim": int(vision_dim),
        "shared_dim": shared_dim,
        "temperature": temperature,
        "contrastive_temperature": temperature,
        "dropout": dropout,
        "evidence_aware": False,
        "interaction_only": False,
        "gated_interaction": False,
        "reliability_only": False,
        "interaction_reliability": False,
        "quality_supervised": True,
    }

    kwargs = _filtered_constructor_kwargs(
        CrossModalAlignmentModel,
        candidates,
    )

    model = CrossModalAlignmentModel(**kwargs).to(device)

    if not bool(getattr(model, "quality_supervised", False)):
        raise RuntimeError(
            "Reconstructed alignment model is not quality_supervised."
        )

    if getattr(model, "text_quality_estimator", None) is None:
        raise RuntimeError(
            "Reconstructed M4q model has no text_quality_estimator."
        )
    if getattr(model, "vision_quality_estimator", None) is None:
        raise RuntimeError(
            "Reconstructed M4q model has no vision_quality_estimator."
        )

    state = checkpoint["alignment_model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    resolved = {
        "text_dim": int(text_dim),
        "vision_dim": int(vision_dim),
        "shared_dim": shared_dim,
        "dropout": dropout,
        "temperature": temperature,
        "constructor_kwargs": kwargs,
        "fusion_architecture": "quality_supervised",
        "quality_supervised": True,
    }
    return model, resolved


# ============================================================================
# Quality inference
# ============================================================================


def evaluate_quality_condition(
    *,
    model_seed: int,
    checkpoint_path: Path,
    alignment_model: CrossModalAlignmentModel,
    sample_ids: Sequence[str],
    condition_record: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    condition = condition_record["condition"]
    text_embeddings = condition_record["text_embeddings"]
    vision_embeddings = condition_record["vision_embeddings"]

    if text_embeddings.shape[0] != vision_embeddings.shape[0]:
        raise RuntimeError("Text/vision condition tensors have unequal size.")

    n = int(text_embeddings.shape[0])
    if len(sample_ids) != n:
        raise RuntimeError(
            f"sample_ids length mismatch: {len(sample_ids)} != {n}"
        )

    text_quality_values: List[float] = []
    vision_quality_values: List[float] = []
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            text_batch = text_embeddings[start:end].to(device)
            vision_batch = vision_embeddings[start:end].to(device)

            outputs = alignment_model(
                text_batch,
                vision_batch,
                compute_loss=False,
            )

            text_quality = outputs.get("text_quality")
            vision_quality = outputs.get("vision_quality")

            if text_quality is None or vision_quality is None:
                raise RuntimeError(
                    "M4q forward output is missing text_quality or vision_quality."
                )

            text_quality = (
                text_quality.detach().cpu().reshape(-1).float()
            )
            vision_quality = (
                vision_quality.detach().cpu().reshape(-1).float()
            )

            if text_quality.numel() != end - start:
                raise RuntimeError("Unexpected text_quality output shape.")
            if vision_quality.numel() != end - start:
                raise RuntimeError("Unexpected vision_quality output shape.")

            for local_index in range(end - start):
                global_index = start + local_index
                tq = float(text_quality[local_index].item())
                vq = float(vision_quality[local_index].item())

                if not math.isfinite(tq) or not math.isfinite(vq):
                    raise RuntimeError("Non-finite quality prediction encountered.")
                if not 0.0 <= tq <= 1.0:
                    raise RuntimeError(
                        f"text_quality outside [0,1]: {tq}"
                    )
                if not 0.0 <= vq <= 1.0:
                    raise RuntimeError(
                        f"vision_quality outside [0,1]: {vq}"
                    )

                text_quality_values.append(tq)
                vision_quality_values.append(vq)

                rows.append(
                    {
                        "aegis_version": VERSION,
                        "model_variant": "M4q",
                        "model_seed": int(model_seed),
                        "checkpoint_path": str(checkpoint_path),
                        "corruption_type": condition["corruption_type"],
                        "corrupted_modality": condition["corrupted_modality"],
                        "severity": condition["severity"],
                        "corruption_seed": condition["corruption_seed"],
                        "row_index": global_index,
                        "sample_id": str(sample_ids[global_index]),
                        "text_quality": tq,
                        "vision_quality": vq,
                        "test_split_accessed": False,
                    }
                )

    text_stats = population_stats(text_quality_values)
    vision_stats = population_stats(vision_quality_values)

    summary: Dict[str, Any] = {
        "aegis_version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "model_variant": "M4q",
        "fusion_architecture": "quality_supervised",
        "model_seed": int(model_seed),
        "checkpoint_path": str(checkpoint_path),
        "corruption_type": condition["corruption_type"],
        "corrupted_modality": condition["corrupted_modality"],
        "severity": condition["severity"],
        "corruption_seed": condition["corruption_seed"],
        "sample_count": n,
        "text_quality_mean": text_stats["mean"],
        "text_quality_std_population": text_stats["std_population"],
        "text_quality_min": text_stats["min"],
        "text_quality_max": text_stats["max"],
        "vision_quality_mean": vision_stats["mean"],
        "vision_quality_std_population": vision_stats["std_population"],
        "vision_quality_min": vision_stats["min"],
        "vision_quality_max": vision_stats["max"],
        "text_corruption_tensor_sha256": condition_record["text_sha256"],
        "vision_corruption_tensor_sha256": condition_record["vision_sha256"],
        "test_split_accessed": False,
    }
    return rows, summary


def add_clean_relative_quality_deltas(
    summaries: List[Dict[str, Any]],
) -> None:
    clean_rows = [
        summary
        for summary in summaries
        if summary["corruption_type"] == "clean"
    ]
    if len(clean_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one clean summary, found {len(clean_rows)}."
        )

    clean = clean_rows[0]
    clean_text = float(clean["text_quality_mean"])
    clean_vision = float(clean["vision_quality_mean"])

    for summary in summaries:
        current_text = float(summary["text_quality_mean"])
        current_vision = float(summary["vision_quality_mean"])

        delta_text = current_text - clean_text
        delta_vision = current_vision - clean_vision

        summary["clean_text_quality_mean"] = clean_text
        summary["clean_vision_quality_mean"] = clean_vision
        summary["delta_text_quality_from_clean"] = delta_text
        summary["delta_vision_quality_from_clean"] = delta_vision

        modality = summary["corrupted_modality"]

        if modality == "text":
            corrupted_quality = current_text
            intact_quality = current_vision
            corrupted_delta = delta_text
            intact_delta = delta_vision
        elif modality == "vision":
            corrupted_quality = current_vision
            intact_quality = current_text
            corrupted_delta = delta_vision
            intact_delta = delta_text
        else:
            corrupted_quality = None
            intact_quality = None
            corrupted_delta = None
            intact_delta = None

        summary["corrupted_modality_quality"] = corrupted_quality
        summary["intact_modality_quality"] = intact_quality
        summary["corrupted_modality_delta_from_clean"] = corrupted_delta
        summary["intact_modality_delta_from_clean"] = intact_delta

        # Positive means the corrupted modality decreased more than the intact
        # modality: intact_delta - corrupted_delta > 0.
        summary["quality_selectivity"] = (
            None
            if corrupted_delta is None or intact_delta is None
            else float(intact_delta) - float(corrupted_delta)
        )


# ============================================================================
# Main
# ============================================================================


def checkpoint_path_for_seed(
    experiments_root: Path,
    seed: int,
) -> Path:
    return (
        experiments_root
        / f"v027_m4q_seed{int(seed)}"
        / "best_model.pt"
    )


def main() -> None:
    args = parse_args()
    device = resolve_runtime_device(args.device)

    assert_safe_input_path(args.train_cache, "training cache")
    assert_safe_input_path(args.validation_cache, "validation cache")
    assert_safe_input_path(args.experiments_root, "experiments root")

    train_cache, train_batch = load_representation_cache(args.train_cache)
    validation_cache, validation_batch = load_representation_cache(
        args.validation_cache
    )

    train_n = int(train_batch.text_embeddings.shape[0])
    validation_n = int(validation_batch.text_embeddings.shape[0])

    if train_n != args.expected_train_samples:
        raise RuntimeError(
            "Training cache sample count mismatch: "
            f"{train_n} != {args.expected_train_samples}"
        )
    if validation_n != args.expected_validation_samples:
        raise RuntimeError(
            "Validation cache sample count mismatch: "
            f"{validation_n} != {args.expected_validation_samples}"
        )

    if int(train_batch.vision_embeddings.shape[0]) != train_n:
        raise RuntimeError("Training text/vision sample counts differ.")
    if int(validation_batch.vision_embeddings.shape[0]) != validation_n:
        raise RuntimeError("Validation text/vision sample counts differ.")

    text_dim = int(validation_batch.text_embeddings.shape[1])
    vision_dim = int(validation_batch.vision_embeddings.shape[1])

    if int(train_batch.text_embeddings.shape[1]) != text_dim:
        raise RuntimeError("Training/validation text dimensions differ.")
    if int(train_batch.vision_embeddings.shape[1]) != vision_dim:
        raise RuntimeError("Training/validation vision dimensions differ.")

    # Frozen v0.27 rule: Gaussian scale comes from TRAINING cache only.
    text_feature_std = compute_feature_std(
        train_batch.text_embeddings,
        unbiased=False,
    )
    vision_feature_std = compute_feature_std(
        train_batch.vision_embeddings,
        unbiased=False,
    )

    sample_ids = get_sample_ids(validation_batch, validation_n)
    conditions = build_conditions(smoke=args.smoke)

    realized_conditions = precompute_validation_conditions(
        text_embeddings=validation_batch.text_embeddings,
        vision_embeddings=validation_batch.vision_embeddings,
        text_feature_std=text_feature_std,
        vision_feature_std=vision_feature_std,
        conditions=conditions,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    protocol_manifest = {
        "aegis_version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "scientific_status": (
            "validation-only diagnostic; not independent test generalization"
        ),
        "model_variant": "M4q",
        "fusion_architecture": "quality_supervised",
        "model_seeds": [int(seed) for seed in args.seeds],
        "training_cache": str(args.train_cache),
        "validation_cache": str(args.validation_cache),
        "training_samples": train_n,
        "validation_samples": validation_n,
        "text_dimension": text_dim,
        "vision_dimension": vision_dim,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "conditions_per_checkpoint": len(conditions),
        "number_of_condition_evaluations": (
            len(conditions) * len(args.seeds)
        ),
        "continuous_curve_zero_point": (
            "clean condition reused as severity 0 for Gaussian and attenuation"
        ),
        "gaussian_scaling": (
            "training-cache per-feature population standard deviation"
        ),
        "text_feature_std_sha256": tensor_sha256(text_feature_std),
        "vision_feature_std_sha256": tensor_sha256(vision_feature_std),
        "validation_corruption_seed_independent_of_model_seed": True,
        "validation_corruption_base_seed": VALIDATION_CORRUPTION_BASE_SEED,
        "validation_corruption_seed_formula": (
            "base + 10000*corruption_id + 1000*modality_id + severity_code"
        ),
        "corruption_realizations_precomputed_once_and_reused_across_models": True,
        "conditions": conditions,
        "hypotheses_deferred_to_step10": ["H1_Q", "H2_Q", "H3_Q", "H6_CLS"],
        "permutation_quality_diagnostics_in_step9": False,
        "compatibility_diagnostics_in_step9": False,
        "official_test_split_accessed": False,
    }
    save_json(args.output_root / "protocol_manifest.json", protocol_manifest)

    print("=" * 78)
    print("AEGIS v0.27 STEP 9 - M4q INTRINSIC-QUALITY DIAGNOSTICS")
    print("=" * 78)
    print("Device:", device)
    print("Training samples:", train_n)
    print("Validation samples:", validation_n)
    print("Model seeds:", args.seeds)
    print("Conditions per checkpoint:", len(conditions))
    print(
        "Total condition evaluations:",
        len(conditions) * len(args.seeds),
    )
    print("Gaussian scale: TRAINING-cache population std")
    print("Validation corruption seed depends on model seed: NO")
    print("Official test split: SEALED / NOT ACCESSED")
    print()

    all_summaries: List[Dict[str, Any]] = []

    for seed in args.seeds:
        checkpoint_path = checkpoint_path_for_seed(
            args.experiments_root,
            seed,
        )
        assert_safe_input_path(
            checkpoint_path,
            f"M4q seed {seed} checkpoint",
        )

        if checkpoint_path.name != "best_model.pt":
            raise RuntimeError(
                "Step 9 is restricted to best_model.pt checkpoints."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        print(f"Loading M4q seed {seed}: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        alignment_model, resolved = build_m4q_alignment_from_checkpoint(
            checkpoint=checkpoint,
            text_dim=text_dim,
            vision_dim=vision_dim,
            device=device,
            expected_seed=seed,
        )

        before_fingerprint = module_state_fingerprint(alignment_model)
        seed_dir = args.output_root / f"seed{seed}"
        seed_summaries: List[Dict[str, Any]] = []

        for slug, condition_record in realized_conditions.items():
            rows, summary = evaluate_quality_condition(
                model_seed=seed,
                checkpoint_path=checkpoint_path,
                alignment_model=alignment_model,
                sample_ids=sample_ids,
                condition_record=condition_record,
                batch_size=args.batch_size,
                device=device,
            )

            summary["checkpoint_epoch"] = checkpoint.get("epoch")
            summary["checkpoint_validation_metrics"] = checkpoint.get(
                "validation_metrics"
            )
            summary["resolved_model_configuration"] = resolved

            save_csv(seed_dir / f"{slug}_quality_samples.csv", rows)
            seed_summaries.append(summary)

            print(
                f"  {slug:34s} "
                f"qT={summary['text_quality_mean']:.4f} "
                f"qV={summary['vision_quality_mean']:.4f}"
            )

        after_fingerprint = module_state_fingerprint(alignment_model)
        if before_fingerprint != after_fingerprint:
            raise RuntimeError(
                f"M4q seed {seed} parameters changed during diagnostics."
            )

        add_clean_relative_quality_deltas(seed_summaries)

        for summary in seed_summaries:
            summary["model_state_fingerprint_before"] = before_fingerprint
            summary["model_state_fingerprint_after"] = after_fingerprint
            summary["parameters_unchanged"] = True

            save_json(
                seed_dir
                / f"{condition_slug(summary)}_quality_summary.json",
                summary,
            )

        save_csv(
            seed_dir / "condition_summary.csv",
            seed_summaries,
        )
        save_json(
            seed_dir / "run_manifest.json",
            {
                "aegis_version": VERSION,
                "experiment": EXPERIMENT_NAME,
                "model_variant": "M4q",
                "seed": int(seed),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "resolved_model_configuration": resolved,
                "conditions_evaluated": len(seed_summaries),
                "model_state_fingerprint_before": before_fingerprint,
                "model_state_fingerprint_after": after_fingerprint,
                "parameters_unchanged": True,
                "validation_corruption_seed_independent_of_model_seed": True,
                "official_test_split_accessed": False,
            },
        )

        all_summaries.extend(seed_summaries)
        print()

    aggregate_dir = args.output_root / "aggregate"
    save_csv(
        aggregate_dir / "all_condition_summaries.csv",
        all_summaries,
    )
    save_json(
        aggregate_dir / "all_condition_summaries.json",
        {
            "aegis_version": VERSION,
            "experiment": EXPERIMENT_NAME,
            "number_of_condition_evaluations": len(all_summaries),
            "model_seeds": [int(seed) for seed in args.seeds],
            "conditions_per_checkpoint": len(conditions),
            "validation_corruption_seed_independent_of_model_seed": True,
            "gaussian_scale_source": (
                "training_cache_per_feature_population_std"
            ),
            "official_test_split_accessed": False,
            "scientific_note": (
                "Step 9 records validation-only intrinsic-quality diagnostics. "
                "Formal H1-Q/H2-Q/H3-Q decisions are deferred to Step 10."
            ),
            "rows": all_summaries,
        },
    )

    print("=" * 78)
    print("STEP 9 QUALITY DIAGNOSTIC RUN COMPLETE")
    print("=" * 78)
    print("Condition evaluations:", len(all_summaries))
    print("Model parameters changed: NO")
    print("Official test samples accessed: 0")
    print("Formal H1-Q/H2-Q/H3-Q decision: DEFERRED TO STEP 10")
    print("Outputs:", args.output_root)


if __name__ == "__main__":
    main()
