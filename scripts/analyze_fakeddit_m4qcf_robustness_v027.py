#!/usr/bin/env python3
"""
AEGIS v0.27 Step14B formal M4qcf robustness analysis.

This analyzer is deliberately read-only with respect to the frozen
Step14A diagnostic evidence. It operationalizes the Step14B decision
rules frozen before formal result computation.

Primary hypotheses:
    H8-F  corrupted-modality down-weighting
    H9-F  robustness improvement over M1b
    H10-F mismatch interaction suppression

The official Fakeddit test split must remain sealed.

Scientific scope:
    controlled Fakeddit validation representation corruptions and
    deterministic class-preserving text-vision mismatch only.

This script does not train, tune, reselect checkpoints, or access the
official test set.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ANALYZER_VERSION = "0.27.0-step14b"
INPUT_PROTOCOL_VERSION = "0.27.0-step14a"
PARENT_PROTOCOL_VERSION = "0.27.0-step13a"
OPERATIONALIZATION_COMMIT = "8ac69d5"

EXPERIMENT_NAME = "fakeddit_m4qcf_robustness_diagnostics_v027"

FORMAL_SEEDS: Tuple[int, ...] = (42, 43, 44)
FORMAL_ARCHITECTURES: Tuple[str, ...] = ("M1b", "M4qc", "M4qcf")

ARCH_DIR = {
    "M1b": "m1b",
    "M4qc": "m4qc",
    "M4qcf": "m4qcf",
}

EXPECTED_VALIDATION_SAMPLES = 1000
EXPECTED_QUALITY_CONDITIONS = 19
EXPECTED_NONCLEAN_CONDITIONS = 18
EXPECTED_H8_H9_OBSERVATIONS = 54
EXPECTED_QUALITY_SUMMARIES = 171
EXPECTED_MISMATCH_SUMMARIES = 18

WEIGHT_SUM_TOLERANCE = 1e-5

H8_POSITIVE_PROPORTION_THRESHOLD = 0.80
H9_MEAN_IMPROVEMENT_THRESHOLD = 0.01
H9_POSITIVE_PROPORTION_THRESHOLD = 0.70

EXPECTED_MAPPING_SHA256 = (
    "975967d77622bcdad7e28597cb6f9f967"
    "c73bfe16d4a7455c1ab56cc7fbc6464"
)

NOT_COMPUTED_DECISIONS = {
    "H8_F": "NOT_COMPUTED",
    "H9_F": "NOT_COMPUTED",
    "H10_F": "NOT_COMPUTED",
}

REQUIRED_M4QCF_DIAGNOSTICS: Tuple[str, ...] = (
    "text_quality",
    "vision_quality",
    "compatibility_score",
    "effective_text_reliability",
    "effective_vision_reliability",
    "text_weight",
    "vision_weight",
    "interaction_multiplier",
)


class FormalAnalysisError(RuntimeError):
    """Raised when frozen Step14A evidence violates the formal contract."""


def fail(message: str) -> None:
    raise FormalAnalysisError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"Missing required JSON artifact: {path}")

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Could not parse JSON {path}: {exc}")

    require(isinstance(obj, dict), f"Expected JSON object in {path}")
    return obj


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    require(path.is_file(), f"Missing required JSONL artifact: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                fail(f"Blank JSONL line in {path}:{line_number}")

            try:
                row = json.loads(line)
            except Exception as exc:
                fail(f"Malformed JSONL in {path}:{line_number}: {exc}")

            require(
                isinstance(row, dict),
                f"Expected JSON object in {path}:{line_number}",
            )
            yield line_number, row


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric, got {type(value).__name__}",
    )
    out = float(value)
    require(math.isfinite(out), f"{label} must be finite, got {out}")
    return out


def probability(value: Any, label: str) -> float:
    out = finite_number(value, label)
    require(0.0 <= out <= 1.0, f"{label} outside [0,1]: {out}")
    return out


def strict_probability(value: Any, label: str) -> float:
    out = probability(value, label)
    require(0.0 < out < 1.0, f"{label} must be strictly inside (0,1): {out}")
    return out


def validate_not_computed(value: Any, label: str) -> None:
    require(
        value == NOT_COMPUTED_DECISIONS,
        f"{label} must remain {NOT_COMPUTED_DECISIONS}, got {value!r}",
    )


def validate_base_documents(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    quality_doc: Mapping[str, Any],
    mapping_doc: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    require(
        manifest.get("experiment") == EXPERIMENT_NAME,
        "Unexpected Step14A experiment name",
    )
    require(
        summary.get("experiment") == EXPERIMENT_NAME,
        "Unexpected Step14A summary experiment name",
    )

    require(
        manifest.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Manifest protocol version mismatch",
    )
    require(
        summary.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Summary protocol version mismatch",
    )
    require(
        quality_doc.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Quality-condition protocol version mismatch",
    )
    require(
        mapping_doc.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Mismatch-mapping protocol version mismatch",
    )

    require(
        manifest.get("parent_protocol_version") == PARENT_PROTOCOL_VERSION,
        "Manifest parent protocol mismatch",
    )
    require(
        summary.get("parent_protocol_version") == PARENT_PROTOCOL_VERSION,
        "Summary parent protocol mismatch",
    )

    require(manifest.get("smoke") is False, "Formal manifest cannot be smoke")
    require(summary.get("smoke") is False, "Formal summary cannot be smoke")

    require(
        summary.get("status") == "STEP14A_DIAGNOSTIC_EVIDENCE_COMPLETE",
        "Step14A evidence status is not complete",
    )

    for label, obj in (
        ("manifest", manifest),
        ("summary", summary),
        ("quality_conditions", quality_doc),
        ("mapping", mapping_doc),
    ):
        require(
            obj.get("official_test_accessed") is False,
            f"Official test access detected in {label}",
        )

    require(
        manifest.get("official_test_samples_accessed") == 0,
        "Manifest reports official-test sample access",
    )
    require(
        summary.get("official_test_samples_accessed") == 0,
        "Summary reports official-test sample access",
    )

    for field in (
        "training_performed",
        "checkpoint_reselection_performed",
        "threshold_tuning_performed",
    ):
        require(manifest.get(field) is False, f"Manifest {field} must be false")
        require(summary.get(field) is False, f"Summary {field} must be false")

    validate_not_computed(manifest.get("hypotheses"), "manifest hypotheses")
    validate_not_computed(
        summary.get("formal_hypothesis_decisions"),
        "summary formal_hypothesis_decisions",
    )

    require(
        manifest.get("formal_seeds") == list(FORMAL_SEEDS),
        "Unexpected formal seed set",
    )
    require(
        summary.get("seeds") == list(FORMAL_SEEDS),
        "Unexpected summary seed set",
    )
    require(
        manifest.get("architectures") == list(FORMAL_ARCHITECTURES),
        "Unexpected formal architecture set",
    )
    require(
        summary.get("architectures") == list(FORMAL_ARCHITECTURES),
        "Unexpected summary architecture set",
    )

    require(
        manifest.get("validation_samples") == EXPECTED_VALIDATION_SAMPLES,
        "Unexpected validation sample count in manifest",
    )
    require(
        summary.get("validation_samples") == EXPECTED_VALIDATION_SAMPLES,
        "Unexpected validation sample count in summary",
    )

    require(
        manifest.get("formal_quality_condition_level_evaluations")
        == EXPECTED_QUALITY_SUMMARIES,
        "Unexpected formal quality evaluation count",
    )
    require(
        manifest.get("quality_condition_level_evaluations")
        == EXPECTED_QUALITY_SUMMARIES,
        "Unexpected quality evaluation count",
    )
    require(
        summary.get("quality_summary_rows") == EXPECTED_QUALITY_SUMMARIES,
        "Unexpected quality summary row count",
    )
    require(
        summary.get("mismatch_summary_rows") == EXPECTED_MISMATCH_SUMMARIES,
        "Unexpected mismatch summary row count",
    )

    require(
        manifest.get("quality_conditions_per_architecture_seed")
        == EXPECTED_QUALITY_CONDITIONS,
        "Unexpected quality condition count in manifest",
    )
    require(
        summary.get("quality_conditions_per_architecture_seed")
        == EXPECTED_QUALITY_CONDITIONS,
        "Unexpected quality condition count in summary",
    )
    require(
        quality_doc.get("conditions_per_architecture_seed")
        == EXPECTED_QUALITY_CONDITIONS,
        "Unexpected quality condition count in quality_conditions.json",
    )

    require(
        quality_doc.get("corruption_realizations_shared_across_architectures")
        is True,
        "Corruption realizations are not shared across architectures",
    )
    require(
        quality_doc.get("corruption_realizations_shared_across_model_seeds")
        is True,
        "Corruption realizations are not shared across model seeds",
    )

    require(
        manifest.get("mismatch_mapping_shared_across_architectures") is True,
        "Mismatch mapping is not shared across architectures",
    )
    require(
        manifest.get("mismatch_mapping_shared_across_model_seeds") is True,
        "Mismatch mapping is not shared across seeds",
    )

    for obj, label in (
        (manifest, "manifest"),
        (summary, "summary"),
    ):
        require(
            obj.get("mismatch_mapping_sha256") == EXPECTED_MAPPING_SHA256,
            f"Unexpected mismatch mapping hash in {label}",
        )

    require(
        mapping_doc.get("mapping_sha256") == EXPECTED_MAPPING_SHA256,
        "Unexpected mapping SHA256",
    )
    require(
        mapping_doc.get("construction")
        == "frozen_step12b_deterministic_class_preserving_derangement",
        "Unexpected mismatch mapping construction",
    )
    require(
        mapping_doc.get("sample_count") == EXPECTED_VALIDATION_SAMPLES,
        "Unexpected mismatch mapping sample count",
    )

    conditions = quality_doc.get("conditions")
    require(isinstance(conditions, list), "conditions must be a list")
    require(
        len(conditions) == EXPECTED_QUALITY_CONDITIONS,
        f"Expected {EXPECTED_QUALITY_CONDITIONS} quality conditions",
    )

    slugs: List[str] = []
    clean_count = 0

    for index, raw in enumerate(conditions):
        require(isinstance(raw, dict), f"Condition {index} must be an object")

        slug = raw.get("slug")
        require(isinstance(slug, str) and slug, f"Invalid condition slug at {index}")
        slugs.append(slug)

        if slug == "clean":
            clean_count += 1
            require(raw.get("corrupted_modality") == "none", "Clean modality mismatch")
            require(raw.get("corruption_type") == "clean", "Clean type mismatch")
        else:
            require(
                raw.get("corrupted_modality") in {"text", "vision"},
                f"Non-clean condition {slug} must corrupt one modality",
            )
            require(
                raw.get("corruption_type")
                in {"gaussian_noise", "attenuation", "zero_dropout"},
                f"Unexpected corruption type for {slug}",
            )

        for hash_field in ("text_sha256", "vision_sha256"):
            value = raw.get(hash_field)
            require(
                isinstance(value, str) and len(value) == 64,
                f"Invalid {hash_field} for {slug}",
            )

    require(len(set(slugs)) == len(slugs), "Duplicate quality condition slug")
    require(clean_count == 1, "Expected exactly one clean condition")
    require(
        len([slug for slug in slugs if slug != "clean"])
        == EXPECTED_NONCLEAN_CONDITIONS,
        "Expected exactly 18 non-clean conditions",
    )

    validate_mapping_rows(mapping_doc)
    return [dict(item) for item in conditions]


def validate_mapping_rows(mapping_doc: Mapping[str, Any]) -> None:
    rows = mapping_doc.get("rows")
    require(isinstance(rows, list), "Mapping rows must be a list")
    require(
        len(rows) == EXPECTED_VALIDATION_SAMPLES,
        "Mapping must contain exactly 1000 rows",
    )

    receivers: List[int] = []
    donors: List[int] = []

    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"Mapping row {index} is malformed")
        require(row.get("self_pair") is False, f"Self-pair at mapping row {index}")
        require(
            row.get("class_preserved") is True,
            f"Class preservation false at mapping row {index}",
        )
        require(
            row.get("receiver_class") == row.get("donor_class"),
            f"Class-changing mismatch at mapping row {index}",
        )

        receiver = int(row.get("receiver_index"))
        donor = int(row.get("vision_donor_index"))

        require(receiver != donor, f"Self donor at mapping row {index}")

        receivers.append(receiver)
        donors.append(donor)

    require(
        sorted(receivers) == list(range(EXPECTED_VALIDATION_SAMPLES)),
        "Mismatch receiver indices are not exactly 0..999",
    )
    require(
        len(set(donors)) == EXPECTED_VALIDATION_SAMPLES,
        "Mismatch donor reuse violates derangement",
    )


def condition_map(conditions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(item["slug"]): dict(item) for item in conditions}


def quality_summary_path(
    root: Path,
    seed: int,
    architecture: str,
    condition: str,
) -> Path:
    return (
        root
        / f"seed{seed}"
        / ARCH_DIR[architecture]
        / "quality"
        / f"{condition}_summary.json"
    )


def quality_samples_path(
    root: Path,
    seed: int,
    architecture: str,
    condition: str,
) -> Path:
    return (
        root
        / f"seed{seed}"
        / ARCH_DIR[architecture]
        / "quality"
        / f"{condition}_samples.jsonl"
    )


def mismatch_summary_path(
    root: Path,
    seed: int,
    architecture: str,
    state: str,
) -> Path:
    return (
        root
        / f"seed{seed}"
        / ARCH_DIR[architecture]
        / "mismatch"
        / f"{state}_summary.json"
    )


def mismatch_samples_path(
    root: Path,
    seed: int,
    architecture: str,
    state: str,
) -> Path:
    return (
        root
        / f"seed{seed}"
        / ARCH_DIR[architecture]
        / "mismatch"
        / f"{state}_samples.jsonl"
    )


def validate_summary_common(
    obj: Mapping[str, Any],
    *,
    seed: int,
    architecture: str,
    sample_count: int = EXPECTED_VALIDATION_SAMPLES,
) -> None:
    require(obj.get("experiment") == EXPERIMENT_NAME, "Summary experiment mismatch")
    require(
        obj.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Summary protocol mismatch",
    )
    require(obj.get("model_seed") == seed, "Summary model seed mismatch")
    require(obj.get("architecture") == architecture, "Summary architecture mismatch")
    require(
        obj.get("official_test_accessed") is False,
        "Official test access detected in condition summary",
    )
    validate_not_computed(
        obj.get("formal_hypothesis_decisions"),
        "condition formal_hypothesis_decisions",
    )

    metrics = obj.get("metrics")
    require(isinstance(metrics, dict), "Condition metrics missing")

    require(
        metrics.get("sample_count") == sample_count,
        f"Expected sample_count={sample_count}",
    )

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "class0_f1",
        "class1_f1",
    ):
        probability(metrics.get(key), f"{architecture}/seed{seed}/{key}")

    loss = finite_number(
        metrics.get("classification_loss"),
        f"{architecture}/seed{seed}/classification_loss",
    )
    require(loss >= 0.0, "Classification loss cannot be negative")


def validate_m4qcf_diagnostics(
    obj: Mapping[str, Any],
    label: str,
) -> None:
    diagnostics = obj.get("diagnostics")
    require(isinstance(diagnostics, dict), f"{label}: diagnostics missing")

    for field in REQUIRED_M4QCF_DIAGNOSTICS:
        stat = diagnostics.get(field)
        require(isinstance(stat, dict), f"{label}: missing diagnostic {field}")

        for stat_name in ("min", "max", "mean", "std_population"):
            finite_number(
                stat.get(stat_name),
                f"{label}/{field}/{stat_name}",
            )

        minimum = probability(stat.get("min"), f"{label}/{field}/min")
        maximum = probability(stat.get("max"), f"{label}/{field}/max")
        mean = probability(stat.get("mean"), f"{label}/{field}/mean")

        require(minimum <= mean <= maximum, f"{label}/{field}: mean outside range")

        std = finite_number(
            stat.get("std_population"),
            f"{label}/{field}/std_population",
        )
        require(std >= 0.0, f"{label}/{field}: negative std")

    alpha_t = probability(
        diagnostics["text_weight"]["mean"],
        f"{label}/text_weight/mean",
    )
    alpha_v = probability(
        diagnostics["vision_weight"]["mean"],
        f"{label}/vision_weight/mean",
    )

    require(
        abs(alpha_t + alpha_v - 1.0) <= WEIGHT_SUM_TOLERANCE,
        f"{label}: mean fusion weights do not sum to one",
    )


def load_quality_summaries(
    root: Path,
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, str, str], Dict[str, Any]]:
    result: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    cmap = condition_map(conditions)

    for seed in FORMAL_SEEDS:
        for architecture in FORMAL_ARCHITECTURES:
            for slug, condition in cmap.items():
                path = quality_summary_path(root, seed, architecture, slug)
                obj = read_json(path)

                validate_summary_common(
                    obj,
                    seed=seed,
                    architecture=architecture,
                )

                require(obj.get("condition") == slug, f"{path}: condition mismatch")
                require(obj.get("mismatch_state") is None, f"{path}: unexpected mismatch")

                require(
                    obj.get("text_corruption_tensor_sha256")
                    == condition["text_sha256"],
                    f"{path}: text corruption hash mismatch",
                )
                require(
                    obj.get("vision_corruption_tensor_sha256")
                    == condition["vision_sha256"],
                    f"{path}: vision corruption hash mismatch",
                )

                if architecture == "M4qcf":
                    validate_m4qcf_diagnostics(
                        obj,
                        f"seed{seed}/{architecture}/{slug}",
                    )

                key = (seed, architecture, slug)
                require(key not in result, f"Duplicate quality summary: {key}")
                result[key] = obj

    require(
        len(result) == EXPECTED_QUALITY_SUMMARIES,
        "Incomplete quality-summary matrix",
    )
    return result


def load_mismatch_summaries(
    root: Path,
) -> Dict[Tuple[int, str, str], Dict[str, Any]]:
    result: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    for seed in FORMAL_SEEDS:
        for architecture in FORMAL_ARCHITECTURES:
            for state in ("matched", "mismatched"):
                path = mismatch_summary_path(root, seed, architecture, state)
                obj = read_json(path)

                validate_summary_common(
                    obj,
                    seed=seed,
                    architecture=architecture,
                )

                require(
                    obj.get("mismatch_state") == state,
                    f"{path}: mismatch_state mismatch",
                )

                expected_condition = (
                    "matched"
                    if state == "matched"
                    else "class_preserving_mismatched"
                )
                require(
                    obj.get("condition") == expected_condition,
                    f"{path}: mismatch condition mismatch",
                )
                require(
                    obj.get("mapping_sha256") == EXPECTED_MAPPING_SHA256,
                    f"{path}: mismatch mapping hash mismatch",
                )

                if architecture == "M4qcf":
                    validate_m4qcf_diagnostics(
                        obj,
                        f"seed{seed}/{architecture}/{state}",
                    )

                key = (seed, architecture, state)
                require(key not in result, f"Duplicate mismatch summary: {key}")
                result[key] = obj

    require(
        len(result) == EXPECTED_MISMATCH_SUMMARIES,
        "Incomplete mismatch-summary matrix",
    )
    return result


def validate_jsonl_evidence(
    root: Path,
    conditions: Sequence[Mapping[str, Any]],
    mapping_doc: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Validate persisted sample evidence without using it to tune or redefine
    any hypothesis criterion.
    """
    quality_rows = 0
    mismatch_rows = 0
    m4qcf_rows = 0
    max_weight_error = 0.0

    mapping_by_receiver = {
        int(row["receiver_index"]): row
        for row in mapping_doc["rows"]
    }

    for seed in FORMAL_SEEDS:
        for architecture in FORMAL_ARCHITECTURES:
            for condition in conditions:
                slug = str(condition["slug"])
                path = quality_samples_path(
                    root,
                    seed,
                    architecture,
                    slug,
                )

                seen_ids = set()
                count = 0

                for line_number, row in read_jsonl(path):
                    count += 1
                    quality_rows += 1

                    require(
                        row.get("official_test_accessed") is False,
                        f"{path}:{line_number}: test access",
                    )
                    require(
                        row.get("protocol_version") == INPUT_PROTOCOL_VERSION,
                        f"{path}:{line_number}: protocol mismatch",
                    )
                    require(
                        row.get("model_seed") == seed,
                        f"{path}:{line_number}: seed mismatch",
                    )
                    require(
                        row.get("architecture") == architecture,
                        f"{path}:{line_number}: architecture mismatch",
                    )
                    require(
                        row.get("condition") == slug,
                        f"{path}:{line_number}: condition mismatch",
                    )

                    sample_id = row.get("sample_id")
                    require(
                        isinstance(sample_id, str) and sample_id,
                        f"{path}:{line_number}: invalid sample id",
                    )
                    require(
                        sample_id not in seen_ids,
                        f"{path}:{line_number}: duplicate sample id",
                    )
                    seen_ids.add(sample_id)

                    target = row.get("target")
                    prediction = row.get("prediction")
                    require(target in (0, 1), f"{path}:{line_number}: invalid target")
                    require(
                        prediction in (0, 1),
                        f"{path}:{line_number}: invalid prediction",
                    )

                    finite_number(
                        row.get("classification_loss"),
                        f"{path}:{line_number}: classification_loss",
                    )

                    if architecture == "M4qcf":
                        for field in REQUIRED_M4QCF_DIAGNOSTICS:
                            probability(
                                row.get(field),
                                f"{path}:{line_number}:{field}",
                            )

                        alpha_t = strict_probability(
                            row["text_weight"],
                            f"{path}:{line_number}:text_weight",
                        )
                        alpha_v = strict_probability(
                            row["vision_weight"],
                            f"{path}:{line_number}:vision_weight",
                        )
                        error = abs(alpha_t + alpha_v - 1.0)
                        max_weight_error = max(max_weight_error, error)
                        require(
                            error <= WEIGHT_SUM_TOLERANCE,
                            f"{path}:{line_number}: fusion weights invalid",
                        )
                        m4qcf_rows += 1

                require(
                    count == EXPECTED_VALIDATION_SAMPLES,
                    f"{path}: expected 1000 rows, got {count}",
                )
                require(
                    len(seen_ids) == EXPECTED_VALIDATION_SAMPLES,
                    f"{path}: sample identity coverage failure",
                )

            for state in ("matched", "mismatched"):
                path = mismatch_samples_path(
                    root,
                    seed,
                    architecture,
                    state,
                )

                count = 0
                seen_receivers = set()

                for line_number, row in read_jsonl(path):
                    count += 1
                    mismatch_rows += 1

                    require(
                        row.get("official_test_accessed") is False,
                        f"{path}:{line_number}: test access",
                    )
                    require(
                        row.get("protocol_version") == INPUT_PROTOCOL_VERSION,
                        f"{path}:{line_number}: protocol mismatch",
                    )
                    require(
                        row.get("model_seed") == seed,
                        f"{path}:{line_number}: seed mismatch",
                    )
                    require(
                        row.get("architecture") == architecture,
                        f"{path}:{line_number}: architecture mismatch",
                    )
                    require(
                        row.get("mismatch_state") == state,
                        f"{path}:{line_number}: mismatch_state mismatch",
                    )

                    receiver_index = int(row.get("row_index"))
                    receiver_sample_id = row.get("receiver_sample_id")

                    require(
                        receiver_index not in seen_receivers,
                        f"{path}:{line_number}: duplicate receiver index",
                    )
                    seen_receivers.add(receiver_index)

                    if state == "mismatched":
                        require(
                            row.get("receiver_class") == row.get("donor_class"),
                            f"{path}:{line_number}: mismatch changes Stage1 class",
                        )

                        donor_index = int(row.get("vision_donor_index"))
                        require(
                            donor_index != receiver_index,
                            f"{path}:{line_number}: mismatch self-pair",
                        )

                        frozen = mapping_by_receiver.get(receiver_index)
                        require(
                            frozen is not None,
                            f"{path}:{line_number}: unknown receiver index",
                        )
                        require(
                            frozen["receiver_sample_id"] == receiver_sample_id,
                            f"{path}:{line_number}: receiver identity mismatch",
                        )
                        require(
                            int(frozen["vision_donor_index"]) == donor_index,
                            f"{path}:{line_number}: donor index mismatch",
                        )
                        require(
                            frozen["vision_donor_sample_id"]
                            == row.get("vision_donor_sample_id"),
                            f"{path}:{line_number}: donor identity mismatch",
                        )

                    if architecture == "M4qcf":
                        for field in REQUIRED_M4QCF_DIAGNOSTICS:
                            probability(
                                row.get(field),
                                f"{path}:{line_number}:{field}",
                            )

                        alpha_t = strict_probability(
                            row["text_weight"],
                            f"{path}:{line_number}:text_weight",
                        )
                        alpha_v = strict_probability(
                            row["vision_weight"],
                            f"{path}:{line_number}:vision_weight",
                        )
                        error = abs(alpha_t + alpha_v - 1.0)
                        max_weight_error = max(max_weight_error, error)
                        require(
                            error <= WEIGHT_SUM_TOLERANCE,
                            f"{path}:{line_number}: fusion weights invalid",
                        )
                        m4qcf_rows += 1

                require(
                    count == EXPECTED_VALIDATION_SAMPLES,
                    f"{path}: expected 1000 rows, got {count}",
                )
                require(
                    len(seen_receivers) == EXPECTED_VALIDATION_SAMPLES,
                    f"{path}: receiver coverage failure",
                )

    require(
        quality_rows == EXPECTED_QUALITY_SUMMARIES * EXPECTED_VALIDATION_SAMPLES,
        "Unexpected total quality JSONL rows",
    )
    require(
        mismatch_rows == EXPECTED_MISMATCH_SUMMARIES * EXPECTED_VALIDATION_SAMPLES,
        "Unexpected total mismatch JSONL rows",
    )
    require(
        m4qcf_rows == 63000,
        f"Expected 63000 M4qcf reliability rows, got {m4qcf_rows}",
    )

    return {
        "quality_sample_rows": quality_rows,
        "mismatch_sample_rows": mismatch_rows,
        "m4qcf_reliability_rows": m4qcf_rows,
        "max_weight_sum_error": max_weight_error,
        "status": "PASS",
    }


def mean_diag(
    summary: Mapping[str, Any],
    field: str,
    label: str,
) -> float:
    diagnostics = summary.get("diagnostics")
    require(isinstance(diagnostics, dict), f"{label}: diagnostics missing")
    stat = diagnostics.get(field)
    require(isinstance(stat, dict), f"{label}: {field} missing")
    return probability(stat.get("mean"), f"{label}/{field}/mean")


def macro_f1(summary: Mapping[str, Any], label: str) -> float:
    metrics = summary.get("metrics")
    require(isinstance(metrics, dict), f"{label}: metrics missing")
    return probability(metrics.get("macro_f1"), f"{label}/macro_f1")


def compute_h8(
    quality: Mapping[Tuple[int, str, str], Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []

    eligible = [
        condition
        for condition in conditions
        if condition["slug"] != "clean"
    ]

    require(
        len(eligible) == EXPECTED_NONCLEAN_CONDITIONS,
        "H8 eligible condition count mismatch",
    )

    for seed in FORMAL_SEEDS:
        clean = quality[(seed, "M4qcf", "clean")]

        clean_text_weight = mean_diag(
            clean,
            "text_weight",
            f"H8 seed{seed} clean",
        )
        clean_vision_weight = mean_diag(
            clean,
            "vision_weight",
            f"H8 seed{seed} clean",
        )

        for condition in eligible:
            slug = str(condition["slug"])
            modality = str(condition["corrupted_modality"])

            corrupt = quality[(seed, "M4qcf", slug)]

            if modality == "text":
                clean_weight = clean_text_weight
                corrupt_weight = mean_diag(
                    corrupt,
                    "text_weight",
                    f"H8 seed{seed} {slug}",
                )
            elif modality == "vision":
                clean_weight = clean_vision_weight
                corrupt_weight = mean_diag(
                    corrupt,
                    "vision_weight",
                    f"H8 seed{seed} {slug}",
                )
            else:
                fail(f"H8 unexpected corrupted modality: {modality}")

            down_weighting = clean_weight - corrupt_weight

            observations.append(
                {
                    "seed": seed,
                    "condition": slug,
                    "corrupted_modality": modality,
                    "corruption_type": condition["corruption_type"],
                    "severity": condition["severity"],
                    "clean_weight": clean_weight,
                    "corrupted_weight": corrupt_weight,
                    "down_weighting": down_weighting,
                    "strict_positive": down_weighting > 0.0,
                }
            )

    return decide_h8(observations)


def decide_h8(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(
        len(observations) == EXPECTED_H8_H9_OBSERVATIONS,
        f"H8 requires 54 observations, got {len(observations)}",
    )

    values = [
        finite_number(item["down_weighting"], "H8 down_weighting")
        for item in observations
    ]

    positive_count = sum(value > 0.0 for value in values)
    positive_proportion = positive_count / len(values)
    mean_down_weighting = statistics.fmean(values)

    supported = (
        positive_proportion >= H8_POSITIVE_PROPORTION_THRESHOLD
        and mean_down_weighting > 0.0
    )

    return {
        "hypothesis": "H8_F",
        "decision": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "criterion": {
            "positive_proportion_threshold": H8_POSITIVE_PROPORTION_THRESHOLD,
            "mean_down_weighting_strictly_greater_than": 0.0,
            "minimum_positive_count_for_54": 44,
        },
        "eligible_observations": len(values),
        "strict_positive_count": positive_count,
        "strict_positive_proportion": positive_proportion,
        "mean_down_weighting": mean_down_weighting,
        "observations": [dict(item) for item in observations],
    }


def compute_h9(
    quality: Mapping[Tuple[int, str, str], Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []

    eligible = [
        condition
        for condition in conditions
        if condition["slug"] != "clean"
    ]

    require(
        len(eligible) == EXPECTED_NONCLEAN_CONDITIONS,
        "H9 eligible condition count mismatch",
    )

    for seed in FORMAL_SEEDS:
        for condition in eligible:
            slug = str(condition["slug"])

            m1b = macro_f1(
                quality[(seed, "M1b", slug)],
                f"H9 seed{seed} M1b {slug}",
            )
            m4qc = macro_f1(
                quality[(seed, "M4qc", slug)],
                f"H9 seed{seed} M4qc {slug}",
            )
            m4qcf = macro_f1(
                quality[(seed, "M4qcf", slug)],
                f"H9 seed{seed} M4qcf {slug}",
            )

            improvement = m4qcf - m1b

            observations.append(
                {
                    "seed": seed,
                    "condition": slug,
                    "corrupted_modality": condition["corrupted_modality"],
                    "corruption_type": condition["corruption_type"],
                    "severity": condition["severity"],
                    "m1b_macro_f1": m1b,
                    "m4qc_macro_f1": m4qc,
                    "m4qcf_macro_f1": m4qcf,
                    "m4qcf_minus_m1b": improvement,
                    "strict_positive": improvement > 0.0,
                    "m4qcf_minus_m4qc_descriptive": m4qcf - m4qc,
                }
            )

    return decide_h9(observations)


def decide_h9(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(
        len(observations) == EXPECTED_H8_H9_OBSERVATIONS,
        f"H9 requires 54 observations, got {len(observations)}",
    )

    values = [
        finite_number(item["m4qcf_minus_m1b"], "H9 improvement")
        for item in observations
    ]

    positive_count = sum(value > 0.0 for value in values)
    positive_proportion = positive_count / len(values)
    mean_improvement = statistics.fmean(values)

    supported = (
        mean_improvement >= H9_MEAN_IMPROVEMENT_THRESHOLD
        and positive_proportion >= H9_POSITIVE_PROPORTION_THRESHOLD
    )

    return {
        "hypothesis": "H9_F",
        "decision": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "criterion": {
            "mean_improvement_at_least": H9_MEAN_IMPROVEMENT_THRESHOLD,
            "positive_proportion_threshold": H9_POSITIVE_PROPORTION_THRESHOLD,
            "minimum_positive_count_for_54": 38,
        },
        "eligible_observations": len(values),
        "strict_positive_count": positive_count,
        "strict_positive_proportion": positive_proportion,
        "mean_macro_f1_improvement": mean_improvement,
        "observations": [dict(item) for item in observations],
    }


def compute_h10(
    mismatch: Mapping[Tuple[int, str, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    seed_results: List[Dict[str, Any]] = []

    for seed in FORMAL_SEEDS:
        m4qcf_matched = mismatch[(seed, "M4qcf", "matched")]
        m4qcf_mismatched = mismatch[(seed, "M4qcf", "mismatched")]

        interaction_matched = mean_diag(
            m4qcf_matched,
            "interaction_multiplier",
            f"H10 seed{seed} M4qcf matched",
        )
        interaction_mismatched = mean_diag(
            m4qcf_mismatched,
            "interaction_multiplier",
            f"H10 seed{seed} M4qcf mismatched",
        )
        d_i = interaction_matched - interaction_mismatched

        m1b_matched = macro_f1(
            mismatch[(seed, "M1b", "matched")],
            f"H10 seed{seed} M1b matched",
        )
        m1b_mismatched = macro_f1(
            mismatch[(seed, "M1b", "mismatched")],
            f"H10 seed{seed} M1b mismatched",
        )

        m4qcf_matched_f1 = macro_f1(
            m4qcf_matched,
            f"H10 seed{seed} M4qcf matched",
        )
        m4qcf_mismatched_f1 = macro_f1(
            m4qcf_mismatched,
            f"H10 seed{seed} M4qcf mismatched",
        )

        m4qc_matched = macro_f1(
            mismatch[(seed, "M4qc", "matched")],
            f"H10 seed{seed} M4qc matched",
        )
        m4qc_mismatched = macro_f1(
            mismatch[(seed, "M4qc", "mismatched")],
            f"H10 seed{seed} M4qc mismatched",
        )

        delta_m1b = m1b_matched - m1b_mismatched
        delta_m4qcf = m4qcf_matched_f1 - m4qcf_mismatched_f1
        delta_m4qc = m4qc_matched - m4qc_mismatched

        g = delta_m1b - delta_m4qcf

        seed_results.append(
            {
                "seed": seed,
                "interaction_multiplier_matched_mean": interaction_matched,
                "interaction_multiplier_mismatched_mean": interaction_mismatched,
                "interaction_suppression_D_I": d_i,
                "interaction_suppression_strict_positive": d_i > 0.0,
                "m1b_matched_macro_f1": m1b_matched,
                "m1b_mismatched_macro_f1": m1b_mismatched,
                "m1b_mismatch_drop": delta_m1b,
                "m4qcf_matched_macro_f1": m4qcf_matched_f1,
                "m4qcf_mismatched_macro_f1": m4qcf_mismatched_f1,
                "m4qcf_mismatch_drop": delta_m4qcf,
                "G": g,
                "m4qc_matched_macro_f1_descriptive": m4qc_matched,
                "m4qc_mismatched_macro_f1_descriptive": m4qc_mismatched,
                "m4qc_mismatch_drop_descriptive": delta_m4qc,
            }
        )

    return decide_h10(seed_results)


def decide_h10(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(len(seed_results) == 3, "H10 requires exactly three seed results")

    d_i_values = [
        finite_number(item["interaction_suppression_D_I"], "H10 D_I")
        for item in seed_results
    ]
    g_values = [
        finite_number(item["G"], "H10 G")
        for item in seed_results
    ]

    all_interaction_positive = all(value > 0.0 for value in d_i_values)
    mean_g = statistics.fmean(g_values)

    supported = all_interaction_positive and mean_g > 0.0

    return {
        "hypothesis": "H10_F",
        "decision": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "criterion": {
            "interaction_suppression_positive_all_seeds": True,
            "mean_G_strictly_greater_than": 0.0,
        },
        "interaction_suppression_positive_seed_count": sum(
            value > 0.0 for value in d_i_values
        ),
        "all_three_interaction_suppression_positive": all_interaction_positive,
        "mean_interaction_suppression_D_I": statistics.fmean(d_i_values),
        "mean_G": mean_g,
        "seed_results": [dict(item) for item in seed_results],
    }


def analyze_formal(
    input_root: Path,
    output_root: Path,
    *,
    validate_jsonl: bool = True,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    manifest = read_json(input_root / "manifest.json")
    step14a_summary = read_json(input_root / "summary.json")
    quality_doc = read_json(input_root / "quality_conditions.json")
    mapping_doc = read_json(
        input_root / "validation_derangement_mapping.json"
    )

    conditions = validate_base_documents(
        manifest,
        step14a_summary,
        quality_doc,
        mapping_doc,
    )

    quality = load_quality_summaries(input_root, conditions)
    mismatch = load_mismatch_summaries(input_root)

    evidence_audit = (
        validate_jsonl_evidence(input_root, conditions, mapping_doc)
        if validate_jsonl
        else {
            "status": "SKIPPED_FOR_SYNTHETIC_TEST_ONLY",
            "quality_sample_rows": None,
            "mismatch_sample_rows": None,
            "m4qcf_reliability_rows": None,
            "max_weight_sum_error": None,
        }
    )

    h8 = compute_h8(quality, conditions)
    h9 = compute_h9(quality, conditions)
    h10 = compute_h10(mismatch)

    decisions = {
        "H8_F": h8["decision"],
        "H9_F": h9["decision"],
        "H10_F": h10["decision"],
    }

    analysis = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "parent_protocol_version": PARENT_PROTOCOL_VERSION,
        "operationalization_commit": OPERATIONALIZATION_COMMIT,
        "experiment": "fakeddit_m4qcf_robustness_formal_analysis_v027",
        "input_root": str(input_root),
        "official_test_accessed": False,
        "official_test_samples_accessed": 0,
        "training_performed": False,
        "checkpoint_reselection_performed": False,
        "threshold_tuning_performed": False,
        "analysis_only": True,
        "formal_seeds": list(FORMAL_SEEDS),
        "architectures": list(FORMAL_ARCHITECTURES),
        "eligible_single_modality_conditions_per_seed": (
            EXPECTED_NONCLEAN_CONDITIONS
        ),
        "h8_h9_seed_condition_observations": (
            EXPECTED_H8_H9_OBSERVATIONS
        ),
        "mapping_sha256": EXPECTED_MAPPING_SHA256,
        "evidence_audit": evidence_audit,
        "formal_decisions": decisions,
        "H8_F": h8,
        "H9_F": h9,
        "H10_F": h10,
        "interpretation_boundary": [
            "controlled Fakeddit validation representations only",
            "controlled single-modality representation corruptions only",
            "controlled deterministic class-preserving mismatch only",
            "no open-world factual verification claim",
            "no source-credibility claim",
            "no author-intent claim",
            "no human-trust claim",
            "no universal semantic-consistency claim",
            "no arbitrary real-world robustness claim",
            "no external-evidence verification claim",
            "no temporal or geographic reasoning claim",
        ],
    }

    compact_summary = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "operationalization_commit": OPERATIONALIZATION_COMMIT,
        "status": "STEP14B_FORMAL_ANALYSIS_COMPLETE",
        "formal_decisions": decisions,
        "H8_F": {
            "decision": h8["decision"],
            "eligible_observations": h8["eligible_observations"],
            "strict_positive_count": h8["strict_positive_count"],
            "strict_positive_proportion": h8["strict_positive_proportion"],
            "mean_down_weighting": h8["mean_down_weighting"],
        },
        "H9_F": {
            "decision": h9["decision"],
            "eligible_observations": h9["eligible_observations"],
            "strict_positive_count": h9["strict_positive_count"],
            "strict_positive_proportion": h9["strict_positive_proportion"],
            "mean_macro_f1_improvement": h9[
                "mean_macro_f1_improvement"
            ],
        },
        "H10_F": {
            "decision": h10["decision"],
            "interaction_suppression_positive_seed_count": h10[
                "interaction_suppression_positive_seed_count"
            ],
            "all_three_interaction_suppression_positive": h10[
                "all_three_interaction_suppression_positive"
            ],
            "mean_interaction_suppression_D_I": h10[
                "mean_interaction_suppression_D_I"
            ],
            "mean_G": h10["mean_G"],
        },
        "official_test_accessed": False,
        "official_test_samples_accessed": 0,
        "training_performed": False,
        "checkpoint_reselection_performed": False,
        "threshold_tuning_performed": False,
        "analysis_only": True,
        "evidence_audit_status": evidence_audit["status"],
    }

    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)

        (output_root / "formal_analysis.json").write_text(
            json.dumps(
                analysis,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        (output_root / "summary.json").write_text(
            json.dumps(
                compact_summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return analysis


def print_result(analysis: Mapping[str, Any], output_root: Path) -> None:
    print("=" * 78)
    print("AEGIS v0.27 STEP14B - FORMAL M4qcf ROBUSTNESS ANALYSIS")
    print("=" * 78)

    h8 = analysis["H8_F"]
    h9 = analysis["H9_F"]
    h10 = analysis["H10_F"]

    print()
    print(
        "H8-F:",
        h8["decision"],
        f"| positive={h8['strict_positive_count']}/"
        f"{h8['eligible_observations']}",
        f"| proportion={h8['strict_positive_proportion']:.12f}",
        f"| mean_down_weighting={h8['mean_down_weighting']:.12f}",
    )

    print(
        "H9-F:",
        h9["decision"],
        f"| positive={h9['strict_positive_count']}/"
        f"{h9['eligible_observations']}",
        f"| proportion={h9['strict_positive_proportion']:.12f}",
        f"| mean_improvement={h9['mean_macro_f1_improvement']:.12f}",
    )

    print(
        "H10-F:",
        h10["decision"],
        "| interaction_positive_seeds="
        f"{h10['interaction_suppression_positive_seed_count']}/3",
        f"| mean_D_I={h10['mean_interaction_suppression_D_I']:.12f}",
        f"| mean_G={h10['mean_G']:.12f}",
    )

    print()
    print("Official test accessed: NO")
    print("Training performed: NO")
    print("Checkpoint reselection: NO")
    print("Threshold tuning: NO")
    print(
        "Evidence audit:",
        analysis["evidence_audit"]["status"],
    )
    print("Output:", output_root)

    print()
    print("=" * 78)
    print("STEP14B_FORMAL_ANALYSIS_COMPLETE")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Step14B analysis of frozen AEGIS v0.27 "
            "M4qcf robustness diagnostics."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "experiments/fakeddit/"
            "v027_m4qcf_robustness_diagnostics"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "experiments/fakeddit/"
            "v027_m4qcf_robustness_analysis"
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        analysis = analyze_formal(
            args.input_root,
            args.output_root,
            validate_jsonl=True,
            write_outputs=True,
        )
    except FormalAnalysisError as exc:
        print("=" * 78)
        print("STEP14B FORMAL ANALYSIS FAILED LOUDLY")
        print("=" * 78)
        print(str(exc))
        return 1

    print_result(analysis, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
