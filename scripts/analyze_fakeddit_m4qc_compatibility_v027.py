from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROTOCOL_VERSION = "0.27.0-step11a"
ANALYZER_VERSION = "0.27.0-step12c"
FORMAL_SEEDS = (42, 43, 44)
EXPECTED_ROWS_PER_SEED = 1000

H4_MEAN_AUC_THRESHOLD = 0.80
H5_TEXT_MEAN_ABS_DELTA_THRESHOLD = 0.10
H5_VISION_MEAN_ABS_DELTA_THRESHOLD = 0.10
H5_COMPATIBILITY_DROP_THRESHOLD = 0.0

DEFAULT_DIAGNOSTICS_ROOT = Path(
    "experiments/fakeddit/v027_m4qc_compatibility_diagnostics"
)
DEFAULT_STEP11E_RECORD = Path(
    "docs/experiments/v027/step11e_m4qc_training_results.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "experiments/fakeddit/v027_m4qc_compatibility_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "AEGIS v0.27 Step12C formal H4-C/H5-C analysis from the "
            "frozen Step12B deterministic compatibility diagnostics."
        )
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_ROOT,
    )
    parser.add_argument(
        "--step11e-record",
        type=Path,
        default=DEFAULT_STEP11E_RECORD,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing Step12C output root. This never modifies "
            "the frozen Step12B diagnostics."
        ),
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSON in {path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Expected JSON object in {path} line {line_number}."
                )
            rows.append(row)
    return rows


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _finite_unit_interval(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite: {value!r}")
    if result < 0.0 or result > 1.0:
        raise RuntimeError(f"{name} is outside [0,1]: {result}")
    return result


def roc_auc_binary(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> float:
    """
    Compute binary ROC-AUC using the Mann-Whitney interpretation.

    Ties contribute 0.5. This implementation is dependency-free and exact
    for the finite score arrays used by Step12C.
    """
    if not positive_scores or not negative_scores:
        raise ValueError("ROC-AUC requires non-empty positive and negative scores.")

    positives = [float(x) for x in positive_scores]
    negatives = [float(x) for x in negative_scores]

    for i, value in enumerate(positives):
        if not math.isfinite(value):
            raise ValueError(f"Positive score {i} is non-finite.")
    for i, value in enumerate(negatives):
        if not math.isfinite(value):
            raise ValueError(f"Negative score {i} is non-finite.")

    combined: List[Tuple[float, int]] = (
        [(value, 1) for value in positives]
        + [(value, 0) for value in negatives]
    )
    combined.sort(key=lambda item: item[0])

    rank = 1
    positive_rank_sum = 0.0
    index = 0

    while index < len(combined):
        end = index + 1
        score = combined[index][0]
        while end < len(combined) and combined[end][0] == score:
            end += 1

        group_size = end - index
        average_rank = (rank + (rank + group_size - 1)) / 2.0
        positive_count = sum(label for _, label in combined[index:end])
        positive_rank_sum += positive_count * average_rank

        rank += group_size
        index = end

    n_pos = len(positives)
    n_neg = len(negatives)
    u_statistic = positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    auc = u_statistic / (n_pos * n_neg)

    if not math.isfinite(auc) or auc < 0.0 or auc > 1.0:
        raise RuntimeError(f"Computed invalid ROC-AUC: {auc}")
    return float(auc)


def validate_mapping_payload(
    mapping_payload: Dict[str, Any],
) -> Dict[int, Dict[str, Any]]:
    _require(
        mapping_payload.get("protocol_version") == PROTOCOL_VERSION,
        "Step12B mapping protocol-version mismatch.",
    )
    _require(
        mapping_payload.get("official_test_accessed") is False,
        "Step12B mapping reports official test access.",
    )
    _require(
        int(mapping_payload.get("validation_sample_count", -1))
        == EXPECTED_ROWS_PER_SEED,
        "Step12B mapping must contain exactly 1000 validation samples.",
    )

    invariants = mapping_payload.get("invariants")
    _require(isinstance(invariants, dict), "Mapping invariants are missing.")
    for key in (
        "no_self_pairs",
        "class_preserving",
        "one_mismatch_partner_per_text",
        "each_vision_used_exactly_once_within_class",
        "mapping_shared_across_model_seeds",
        "mapping_independent_of_training_rng",
    ):
        _require(
            invariants.get(key) is True,
            f"Frozen mapping invariant failed or missing: {key}",
        )

    mapping = mapping_payload.get("mapping")
    _require(isinstance(mapping, list), "Mapping rows are missing.")
    _require(
        len(mapping) == EXPECTED_ROWS_PER_SEED,
        "Mapping must contain exactly 1000 rows.",
    )

    by_index: Dict[int, Dict[str, Any]] = {}
    donors: List[int] = []

    for row in mapping:
        _require(isinstance(row, dict), "Malformed mapping row.")
        index = int(row["validation_row_index"])
        donor = int(row["donor_vision_index"])
        label = int(row["stage1_label"])
        donor_label = int(row["donor_stage1_label"])

        _require(
            0 <= index < EXPECTED_ROWS_PER_SEED,
            f"Mapping row index out of range: {index}",
        )
        _require(
            0 <= donor < EXPECTED_ROWS_PER_SEED,
            f"Mapping donor index out of range: {donor}",
        )
        _require(index != donor, f"Mapping contains self-pair at row {index}.")
        _require(
            row.get("self_pair") is False,
            f"Mapping self_pair flag is not false at row {index}.",
        )
        _require(
            label == donor_label,
            f"Mapping changes Stage-1 class at row {index}.",
        )
        _require(index not in by_index, f"Duplicate mapping row index: {index}")

        by_index[index] = row
        donors.append(donor)

    _require(
        set(by_index) == set(range(EXPECTED_ROWS_PER_SEED)),
        "Mapping does not cover validation indices 0..999 exactly once.",
    )
    _require(
        len(set(donors)) == EXPECTED_ROWS_PER_SEED,
        "Mapping donor indices are not globally one-to-one.",
    )
    _require(
        set(donors) == set(range(EXPECTED_ROWS_PER_SEED)),
        "Mapping does not use every validation vision exactly once.",
    )

    return by_index


def validate_step12b_manifest(manifest: Dict[str, Any]) -> None:
    _require(
        manifest.get("protocol_version") == PROTOCOL_VERSION,
        "Step12B manifest protocol-version mismatch.",
    )
    _require(
        manifest.get("stage") == "Step12B",
        "Expected Step12B manifest.",
    )
    _require(
        manifest.get("official_test_split_accessed") is False,
        "Step12B manifest reports official test access.",
    )
    _require(
        manifest.get("formal_seeds") == list(FORMAL_SEEDS),
        "Step12B manifest formal seeds must be [42,43,44].",
    )
    _require(
        int(manifest.get("validation_sample_count", -1))
        == EXPECTED_ROWS_PER_SEED,
        "Step12B manifest validation sample count mismatch.",
    )
    _require(
        int(manifest.get("matched_observations_per_seed", -1))
        == EXPECTED_ROWS_PER_SEED,
        "Step12B matched observation count mismatch.",
    )
    _require(
        int(manifest.get("mismatched_observations_per_seed", -1))
        == EXPECTED_ROWS_PER_SEED,
        "Step12B mismatched observation count mismatch.",
    )

    boundary = manifest.get("analysis_boundary")
    _require(isinstance(boundary, dict), "Step12B analysis boundary is missing.")
    forbidden_true = (
        "roc_auc_computed",
        "h4_c_decision_computed",
        "h5_c_decision_computed",
        "classification_preservation_recomputed",
        "threshold_tuning_performed",
        "checkpoint_reselection_performed",
        "training_performed",
        "m4qcf_computed",
    )
    for key in forbidden_true:
        _require(
            boundary.get(key) is False,
            f"Step12B boundary violation: {key} is not false.",
        )


def validate_step11e_record(record: Dict[str, Any]) -> Dict[str, Any]:
    _require(
        record.get("record_version") == "0.27.0-step11e",
        "Unexpected Step11E record version.",
    )
    _require(
        record.get("stage") == "Step11E",
        "Expected Step11E record.",
    )
    _require(
        record.get("status") == "COMPLETE_AND_FROZEN",
        "Step11E record is not COMPLETE_AND_FROZEN.",
    )

    scope = record.get("scientific_scope")
    _require(isinstance(scope, dict), "Step11E scientific_scope is missing.")
    _require(
        scope.get("official_test_accessed") is False,
        "Step11E reports official test access.",
    )
    _require(
        scope.get("quality_affects_primary_fusion") is False,
        "Step11E reports quality affecting primary fusion.",
    )
    _require(
        scope.get("compatibility_affects_primary_fusion") is False,
        "Step11E reports compatibility affecting primary fusion.",
    )

    gate = record.get("classification_preservation_gate")
    _require(
        isinstance(gate, dict),
        "Step11E classification-preservation gate is missing.",
    )
    _require(
        gate.get("passed") is True,
        "Frozen Step11E classification-preservation gate did not pass.",
    )

    delta = float(gate["delta_vs_m1b"])
    margin = float(gate["frozen_margin"])
    _require(
        math.isfinite(delta) and math.isfinite(margin),
        "Step11E classification gate contains non-finite values.",
    )
    _require(
        delta >= margin,
        "Step11E classification gate is internally inconsistent.",
    )

    pending = record.get("pending_hypotheses")
    _require(isinstance(pending, dict), "Step11E pending_hypotheses missing.")
    _require(
        pending.get("H4-C", {}).get("status") == "PENDING_NOT_COMPUTED",
        "Step11E H4-C status was not pending.",
    )
    _require(
        pending.get("H5-C", {}).get("status") == "PENDING_NOT_COMPUTED",
        "Step11E H5-C status was not pending.",
    )

    return gate


def validate_seed_rows(
    rows: List[Dict[str, Any]],
    *,
    seed: int,
    mapping_by_index: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    _require(
        len(rows) == EXPECTED_ROWS_PER_SEED,
        f"Seed {seed}: expected 1000 diagnostic rows; got {len(rows)}.",
    )

    by_index: Dict[int, Dict[str, Any]] = {}

    for row in rows:
        _require(
            row.get("protocol_version") == PROTOCOL_VERSION,
            f"Seed {seed}: protocol-version mismatch.",
        )
        _require(int(row.get("seed", -1)) == seed, f"Seed {seed}: seed mismatch.")
        _require(
            row.get("official_test_accessed") is False,
            f"Seed {seed}: diagnostic row reports official test access.",
        )

        index = int(row["validation_row_index"])
        _require(
            0 <= index < EXPECTED_ROWS_PER_SEED,
            f"Seed {seed}: row index out of range: {index}",
        )
        _require(
            index not in by_index,
            f"Seed {seed}: duplicate diagnostic row index {index}.",
        )

        mapping = mapping_by_index[index]
        donor = int(mapping["donor_vision_index"])

        _require(
            int(row["text_source_index"]) == index,
            f"Seed {seed} row {index}: text source identity mismatch.",
        )
        _require(
            int(row["stage1_label"]) == int(mapping["stage1_label"]),
            f"Seed {seed} row {index}: label mismatch with frozen mapping.",
        )

        matched = row.get("matched")
        mismatched = row.get("mismatched")
        _require(isinstance(matched, dict), f"Seed {seed}: malformed matched row.")
        _require(
            isinstance(mismatched, dict),
            f"Seed {seed}: malformed mismatched row.",
        )

        _require(
            int(matched["vision_source_index"]) == index,
            f"Seed {seed} row {index}: matched vision identity mismatch.",
        )
        _require(
            int(mismatched["vision_source_index"]) == donor,
            f"Seed {seed} row {index}: mismatched donor identity mismatch.",
        )
        _require(
            int(mismatched["vision_source_stage1_label"])
            == int(row["stage1_label"]),
            f"Seed {seed} row {index}: mismatched donor changes class.",
        )
        _require(
            int(matched["compatibility_target"]) == 1,
            f"Seed {seed} row {index}: matched target must be 1.",
        )
        _require(
            int(mismatched["compatibility_target"]) == 0,
            f"Seed {seed} row {index}: mismatched target must be 0.",
        )

        for condition_name, condition in (
            ("matched", matched),
            ("mismatched", mismatched),
        ):
            for key in ("text_quality", "vision_quality", "compatibility"):
                _finite_unit_interval(
                    condition[key],
                    f"seed {seed} row {index} {condition_name}.{key}",
                )

        by_index[index] = row

    _require(
        set(by_index) == set(range(EXPECTED_ROWS_PER_SEED)),
        f"Seed {seed}: diagnostic rows do not cover indices 0..999 exactly once.",
    )
    return by_index


def analyze_seed(
    *,
    seed: int,
    rows_by_index: Dict[int, Dict[str, Any]],
    mapping_by_index: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    matched_compatibility = [
        float(rows_by_index[index]["matched"]["compatibility"])
        for index in range(EXPECTED_ROWS_PER_SEED)
    ]
    mismatched_compatibility = [
        float(rows_by_index[index]["mismatched"]["compatibility"])
        for index in range(EXPECTED_ROWS_PER_SEED)
    ]

    auc = roc_auc_binary(
        positive_scores=matched_compatibility,
        negative_scores=mismatched_compatibility,
    )

    text_abs_deltas: List[float] = []
    vision_abs_deltas: List[float] = []

    for index in range(EXPECTED_ROWS_PER_SEED):
        row = rows_by_index[index]
        donor = int(mapping_by_index[index]["donor_vision_index"])

        matched_text_q = float(row["matched"]["text_quality"])
        mismatched_text_q = float(row["mismatched"]["text_quality"])
        text_abs_deltas.append(abs(mismatched_text_q - matched_text_q))

        # H5-C vision quality is aligned by representation identity.
        # The mismatched row i contains vision h_V^(pi(i)); its matched
        # reference is therefore the matched q_V stored on row pi(i).
        mismatched_vision_q = float(row["mismatched"]["vision_quality"])
        matched_same_vision_q = float(
            rows_by_index[donor]["matched"]["vision_quality"]
        )
        vision_abs_deltas.append(
            abs(mismatched_vision_q - matched_same_vision_q)
        )

    mean_matched_c = mean(matched_compatibility)
    mean_mismatched_c = mean(mismatched_compatibility)
    compatibility_drop = mean_matched_c - mean_mismatched_c

    return {
        "seed": seed,
        "observations": {
            "matched": EXPECTED_ROWS_PER_SEED,
            "mismatched": EXPECTED_ROWS_PER_SEED,
            "total_compatibility": 2 * EXPECTED_ROWS_PER_SEED,
        },
        "h4_c": {
            "roc_auc": auc,
            "mean_matched_compatibility": mean_matched_c,
            "mean_mismatched_compatibility": mean_mismatched_c,
        },
        "h5_c": {
            "mean_abs_text_quality_change": mean(text_abs_deltas),
            "mean_abs_vision_quality_change": mean(vision_abs_deltas),
            "compatibility_drop": compatibility_drop,
            "quality_alignment_rule": "representation_identity",
        },
        "official_test_accessed": False,
    }


def compute_formal_decisions(
    seed_results: Sequence[Dict[str, Any]],
    classification_gate: Dict[str, Any],
) -> Dict[str, Any]:
    _require(
        [int(result["seed"]) for result in seed_results] == list(FORMAL_SEEDS),
        "Formal seed results must be ordered [42,43,44].",
    )

    aucs = [float(result["h4_c"]["roc_auc"]) for result in seed_results]
    text_changes = [
        float(result["h5_c"]["mean_abs_text_quality_change"])
        for result in seed_results
    ]
    vision_changes = [
        float(result["h5_c"]["mean_abs_vision_quality_change"])
        for result in seed_results
    ]
    compatibility_drops = [
        float(result["h5_c"]["compatibility_drop"])
        for result in seed_results
    ]

    mean_auc = mean(aucs)
    mean_text_change = mean(text_changes)
    mean_vision_change = mean(vision_changes)
    mean_compatibility_drop = mean(compatibility_drops)

    h4_supported = mean_auc >= H4_MEAN_AUC_THRESHOLD
    h5_text_pass = mean_text_change < H5_TEXT_MEAN_ABS_DELTA_THRESHOLD
    h5_vision_pass = mean_vision_change < H5_VISION_MEAN_ABS_DELTA_THRESHOLD
    h5_drop_pass = mean_compatibility_drop > H5_COMPATIBILITY_DROP_THRESHOLD
    h5_supported = h5_text_pass and h5_vision_pass and h5_drop_pass

    return {
        "H4-C": {
            "status": "SUPPORTED" if h4_supported else "NOT_SUPPORTED",
            "seed_roc_auc": {
                str(result["seed"]): float(result["h4_c"]["roc_auc"])
                for result in seed_results
            },
            "mean_roc_auc": mean_auc,
            "criterion": "mean_roc_auc >= 0.80",
            "threshold": H4_MEAN_AUC_THRESHOLD,
            "supported": h4_supported,
            "no_per_seed_minimum": True,
        },
        "H5-C": {
            "status": "SUPPORTED" if h5_supported else "NOT_SUPPORTED",
            "seed_mean_abs_text_quality_change": {
                str(result["seed"]): float(
                    result["h5_c"]["mean_abs_text_quality_change"]
                )
                for result in seed_results
            },
            "seed_mean_abs_vision_quality_change": {
                str(result["seed"]): float(
                    result["h5_c"]["mean_abs_vision_quality_change"]
                )
                for result in seed_results
            },
            "seed_compatibility_drop": {
                str(result["seed"]): float(
                    result["h5_c"]["compatibility_drop"]
                )
                for result in seed_results
            },
            "aggregate": {
                "mean_abs_text_quality_change": mean_text_change,
                "mean_abs_vision_quality_change": mean_vision_change,
                "mean_compatibility_drop": mean_compatibility_drop,
            },
            "criteria": {
                "text": "mean_abs_text_quality_change < 0.10",
                "vision": "mean_abs_vision_quality_change < 0.10",
                "compatibility_drop": "mean_compatibility_drop > 0",
            },
            "component_pass": {
                "text_quality_stability": h5_text_pass,
                "vision_quality_stability": h5_vision_pass,
                "positive_compatibility_drop": h5_drop_pass,
            },
            "supported": h5_supported,
            "quality_alignment_rule": "representation_identity",
        },
        "classification_preservation": {
            "status": "PASS",
            "source": "frozen Step11E record",
            "reference_model": classification_gate["reference_model"],
            "reference_macro_f1": float(
                classification_gate["reference_macro_f1"]
            ),
            "m4qc_mean_macro_f1": float(
                classification_gate["m4qc_mean_macro_f1"]
            ),
            "delta_vs_m1b": float(classification_gate["delta_vs_m1b"]),
            "frozen_margin": float(classification_gate["frozen_margin"]),
            "criterion": classification_gate["criterion"],
            "passed": True,
            "interpretation": classification_gate["interpretation"],
        },
    }


def _prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. "
                "Use --overwrite only for an intentional rerun of the "
                "same frozen Step12C analyzer."
            )
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parse_args()

    diagnostics_root = args.diagnostics_root
    if not diagnostics_root.is_dir():
        raise FileNotFoundError(
            f"Step12B diagnostics root not found: {diagnostics_root}"
        )

    manifest = _load_json(diagnostics_root / "manifest.json")
    mapping_payload = _load_json(
        diagnostics_root / "validation_derangement_mapping.json"
    )
    step11e_record = _load_json(args.step11e_record)

    validate_step12b_manifest(manifest)
    mapping_by_index = validate_mapping_payload(mapping_payload)
    classification_gate = validate_step11e_record(step11e_record)

    seed_results: List[Dict[str, Any]] = []

    for seed in FORMAL_SEEDS:
        rows = _load_jsonl(
            diagnostics_root / f"seed{seed}_diagnostics.jsonl"
        )
        rows_by_index = validate_seed_rows(
            rows,
            seed=seed,
            mapping_by_index=mapping_by_index,
        )
        seed_results.append(
            analyze_seed(
                seed=seed,
                rows_by_index=rows_by_index,
                mapping_by_index=mapping_by_index,
            )
        )

    decisions = compute_formal_decisions(
        seed_results,
        classification_gate,
    )

    _prepare_output_root(args.output_root, args.overwrite)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "stage": "Step12C",
        "status": "FORMAL_ANALYSIS_COMPLETE",
        "source_diagnostics_root": str(diagnostics_root).replace("\\", "/"),
        "source_step11e_record": str(args.step11e_record).replace("\\", "/"),
        "formal_seeds": list(FORMAL_SEEDS),
        "validation_samples_per_seed": EXPECTED_ROWS_PER_SEED,
        "seed_results": seed_results,
        "formal_decisions": decisions,
        "analysis_invariants": {
            "frozen_step12b_mapping_used": True,
            "mapping_shared_across_seeds": True,
            "quality_aligned_by_representation_identity": True,
            "checkpoint_reselection_performed": False,
            "training_performed": False,
            "threshold_tuning_performed": False,
            "official_test_split_accessed": False,
            "m4qcf_computed": False,
        },
        "interpretation_boundaries": [
            (
                "H4-C measures controlled matched-versus-class-preserving-"
                "mismatched compatibility discrimination on the frozen "
                "Fakeddit validation protocol."
            ),
            (
                "H5-C measures separation of intrinsic modality quality from "
                "pairwise compatibility under the same controlled protocol."
            ),
            (
                "These results do not establish factual truth, factual "
                "verification, source credibility, intent, human trust, "
                "universal semantic consistency, external-world evidence, "
                "raw-world generalization, temporal/geographic verification, "
                "or reliability-informed fusion benefit."
            ),
            "The official Fakeddit test split remained sealed.",
        ],
        "next_stage": "Step12D tracked scientific results freeze",
    }

    _save_json(args.output_root / "formal_analysis.json", result)

    summary = {
        "stage": "Step12C",
        "status": "complete",
        "H4-C": decisions["H4-C"],
        "H5-C": decisions["H5-C"],
        "classification_preservation": decisions[
            "classification_preservation"
        ],
        "official_test_accessed": False,
        "next_stage": "Step12D",
    }
    _save_json(args.output_root / "summary.json", summary)

    print("=" * 78)
    print("AEGIS v0.27 STEP12C FORMAL M4qc ANALYSIS")
    print("=" * 78)

    for seed_result in seed_results:
        seed = seed_result["seed"]
        print(
            f"Seed {seed}: "
            f"AUC={seed_result['h4_c']['roc_auc']:.12f} | "
            f"|ΔqT|={seed_result['h5_c']['mean_abs_text_quality_change']:.12f} | "
            f"|ΔqV|={seed_result['h5_c']['mean_abs_vision_quality_change']:.12f} | "
            f"Δc={seed_result['h5_c']['compatibility_drop']:.12f}"
        )

    h4 = decisions["H4-C"]
    h5 = decisions["H5-C"]
    cls = decisions["classification_preservation"]

    print()
    print(
        "H4-C mean ROC-AUC:",
        f"{h4['mean_roc_auc']:.12f}",
        "=>",
        h4["status"],
    )
    print(
        "H5-C mean |ΔqT|:",
        f"{h5['aggregate']['mean_abs_text_quality_change']:.12f}",
    )
    print(
        "H5-C mean |ΔqV|:",
        f"{h5['aggregate']['mean_abs_vision_quality_change']:.12f}",
    )
    print(
        "H5-C mean compatibility drop:",
        f"{h5['aggregate']['mean_compatibility_drop']:.12f}",
        "=>",
        h5["status"],
    )
    print(
        "Classification preservation:",
        cls["status"],
        f"(Δ={cls['delta_vs_m1b']:.12f})",
    )
    print("Official test accessed: FALSE")
    print("M4qcf computed: FALSE")
    print()
    print("Wrote:", args.output_root / "formal_analysis.json")
    print("Wrote:", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
