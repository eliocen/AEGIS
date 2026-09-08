from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ANALYZER_VERSION = "0.27.0-step15b"
INPUT_PROTOCOL_VERSION = "0.27.0-step15a"
EXPECTED_PROTOCOL_SHA256 = (
    "0c87544611beb989930e895fd9a2045bf32e6ea2e8bfff18810e91fe5211a77c"
)
EXPECTED_FORMAL_ANALYSIS_SHA256 = (
    "82179d5c7eaa60db4f5a188f0625087c2a85479edb78bb5f897dee8bf71dd2b4"
)
EXPECTED_FORMAL_SUMMARY_SHA256 = (
    "2c71db7d513157a580e2d902f12d2461a47e4f725544683b09bb95778a45decd"
)
EXPECTED_STEP14_ANALYZER_VERSION = "0.27.0-step14b"
EXPECTED_STEP14_PROTOCOL_VERSION = "0.27.0-step14a"
EXPECTED_OBSERVATIONS = 54
FORMAL_SEEDS = (42, 43, 44)
EXPECTED_DECISIONS = {
    "H8_F": "SUPPORTED",
    "H9_F": "NOT_SUPPORTED",
    "H10_F": "NOT_SUPPORTED",
}
FLOAT_TOLERANCE = 1e-12


class ExploratoryAnalysisError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ExploratoryAnalysisError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"required input missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read valid JSON from {path}: {exc}")
    require(isinstance(value, dict), f"{path}: top level must be an object")
    return value


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"required hash input missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: expected numeric value",
    )
    result = float(value)
    require(math.isfinite(result), f"{label}: expected finite value")
    return result


def close(left: Any, right: Any, label: str) -> None:
    lhs = finite_number(left, f"{label}/left")
    rhs = finite_number(right, f"{label}/right")
    require(math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE), label)


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(
        protocol.get("protocol_version") == INPUT_PROTOCOL_VERSION,
        "Step15A protocol version mismatch",
    )
    require(
        protocol.get("analysis_type") == "exploratory_post_hoc_characterization",
        "Step15A analysis type mismatch",
    )
    require(protocol.get("frozen_step14_decisions") == EXPECTED_DECISIONS,
            "Step15A frozen decisions mismatch")
    boundary = protocol.get("data_boundary")
    require(isinstance(boundary, dict), "Step15A data boundary missing")
    for key in (
        "official_test_access_permitted",
        "new_training_permitted",
        "checkpoint_reselection_permitted",
        "new_corruption_generation_permitted",
        "new_mismatch_generation_permitted",
    ):
        require(boundary.get(key) is False, f"Step15A boundary violation: {key}")
    spec = protocol.get("step15b_h9_decomposition")
    require(isinstance(spec, dict), "Step15B protocol section missing")
    require(
        spec.get("interpretation_rule")
        == "The previously frozen H9-F decision remains NOT_SUPPORTED regardless of any favorable subgroup.",
        "Step15B interpretation rule mismatch",
    )


def validate_frozen_inputs(
    formal: Mapping[str, Any], summary: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    require(formal.get("analyzer_version") == EXPECTED_STEP14_ANALYZER_VERSION,
            "formal analysis analyzer version mismatch")
    require(formal.get("input_protocol_version") == EXPECTED_STEP14_PROTOCOL_VERSION,
            "formal analysis protocol version mismatch")
    require(formal.get("formal_decisions") == EXPECTED_DECISIONS,
            "formal analysis decisions changed")
    require(summary.get("formal_decisions") == EXPECTED_DECISIONS,
            "formal summary decisions changed")
    require(formal.get("official_test_accessed") is False,
            "formal analysis reports official test access")
    require(formal.get("official_test_samples_accessed") == 0,
            "formal analysis reports official test samples")
    require(formal.get("training_performed") is False,
            "formal analysis reports training")
    require(formal.get("checkpoint_reselection_performed") is False,
            "formal analysis reports checkpoint reselection")
    require(formal.get("threshold_tuning_performed") is False,
            "formal analysis reports threshold tuning")
    h9 = formal.get("H9_F")
    require(isinstance(h9, dict), "formal H9_F section missing")
    require(h9.get("decision") == "NOT_SUPPORTED", "frozen H9-F decision changed")
    rows = h9.get("observations")
    require(isinstance(rows, list), "formal H9 observations missing")
    require(len(rows) == EXPECTED_OBSERVATIONS,
            f"expected 54 H9 observations, got {len(rows)}")
    result = [validate_observation(row, index) for index, row in enumerate(rows)]
    require(len({(r["seed"], r["condition"]) for r in result}) == 54,
            "duplicate seed-condition observation")
    require({r["seed"] for r in result} == set(FORMAL_SEEDS),
            "formal seed set mismatch")
    require(all(sum(r["seed"] == seed for r in result) == 18 for seed in FORMAL_SEEDS),
            "each formal seed must contain 18 observations")
    values = [r["R_i"] for r in result]
    positive = sum(v > 0.0 for v in values)
    require(positive == h9.get("strict_positive_count") == 29,
            "frozen H9 positive count mismatch")
    close(statistics.fmean(values), h9.get("mean_macro_f1_improvement"),
          "frozen H9 mean mismatch")
    return result


def validate_observation(row: Any, index: int) -> Dict[str, Any]:
    label = f"H9 observation {index}"
    require(isinstance(row, dict), f"{label}: expected object")
    seed = row.get("seed")
    require(seed in FORMAL_SEEDS, f"{label}: unexpected seed")
    condition = row.get("condition")
    modality = row.get("corrupted_modality")
    family = row.get("corruption_type")
    require(isinstance(condition, str) and condition, f"{label}: condition missing")
    require(modality in {"text", "vision"}, f"{label}: modality invalid")
    require(isinstance(family, str) and family, f"{label}: family missing")
    severity = finite_number(row.get("severity"), f"{label}/severity")
    m1b = finite_number(row.get("m1b_macro_f1"), f"{label}/m1b")
    m4qc = finite_number(row.get("m4qc_macro_f1"), f"{label}/m4qc")
    m4qcf = finite_number(row.get("m4qcf_macro_f1"), f"{label}/m4qcf")
    r_i = finite_number(row.get("m4qcf_minus_m1b"), f"{label}/R_i")
    close(m4qcf - m1b, r_i, f"{label}: stored R_i mismatch")
    require(row.get("strict_positive") is (r_i > 0.0),
            f"{label}: strict-positive flag mismatch")
    return {
        "seed": seed,
        "condition": condition,
        "corrupted_modality": modality,
        "corruption_family": family,
        "severity": severity,
        "m1b_macro_f1": m1b,
        "m4qc_macro_f1": m4qc,
        "m4qcf_macro_f1": m4qcf,
        "R_i": r_i,
        "sign": "positive" if r_i > 0.0 else "negative" if r_i < 0.0 else "zero",
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(bool(rows), "cannot summarize an empty group")
    values = [finite_number(row["R_i"], "group R_i") for row in rows]
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    zero = len(values) - positive - negative
    return {
        "count": len(values),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_proportion": positive / len(values),
        "mean_R_i": statistics.fmean(values),
        "median_R_i": statistics.median(values),
        "min_R_i": min(values),
        "max_R_i": max(values),
    }


def stratify(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        group = {field: value for field, value in zip(fields, key)}
        group.update(summarize_rows(groups[key]))
        output.append(group)
    return output


def severity_trajectories(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["corrupted_modality"]), str(row["corruption_family"]))].append(row)
    trajectories = []
    for modality, family in sorted(groups):
        points = stratify(groups[(modality, family)], ("severity",))
        points.sort(key=lambda item: item["severity"])
        trajectories.append({
            "corrupted_modality": modality,
            "corruption_family": family,
            "points": points,
        })
    return trajectories


def rank_observations(
    rows: Iterable[Mapping[str, Any]], *, reverse: bool
) -> List[Dict[str, Any]]:
    return [dict(row) for row in sorted(
        rows,
        key=lambda row: (
            -row["R_i"] if reverse else row["R_i"],
            row["seed"], row["condition"],
        ),
    )]


def analyze_h9_failure_decomposition(
    protocol_path: Path,
    formal_analysis_path: Path,
    formal_summary_path: Path,
    output_root: Path,
    *,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    protocol_hash = verify_sha256(
        protocol_path, EXPECTED_PROTOCOL_SHA256, "Step15A protocol"
    )
    formal_hash = verify_sha256(
        formal_analysis_path, EXPECTED_FORMAL_ANALYSIS_SHA256, "formal analysis"
    )
    summary_hash = verify_sha256(
        formal_summary_path, EXPECTED_FORMAL_SUMMARY_SHA256, "formal summary"
    )
    protocol = read_json(protocol_path)
    formal = read_json(formal_analysis_path)
    summary = read_json(formal_summary_path)
    validate_protocol(protocol)
    rows = validate_frozen_inputs(formal, summary)
    rows = sorted(rows, key=lambda row: (row["seed"], row["condition"]))

    stratifications = {
        "seed": stratify(rows, ("seed",)),
        "corrupted_modality": stratify(rows, ("corrupted_modality",)),
        "corruption_family": stratify(rows, ("corruption_family",)),
        "severity": stratify(rows, ("severity",)),
        "corrupted_modality_x_corruption_family": stratify(
            rows, ("corrupted_modality", "corruption_family")
        ),
        "corruption_family_x_severity": stratify(
            rows, ("corruption_family", "severity")
        ),
        "corrupted_modality_x_severity": stratify(
            rows, ("corrupted_modality", "severity")
        ),
    }
    overall = summarize_rows(rows)
    require(overall["positive_count"] == 29, "Step15B positive count drift")
    require(overall["negative_count"] + overall["zero_count"] == 25,
            "Step15B nonpositive count drift")

    analysis = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "analysis_type": "exploratory_post_hoc_characterization",
        "status": "STEP15B_H9_FAILURE_DECOMPOSITION_COMPLETE",
        "experiment": "fakeddit_h9_failure_decomposition_v027",
        "frozen_formal_decisions": dict(EXPECTED_DECISIONS),
        "H9_F_decision_preserved": "NOT_SUPPORTED",
        "interpretation_rule": (
            "The frozen H9-F decision remains NOT_SUPPORTED regardless of any favorable subgroup."
        ),
        "input_sha256": {
            "step15a_protocol_json": protocol_hash,
            "formal_analysis_json": formal_hash,
            "formal_summary_json": summary_hash,
        },
        "safety_audit": {
            "official_test_accessed": False,
            "official_test_samples_accessed": 0,
            "training_performed": False,
            "checkpoint_reselection_performed": False,
            "threshold_tuning_performed": False,
            "architecture_modification_performed": False,
            "new_corruption_generation_performed": False,
            "new_mismatch_generation_performed": False,
            "conditions_dropped": False,
            "seeds_dropped": False,
            "confirmatory_claim_made": False,
        },
        "primary_quantity": "R_i = MacroF1_M4qcf_i - MacroF1_M1b_i",
        "overall": overall,
        "all_54_seed_condition_R_i_values": rows,
        "stratifications": stratifications,
        "severity_trajectories": severity_trajectories(rows),
        "largest_positive_observations": rank_observations(
            (row for row in rows if row["R_i"] > 0.0), reverse=True
        ),
        "largest_negative_observations": rank_observations(
            (row for row in rows if row["R_i"] < 0.0), reverse=False
        ),
        "zero_observations": rank_observations(
            (row for row in rows if row["R_i"] == 0.0), reverse=False
        ),
        "interpretation_boundary": protocol["interpretation_boundary"],
    }

    compact_summary = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "status": analysis["status"],
        "analysis_type": analysis["analysis_type"],
        "H9_F_decision_preserved": "NOT_SUPPORTED",
        "observation_count": overall["count"],
        "positive_count": overall["positive_count"],
        "negative_count": overall["negative_count"],
        "zero_count": overall["zero_count"],
        "positive_proportion": overall["positive_proportion"],
        "mean_R_i": overall["mean_R_i"],
        "median_R_i": overall["median_R_i"],
        "min_R_i": overall["min_R_i"],
        "max_R_i": overall["max_R_i"],
        "stratification_count": len(stratifications),
        "input_sha256": analysis["input_sha256"],
        "safety_audit": analysis["safety_audit"],
    }

    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "h9_failure_decomposition.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_root / "summary.json").write_text(
            json.dumps(compact_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return analysis


def print_result(analysis: Mapping[str, Any], output_root: Path) -> None:
    overall = analysis["overall"]
    print("=" * 78)
    print("AEGIS v0.27 STEP15B - EXPLORATORY H9 FAILURE DECOMPOSITION")
    print("=" * 78)
    print()
    print("H9-F frozen decision: NOT_SUPPORTED (UNCHANGED)")
    print(
        f"R_i observations: {overall['count']} | positive={overall['positive_count']} "
        f"| negative={overall['negative_count']} | zero={overall['zero_count']}"
    )
    print(
        f"mean={overall['mean_R_i']:.12f} | median={overall['median_R_i']:.12f} "
        f"| min={overall['min_R_i']:.12f} | max={overall['max_R_i']:.12f}"
    )
    print()
    print("Official test accessed: NO")
    print("Training performed: NO")
    print("Checkpoint reselection: NO")
    print("Tuning performed: NO")
    print("Architecture modified: NO")
    print("Analysis status: EXPLORATORY / POST-HOC")
    print("Output:", output_root)
    print()
    print("=" * 78)
    print("STEP15B_H9_FAILURE_DECOMPOSITION_COMPLETE")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory Step15B decomposition of the frozen v0.27 H9-F result."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"),
    )
    parser.add_argument(
        "--formal-analysis",
        type=Path,
        default=Path("experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"),
    )
    parser.add_argument(
        "--formal-summary",
        type=Path,
        default=Path("experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/fakeddit/v027_h9_failure_decomposition"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analysis = analyze_h9_failure_decomposition(
            args.protocol,
            args.formal_analysis,
            args.formal_summary,
            args.output_root,
        )
    except ExploratoryAnalysisError as exc:
        print("=" * 78)
        print("STEP15B EXPLORATORY ANALYSIS FAILED LOUDLY")
        print("=" * 78)
        print(str(exc))
        return 1
    print_result(analysis, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
