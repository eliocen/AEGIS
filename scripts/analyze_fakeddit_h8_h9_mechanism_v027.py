from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ANALYZER_VERSION = "0.27.0-step15c"
INPUT_PROTOCOL_VERSION = "0.27.0-step15a"
EXPECTED_PROTOCOL_SHA256 = "0c87544611beb989930e895fd9a2045bf32e6ea2e8bfff18810e91fe5211a77c"
EXPECTED_FORMAL_ANALYSIS_SHA256 = "82179d5c7eaa60db4f5a188f0625087c2a85479edb78bb5f897dee8bf71dd2b4"
EXPECTED_FORMAL_SUMMARY_SHA256 = "2c71db7d513157a580e2d902f12d2461a47e4f725544683b09bb95778a45decd"
EXPECTED_OBSERVATIONS = 54
FORMAL_SEEDS = (42, 43, 44)
EXPECTED_DECISIONS = {
    "H8_F": "SUPPORTED",
    "H9_F": "NOT_SUPPORTED",
    "H10_F": "NOT_SUPPORTED",
}
ABS_TOL = 1e-12


class MechanismAnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MechanismAnalysisError(message)


def finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label}: expected number")
    result = float(value)
    require(math.isfinite(result), f"{label}: expected finite number")
    return result


def read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing required input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MechanismAnalysisError(f"invalid JSON at {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: top level must be an object")
    return value


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing hash input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    require(actual == expected, f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    require(len(x) == len(y), "correlation length mismatch")
    require(len(x) >= 2, "correlation requires at least two observations")
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    ss_x = sum(value * value for value in centered_x)
    ss_y = sum(value * value for value in centered_y)
    if ss_x == 0.0 or ss_y == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(ss_x * ss_y)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return pearson(average_ranks(x), average_ranks(y))


def association(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(bool(rows), "cannot analyze empty association group")
    d_values = [finite(row["D_m"], "D_m") for row in rows]
    r_values = [finite(row["R_i"], "R_i") for row in rows]
    pearson_r = pearson(d_values, r_values) if len(rows) >= 2 else None
    spearman_rho = spearman(d_values, r_values) if len(rows) >= 2 else None
    return {
        "count": len(rows),
        "mean_D_m": statistics.fmean(d_values),
        "median_D_m": statistics.median(d_values),
        "min_D_m": min(d_values),
        "max_D_m": max(d_values),
        "mean_R_i": statistics.fmean(r_values),
        "median_R_i": statistics.median(r_values),
        "min_R_i": min(r_values),
        "max_R_i": max(r_values),
        "R_i_positive_count": sum(value > 0.0 for value in r_values),
        "R_i_nonpositive_count": sum(value <= 0.0 for value in r_values),
        "pearson_r_descriptive": pearson_r,
        "spearman_rho_descriptive": spearman_rho,
        "correlation_defined": pearson_r is not None and spearman_rho is not None,
        "significance_test_performed": False,
        "causal_claim_permitted": False,
    }


def stratify(rows: Sequence[Mapping[str, Any]], field: str) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    output = []
    for key in sorted(groups, key=lambda value: (isinstance(value, str), str(value))):
        item = {field: key}
        item.update(association(groups[key]))
        output.append(item)
    return output


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol.get("protocol_version") == INPUT_PROTOCOL_VERSION,
            "Step15A protocol version mismatch")
    require(protocol.get("analysis_type") == "exploratory_post_hoc_characterization",
            "Step15A analysis type mismatch")
    require(protocol.get("frozen_step14_decisions") == EXPECTED_DECISIONS,
            "frozen decision mismatch")
    spec = protocol.get("step15c_h8_to_h9_mechanism")
    require(isinstance(spec, dict), "Step15C protocol section missing")
    require(spec.get("causal_claim_permitted") is False,
            "Step15C causal boundary mismatch")
    require(spec.get("significance_claim_permitted") is False,
            "Step15C significance boundary mismatch")


def validate_formal(formal: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    require(formal.get("formal_decisions") == EXPECTED_DECISIONS,
            "formal decisions changed")
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
    require(formal.get("H8_F", {}).get("decision") == "SUPPORTED",
            "H8-F decision changed")
    require(formal.get("H9_F", {}).get("decision") == "NOT_SUPPORTED",
            "H9-F decision changed")


def pair_observations(formal: Mapping[str, Any]) -> List[Dict[str, Any]]:
    h8_rows = formal["H8_F"].get("observations")
    h9_rows = formal["H9_F"].get("observations")
    require(isinstance(h8_rows, list) and len(h8_rows) == EXPECTED_OBSERVATIONS,
            "expected 54 H8 observations")
    require(isinstance(h9_rows, list) and len(h9_rows) == EXPECTED_OBSERVATIONS,
            "expected 54 H9 observations")
    h8_by_key = {(row.get("seed"), row.get("condition")): row for row in h8_rows}
    h9_by_key = {(row.get("seed"), row.get("condition")): row for row in h9_rows}
    require(len(h8_by_key) == 54 and len(h9_by_key) == 54,
            "duplicate seed-condition observations")
    require(set(h8_by_key) == set(h9_by_key), "H8/H9 pairing keys differ")
    paired = []
    for seed, condition in sorted(h8_by_key):
        h8 = h8_by_key[(seed, condition)]
        h9 = h9_by_key[(seed, condition)]
        for field in ("corrupted_modality", "corruption_type", "severity"):
            require(h8.get(field) == h9.get(field),
                    f"pair metadata mismatch for seed {seed} {condition}: {field}")
        d_m = finite(h8.get("down_weighting"), "H8 down_weighting")
        r_i = finite(h9.get("m4qcf_minus_m1b"), "H9 R_i")
        require(d_m > 0.0, "frozen H8 observation is not strictly positive")
        require(h8.get("strict_positive") is True, "H8 positive flag mismatch")
        require(h9.get("strict_positive") is (r_i > 0.0), "H9 positive flag mismatch")
        paired.append({
            "seed": seed,
            "condition": condition,
            "corrupted_modality": h8["corrupted_modality"],
            "corruption_family": h8["corruption_type"],
            "severity": finite(h8["severity"], "severity"),
            "clean_corrupted_modality_weight": finite(h8["clean_weight"], "clean weight"),
            "corrupted_modality_weight": finite(h8["corrupted_weight"], "corrupted weight"),
            "D_m": d_m,
            "m1b_macro_f1": finite(h9["m1b_macro_f1"], "M1b Macro-F1"),
            "m4qcf_macro_f1": finite(h9["m4qcf_macro_f1"], "M4qcf Macro-F1"),
            "R_i": r_i,
            "quadrant": (
                "D_m_positive_R_i_positive" if r_i > 0.0
                else "D_m_positive_R_i_nonpositive"
            ),
        })
    return paired


def characterize_quadrants(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    names = (
        "D_m_positive_R_i_positive",
        "D_m_positive_R_i_nonpositive",
        "D_m_nonpositive_R_i_positive",
        "D_m_nonpositive_R_i_nonpositive",
    )
    output = []
    for name in names:
        selected = [row for row in rows if row["quadrant"] == name]
        item: Dict[str, Any] = {
            "quadrant": name,
            "count": len(selected),
            "proportion": len(selected) / len(rows),
        }
        if selected:
            item.update(association(selected))
        else:
            item.update({
                "mean_D_m": None,
                "mean_R_i": None,
                "correlation_defined": False,
                "significance_test_performed": False,
                "causal_claim_permitted": False,
            })
        output.append(item)
    return output


def analyze(
    protocol_path: Path,
    formal_path: Path,
    summary_path: Path,
    output_root: Path,
    *,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    hashes = {
        "step15a_protocol_json": verify_hash(
            protocol_path, EXPECTED_PROTOCOL_SHA256, "Step15A protocol"
        ),
        "formal_analysis_json": verify_hash(
            formal_path, EXPECTED_FORMAL_ANALYSIS_SHA256, "formal analysis"
        ),
        "formal_summary_json": verify_hash(
            summary_path, EXPECTED_FORMAL_SUMMARY_SHA256, "formal summary"
        ),
    }
    protocol = read_json(protocol_path)
    formal = read_json(formal_path)
    summary = read_json(summary_path)
    validate_protocol(protocol)
    validate_formal(formal, summary)
    rows = pair_observations(formal)
    require(len(rows) == 54, "paired observation count mismatch")
    overall = association(rows)
    quadrants = characterize_quadrants(rows)
    quadrant_counts = {item["quadrant"]: item["count"] for item in quadrants}
    require(quadrant_counts == {
        "D_m_positive_R_i_positive": 29,
        "D_m_positive_R_i_nonpositive": 25,
        "D_m_nonpositive_R_i_positive": 0,
        "D_m_nonpositive_R_i_nonpositive": 0,
    }, "quadrant count drift")

    analysis = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "analysis_type": "exploratory_post_hoc_mechanism_characterization",
        "status": "STEP15C_H8_TO_H9_MECHANISM_CHARACTERIZATION_COMPLETE",
        "experiment": "fakeddit_h8_to_h9_mechanism_characterization_v027",
        "frozen_formal_decisions": dict(EXPECTED_DECISIONS),
        "H8_F_decision_preserved": "SUPPORTED",
        "H9_F_decision_preserved": "NOT_SUPPORTED",
        "variables": {
            "D_m": "clean minus corrupted mean fusion weight for the corrupted modality",
            "R_i": "MacroF1_M4qcf minus MacroF1_M1b for the same seed-condition observation",
        },
        "input_sha256": hashes,
        "paired_D_m_and_R_i_table_for_all_54_observations": rows,
        "overall_descriptive_association": overall,
        "grouped_association_by_modality": stratify(rows, "corrupted_modality"),
        "grouped_association_by_corruption_family": stratify(rows, "corruption_family"),
        "grouped_association_by_severity": stratify(rows, "severity"),
        "quadrant_characterization": quadrants,
        "interpretation_rule": protocol["step15c_h8_to_h9_mechanism"]["interpretation_rule"],
        "safety_audit": {
            "official_test_accessed": False,
            "official_test_samples_accessed": 0,
            "training_performed": False,
            "checkpoint_reselection_performed": False,
            "threshold_tuning_performed": False,
            "architecture_modification_performed": False,
            "new_corruption_generation_performed": False,
            "conditions_dropped": False,
            "seeds_dropped": False,
            "significance_test_performed": False,
            "causal_claim_made": False,
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    compact = {
        "analyzer_version": ANALYZER_VERSION,
        "status": analysis["status"],
        "analysis_type": analysis["analysis_type"],
        "H8_F_decision_preserved": "SUPPORTED",
        "H9_F_decision_preserved": "NOT_SUPPORTED",
        "paired_observation_count": 54,
        "pearson_r_descriptive": overall["pearson_r_descriptive"],
        "spearman_rho_descriptive": overall["spearman_rho_descriptive"],
        "quadrant_counts": quadrant_counts,
        "significance_test_performed": False,
        "causal_claim_made": False,
        "input_sha256": hashes,
        "safety_audit": analysis["safety_audit"],
    }
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "mechanism_characterization.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_root / "summary.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return analysis


def print_result(result: Mapping[str, Any], output_root: Path) -> None:
    overall = result["overall_descriptive_association"]
    quadrants = {row["quadrant"]: row["count"] for row in result["quadrant_characterization"]}
    print("=" * 78)
    print("AEGIS v0.27 STEP15C - EXPLORATORY H8-TO-H9 MECHANISM CHARACTERIZATION")
    print("=" * 78)
    print()
    print("H8-F frozen decision: SUPPORTED (UNCHANGED)")
    print("H9-F frozen decision: NOT_SUPPORTED (UNCHANGED)")
    print("Paired observations: 54")
    print(f"Pearson r (descriptive): {overall['pearson_r_descriptive']:.12f}")
    print(f"Spearman rho (descriptive): {overall['spearman_rho_descriptive']:.12f}")
    print("D_m>0, R_i>0:", quadrants["D_m_positive_R_i_positive"])
    print("D_m>0, R_i<=0:", quadrants["D_m_positive_R_i_nonpositive"])
    print()
    print("Significance test performed: NO")
    print("Causal claim made: NO")
    print("Official test accessed: NO")
    print("Output:", output_root)
    print()
    print("=" * 78)
    print("STEP15C_H8_TO_H9_MECHANISM_CHARACTERIZATION_COMPLETE")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory Step15C association of frozen H8 D_m and H9 R_i."
    )
    parser.add_argument("--protocol", type=Path, default=Path(
        "docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"))
    parser.add_argument("--formal-analysis", type=Path, default=Path(
        "experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"))
    parser.add_argument("--formal-summary", type=Path, default=Path(
        "experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "experiments/fakeddit/v027_h8_h9_mechanism_characterization"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze(
            args.protocol, args.formal_analysis, args.formal_summary, args.output_root
        )
    except MechanismAnalysisError as exc:
        print("=" * 78)
        print("STEP15C EXPLORATORY ANALYSIS FAILED LOUDLY")
        print("=" * 78)
        print(str(exc))
        return 1
    print_result(result, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
