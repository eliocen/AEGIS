"""AEGIS v0.28 Step5 prospectively specified formal analyzer.

Freeze this implementation before applying it to the frozen Step4 record.
The analyzer performs no training, tuning, checkpoint selection, condition
selection, or official-test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VERSION = "0.28.0-step5"
EXPECTED_STEP4_SHA256 = "d27591e33fe82528682c00f73b2323b47c430896e71fae059b02c942ab32f8fa"
DEFAULT_PROTOCOL = Path("docs/experiments/v028/step1_selective_reliability_mismatch_protocol.json")
DEFAULT_INPUT = Path("docs/experiments/v028/step4_unified_evaluation_results.json")
DEFAULT_OUTPUT = Path("experiments/fakeddit/v028_formal_analysis")
ARCHITECTURES = ("M1b", "M4qc", "M4qcf", "M4qcs-w", "M4qcs-i", "M4qcs")
SEEDS = (42, 43, 44)
GRADED_CONDITIONS = tuple(
    f"{modality}_{family}_s{severity}"
    for modality in ("text", "vision")
    for family in ("gaussian_noise", "attenuation")
    for severity in ("0p25", "0p5", "0p75")
)
TEXT_GAUSSIAN_CONDITIONS = tuple(
    f"text_gaussian_noise_s{severity}" for severity in ("0p25", "0p5", "0p75", "1")
)
CATASTROPHIC_CONDITIONS = (
    "text_attenuation_s1",
    "vision_attenuation_s1",
    "text_zero_dropout_s1",
    "vision_zero_dropout_s1",
)
ALL_NONCLEAN_CONDITIONS = tuple(
    f"{modality}_{family}_s{severity}"
    for modality in ("text", "vision")
    for family in ("gaussian_noise", "attenuation")
    for severity in ("0p25", "0p5", "0p75", "1")
) + ("text_zero_dropout_s1", "vision_zero_dropout_s1")
OTHER_NONCLEAN_CONDITIONS = tuple(
    condition for condition in ALL_NONCLEAN_CONDITIONS if condition not in TEXT_GAUSSIAN_CONDITIONS
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite(value: Any, name: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"Non-finite {name}.")
    return result


def mean(values: Iterable[float]) -> float:
    items = [finite(value, "mean input") for value in values]
    require(bool(items), "Mean input is empty.")
    return float(statistics.fmean(items))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(gates: Mapping[str, bool]) -> str:
    require(bool(gates), "A formal decision requires at least one gate.")
    return "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED"


def quality_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    output: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["architecture"]), int(row["model_seed"]), str(row["condition"]))
        require(key not in output, f"Duplicate quality observation: {key}")
        output[key] = row
    require(len(output) == 342, "Quality matrix must contain exactly 342 unique observations.")
    return output


def macro_f1(index: Mapping[tuple[str, int, str], Mapping[str, Any]], architecture: str, seed: int, condition: str) -> float:
    key = (architecture, seed, condition)
    require(key in index, f"Missing quality observation: {key}")
    return finite(index[key]["metrics"]["macro_f1"], f"Macro-F1 for {key}")


def paired_deltas(index: Mapping[tuple[str, int, str], Mapping[str, Any]], primary: str, comparator: str, conditions: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "condition": condition,
            "primary_macro_f1": macro_f1(index, primary, seed, condition),
            "comparator_macro_f1": macro_f1(index, comparator, seed, condition),
            "delta": macro_f1(index, primary, seed, condition) - macro_f1(index, comparator, seed, condition),
        }
        for seed in SEEDS
        for condition in conditions
    ]


def mismatch_aggregate(rows: Sequence[Mapping[str, Any]], architecture: str) -> dict[str, Any]:
    selected = [row for row in rows if row["architecture"] == architecture]
    matched = {int(row["model_seed"]): row for row in selected if row["mismatch_state"] == "matched"}
    mismatched = {int(row["model_seed"]): row for row in selected if row["mismatch_state"] == "mismatched"}
    require(set(matched) == set(SEEDS), f"{architecture}: incomplete matched seed set.")
    require(set(mismatched) == set(SEEDS), f"{architecture}: incomplete mismatched seed set.")
    transitions = [mismatched[seed]["transition_metrics"] for seed in SEEDS]
    harmful = sum(int(item["harmful_flip_count"]) for item in transitions)
    beneficial = sum(int(item["beneficial_flip_count"]) for item in transitions)
    flips = sum(int(item["prediction_flip_count"]) for item in transitions)
    pairs = sum(int(item["sample_pairs"]) for item in transitions)
    require(beneficial > 0, f"{architecture}: harmful/beneficial ratio is undefined.")
    gaps = [
        finite(matched[seed]["metrics"]["macro_f1"], "matched Macro-F1")
        - finite(mismatched[seed]["metrics"]["macro_f1"], "mismatched Macro-F1")
        for seed in SEEDS
    ]
    return {
        "architecture": architecture,
        "mean_matched_macro_f1": mean(matched[seed]["metrics"]["macro_f1"] for seed in SEEDS),
        "mean_mismatched_macro_f1": mean(mismatched[seed]["metrics"]["macro_f1"] for seed in SEEDS),
        "per_seed_mismatch_gap": [{"seed": seed, "gap": gaps[position]} for position, seed in enumerate(SEEDS)],
        "mean_mismatch_gap": mean(gaps),
        "sample_pairs": pairs,
        "prediction_flip_count": flips,
        "prediction_flip_rate": flips / pairs,
        "harmful_flip_count": harmful,
        "beneficial_flip_count": beneficial,
        "net_harmful_flip_count": harmful - beneficial,
        "harmful_to_beneficial_ratio": harmful / beneficial,
    }


def diagnostic_is_constant(rows: Sequence[Mapping[str, Any]], architecture: str, name: str, expected: float, tolerance: float = 1e-12) -> bool:
    selected = [row for row in rows if row["architecture"] == architecture]
    require(bool(selected), f"No diagnostic rows for {architecture}.")
    for row in selected:
        diagnostic = row["diagnostics"].get(name)
        if not isinstance(diagnostic, Mapping):
            return False
        for key in ("min", "mean", "max"):
            if abs(finite(diagnostic[key], f"{architecture} {name} {key}") - expected) > tolerance:
                return False
    return True


def diagnostic_has_variation(rows: Sequence[Mapping[str, Any]], architecture: str, name: str, tolerance: float = 1e-12) -> bool:
    selected = [row for row in rows if row["architecture"] == architecture]
    require(bool(selected), f"No diagnostic rows for {architecture}.")
    return any(
        isinstance(row["diagnostics"].get(name), Mapping)
        and finite(row["diagnostics"][name]["max"], "diagnostic max")
        - finite(row["diagnostics"][name]["min"], "diagnostic min") > tolerance
        for row in selected
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol["protocol_version"] == "0.28.0-step1", "Wrong Step1 protocol version.")
    require(protocol["status"] == "FROZEN_PROSPECTIVE_PROTOCOL", "Step1 protocol is not frozen.")
    policy = protocol["formal_decision_policy"]
    require(policy["all_hypothesis_gates_conjunctive"] is True, "Formal gates are not conjunctive.")
    require(policy["aggregate_mean_may_override_failed_coverage_gate"] is False, "Mean override is prohibited.")
    hypotheses = protocol["formal_hypotheses"]
    require(set(hypotheses) == {"V28_H1", "V28_H2", "V28_H3", "V28_H4"}, "Wrong formal hypothesis set.")
    require(hypotheses["V28_H1"]["support_gates"]["graded_mean_macro_f1_improvement_min"] == 0.01, "H1 mean threshold changed.")
    require(hypotheses["V28_H1"]["support_gates"]["graded_minimum_positive_count_for_36"] == 26, "H1 coverage threshold changed.")
    require(hypotheses["V28_H1"]["support_gates"]["catastrophic_mean_delta_vs_M4qcf_min"] == -0.01, "H1 catastrophic threshold changed.")
    require(hypotheses["V28_H4"]["support_gates"]["text_gaussian_mean_macro_f1_delta_min"] == 0.01, "H4 mean threshold changed.")
    require(hypotheses["V28_H4"]["support_gates"]["text_gaussian_minimum_positive_count_for_12"] == 9, "H4 coverage threshold changed.")


def analyze(protocol: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    require(record["record_version"] == "0.28.0-step4-results", "Wrong Step4 record version.")
    require(record["formal_hypothesis_decisions"] == {key: "NOT_COMPUTED" for key in ("V28_H1", "V28_H2", "V28_H3", "V28_H4")}, "Step4 contains formal decisions.")
    quality_rows = record["all_342_quality_condition_summaries"]
    mismatch_rows = record["all_36_mismatch_summaries"]
    require(len(quality_rows) == 342, "Expected 342 quality rows.")
    require(len(mismatch_rows) == 36, "Expected 36 mismatch rows.")
    qindex = quality_index(quality_rows)

    h1_graded = paired_deltas(qindex, "M4qcs", "M1b", GRADED_CONDITIONS)
    h1_catastrophic = paired_deltas(qindex, "M4qcs", "M4qcf", CATASTROPHIC_CONDITIONS)
    h1_positive = sum(row["delta"] > 0.0 for row in h1_graded)
    h1_mean = mean(row["delta"] for row in h1_graded)
    h1_cat_mean = mean(row["delta"] for row in h1_catastrophic)
    h1_gates = {
        "graded_mean_macro_f1_improvement_at_least_0p01": h1_mean >= 0.01,
        "graded_positive_count_at_least_26_of_36": h1_positive >= 26,
        "graded_positive_fraction_at_least_0p70": h1_positive / 36 >= 0.70,
        "catastrophic_mean_delta_vs_M4qcf_at_least_minus_0p01": h1_cat_mean >= -0.01,
    }

    mismatch = {architecture: mismatch_aggregate(mismatch_rows, architecture) for architecture in ARCHITECTURES}
    primary = mismatch["M4qcs"]
    m1b = mismatch["M1b"]
    m4qcf = mismatch["M4qcf"]
    gap_improvement = m1b["mean_mismatch_gap"] - primary["mean_mismatch_gap"]
    h2_gates = {
        "harmful_to_beneficial_ratio_lower_than_M1b": primary["harmful_to_beneficial_ratio"] < m1b["harmful_to_beneficial_ratio"],
        "harmful_to_beneficial_ratio_lower_than_M4qcf": primary["harmful_to_beneficial_ratio"] < m4qcf["harmful_to_beneficial_ratio"],
        "net_harmful_flips_lower_than_M1b": primary["net_harmful_flip_count"] < m1b["net_harmful_flip_count"],
        "net_harmful_flips_lower_than_M4qcf": primary["net_harmful_flip_count"] < m4qcf["net_harmful_flip_count"],
        "mean_mismatch_gap_improvement_vs_M1b_strictly_positive": gap_improvement > 0.0,
        "total_flip_rate_not_above_M1b": primary["prediction_flip_rate"] <= m1b["prediction_flip_rate"],
    }

    invariant_rows = list(quality_rows) + list(mismatch_rows)
    controller_invariants = {
        "M4qcs_w_interaction_fixed_to_one": diagnostic_is_constant(invariant_rows, "M4qcs-w", "interaction_multiplier", 1.0),
        "M4qcs_w_weights_are_active": diagnostic_has_variation(invariant_rows, "M4qcs-w", "text_weight"),
        "M4qcs_i_text_weight_fixed_to_half": diagnostic_is_constant(invariant_rows, "M4qcs-i", "text_weight", 0.5),
        "M4qcs_i_vision_weight_fixed_to_half": diagnostic_is_constant(invariant_rows, "M4qcs-i", "vision_weight", 0.5),
        "M4qcs_i_interaction_is_active": diagnostic_has_variation(invariant_rows, "M4qcs-i", "interaction_multiplier"),
        "M4qcs_combined_weights_are_active": diagnostic_has_variation(invariant_rows, "M4qcs", "text_weight"),
        "M4qcs_combined_interaction_is_active": diagnostic_has_variation(invariant_rows, "M4qcs", "interaction_multiplier"),
    }
    controls_independent = all(controller_invariants.values())
    h3_gates = {
        "M4qcs_mismatch_macro_f1_at_least_M4qcs_i": primary["mean_mismatched_macro_f1"] >= mismatch["M4qcs-i"]["mean_mismatched_macro_f1"],
        "M4qcs_mismatch_macro_f1_at_least_M4qcs_w": primary["mean_mismatched_macro_f1"] >= mismatch["M4qcs-w"]["mean_mismatched_macro_f1"],
        "M4qcs_net_harmful_flips_lower_than_M4qcs_i": primary["net_harmful_flip_count"] < mismatch["M4qcs-i"]["net_harmful_flip_count"],
        "M4qcs_net_harmful_flips_lower_than_M4qcs_w": primary["net_harmful_flip_count"] < mismatch["M4qcs-w"]["net_harmful_flip_count"],
        "weight_and_interaction_controls_verified_independent": controls_independent,
    }

    h4_text = paired_deltas(qindex, "M4qcs", "M4qcf", TEXT_GAUSSIAN_CONDITIONS)
    h4_other = paired_deltas(qindex, "M4qcs", "M4qcf", OTHER_NONCLEAN_CONDITIONS)
    h4_clean = paired_deltas(qindex, "M4qcs", "M4qcf", ("clean",))
    h4_positive = sum(row["delta"] > 0.0 for row in h4_text)
    h4_text_mean = mean(row["delta"] for row in h4_text)
    h4_other_mean = mean(row["delta"] for row in h4_other)
    h4_clean_mean = mean(row["delta"] for row in h4_clean)
    h4_gates = {
        "text_gaussian_mean_macro_f1_delta_at_least_0p01": h4_text_mean >= 0.01,
        "text_gaussian_positive_count_at_least_9_of_12": h4_positive >= 9,
        "text_gaussian_positive_fraction_at_least_0p75": h4_positive / 12 >= 0.75,
        "other_conditions_mean_macro_f1_delta_at_least_minus_0p01": h4_other_mean >= -0.01,
        "clean_macro_f1_delta_vs_M4qcf_at_least_minus_0p01": h4_clean_mean >= -0.01,
    }

    clean_deltas = {
        comparator: mean(row["delta"] for row in paired_deltas(qindex, "M4qcs", comparator, ("clean",)))
        for comparator in ("M1b", "M4qcf")
    }
    clean_gates = {f"mean_clean_delta_vs_{comparator}_at_least_minus_0p01": value >= -0.01 for comparator, value in clean_deltas.items()}

    return {
        "analyzer_version": VERSION,
        "analysis_type": "prospectively_specified_formal_analysis",
        "formal_decisions": {
            "V28_H1": decision(h1_gates),
            "V28_H2": decision(h2_gates),
            "V28_H3": decision(h3_gates),
            "V28_H4": decision(h4_gates),
        },
        "V28_H1": {
            "decision": decision(h1_gates), "gates": h1_gates,
            "graded_observations": h1_graded, "graded_positive_count": h1_positive,
            "graded_positive_fraction": h1_positive / 36, "graded_mean_delta": h1_mean,
            "catastrophic_observations": h1_catastrophic, "catastrophic_mean_delta": h1_cat_mean,
        },
        "V28_H2": {
            "decision": decision(h2_gates), "gates": h2_gates,
            "architecture_aggregates": [mismatch[a] for a in ARCHITECTURES],
            "mean_mismatch_gap_improvement_vs_M1b": gap_improvement,
        },
        "V28_H3": {
            "decision": decision(h3_gates), "gates": h3_gates,
            "controller_invariants": controller_invariants,
            "architecture_aggregates": [mismatch[a] for a in ("M4qcs-w", "M4qcs-i", "M4qcs")],
        },
        "V28_H4": {
            "decision": decision(h4_gates), "gates": h4_gates,
            "text_gaussian_observations": h4_text, "text_gaussian_positive_count": h4_positive,
            "text_gaussian_positive_fraction": h4_positive / 12,
            "text_gaussian_mean_delta": h4_text_mean,
            "other_nonclean_observations": h4_other, "other_nonclean_mean_delta": h4_other_mean,
            "clean_observations": h4_clean, "clean_mean_delta": h4_clean_mean,
        },
        "clean_performance_safeguard": {
            "decision": "PASSED" if all(clean_gates.values()) else "FAILED",
            "gates": clean_gates, "mean_clean_deltas": clean_deltas,
            "failure_consequence": "M4qcs cannot be recommended as the v0.28 successor if this safeguard fails.",
        },
        "safety_audit": {
            "official_test_accessed": False,
            "official_test_samples_accessed": 0,
            "training_performed": False,
            "checkpoint_reselection_performed": False,
            "threshold_tuning_performed": False,
            "architecture_modification_performed": False,
            "conditions_dropped": False,
            "seeds_dropped": False,
            "exploratory_analysis_performed_before_formal_decisions": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AEGIS v0.28 Step5 frozen formal analyzer")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    require(args.protocol.is_file(), f"Missing protocol: {args.protocol}")
    require(args.input.is_file(), f"Missing Step4 record: {args.input}")
    require(sha256(args.input) == EXPECTED_STEP4_SHA256, "Frozen Step4 record SHA256 mismatch.")
    require(not args.output_root.exists(), f"Output root already exists: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    record = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(protocol, record)
    result.update({
        "experiment": "fakeddit_v028_formal_analysis",
        "status": "STEP5_FORMAL_ANALYSIS_COMPLETE",
        "input_step4_record": str(args.input),
        "input_step4_sha256": sha256(args.input),
        "input_protocol": str(args.protocol),
        "input_protocol_sha256": sha256(args.protocol),
    })
    summary = {
        "analyzer_version": VERSION,
        "status": result["status"],
        "formal_decisions": result["formal_decisions"],
        "clean_performance_safeguard": result["clean_performance_safeguard"]["decision"],
        "official_test_accessed": False,
        "official_test_samples_accessed": 0,
    }
    save_json(args.output_root / "formal_analysis.json", result)
    save_json(args.output_root / "summary.json", summary)
    print("=" * 78)
    print("AEGIS v0.28 STEP5 - FROZEN FORMAL ANALYSIS")
    print("=" * 78)
    for hypothesis, value in result["formal_decisions"].items():
        print(f"{hypothesis}: {value}")
    print("Clean safeguard:", result["clean_performance_safeguard"]["decision"])
    print("Official test samples accessed: 0")
    print("Output:", args.output_root)
    print("=" * 78)


if __name__ == "__main__":
    main()
