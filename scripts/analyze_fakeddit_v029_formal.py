"""AEGIS v0.29 Step5 frozen formal analyzer.

Consumes only the frozen Step4 validation-development evidence and applies the
prospectively frozen v0.29 gates plus the Step4A cardinality clarification.
No training, checkpoint selection, tuning, or official-test access occurs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

VERSION = "0.29.0-step5"
EXPECTED_PROTOCOL_VERSION = "0.29.0-step1"
EXPECTED_EVALUATION_VERSION = "0.29.0-step4"
ARCHITECTURES = ("M1b", "M4qcf", "M4qcs-w", "M4qcs", "M4qgr", "M4qtc", "M4qgrt")
SEEDS = (42, 43, 44)

GRADED_CONDITIONS = (
    "text_gaussian_noise_s0p25", "text_gaussian_noise_s0p5", "text_gaussian_noise_s0p75",
    "vision_gaussian_noise_s0p25", "vision_gaussian_noise_s0p5", "vision_gaussian_noise_s0p75",
    "text_attenuation_s0p25", "text_attenuation_s0p5", "text_attenuation_s0p75",
    "vision_attenuation_s0p25", "vision_attenuation_s0p5", "vision_attenuation_s0p75",
)
CATASTROPHIC_CONDITIONS = (
    "text_attenuation_s1", "vision_attenuation_s1",
    "text_zero_dropout_s1", "vision_zero_dropout_s1",
)
TEXT_GAUSSIAN_CONDITIONS = (
    "text_gaussian_noise_s0p25", "text_gaussian_noise_s0p5", "text_gaussian_noise_s0p75",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=Path("docs/experiments/v029/step1_graded_reliability_transition_protocol.json"))
    p.add_argument("--step4-operationalization", type=Path, default=Path("docs/experiments/v029/step4_unified_evaluation_operationalization.json"))
    p.add_argument("--evaluation-root", type=Path, default=Path("experiments/fakeddit/v029_unified_evaluation"))
    p.add_argument("--output-json", type=Path, default=Path("docs/experiments/v029/step5_formal_analysis_results.json"))
    p.add_argument("--output-md", type=Path, default=Path("docs/experiments/v029/step5_formal_analysis_results.md"))
    return p.parse_args()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def f(x: Any, label: str) -> float:
    value = float(x)
    require(math.isfinite(value), f"Non-finite {label}")
    return value


def quality_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    out = {}
    for row in rows:
        key = (str(row["architecture"]), int(row["model_seed"]), str(row["condition"]))
        require(key not in out, f"Duplicate quality row {key}")
        out[key] = row
    return out


def mismatch_index(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    out = {}
    for row in rows:
        state = row.get("mismatch_state")
        key = (str(row["architecture"]), int(row["model_seed"]), str(state))
        require(key not in out, f"Duplicate mismatch row {key}")
        out[key] = row
    return out


def macro(row: Mapping[str, Any]) -> float:
    return f(row["metrics"]["macro_f1"], "macro_f1")


def paired_deltas(qi, primary: str, comparator: str, conditions: Sequence[str]) -> List[float]:
    vals = []
    for seed in SEEDS:
        for condition in conditions:
            vals.append(macro(qi[(primary, seed, condition)]) - macro(qi[(comparator, seed, condition)]))
    return vals


def mean_quality(qi, arch: str, conditions: Sequence[str]) -> float:
    vals = [macro(qi[(arch, seed, c)]) for seed in SEEDS for c in conditions]
    return statistics.fmean(vals)


def mismatch_aggregate(mi, arch: str) -> Dict[str, Any]:
    matched = []
    mismatched = []
    gaps = []
    flips = harmful = beneficial = pairs = 0
    seed_rows = []
    for seed in SEEDS:
        ma = mi[(arch, seed, "matched")]
        mm = mi[(arch, seed, "mismatched")]
        m1 = macro(ma); m2 = macro(mm)
        t = mm.get("transition_metrics") or {}
        sf = int(t["prediction_flip_count"])
        sh = int(t["harmful_flip_count"])
        sb = int(t["beneficial_flip_count"])
        sp = int(t["sample_pairs"])
        matched.append(m1); mismatched.append(m2); gaps.append(m1-m2)
        flips += sf; harmful += sh; beneficial += sb; pairs += sp
        seed_rows.append({"seed": seed, "matched_macro_f1": m1, "mismatched_macro_f1": m2,
                          "mismatch_gap": m1-m2, "prediction_flip_count": sf,
                          "harmful_flip_count": sh, "beneficial_flip_count": sb,
                          "net_harmful_flips": sh-sb})
    if beneficial > 0:
        ratio = harmful / beneficial
        ratio_repr: Any = ratio
    elif harmful == 0:
        ratio = 0.0
        ratio_repr = 0.0
    else:
        ratio = math.inf
        ratio_repr = "Infinity"
    return {
        "mean_matched_macro_f1": statistics.fmean(matched),
        "mean_mismatched_macro_f1": statistics.fmean(mismatched),
        "mean_mismatch_gap": statistics.fmean(gaps),
        "prediction_flip_count": flips,
        "prediction_flip_rate": flips / pairs,
        "harmful_flip_count": harmful,
        "beneficial_flip_count": beneficial,
        "net_harmful_flips": harmful-beneficial,
        "harmful_to_beneficial_ratio": ratio_repr,
        "_ratio_numeric": ratio,
        "sample_pairs": pairs,
        "seed_results": seed_rows,
    }


def gate(value: bool, observed: Any, criterion: str) -> Dict[str, Any]:
    return {"passed": bool(value), "observed": observed, "criterion": criterion}


def decide(gates: Mapping[str, Mapping[str, Any]]) -> str:
    return "SUPPORTED" if all(bool(x["passed"]) for x in gates.values()) else "NOT_SUPPORTED"


def main() -> None:
    a = parse_args()
    protocol = load_json(a.protocol)
    op = load_json(a.step4_operationalization)
    summary = load_json(a.evaluation_root / "summary.json")
    qdoc = load_json(a.evaluation_root / "quality_condition_summaries.json")
    mdoc = load_json(a.evaluation_root / "mismatch_summaries.json")

    require(protocol["protocol_version"] == EXPECTED_PROTOCOL_VERSION, "Protocol version mismatch")
    require(op["status"] == "FROZEN_BEFORE_UNIFIED_V029_EVALUATION", "Step4 operationalization not frozen")
    require(op["formal_analysis_contract"]["h4_text_gaussian_observation_count"] == 9,
            "Step4A H4 cardinality clarification unavailable")
    require(summary["protocol_version"] == EXPECTED_EVALUATION_VERSION, "Evaluation version mismatch")
    require(summary["status"] == "STEP4_DIAGNOSTIC_EVIDENCE_COMPLETE", "Formal Step4 evaluation incomplete")
    require(summary["quality_summary_rows"] == 399, "Expected 399 quality summaries")
    require(summary["mismatch_summary_rows"] == 42, "Expected 42 mismatch summaries")
    require(summary["architectures"] == list(ARCHITECTURES), "Architecture order mismatch")
    require(summary["seeds"] == list(SEEDS), "Seed matrix mismatch")
    require(summary["official_test_accessed"] is False and summary["official_test_samples_accessed"] == 0,
            "Official test boundary violated")
    require(summary["training_performed"] is False, "Step4 performed training")
    require(summary["checkpoint_reselection_performed"] is False, "Checkpoint reselection occurred")
    require(summary["threshold_tuning_performed"] is False, "Threshold tuning occurred")

    qi = quality_index(qdoc["rows"])
    mi = mismatch_index(mdoc["rows"])
    require(len(qi) == 399, "Quality index cardinality mismatch")
    require(len(mi) == 42, "Mismatch index cardinality mismatch")

    all_conditions = list(protocol["evaluation_protocol"]["quality_conditions"])
    require(len(all_conditions) == 19, "Frozen condition count mismatch")
    other_nonclean = [c for c in all_conditions if c != "clean" and c not in TEXT_GAUSSIAN_CONDITIONS]

    # H1
    h1_grade = paired_deltas(qi, "M4qgrt", "M4qcs", GRADED_CONDITIONS)
    h1_cat = paired_deltas(qi, "M4qgrt", "M4qcf", CATASTROPHIC_CONDITIONS)
    h1_pos = sum(x > 0 for x in h1_grade)
    h1_gates = {
        "graded_mean_delta": gate(statistics.fmean(h1_grade) >= 0.01, statistics.fmean(h1_grade), ">=0.01"),
        "graded_positive_count": gate(h1_pos >= 26, h1_pos, ">=26 of 36"),
        "graded_positive_fraction": gate(h1_pos/len(h1_grade) >= 0.70, h1_pos/len(h1_grade), ">=0.70"),
        "catastrophic_mean_delta": gate(statistics.fmean(h1_cat) >= -0.01, statistics.fmean(h1_cat), ">=-0.01 across 12"),
    }

    # Mismatch aggregates once for H2/H3.
    mm = {arch: mismatch_aggregate(mi, arch) for arch in ("M1b", "M4qcs-w", "M4qgr", "M4qtc", "M4qgrt")}
    p = mm["M4qgrt"]
    h2_gates = {
        "mean_gap_vs_M1b": gate(p["mean_mismatch_gap"] < mm["M1b"]["mean_mismatch_gap"],
                                p["mean_mismatch_gap"] - mm["M1b"]["mean_mismatch_gap"], "M4qgrt gap strictly lower"),
        "mean_gap_vs_M4qcs_w": gate(p["mean_mismatch_gap"] < mm["M4qcs-w"]["mean_mismatch_gap"],
                                    p["mean_mismatch_gap"] - mm["M4qcs-w"]["mean_mismatch_gap"], "M4qgrt gap strictly lower"),
        "ratio_vs_M1b": gate(p["_ratio_numeric"] < mm["M1b"]["_ratio_numeric"],
                              {"M4qgrt": p["harmful_to_beneficial_ratio"], "M1b": mm["M1b"]["harmful_to_beneficial_ratio"]}, "strictly lower"),
        "ratio_vs_M4qcs_w": gate(p["_ratio_numeric"] < mm["M4qcs-w"]["_ratio_numeric"],
                                  {"M4qgrt": p["harmful_to_beneficial_ratio"], "M4qcs-w": mm["M4qcs-w"]["harmful_to_beneficial_ratio"]}, "strictly lower"),
        "net_harmful_vs_M1b": gate(p["net_harmful_flips"] < mm["M1b"]["net_harmful_flips"],
                                    {"M4qgrt": p["net_harmful_flips"], "M1b": mm["M1b"]["net_harmful_flips"]}, "strictly lower"),
        "net_harmful_vs_M4qcs_w": gate(p["net_harmful_flips"] < mm["M4qcs-w"]["net_harmful_flips"],
                                        {"M4qgrt": p["net_harmful_flips"], "M4qcs-w": mm["M4qcs-w"]["net_harmful_flips"]}, "strictly lower"),
        "flip_rate_vs_M1b": gate(p["prediction_flip_rate"] <= mm["M1b"]["prediction_flip_rate"],
                                  {"M4qgrt": p["prediction_flip_rate"], "M1b": mm["M1b"]["prediction_flip_rate"]}, "<= M1b"),
    }

    # H3
    graded_means = {arch: mean_quality(qi, arch, GRADED_CONDITIONS) for arch in ("M4qgr", "M4qtc", "M4qgrt")}
    h3_gates = {
        "graded_above_M4qgr": gate(graded_means["M4qgrt"] > graded_means["M4qgr"],
                                    graded_means["M4qgrt"]-graded_means["M4qgr"], "strictly above"),
        "graded_above_M4qtc": gate(graded_means["M4qgrt"] > graded_means["M4qtc"],
                                    graded_means["M4qgrt"]-graded_means["M4qtc"], "strictly above"),
        "mismatch_above_M4qgr": gate(mm["M4qgrt"]["mean_mismatched_macro_f1"] > mm["M4qgr"]["mean_mismatched_macro_f1"],
                                      mm["M4qgrt"]["mean_mismatched_macro_f1"]-mm["M4qgr"]["mean_mismatched_macro_f1"], "strictly above"),
        "mismatch_above_M4qtc": gate(mm["M4qgrt"]["mean_mismatched_macro_f1"] > mm["M4qtc"]["mean_mismatched_macro_f1"],
                                      mm["M4qgrt"]["mean_mismatched_macro_f1"]-mm["M4qtc"]["mean_mismatched_macro_f1"], "strictly above"),
        "net_harmful_below_M4qgr": gate(mm["M4qgrt"]["net_harmful_flips"] < mm["M4qgr"]["net_harmful_flips"],
                                        {"M4qgrt": mm["M4qgrt"]["net_harmful_flips"], "M4qgr": mm["M4qgr"]["net_harmful_flips"]}, "strictly lower"),
        "net_harmful_below_M4qtc": gate(mm["M4qgrt"]["net_harmful_flips"] < mm["M4qtc"]["net_harmful_flips"],
                                        {"M4qgrt": mm["M4qgrt"]["net_harmful_flips"], "M4qtc": mm["M4qtc"]["net_harmful_flips"]}, "strictly lower"),
        "separation_invariants": gate(bool(op["formal_analysis_contract"]["separation_invariants_verified_before_evaluation"]), True, "all frozen invariants true"),
    }

    # H4. The frozen condition list has 9 text-Gaussian observations, not 12.
    h4_comparators = {}
    h4_all_gates = {}
    for comp in ("M4qcf", "M4qcs"):
        tg = paired_deltas(qi, "M4qgrt", comp, TEXT_GAUSSIAN_CONDITIONS)
        pos = sum(x > 0 for x in tg)
        clean = paired_deltas(qi, "M4qgrt", comp, ("clean",))
        other = paired_deltas(qi, "M4qgrt", comp, other_nonclean)
        gates = {
            "text_gaussian_mean_delta": gate(statistics.fmean(tg) >= 0.01, statistics.fmean(tg), ">=0.01"),
            "text_gaussian_positive_count": gate(pos >= 9, pos, ">=9 of 9 available frozen observations"),
            "text_gaussian_positive_fraction": gate(pos/len(tg) >= 0.75, pos/len(tg), ">=0.75"),
            "clean_mean_delta": gate(statistics.fmean(clean) >= -0.01, statistics.fmean(clean), ">=-0.01"),
            "other_nonclean_mean_delta": gate(statistics.fmean(other) >= -0.01, statistics.fmean(other), ">=-0.01"),
        }
        h4_comparators[comp] = {"observation_count_text_gaussian": len(tg), "gates": gates}
        for k,v in gates.items(): h4_all_gates[f"{k}_vs_{comp}"] = v

    # Separate clean successor safeguard.
    safeguard = {}
    for comp in ("M1b", "M4qcf", "M4qcs-w"):
        d = statistics.fmean(paired_deltas(qi, "M4qgrt", comp, ("clean",)))
        safeguard[comp] = gate(d >= -0.01, d, ">=-0.01")
    safeguard_passed = all(x["passed"] for x in safeguard.values())

    decisions = {
        "V29_H1": {"decision": decide(h1_gates), "gates": h1_gates},
        "V29_H2": {"decision": decide(h2_gates), "gates": h2_gates, "mismatch_aggregates": {k:{kk:vv for kk,vv in v.items() if kk != "_ratio_numeric"} for k,v in mm.items()}},
        "V29_H3": {"decision": decide(h3_gates), "gates": h3_gates, "graded_means": graded_means},
        "V29_H4": {"decision": decide(h4_all_gates), "gates": h4_all_gates, "comparators": h4_comparators,
                    "cardinality_clarification": "Frozen matrix contains 3 text-Gaussian severities x 3 seeds = 9 observations. Numeric count threshold >=9 and fraction >=0.75 are unchanged."},
    }

    result = {
        "record_version": VERSION,
        "status": "V029_FORMAL_ANALYSIS_COMPLETE",
        "evidence_classification": "CONTROLLED_VALIDATION_DEVELOPMENT_EVIDENCE",
        "formal_decisions": decisions,
        "clean_performance_safeguard": {"passed": safeguard_passed, "comparators": safeguard},
        "successor_recommendation_permitted": safeguard_passed,
        "scientific_safety": {"training_performed": False, "evaluation_rerun_by_analyzer": False,
                              "checkpoint_reselection_performed": False, "threshold_tuning_performed": False,
                              "official_test_accessed": False, "official_test_samples_accessed": 0},
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")

    lines = ["# AEGIS v0.29 Step5 Formal Analysis", "", "Evidence: controlled validation-development evidence.", "",
             "## Formal decisions", ""]
    for h in ("V29_H1","V29_H2","V29_H3","V29_H4"):
        lines.append(f"- **{h}: {decisions[h]['decision']}**")
    lines += ["", f"Clean-performance safeguard: **{'PASSED' if safeguard_passed else 'FAILED'}**", "",
              "Official test samples accessed: **0**", "",
              "No checkpoint reselection, tuning, retraining, or official-test evaluation was performed.", ""]
    a.output_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print("="*78)
    print("AEGIS V0.29 STEP5 FORMAL ANALYSIS: COMPLETE")
    for h in ("V29_H1","V29_H2","V29_H3","V29_H4"):
        print(f"{h}: {decisions[h]['decision']}")
    print("Clean safeguard:", "PASSED" if safeguard_passed else "FAILED")
    print("Official test samples accessed: 0")
    print("V029_STEP5_FORMAL_ANALYSIS_COMPLETE")

if __name__ == "__main__":
    main()
