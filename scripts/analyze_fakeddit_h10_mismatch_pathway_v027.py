from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ANALYZER_VERSION = "0.27.0-step15d"
INPUT_PROTOCOL_VERSION = "0.27.0-step15a"
EXPECTED_PROTOCOL_SHA256 = "0c87544611beb989930e895fd9a2045bf32e6ea2e8bfff18810e91fe5211a77c"
EXPECTED_FORMAL_ANALYSIS_SHA256 = "82179d5c7eaa60db4f5a188f0625087c2a85479edb78bb5f897dee8bf71dd2b4"
EXPECTED_FORMAL_SUMMARY_SHA256 = "2c71db7d513157a580e2d902f12d2461a47e4f725544683b09bb95778a45decd"
EXPECTED_MAPPING_SHA256 = "975967d77622bcdad7e28597cb6f9f967c73bfe16d4a7455c1ab56cc7fbc6464"
FORMAL_SEEDS = (42, 43, 44)
ARCHITECTURES = ("M1b", "M4qc", "M4qcf")
STATES = ("matched", "mismatched")
EXPECTED_SAMPLES = 1000
EXPECTED_DECISIONS = {
    "H8_F": "SUPPORTED",
    "H9_F": "NOT_SUPPORTED",
    "H10_F": "NOT_SUPPORTED",
}
TRANSITIONS = (
    "matched_correct_to_mismatched_correct",
    "matched_correct_to_mismatched_incorrect",
    "matched_incorrect_to_mismatched_correct",
    "matched_incorrect_to_mismatched_incorrect",
)
ABS_TOL = 1e-12


class MismatchPathwayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MismatchPathwayError(message)


def finite(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label}: expected number")
    result = float(value)
    require(math.isfinite(result), f"{label}: expected finite number")
    return result


def probability(value: Any, label: str) -> float:
    result = finite(value, label)
    require(0.0 <= result <= 1.0, f"{label}: outside [0,1]")
    return result


def read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing required file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MismatchPathwayError(f"invalid JSON at {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    require(path.is_file(), f"missing required file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MismatchPathwayError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


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


def close(left: Any, right: Any, label: str) -> None:
    require(math.isclose(finite(left, label), finite(right, label),
                         rel_tol=0.0, abs_tol=ABS_TOL), label)


def diagnostics_mean(summary: Mapping[str, Any], name: str, label: str) -> float:
    diagnostics = summary.get("diagnostics")
    require(isinstance(diagnostics, dict), f"{label}: diagnostics missing")
    item = diagnostics.get(name)
    require(isinstance(item, dict), f"{label}: {name} missing")
    return finite(item.get("mean"), f"{label}/{name}/mean")


def metrics(summary: Mapping[str, Any], label: str) -> Tuple[float, float]:
    values = summary.get("metrics")
    require(isinstance(values, dict), f"{label}: metrics missing")
    require(values.get("sample_count") == EXPECTED_SAMPLES, f"{label}: sample count")
    return (
        probability(values.get("macro_f1"), f"{label}/macro_f1"),
        probability(values.get("accuracy"), f"{label}/accuracy"),
    )


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol.get("protocol_version") == INPUT_PROTOCOL_VERSION,
            "Step15A protocol version mismatch")
    require(protocol.get("analysis_type") == "exploratory_post_hoc_characterization",
            "Step15A analysis type mismatch")
    require(protocol.get("frozen_step14_decisions") == EXPECTED_DECISIONS,
            "frozen decision mismatch")
    spec = protocol.get("step15d_h10_mismatch_decomposition")
    require(isinstance(spec, dict), "Step15D protocol section missing")
    require(spec.get("architectures") == list(ARCHITECTURES),
            "Step15D architecture set mismatch")
    require("prediction_flip_rate" in spec.get("required_sample_level_characterization", []),
            "Step15D prediction-flip requirement missing")


def validate_formal(formal: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    require(formal.get("formal_decisions") == EXPECTED_DECISIONS, "formal decision drift")
    require(summary.get("formal_decisions") == EXPECTED_DECISIONS, "summary decision drift")
    require(formal.get("H10_F", {}).get("decision") == "NOT_SUPPORTED", "H10-F drift")
    require(formal.get("official_test_accessed") is False, "official test access reported")
    require(formal.get("official_test_samples_accessed") == 0, "official test samples reported")
    require(formal.get("training_performed") is False, "training reported")
    require(formal.get("checkpoint_reselection_performed") is False, "reselection reported")
    require(formal.get("threshold_tuning_performed") is False, "tuning reported")
    rows = formal["H10_F"].get("seed_results")
    require(isinstance(rows, list) and len(rows) == 3, "expected three formal H10 seed results")
    by_seed = {row.get("seed"): row for row in rows}
    require(set(by_seed) == set(FORMAL_SEEDS), "formal H10 seed set mismatch")
    return by_seed


def architecture_dir(architecture: str) -> str:
    return architecture.lower()


def load_summaries(input_root: Path) -> Tuple[Dict[Tuple[int, str, str], Dict[str, Any]], Dict[str, str]]:
    summaries = {}
    hashes = {}
    for seed in FORMAL_SEEDS:
        for architecture in ARCHITECTURES:
            for state in STATES:
                path = (input_root / f"seed{seed}" / architecture_dir(architecture)
                        / "mismatch" / f"{state}_summary.json")
                document = read_json(path)
                label = f"seed{seed}/{architecture}/{state}"
                require(document.get("architecture") == architecture, f"{label}: architecture")
                require(document.get("model_seed") == seed, f"{label}: seed")
                require(document.get("mismatch_state") == state, f"{label}: state")
                require(document.get("official_test_accessed") is False, f"{label}: test access")
                require(document.get("mapping_sha256") == EXPECTED_MAPPING_SHA256,
                        f"{label}: mapping hash")
                metrics(document, label)
                summaries[(seed, architecture, state)] = document
                hashes[str(path).replace("\\", "/")] = sha256(path)
    return summaries, hashes


def validate_sample(row: Mapping[str, Any], seed: int, architecture: str,
                    state: str, index: int) -> Dict[str, Any]:
    label = f"seed{seed}/{architecture}/{state}/row{index}"
    require(row.get("architecture") == architecture, f"{label}: architecture")
    require(row.get("model_seed") == seed, f"{label}: seed")
    require(row.get("mismatch_state") == state, f"{label}: state")
    require(row.get("row_index") == index, f"{label}: row index")
    require(row.get("official_test_accessed") is False, f"{label}: official test")
    require(row.get("target") in (0, 1), f"{label}: target")
    require(row.get("prediction") in (0, 1), f"{label}: prediction")
    require(isinstance(row.get("correct"), bool), f"{label}: correct")
    require(row["correct"] is (row["prediction"] == row["target"]),
            f"{label}: correctness mismatch")
    p0 = probability(row.get("probability_class0"), f"{label}/p0")
    p1 = probability(row.get("probability_class1"), f"{label}/p1")
    require(math.isclose(p0 + p1, 1.0, rel_tol=0.0, abs_tol=2e-6),
            f"{label}: probabilities do not sum to one")
    sample_id = row.get("sample_id")
    require(isinstance(sample_id, str) and sample_id, f"{label}: sample id")
    return {
        "row_index": index,
        "sample_id": sample_id,
        "receiver_sample_id": row.get("receiver_sample_id"),
        "target": row["target"],
        "prediction": row["prediction"],
        "correct": row["correct"],
        "confidence": p0 if row["prediction"] == 0 else p1,
        "probability_class0": p0,
        "probability_class1": p1,
        "vision_donor_sample_id": row.get("vision_donor_sample_id"),
        "receiver_class": row.get("receiver_class"),
        "donor_class": row.get("donor_class"),
    }


def load_samples(input_root: Path, seed: int, architecture: str,
                 state: str) -> Tuple[List[Dict[str, Any]], str]:
    path = (input_root / f"seed{seed}" / architecture_dir(architecture)
            / "mismatch" / f"{state}_samples.jsonl")
    raw = read_jsonl(path)
    require(len(raw) == EXPECTED_SAMPLES,
            f"seed{seed}/{architecture}/{state}: expected 1000 rows")
    rows = [validate_sample(row, seed, architecture, state, index)
            for index, row in enumerate(raw)]
    return rows, sha256(path)


def describe(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def transition_name(matched_correct: bool, mismatched_correct: bool) -> str:
    if matched_correct and mismatched_correct:
        return TRANSITIONS[0]
    if matched_correct and not mismatched_correct:
        return TRANSITIONS[1]
    if not matched_correct and mismatched_correct:
        return TRANSITIONS[2]
    return TRANSITIONS[3]


def transition_analysis(matched: Sequence[Mapping[str, Any]],
                        mismatched: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(len(matched) == len(mismatched) == EXPECTED_SAMPLES,
            "transition sample count mismatch")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    flip_count = 0
    for left, right in zip(matched, mismatched):
        for field in ("row_index", "sample_id", "receiver_sample_id", "target", "receiver_class"):
            require(left[field] == right[field], f"transition alignment mismatch: {field}")
        require(right["donor_class"] == right["receiver_class"],
                "mismatch mapping is not class-preserving")
        name = transition_name(left["correct"], right["correct"])
        flipped = left["prediction"] != right["prediction"]
        flip_count += flipped
        groups[name].append({
            "row_index": left["row_index"],
            "sample_id": left["sample_id"],
            "target": left["target"],
            "matched_prediction": left["prediction"],
            "mismatched_prediction": right["prediction"],
            "prediction_flipped": flipped,
            "matched_confidence": left["confidence"],
            "mismatched_confidence": right["confidence"],
            "confidence_change": right["confidence"] - left["confidence"],
        })
    output_groups = []
    for name in TRANSITIONS:
        items = groups[name]
        changes = [item["confidence_change"] for item in items]
        output_groups.append({
            "transition": name,
            "count": len(items),
            "proportion": len(items) / EXPECTED_SAMPLES,
            "prediction_flip_count": sum(item["prediction_flipped"] for item in items),
            "confidence_change": describe(changes),
            "matched_confidence": describe([item["matched_confidence"] for item in items]),
            "mismatched_confidence": describe([item["mismatched_confidence"] for item in items]),
        })
    require(sum(group["count"] for group in output_groups) == EXPECTED_SAMPLES,
            "transition groups do not cover all samples")
    return {
        "sample_count": EXPECTED_SAMPLES,
        "prediction_flip_count": flip_count,
        "prediction_flip_rate": flip_count / EXPECTED_SAMPLES,
        "confidence_definition": "probability assigned to the predicted class",
        "confidence_change_definition": "mismatched confidence minus matched confidence",
        "transition_groups": output_groups,
    }


def summarize_architectures(
    summaries: Mapping[Tuple[int, str, str], Mapping[str, Any]],
    formal_by_seed: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output = []
    for seed in FORMAL_SEEDS:
        architecture_metrics = {}
        for architecture in ARCHITECTURES:
            matched_f1, matched_accuracy = metrics(
                summaries[(seed, architecture, "matched")], "matched summary")
            mismatched_f1, mismatched_accuracy = metrics(
                summaries[(seed, architecture, "mismatched")], "mismatched summary")
            architecture_metrics[architecture] = {
                "matched_macro_f1": matched_f1,
                "mismatched_macro_f1": mismatched_f1,
                "matched_minus_mismatched_macro_f1": matched_f1 - mismatched_f1,
                "matched_accuracy": matched_accuracy,
                "mismatched_accuracy": mismatched_accuracy,
                "matched_minus_mismatched_accuracy": matched_accuracy - mismatched_accuracy,
            }
        delta_m1b = architecture_metrics["M1b"]["matched_minus_mismatched_macro_f1"]
        delta_m4qcf = architecture_metrics["M4qcf"]["matched_minus_mismatched_macro_f1"]
        g_value = delta_m1b - delta_m4qcf
        formal = formal_by_seed[seed]
        close(g_value, formal["G"], f"seed{seed} G mismatch")
        close(delta_m1b, formal["m1b_mismatch_drop"], f"seed{seed} M1b drop")
        close(delta_m4qcf, formal["m4qcf_mismatch_drop"], f"seed{seed} M4qcf drop")
        output.append({
            "seed": seed,
            "architectures": architecture_metrics,
            "M1b_vs_M4qcf_mismatch_degradation": {
                "M1b_drop": delta_m1b,
                "M4qcf_drop": delta_m4qcf,
                "G": g_value,
            },
            "M4qc_vs_M4qcf_mismatch_degradation_descriptive": {
                "M4qc_drop": architecture_metrics["M4qc"]["matched_minus_mismatched_macro_f1"],
                "M4qcf_drop": delta_m4qcf,
                "M4qc_drop_minus_M4qcf_drop": (
                    architecture_metrics["M4qc"]["matched_minus_mismatched_macro_f1"]
                    - delta_m4qcf
                ),
            },
        })
    return output


def m4qcf_mechanism(
    summaries: Mapping[Tuple[int, str, str], Mapping[str, Any]],
    formal_by_seed: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    names = (
        "compatibility_score", "interaction_multiplier", "text_weight", "vision_weight",
        "effective_text_reliability", "effective_vision_reliability",
    )
    output = []
    for seed in FORMAL_SEEDS:
        matched = summaries[(seed, "M4qcf", "matched")]
        mismatched = summaries[(seed, "M4qcf", "mismatched")]
        item: Dict[str, Any] = {"seed": seed}
        for name in names:
            matched_mean = diagnostics_mean(matched, name, f"seed{seed}/matched")
            mismatched_mean = diagnostics_mean(mismatched, name, f"seed{seed}/mismatched")
            item[f"matched_{name}"] = matched_mean
            item[f"mismatched_{name}"] = mismatched_mean
            item[f"matched_minus_mismatched_{name}"] = matched_mean - mismatched_mean
        item["compatibility_drop"] = item["matched_minus_mismatched_compatibility_score"]
        item["interaction_suppression_D_I"] = item["matched_minus_mismatched_interaction_multiplier"]
        close(item["interaction_suppression_D_I"],
              formal_by_seed[seed]["interaction_suppression_D_I"],
              f"seed{seed} D_I mismatch")
        output.append(item)
    return output


def analyze(protocol_path: Path, formal_path: Path, formal_summary_path: Path,
            input_root: Path, output_root: Path, *, write_outputs: bool = True) -> Dict[str, Any]:
    base_hashes = {
        "step15a_protocol_json": verify_hash(protocol_path, EXPECTED_PROTOCOL_SHA256, "protocol"),
        "formal_analysis_json": verify_hash(formal_path, EXPECTED_FORMAL_ANALYSIS_SHA256, "formal analysis"),
        "formal_summary_json": verify_hash(formal_summary_path, EXPECTED_FORMAL_SUMMARY_SHA256, "formal summary"),
    }
    protocol = read_json(protocol_path)
    formal = read_json(formal_path)
    formal_summary = read_json(formal_summary_path)
    validate_protocol(protocol)
    formal_by_seed = validate_formal(formal, formal_summary)
    summaries, input_hashes = load_summaries(input_root)
    architecture_results = summarize_architectures(summaries, formal_by_seed)
    mechanism_results = m4qcf_mechanism(summaries, formal_by_seed)

    transitions = []
    total_sample_rows = 0
    for seed in FORMAL_SEEDS:
        for architecture in ARCHITECTURES:
            matched, matched_hash = load_samples(input_root, seed, architecture, "matched")
            mismatched, mismatched_hash = load_samples(input_root, seed, architecture, "mismatched")
            total_sample_rows += len(matched) + len(mismatched)
            for state, digest in (("matched", matched_hash), ("mismatched", mismatched_hash)):
                path = (input_root / f"seed{seed}" / architecture_dir(architecture)
                        / "mismatch" / f"{state}_samples.jsonl")
                input_hashes[str(path).replace("\\", "/")] = digest
            transition = transition_analysis(matched, mismatched)
            transition.update({"seed": seed, "architecture": architecture})
            transitions.append(transition)
    require(total_sample_rows == 18000, "expected 18,000 mismatch sample rows")

    g_values = [row["M1b_vs_M4qcf_mismatch_degradation"]["G"]
                for row in architecture_results]
    d_i_values = [row["interaction_suppression_D_I"] for row in mechanism_results]
    mean_g = statistics.fmean(g_values)
    mean_d_i = statistics.fmean(d_i_values)
    close(mean_g, formal["H10_F"]["mean_G"], "mean G mismatch")
    close(mean_d_i, formal["H10_F"]["mean_interaction_suppression_D_I"], "mean D_I mismatch")

    analysis = {
        "analyzer_version": ANALYZER_VERSION,
        "input_protocol_version": INPUT_PROTOCOL_VERSION,
        "analysis_type": "exploratory_post_hoc_mismatch_pathway_decomposition",
        "status": "STEP15D_H10_MISMATCH_PATHWAY_DECOMPOSITION_COMPLETE",
        "experiment": "fakeddit_h10_mismatch_pathway_decomposition_v027",
        "frozen_formal_decisions": dict(EXPECTED_DECISIONS),
        "H10_F_decision_preserved": "NOT_SUPPORTED",
        "base_input_sha256": base_hashes,
        "persisted_mismatch_input_sha256": dict(sorted(input_hashes.items())),
        "persisted_mismatch_file_count": len(input_hashes),
        "persisted_mismatch_sample_rows": total_sample_rows,
        "architecture_level_metrics_by_seed": architecture_results,
        "M4qcf_mechanism_metrics_by_seed": mechanism_results,
        "per_seed_G": [{"seed": row["seed"], "G": row["M1b_vs_M4qcf_mismatch_degradation"]["G"]}
                       for row in architecture_results],
        "mean_G": mean_g,
        "per_seed_interaction_suppression_D_I": [
            {"seed": row["seed"], "interaction_suppression_D_I": row["interaction_suppression_D_I"]}
            for row in mechanism_results
        ],
        "mean_interaction_suppression_D_I": mean_d_i,
        "sample_level_transition_characterization": transitions,
        "confidence_definition": "probability assigned to the predicted class",
        "interpretation_rule": (
            "Interaction suppression and prediction transitions are descriptive and do not establish "
            "that compatibility suppression caused mismatch robustness or degradation."
        ),
        "safety_audit": {
            "official_test_accessed": False,
            "official_test_samples_accessed": 0,
            "training_performed": False,
            "checkpoint_reselection_performed": False,
            "threshold_tuning_performed": False,
            "architecture_modification_performed": False,
            "new_mismatch_generation_performed": False,
            "mismatch_policy_modification_performed": False,
            "samples_dropped": False,
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
        "H10_F_decision_preserved": "NOT_SUPPORTED",
        "mean_G": mean_g,
        "mean_interaction_suppression_D_I": mean_d_i,
        "persisted_mismatch_sample_rows": total_sample_rows,
        "prediction_flip_rates": [
            {"seed": row["seed"], "architecture": row["architecture"],
             "prediction_flip_rate": row["prediction_flip_rate"]}
            for row in transitions
        ],
        "official_test_accessed": False,
        "new_mismatch_generation_performed": False,
        "causal_claim_made": False,
    }
    if write_outputs:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "mismatch_pathway_decomposition.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_root / "summary.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return analysis


def print_result(result: Mapping[str, Any], output_root: Path) -> None:
    print("=" * 78)
    print("AEGIS v0.27 STEP15D - EXPLORATORY H10 MISMATCH PATHWAY DECOMPOSITION")
    print("=" * 78)
    print()
    print("H10-F frozen decision: NOT_SUPPORTED (UNCHANGED)")
    print(f"Mean interaction suppression D_I: {result['mean_interaction_suppression_D_I']:.12f}")
    print(f"Mean G: {result['mean_G']:.12f}")
    print("Persisted mismatch sample rows:", result["persisted_mismatch_sample_rows"])
    print("New mismatch generation: NO")
    print("Official test accessed: NO")
    print("Significance test performed: NO")
    print("Causal claim made: NO")
    print("Output:", output_root)
    print()
    print("=" * 78)
    print("STEP15D_H10_MISMATCH_PATHWAY_DECOMPOSITION_COMPLETE")
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory Step15D H10 mismatch pathway decomposition.")
    parser.add_argument("--protocol", type=Path, default=Path(
        "docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"))
    parser.add_argument("--formal-analysis", type=Path, default=Path(
        "experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"))
    parser.add_argument("--formal-summary", type=Path, default=Path(
        "experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"))
    parser.add_argument("--input-root", type=Path, default=Path(
        "experiments/fakeddit/v027_m4qcf_robustness_diagnostics"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "experiments/fakeddit/v027_h10_mismatch_pathway_decomposition"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze(args.protocol, args.formal_analysis, args.formal_summary,
                         args.input_root, args.output_root)
    except MismatchPathwayError as exc:
        print("=" * 78)
        print("STEP15D EXPLORATORY ANALYSIS FAILED LOUDLY")
        print("=" * 78)
        print(str(exc))
        return 1
    print_result(result, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
