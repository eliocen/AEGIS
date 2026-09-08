from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_fakeddit_h9_failure_decomposition_v027.py"

spec = importlib.util.spec_from_file_location(
    "analyze_fakeddit_h9_failure_decomposition_v027", SCRIPT
)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_row(index: int, value: float):
    seed = 42 + index // 18
    modality = "text" if index % 2 == 0 else "vision"
    family = ("gaussian_noise", "attenuation", "zero_dropout")[index % 3]
    severity = (0.25, 0.5, 0.75, 1.0)[index % 4]
    return {
        "seed": seed,
        "condition": f"{modality}_{family}_{index}",
        "corrupted_modality": modality,
        "corruption_family": family,
        "severity": severity,
        "m1b_macro_f1": 0.70,
        "m4qc_macro_f1": 0.71,
        "m4qcf_macro_f1": 0.70 + value,
        "R_i": value,
        "sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
    }


def test_versions_and_frozen_decisions_are_exact():
    assert mod.ANALYZER_VERSION == "0.27.0-step15b"
    assert mod.INPUT_PROTOCOL_VERSION == "0.27.0-step15a"
    assert mod.EXPECTED_DECISIONS == {
        "H8_F": "SUPPORTED",
        "H9_F": "NOT_SUPPORTED",
        "H10_F": "NOT_SUPPORTED",
    }


def test_summarize_rows_reports_positive_negative_and_zero():
    rows = [make_row(0, 0.2), make_row(1, -0.1), make_row(2, 0.0)]
    result = mod.summarize_rows(rows)
    assert result["count"] == 3
    assert result["positive_count"] == 1
    assert result["negative_count"] == 1
    assert result["zero_count"] == 1
    assert math.isclose(result["positive_proportion"], 1 / 3, abs_tol=1e-12)
    assert math.isclose(result["mean_R_i"], 1 / 30, abs_tol=1e-12)
    assert result["median_R_i"] == 0.0
    assert result["min_R_i"] == -0.1
    assert result["max_R_i"] == 0.2


def test_summarize_rows_rejects_empty_group():
    with pytest.raises(mod.ExploratoryAnalysisError):
        mod.summarize_rows([])


def test_stratify_preserves_every_observation():
    rows = [make_row(index, 0.01 * (index - 2)) for index in range(6)]
    groups = mod.stratify(rows, ("corrupted_modality", "corruption_family"))
    assert sum(group["count"] for group in groups) == len(rows)


def test_severity_trajectories_are_ordered():
    rows = [make_row(index, 0.01) for index in range(18)]
    trajectories = mod.severity_trajectories(rows)
    assert trajectories
    for trajectory in trajectories:
        severities = [point["severity"] for point in trajectory["points"]]
        assert severities == sorted(severities)


def test_positive_ranking_is_descending():
    rows = [make_row(0, 0.1), make_row(1, 0.3), make_row(2, 0.2)]
    ranked = mod.rank_observations(rows, reverse=True)
    assert [row["R_i"] for row in ranked] == [0.3, 0.2, 0.1]


def test_negative_ranking_is_ascending():
    rows = [make_row(0, -0.1), make_row(1, -0.3), make_row(2, -0.2)]
    ranked = mod.rank_observations(rows, reverse=False)
    assert [row["R_i"] for row in ranked] == [-0.3, -0.2, -0.1]


def test_validate_observation_recomputes_r_i():
    source = {
        "seed": 42,
        "condition": "text_attenuation_s0p25",
        "corrupted_modality": "text",
        "corruption_type": "attenuation",
        "severity": 0.25,
        "m1b_macro_f1": 0.70,
        "m4qc_macro_f1": 0.71,
        "m4qcf_macro_f1": 0.72,
        "m4qcf_minus_m1b": 0.02,
        "strict_positive": True,
    }
    result = mod.validate_observation(source, 0)
    assert math.isclose(result["R_i"], 0.02, abs_tol=1e-12)
    assert result["sign"] == "positive"


def test_validate_observation_rejects_inconsistent_r_i():
    source = {
        "seed": 42,
        "condition": "text_attenuation_s0p25",
        "corrupted_modality": "text",
        "corruption_type": "attenuation",
        "severity": 0.25,
        "m1b_macro_f1": 0.70,
        "m4qc_macro_f1": 0.71,
        "m4qcf_macro_f1": 0.72,
        "m4qcf_minus_m1b": 0.03,
        "strict_positive": True,
    }
    with pytest.raises(mod.ExploratoryAnalysisError):
        mod.validate_observation(source, 0)


def test_validate_protocol_preserves_exploratory_boundary():
    protocol = {
        "protocol_version": "0.27.0-step15a",
        "analysis_type": "exploratory_post_hoc_characterization",
        "frozen_step14_decisions": mod.EXPECTED_DECISIONS,
        "data_boundary": {
            "official_test_access_permitted": False,
            "new_training_permitted": False,
            "checkpoint_reselection_permitted": False,
            "new_corruption_generation_permitted": False,
            "new_mismatch_generation_permitted": False,
        },
        "step15b_h9_decomposition": {
            "interpretation_rule": (
                "The previously frozen H9-F decision remains NOT_SUPPORTED "
                "regardless of any favorable subgroup."
            )
        },
    }
    mod.validate_protocol(protocol)


def test_validate_protocol_rejects_changed_h9_decision():
    protocol = {
        "protocol_version": "0.27.0-step15a",
        "analysis_type": "exploratory_post_hoc_characterization",
        "frozen_step14_decisions": {
            "H8_F": "SUPPORTED",
            "H9_F": "SUPPORTED",
            "H10_F": "NOT_SUPPORTED",
        },
        "data_boundary": {},
        "step15b_h9_decomposition": {},
    }
    with pytest.raises(mod.ExploratoryAnalysisError):
        mod.validate_protocol(protocol)


def test_finite_number_rejects_nan_and_inf():
    with pytest.raises(mod.ExploratoryAnalysisError):
        mod.finite_number(float("nan"), "x")
    with pytest.raises(mod.ExploratoryAnalysisError):
        mod.finite_number(float("inf"), "x")


def test_real_frozen_inputs_and_outputs(tmp_path):
    protocol = ROOT / "docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"
    formal = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"
    summary = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"
    if not all(path.is_file() for path in (protocol, formal, summary)):
        pytest.skip("frozen Step15B inputs are not present")
    result = mod.analyze_h9_failure_decomposition(
        protocol, formal, summary, tmp_path, write_outputs=True
    )
    assert result["status"] == "STEP15B_H9_FAILURE_DECOMPOSITION_COMPLETE"
    assert result["H9_F_decision_preserved"] == "NOT_SUPPORTED"
    assert result["overall"]["count"] == 54
    assert result["overall"]["positive_count"] == 29
    assert result["overall"]["negative_count"] + result["overall"]["zero_count"] == 25
    assert math.isclose(
        result["overall"]["mean_R_i"], 0.051111288731590224, abs_tol=1e-12
    )
    assert len(result["stratifications"]) == 7
    assert (tmp_path / "h9_failure_decomposition.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["H9_F_decision_preserved"] == "NOT_SUPPORTED"
