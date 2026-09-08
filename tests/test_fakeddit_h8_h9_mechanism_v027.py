from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_fakeddit_h8_h9_mechanism_v027.py"
spec = importlib.util.spec_from_file_location("analyze_fakeddit_h8_h9_mechanism_v027", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def rows(d_values=(0.1, 0.2, 0.3), r_values=(-1.0, 0.0, 1.0)):
    return [
        {
            "seed": 42,
            "condition": f"c{index}",
            "corrupted_modality": "text" if index % 2 == 0 else "vision",
            "corruption_family": "gaussian_noise",
            "severity": 0.25,
            "D_m": d_value,
            "R_i": r_value,
            "quadrant": (
                "D_m_positive_R_i_positive" if d_value > 0 and r_value > 0
                else "D_m_positive_R_i_nonpositive" if d_value > 0
                else "D_m_nonpositive_R_i_positive" if r_value > 0
                else "D_m_nonpositive_R_i_nonpositive"
            ),
        }
        for index, (d_value, r_value) in enumerate(zip(d_values, r_values))
    ]


def test_versions_and_decisions_are_frozen():
    assert mod.ANALYZER_VERSION == "0.27.0-step15c"
    assert mod.INPUT_PROTOCOL_VERSION == "0.27.0-step15a"
    assert mod.EXPECTED_DECISIONS["H8_F"] == "SUPPORTED"
    assert mod.EXPECTED_DECISIONS["H9_F"] == "NOT_SUPPORTED"


def test_average_ranks_handles_ties():
    assert mod.average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]


def test_pearson_perfect_positive():
    assert math.isclose(mod.pearson([1, 2, 3], [2, 4, 6]), 1.0, abs_tol=1e-12)


def test_pearson_perfect_negative():
    assert math.isclose(mod.pearson([1, 2, 3], [6, 4, 2]), -1.0, abs_tol=1e-12)


def test_pearson_is_undefined_for_constant_input():
    assert mod.pearson([1, 1, 1], [1, 2, 3]) is None


def test_spearman_uses_tie_aware_ranks():
    result = mod.spearman([1, 2, 2, 3], [10, 20, 20, 30])
    assert math.isclose(result, 1.0, abs_tol=1e-12)


def test_association_is_descriptive_only():
    result = mod.association(rows())
    assert result["count"] == 3
    assert result["R_i_positive_count"] == 1
    assert result["R_i_nonpositive_count"] == 2
    assert result["significance_test_performed"] is False
    assert result["causal_claim_permitted"] is False


def test_stratification_preserves_all_rows():
    sample = rows((0.1, 0.2, 0.3, 0.4), (-1, 1, -1, 1))
    groups = mod.stratify(sample, "corrupted_modality")
    assert sum(group["count"] for group in groups) == 4


def test_quadrant_characterization_emits_all_four_quadrants():
    sample = rows((0.1, 0.2, -0.1, -0.2), (1.0, -1.0, 1.0, -1.0))
    result = mod.characterize_quadrants(sample)
    counts = {item["quadrant"]: item["count"] for item in result}
    assert counts == {
        "D_m_positive_R_i_positive": 1,
        "D_m_positive_R_i_nonpositive": 1,
        "D_m_nonpositive_R_i_positive": 1,
        "D_m_nonpositive_R_i_nonpositive": 1,
    }


def test_validate_protocol_rejects_causal_permission():
    protocol = {
        "protocol_version": "0.27.0-step15a",
        "analysis_type": "exploratory_post_hoc_characterization",
        "frozen_step14_decisions": mod.EXPECTED_DECISIONS,
        "step15c_h8_to_h9_mechanism": {
            "causal_claim_permitted": True,
            "significance_claim_permitted": False,
        },
    }
    with pytest.raises(mod.MechanismAnalysisError):
        mod.validate_protocol(protocol)


def test_real_frozen_inputs_and_outputs(tmp_path):
    protocol = ROOT / "docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"
    formal = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"
    summary = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"
    if not all(path.is_file() for path in (protocol, formal, summary)):
        pytest.skip("frozen Step15C inputs are not present")
    result = mod.analyze(protocol, formal, summary, tmp_path)
    assert result["status"] == "STEP15C_H8_TO_H9_MECHANISM_CHARACTERIZATION_COMPLETE"
    assert len(result["paired_D_m_and_R_i_table_for_all_54_observations"]) == 54
    assert result["H8_F_decision_preserved"] == "SUPPORTED"
    assert result["H9_F_decision_preserved"] == "NOT_SUPPORTED"
    counts = {row["quadrant"]: row["count"] for row in result["quadrant_characterization"]}
    assert counts["D_m_positive_R_i_positive"] == 29
    assert counts["D_m_positive_R_i_nonpositive"] == 25
    assert (tmp_path / "mechanism_characterization.json").is_file()
    assert (tmp_path / "summary.json").is_file()
