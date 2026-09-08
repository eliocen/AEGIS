from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_fakeddit_h10_mismatch_pathway_v027.py"
spec = importlib.util.spec_from_file_location("analyze_fakeddit_h10_mismatch_pathway_v027", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample(index, *, correct, prediction, confidence):
    target = prediction if correct else 1 - prediction
    return {
        "row_index": index,
        "sample_id": f"sample-{index}",
        "receiver_sample_id": f"sample-{index}",
        "target": target,
        "prediction": prediction,
        "correct": correct,
        "confidence": confidence,
        "probability_class0": confidence if prediction == 0 else 1 - confidence,
        "probability_class1": confidence if prediction == 1 else 1 - confidence,
        "vision_donor_sample_id": f"donor-{index}",
        "receiver_class": target,
        "donor_class": target,
    }


def expand(items):
    output = []
    for index in range(mod.EXPECTED_SAMPLES):
        template = items[index % len(items)]
        output.append(sample(index, **template))
    return output


def test_versions_and_frozen_decision():
    assert mod.ANALYZER_VERSION == "0.27.0-step15d"
    assert mod.INPUT_PROTOCOL_VERSION == "0.27.0-step15a"
    assert mod.EXPECTED_DECISIONS["H10_F"] == "NOT_SUPPORTED"


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (True, True, mod.TRANSITIONS[0]),
        (True, False, mod.TRANSITIONS[1]),
        (False, True, mod.TRANSITIONS[2]),
        (False, False, mod.TRANSITIONS[3]),
    ],
)
def test_transition_name(left, right, expected):
    assert mod.transition_name(left, right) == expected


def test_describe_empty_is_explicitly_undefined():
    assert mod.describe([]) == {
        "count": 0, "mean": None, "median": None, "min": None, "max": None
    }


def test_describe_values():
    result = mod.describe([-0.2, 0.0, 0.4])
    assert result["count"] == 3
    assert math.isclose(result["mean"], 0.2 / 3, abs_tol=1e-12)
    assert result["median"] == 0.0


def test_transition_analysis_covers_all_rows():
    matched = expand([
        {"correct": True, "prediction": 1, "confidence": 0.8},
        {"correct": False, "prediction": 0, "confidence": 0.7},
    ])
    mismatched = expand([
        {"correct": True, "prediction": 1, "confidence": 0.6},
        {"correct": True, "prediction": 1, "confidence": 0.9},
    ])
    result = mod.transition_analysis(matched, mismatched)
    assert sum(group["count"] for group in result["transition_groups"]) == 1000
    assert result["prediction_flip_count"] == 500
    assert result["prediction_flip_rate"] == 0.5


def test_transition_analysis_rejects_sample_misalignment():
    matched = expand([{"correct": True, "prediction": 1, "confidence": 0.8}])
    mismatched = expand([{"correct": True, "prediction": 1, "confidence": 0.7}])
    mismatched[0]["sample_id"] = "wrong"
    with pytest.raises(mod.MismatchPathwayError):
        mod.transition_analysis(matched, mismatched)


def test_probability_rejects_out_of_range():
    with pytest.raises(mod.MismatchPathwayError):
        mod.probability(1.1, "p")


def test_real_frozen_inputs_and_outputs(tmp_path):
    protocol = ROOT / "docs/experiments/v027/step15a_failure_mode_characterization_protocol.json"
    formal = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/formal_analysis.json"
    summary = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_analysis/summary.json"
    inputs = ROOT / "experiments/fakeddit/v027_m4qcf_robustness_diagnostics"
    if not all(path.exists() for path in (protocol, formal, summary, inputs)):
        pytest.skip("frozen Step15D inputs are not present")
    result = mod.analyze(protocol, formal, summary, inputs, tmp_path)
    assert result["status"] == "STEP15D_H10_MISMATCH_PATHWAY_DECOMPOSITION_COMPLETE"
    assert result["H10_F_decision_preserved"] == "NOT_SUPPORTED"
    assert result["persisted_mismatch_sample_rows"] == 18000
    assert len(result["architecture_level_metrics_by_seed"]) == 3
    assert len(result["M4qcf_mechanism_metrics_by_seed"]) == 3
    assert len(result["sample_level_transition_characterization"]) == 9
    assert math.isclose(result["mean_G"], -0.007623939063888622, abs_tol=1e-12)
    assert math.isclose(result["mean_interaction_suppression_D_I"],
                        0.42031526107986356, abs_tol=1e-12)
    assert (tmp_path / "mismatch_pathway_decomposition.json").is_file()
    assert (tmp_path / "summary.json").is_file()
