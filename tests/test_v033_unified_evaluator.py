from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "run_fakeddit_unified_evaluation_v033.py"
PARENT_PATH = REPO_ROOT / "scripts" / "run_fakeddit_unified_evaluation_v031.py"

spec = importlib.util.spec_from_file_location("v033_eval", MODULE_PATH)
assert spec is not None and spec.loader is not None
v033 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v033)

parent_spec = importlib.util.spec_from_file_location("v031_eval_parent", PARENT_PATH)
assert parent_spec is not None and parent_spec.loader is not None
v031 = importlib.util.module_from_spec(parent_spec)
parent_spec.loader.exec_module(v031)


def test_v033_exact_formal_roster():
    assert v033.ARCHITECTURES == (
        "M1b",
        "M4qcf",
        "M4qcs-w",
        "M4qusli",
    )
    assert v033.DEFAULT_SEEDS == (42, 43, 44)


def test_v033_operationalization_and_training_provenance_paths():
    assert str(v033.DEFAULT_OPERATIONALIZATION).replace("\\", "/") == (
        "docs/experiments/v033/step9_formal_evaluation_operationalization.json"
    )
    assert str(v033.DEFAULT_STEP8_PROVENANCE).replace("\\", "/") == (
        "docs/experiments/v033/step8_formal_training_evidence_and_provenance.json"
    )


def test_v033_m4qusli_architecture_routing_contract():
    spec = v033.ARCHITECTURE_SPECS["M4qusli"]
    assert spec["fusion_architecture"] == (
        "quality_compatibility_utility_supervised_intervention"
    )
    assert spec["directory_pattern"] == "v033/formal_training/m4qusli_seed{seed}"
    assert v033.V033_UTILITY_SUPERVISED_ARCHITECTURES == ("M4qusli",)


def test_v033_required_utility_supervision_diagnostics():
    assert set(v033.V033_DIAGNOSTICS) == {
        "utility_probability",
        "utility_gate",
        "intervention_gate",
        "active_intervention_indicator",
        "p_ref_true",
        "p_candidate_true",
        "delta_u",
        "u_target",
    }
    assert v033.V033_ACTIVE_UTILITY_THRESHOLD == 0.60
    assert v033.V033_MAX_INTERVENTION_STRENGTH == 0.15


def test_v033_source_recomputes_counterfactual_targets_from_frozen_classifier():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'outputs.get("stable_reference_fused_embedding")' in source
    assert 'outputs.get("candidate_fused_embedding")' in source
    assert ".utility_target_from_logits(" in source
    assert '"p_ref_true"' in source
    assert '"p_candidate_true"' in source
    assert '"delta_u"' in source
    assert '"u_target"' in source


def test_v033_active_indicator_uses_frozen_point60_utility_threshold():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "V033_ACTIVE_UTILITY_THRESHOLD = 0.60" in source
    assert "utility_probability >= V033_ACTIVE_UTILITY_THRESHOLD" in source
    assert "(utility_probability - V033_ACTIVE_UTILITY_THRESHOLD)" in source
    assert "/ (1.0 - V033_ACTIVE_UTILITY_THRESHOLD)" in source


def test_v033_preserves_inherited_mismatch_functions_exactly():
    assert inspect.getsource(v033.mismatch_transition_metrics) == inspect.getsource(
        v031.mismatch_transition_metrics
    )
    assert inspect.getsource(v033.build_frozen_mismatch_mapping) == inspect.getsource(
        v031.build_frozen_mismatch_mapping
    )
    assert inspect.getsource(v033.mapping_records) == inspect.getsource(
        v031.mapping_records
    )


def test_v033_source_preserves_no_training_and_test_seal_markers():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert '"training_performed": False' in text
    assert '"official_test_accessed": False' in text
    assert '"official_test_samples_accessed": 0' in text
    assert '"V33_H5": "NOT_COMPUTED"' in text
    assert "quality_compatibility_utility_supervised_intervention" in text


def test_v033_formal_cardinalities_are_frozen():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '== 228' in source
    assert '== 24' in source
    assert 'len(expected_checkpoint_hashes) == 12' in source
