from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "run_fakeddit_unified_evaluation_v031.py"

spec = importlib.util.spec_from_file_location("v031_eval", MODULE_PATH)
assert spec is not None and spec.loader is not None
v031 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v031)


def test_v031_exact_formal_roster():
    assert v031.ARCHITECTURES == (
        "M1b",
        "M4qcf",
        "M4qcs-w",
        "M4qgrt",
        "M4qcesi-c",
        "M4qcesi-u",
        "M4qcesi",
    )
    assert v031.DEFAULT_SEEDS == (42, 43, 44)


def test_v031_operationalization_and_provenance_paths():
    assert str(v031.DEFAULT_OPERATIONALIZATION).replace("\\", "/") == (
        "docs/experiments/v031/step7_formal_evaluation_operationalization.json"
    )
    assert str(v031.DEFAULT_STEP6_PROVENANCE).replace("\\", "/") == (
        "docs/experiments/v031/step6_formal_training_evidence_provenance.json"
    )


def test_v031_architecture_routing_contract():
    specs = v031.ARCHITECTURE_SPECS
    assert specs["M4qcesi-c"]["fusion_architecture"] == (
        "quality_compatibility_calibrated_intervention_calibration"
    )
    assert specs["M4qcesi-c"]["controller_mode"] == "calibration_only"
    assert specs["M4qcesi-u"]["fusion_architecture"] == (
        "quality_compatibility_calibrated_intervention_utility"
    )
    assert specs["M4qcesi-u"]["controller_mode"] == "utility_only"
    assert specs["M4qcesi"]["fusion_architecture"] == (
        "quality_compatibility_calibrated_intervention"
    )
    assert specs["M4qcesi"]["controller_mode"] == "combined"


def test_v031_required_calibration_utility_diagnostics():
    required = {
        "q_text_raw",
        "q_vision_raw",
        "compatibility_raw",
        "q_text_calibrated",
        "q_vision_calibrated",
        "compatibility_calibrated",
        "calibrated_risk",
        "utility_probability",
        "utility_gate",
        "calibration_only_gate",
    }
    assert required == set(v031.V031_CALIBRATION_UTILITY_DIAGNOSTICS)
    assert set(v031.V031_INTERVENTION_DIAGNOSTICS) == {
        "intervention_gate",
        "active_intervention_indicator",
        "modality_weight_intervention_magnitude",
        "interaction_suppression_magnitude",
    }


def test_v031_frozen_intervention_constants():
    assert v031.V031_ACTIVE_INTERVENTION_THRESHOLD == 0.50
    assert v031.V031_MAX_WEIGHT_SHIFT == 0.15
    assert v031.V031_MAX_INTERACTION_SUPPRESSION == 0.50


def test_v031_source_preserves_no_training_and_test_seal_markers():
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert '"training_performed": False' in text
    assert '"official_test_accessed": False' in text
    assert '"official_test_samples_accessed": 0' in text
    assert '"V31_H5": "NOT_COMPUTED"' in text
    assert "quality_compatibility_calibrated_intervention" in text

def test_v031_checkpoint_mode_validation_uses_frozen_fusion_architecture_schema():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'explicit_mode = config.get(' in source
    assert '"quality_compatibility_calibrated_intervention"' in source
    assert "explicit_mode in (None, expected_mode)" in source
    assert 'transition_meta = config.get("transition_objective")' in source
    assert 'float(transition_meta.get("effective_weight")) == 0.0' in source
    assert 'float(config.get("transition_objective_weight", 0.0))' not in source

def test_v031_does_not_require_legacy_effective_reliability_fields():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "V031_BASE_DIAGNOSTICS = (" in source
    assert '*V031_BASE_DIAGNOSTICS,' in source
    block = source.split("V031_BASE_DIAGNOSTICS = (", 1)[1].split(")", 1)[0]
    assert '"effective_text_reliability"' not in block
    assert '"effective_vision_reliability"' not in block
    assert '"text_weight"' in block
    assert '"vision_weight"' in block
    assert '"interaction_multiplier"' in block
