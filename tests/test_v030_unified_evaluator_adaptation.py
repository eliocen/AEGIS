from __future__ import annotations
import ast, pathlib, re

EVALUATOR = pathlib.Path("scripts/run_fakeddit_unified_evaluation_v030.py")
BASE = pathlib.Path("scripts/run_fakeddit_unified_evaluation_v029.py")

EXPECTED_ARCHITECTURES = [
    "M1b","M4qcf","M4qcs-w","M4qgrt",
    "M4qesri-w","M4qesri-t","M4qesri",
]

def src():
    return EVALUATOR.read_text(encoding="utf-8-sig")

def fn(name):
    s = src()
    tree = ast.parse(s)
    lines = s.splitlines()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(lines[node.lineno - 1:node.end_lineno])

def test_separate_new_evaluator_and_valid_python():
    assert BASE.exists()
    ast.parse(src())

def test_exact_step7_roster():
    s = src()
    m = re.search(r"ARCHITECTURES\s*=\s*\((.*?)\)\s*\n\s*ARCHITECTURE_SPECS", s, re.S)
    assert m
    assert re.findall(r'"([^"]+)"', m.group(1)) == EXPECTED_ARCHITECTURES

def test_v030_specs_are_exact():
    s = src()
    assert '"quality_compatibility_selective_intervention_weights"' in s
    assert '"quality_compatibility_selective_intervention_transition"' in s
    assert '"quality_compatibility_selective_intervention"' in s
    assert '"v030_m4qesri_w_seed{seed}"' in s
    assert '"v030_m4qesri_t_seed{seed}"' in s
    assert '"v030_m4qesri_seed{seed}"' in s

def test_build_models_uses_new_constructor():
    s = fn("build_models")
    assert 'kwargs["quality_compatibility_selective_intervention"]' in s
    assert "alignment_model.quality_compatibility_selective_intervention" in s
    assert "alignment_model.selective_intervention_controller is not None" in s

def test_checkpoint_validation_binds_v030_mode():
    s = fn("load_checkpoint")
    assert 'config.get("quality_compatibility_selective_intervention")' in s
    assert 'config.get("v030_selective_intervention_protocol")' in s
    assert 'v030_protocol.get("mode") == expected_mode' in s

def test_step7_is_uniform_hash_authority():
    s = fn("main")
    assert "expected_checkpoint_hashes" in s
    assert "formal_evaluation_roster" in s
    assert "best_checkpoint" in s
    assert "len(expected_checkpoint_hashes) == 21" in s
    assert "expected_v029_hashes" not in s
    assert "expected_comparator_hashes" not in s

def test_inherited_counts_preserved():
    s = src()
    assert "expected_condition_count = 2 if args.smoke else 19" in s
    assert "399 if not args.smoke else None" in s
    assert "expected_mismatch_summaries" in s

def test_required_v030_diagnostics_present():
    s = src()
    for key in [
        "intervention_gate",
        "active_intervention_indicator",
        "modality_weight_intervention_magnitude",
        "interaction_suppression_magnitude",
    ]:
        assert key in s
    assert "V030_ACTIVE_INTERVENTION_THRESHOLD = 0.50" in s
    assert "V030_MAX_WEIGHT_SHIFT = 0.15" in s
    assert "V030_MAX_INTERACTION_SUPPRESSION = 0.50" in s

def test_ablation_specific_invariants_present():
    s = fn("evaluate_embeddings")
    assert 'architecture == "M4qesri-w"' in s
    assert 'architecture == "M4qesri-t"' in s
    assert "unexpectedly reports transition intervention" in s
    assert "unexpectedly reports weight intervention" in s

def test_hypotheses_not_computed():
    s = src()
    for h in ["V30_H1","V30_H2","V30_H3","V30_H4"]:
        assert h in s
    for h in ["V29_H1","V29_H2","V29_H3","V29_H4"]:
        assert h not in s

def test_official_test_sealed():
    s = src()
    assert '"official_test_accessed": False' in s
    assert '"official_test_samples_accessed": 0' in s
    assert "Official test accessed: NO" in s

def test_no_training_or_reselection():
    s = src()
    assert '"training_performed": False' in s
    assert '"checkpoint_reselection_performed": False' in s
    assert '"threshold_tuning_performed": False' in s
