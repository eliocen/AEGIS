from __future__ import annotations

import json
import math

from scripts import run_fakeddit_ablation as runner


def test_v034_independent_recoverability_from_persisted_artifacts_only(tmp_path):
    samples = [
        {"epoch":1,"optimization_step":1,"sample_id_or_stable_index":"a","p_ref_true":0.50,"p_candidate_true":0.54,"delta_u":0.04,"u_target":0.70,"utility_probability":0.62,"utility_gate":0.05,"intervention_gate":0.05,"active_intervention_indicator":1.0},
        {"epoch":1,"optimization_step":1,"sample_id_or_stable_index":"b","p_ref_true":0.55,"p_candidate_true":0.52,"delta_u":-0.03,"u_target":0.35,"utility_probability":0.48,"utility_gate":0.0,"intervention_gate":0.0,"active_intervention_indicator":0.0},
    ]
    steps = [{"epoch":1,"optimization_step":1,"batch_sample_count":2,"non_neutral_target_count":2,"non_neutral_target_fraction":1.0,"utility_loss":0.7,"classification_loss":0.4,"selector_gradient_norm":0.03,"selector_parameter_l2_update_this_step":0.001,"selector_parameter_l2_movement_from_initial":0.002,"effective_classification_loss_weight":1.0,"effective_utility_loss_weight":0.50,"effective_transition_loss_weight_if_present":0.0}]
    (tmp_path / "training_sample_observability.jsonl").write_text("".join(json.dumps(x)+"\n" for x in samples), encoding="utf-8")
    (tmp_path / "optimization_step_observability.jsonl").write_text("".join(json.dumps(x)+"\n" for x in steps), encoding="utf-8")
    summary = runner.verify_v034_observability_artifacts(tmp_path)
    assert math.isclose(summary["delta_u_abs_ge_0p01_fraction"], 1.0)
    assert math.isclose(summary["selector_gradient_norm_median_non_neutral_steps"], 0.03)
    assert math.isclose(summary["selector_parameter_l2_movement_from_initial"], 0.002)


def test_v034_recoverability_rejects_bad_counterfactual_identity(tmp_path):
    sample = {"epoch":1,"optimization_step":1,"sample_id_or_stable_index":"a","p_ref_true":0.50,"p_candidate_true":0.54,"delta_u":0.03,"u_target":0.65,"utility_probability":0.62,"utility_gate":0.05,"intervention_gate":0.05,"active_intervention_indicator":1.0}
    step = {"epoch":1,"optimization_step":1,"batch_sample_count":1,"non_neutral_target_count":1,"non_neutral_target_fraction":1.0,"utility_loss":0.7,"classification_loss":0.4,"selector_gradient_norm":0.03,"selector_parameter_l2_update_this_step":0.001,"selector_parameter_l2_movement_from_initial":0.002,"effective_classification_loss_weight":1.0,"effective_utility_loss_weight":0.50,"effective_transition_loss_weight_if_present":0.0}
    (tmp_path / "training_sample_observability.jsonl").write_text(json.dumps(sample)+"\n", encoding="utf-8")
    (tmp_path / "optimization_step_observability.jsonl").write_text(json.dumps(step)+"\n", encoding="utf-8")
    try:
        runner.verify_v034_observability_artifacts(tmp_path)
    except RuntimeError as exc:
        assert "counterfactual identity" in str(exc)
    else:
        raise AssertionError("bad persisted delta_u was not rejected")
