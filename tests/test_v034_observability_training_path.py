from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from scripts import run_fakeddit_ablation as runner


def _payload():
    delta = [0.04, -0.03, 0.0, 0.02]
    pref = [0.50, 0.55, 0.60, 0.45]
    pcand = [a + b for a, b in zip(pref, delta)]
    target = [max(0.0, min(1.0, 0.5 + value / 0.20)) for value in delta]
    return {
        "optimization_step": 1,
        "p_ref_true": pref,
        "p_candidate_true": pcand,
        "delta_u": delta,
        "u_target": target,
        "utility_probability": [0.62, 0.48, 0.50, 0.61],
        "utility_gate": [0.05, 0.0, 0.0, 0.025],
        "intervention_gate": [0.05, 0.0, 0.0, 0.025],
        "active_intervention_indicator": [1.0, 0.0, 0.0, 1.0],
        "selector_parameter_l2_movement_from_initial": 0.002,
        "effective_transition_loss_weight": 0.0,
    }


def test_v034_observability_training_path_persists_live_payload(monkeypatch, tmp_path):
    batch = SimpleNamespace(batch_size=4, sample_ids=["s0", "s1", "s2", "s3"])

    class FakeTrainer:
        fusion_architecture = "quality_compatibility_utility_supervised_intervention"
        classification_loss_weight = 1.0

        def train_step(self, *args, **kwargs):
            del args, kwargs
            return {
                "loss": 1.0,
                "alignment_loss": 0.1,
                "classification_loss": 0.4,
                "quality_loss": 0.2,
                "text_quality_loss": 0.2,
                "vision_quality_loss": 0.2,
                "compatibility_loss": 0.1,
                "utility_loss": 0.7,
                "selector_gradient_norm": 0.03,
                "selector_parameter_l2_movement": 0.001,
                "utility_probability_mean": 0.5525,
                "active_intervention_rate": 0.5,
                "delta_u_mean": 0.0075,
                "delta_u_std": 0.0273,
                "u_target_mean": 0.5375,
                "effective_utility_loss_weight": runner.V033_UTILITY_LOSS_WEIGHT,
                "transition_loss": 0.0,
                "_v034_observability": _payload(),
            }

    monkeypatch.setattr(runner, "iter_training_batches", lambda **kwargs: [batch])
    monkeypatch.setattr(
        runner,
        "prepare_m4qcs_training_batch",
        lambda batch, **kwargs: (
            batch,
            (torch.ones(4, 1), torch.ones(4, 1)),
            torch.ones(4, 1),
            (torch.ones(4, 1), torch.ones(4, 1)),
            {"clean": 4},
        ),
    )

    metrics = runner.train_one_epoch(
        FakeTrainer(), batch, 4, 1, 42,
        quality_loss_weight=1.0,
        compatibility_loss_weight=1.0,
        transition_loss_weight=0.0,
        text_feature_std=torch.ones(1),
        vision_feature_std=torch.ones(1),
        observability_root=tmp_path,
    )

    samples = [json.loads(x) for x in (tmp_path / "training_sample_observability.jsonl").read_text().splitlines()]
    steps = [json.loads(x) for x in (tmp_path / "optimization_step_observability.jsonl").read_text().splitlines()]
    assert len(samples) == 4
    assert len(steps) == 1
    assert steps[0]["non_neutral_target_count"] == 3
    assert steps[0]["selector_gradient_norm"] > 1e-8
    assert steps[0]["selector_parameter_l2_update_this_step"] > 1e-8
    assert steps[0]["effective_utility_loss_weight"] == 0.50
    assert metrics["delta_u_abs_ge_0p01_fraction"] == 0.75
    assert metrics["selector_gradient_norm_median_non_neutral_steps"] == 0.03
    assert metrics["selector_parameter_l2_movement_from_initial"] == 0.002
    assert any(row["delta_u"] > 0.01 for row in samples)
    assert any(row["delta_u"] < -0.01 for row in samples)
    assert all(math.isclose(row["p_candidate_true"] - row["p_ref_true"], row["delta_u"], abs_tol=1e-7) for row in samples)


def test_v034_summary_has_exact_step1_distribution_fields():
    payload = _payload()
    samples = []
    for i in range(4):
        samples.append({
            "p_ref_true": payload["p_ref_true"][i],
            "p_candidate_true": payload["p_candidate_true"][i],
            "delta_u": payload["delta_u"][i],
            "u_target": payload["u_target"][i],
            "utility_probability": payload["utility_probability"][i],
            "active_intervention_indicator": payload["active_intervention_indicator"][i],
        })
    steps = [{
        "non_neutral_target_count": 3,
        "selector_gradient_norm": 0.03,
        "utility_loss": 0.7,
        "selector_parameter_l2_movement_from_initial": 0.002,
    }]
    summary = runner._v034_summarize_observability(samples, steps)
    required = {
        "delta_u_mean", "delta_u_std", "delta_u_median", "delta_u_q25", "delta_u_q75",
        "delta_u_min", "delta_u_max", "delta_u_positive_fraction", "delta_u_negative_fraction",
        "delta_u_abs_ge_0p01_fraction", "u_target_mean", "u_target_std", "u_target_median",
        "u_target_q25", "u_target_q75", "u_target_neutral_0p45_0p55_fraction",
        "u_target_ge_0p60_fraction", "u_target_le_0p40_fraction", "utility_probability_mean",
        "utility_probability_std", "utility_loss_mean", "active_intervention_rate",
        "selector_gradient_norm_median_non_neutral_steps", "selector_parameter_l2_movement_from_initial",
    }
    assert required == set(summary)



def test_v034_formal_main_loop_wires_observability_root_from_experiment_root():
    import ast
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(runner.main))
    tree = ast.parse(source)
    calls = []
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name == "train_one_epoch":
                calls.append(node)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "v034_observability_root":
                    assignments.append(node)
    assert len(assignments) == 1
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords if kw.arg}
    assert keywords["observability_root"] == "v034_observability_root"
    assignment_text = ast.unparse(assignments[0].value)
    assert "args.experiment_root" in assignment_text
    assert "'v034_observability'" in assignment_text
    assert "is_v033_utility_supervised_architecture(args.fusion_architecture)" in assignment_text


def test_v034_formal_observability_activation_is_restricted_to_m4qusli(tmp_path):
    primary = tmp_path / "v034_observability" if runner.is_v033_utility_supervised_architecture("quality_compatibility_utility_supervised_intervention") else None
    inherited = tmp_path / "v034_observability" if runner.is_v033_utility_supervised_architecture("quality_compatibility_calibrated_intervention") else None
    assert primary == tmp_path / "v034_observability"
    assert inherited is None
