"""AEGIS v0.30 Step2 selective-intervention tensor-contract tests."""

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.reliability_controller import (
    EvidenceConditionedSelectiveInterventionController,
)


def _scores():
    return (
        torch.tensor([[1.0], [0.2], [0.9]], requires_grad=True),
        torch.tensor([[1.0], [0.9], [0.1]], requires_grad=True),
        torch.tensor([[1.0], [0.2], [0.4]], requires_grad=True),
    )


@pytest.mark.parametrize("mode", ["weights_only", "transition_only", "combined"])
def test_v030_controller_is_parameter_free_bounded_and_detached(mode):
    q_t, q_v, c = _scores()
    controller = EvidenceConditionedSelectiveInterventionController(mode=mode)
    out = controller(q_t, q_v, c)
    assert controller.parameter_count == 0
    assert torch.all(out.intervention_gate >= 0.0)
    assert torch.all(out.intervention_gate <= 1.0)

    if mode in {"weights_only", "combined"}:
        assert torch.all(out.text_weight >= 0.1)
        assert torch.all(out.text_weight <= 0.9)
        assert torch.all(out.vision_weight >= 0.1)
        assert torch.all(out.vision_weight <= 0.9)
    else:
        assert torch.equal(out.text_weight, out.reference_text_weight)
        assert torch.equal(out.vision_weight, out.reference_vision_weight)

    assert torch.allclose(
        out.text_weight + out.vision_weight,
        torch.ones_like(out.text_weight),
    )
    assert torch.all(out.interaction_multiplier >= 0.5)
    assert torch.all(out.interaction_multiplier <= 1.0)
    assert not out.text_weight.requires_grad
    assert not out.intervention_gate.requires_grad
    assert not out.interaction_multiplier.requires_grad


def test_v030_gate_zero_exactly_preserves_stable_reference():
    controller = EvidenceConditionedSelectiveInterventionController(mode="combined")
    q = torch.ones(4, 1)
    c = torch.ones(4, 1)
    out = controller(q, q, c)
    assert torch.equal(out.intervention_gate, torch.zeros_like(out.intervention_gate))
    assert torch.equal(out.text_weight, out.reference_text_weight)
    assert torch.equal(out.vision_weight, out.reference_vision_weight)
    assert torch.equal(out.interaction_multiplier, torch.ones_like(out.interaction_multiplier))
    assert torch.equal(out.weight_intervention_magnitude, torch.zeros_like(out.weight_intervention_magnitude))
    assert torch.equal(out.transition_intervention_magnitude, torch.zeros_like(out.transition_intervention_magnitude))


def test_v030_transition_risk_and_gate_match_frozen_equations():
    controller = EvidenceConditionedSelectiveInterventionController(mode="combined")
    q_t = torch.tensor([[0.2]])
    q_v = torch.tensor([[0.8]])
    c = torch.tensor([[0.4]])
    out = controller(q_t, q_v, c)
    quality_deficit = 1.0 - 0.5 * (q_t + q_v)
    compatibility_deficit = 1.0 - c
    disagreement = (q_t - q_v).abs()
    expected_risk = (0.50 * compatibility_deficit + 0.25 * quality_deficit + 0.25 * disagreement).clamp(0.0, 1.0)
    expected_gate = ((expected_risk - 0.25) / 0.50).clamp(0.0, 1.0)
    assert torch.allclose(out.transition_risk, expected_risk)
    assert torch.allclose(out.intervention_gate, expected_gate)


def test_v030_intervention_increases_for_degraded_evidence():
    controller = EvidenceConditionedSelectiveInterventionController(mode="combined")
    clean = controller(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]))
    degraded = controller(torch.tensor([[0.1]]), torch.tensor([[0.9]]), torch.tensor([[0.1]]))
    assert degraded.transition_risk.item() > clean.transition_risk.item()
    assert degraded.intervention_gate.item() > clean.intervention_gate.item()


def test_v030_ablation_separation_is_exact():
    q_t, q_v, c = _scores()
    weights_only = EvidenceConditionedSelectiveInterventionController(mode="weights_only")(q_t, q_v, c)
    transition_only = EvidenceConditionedSelectiveInterventionController(mode="transition_only")(q_t, q_v, c)
    combined = EvidenceConditionedSelectiveInterventionController(mode="combined")(q_t, q_v, c)
    assert torch.equal(weights_only.interaction_multiplier, torch.ones_like(weights_only.interaction_multiplier))
    assert torch.equal(transition_only.text_weight, transition_only.reference_text_weight)
    assert torch.allclose(combined.text_weight, weights_only.text_weight)
    assert torch.allclose(combined.interaction_multiplier, transition_only.interaction_multiplier)


@pytest.mark.parametrize("mode", ["weights_only", "transition_only", "combined"])
def test_v030_model_emits_exact_fusion_identity(mode):
    torch.manual_seed(30)
    model = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_selective_intervention=mode,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    ).eval()
    out = model(torch.randn(4, 768), torch.randn(4, 512))
    expected = out["text_weight"] * out["aligned_text"] + out["vision_weight"] * out["aligned_vision"] + out["interaction_multiplier"] * out["scaled_interaction"]
    stable = out["reference_text_weight"] * out["aligned_text"] + out["reference_vision_weight"] * out["aligned_vision"] + out["scaled_interaction"]
    assert torch.allclose(out["fused_embedding"], expected)
    assert torch.allclose(out["stable_reference_fused_embedding"], stable)
    assert model.selective_intervention_controller.parameter_count == 0


def test_v030_rejects_combination_with_v029_controller():
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(
            quality_compatibility_selective_intervention="combined",
            quality_compatibility_graded_transition_fusion="combined",
        )
