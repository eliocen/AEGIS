"""Frozen-contract tests for the AEGIS v0.28 selective controller."""

import pytest
import torch

from aegis.reliability.reliability_controller import (
    SelectiveReliabilityController,
)


def _scores():
    return (
        torch.tensor([[0.2], [0.8]], requires_grad=True),
        torch.tensor([[0.8], [0.2]], requires_grad=True),
        torch.tensor([[0.0], [1.0]], requires_grad=True),
    )


@pytest.mark.parametrize("mode", ["weights_only", "interaction_only", "combined"])
def test_controller_is_parameter_free_and_normalized(mode):
    controller = SelectiveReliabilityController(mode=mode)
    out = controller(*_scores())
    assert controller.parameter_count == 0
    assert torch.allclose(out.text_weight + out.vision_weight,
                          torch.ones_like(out.text_weight))


def test_combined_implements_exact_step1_equations():
    q_t, q_v, c = _scores()
    out = SelectiveReliabilityController(mode="combined")(q_t, q_v, c)
    s_t = (0.1 + q_t.detach()).pow(2.0)
    s_v = (0.1 + q_v.detach()).pow(2.0)
    assert torch.allclose(out.effective_text_reliability, s_t)
    assert torch.allclose(out.effective_vision_reliability, s_v)
    assert torch.allclose(out.text_weight, s_t / (s_t + s_v))
    assert torch.allclose(out.vision_weight, s_v / (s_t + s_v))
    assert torch.allclose(out.interaction_multiplier,
                          0.1 + 0.9 * c.detach())


def test_weights_only_fixes_interaction_multiplier_at_one():
    out = SelectiveReliabilityController(mode="weights_only")(*_scores())
    assert torch.equal(out.interaction_multiplier,
                       torch.ones_like(out.interaction_multiplier))


def test_interaction_only_uses_equal_allocation():
    out = SelectiveReliabilityController(mode="interaction_only")(*_scores())
    assert torch.equal(out.text_weight, torch.full_like(out.text_weight, 0.5))
    assert torch.equal(out.vision_weight, torch.full_like(out.vision_weight, 0.5))


def test_all_controller_inputs_are_stop_gradient():
    q_t, q_v, c = _scores()
    out = SelectiveReliabilityController(mode="combined")(q_t, q_v, c)
    assert not out.text_weight.requires_grad
    assert not out.vision_weight.requires_grad
    assert not out.interaction_multiplier.requires_grad


def test_allocation_is_independent_of_compatibility():
    q_t, q_v, _ = _scores()
    controller = SelectiveReliabilityController(mode="combined")
    low = controller(q_t, q_v, torch.zeros(2, 1))
    high = controller(q_t, q_v, torch.ones(2, 1))
    assert torch.equal(low.weights, high.weights)


def test_interaction_is_independent_of_quality():
    controller = SelectiveReliabilityController(mode="combined")
    c = torch.tensor([[0.3], [0.7]])
    low = controller(torch.zeros(2, 1), torch.zeros(2, 1), c)
    high = controller(torch.ones(2, 1), torch.ones(2, 1), c)
    assert torch.equal(low.interaction_multiplier, high.interaction_multiplier)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        SelectiveReliabilityController(mode="searched")
