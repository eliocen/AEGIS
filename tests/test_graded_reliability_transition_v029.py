"""Frozen-contract invariants for AEGIS v0.29 Step2 variants."""

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.reliability_controller import (
    GradedReliabilityTransitionController,
)
from scripts.run_fakeddit_ablation import requires_quality_feature_statistics


def _scores():
    return (
        torch.tensor([[0.0], [0.2], [1.0]], requires_grad=True),
        torch.tensor([[1.0], [0.8], [0.0]], requires_grad=True),
        torch.tensor([[0.9], [0.5], [0.1]], requires_grad=True),
    )


@pytest.mark.parametrize(
    "mode", ["graded_weights_only", "transition_only", "combined"],
)
def test_v029_controller_is_parameter_free_and_normalized(mode):
    controller = GradedReliabilityTransitionController(mode=mode)
    out = controller(*_scores())
    assert controller.parameter_count == 0
    assert torch.allclose(
        out.text_weight + out.vision_weight,
        torch.ones_like(out.text_weight),
    )
    assert not out.weights.requires_grad


def test_graded_weights_respect_the_frozen_bounds():
    out = GradedReliabilityTransitionController(mode="graded_weights_only")(
        *_scores()
    )
    assert torch.all(out.text_weight >= 0.1)
    assert torch.all(out.text_weight <= 0.9)
    assert torch.equal(out.interaction_multiplier, torch.ones_like(out.interaction_multiplier))


def test_transition_control_uses_frozen_disagreement_margin():
    q_t, q_v, c = _scores()
    out = GradedReliabilityTransitionController(mode="transition_only")(q_t, q_v, c)
    expected = 1e-6 + (1.0 - 1e-6) * (c.detach() - 0.1 * (q_t.detach() - q_v.detach()).abs()).clamp(0.0, 1.0)
    assert torch.allclose(out.interaction_multiplier, expected)
    assert torch.equal(out.text_weight, torch.full_like(out.text_weight, 0.5))


@pytest.mark.parametrize(
    "mode", ["graded_weights_only", "transition_only", "combined"],
)
def test_model_variants_emit_the_frozen_fusion_identity(mode):
    torch.manual_seed(29)
    model = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_graded_transition_fusion=mode,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    ).eval()
    out = model(torch.randn(4, 768), torch.randn(4, 512))
    expected = (
        out["text_weight"] * out["aligned_text"]
        + out["vision_weight"] * out["aligned_vision"]
        + out["interaction_multiplier"] * out["scaled_interaction"]
    )
    assert torch.allclose(out["fused_embedding"], expected)
    assert model.graded_transition_reliability_controller.parameter_count == 0


def test_runner_recognizes_all_v029_architectures_as_quality_training():
    for architecture in (
        "quality_compatibility_graded_weights",
        "quality_compatibility_transition_control",
        "quality_compatibility_graded_transition_fusion",
    ):
        assert requires_quality_feature_statistics(architecture)


def test_new_path_rejects_experimental_flag_combinations():
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(
            quality_compatibility_graded_transition_fusion="combined",
            quality_compatibility_selective_fusion="combined",
        )
