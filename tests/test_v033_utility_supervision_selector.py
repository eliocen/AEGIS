from __future__ import annotations

import math

import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.reliability_controller import (
    CalibratedEvidenceUtilityInterventionController,
    UtilitySupervisedLearnedInterventionController,
)


def _binary_logits_for_target_one(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.to(dtype=torch.float32)
    logits = torch.log(probabilities / (1.0 - probabilities))
    return torch.stack((torch.zeros_like(logits), logits), dim=1)


def test_v033_target_anchor_directionality_and_detach() -> None:
    ref_prob = torch.tensor([0.50, 0.50, 0.50, 0.50, 0.50])
    cand_prob = torch.tensor([0.40, 0.49, 0.50, 0.51, 0.60])
    targets = torch.ones(5, dtype=torch.long)

    out = UtilitySupervisedLearnedInterventionController.utility_target_from_logits(
        _binary_logits_for_target_one(ref_prob).requires_grad_(True),
        _binary_logits_for_target_one(cand_prob).requires_grad_(True),
        targets,
    )

    assert out["p_ref_true"].shape == (5,)
    assert out["p_candidate_true"].shape == (5,)
    assert out["delta_u"].shape == (5,)
    assert out["u_target"].shape == (5,)
    assert not out["delta_u"].requires_grad
    assert not out["u_target"].requires_grad

    expected_delta = torch.tensor([-0.10, -0.01, 0.0, 0.01, 0.10])
    expected_target = torch.tensor([0.0, 0.45, 0.50, 0.55, 1.0])
    assert torch.allclose(out["delta_u"], expected_delta, atol=2e-6, rtol=0.0)
    assert torch.allclose(out["u_target"], expected_target, atol=1e-5, rtol=0.0)


def test_v033_inactive_default_and_candidate_action_parity() -> None:
    q_t = torch.tensor([[0.20], [0.80], [0.60]], dtype=torch.float32)
    q_v = torch.tensor([[0.70], [0.30], [0.60]], dtype=torch.float32)
    c = torch.tensor([[0.90], [0.40], [0.75]], dtype=torch.float32)

    v031 = CalibratedEvidenceUtilityInterventionController(
        mode="utility_only",
        utility_initialization_seed=17,
    )
    v033 = UtilitySupervisedLearnedInterventionController(
        utility_initialization_seed=17,
    )
    old = v031(q_t, q_v, c)
    new = v033(q_t, q_v, c)

    assert torch.allclose(
        new.reference_weights, old.reference_weights, atol=1e-7, rtol=1e-6
    )
    assert torch.allclose(
        new.candidate_text_weight,
        old.candidate_text_weight,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        new.candidate_vision_weight,
        old.candidate_vision_weight,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        new.candidate_interaction_multiplier,
        old.candidate_interaction_multiplier,
        atol=1e-7,
        rtol=1e-6,
    )

    # Frozen zero final layer initializes utility_probability exactly at 0.5.
    assert torch.allclose(
        new.utility_probability,
        torch.full_like(new.utility_probability, 0.5),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(new.intervention_gate).item() == 0
    assert torch.count_nonzero(new.active_intervention_indicator).item() == 0
    assert torch.allclose(new.weights, new.reference_weights, atol=0.0, rtol=0.0)
    assert torch.allclose(
        new.interaction_multiplier,
        torch.ones_like(new.interaction_multiplier),
        atol=0.0,
        rtol=0.0,
    )


def test_v033_model_integration_exposes_counterfactual_embeddings_and_logit() -> None:
    torch.manual_seed(3)
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
        quality_compatibility_utility_supervised_intervention=True,
        v033_utility_initialization_seed=23,
    )
    text = torch.randn(5, 8)
    vision = torch.randn(5, 6)
    out = model(text, vision, compute_loss=False)

    assert out["v033_control"] is not None
    assert out["selector_logit"].shape == (5, 1)
    assert out["utility_probability"].shape == (5, 1)
    assert out["stable_reference_fused_embedding"].shape == (5, 4)
    assert out["candidate_fused_embedding"].shape == (5, 4)
    assert out["fused_embedding"].shape == (5, 4)
    assert torch.isfinite(out["selector_logit"]).all()
    assert torch.isfinite(out["utility_probability"]).all()
