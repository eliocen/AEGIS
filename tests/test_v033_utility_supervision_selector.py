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
    ref_logits = _binary_logits_for_target_one(
        torch.tensor([0.45, 0.55, 0.50])
    )
    cand_logits = _binary_logits_for_target_one(
        torch.tensor([0.55, 0.45, 0.60])
    )
    posterior = v033.posterior_context_from_logits(ref_logits, cand_logits)
    new = v033(
        q_t,
        q_v,
        c,
        posterior["reference_positive_probability"],
        posterior["candidate_positive_probability"],
        posterior["candidate_minus_reference_positive_probability"],
    )

    assert new.utility_features.shape == (3, 10)
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


def test_v035_posterior_context_is_exact_label_free_and_detached() -> None:
    controller = UtilitySupervisedLearnedInterventionController(
        utility_initialization_seed=23,
    )
    ref_logits = torch.tensor(
        [[0.0, 0.4], [0.2, -0.1]], requires_grad=True
    )
    cand_logits = torch.tensor(
        [[0.0, 0.8], [-0.2, 0.3]], requires_grad=True
    )
    posterior = controller.posterior_context_from_logits(
        ref_logits, cand_logits
    )
    expected_ref = torch.softmax(ref_logits, dim=1)[:, 1]
    expected_cand = torch.softmax(cand_logits, dim=1)[:, 1]
    assert torch.allclose(
        posterior["reference_positive_probability"], expected_ref
    )
    assert torch.allclose(
        posterior["candidate_positive_probability"], expected_cand
    )
    assert torch.allclose(
        posterior["candidate_minus_reference_positive_probability"],
        expected_cand - expected_ref,
    )
    assert all(not value.requires_grad for value in posterior.values())


def test_v035_two_stage_model_integration_and_anti_leakage() -> None:
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
    prepared = model.prepare_v035_utility_supervised_context(
        text, vision, compute_loss=False
    )
    ref_logits = torch.randn(5, 2, requires_grad=True)
    cand_logits = torch.randn(5, 2, requires_grad=True)
    posterior = (
        model.utility_supervised_intervention_controller
        .posterior_context_from_logits(ref_logits, cand_logits)
    )
    out = model.finalize_v035_utility_supervised_context(
        prepared,
        posterior["reference_positive_probability"],
        posterior["candidate_positive_probability"],
        posterior["candidate_minus_reference_positive_probability"],
    )

    control = out["v033_control"]
    assert control.utility_features.shape == (5, 10)
    assert control.selector_logit.shape == (5, 1)
    assert control.utility_probability.shape == (5, 1)
    assert out["stable_reference_fused_embedding"].shape == (5, 4)
    assert out["candidate_fused_embedding"].shape == (5, 4)
    assert out["fused_embedding"].shape == (5, 4)
    assert torch.isfinite(control.selector_logit).all()
    assert torch.isfinite(control.utility_probability).all()

    control.selector_logit.sum().backward()
    assert ref_logits.grad is None
    assert cand_logits.grad is None


def test_v035_legacy_one_stage_m4qusli_call_is_rejected() -> None:
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
    text = torch.randn(2, 8)
    vision = torch.randn(2, 6)
    try:
        model(text, vision, compute_loss=False)
    except RuntimeError as exc:
        assert "prepare_v035_utility_supervised_context" in str(exc)
    else:
        raise AssertionError("legacy one-stage M4qusli call must be rejected")
