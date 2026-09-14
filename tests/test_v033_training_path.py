from __future__ import annotations

import math

import torch

from aegis.reliability.reliability_controller import (
    UtilitySupervisedLearnedInterventionController,
)
from scripts.run_fakeddit_ablation import (
    V033_UTILITY_LOSS_WEIGHT,
    compute_v033_utility_loss,
)


def _binary_logits_for_target_one(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.to(dtype=torch.float32)
    logits = torch.log(probabilities / (1.0 - probabilities))
    return torch.stack((torch.zeros_like(logits), logits), dim=1)


def test_v033_controlled_non_neutral_supervision_smoke() -> None:
    targets = torch.ones(4, dtype=torch.long)
    ref = _binary_logits_for_target_one(
        torch.tensor([0.50, 0.50, 0.50, 0.50])
    )
    cand = _binary_logits_for_target_one(
        torch.tensor([0.54, 0.52, 0.49, 0.47])
    )

    out = UtilitySupervisedLearnedInterventionController.utility_target_from_logits(
        ref, cand, targets
    )
    assert torch.any(out["delta_u"] >= 0.01)
    assert torch.any(out["delta_u"] <= -0.01)
    assert torch.any(out["u_target"] >= 0.55)
    assert torch.any(out["u_target"] <= 0.45)


def test_v033_real_selector_objective_one_batch_moves_parameters() -> None:
    torch.manual_seed(11)
    controller = UtilitySupervisedLearnedInterventionController(
        utility_initialization_seed=31
    )
    optimizer = torch.optim.Adam(controller.utility_selector.parameters(), lr=1e-2)

    q_t = torch.tensor([[0.90], [0.75], [0.25], [0.45]], dtype=torch.float32)
    q_v = torch.tensor([[0.20], [0.55], [0.85], [0.35]], dtype=torch.float32)
    c = torch.tensor([[0.30], [0.80], [0.40], [0.95]], dtype=torch.float32)
    control = controller(q_t, q_v, c)

    targets = torch.ones(4, dtype=torch.long)
    ref_logits = _binary_logits_for_target_one(
        torch.tensor([0.50, 0.50, 0.50, 0.50])
    )
    cand_logits = _binary_logits_for_target_one(
        torch.tensor([0.56, 0.53, 0.48, 0.46])
    )
    target_out = controller.utility_target_from_logits(
        ref_logits, cand_logits, targets
    )

    before = [
        parameter.detach().clone()
        for parameter in controller.utility_selector.parameters()
    ]
    optimizer.zero_grad(set_to_none=True)
    utility_loss = compute_v033_utility_loss(
        control.selector_logit,
        target_out,
    )
    total_loss = V033_UTILITY_LOSS_WEIGHT * utility_loss
    assert torch.isfinite(utility_loss)
    assert float(utility_loss.detach()) > 0.0
    assert V033_UTILITY_LOSS_WEIGHT == 0.50

    total_loss.backward()
    grad_sq = 0.0
    for parameter in controller.utility_selector.parameters():
        if parameter.grad is not None:
            grad_sq += float(parameter.grad.detach().pow(2).sum())
    grad_norm = math.sqrt(grad_sq)
    assert grad_norm > 1e-8

    optimizer.step()
    movement_sq = 0.0
    for prior, parameter in zip(before, controller.utility_selector.parameters()):
        movement_sq += float((parameter.detach() - prior).pow(2).sum())
    movement = math.sqrt(movement_sq)
    assert movement > 1e-8

    after = controller(q_t, q_v, c)
    assert torch.isfinite(after.utility_probability).all()


def test_v033_magnitude_weight_increases_for_material_delta() -> None:
    delta = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.20])
    weight = UtilitySupervisedLearnedInterventionController.utility_magnitude_weight(
        delta
    )
    expected = torch.tensor([1.0, 1.4, 3.0, 5.0, 5.0])
    assert torch.allclose(weight, expected, atol=1e-7, rtol=0.0)


def test_v033_validation_path_persists_control_diagnostics(monkeypatch) -> None:
    from types import SimpleNamespace

    from scripts import run_fakeddit_ablation as runner_module

    class _EvalOnlyModule:
        def eval(self) -> None:
            return None

    class _FakeTrainer:
        mode = "multimodal"
        alignment_loss_weight = 0.5
        fusion_architecture = (
            "quality_compatibility_utility_supervised_intervention"
        )
        alignment_model = _EvalOnlyModule()
        classification_model = _EvalOnlyModule()

        def forward_batch(self, batch, compute_alignment_loss):
            del compute_alignment_loss
            batch_size = batch.integrity_targets.numel()
            logits = torch.tensor(
                [[1.5, -0.5], [-0.2, 1.2]],
                dtype=torch.float32,
            )[:batch_size]
            control = SimpleNamespace(
                utility_probability=torch.full(
                    (batch_size, 1), 0.65, dtype=torch.float32
                ),
                utility_gate=torch.full(
                    (batch_size, 1), 0.125, dtype=torch.float32
                ),
                intervention_gate=torch.full(
                    (batch_size, 1), 0.125, dtype=torch.float32
                ),
                active_intervention_indicator=torch.ones(
                    (batch_size, 1), dtype=torch.float32
                ),
            )
            zero = torch.tensor(0.0)
            return {
                "loss": zero,
                "alignment_loss": zero,
                "classification_loss": zero,
                "classification_outputs": {"integrity_logits": logits},
                "v033_control": control,
            }

    fake_batch = SimpleNamespace(
        integrity_targets=torch.tensor([0, 1], dtype=torch.long),
        sample_ids=["a", "b"],
    )
    monkeypatch.setattr(
        runner_module,
        "iter_sequential_batches",
        lambda full_batch, batch_size: [full_batch],
    )

    metrics = runner_module.evaluate(
        trainer=_FakeTrainer(),
        full_batch=fake_batch,
        batch_size=2,
    )

    assert math.isclose(
        metrics["utility_probability_mean"], 0.65, rel_tol=0.0, abs_tol=1e-7
    )
    assert math.isclose(
        metrics["utility_gate_mean"], 0.125, rel_tol=0.0, abs_tol=1e-7
    )
    assert math.isclose(
        metrics["intervention_gate_mean"], 0.125, rel_tol=0.0, abs_tol=1e-7
    )
    assert math.isclose(
        metrics["active_intervention_rate"], 1.0, rel_tol=0.0, abs_tol=1e-7
    )
