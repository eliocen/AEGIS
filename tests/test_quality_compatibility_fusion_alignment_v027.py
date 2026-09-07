"""AEGIS v0.27 Step13C tests for M4qcf reliability-informed fusion."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.reliability_controller import (
    DeterministicReliabilityController,
)


class _ConstantScore(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return torch.full(
            (x.shape[0], 1),
            self.value,
            dtype=x.dtype,
            device=x.device,
        )


def _inputs(batch_size: int = 4):
    torch.manual_seed(1303)
    return (
        torch.randn(batch_size, 768),
        torch.randn(batch_size, 512),
    )


def _model(**kwargs):
    return CrossModalAlignmentModel(
        quality_compatibility_fusion=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
        **kwargs,
    )


def test_m4qcf_initializes_required_components_only():
    model = _model()

    assert model.quality_compatibility_fusion is True
    assert model.fusion_architecture == "quality_compatibility_fusion"
    assert model.gated_interaction_fusion is not None
    assert model.text_quality_estimator is not None
    assert model.vision_quality_estimator is not None
    assert model.compatibility_estimator is not None
    assert isinstance(
        model.reliability_controller,
        DeterministicReliabilityController,
    )
    assert model.reliability_controller.parameter_count == 0


def test_m4qcf_rejects_combination_with_existing_experimental_flags():
    incompatible = (
        "evidence_aware",
        "interaction_only",
        "gated_interaction",
        "reliability_only",
        "interaction_reliability",
        "quality_supervised",
        "quality_compatibility_supervised",
    )
    for flag in incompatible:
        with pytest.raises(ValueError):
            CrossModalAlignmentModel(
                quality_compatibility_fusion=True,
                **{flag: True},
            )


def test_existing_modes_reject_m4qcf_combination_symmetrically():
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(
            quality_compatibility_supervised=True,
            quality_compatibility_fusion=True,
        )
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(
            quality_supervised=True,
            quality_compatibility_fusion=True,
        )


def test_m4qcf_output_contract_contains_controller_diagnostics():
    model = _model(shared_dim=32)
    text, vision = _inputs()
    out = model(text, vision)

    expected = {
        "aligned_text",
        "aligned_vision",
        "fused_embedding",
        "evidence_difference",
        "evidence_product",
        "cosine_similarity",
        "interaction_embedding",
        "gated_interaction_base_fusion",
        "m1b_reference_fused_embedding",
        "scaled_interaction",
        "interaction_scale",
        "text_quality",
        "vision_quality",
        "compatibility_score",
        "effective_text_reliability",
        "effective_vision_reliability",
        "text_weight",
        "vision_weight",
        "evidence_weights",
        "interaction_multiplier",
        "reliability_weighted_fusion",
        "compatibility_scaled_interaction",
    }
    assert expected.issubset(out)
    assert out["fused_embedding"].shape == (4, 32)
    assert out["evidence_weights"].shape == (4, 2)


def test_m4qcf_implements_exact_frozen_step13a_fusion_equation():
    model = _model(shared_dim=32)
    model.eval()
    text, vision = _inputs()

    out = model(text, vision)
    expected_base = (
        out["text_weight"] * out["aligned_text"]
        + out["vision_weight"] * out["aligned_vision"]
    )
    expected_interaction = (
        out["interaction_multiplier"] * out["scaled_interaction"]
    )
    expected_fused = expected_base + expected_interaction

    assert torch.allclose(out["reliability_weighted_fusion"], expected_base)
    assert torch.allclose(
        out["compatibility_scaled_interaction"],
        expected_interaction,
    )
    assert torch.allclose(out["fused_embedding"], expected_fused)


def test_m4qcf_controller_weights_are_normalized_and_strictly_bounded():
    model = _model(shared_dim=32)
    model.eval()
    out = model(*_inputs())

    assert torch.all(out["text_weight"] > 0.0)
    assert torch.all(out["text_weight"] < 1.0)
    assert torch.all(out["vision_weight"] > 0.0)
    assert torch.all(out["vision_weight"] < 1.0)
    assert torch.allclose(
        out["text_weight"] + out["vision_weight"],
        torch.ones_like(out["text_weight"]),
    )


def test_m4qcf_parameter_count_equals_m4qc_parameter_count():
    torch.manual_seed(13)
    m4qc = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_supervised=True,
    )
    torch.manual_seed(13)
    m4qcf = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_fusion=True,
    )

    n_qc = sum(p.numel() for p in m4qc.parameters())
    n_qcf = sum(p.numel() for p in m4qcf.parameters())
    assert n_qcf == n_qc


def test_m4qcf_preserves_paired_initialization_of_all_m4qc_trainable_components():
    torch.manual_seed(2713)
    m4qc = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_supervised=True,
    )
    torch.manual_seed(2713)
    m4qcf = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_fusion=True,
    )

    qc_state = m4qc.state_dict()
    qcf_state = m4qcf.state_dict()
    assert set(qc_state) == set(qcf_state)
    for key in qc_state:
        assert torch.equal(qc_state[key], qcf_state[key]), key


def test_classification_path_is_detached_from_quality_and_compatibility_scores():
    model = _model(shared_dim=32)
    model.train()
    out = model(*_inputs())

    out["text_quality"].retain_grad()
    out["vision_quality"].retain_grad()
    out["compatibility_score"].retain_grad()

    out["fused_embedding"].square().mean().backward()

    assert out["text_quality"].grad is None
    assert out["vision_quality"].grad is None
    assert out["compatibility_score"].grad is None


def test_explicit_auxiliary_losses_still_train_q_and_c_heads():
    model = _model(shared_dim=32)
    model.train()
    out = model(*_inputs())

    q_loss = (
        (out["text_quality"] - 1.0).square().mean()
        + (out["vision_quality"] - 1.0).square().mean()
    )
    c_loss = (out["compatibility_score"] - 1.0).square().mean()
    (q_loss + c_loss).backward()

    q_text_grad = sum(
        float(p.grad.abs().sum())
        for p in model.text_quality_estimator.parameters()
        if p.grad is not None
    )
    q_vision_grad = sum(
        float(p.grad.abs().sum())
        for p in model.vision_quality_estimator.parameters()
        if p.grad is not None
    )
    c_grad = sum(
        float(p.grad.abs().sum())
        for p in model.compatibility_estimator.parameters()
        if p.grad is not None
    )
    assert q_text_grad > 0.0
    assert q_vision_grad > 0.0
    assert c_grad > 0.0


def test_higher_text_quality_increases_text_weight():
    model = _model(shared_dim=32)
    model.text_quality_estimator = _ConstantScore(0.9)
    model.vision_quality_estimator = _ConstantScore(0.2)
    model.compatibility_estimator = _ConstantScore(0.7)
    model.eval()

    out = model(*_inputs())
    assert torch.all(out["text_weight"] > out["vision_weight"])


def test_higher_vision_quality_increases_vision_weight():
    model = _model(shared_dim=32)
    model.text_quality_estimator = _ConstantScore(0.2)
    model.vision_quality_estimator = _ConstantScore(0.9)
    model.compatibility_estimator = _ConstantScore(0.7)
    model.eval()

    out = model(*_inputs())
    assert torch.all(out["vision_weight"] > out["text_weight"])


def test_equal_quality_produces_symmetric_modality_weights():
    model = _model(shared_dim=32)
    model.text_quality_estimator = _ConstantScore(0.6)
    model.vision_quality_estimator = _ConstantScore(0.6)
    model.compatibility_estimator = _ConstantScore(0.4)
    model.eval()

    out = model(*_inputs())
    assert torch.allclose(
        out["text_weight"],
        torch.full_like(out["text_weight"], 0.5),
    )
    assert torch.allclose(
        out["vision_weight"],
        torch.full_like(out["vision_weight"], 0.5),
    )


def test_lower_compatibility_suppresses_interaction_residual():
    torch.manual_seed(99)
    high = _model(shared_dim=32)
    torch.manual_seed(99)
    low = _model(shared_dim=32)

    for model, c in ((high, 0.9), (low, 0.1)):
        model.text_quality_estimator = _ConstantScore(0.8)
        model.vision_quality_estimator = _ConstantScore(0.8)
        model.compatibility_estimator = _ConstantScore(c)
        model.eval()

    text, vision = _inputs()
    high_out = high(text, vision)
    low_out = low(text, vision)

    # Common M1b interaction computation is identical under paired state.
    assert torch.allclose(
        high_out["scaled_interaction"],
        low_out["scaled_interaction"],
    )
    assert torch.all(
        high_out["interaction_multiplier"]
        > low_out["interaction_multiplier"]
    )
    assert (
        high_out["compatibility_scaled_interaction"].norm()
        > low_out["compatibility_scaled_interaction"].norm()
    )


def test_m4qcf_alignment_loss_remains_available():
    model = _model(shared_dim=32)
    out = model(*_inputs(), compute_loss=True)
    assert "alignment_loss" in out
    assert out["alignment_loss"].ndim == 0


def test_m4qc_remains_diagnostic_only_and_has_no_controller():
    model = CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )
    model.eval()
    out = model(*_inputs())

    assert model.reliability_controller is None
    assert "text_weight" not in out
    assert "interaction_multiplier" not in out
    assert torch.allclose(
        out["fused_embedding"],
        model.gated_interaction_fusion(
            out["aligned_text"], out["aligned_vision"]
        )["fused_embedding"],
    )
