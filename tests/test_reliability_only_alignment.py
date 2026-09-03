"""Tests for AEGIS M2 reliability-only residual alignment."""

import math

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.alignment.reliability_residual_fusion import (
    ModalityOnlyReliabilityEstimator,
    ReliabilityResidualFusion,
)


def _inputs(batch_size=4):
    torch.manual_seed(7)
    return (
        torch.randn(batch_size, 768),
        torch.randn(batch_size, 512),
    )


def test_modality_only_estimator_output_contract():
    estimator = ModalityOnlyReliabilityEstimator(
        dimension=16,
        hidden_dim=8,
        dropout=0.0,
    )
    out = estimator(torch.randn(3, 16))
    assert out.shape == (3, 1)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_modality_only_estimator_rejects_wrong_dimension():
    estimator = ModalityOnlyReliabilityEstimator(dimension=16)
    with pytest.raises(ValueError):
        estimator(torch.randn(3, 8))


def test_reliability_residual_block_output_contract():
    block = ReliabilityResidualFusion(
        dimension=16,
        reliability_hidden_dim=8,
        reliability_dropout=0.0,
    )

    first = torch.randn(3, 16)
    second = torch.randn(3, 16)
    base = torch.randn(3, 16)

    out = block(first, second, base)

    expected = {
        "first_embedding",
        "second_embedding",
        "gated_base_fusion",
        "first_reliability",
        "second_reliability",
        "first_weight",
        "second_weight",
        "weights",
        "reliability_adaptive_fusion",
        "scaled_reliability_fusion",
        "reliability_scale",
        "fused_embedding",
    }

    assert set(out) == expected
    assert out["fused_embedding"].shape == (3, 16)
    assert out["weights"].shape == (3, 2)


def test_reliability_residual_uses_deterministic_scaling():
    block = ReliabilityResidualFusion(
        dimension=16,
        reliability_hidden_dim=8,
        reliability_dropout=0.0,
    )

    first = torch.randn(2, 16)
    second = torch.randn(2, 16)
    base = torch.randn(2, 16)

    out = block(first, second, base)

    assert block.reliability_scale == pytest.approx(
        1.0 / math.sqrt(16)
    )

    assert torch.allclose(
        out["scaled_reliability_fusion"],
        out["reliability_adaptive_fusion"]
        / math.sqrt(16),
    )


def test_reliability_residual_uses_separate_estimators():
    block = ReliabilityResidualFusion(dimension=16)

    assert (
        block.first_reliability_estimator
        is not block.second_reliability_estimator
    )


def test_reliability_weights_sum_to_one():
    block = ReliabilityResidualFusion(
        dimension=16,
        reliability_hidden_dim=8,
        reliability_dropout=0.0,
    )

    out = block(
        torch.randn(5, 16),
        torch.randn(5, 16),
        torch.randn(5, 16),
    )

    assert torch.allclose(
        out["weights"].sum(dim=-1),
        torch.ones(5),
        atol=1e-6,
    )


def test_reliability_only_model_initializes_only_m2_block():
    model = CrossModalAlignmentModel(
        reliability_only=True
    )

    assert model.reliability_only is True
    assert model.fusion_architecture == "reliability_only"
    assert model.reliability_only_fusion is not None
    assert model.interaction_fusion is None
    assert model.gated_interaction_fusion is None
    assert model.evidence_integration is None


def test_reliability_only_model_output_contract():
    model = CrossModalAlignmentModel(
        reliability_only=True,
        evidence_reliability_dropout=0.0,
    )

    text, vision = _inputs()
    out = model(text, vision)

    assert out["aligned_text"].shape == (4, 512)
    assert out["aligned_vision"].shape == (4, 512)
    assert out["fused_embedding"].shape == (4, 512)

    assert "reliability_base_fusion" in out
    assert "text_reliability" in out
    assert "vision_reliability" in out
    assert "evidence_weights" in out
    assert "reliability_adaptive_fusion" in out
    assert "scaled_reliability_fusion" in out
    assert "reliability_scale" in out

    assert "interaction_embedding" not in out
    assert "cosine_similarity" not in out
    assert "evidence_difference" not in out


def test_reliability_only_base_matches_legacy_gate():
    torch.manual_seed(17)
    model = CrossModalAlignmentModel(
        reliability_only=True,
        dropout=0.0,
        evidence_reliability_dropout=0.0,
    )
    model.eval()

    text, vision = _inputs()

    with torch.no_grad():
        aligned_text, aligned_vision = model.align(
            text,
            vision,
        )
        expected_base = model.fusion(
            aligned_text,
            aligned_vision,
        )
        out = model(text, vision)

    assert torch.allclose(
        out["reliability_base_fusion"],
        expected_base,
        atol=1e-6,
    )


def test_reliability_only_alignment_loss_remains_available():
    model = CrossModalAlignmentModel(
        reliability_only=True,
        evidence_reliability_dropout=0.0,
    )

    text, vision = _inputs()

    out = model(
        text,
        vision,
        compute_loss=True,
    )

    assert "alignment_loss" in out
    assert out["alignment_loss"].ndim == 0


def test_reliability_only_gradients_reach_gate_and_estimators():
    model = CrossModalAlignmentModel(
        reliability_only=True,
        evidence_reliability_dropout=0.0,
    )

    text, vision = _inputs()
    out = model(text, vision)

    out["fused_embedding"].pow(2).mean().backward()

    gate_has_grad = any(
        p.grad is not None
        for p in model.fusion.parameters()
        if p.requires_grad
    )

    block = model.reliability_only_fusion
    assert block is not None

    first_has_grad = any(
        p.grad is not None
        for p in block
        .first_reliability_estimator
        .parameters()
        if p.requires_grad
    )

    second_has_grad = any(
        p.grad is not None
        for p in block
        .second_reliability_estimator
        .parameters()
        if p.requires_grad
    )

    assert gate_has_grad
    assert first_has_grad
    assert second_has_grad


def test_default_path_remains_legacy():
    model = CrossModalAlignmentModel()

    assert model.fusion_architecture == "legacy"
    assert model.reliability_only is False
    assert model.reliability_only_fusion is None


def test_explicit_reliability_false_matches_default_exactly():
    torch.manual_seed(11)
    default = CrossModalAlignmentModel()

    torch.manual_seed(11)
    explicit = CrossModalAlignmentModel(
        reliability_only=False
    )

    explicit.load_state_dict(
        default.state_dict()
    )

    text, vision = _inputs()

    default.eval()
    explicit.eval()

    with torch.no_grad():
        a = default(text, vision)
        b = explicit(text, vision)

    assert torch.equal(
        a["fused_embedding"],
        b["fused_embedding"],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "reliability_only": True,
            "evidence_aware": True,
        },
        {
            "reliability_only": True,
            "interaction_only": True,
        },
        {
            "reliability_only": True,
            "gated_interaction": True,
        },
    ],
)
def test_m2_is_mutually_exclusive_with_other_paths(
    kwargs,
):
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(**kwargs)


def test_invalid_reliability_only_type_fails():
    with pytest.raises(TypeError):
        CrossModalAlignmentModel(
            reliability_only="yes"
        )


def test_historical_m1_m3_error_message_is_preserved():
    with pytest.raises(
        ValueError,
        match=(
            "evidence_aware and interaction_only "
            "cannot both be True"
        ),
    ):
        CrossModalAlignmentModel(
            evidence_aware=True,
            interaction_only=True,
        )
