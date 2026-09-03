"""Tests for AEGIS v0.25.5 M2b interaction-conditioned reliability residual."""
import math
import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.alignment.interaction_reliability_residual_fusion import (
    InteractionConditionedReliabilityResidualFusion,
)


def make_model(**kwargs):
    return CrossModalAlignmentModel(
        text_dim=12,
        vision_dim=10,
        shared_dim=8,
        dropout=0.0,
        evidence_interaction_dropout=0.0,
        evidence_reliability_dropout=0.0,
        interaction_reliability=True,
        **kwargs,
    )


def test_m2b_flag_selects_architecture():
    model = make_model()
    assert model.interaction_reliability is True
    assert model.fusion_architecture == "interaction_reliability"
    assert model.interaction_reliability_fusion is not None


def test_m2b_preserves_parent_legacy_gate_without_duplicate_gate():
    model = make_model()
    block = model.interaction_reliability_fusion
    assert model.fusion is not None
    assert not hasattr(block, "fusion")
    assert not hasattr(block, "gated_fusion")


def test_m2b_forward_shapes_and_diagnostics():
    model = make_model().eval()
    text = torch.randn(4, 12)
    vision = torch.randn(4, 10)
    out = model(text, vision)
    assert out["fused_embedding"].shape == (4, 8)
    assert out["interaction_embedding"].shape == (4, 8)
    assert out["cosine_similarity"].shape == (4, 1)
    assert out["text_reliability"].shape == (4, 1)
    assert out["vision_reliability"].shape == (4, 1)
    assert out["evidence_weights"].shape == (4, 2)


def test_m2b_weights_are_normalized():
    model = make_model().eval()
    out = model(torch.randn(5, 12), torch.randn(5, 10))
    assert torch.allclose(
        out["evidence_weights"].sum(dim=-1),
        torch.ones(5),
        atol=1e-6,
    )


def test_m2b_reliability_is_bounded():
    model = make_model().eval()
    out = model(torch.randn(6, 12), torch.randn(6, 10))
    for key in ("text_reliability", "vision_reliability"):
        assert torch.all(out[key] >= 0.0)
        assert torch.all(out[key] <= 1.0)


def test_m2b_scale_is_inverse_sqrt_dimension():
    model = make_model().eval()
    out = model(torch.randn(2, 12), torch.randn(2, 10))
    assert out["reliability_scale"].item() == pytest.approx(1.0 / math.sqrt(8))


def test_m2b_alignment_loss_remains_available():
    model = make_model().eval()
    out = model(torch.randn(3, 12), torch.randn(3, 10), compute_loss=True)
    assert "alignment_loss" in out
    assert out["alignment_loss"].ndim == 0


def test_m2b_all_active_components_receive_gradients():
    model = make_model().train()
    out = model(torch.randn(7, 12), torch.randn(7, 10))
    out["fused_embedding"].pow(2).mean().backward()
    groups = {
        "text_projection": model.text_projection,
        "vision_projection": model.vision_projection,
        "legacy_fusion": model.fusion,
        "interaction_reliability_fusion": model.interaction_reliability_fusion,
    }
    for name, module in groups.items():
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        ), name


def test_m2b_is_mutually_exclusive_with_other_experimental_flags():
    for flag in (
        "evidence_aware",
        "interaction_only",
        "gated_interaction",
        "reliability_only",
    ):
        with pytest.raises(ValueError):
            CrossModalAlignmentModel(
                text_dim=12,
                vision_dim=10,
                shared_dim=8,
                interaction_reliability=True,
                **{flag: True},
            )


def test_m0_compatibility_default_is_unchanged():
    model = CrossModalAlignmentModel(text_dim=12, vision_dim=10, shared_dim=8)
    assert model.fusion_architecture == "legacy"
    assert model.interaction_reliability is False
    assert model.interaction_reliability_fusion is None


def test_block_rejects_bad_base_shape():
    block = InteractionConditionedReliabilityResidualFusion(
        dimension=8,
        interaction_dropout=0.0,
        reliability_dropout=0.0,
    )
    with pytest.raises(ValueError):
        block(torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 7))


def test_block_exposes_interaction_conditioned_outputs():
    block = InteractionConditionedReliabilityResidualFusion(
        dimension=8,
        interaction_dropout=0.0,
        reliability_dropout=0.0,
    ).eval()
    first = torch.randn(3, 8)
    second = torch.randn(3, 8)
    base = torch.randn(3, 8)
    out = block(first, second, base)
    for key in (
        "difference",
        "product",
        "cosine_similarity",
        "interaction_embedding",
        "first_reliability",
        "second_reliability",
        "weights",
        "fused_embedding",
    ):
        assert key in out
