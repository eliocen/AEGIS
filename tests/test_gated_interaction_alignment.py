"""Tests for AEGIS M1b gated-interaction alignment."""

import math

import pytest
import torch

from aegis.alignment.gated_interaction_fusion import (
    GatedInteractionEvidenceFusion,
)
from aegis.alignment.model import CrossModalAlignmentModel


def _inputs(batch_size=4):
    torch.manual_seed(7)
    return (
        torch.randn(batch_size, 768),
        torch.randn(batch_size, 512),
    )


def test_gated_interaction_block_initializes():
    block = GatedInteractionEvidenceFusion(
        dimension=32,
        interaction_dropout=0.0,
    )
    assert block.dimension == 32
    assert block.interaction_scale == pytest.approx(
        1.0 / math.sqrt(32)
    )


def test_gated_interaction_block_output_contract():
    block = GatedInteractionEvidenceFusion(
        dimension=32,
        interaction_dropout=0.0,
    )
    first = torch.randn(3, 32)
    second = torch.randn(3, 32)
    out = block(first, second)

    expected = {
        "difference",
        "product",
        "cosine_similarity",
        "interaction_embedding",
        "gated_base_fusion",
        "scaled_interaction",
        "interaction_scale",
        "fused_embedding",
    }
    assert expected.issubset(out)
    assert out["fused_embedding"].shape == (3, 32)


def test_gated_interaction_uses_deterministic_scaling():
    block = GatedInteractionEvidenceFusion(
        dimension=16,
        interaction_dropout=0.0,
    )
    first = torch.randn(2, 16)
    second = torch.randn(2, 16)
    out = block(first, second)

    expected = (
        out["interaction_embedding"]
        / math.sqrt(16)
    )
    assert torch.allclose(
        out["scaled_interaction"],
        expected,
    )


def test_gated_interaction_model_initializes_only_m1b_block():
    model = CrossModalAlignmentModel(
        gated_interaction=True,
    )
    assert model.gated_interaction is True
    assert model.fusion_architecture == "gated_interaction"
    assert model.gated_interaction_fusion is not None
    assert model.interaction_fusion is None
    assert model.evidence_integration is None


def test_gated_interaction_model_output_contract():
    model = CrossModalAlignmentModel(
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )
    text, vision = _inputs()
    out = model(text, vision)

    assert out["aligned_text"].shape == (4, 512)
    assert out["aligned_vision"].shape == (4, 512)
    assert out["fused_embedding"].shape == (4, 512)
    assert "gated_interaction_base_fusion" in out
    assert "interaction_embedding" in out
    assert "text_reliability" not in out
    assert "vision_reliability" not in out
    assert "evidence_weights" not in out


def test_gated_interaction_alignment_loss_remains_available():
    model = CrossModalAlignmentModel(
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )
    text, vision = _inputs()
    out = model(
        text,
        vision,
        compute_loss=True,
    )
    assert "alignment_loss" in out
    assert out["alignment_loss"].ndim == 0


def test_gated_interaction_gradients_reach_gate_and_interaction():
    model = CrossModalAlignmentModel(
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )
    text, vision = _inputs()
    out = model(text, vision)
    out["fused_embedding"].sum().backward()

    block = model.gated_interaction_fusion
    assert block is not None

    gate_has_grad = any(
        p.grad is not None
        for p in block.gated_fusion.parameters()
        if p.requires_grad
    )
    interaction_has_grad = any(
        p.grad is not None
        for p in block.interaction.parameters()
        if p.requires_grad
    )

    assert gate_has_grad
    assert interaction_has_grad


def test_default_path_remains_legacy():
    model = CrossModalAlignmentModel()
    assert model.fusion_architecture == "legacy"
    assert model.gated_interaction is False
    assert model.gated_interaction_fusion is None


def test_explicit_false_matches_default_exactly():
    torch.manual_seed(11)
    default = CrossModalAlignmentModel()

    torch.manual_seed(11)
    explicit = CrossModalAlignmentModel(
        gated_interaction=False,
    )

    explicit.load_state_dict(default.state_dict())

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
            "gated_interaction": True,
            "interaction_only": True,
        },
        {
            "gated_interaction": True,
            "evidence_aware": True,
        },
        {
            "interaction_only": True,
            "evidence_aware": True,
        },
    ],
)
def test_experimental_paths_are_mutually_exclusive(kwargs):
    with pytest.raises(ValueError):
        CrossModalAlignmentModel(**kwargs)


def test_invalid_gated_interaction_type_fails():
    with pytest.raises(TypeError):
        CrossModalAlignmentModel(
            gated_interaction="yes",
        )
