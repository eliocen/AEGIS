"""Integration invariants for M4qcs-w, M4qcs-i, and M4qcs."""

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.reliability_controller import SelectiveReliabilityController


def _model(mode):
    return CrossModalAlignmentModel(
        shared_dim=32,
        quality_compatibility_selective_fusion=mode,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )


def _inputs():
    torch.manual_seed(2802)
    return torch.randn(4, 768), torch.randn(4, 512)


@pytest.mark.parametrize("mode", ["weights_only", "interaction_only", "combined"])
def test_variants_initialize_and_emit_required_contract(mode):
    model = _model(mode)
    out = model(*_inputs())
    assert isinstance(model.selective_reliability_controller,
                      SelectiveReliabilityController)
    assert model.reliability_controller is None
    for key in (
        "text_quality", "vision_quality", "compatibility",
        "text_weight", "vision_weight", "interaction_multiplier",
        "effective_text_reliability", "effective_vision_reliability",
    ):
        assert key in out
    assert out["fused_embedding"].shape == (4, 32)


@pytest.mark.parametrize("mode", ["weights_only", "interaction_only", "combined"])
def test_exact_fusion_identity(mode):
    model = _model(mode).eval()
    out = model(*_inputs())
    expected = (
        out["text_weight"] * out["aligned_text"]
        + out["vision_weight"] * out["aligned_vision"]
        + out["interaction_multiplier"] * out["scaled_interaction"]
    )
    assert torch.allclose(out["fused_embedding"], expected)


def test_classification_path_cannot_rewrite_q_or_c():
    model = _model("combined").train()
    out = model(*_inputs())
    out["text_quality"].retain_grad()
    out["vision_quality"].retain_grad()
    out["compatibility"].retain_grad()
    out["fused_embedding"].square().mean().backward()
    assert out["text_quality"].grad is None
    assert out["vision_quality"].grad is None
    assert out["compatibility"].grad is None


def test_selective_path_rejects_all_legacy_experimental_flags():
    for flag in (
        "evidence_aware", "interaction_only", "gated_interaction",
        "reliability_only", "interaction_reliability", "quality_supervised",
        "quality_compatibility_supervised", "quality_compatibility_fusion",
    ):
        with pytest.raises(ValueError):
            CrossModalAlignmentModel(
                quality_compatibility_selective_fusion="combined",
                **{flag: True},
            )


def test_selective_variants_preserve_trainable_parameter_count():
    counts = {
        mode: sum(p.numel() for p in _model(mode).parameters())
        for mode in ("weights_only", "interaction_only", "combined")
    }
    assert len(set(counts.values())) == 1
