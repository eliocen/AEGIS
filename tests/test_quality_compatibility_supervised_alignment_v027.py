"""
AEGIS v0.27 Step 11C M4qc alignment-model integration tests.

M4qc adds explicit cross-modal compatibility supervision to M4q while
preserving the exact M1b primary fusion pathway. The central causal invariant
is therefore:

    z_M4qc == z_M4q == z_M1b

for identical common component state and inputs.
"""

from __future__ import annotations

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.compatibility_estimator import (
    CrossModalCompatibilityEstimator,
)
from aegis.reliability.quality_estimator import ModalityQualityEstimator


def _inputs():
    generator = torch.Generator().manual_seed(11103)
    return (
        torch.randn(6, 24, generator=generator),
        torch.randn(6, 20, generator=generator),
    )


def _m1b():
    return CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )


def _m4q():
    return CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
        quality_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
    )


def _m4qc():
    return CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
        quality_compatibility_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
        compatibility_dropout=0.0,
    )


def test_m4qc_flag_must_be_boolean():
    with pytest.raises(
        TypeError,
        match="quality_compatibility_supervised must be a bool",
    ):
        CrossModalAlignmentModel(
            quality_compatibility_supervised="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "other_flag",
    [
        "evidence_aware",
        "interaction_only",
        "gated_interaction",
        "reliability_only",
        "interaction_reliability",
        "quality_supervised",
    ],
)
def test_m4qc_is_exclusive_with_other_experimental_flags(other_flag):
    kwargs = {
        "quality_compatibility_supervised": True,
        other_flag: True,
    }
    with pytest.raises(
        ValueError,
        match=(
            "quality_compatibility_supervised cannot be combined with "
            "another experimental fusion flag"
        ),
    ):
        CrossModalAlignmentModel(**kwargs)


def test_m4qc_architecture_identity_and_heads():
    model = _m4qc()

    assert model.quality_compatibility_supervised is True
    assert model.quality_supervised is False
    assert model.fusion_architecture == "quality_compatibility_supervised"
    assert model.gated_interaction_fusion is not None
    assert isinstance(model.text_quality_estimator, ModalityQualityEstimator)
    assert isinstance(model.vision_quality_estimator, ModalityQualityEstimator)
    assert isinstance(
        model.compatibility_estimator,
        CrossModalCompatibilityEstimator,
    )


def test_non_m4qc_paths_do_not_create_compatibility_head():
    assert _m1b().compatibility_estimator is None
    assert _m4q().compatibility_estimator is None


def test_m4qc_forward_returns_qt_qv_and_ctv():
    model = _m4qc()
    model.eval()
    text, vision = _inputs()

    out = model(text, vision, compute_loss=False)

    assert out["text_quality"].shape == (6, 1)
    assert out["vision_quality"].shape == (6, 1)
    assert out["compatibility_score"].shape == (6, 1)
    for key in (
        "text_quality",
        "vision_quality",
        "compatibility_score",
    ):
        assert torch.all(out[key] >= 0.0)
        assert torch.all(out[key] <= 1.0)


def test_m4qc_fused_embedding_is_exactly_m1b_for_common_state():
    m1b = _m1b()
    m4qc = _m4qc()

    incompatible = m4qc.load_state_dict(
        m1b.state_dict(),
        strict=False,
    )
    assert all(
        key.startswith(
            (
                "text_quality_estimator.",
                "vision_quality_estimator.",
                "compatibility_estimator.",
            )
        )
        for key in incompatible.missing_keys
    )
    assert incompatible.unexpected_keys == []

    m1b.eval()
    m4qc.eval()
    text, vision = _inputs()

    out_m1b = m1b(text, vision, compute_loss=False)
    out_m4qc = m4qc(text, vision, compute_loss=False)

    assert torch.equal(
        out_m1b["fused_embedding"],
        out_m4qc["fused_embedding"],
    )


def test_m4qc_preserves_m4q_outputs_when_m4q_state_is_loaded():
    m4q = _m4q()
    m4qc = _m4qc()

    incompatible = m4qc.load_state_dict(
        m4q.state_dict(),
        strict=False,
    )
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("compatibility_estimator.")
        for key in incompatible.missing_keys
    )

    m4q.eval()
    m4qc.eval()
    text, vision = _inputs()

    out_q = m4q(text, vision, compute_loss=False)
    out_qc = m4qc(text, vision, compute_loss=False)

    assert torch.equal(out_q["aligned_text"], out_qc["aligned_text"])
    assert torch.equal(out_q["aligned_vision"], out_qc["aligned_vision"])
    assert torch.equal(out_q["fused_embedding"], out_qc["fused_embedding"])
    assert torch.equal(out_q["text_quality"], out_qc["text_quality"])
    assert torch.equal(out_q["vision_quality"], out_qc["vision_quality"])


def test_fused_only_backward_does_not_touch_any_diagnostic_head():
    model = _m4qc()
    model.train()
    text, vision = _inputs()

    out = model(text, vision)
    out["fused_embedding"].square().mean().backward()

    for estimator in (
        model.text_quality_estimator,
        model.vision_quality_estimator,
        model.compatibility_estimator,
    ):
        assert estimator is not None
        for parameter in estimator.parameters():
            assert parameter.grad is None


def test_compatibility_loss_reaches_compatibility_head_and_projections():
    model = _m4qc()
    model.train()
    text, vision = _inputs()

    out = model(text, vision)
    target = torch.tensor(
        [[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]
    )
    loss = torch.nn.functional.binary_cross_entropy(
        out["compatibility_score"],
        target,
    )
    loss.backward()

    assert model.compatibility_estimator is not None
    assert all(
        parameter.grad is not None
        for parameter in model.compatibility_estimator.parameters()
        if parameter.requires_grad
    )
    assert any(
        parameter.grad is not None
        for parameter in model.text_projection.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.vision_projection.parameters()
    )


def test_compatibility_only_backward_does_not_touch_quality_heads():
    model = _m4qc()
    model.train()
    text, vision = _inputs()

    out = model(text, vision)
    out["compatibility_score"].mean().backward()

    for estimator in (
        model.text_quality_estimator,
        model.vision_quality_estimator,
    ):
        assert estimator is not None
        for parameter in estimator.parameters():
            assert parameter.grad is None


def test_m4qc_parameter_increment_over_m4q_is_exactly_one_compatibility_head():
    m4q = _m4q()
    m4qc = _m4qc()

    m4q_count = sum(p.numel() for p in m4q.parameters())
    m4qc_count = sum(p.numel() for p in m4qc.parameters())

    assert m4qc.compatibility_estimator is not None
    expected_increment = sum(
        p.numel()
        for p in m4qc.compatibility_estimator.parameters()
    )
    assert m4qc_count - m4q_count == expected_increment
