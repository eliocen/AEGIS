"""
AEGIS v0.27 M4q alignment-model integration tests.

The central causal invariant is:

    z_M4q == z_M1b

when common parameters and inputs are identical. The quality heads are
diagnostic-only with respect to the forward fusion equation.
"""

from __future__ import annotations

import pytest
import torch

from aegis.alignment.model import CrossModalAlignmentModel
from aegis.reliability.quality_estimator import ModalityQualityEstimator


def _make_inputs(
    *,
    batch_size: int = 6,
    text_dim: int = 24,
    vision_dim: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7001)
    return (
        torch.randn(batch_size, text_dim),
        torch.randn(batch_size, vision_dim),
    )


def _make_m1b() -> CrossModalAlignmentModel:
    return CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
        gated_interaction=True,
        evidence_interaction_dropout=0.0,
    )


def _make_m4q() -> CrossModalAlignmentModel:
    return CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
        quality_supervised=True,
        evidence_interaction_dropout=0.0,
        quality_dropout=0.0,
    )


def _copy_m1b_common_state_into_m4q(
    m1b: CrossModalAlignmentModel,
    m4q: CrossModalAlignmentModel,
) -> None:
    incompatible = m4q.load_state_dict(
        m1b.state_dict(),
        strict=False,
    )

    assert set(incompatible.missing_keys) == {
        "text_quality_estimator.network.0.weight",
        "text_quality_estimator.network.0.bias",
        "text_quality_estimator.network.3.weight",
        "text_quality_estimator.network.3.bias",
        "text_quality_estimator.network.6.weight",
        "text_quality_estimator.network.6.bias",
        "vision_quality_estimator.network.0.weight",
        "vision_quality_estimator.network.0.bias",
        "vision_quality_estimator.network.3.weight",
        "vision_quality_estimator.network.3.bias",
        "vision_quality_estimator.network.6.weight",
        "vision_quality_estimator.network.6.bias",
    }
    assert incompatible.unexpected_keys == []


def test_quality_supervised_flag_must_be_boolean() -> None:
    with pytest.raises(
        TypeError,
        match="quality_supervised must be a bool",
    ):
        CrossModalAlignmentModel(
            quality_supervised="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "other_flag",
    [
        "evidence_aware",
        "interaction_only",
        "gated_interaction",
        "reliability_only",
        "interaction_reliability",
    ],
)
def test_quality_supervised_is_exclusive_with_existing_experimental_flags(
    other_flag: str,
) -> None:
    kwargs = {
        "quality_supervised": True,
        other_flag: True,
    }

    with pytest.raises(
        ValueError,
        match=(
            "quality_supervised cannot be combined with another "
            "experimental fusion flag"
        ),
    ):
        CrossModalAlignmentModel(**kwargs)


def test_quality_supervised_architecture_identity() -> None:
    model = _make_m4q()

    assert model.quality_supervised is True
    assert model.fusion_architecture == "quality_supervised"


def test_m4q_uses_m1b_fusion_module_and_two_separate_quality_heads() -> None:
    model = _make_m4q()

    assert model.gated_interaction_fusion is not None
    assert isinstance(
        model.text_quality_estimator,
        ModalityQualityEstimator,
    )
    assert isinstance(
        model.vision_quality_estimator,
        ModalityQualityEstimator,
    )
    assert model.text_quality_estimator is not model.vision_quality_estimator

    text_ids = {
        id(parameter)
        for parameter in model.text_quality_estimator.parameters()
    }
    vision_ids = {
        id(parameter)
        for parameter in model.vision_quality_estimator.parameters()
    }
    assert text_ids.isdisjoint(vision_ids)


def test_m4q_outputs_quality_scores_with_expected_shapes_and_bounds() -> None:
    model = _make_m4q()
    model.eval()
    text, vision = _make_inputs()

    out = model(text, vision)

    assert out["text_quality"].shape == (6, 1)
    assert out["vision_quality"].shape == (6, 1)
    assert torch.all(out["text_quality"] >= 0.0)
    assert torch.all(out["text_quality"] <= 1.0)
    assert torch.all(out["vision_quality"] >= 0.0)
    assert torch.all(out["vision_quality"] <= 1.0)


def test_m4q_exposes_all_m1b_diagnostics_plus_quality() -> None:
    m1b = _make_m1b()
    m4q = _make_m4q()
    _copy_m1b_common_state_into_m4q(m1b, m4q)

    m1b.eval()
    m4q.eval()

    text, vision = _make_inputs()

    m1b_out = m1b(text, vision)
    m4q_out = m4q(text, vision)

    assert set(m1b_out).issubset(set(m4q_out))
    assert set(m4q_out) - set(m1b_out) == {
        "text_quality",
        "vision_quality",
    }


def test_m4q_fused_embedding_is_exactly_m1b_for_identical_common_state() -> None:
    m1b = _make_m1b()
    m4q = _make_m4q()
    _copy_m1b_common_state_into_m4q(m1b, m4q)

    m1b.eval()
    m4q.eval()

    text, vision = _make_inputs()

    m1b_out = m1b(text, vision)
    m4q_out = m4q(text, vision)

    assert torch.equal(
        m1b_out["fused_embedding"],
        m4q_out["fused_embedding"],
    )


@pytest.mark.parametrize(
    "key",
    [
        "aligned_text",
        "aligned_vision",
        "fused_embedding",
        "evidence_difference",
        "evidence_product",
        "cosine_similarity",
        "interaction_embedding",
        "gated_interaction_base_fusion",
        "scaled_interaction",
        "interaction_scale",
    ],
)
def test_m4q_common_forward_diagnostic_is_exactly_m1b(
    key: str,
) -> None:
    m1b = _make_m1b()
    m4q = _make_m4q()
    _copy_m1b_common_state_into_m4q(m1b, m4q)

    m1b.eval()
    m4q.eval()

    text, vision = _make_inputs()

    m1b_out = m1b(text, vision)
    m4q_out = m4q(text, vision)

    assert torch.equal(m1b_out[key], m4q_out[key])


def test_fused_only_backward_does_not_touch_quality_head_parameters() -> None:
    model = _make_m4q()
    model.train()

    text, vision = _make_inputs()
    out = model(text, vision)

    loss = out["fused_embedding"].square().mean()
    loss.backward()

    assert model.text_quality_estimator is not None
    assert model.vision_quality_estimator is not None

    for parameter in model.text_quality_estimator.parameters():
        assert parameter.grad is None

    for parameter in model.vision_quality_estimator.parameters():
        assert parameter.grad is None


def test_quality_loss_reaches_both_quality_estimators() -> None:
    model = _make_m4q()
    model.train()

    text, vision = _make_inputs()
    out = model(text, vision)

    quality_loss = (
        out["text_quality"].mean()
        + out["vision_quality"].mean()
    )
    quality_loss.backward()

    assert model.text_quality_estimator is not None
    assert model.vision_quality_estimator is not None

    for estimator in (
        model.text_quality_estimator,
        model.vision_quality_estimator,
    ):
        grads = [
            parameter.grad
            for parameter in estimator.parameters()
            if parameter.requires_grad
        ]
        assert grads
        assert all(grad is not None for grad in grads)
        assert all(torch.isfinite(grad).all() for grad in grads)


def test_m1b_does_not_create_quality_estimators_or_quality_outputs() -> None:
    model = _make_m1b()
    model.eval()

    assert model.text_quality_estimator is None
    assert model.vision_quality_estimator is None

    text, vision = _make_inputs()
    out = model(text, vision)

    assert "text_quality" not in out
    assert "vision_quality" not in out


def test_legacy_path_remains_quality_free() -> None:
    model = CrossModalAlignmentModel(
        text_dim=24,
        vision_dim=20,
        shared_dim=16,
        dropout=0.0,
    )
    model.eval()

    assert model.fusion_architecture == "legacy"
    assert model.quality_supervised is False
    assert model.text_quality_estimator is None
    assert model.vision_quality_estimator is None

    text, vision = _make_inputs()
    out = model(text, vision)

    assert set(out) == {
        "aligned_text",
        "aligned_vision",
        "fused_embedding",
    }


def test_compute_loss_semantics_are_preserved_for_m4q() -> None:
    model = _make_m4q()
    model.eval()

    text, vision = _make_inputs()
    out = model(
        text,
        vision,
        compute_loss=True,
    )

    assert "alignment_loss" in out
    assert out["alignment_loss"].ndim == 0
    assert torch.isfinite(out["alignment_loss"])


def test_quality_head_parameter_increment_matches_two_estimators() -> None:
    m1b = _make_m1b()
    m4q = _make_m4q()

    m1b_count = sum(
        parameter.numel()
        for parameter in m1b.parameters()
    )
    m4q_count = sum(
        parameter.numel()
        for parameter in m4q.parameters()
    )

    one_quality_head = (
        16 * 256
        + 256
        + 256 * 64
        + 64
        + 64
        + 1
    )

    assert m4q_count - m1b_count == 2 * one_quality_head
