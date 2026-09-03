"""
Tests for AEGIS v0.25.2 M1 interaction-only alignment.
"""

from __future__ import annotations

import math

import pytest
import torch

from aegis.alignment.interaction_fusion import (
    InteractionOnlyEvidenceFusion,
)
from aegis.alignment.model import (
    CrossModalAlignmentModel,
)


def make_inputs(
    batch_size: int = 4,
    text_dim: int = 8,
    vision_dim: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)

    text = torch.randn(
        batch_size,
        text_dim,
    )

    vision = torch.randn(
        batch_size,
        vision_dim,
    )

    return text, vision


def test_interaction_only_block_shapes():
    dimension = 8

    block = InteractionOnlyEvidenceFusion(
        dimension=dimension,
        interaction_dropout=0.0,
    )

    first = torch.randn(5, dimension)
    second = torch.randn(5, dimension)

    output = block(
        first,
        second,
    )

    assert output["difference"].shape == (5, dimension)
    assert output["product"].shape == (5, dimension)
    assert output["cosine_similarity"].shape == (5, 1)
    assert output["interaction_embedding"].shape == (5, dimension)
    assert output["base_fusion"].shape == (5, dimension)
    assert output["scaled_interaction"].shape == (5, dimension)
    assert output["fused_embedding"].shape == (5, dimension)


def test_interaction_scale_is_deterministic_inverse_sqrt_dimension():
    dimension = 16

    block = InteractionOnlyEvidenceFusion(
        dimension=dimension,
        interaction_dropout=0.0,
    )

    assert block.interaction_scale == pytest.approx(
        1.0 / math.sqrt(dimension)
    )


def test_interaction_only_contains_no_reliability_estimators():
    block = InteractionOnlyEvidenceFusion(
        dimension=8,
    )

    parameter_names = [
        name for name, _ in block.named_parameters()
    ]

    assert not any(
        "reliability" in name
        for name in parameter_names
    )

    assert not any(
        "adaptive" in name
        for name in parameter_names
    )


def test_default_model_remains_legacy():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
    )

    assert model.fusion_architecture == "legacy"
    assert model.evidence_aware is False
    assert model.interaction_only is False
    assert model.interaction_fusion is None
    assert model.evidence_integration is None


def test_explicit_false_flags_match_default_exactly():
    torch.manual_seed(55)

    default_model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
    )

    explicit_model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        evidence_aware=False,
        interaction_only=False,
    )

    explicit_model.load_state_dict(
        default_model.state_dict(),
        strict=True,
    )

    default_model.eval()
    explicit_model.eval()

    text, vision = make_inputs(
        text_dim=8,
        vision_dim=6,
    )

    with torch.no_grad():
        default_output = default_model(
            text,
            vision,
        )

        explicit_output = explicit_model(
            text,
            vision,
        )

    assert torch.equal(
        default_output["aligned_text"],
        explicit_output["aligned_text"],
    )

    assert torch.equal(
        default_output["aligned_vision"],
        explicit_output["aligned_vision"],
    )

    assert torch.equal(
        default_output["fused_embedding"],
        explicit_output["fused_embedding"],
    )


def test_m1_initializes_only_interaction_ablation_block():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        interaction_only=True,
    )

    assert model.fusion_architecture == "interaction_only"
    assert model.interaction_fusion is not None
    assert model.evidence_integration is None


def test_m1_output_contract_and_diagnostics():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        interaction_only=True,
        evidence_interaction_dropout=0.0,
    )

    model.eval()

    text, vision = make_inputs(
        text_dim=8,
        vision_dim=6,
    )

    with torch.no_grad():
        output = model(
            text,
            vision,
            compute_loss=True,
        )

    required = {
        "aligned_text",
        "aligned_vision",
        "fused_embedding",
        "evidence_difference",
        "evidence_product",
        "cosine_similarity",
        "interaction_embedding",
        "interaction_base_fusion",
        "scaled_interaction",
        "interaction_scale",
        "alignment_loss",
    }

    assert required.issubset(
        output.keys()
    )

    assert output["fused_embedding"].shape == (4, 4)
    assert output["cosine_similarity"].shape == (4, 1)
    assert output["alignment_loss"].ndim == 0


def test_m1_does_not_expose_reliability_or_adaptive_weights():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        interaction_only=True,
    )

    model.eval()

    text, vision = make_inputs(
        text_dim=8,
        vision_dim=6,
    )

    with torch.no_grad():
        output = model(
            text,
            vision,
        )

    forbidden = {
        "text_reliability",
        "vision_reliability",
        "text_weight",
        "vision_weight",
        "evidence_weights",
    }

    assert forbidden.isdisjoint(
        output.keys()
    )


def test_m1_gradients_reach_interaction_projection():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        dropout=0.0,
        interaction_only=True,
        evidence_interaction_dropout=0.0,
    )

    text, vision = make_inputs(
        text_dim=8,
        vision_dim=6,
    )

    output = model(
        text,
        vision,
    )

    # Avoid a LayerNorm-invariant squared-norm objective.
    loss = output["fused_embedding"][:, 0].sum()

    loss.backward()

    interaction_parameters = list(
        model
        .interaction_fusion
        .interaction
        .parameters()
    )

    assert interaction_parameters
    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in interaction_parameters
    )


def test_m1_and_m3_are_mutually_exclusive():
    with pytest.raises(
        ValueError,
        match=(
            "evidence_aware and interaction_only "
            "cannot both be True"
        ),
    ):
        CrossModalAlignmentModel(
            text_dim=8,
            vision_dim=6,
            shared_dim=4,
            evidence_aware=True,
            interaction_only=True,
        )


def test_invalid_interaction_only_type_fails():
    with pytest.raises(
        TypeError,
        match="interaction_only must be a bool",
    ):
        CrossModalAlignmentModel(
            interaction_only="yes",
        )


def test_m3_remains_available():
    model = CrossModalAlignmentModel(
        text_dim=8,
        vision_dim=6,
        shared_dim=4,
        evidence_aware=True,
    )

    assert model.fusion_architecture == "evidence_aware"
    assert model.evidence_integration is not None
    assert model.interaction_fusion is None
