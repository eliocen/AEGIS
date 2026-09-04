"""
AEGIS v0.27 Step 11B standalone M4qc compatibility-estimator tests.

These tests validate only the compatibility estimator frozen by Step 11A.
They do not integrate M4qc into CrossModalAlignmentModel, modify fusion,
construct the training loss, train checkpoints, or evaluate H4-C/H5-C.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from aegis.reliability.compatibility_estimator import (
    CrossModalCompatibilityEstimator,
)


def _make_pair(
    *,
    batch_size: int = 7,
    shared_dim: int = 16,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11027)
    text = torch.randn(
        batch_size,
        shared_dim,
        generator=generator,
        dtype=dtype,
    )
    vision = torch.randn(
        batch_size,
        shared_dim,
        generator=generator,
        dtype=dtype,
    )
    return text, vision


def test_default_architecture_matches_step11a_protocol() -> None:
    model = CrossModalCompatibilityEstimator(128)

    assert model.shared_dim == 128
    assert model.feature_dim == 513
    assert model.hidden_dims == (256, 64)
    assert model.dropout_probability == 0.1
    assert model.cosine_eps == 1e-8

    layers = list(model.network)

    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 4 * 128 + 1
    assert layers[0].out_features == 256

    assert isinstance(layers[1], nn.GELU)
    assert isinstance(layers[2], nn.Dropout)
    assert layers[2].p == 0.1

    assert isinstance(layers[3], nn.Linear)
    assert layers[3].in_features == 256
    assert layers[3].out_features == 64

    assert isinstance(layers[4], nn.GELU)
    assert isinstance(layers[5], nn.Dropout)
    assert layers[5].p == 0.1

    assert isinstance(layers[6], nn.Linear)
    assert layers[6].in_features == 64
    assert layers[6].out_features == 1

    assert isinstance(layers[7], nn.Sigmoid)


def test_pair_feature_shape_is_4d_plus_1() -> None:
    model = CrossModalCompatibilityEstimator(11)
    text, vision = _make_pair(batch_size=5, shared_dim=11)

    features = model.build_pair_features(text, vision)

    assert features.shape == (5, 45)


def test_pair_feature_construction_matches_frozen_equation() -> None:
    model = CrossModalCompatibilityEstimator(3, dropout=0.0)

    text = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 0.0, -1.0],
        ]
    )
    vision = torch.tensor(
        [
            [3.0, 1.0, -1.0],
            [1.0, 4.0, 2.0],
        ]
    )

    features = model.build_pair_features(text, vision)

    expected_cosine = torch.nn.functional.cosine_similarity(
        text,
        vision,
        dim=1,
        eps=1e-8,
    ).unsqueeze(1)

    expected = torch.cat(
        (
            text,
            vision,
            torch.abs(text - vision),
            text * vision,
            expected_cosine,
        ),
        dim=1,
    )

    assert torch.equal(features, expected)


def test_pair_feature_block_order_is_text_vision_abs_product_cosine() -> None:
    model = CrossModalCompatibilityEstimator(2, dropout=0.0)

    text = torch.tensor([[1.0, 2.0]])
    vision = torch.tensor([[3.0, 4.0]])

    features = model.build_pair_features(text, vision)

    assert torch.equal(features[:, 0:2], text)
    assert torch.equal(features[:, 2:4], vision)
    assert torch.equal(features[:, 4:6], torch.tensor([[2.0, 2.0]]))
    assert torch.equal(features[:, 6:8], torch.tensor([[3.0, 8.0]]))
    assert features[:, 8:9].shape == (1, 1)


def test_identical_nonzero_vectors_have_unit_cosine_feature() -> None:
    model = CrossModalCompatibilityEstimator(4, dropout=0.0)
    text = torch.tensor([[1.0, 2.0, -3.0, 4.0]])
    vision = text.clone()

    features = model.build_pair_features(text, vision)

    cosine = features[:, -1]
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-7)


def test_zero_vectors_produce_finite_cosine_feature() -> None:
    model = CrossModalCompatibilityEstimator(4, dropout=0.0)
    text = torch.zeros(3, 4)
    vision = torch.zeros(3, 4)

    features = model.build_pair_features(text, vision)

    assert torch.isfinite(features).all()
    assert torch.equal(features[:, -1], torch.zeros(3))


def test_forward_returns_batch_by_one() -> None:
    model = CrossModalCompatibilityEstimator(16)
    text, vision = _make_pair(shared_dim=16)

    out = model(text, vision)

    assert out.shape == (7, 1)


def test_output_is_bounded_between_zero_and_one() -> None:
    model = CrossModalCompatibilityEstimator(16)
    text, vision = _make_pair(
        batch_size=100,
        shared_dim=16,
    )

    out = model(text * 10.0, vision * 10.0)

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_float64_is_supported_when_model_is_float64() -> None:
    model = CrossModalCompatibilityEstimator(8).double()
    text, vision = _make_pair(
        batch_size=4,
        shared_dim=8,
        dtype=torch.float64,
    )

    out = model(text, vision)

    assert out.dtype == torch.float64


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_estimator_runs_on_cuda() -> None:
    model = CrossModalCompatibilityEstimator(8).cuda()
    text = torch.randn(4, 8, device="cuda")
    vision = torch.randn(4, 8, device="cuda")

    out = model(text, vision)

    assert out.is_cuda
    assert out.shape == (4, 1)


def test_gradients_reach_both_inputs_and_all_linear_parameters() -> None:
    model = CrossModalCompatibilityEstimator(8)
    text, vision = _make_pair(batch_size=10, shared_dim=8)
    text.requires_grad_(True)
    vision.requires_grad_(True)

    target = torch.rand(10, 1)
    out = model(text, vision)
    loss = nn.functional.binary_cross_entropy(out, target)
    loss.backward()

    assert text.grad is not None
    assert vision.grad is not None
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(vision.grad).all()

    linear_parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    ]
    assert linear_parameters
    for parameter in linear_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_eval_mode_is_deterministic() -> None:
    model = CrossModalCompatibilityEstimator(16)
    model.eval()
    text, vision = _make_pair(shared_dim=16)

    first = model(text, vision)
    second = model(text, vision)

    assert torch.equal(first, second)


def test_training_mode_uses_dropout() -> None:
    torch.manual_seed(42)
    model = CrossModalCompatibilityEstimator(
        32,
        hidden_dims=(256, 64),
        dropout=0.5,
    )
    model.train()
    text, vision = _make_pair(batch_size=64, shared_dim=32)

    first = model(text, vision)
    second = model(text, vision)

    assert not torch.equal(first, second)


def test_custom_valid_dimensions_are_supported() -> None:
    model = CrossModalCompatibilityEstimator(
        20,
        hidden_dims=(40, 10),
        dropout=0.2,
    )
    text, vision = _make_pair(batch_size=3, shared_dim=20)

    assert model.shared_dim == 20
    assert model.feature_dim == 81
    assert model.hidden_dims == (40, 10)
    assert model.dropout_probability == 0.2
    assert model(text, vision).shape == (3, 1)


def test_architecture_metadata_is_stable_and_json_serializable() -> None:
    model = CrossModalCompatibilityEstimator(128)

    metadata = model.architecture_metadata()

    assert metadata == {
        "module": "CrossModalCompatibilityEstimator",
        "shared_dim": 128,
        "pair_feature_dim": 513,
        "pair_features": [
            "text",
            "vision",
            "absolute_difference",
            "elementwise_product",
            "cosine_similarity",
        ],
        "hidden_dims": [256, 64],
        "dropout": 0.1,
        "cosine_eps": 1e-8,
        "output_dim": 1,
        "hidden_activation": "GELU",
        "output_activation": "Sigmoid",
        "semantic_role": "cross_modal_compatibility",
        "cross_modal_input": True,
        "ordered_modalities": ["text", "vision"],
        "affects_primary_fusion": False,
    }

    json.dumps(metadata)


@pytest.mark.parametrize("bad_dim", [0, -1])
def test_invalid_shared_dim_is_rejected(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="shared_dim must be > 0"):
        CrossModalCompatibilityEstimator(bad_dim)


def test_boolean_shared_dim_is_rejected() -> None:
    with pytest.raises(TypeError, match="shared_dim must be an integer"):
        CrossModalCompatibilityEstimator(True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_hidden_dims",
    [
        (),
        (256,),
        (256, 64, 32),
        (0, 64),
        (256, 0),
        (-1, 64),
    ],
)
def test_invalid_hidden_dims_are_rejected(
    bad_hidden_dims: tuple[int, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CrossModalCompatibilityEstimator(
            16,
            hidden_dims=bad_hidden_dims,
        )


@pytest.mark.parametrize("bad_dropout", [-0.01, 1.0, 1.5])
def test_invalid_dropout_is_rejected(bad_dropout: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        CrossModalCompatibilityEstimator(
            16,
            dropout=bad_dropout,
        )


def test_boolean_dropout_is_rejected() -> None:
    with pytest.raises(TypeError, match="dropout must be a real number"):
        CrossModalCompatibilityEstimator(
            16,
            dropout=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_eps", [0.0, -1e-8])
def test_nonpositive_cosine_eps_is_rejected(bad_eps: float) -> None:
    with pytest.raises(ValueError, match="cosine_eps must be > 0"):
        CrossModalCompatibilityEstimator(
            16,
            cosine_eps=bad_eps,
        )


def test_boolean_cosine_eps_is_rejected() -> None:
    with pytest.raises(TypeError, match="cosine_eps must be a positive real"):
        CrossModalCompatibilityEstimator(
            16,
            cosine_eps=True,  # type: ignore[arg-type]
        )


def test_forward_rejects_non_tensor_text() -> None:
    model = CrossModalCompatibilityEstimator(4)
    vision = torch.randn(3, 4)

    with pytest.raises(TypeError, match="text_embedding must be a torch.Tensor"):
        model([[1.0] * 4] * 3, vision)  # type: ignore[arg-type]


def test_forward_rejects_non_tensor_vision() -> None:
    model = CrossModalCompatibilityEstimator(4)
    text = torch.randn(3, 4)

    with pytest.raises(TypeError, match="vision_embedding must be a torch.Tensor"):
        model(text, [[1.0] * 4] * 3)  # type: ignore[arg-type]


def test_forward_rejects_wrong_text_rank() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match=r"text_embedding must have shape"):
        model(torch.randn(2, 3, 4), torch.randn(2, 4))


def test_forward_rejects_wrong_vision_rank() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match=r"vision_embedding must have shape"):
        model(torch.randn(2, 4), torch.randn(2, 3, 4))


def test_forward_rejects_empty_batch() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match="at least one sample"):
        model(torch.empty(0, 4), torch.empty(0, 4))


def test_forward_rejects_unequal_batch_sizes() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match="batch sizes must match"):
        model(torch.randn(3, 4), torch.randn(4, 4))


def test_forward_rejects_wrong_text_feature_dimension() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match="text_embedding feature dimension"):
        model(torch.randn(3, 5), torch.randn(3, 4))


def test_forward_rejects_wrong_vision_feature_dimension() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(ValueError, match="vision_embedding feature dimension"):
        model(torch.randn(3, 4), torch.randn(3, 5))


def test_forward_rejects_integer_text_tensor() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(TypeError, match="text_embedding must have a floating-point"):
        model(
            torch.ones(3, 4, dtype=torch.int64),
            torch.randn(3, 4),
        )


def test_forward_rejects_integer_vision_tensor() -> None:
    model = CrossModalCompatibilityEstimator(4)

    with pytest.raises(TypeError, match="vision_embedding must have a floating-point"):
        model(
            torch.randn(3, 4),
            torch.ones(3, 4, dtype=torch.int64),
        )


def test_forward_rejects_mismatched_dtypes() -> None:
    model = CrossModalCompatibilityEstimator(4).double()

    with pytest.raises(TypeError, match="must have the same dtype"):
        model(
            torch.randn(3, 4, dtype=torch.float64),
            torch.randn(3, 4, dtype=torch.float32),
        )


def test_forward_rejects_nonfinite_text_values() -> None:
    model = CrossModalCompatibilityEstimator(4)
    text = torch.randn(3, 4)
    vision = torch.randn(3, 4)
    text[0, 0] = float("nan")

    with pytest.raises(ValueError, match="text_embedding contains NaN or infinite"):
        model(text, vision)


def test_forward_rejects_nonfinite_vision_values() -> None:
    model = CrossModalCompatibilityEstimator(4)
    text = torch.randn(3, 4)
    vision = torch.randn(3, 4)
    vision[0, 0] = float("inf")

    with pytest.raises(ValueError, match="vision_embedding contains NaN or infinite"):
        model(text, vision)


def test_feature_construction_does_not_modify_inputs() -> None:
    model = CrossModalCompatibilityEstimator(8)
    text, vision = _make_pair(batch_size=5, shared_dim=8)
    text_original = text.clone()
    vision_original = vision.clone()

    _ = model.build_pair_features(text, vision)

    assert torch.equal(text, text_original)
    assert torch.equal(vision, vision_original)


def test_forward_does_not_modify_inputs() -> None:
    model = CrossModalCompatibilityEstimator(8)
    text, vision = _make_pair(batch_size=5, shared_dim=8)
    text_original = text.clone()
    vision_original = vision.clone()

    _ = model(text, vision)

    assert torch.equal(text, text_original)
    assert torch.equal(vision, vision_original)


def test_parameter_count_matches_frozen_architecture() -> None:
    shared_dim = 100
    model = CrossModalCompatibilityEstimator(shared_dim)

    actual = sum(parameter.numel() for parameter in model.parameters())

    feature_dim = 4 * shared_dim + 1
    expected = (
        feature_dim * 256
        + 256
        + 256 * 64
        + 64
        + 64 * 1
        + 1
    )

    assert actual == expected


def test_estimator_contains_no_quality_or_fusion_modules() -> None:
    model = CrossModalCompatibilityEstimator(16)

    module_names = {name for name, _ in model.named_modules()}

    assert "text_quality_estimator" not in module_names
    assert "vision_quality_estimator" not in module_names
    assert "fusion" not in module_names


def test_step11b_scope_does_not_encode_a_compatibility_threshold() -> None:
    model = CrossModalCompatibilityEstimator(16)
    metadata = model.architecture_metadata()

    forbidden_keys = {
        "classification_threshold",
        "compatibility_threshold",
        "h4_threshold",
        "h5_threshold",
        "auc_threshold",
    }

    assert forbidden_keys.isdisjoint(metadata.keys())
