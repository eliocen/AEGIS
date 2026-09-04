"""
Tests for the AEGIS v0.27 ModalityQualityEstimator.

These tests validate the standalone learned quality estimator only. M4q
integration into the alignment model is intentionally a later protocol step.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from aegis.reliability.quality_estimator import (
    ModalityQualityEstimator,
)


def test_default_architecture_matches_frozen_protocol() -> None:
    model = ModalityQualityEstimator(128)

    assert model.input_dim == 128
    assert model.hidden_dims == (256, 64)
    assert model.dropout_probability == 0.1

    layers = list(model.network)

    assert isinstance(layers[0], nn.Linear)
    assert layers[0].in_features == 128
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


def test_forward_returns_batch_by_one() -> None:
    model = ModalityQualityEstimator(32)
    x = torch.randn(7, 32)

    out = model(x)

    assert out.shape == (7, 1)


def test_output_is_bounded_between_zero_and_one() -> None:
    model = ModalityQualityEstimator(32)
    x = torch.randn(100, 32) * 10.0

    out = model(x)

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_estimator_accepts_float64_when_model_is_float64() -> None:
    model = ModalityQualityEstimator(16).double()
    x = torch.randn(4, 16, dtype=torch.float64)

    out = model(x)

    assert out.dtype == torch.float64


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_estimator_runs_on_cuda() -> None:
    model = ModalityQualityEstimator(16).cuda()
    x = torch.randn(4, 16, device="cuda")

    out = model(x)

    assert out.is_cuda
    assert out.shape == (4, 1)


def test_gradients_reach_all_linear_parameters() -> None:
    model = ModalityQualityEstimator(16)
    x = torch.randn(8, 16)
    target = torch.rand(8, 1)

    out = model(x)
    loss = torch.mean((out - target) ** 2)
    loss.backward()

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


def test_two_estimators_do_not_share_parameters() -> None:
    text_quality = ModalityQualityEstimator(32)
    vision_quality = ModalityQualityEstimator(32)

    text_parameter_ids = {
        id(parameter) for parameter in text_quality.parameters()
    }
    vision_parameter_ids = {
        id(parameter) for parameter in vision_quality.parameters()
    }

    assert text_parameter_ids.isdisjoint(vision_parameter_ids)


def test_eval_mode_is_deterministic() -> None:
    model = ModalityQualityEstimator(32)
    model.eval()
    x = torch.randn(5, 32)

    first = model(x)
    second = model(x)

    assert torch.equal(first, second)


def test_training_mode_uses_dropout() -> None:
    torch.manual_seed(42)

    model = ModalityQualityEstimator(
        64,
        hidden_dims=(256, 64),
        dropout=0.5,
    )
    model.train()
    x = torch.randn(64, 64)

    first = model(x)
    second = model(x)

    assert not torch.equal(first, second)


def test_custom_valid_dimensions_are_supported() -> None:
    model = ModalityQualityEstimator(
        20,
        hidden_dims=(40, 10),
        dropout=0.2,
    )

    assert model.input_dim == 20
    assert model.hidden_dims == (40, 10)
    assert model.dropout_probability == 0.2
    assert model(torch.randn(3, 20)).shape == (3, 1)


def test_architecture_metadata_is_stable_and_serializable() -> None:
    model = ModalityQualityEstimator(768)

    metadata = model.architecture_metadata()

    assert metadata == {
        "module": "ModalityQualityEstimator",
        "input_dim": 768,
        "hidden_dims": [256, 64],
        "dropout": 0.1,
        "output_dim": 1,
        "hidden_activation": "GELU",
        "output_activation": "Sigmoid",
        "semantic_role": "intrinsic_modality_quality",
        "cross_modal_input": False,
        "affects_primary_fusion": False,
    }


@pytest.mark.parametrize("bad_dim", [0, -1])
def test_invalid_input_dim_is_rejected(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="input_dim must be > 0"):
        ModalityQualityEstimator(bad_dim)


def test_boolean_input_dim_is_rejected() -> None:
    with pytest.raises(TypeError, match="input_dim must be an integer"):
        ModalityQualityEstimator(True)  # type: ignore[arg-type]


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
        ModalityQualityEstimator(
            32,
            hidden_dims=bad_hidden_dims,
        )


@pytest.mark.parametrize("bad_dropout", [-0.01, 1.0, 1.5])
def test_invalid_dropout_is_rejected(bad_dropout: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        ModalityQualityEstimator(
            32,
            dropout=bad_dropout,
        )


def test_boolean_dropout_is_rejected() -> None:
    with pytest.raises(TypeError, match="dropout must be a real number"):
        ModalityQualityEstimator(
            32,
            dropout=True,  # type: ignore[arg-type]
        )


def test_forward_rejects_wrong_rank() -> None:
    model = ModalityQualityEstimator(32)

    with pytest.raises(ValueError, match=r"\[batch, input_dim\]"):
        model(torch.randn(2, 3, 32))


def test_forward_rejects_wrong_feature_dimension() -> None:
    model = ModalityQualityEstimator(32)

    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(4, 31))


def test_forward_rejects_integer_tensor() -> None:
    model = ModalityQualityEstimator(4)

    with pytest.raises(TypeError, match="floating-point dtype"):
        model(torch.ones(3, 4, dtype=torch.int64))


def test_forward_rejects_nonfinite_values() -> None:
    model = ModalityQualityEstimator(4)
    x = torch.randn(3, 4)
    x[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        model(x)


def test_forward_does_not_modify_input_tensor() -> None:
    model = ModalityQualityEstimator(8)
    x = torch.randn(5, 8)
    original = x.clone()

    _ = model(x)

    assert torch.equal(x, original)


def test_parameter_count_matches_architecture() -> None:
    input_dim = 100
    model = ModalityQualityEstimator(input_dim)

    actual = sum(parameter.numel() for parameter in model.parameters())

    expected = (
        input_dim * 256
        + 256
        + 256 * 64
        + 64
        + 64 * 1
        + 1
    )

    assert actual == expected
