from __future__ import annotations

import json
import math

import pytest
import torch

from aegis.reliability.reliability_controller import (
    DEFAULT_EPSILON,
    STEP13A_PROTOCOL_VERSION,
    DeterministicReliabilityController,
    ReliabilityControllerOutput,
)


def _scores(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    q_text = torch.tensor(
        [[0.9], [0.6], [0.2]],
        dtype=dtype,
        device=device,
    )
    q_vision = torch.tensor(
        [[0.7], [0.8], [0.3]],
        dtype=dtype,
        device=device,
    )
    compatibility = torch.tensor(
        [[0.95], [0.5], [0.1]],
        dtype=dtype,
        device=device,
    )
    return q_text, q_vision, compatibility


def test_default_protocol_epsilon_is_frozen_value():
    controller = DeterministicReliabilityController()
    assert controller.epsilon == pytest.approx(0.10)
    assert DEFAULT_EPSILON == pytest.approx(0.10)


def test_controller_is_parameter_free():
    controller = DeterministicReliabilityController()
    assert list(controller.parameters()) == []
    assert controller.parameter_count == 0


def test_forward_returns_structured_output():
    controller = DeterministicReliabilityController()
    output = controller(*_scores())

    assert isinstance(output, ReliabilityControllerOutput)
    assert output.effective_text_reliability.shape == (3, 1)
    assert output.effective_vision_reliability.shape == (3, 1)
    assert output.text_weight.shape == (3, 1)
    assert output.vision_weight.shape == (3, 1)
    assert output.weights.shape == (3, 2)
    assert output.interaction_multiplier.shape == (3, 1)


def test_exact_frozen_equations():
    controller = DeterministicReliabilityController(epsilon=0.10)
    q_text, q_vision, compatibility = _scores()

    output = controller(
        q_text,
        q_vision,
        compatibility,
    )

    factor = 0.10 + 0.90 * compatibility
    expected_r_text = q_text * factor
    expected_r_vision = q_vision * factor
    denominator = expected_r_text + expected_r_vision + 0.20
    expected_alpha_text = (expected_r_text + 0.10) / denominator
    expected_alpha_vision = (expected_r_vision + 0.10) / denominator

    assert torch.allclose(
        output.effective_text_reliability,
        expected_r_text,
    )
    assert torch.allclose(
        output.effective_vision_reliability,
        expected_r_vision,
    )
    assert torch.allclose(
        output.text_weight,
        expected_alpha_text,
    )
    assert torch.allclose(
        output.vision_weight,
        expected_alpha_vision,
    )
    assert torch.allclose(
        output.interaction_multiplier,
        compatibility,
    )


def test_weights_are_strictly_bounded_and_normalized():
    controller = DeterministicReliabilityController()

    q_text = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    q_vision = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    compatibility = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

    output = controller(q_text, q_vision, compatibility)

    assert torch.all(output.text_weight > 0.0)
    assert torch.all(output.text_weight < 1.0)
    assert torch.all(output.vision_weight > 0.0)
    assert torch.all(output.vision_weight < 1.0)
    assert torch.allclose(
        output.text_weight + output.vision_weight,
        torch.ones_like(output.text_weight),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "quality,compatibility",
    [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.5, 0.25),
        (0.8, 0.75),
        (1.0, 1.0),
    ],
)
def test_equal_quality_gives_exactly_symmetric_weights(
    quality: float,
    compatibility: float,
):
    controller = DeterministicReliabilityController()

    q_text = torch.tensor([[quality]], dtype=torch.float64)
    q_vision = torch.tensor([[quality]], dtype=torch.float64)
    c_tv = torch.tensor([[compatibility]], dtype=torch.float64)

    output = controller(q_text, q_vision, c_tv)

    assert output.text_weight.item() == pytest.approx(0.5)
    assert output.vision_weight.item() == pytest.approx(0.5)


@pytest.mark.parametrize("compatibility", [0.0, 0.2, 0.5, 1.0])
def test_higher_text_quality_produces_higher_text_weight(
    compatibility: float,
):
    controller = DeterministicReliabilityController()

    output = controller(
        torch.tensor([[0.9]], dtype=torch.float64),
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[compatibility]], dtype=torch.float64),
    )

    assert output.text_weight.item() > output.vision_weight.item()


@pytest.mark.parametrize("compatibility", [0.0, 0.2, 0.5, 1.0])
def test_higher_vision_quality_produces_higher_vision_weight(
    compatibility: float,
):
    controller = DeterministicReliabilityController()

    output = controller(
        torch.tensor([[0.3]], dtype=torch.float64),
        torch.tensor([[0.9]], dtype=torch.float64),
        torch.tensor([[compatibility]], dtype=torch.float64),
    )

    assert output.vision_weight.item() > output.text_weight.item()


def test_interaction_multiplier_is_monotonic_in_compatibility():
    controller = DeterministicReliabilityController()

    q_text = torch.ones((5, 1))
    q_vision = torch.ones((5, 1))
    compatibility = torch.tensor(
        [[0.0], [0.1], [0.4], [0.7], [1.0]]
    )

    output = controller(q_text, q_vision, compatibility)
    multipliers = output.interaction_multiplier.reshape(-1)

    assert torch.all(multipliers[1:] > multipliers[:-1])
    assert torch.equal(multipliers, compatibility.reshape(-1))


def test_effective_reliability_is_monotonic_in_compatibility_for_positive_quality():
    controller = DeterministicReliabilityController()

    q_text = torch.full((5, 1), 0.8)
    q_vision = torch.full((5, 1), 0.6)
    compatibility = torch.tensor(
        [[0.0], [0.1], [0.4], [0.7], [1.0]]
    )

    output = controller(q_text, q_vision, compatibility)

    r_text = output.effective_text_reliability.reshape(-1)
    r_vision = output.effective_vision_reliability.reshape(-1)

    assert torch.all(r_text[1:] > r_text[:-1])
    assert torch.all(r_vision[1:] > r_vision[:-1])


def test_controller_is_deterministic():
    controller = DeterministicReliabilityController()
    inputs = _scores()

    first = controller(*inputs)
    second = controller(*inputs)

    assert torch.equal(
        first.effective_text_reliability,
        second.effective_text_reliability,
    )
    assert torch.equal(
        first.effective_vision_reliability,
        second.effective_vision_reliability,
    )
    assert torch.equal(first.text_weight, second.text_weight)
    assert torch.equal(first.vision_weight, second.vision_weight)
    assert torch.equal(
        first.interaction_multiplier,
        second.interaction_multiplier,
    )


def test_controller_does_not_mutate_inputs():
    controller = DeterministicReliabilityController()
    q_text, q_vision, compatibility = _scores()

    originals = (
        q_text.clone(),
        q_vision.clone(),
        compatibility.clone(),
    )

    _ = controller(q_text, q_vision, compatibility)

    assert torch.equal(q_text, originals[0])
    assert torch.equal(q_vision, originals[1])
    assert torch.equal(compatibility, originals[2])


def test_stop_gradient_blocks_classification_side_gradients():
    controller = DeterministicReliabilityController()

    q_text = torch.tensor(
        [[0.8], [0.6]],
        requires_grad=True,
    )
    q_vision = torch.tensor(
        [[0.7], [0.9]],
        requires_grad=True,
    )
    compatibility = torch.tensor(
        [[0.9], [0.3]],
        requires_grad=True,
    )

    # Simulate differentiable modality/interaction representations that would
    # be downstream of the reliability controller during M4qcf integration.
    h_text = torch.randn(2, 4, requires_grad=True)
    h_vision = torch.randn(2, 4, requires_grad=True)
    h_interaction = torch.randn(2, 4, requires_grad=True)

    output = controller(q_text, q_vision, compatibility)

    fused = (
        output.text_weight * h_text
        + output.vision_weight * h_vision
        + output.interaction_multiplier * h_interaction
    )
    loss = fused.square().sum()
    loss.backward()

    assert q_text.grad is None
    assert q_vision.grad is None
    assert compatibility.grad is None

    assert h_text.grad is not None
    assert h_vision.grad is not None
    assert h_interaction.grad is not None


def test_explicit_auxiliary_losses_can_still_train_score_sources():
    # The controller detaches only its own control path.  Explicit q/c losses
    # computed from the original scores must remain differentiable.
    controller = DeterministicReliabilityController()

    q_text = torch.tensor([[0.8]], requires_grad=True)
    q_vision = torch.tensor([[0.7]], requires_grad=True)
    compatibility = torch.tensor([[0.6]], requires_grad=True)

    _ = controller(q_text, q_vision, compatibility)

    auxiliary_loss = (
        (q_text - 1.0).square().mean()
        + (q_vision - 1.0).square().mean()
        + (compatibility - 1.0).square().mean()
    )
    auxiliary_loss.backward()

    assert q_text.grad is not None
    assert q_vision.grad is not None
    assert compatibility.grad is not None
    assert q_text.grad.abs().sum().item() > 0.0
    assert q_vision.grad.abs().sum().item() > 0.0
    assert compatibility.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_supported_floating_dtype(dtype: torch.dtype):
    controller = DeterministicReliabilityController()
    output = controller(*_scores(dtype=dtype))

    assert output.text_weight.dtype == dtype
    assert output.vision_weight.dtype == dtype
    assert output.interaction_multiplier.dtype == dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_device_preservation():
    controller = DeterministicReliabilityController().cuda()
    inputs = _scores(device="cuda")
    output = controller(*inputs)

    assert output.text_weight.is_cuda
    assert output.vision_weight.is_cuda
    assert output.interaction_multiplier.is_cuda


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1, 1.1])
def test_invalid_epsilon_rejected(epsilon: float):
    with pytest.raises(ValueError):
        DeterministicReliabilityController(epsilon=epsilon)


@pytest.mark.parametrize("epsilon", [True, "0.1", None])
def test_non_numeric_epsilon_rejected(epsilon):
    with pytest.raises(TypeError):
        DeterministicReliabilityController(epsilon=epsilon)


@pytest.mark.parametrize(
    "bad_shape",
    [
        torch.tensor([0.5]),
        torch.tensor([[0.5, 0.5]]),
        torch.empty((0, 1)),
        torch.ones((1, 1, 1)),
    ],
)
def test_bad_score_shapes_rejected(bad_shape: torch.Tensor):
    controller = DeterministicReliabilityController()
    good = torch.tensor([[0.5]])

    with pytest.raises(ValueError):
        controller(bad_shape, good, good)


@pytest.mark.parametrize(
    "bad_score",
    [
        torch.tensor([[-0.01]]),
        torch.tensor([[1.01]]),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ],
)
def test_invalid_score_values_rejected(bad_score: torch.Tensor):
    controller = DeterministicReliabilityController()
    good = torch.tensor([[0.5]])

    with pytest.raises(ValueError):
        controller(good, bad_score, good)


def test_integer_scores_rejected():
    controller = DeterministicReliabilityController()
    score = torch.tensor([[1]], dtype=torch.int64)

    with pytest.raises(TypeError):
        controller(score, score, score)


def test_mismatched_batch_shapes_rejected():
    controller = DeterministicReliabilityController()

    with pytest.raises(ValueError, match="identical shapes"):
        controller(
            torch.ones((2, 1)),
            torch.ones((3, 1)),
            torch.ones((2, 1)),
        )


def test_mismatched_dtype_rejected():
    controller = DeterministicReliabilityController()

    with pytest.raises(TypeError, match="same dtype"):
        controller(
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float32),
        )


def test_as_dict_has_stable_integration_keys():
    controller = DeterministicReliabilityController()
    output = controller(*_scores())
    payload = output.as_dict()

    assert set(payload) == {
        "effective_text_reliability",
        "effective_vision_reliability",
        "text_weight",
        "vision_weight",
        "weights",
        "interaction_multiplier",
    }


def test_architecture_metadata_records_step13b_boundaries():
    controller = DeterministicReliabilityController()
    metadata = controller.architecture_metadata()

    assert metadata["protocol_version"] == STEP13A_PROTOCOL_VERSION
    assert metadata["epsilon"] == pytest.approx(0.10)
    assert metadata["parameter_free"] is True
    assert metadata["trainable_parameter_count"] == 0
    assert metadata["deterministic"] is True
    assert metadata["classification_gradient_to_quality_inputs"] is False
    assert (
        metadata["classification_gradient_to_compatibility_input"]
        is False
    )
    assert metadata["affects_primary_fusion"] is False
    assert metadata["integration_status"] == "standalone_step13b_only"
    assert metadata["official_test_accessed"] is False

    # Stable metadata must remain JSON serializable.
    json.dumps(metadata)


def test_low_compatibility_preserves_relative_quality_ordering():
    controller = DeterministicReliabilityController()

    output = controller(
        torch.tensor([[0.9]], dtype=torch.float64),
        torch.tensor([[0.2]], dtype=torch.float64),
        torch.tensor([[0.0]], dtype=torch.float64),
    )

    assert (
        output.effective_text_reliability.item()
        > output.effective_vision_reliability.item()
    )
    assert output.text_weight.item() > output.vision_weight.item()


def test_zero_quality_both_modalities_remains_symmetric_and_finite():
    controller = DeterministicReliabilityController()

    output = controller(
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
        torch.tensor([[0.0]]),
    )

    assert output.effective_text_reliability.item() == pytest.approx(0.0)
    assert output.effective_vision_reliability.item() == pytest.approx(0.0)
    assert output.text_weight.item() == pytest.approx(0.5)
    assert output.vision_weight.item() == pytest.approx(0.5)
    assert math.isfinite(output.text_weight.item())
    assert math.isfinite(output.vision_weight.item())
