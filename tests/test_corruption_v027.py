"""
Unit tests for AEGIS v0.27 deterministic corruption utilities.

These tests intentionally cover corruption mechanics only. Quality and
compatibility target generation belongs to the next protocol step.
"""

from __future__ import annotations

import torch
import pytest

from aegis.reliability.corruption import (
    attenuation,
    compute_feature_std,
    corrupt_representation,
    deterministic_derangement_indices,
    gaussian_noise,
    permutation_mismatch,
    validate_feature_std,
    zero_dropout,
)


def make_representation() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )


def test_compute_feature_std_population_matches_torch() -> None:
    x = make_representation()
    expected = torch.std(x, dim=0, unbiased=False)
    actual = compute_feature_std(x)

    assert actual.shape == (x.shape[1],)
    assert torch.allclose(actual, expected)


def test_validate_feature_std_accepts_vector_and_row_vector() -> None:
    x = make_representation()
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    row = vector.unsqueeze(0)

    vector_result = validate_feature_std(vector, x)
    row_result = validate_feature_std(row, x)

    assert vector_result.shape == (1, 4)
    assert row_result.shape == (1, 4)
    assert torch.equal(vector_result, row_result)


def test_gaussian_noise_is_deterministic_for_same_seed() -> None:
    x = make_representation()
    std = compute_feature_std(x)

    first = gaussian_noise(x, std, severity=0.5, seed=12345)
    second = gaussian_noise(x, std, severity=0.5, seed=12345)

    assert torch.equal(first.corrupted, second.corrupted)
    assert first.family == "gaussian_noise"
    assert first.severity == 0.5
    assert first.seed == 12345


def test_gaussian_noise_changes_with_different_seed() -> None:
    x = make_representation()
    std = compute_feature_std(x)

    first = gaussian_noise(x, std, severity=0.5, seed=1)
    second = gaussian_noise(x, std, severity=0.5, seed=2)

    assert not torch.equal(first.corrupted, second.corrupted)


def test_gaussian_noise_severity_zero_returns_equal_clone() -> None:
    x = make_representation()
    std = compute_feature_std(x)

    result = gaussian_noise(x, std, severity=0.0, seed=7)

    assert torch.equal(result.corrupted, x)
    assert result.corrupted.data_ptr() != x.data_ptr()


def test_gaussian_noise_does_not_modify_global_rng_state() -> None:
    x = make_representation()
    std = compute_feature_std(x)

    torch.manual_seed(999)
    state_before = torch.random.get_rng_state().clone()

    _ = gaussian_noise(x, std, severity=0.5, seed=42)

    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


def test_gaussian_noise_preserves_dtype_and_device() -> None:
    x = make_representation().to(dtype=torch.float64)
    std = compute_feature_std(x)

    result = gaussian_noise(x, std, severity=0.25, seed=42)

    assert result.corrupted.dtype == x.dtype
    assert result.corrupted.device == x.device


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_gaussian_noise_cpu_and_cuda_use_same_canonical_noise() -> None:
    x_cpu = make_representation()
    std_cpu = compute_feature_std(x_cpu)

    cpu_result = gaussian_noise(
        x_cpu,
        std_cpu,
        severity=0.5,
        seed=31415,
    )

    x_cuda = x_cpu.cuda()
    std_cuda = std_cpu.cuda()
    cuda_result = gaussian_noise(
        x_cuda,
        std_cuda,
        severity=0.5,
        seed=31415,
    )

    assert torch.allclose(
        cpu_result.corrupted,
        cuda_result.corrupted.cpu(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_attenuation_matches_protocol_equation() -> None:
    x = make_representation()
    result = attenuation(x, severity=0.25)

    assert torch.allclose(result.corrupted, 0.75 * x)
    assert result.family == "attenuation"
    assert result.severity == 0.25


def test_attenuation_severity_one_is_zero() -> None:
    x = make_representation()
    result = attenuation(x, severity=1.0)

    assert torch.equal(result.corrupted, torch.zeros_like(x))


def test_zero_dropout_returns_zeros_without_mutating_input() -> None:
    x = make_representation()
    original = x.clone()

    result = zero_dropout(x)

    assert torch.equal(result.corrupted, torch.zeros_like(x))
    assert torch.equal(x, original)
    assert result.family == "zero_dropout"
    assert result.severity == 1.0


def test_derangement_has_no_fixed_points() -> None:
    indices = deterministic_derangement_indices(100, seed=42)
    original = torch.arange(100)

    assert indices.shape == (100,)
    assert torch.unique(indices).numel() == 100
    assert torch.all(indices != original)


def test_derangement_is_deterministic_for_same_seed() -> None:
    first = deterministic_derangement_indices(12, seed=77)
    second = deterministic_derangement_indices(12, seed=77)

    assert torch.equal(first, second)


def test_derangement_rejects_batch_size_one() -> None:
    with pytest.raises(ValueError, match="batch_size >= 2"):
        deterministic_derangement_indices(1, seed=42)


def test_permutation_mismatch_uses_derangement_and_preserves_rows() -> None:
    x = make_representation()
    result = permutation_mismatch(x, seed=42)

    assert result.permutation_indices is not None
    assert torch.all(
        result.permutation_indices != torch.arange(x.shape[0])
    )

    expected = x.index_select(0, result.permutation_indices)
    assert torch.equal(result.corrupted, expected)

    assert {
        tuple(row.tolist())
        for row in result.corrupted
    } == {
        tuple(row.tolist())
        for row in x
    }


def test_permutation_mismatch_does_not_modify_global_rng_state() -> None:
    x = make_representation()

    torch.manual_seed(2026)
    state_before = torch.random.get_rng_state().clone()

    _ = permutation_mismatch(x, seed=42)

    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


def test_dispatcher_routes_all_families() -> None:
    x = make_representation()
    std = compute_feature_std(x)

    gaussian = corrupt_representation(
        x,
        family="gaussian_noise",
        severity=0.25,
        seed=42,
        feature_std=std,
    )
    attenuated = corrupt_representation(
        x,
        family="attenuation",
        severity=0.25,
    )
    zeroed = corrupt_representation(
        x,
        family="zero_dropout",
    )
    permuted = corrupt_representation(
        x,
        family="permutation",
        seed=42,
    )

    assert gaussian.family == "gaussian_noise"
    assert attenuated.family == "attenuation"
    assert zeroed.family == "zero_dropout"
    assert permuted.family == "permutation"


@pytest.mark.parametrize("severity", [-0.01, 1.01])
def test_continuous_corruptions_reject_invalid_severity(
    severity: float,
) -> None:
    x = make_representation()
    std = compute_feature_std(x)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gaussian_noise(x, std, severity=severity, seed=42)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        attenuation(x, severity=severity)


def test_gaussian_noise_rejects_wrong_feature_dimension() -> None:
    x = make_representation()
    bad_std = torch.ones(3)

    with pytest.raises(ValueError, match="feature dimension"):
        gaussian_noise(
            x,
            bad_std,
            severity=0.5,
            seed=42,
        )


def test_feature_std_rejects_negative_values() -> None:
    x = make_representation()
    bad_std = torch.tensor([1.0, -1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        validate_feature_std(bad_std, x)


def test_corruptions_do_not_modify_input_tensor() -> None:
    x = make_representation()
    original = x.clone()
    std = compute_feature_std(x)

    _ = gaussian_noise(x, std, severity=0.5, seed=42)
    _ = attenuation(x, severity=0.5)
    _ = zero_dropout(x)
    _ = permutation_mismatch(x, seed=42)

    assert torch.equal(x, original)
