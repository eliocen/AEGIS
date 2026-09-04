"""
Unit tests for AEGIS v0.27 explicit modality-quality targets.

These tests freeze the semantic distinction between:
    - intrinsic modality degradation; and
    - cross-modal permutation mismatch.

Compatibility targets are intentionally excluded from this test file.
"""

from __future__ import annotations

import pytest
import torch

from aegis.reliability.targets import (
    continuous_quality_target,
    make_quality_targets,
    scalar_quality_targets,
)


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (0.0, 1.0),
        (0.25, 0.75),
        (0.50, 0.50),
        (0.75, 0.25),
        (1.0, 0.0),
    ],
)
def test_continuous_quality_mapping_is_one_minus_severity(
    severity: float,
    expected: float,
) -> None:
    assert continuous_quality_target(severity) == expected


@pytest.mark.parametrize("severity", [-0.01, 1.01])
def test_continuous_quality_rejects_out_of_range_severity(
    severity: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        continuous_quality_target(severity)


def test_clean_targets_are_one_for_both_modalities() -> None:
    text, vision = scalar_quality_targets(condition="clean")

    assert text == 1.0
    assert vision == 1.0


@pytest.mark.parametrize(
    ("condition", "severity", "expected_text"),
    [
        ("gaussian_noise", 0.25, 0.75),
        ("gaussian_noise", 0.50, 0.50),
        ("attenuation", 0.75, 0.25),
        ("attenuation", 1.00, 0.00),
    ],
)
def test_text_continuous_corruption_only_reduces_text_quality(
    condition: str,
    severity: float,
    expected_text: float,
) -> None:
    text, vision = scalar_quality_targets(
        condition=condition,  # type: ignore[arg-type]
        corrupted_modality="text",
        severity=severity,
    )

    assert text == expected_text
    assert vision == 1.0


@pytest.mark.parametrize(
    ("condition", "severity", "expected_vision"),
    [
        ("gaussian_noise", 0.25, 0.75),
        ("gaussian_noise", 0.50, 0.50),
        ("attenuation", 0.75, 0.25),
        ("attenuation", 1.00, 0.00),
    ],
)
def test_vision_continuous_corruption_only_reduces_vision_quality(
    condition: str,
    severity: float,
    expected_vision: float,
) -> None:
    text, vision = scalar_quality_targets(
        condition=condition,  # type: ignore[arg-type]
        corrupted_modality="vision",
        severity=severity,
    )

    assert text == 1.0
    assert vision == expected_vision


def test_zero_dropout_text_target_is_zero_and_vision_remains_one() -> None:
    text, vision = scalar_quality_targets(
        condition="zero_dropout",
        corrupted_modality="text",
    )

    assert text == 0.0
    assert vision == 1.0


def test_zero_dropout_vision_target_is_zero_and_text_remains_one() -> None:
    text, vision = scalar_quality_targets(
        condition="zero_dropout",
        corrupted_modality="vision",
    )

    assert text == 1.0
    assert vision == 0.0


@pytest.mark.parametrize("modality", ["text", "vision"])
def test_permutation_preserves_intrinsic_quality_targets(
    modality: str,
) -> None:
    text, vision = scalar_quality_targets(
        condition="permutation",
        corrupted_modality=modality,  # type: ignore[arg-type]
    )

    assert text == 1.0
    assert vision == 1.0


def test_batch_quality_targets_have_expected_shape_and_values() -> None:
    result = make_quality_targets(
        8,
        condition="gaussian_noise",
        corrupted_modality="text",
        severity=0.50,
    )

    assert result.text.shape == (8, 1)
    assert result.vision.shape == (8, 1)
    assert torch.all(result.text == 0.50)
    assert torch.all(result.vision == 1.0)
    assert result.condition == "gaussian_noise"
    assert result.corrupted_modality == "text"
    assert result.severity == 0.50


def test_batch_quality_targets_preserve_requested_dtype() -> None:
    result = make_quality_targets(
        3,
        condition="attenuation",
        corrupted_modality="vision",
        severity=0.25,
        dtype=torch.float64,
    )

    assert result.text.dtype == torch.float64
    assert result.vision.dtype == torch.float64


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_batch_quality_targets_support_cuda_device() -> None:
    result = make_quality_targets(
        4,
        condition="zero_dropout",
        corrupted_modality="vision",
        device="cuda",
    )

    assert result.text.is_cuda
    assert result.vision.is_cuda


def test_clean_rejects_corrupted_modality() -> None:
    with pytest.raises(
        ValueError,
        match="must not specify corrupted_modality",
    ):
        scalar_quality_targets(
            condition="clean",
            corrupted_modality="text",
        )


def test_clean_rejects_severity() -> None:
    with pytest.raises(
        ValueError,
        match="must not specify severity",
    ):
        scalar_quality_targets(
            condition="clean",
            severity=0.0,
        )


@pytest.mark.parametrize(
    "condition",
    ["gaussian_noise", "attenuation"],
)
def test_continuous_corruption_requires_modality(
    condition: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="corrupted_modality is required",
    ):
        scalar_quality_targets(
            condition=condition,  # type: ignore[arg-type]
            severity=0.5,
        )


@pytest.mark.parametrize(
    "condition",
    ["gaussian_noise", "attenuation"],
)
def test_continuous_corruption_requires_severity(
    condition: str,
) -> None:
    with pytest.raises(ValueError, match="requires severity"):
        scalar_quality_targets(
            condition=condition,  # type: ignore[arg-type]
            corrupted_modality="text",
        )


def test_zero_dropout_rejects_explicit_severity() -> None:
    with pytest.raises(
        ValueError,
        match="discrete condition",
    ):
        scalar_quality_targets(
            condition="zero_dropout",
            corrupted_modality="text",
            severity=1.0,
        )


def test_permutation_requires_modality_for_condition_metadata() -> None:
    with pytest.raises(
        ValueError,
        match="corrupted_modality is required",
    ):
        scalar_quality_targets(
            condition="permutation",
        )


def test_permutation_rejects_severity() -> None:
    with pytest.raises(
        ValueError,
        match="must not specify severity",
    ):
        scalar_quality_targets(
            condition="permutation",
            corrupted_modality="vision",
            severity=1.0,
        )


def test_invalid_modality_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be 'text' or 'vision'",
    ):
        scalar_quality_targets(
            condition="zero_dropout",
            corrupted_modality="audio",  # type: ignore[arg-type]
        )


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        make_quality_targets(
            0,
            condition="clean",
        )


def test_nonfloating_target_dtype_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="must be floating point",
    ):
        make_quality_targets(
            2,
            condition="clean",
            dtype=torch.int64,
        )
