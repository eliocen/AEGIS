"""
Tests for AEGIS v0.27 compatibility targets and mismatch planning.
"""

from __future__ import annotations

import pytest
import torch

from aegis.reliability.compatibility import (
    apply_permutation_mismatch_plan,
    make_compatibility_targets,
    make_permutation_mismatch_plan,
    scalar_compatibility_target,
)


def make_pair() -> tuple[torch.Tensor, torch.Tensor]:
    text = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    vision = torch.tensor(
        [
            [11.0, 12.0, 13.0],
            [14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0],
        ]
    )
    return text, vision


def test_matched_pair_target_is_one_and_defined() -> None:
    value, defined = scalar_compatibility_target(condition="matched")
    assert value == 1.0
    assert defined is True


@pytest.mark.parametrize("modality", ["text", "vision"])
def test_permutation_target_is_zero_and_defined(modality: str) -> None:
    value, defined = scalar_compatibility_target(
        condition="permutation",
        corrupted_modality=modality,  # type: ignore[arg-type]
    )
    assert value == 0.0
    assert defined is True


@pytest.mark.parametrize(
    "condition",
    ["gaussian_noise", "attenuation", "zero_dropout"],
)
@pytest.mark.parametrize("modality", ["text", "vision"])
def test_quality_corruption_has_no_primary_compatibility_target(
    condition: str,
    modality: str,
) -> None:
    value, defined = scalar_compatibility_target(
        condition=condition,  # type: ignore[arg-type]
        corrupted_modality=modality,  # type: ignore[arg-type]
    )
    assert value == 0.0
    assert defined is False


def test_matched_batch_targets_are_ones_with_true_mask() -> None:
    result = make_compatibility_targets(
        5,
        condition="matched",
    )
    assert result.target.shape == (5, 1)
    assert result.mask.shape == (5, 1)
    assert torch.all(result.target == 1.0)
    assert torch.all(result.mask)


def test_permutation_batch_targets_are_zeros_with_true_mask() -> None:
    result = make_compatibility_targets(
        5,
        condition="permutation",
        corrupted_modality="vision",
    )
    assert torch.all(result.target == 0.0)
    assert torch.all(result.mask)


def test_quality_corruption_batch_mask_is_false() -> None:
    result = make_compatibility_targets(
        5,
        condition="attenuation",
        corrupted_modality="text",
    )
    assert torch.all(result.target == 0.0)
    assert not torch.any(result.mask)


def test_matched_rejects_modality() -> None:
    with pytest.raises(
        ValueError,
        match="must not specify corrupted_modality",
    ):
        scalar_compatibility_target(
            condition="matched",
            corrupted_modality="text",
        )


@pytest.mark.parametrize(
    "condition",
    [
        "permutation",
        "gaussian_noise",
        "attenuation",
        "zero_dropout",
    ],
)
def test_nonmatched_condition_requires_modality(condition: str) -> None:
    with pytest.raises(
        ValueError,
        match="corrupted_modality is required",
    ):
        scalar_compatibility_target(
            condition=condition,  # type: ignore[arg-type]
        )


def test_invalid_modality_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be 'text' or 'vision'",
    ):
        scalar_compatibility_target(
            condition="permutation",
            corrupted_modality="audio",  # type: ignore[arg-type]
        )


def test_mismatch_plan_is_deterministic() -> None:
    first = make_permutation_mismatch_plan(
        20,
        modality="vision",
        seed=42,
    )
    second = make_permutation_mismatch_plan(
        20,
        modality="vision",
        seed=42,
    )
    assert torch.equal(first.indices, second.indices)


def test_mismatch_plan_has_no_fixed_points() -> None:
    plan = make_permutation_mismatch_plan(
        20,
        modality="text",
        seed=42,
    )
    assert torch.all(plan.indices != torch.arange(20))
    assert torch.unique(plan.indices).numel() == 20


def test_apply_text_mismatch_changes_only_text() -> None:
    text, vision = make_pair()
    text_original = text.clone()
    vision_original = vision.clone()

    plan = make_permutation_mismatch_plan(
        text.shape[0],
        modality="text",
        seed=42,
    )
    text_out, vision_out = apply_permutation_mismatch_plan(
        text,
        vision,
        plan,
    )

    assert not torch.equal(text_out, text_original)
    assert torch.equal(vision_out, vision_original)
    assert torch.equal(text, text_original)
    assert torch.equal(vision, vision_original)


def test_apply_vision_mismatch_changes_only_vision() -> None:
    text, vision = make_pair()
    text_original = text.clone()
    vision_original = vision.clone()

    plan = make_permutation_mismatch_plan(
        text.shape[0],
        modality="vision",
        seed=42,
    )
    text_out, vision_out = apply_permutation_mismatch_plan(
        text,
        vision,
        plan,
    )

    assert torch.equal(text_out, text_original)
    assert not torch.equal(vision_out, vision_original)
    assert torch.equal(text, text_original)
    assert torch.equal(vision, vision_original)


def test_apply_mismatch_preserves_row_sets() -> None:
    text, vision = make_pair()

    text_plan = make_permutation_mismatch_plan(
        text.shape[0],
        modality="text",
        seed=77,
    )
    text_out, _ = apply_permutation_mismatch_plan(
        text,
        vision,
        text_plan,
    )

    assert {
        tuple(row.tolist()) for row in text_out
    } == {
        tuple(row.tolist()) for row in text
    }


def test_mismatch_plan_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="batch_size >= 2"):
        make_permutation_mismatch_plan(
            1,
            modality="vision",
            seed=42,
        )


def test_apply_rejects_batch_size_mismatch() -> None:
    text = torch.ones(4, 2)
    vision = torch.ones(3, 3)
    plan = make_permutation_mismatch_plan(
        4,
        modality="text",
        seed=42,
    )

    with pytest.raises(ValueError, match="batch sizes must match"):
        apply_permutation_mismatch_plan(
            text,
            vision,
            plan,
        )


def test_compatibility_target_preserves_float64_dtype() -> None:
    result = make_compatibility_targets(
        3,
        condition="matched",
        dtype=torch.float64,
    )
    assert result.target.dtype == torch.float64
    assert result.mask.dtype == torch.bool


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_compatibility_targets_and_plan_support_cuda() -> None:
    result = make_compatibility_targets(
        4,
        condition="permutation",
        corrupted_modality="vision",
        device="cuda",
    )
    plan = make_permutation_mismatch_plan(
        4,
        modality="vision",
        seed=42,
        device="cuda",
    )

    assert result.target.is_cuda
    assert result.mask.is_cuda
    assert plan.indices.is_cuda
