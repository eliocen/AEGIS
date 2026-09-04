"""
AEGIS v0.27 — Cross-modal compatibility target utilities.

This module implements the frozen v0.27 compatibility semantics only.

Matched original text-image pair:
    y_compat = 1.0

Deterministically permuted/mismatched pair:
    y_compat = 0.0

Important scientific boundary
-----------------------------
Intrinsic modality degradation (Gaussian noise, attenuation, zero dropout)
does NOT automatically define a compatibility target in the first v0.27
experiment. Those conditions return an undefined compatibility mask and must
not contribute to the primary compatibility loss.

Permutation mismatch is different: each substituted modality representation
remains intrinsically intact, while the text-image relationship is explicitly
supervised as incompatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

from .corruption import deterministic_derangement_indices


CompatibilityCondition = Literal[
    "matched",
    "permutation",
    "gaussian_noise",
    "attenuation",
    "zero_dropout",
]

ModalityName = Literal["text", "vision"]


@dataclass(frozen=True)
class CompatibilityTargets:
    """
    Compatibility targets and validity mask.

    target:
        [batch, 1] float tensor. Values are meaningful only where mask=True.

    mask:
        [batch, 1] bool tensor. False means the example must not contribute
        to the primary compatibility loss.
    """

    target: Tensor
    mask: Tensor
    condition: CompatibilityCondition
    corrupted_modality: Optional[ModalityName]


@dataclass(frozen=True)
class PermutationMismatchPlan:
    """
    Reproducible batch-level permutation plan for one modality.
    """

    indices: Tensor
    modality: ModalityName
    seed: int


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return batch_size


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be >= 0.")
    return seed


def _validate_modality(
    modality: Optional[str],
    *,
    required: bool,
) -> Optional[ModalityName]:
    if modality is None:
        if required:
            raise ValueError("corrupted_modality is required.")
        return None

    if modality not in ("text", "vision"):
        raise ValueError(
            "corrupted_modality must be 'text' or 'vision'; "
            f"got {modality!r}."
        )
    return modality  # type: ignore[return-value]


def scalar_compatibility_target(
    *,
    condition: CompatibilityCondition,
    corrupted_modality: Optional[ModalityName] = None,
) -> tuple[float, bool]:
    """
    Return (target_value, target_is_defined).

    Frozen semantics:
        matched     -> (1.0, True)
        permutation -> (0.0, True)
        quality corruption without permutation -> (0.0, False)

    The placeholder value 0.0 for undefined conditions MUST be ignored using
    the returned mask. It has no scientific target meaning.
    """
    if condition == "matched":
        if corrupted_modality is not None:
            raise ValueError(
                "matched condition must not specify corrupted_modality."
            )
        return 1.0, True

    if condition == "permutation":
        _validate_modality(corrupted_modality, required=True)
        return 0.0, True

    if condition in (
        "gaussian_noise",
        "attenuation",
        "zero_dropout",
    ):
        _validate_modality(corrupted_modality, required=True)

        # Frozen v0.27 rule: low intrinsic quality must not be silently
        # relabelled as semantic incompatibility.
        return 0.0, False

    raise ValueError(
        f"Unsupported compatibility condition {condition!r}. Expected one "
        "of: matched, permutation, gaussian_noise, attenuation, zero_dropout."
    )


def make_compatibility_targets(
    batch_size: int,
    *,
    condition: CompatibilityCondition,
    corrupted_modality: Optional[ModalityName] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> CompatibilityTargets:
    """
    Construct [batch, 1] compatibility targets plus a validity mask.
    """
    batch_size = _validate_batch_size(batch_size)

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    if not dtype.is_floating_point:
        raise TypeError(
            f"compatibility-target dtype must be floating point; got {dtype}."
        )

    value, defined = scalar_compatibility_target(
        condition=condition,
        corrupted_modality=corrupted_modality,
    )

    target = torch.full(
        (batch_size, 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    mask = torch.full(
        (batch_size, 1),
        fill_value=defined,
        device=device,
        dtype=torch.bool,
    )

    return CompatibilityTargets(
        target=target,
        mask=mask,
        condition=condition,
        corrupted_modality=corrupted_modality,
    )


def make_permutation_mismatch_plan(
    batch_size: int,
    *,
    modality: ModalityName,
    seed: int,
    device: Optional[torch.device | str] = None,
) -> PermutationMismatchPlan:
    """
    Build a deterministic derangement plan for text or vision.

    This delegates index construction to the already-tested v0.27 corruption
    primitive so training and diagnostic code cannot silently use different
    permutation semantics.
    """
    modality = _validate_modality(modality, required=True)  # type: ignore[assignment]
    seed = _validate_seed(seed)

    indices = deterministic_derangement_indices(
        batch_size,
        seed=seed,
    )

    if device is not None:
        indices = indices.to(device=device)

    return PermutationMismatchPlan(
        indices=indices,
        modality=modality,  # type: ignore[arg-type]
        seed=seed,
    )


def apply_permutation_mismatch_plan(
    text: Tensor,
    vision: Tensor,
    plan: PermutationMismatchPlan,
) -> tuple[Tensor, Tensor]:
    """
    Apply one permutation plan while leaving the other modality unchanged.

    Returns cloned/index-selected tensors and never mutates caller inputs.
    """
    if not isinstance(text, Tensor) or not isinstance(vision, Tensor):
        raise TypeError("text and vision must be torch.Tensor objects.")

    if text.ndim != 2 or vision.ndim != 2:
        raise ValueError(
            "text and vision must both have shape [batch, features]."
        )

    if text.shape[0] != vision.shape[0]:
        raise ValueError(
            "text and vision batch sizes must match; "
            f"got {text.shape[0]} and {vision.shape[0]}."
        )

    if text.shape[0] <= 1:
        raise ValueError(
            "Permutation mismatch requires batch_size >= 2."
        )

    if not isinstance(plan, PermutationMismatchPlan):
        raise TypeError("plan must be a PermutationMismatchPlan.")

    indices = plan.indices
    if indices.ndim != 1 or indices.numel() != text.shape[0]:
        raise ValueError(
            "plan.indices must contain exactly one index per batch sample."
        )

    indices = indices.to(device=text.device)

    expected = torch.arange(
        text.shape[0],
        device=text.device,
        dtype=torch.long,
    )

    if indices.dtype != torch.long:
        indices = indices.to(dtype=torch.long)

    if torch.unique(indices).numel() != text.shape[0]:
        raise ValueError("plan.indices must be a permutation.")

    if int(indices.min().item()) < 0 or int(indices.max().item()) >= text.shape[0]:
        raise ValueError("plan.indices contains an out-of-range index.")

    if torch.any(indices == expected):
        raise ValueError(
            "plan.indices must be a derangement with no fixed points."
        )

    if plan.modality == "text":
        return (
            text.index_select(0, indices),
            vision.clone(),
        )

    # The same numeric indices can be moved independently to the vision
    # device if text/vision are ever stored on different devices.
    vision_indices = indices.to(device=vision.device)
    return (
        text.clone(),
        vision.index_select(0, vision_indices),
    )
