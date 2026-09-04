"""
AEGIS v0.27 — Deterministic representation-level corruption utilities.

This module implements ONLY corruption generation. It does not define quality
targets, compatibility targets, model architecture, training losses, or fusion.

Scientific scope
----------------
Corruption is applied to frozen encoder representations before learned AEGIS
projection heads.

Implemented corruption families:
    - Gaussian representation noise
    - Attenuation
    - Zero dropout
    - Deterministic permutation mismatch

Design requirements:
    - deterministic under an explicit seed;
    - no mutation of PyTorch's global RNG state;
    - no in-place modification of caller tensors;
    - device/dtype preserving outputs;
    - validation of severity, shapes, and feature scales;
    - deterministic permutation is a derangement for batch size > 1.

Important:
    Permutation is a compatibility corruption, not an intrinsic-quality
    corruption. Quality/compatibility target generation is intentionally
    implemented separately in a later v0.27 step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor


CorruptionFamily = Literal[
    "gaussian_noise",
    "attenuation",
    "zero_dropout",
    "permutation",
]


@dataclass(frozen=True)
class CorruptionResult:
    corrupted: Tensor
    family: CorruptionFamily
    severity: Optional[float]
    seed: Optional[int]
    permutation_indices: Optional[Tensor] = None


def _validate_representation(x: Tensor, *, name: str = "representation") -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, features]; got {tuple(x.shape)}."
        )
    if x.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if x.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature.")
    if not torch.is_floating_point(x):
        raise TypeError(
            f"{name} must have a floating-point dtype; got {x.dtype}."
        )
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be >= 0.")
    return seed


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise TypeError("severity must be a real number in [0, 1].")
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"severity must be in [0, 1]; got {value}.")
    return value


def validate_feature_std(feature_std: Tensor, representation: Tensor) -> Tensor:
    _validate_representation(representation)

    if not isinstance(feature_std, Tensor):
        raise TypeError("feature_std must be a torch.Tensor.")

    if feature_std.ndim == 1:
        feature_std = feature_std.unsqueeze(0)

    if feature_std.ndim != 2 or feature_std.shape[0] != 1:
        raise ValueError(
            "feature_std must have shape [features] or [1, features]; "
            f"got {tuple(feature_std.shape)}."
        )

    if feature_std.shape[1] != representation.shape[1]:
        raise ValueError(
            "feature_std feature dimension does not match representation: "
            f"{feature_std.shape[1]} != {representation.shape[1]}."
        )

    if not torch.is_floating_point(feature_std):
        raise TypeError(
            f"feature_std must have a floating-point dtype; got {feature_std.dtype}."
        )

    if not torch.isfinite(feature_std).all():
        raise ValueError("feature_std contains NaN or infinite values.")

    if (feature_std < 0).any():
        raise ValueError("feature_std must be non-negative.")

    return feature_std.to(
        device=representation.device,
        dtype=representation.dtype,
    )


def compute_feature_std(
    representations: Tensor,
    *,
    unbiased: bool = False,
    minimum_std: float = 0.0,
) -> Tensor:
    _validate_representation(representations, name="representations")

    if isinstance(minimum_std, bool) or not isinstance(
        minimum_std, (int, float)
    ):
        raise TypeError("minimum_std must be a non-negative real number.")

    minimum_std = float(minimum_std)
    if minimum_std < 0.0:
        raise ValueError("minimum_std must be >= 0.")

    std = torch.std(
        representations,
        dim=0,
        unbiased=unbiased,
    )

    if minimum_std > 0.0:
        std = torch.clamp(std, min=minimum_std)

    return std


def _cpu_standard_normal(
    shape: Tuple[int, ...],
    *,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_validate_seed(seed))
    return torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


def gaussian_noise(
    representation: Tensor,
    feature_std: Tensor,
    *,
    severity: float,
    seed: int,
) -> CorruptionResult:
    _validate_representation(representation)
    severity = _validate_severity(severity)
    seed = _validate_seed(seed)
    feature_std = validate_feature_std(feature_std, representation)

    if severity == 0.0:
        return CorruptionResult(
            corrupted=representation.clone(),
            family="gaussian_noise",
            severity=severity,
            seed=seed,
        )

    noise = _cpu_standard_normal(
        tuple(representation.shape),
        seed=seed,
    ).to(
        device=representation.device,
        dtype=representation.dtype,
    )

    corrupted = representation + severity * feature_std * noise

    return CorruptionResult(
        corrupted=corrupted,
        family="gaussian_noise",
        severity=severity,
        seed=seed,
    )


def attenuation(
    representation: Tensor,
    *,
    severity: float,
) -> CorruptionResult:
    _validate_representation(representation)
    severity = _validate_severity(severity)

    corrupted = representation * (1.0 - severity)

    return CorruptionResult(
        corrupted=corrupted,
        family="attenuation",
        severity=severity,
        seed=None,
    )


def zero_dropout(representation: Tensor) -> CorruptionResult:
    _validate_representation(representation)

    return CorruptionResult(
        corrupted=torch.zeros_like(representation),
        family="zero_dropout",
        severity=1.0,
        seed=None,
    )


def deterministic_derangement_indices(
    batch_size: int,
    *,
    seed: int,
) -> Tensor:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 1:
        raise ValueError(
            "Permutation mismatch requires batch_size >= 2 because a "
            "derangement does not exist for a single sample."
        )

    seed = _validate_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    shift = int(
        torch.randint(
            low=1,
            high=batch_size,
            size=(1,),
            generator=generator,
            device="cpu",
        ).item()
    )

    base = torch.arange(batch_size, dtype=torch.long)
    return (base + shift) % batch_size


def permutation_mismatch(
    representation: Tensor,
    *,
    seed: int,
) -> CorruptionResult:
    _validate_representation(representation)

    seed = _validate_seed(seed)
    indices_cpu = deterministic_derangement_indices(
        representation.shape[0],
        seed=seed,
    )
    indices = indices_cpu.to(device=representation.device)
    corrupted = representation.index_select(0, indices)

    return CorruptionResult(
        corrupted=corrupted,
        family="permutation",
        severity=None,
        seed=seed,
        permutation_indices=indices,
    )


def corrupt_representation(
    representation: Tensor,
    *,
    family: CorruptionFamily,
    severity: Optional[float] = None,
    seed: Optional[int] = None,
    feature_std: Optional[Tensor] = None,
) -> CorruptionResult:
    if family == "gaussian_noise":
        if severity is None:
            raise ValueError("gaussian_noise requires severity.")
        if seed is None:
            raise ValueError("gaussian_noise requires seed.")
        if feature_std is None:
            raise ValueError("gaussian_noise requires feature_std.")
        return gaussian_noise(
            representation,
            feature_std,
            severity=severity,
            seed=seed,
        )

    if family == "attenuation":
        if severity is None:
            raise ValueError("attenuation requires severity.")
        return attenuation(
            representation,
            severity=severity,
        )

    if family == "zero_dropout":
        return zero_dropout(representation)

    if family == "permutation":
        if seed is None:
            raise ValueError("permutation requires seed.")
        return permutation_mismatch(
            representation,
            seed=seed,
        )

    raise ValueError(
        "Unsupported corruption family "
        f"{family!r}. Expected one of: gaussian_noise, attenuation, "
        "zero_dropout, permutation."
    )
