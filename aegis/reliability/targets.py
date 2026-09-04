"""
AEGIS v0.27 — Explicit modality-quality target generation.

This module implements ONLY intrinsic modality-quality targets from the frozen
v0.27 protocol. It deliberately does not generate compatibility targets and
does not modify model architecture or fusion.

Frozen quality semantics
------------------------
Clean modality:
    q = 1.0

Gaussian noise / attenuation at severity lambda:
    q_corrupted = 1 - lambda
    q_intact = 1.0

Zero dropout:
    q_corrupted = 0.0
    q_intact = 1.0

Permutation mismatch:
    q_text = 1.0
    q_vision = 1.0

Permutation is treated as cross-modal incompatibility, not intrinsic modality
degradation. Compatibility supervision is implemented separately in the next
v0.27 protocol step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor


QualityCondition = Literal[
    "clean",
    "gaussian_noise",
    "attenuation",
    "zero_dropout",
    "permutation",
]

ModalityName = Literal["text", "vision"]


@dataclass(frozen=True)
class QualityTargets:
    """
    Batch quality targets for text and vision.

    Both tensors have shape [batch, 1].
    """

    text: Tensor
    vision: Tensor
    condition: QualityCondition
    corrupted_modality: Optional[ModalityName]
    severity: Optional[float]


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return batch_size


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


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise TypeError("severity must be a real number in [0, 1].")
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"severity must be in [0, 1]; got {value}.")
    return value


def continuous_quality_target(severity: float) -> float:
    """
    Frozen v0.27 synthetic quality mapping:

        y_quality = 1 - severity
    """
    severity = _validate_severity(severity)
    return 1.0 - severity


def scalar_quality_targets(
    *,
    condition: QualityCondition,
    corrupted_modality: Optional[ModalityName] = None,
    severity: Optional[float] = None,
) -> tuple[float, float]:
    """
    Return scalar (text_quality, vision_quality) targets.

    This function encodes the scientific semantics of the frozen protocol.
    """
    if condition == "clean":
        if corrupted_modality is not None:
            raise ValueError(
                "clean condition must not specify corrupted_modality."
            )
        if severity is not None:
            raise ValueError("clean condition must not specify severity.")
        return 1.0, 1.0

    if condition in ("gaussian_noise", "attenuation"):
        modality = _validate_modality(
            corrupted_modality,
            required=True,
        )
        if severity is None:
            raise ValueError(
                f"{condition} requires severity."
            )

        quality = continuous_quality_target(severity)

        if modality == "text":
            return quality, 1.0
        return 1.0, quality

    if condition == "zero_dropout":
        modality = _validate_modality(
            corrupted_modality,
            required=True,
        )
        if severity is not None:
            raise ValueError(
                "zero_dropout is a discrete condition and must not specify "
                "severity."
            )

        if modality == "text":
            return 0.0, 1.0
        return 1.0, 0.0

    if condition == "permutation":
        _validate_modality(
            corrupted_modality,
            required=True,
        )
        if severity is not None:
            raise ValueError(
                "permutation is a discrete compatibility corruption and "
                "must not specify severity."
            )

        # Critical v0.27 semantic distinction:
        # the substituted representation remains intrinsically intact.
        return 1.0, 1.0

    raise ValueError(
        f"Unsupported quality condition {condition!r}. Expected one of: "
        "clean, gaussian_noise, attenuation, zero_dropout, permutation."
    )


def make_quality_targets(
    batch_size: int,
    *,
    condition: QualityCondition,
    corrupted_modality: Optional[ModalityName] = None,
    severity: Optional[float] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
) -> QualityTargets:
    """
    Construct [batch, 1] text and vision quality-target tensors.

    The function is suitable for homogeneous corruption conditions and
    diagnostic batches. Mixed-condition training batches can be assembled by
    concatenating or indexing outputs from this function, while preserving the
    same frozen scalar target rules.
    """
    batch_size = _validate_batch_size(batch_size)

    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    if not dtype.is_floating_point:
        raise TypeError(
            f"quality-target dtype must be floating point; got {dtype}."
        )

    text_value, vision_value = scalar_quality_targets(
        condition=condition,
        corrupted_modality=corrupted_modality,
        severity=severity,
    )

    text = torch.full(
        (batch_size, 1),
        fill_value=text_value,
        device=device,
        dtype=dtype,
    )
    vision = torch.full(
        (batch_size, 1),
        fill_value=vision_value,
        device=device,
        dtype=dtype,
    )

    return QualityTargets(
        text=text,
        vision=vision,
        condition=condition,
        corrupted_modality=corrupted_modality,
        severity=(
            None
            if severity is None
            else float(severity)
        ),
    )
