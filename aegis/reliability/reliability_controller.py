"""
AEGIS v0.27 Step13B — deterministic M4qcf reliability controller.

Protocol
--------
Step13A freezes the first M4qcf reliability-informed fusion controller as

    b(c) = epsilon + (1 - epsilon) * c

    r_T = q_T * b(c)
    r_V = q_V * b(c)

    alpha_T = (r_T + epsilon) / (r_T + r_V + 2 * epsilon)
    alpha_V = (r_V + epsilon) / (r_T + r_V + 2 * epsilon)

    g_I = c

with epsilon = 0.10.

The controller is deterministic and parameter-free.  Crucially, q_T, q_V and
c_TV are detached before they enter the reliability-control path.  Therefore
classification-side gradients cannot manipulate the quality/compatibility
estimators through this controller.  The quality and compatibility heads remain
trainable through their explicit auxiliary losses outside this module.

Step13B implements only the standalone controller.  It does not perform M4qcf
model integration, training, robustness evaluation, checkpoint selection, or
formal H7-F/H8-F/H9-F/H10-F analysis.

Scientific boundary
-------------------
The outputs are controlled learned reliability/fusion quantities.  They do not
establish factual truth, factual verification, source credibility, intent,
human trust, or external-world evidence support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


STEP13A_PROTOCOL_VERSION = "0.27.0-step13a"
STEP13B_IMPLEMENTATION_VERSION = "0.27.0-step13b"
DEFAULT_EPSILON = 0.10
V028_STEP1_PROTOCOL_VERSION = "0.28.0-step1"
V028_STEP2_IMPLEMENTATION_VERSION = "0.28.0-step2"
DEFAULT_ALLOCATION_EXPONENT = 2.0
DEFAULT_INTERACTION_EXPONENT = 1.0
SELECTIVE_CONTROLLER_MODES = {
    "weights_only",
    "interaction_only",
    "combined",
}


@dataclass(frozen=True)
class ReliabilityControllerConfig:
    """
    Immutable configuration for the deterministic reliability controller.

    epsilon:
        Frozen Step13A stabilizer/floor.  The formal M4qcf experiment uses 0.10.
    """

    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True)
class ReliabilityControllerOutput:
    """
    Structured M4qcf reliability-controller outputs.

    effective_text_reliability:
        r_T with shape [batch, 1].
    effective_vision_reliability:
        r_V with shape [batch, 1].
    text_weight:
        alpha_T with shape [batch, 1].
    vision_weight:
        alpha_V with shape [batch, 1].
    weights:
        Concatenated [alpha_T, alpha_V] with shape [batch, 2].
    interaction_multiplier:
        g_I = c_TV with shape [batch, 1].

    All returned tensors belong to the detached reliability-control path.
    """

    effective_text_reliability: Tensor
    effective_vision_reliability: Tensor
    text_weight: Tensor
    vision_weight: Tensor
    weights: Tensor
    interaction_multiplier: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        """Return the output using stable integration-facing field names."""
        return {
            "effective_text_reliability": self.effective_text_reliability,
            "effective_vision_reliability": self.effective_vision_reliability,
            "text_weight": self.text_weight,
            "vision_weight": self.vision_weight,
            "weights": self.weights,
            "interaction_multiplier": self.interaction_multiplier,
        }


def _validate_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a real number in (0, 1).")

    value = float(epsilon)
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"epsilon must satisfy 0 < epsilon < 1; got {value}."
        )
    return value


class DeterministicReliabilityController(nn.Module):
    """
    Parameter-free Step13B M4qcf reliability controller.

    The controller accepts explicitly supervised intrinsic modality-quality
    scores q_T/q_V and pairwise compatibility c_TV, all shaped [batch, 1].

    Before computing any reliability-informed fusion quantity, each input is
    detached.  This implements the frozen Step13A stop-gradient requirement:

        q~_T = stopgrad(q_T)
        q~_V = stopgrad(q_V)
        c~   = stopgrad(c_TV)

    No learnable parameter, buffer, random operation, threshold, temperature,
    or auxiliary MLP is introduced.
    """

    def __init__(
        self,
        *,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        super().__init__()

        epsilon = _validate_epsilon(epsilon)
        self.config = ReliabilityControllerConfig(
            epsilon=epsilon,
        )

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    @property
    def parameter_count(self) -> int:
        """The frozen Step13B controller must have zero trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_score(
        score: Tensor,
        *,
        name: str,
    ) -> None:
        if not isinstance(score, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if score.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch, 1]; "
                f"got {tuple(score.shape)}."
            )

        if score.shape[0] < 1:
            raise ValueError(
                f"{name} must contain at least one sample."
            )

        if score.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape [batch, 1]; "
                f"got {tuple(score.shape)}."
            )

        if not torch.is_floating_point(score):
            raise TypeError(
                f"{name} must have a floating-point dtype; "
                f"got {score.dtype}."
            )

        if not torch.isfinite(score).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        if torch.any(score < 0.0) or torch.any(score > 1.0):
            minimum = float(score.detach().min().cpu())
            maximum = float(score.detach().max().cpu())
            raise ValueError(
                f"{name} must be bounded to [0, 1]; "
                f"observed min={minimum}, max={maximum}."
            )

    @classmethod
    def _validate_triplet(
        cls,
        text_quality: Tensor,
        vision_quality: Tensor,
        compatibility: Tensor,
    ) -> None:
        cls._validate_score(
            text_quality,
            name="text_quality",
        )
        cls._validate_score(
            vision_quality,
            name="vision_quality",
        )
        cls._validate_score(
            compatibility,
            name="compatibility",
        )

        reference_shape = text_quality.shape
        reference_dtype = text_quality.dtype
        reference_device = text_quality.device

        for name, score in (
            ("vision_quality", vision_quality),
            ("compatibility", compatibility),
        ):
            if score.shape != reference_shape:
                raise ValueError(
                    "text_quality, vision_quality, and compatibility "
                    "must have identical shapes; "
                    f"text_quality={tuple(reference_shape)}, "
                    f"{name}={tuple(score.shape)}."
                )

            if score.dtype != reference_dtype:
                raise TypeError(
                    "text_quality, vision_quality, and compatibility "
                    "must have the same dtype; "
                    f"text_quality={reference_dtype}, "
                    f"{name}={score.dtype}."
                )

            if score.device != reference_device:
                raise ValueError(
                    "text_quality, vision_quality, and compatibility "
                    "must be on the same device; "
                    f"text_quality={reference_device}, "
                    f"{name}={score.device}."
                )

    def forward(
        self,
        text_quality: Tensor,
        vision_quality: Tensor,
        compatibility: Tensor,
    ) -> ReliabilityControllerOutput:
        """
        Compute the frozen Step13A deterministic reliability-control terms.

        Classification gradients are intentionally blocked at the controller
        boundary by detaching all three reliability inputs.
        """
        self._validate_triplet(
            text_quality,
            vision_quality,
            compatibility,
        )

        q_text = text_quality.detach()
        q_vision = vision_quality.detach()
        c_tv = compatibility.detach()

        epsilon = self.epsilon

        compatibility_factor = (
            epsilon
            + (1.0 - epsilon) * c_tv
        )

        effective_text_reliability = (
            q_text * compatibility_factor
        )
        effective_vision_reliability = (
            q_vision * compatibility_factor
        )

        denominator = (
            effective_text_reliability
            + effective_vision_reliability
            + 2.0 * epsilon
        )

        text_weight = (
            effective_text_reliability + epsilon
        ) / denominator

        vision_weight = (
            effective_vision_reliability + epsilon
        ) / denominator

        weights = torch.cat(
            (text_weight, vision_weight),
            dim=1,
        )

        interaction_multiplier = c_tv

        self._validate_outputs(
            effective_text_reliability=effective_text_reliability,
            effective_vision_reliability=effective_vision_reliability,
            text_weight=text_weight,
            vision_weight=vision_weight,
            weights=weights,
            interaction_multiplier=interaction_multiplier,
        )

        return ReliabilityControllerOutput(
            effective_text_reliability=effective_text_reliability,
            effective_vision_reliability=effective_vision_reliability,
            text_weight=text_weight,
            vision_weight=vision_weight,
            weights=weights,
            interaction_multiplier=interaction_multiplier,
        )

    @staticmethod
    def _validate_outputs(
        *,
        effective_text_reliability: Tensor,
        effective_vision_reliability: Tensor,
        text_weight: Tensor,
        vision_weight: Tensor,
        weights: Tensor,
        interaction_multiplier: Tensor,
    ) -> None:
        tensors = {
            "effective_text_reliability": effective_text_reliability,
            "effective_vision_reliability": effective_vision_reliability,
            "text_weight": text_weight,
            "vision_weight": vision_weight,
            "weights": weights,
            "interaction_multiplier": interaction_multiplier,
        }

        for name, value in tensors.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Internal Step13B invariant failed: {name} "
                    "contains NaN or infinite values."
                )

        if torch.any(text_weight <= 0.0) or torch.any(text_weight >= 1.0):
            raise RuntimeError(
                "Internal Step13B invariant failed: "
                "text_weight must satisfy 0 < alpha_T < 1."
            )

        if torch.any(vision_weight <= 0.0) or torch.any(vision_weight >= 1.0):
            raise RuntimeError(
                "Internal Step13B invariant failed: "
                "vision_weight must satisfy 0 < alpha_V < 1."
            )

        weight_sum = text_weight + vision_weight
        ones = torch.ones_like(weight_sum)

        if not torch.allclose(
            weight_sum,
            ones,
            rtol=1e-6,
            atol=1e-7,
        ):
            maximum_error = float(
                (weight_sum - ones)
                .abs()
                .max()
                .detach()
                .cpu()
            )
            raise RuntimeError(
                "Internal Step13B invariant failed: "
                "alpha_T + alpha_V must equal 1; "
                f"max error={maximum_error}."
            )

        if weights.ndim != 2 or weights.shape[1] != 2:
            raise RuntimeError(
                "Internal Step13B invariant failed: "
                "weights must have shape [batch, 2]."
            )

        if torch.any(interaction_multiplier < 0.0) or torch.any(
            interaction_multiplier > 1.0
        ):
            raise RuntimeError(
                "Internal Step13B invariant failed: "
                "interaction_multiplier must lie in [0, 1]."
            )

    def architecture_metadata(self) -> dict[str, object]:
        """
        Return stable JSON-serializable Step13B metadata.
        """
        return {
            "module": self.__class__.__name__,
            "protocol_version": STEP13A_PROTOCOL_VERSION,
            "implementation_version": STEP13B_IMPLEMENTATION_VERSION,
            "semantic_role": "deterministic_reliability_controller",
            "epsilon": self.epsilon,
            "parameter_free": True,
            "trainable_parameter_count": self.parameter_count,
            "deterministic": True,
            "classification_gradient_to_quality_inputs": False,
            "classification_gradient_to_compatibility_input": False,
            "input_semantics": [
                "intrinsic_text_quality",
                "intrinsic_vision_quality",
                "cross_modal_compatibility",
            ],
            "output_semantics": [
                "effective_text_reliability",
                "effective_vision_reliability",
                "text_weight",
                "vision_weight",
                "interaction_multiplier",
            ],
            "text_reliability_equation": (
                "q_T * (epsilon + (1 - epsilon) * c_TV)"
            ),
            "vision_reliability_equation": (
                "q_V * (epsilon + (1 - epsilon) * c_TV)"
            ),
            "weight_normalization": (
                "(r_m + epsilon) / "
                "(r_T + r_V + 2 * epsilon)"
            ),
            "interaction_multiplier_equation": "c_TV",
            "affects_primary_fusion": False,
            "integration_status": "standalone_step13b_only",
            "official_test_accessed": False,
        }


class SelectiveReliabilityController(DeterministicReliabilityController):
    """Parameter-free v0.28 selective reliability controller.

    Quality controls modality allocation and compatibility independently
    controls the interaction multiplier. All inputs are detached before use.
    """

    def __init__(
        self,
        *,
        mode: str = "combined",
        epsilon: float = DEFAULT_EPSILON,
        allocation_exponent: float = DEFAULT_ALLOCATION_EXPONENT,
        interaction_exponent: float = DEFAULT_INTERACTION_EXPONENT,
    ) -> None:
        super().__init__(epsilon=epsilon)
        if mode not in SELECTIVE_CONTROLLER_MODES:
            raise ValueError(
                f"mode must be one of {sorted(SELECTIVE_CONTROLLER_MODES)}; "
                f"got {mode!r}."
            )
        for name, value in (
            ("allocation_exponent", allocation_exponent),
            ("interaction_exponent", interaction_exponent),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a positive real number.")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        self.mode = mode
        self.allocation_exponent = float(allocation_exponent)
        self.interaction_exponent = float(interaction_exponent)

    @property
    def selective_weight_reallocation(self) -> bool:
        return self.mode in {"weights_only", "combined"}

    @property
    def interaction_suppression(self) -> bool:
        return self.mode in {"interaction_only", "combined"}

    def forward(
        self,
        text_quality: Tensor,
        vision_quality: Tensor,
        compatibility: Tensor,
    ) -> ReliabilityControllerOutput:
        self._validate_triplet(text_quality, vision_quality, compatibility)
        q_text = text_quality.detach()
        q_vision = vision_quality.detach()
        c_tv = compatibility.detach()

        if self.selective_weight_reallocation:
            text_score = (self.epsilon + q_text).pow(
                self.allocation_exponent
            )
            vision_score = (self.epsilon + q_vision).pow(
                self.allocation_exponent
            )
            denominator = text_score + vision_score
            text_weight = text_score / denominator
            vision_weight = vision_score / denominator
        else:
            text_score = torch.ones_like(q_text)
            vision_score = torch.ones_like(q_vision)
            text_weight = torch.full_like(q_text, 0.5)
            vision_weight = torch.full_like(q_vision, 0.5)

        if self.interaction_suppression:
            interaction_multiplier = self.epsilon + (
                1.0 - self.epsilon
            ) * c_tv.pow(self.interaction_exponent)
        else:
            interaction_multiplier = torch.ones_like(c_tv)

        weights = torch.cat((text_weight, vision_weight), dim=1)
        self._validate_outputs(
            effective_text_reliability=text_score,
            effective_vision_reliability=vision_score,
            text_weight=text_weight,
            vision_weight=vision_weight,
            weights=weights,
            interaction_multiplier=interaction_multiplier,
        )
        return ReliabilityControllerOutput(
            effective_text_reliability=text_score,
            effective_vision_reliability=vision_score,
            text_weight=text_weight,
            vision_weight=vision_weight,
            weights=weights,
            interaction_multiplier=interaction_multiplier,
        )

    def architecture_metadata(self) -> dict[str, object]:
        return {
            "module": self.__class__.__name__,
            "protocol_version": V028_STEP1_PROTOCOL_VERSION,
            "implementation_version": V028_STEP2_IMPLEMENTATION_VERSION,
            "mode": self.mode,
            "epsilon": self.epsilon,
            "allocation_exponent_beta": self.allocation_exponent,
            "interaction_exponent_gamma": self.interaction_exponent,
            "selective_weight_reallocation": self.selective_weight_reallocation,
            "interaction_suppression": self.interaction_suppression,
            "parameter_free": True,
            "trainable_parameter_count": self.parameter_count,
            "stop_gradient_controller_inputs": True,
            "compatibility_affects_allocation": False,
            "quality_affects_interaction_multiplier": False,
            "official_test_accessed": False,
        }
