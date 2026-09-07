"""
AEGIS v0.27 Step13D — M4qcf gradient/causal audit.

This is a pre-formal-training implementation audit.  It does not evaluate
H7-F/H8-F/H9-F/H10-F, select checkpoints, access the official test split, or
make a robustness claim.

Required invariants
-------------------
1. The Step13B reliability controller is parameter-free.
2. alpha_T and alpha_V are finite, strictly bounded, and sum to one.
3. The M4qcf fused representation is exactly
       alpha_T*h_T + alpha_V*h_V + c_TV*z_I
   where z_I is M1b's deterministic scaled interaction residual.
4. Classification-side gradients reach the representation pathway but do not
   reach q_T/q_V/c_TV estimator parameters through the reliability controller.
5. Explicit quality/compatibility auxiliary losses still train their heads.
6. Lower compatibility suppresses the interaction contribution monotonically.
7. No official test data is accessed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from aegis.alignment.model import CrossModalAlignmentModel


AUDIT_VERSION = "0.27.0-step13d"
PROTOCOL_VERSION = "0.27.0-step13a"


def _parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm().cpu().item())
    return total


def _clear_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf.")


def run_audit(
    *,
    seed: int = 42,
    batch_size: int = 8,
    shared_dim: int = 32,
    device: str = "cpu",
) -> dict:
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2.")
    if shared_dim <= 0:
        raise ValueError("shared_dim must be > 0.")

    torch.manual_seed(seed)
    resolved_device = torch.device(device)

    model = CrossModalAlignmentModel(
        text_dim=768,
        vision_dim=512,
        shared_dim=shared_dim,
        temperature=0.07,
        evidence_interaction_dropout=0.0,
        quality_compatibility_fusion=True,
    ).to(resolved_device)
    model.eval()

    if model.reliability_controller is None:
        raise RuntimeError("M4qcf reliability controller is unavailable.")
    if model.text_quality_estimator is None:
        raise RuntimeError("M4qcf text quality estimator is unavailable.")
    if model.vision_quality_estimator is None:
        raise RuntimeError("M4qcf vision quality estimator is unavailable.")
    if model.compatibility_estimator is None:
        raise RuntimeError("M4qcf compatibility estimator is unavailable.")

    controller_parameters = sum(
        p.numel() for p in model.reliability_controller.parameters()
    )
    if controller_parameters != 0:
        raise RuntimeError(
            "Protocol violation: M4qcf reliability controller gained "
            f"{controller_parameters} trainable parameters."
        )

    text = torch.randn(
        batch_size, 768, device=resolved_device, requires_grad=True
    )
    vision = torch.randn(
        batch_size, 512, device=resolved_device, requires_grad=True
    )

    output = model(text, vision, compute_loss=False)

    required = {
        "aligned_text",
        "aligned_vision",
        "fused_embedding",
        "text_quality",
        "vision_quality",
        "compatibility_score",
        "effective_text_reliability",
        "effective_vision_reliability",
        "text_weight",
        "vision_weight",
        "evidence_weights",
        "interaction_multiplier",
        "reliability_weighted_fusion",
        "scaled_interaction",
        "compatibility_scaled_interaction",
    }
    missing = sorted(required.difference(output))
    if missing:
        raise RuntimeError(f"M4qcf output contract is incomplete: {missing}")

    for key in required:
        _assert_finite(key, output[key])

    alpha_text = output["text_weight"]
    alpha_vision = output["vision_weight"]
    if not torch.all(alpha_text > 0.0) or not torch.all(alpha_text < 1.0):
        raise RuntimeError("alpha_T is not strictly inside (0,1).")
    if not torch.all(alpha_vision > 0.0) or not torch.all(alpha_vision < 1.0):
        raise RuntimeError("alpha_V is not strictly inside (0,1).")
    if not torch.allclose(
        alpha_text + alpha_vision,
        torch.ones_like(alpha_text),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("alpha_T + alpha_V != 1.")

    expected_base = (
        alpha_text * output["aligned_text"]
        + alpha_vision * output["aligned_vision"]
    )
    expected_interaction = (
        output["interaction_multiplier"] * output["scaled_interaction"]
    )
    expected_fused = expected_base + expected_interaction

    if not torch.allclose(
        output["reliability_weighted_fusion"],
        expected_base,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("Reliability-weighted base violates Step13A.")

    if not torch.allclose(
        output["compatibility_scaled_interaction"],
        expected_interaction,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("Compatibility-scaled interaction violates Step13A.")

    if not torch.allclose(
        output["fused_embedding"],
        expected_fused,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("M4qcf fused representation violates Step13A.")

    # ---------------------------------------------------------------
    # Classification-side causal audit.
    # ---------------------------------------------------------------
    _clear_gradients(model)
    if text.grad is not None:
        text.grad = None
    if vision.grad is not None:
        vision.grad = None

    output_cls = model(text, vision, compute_loss=False)
    classification_surrogate = output_cls["fused_embedding"].square().mean()
    classification_surrogate.backward()

    q_text_cls_grad = _parameter_grad_norm(
        model.text_quality_estimator.parameters()
    )
    q_vision_cls_grad = _parameter_grad_norm(
        model.vision_quality_estimator.parameters()
    )
    compatibility_cls_grad = _parameter_grad_norm(
        model.compatibility_estimator.parameters()
    )

    if q_text_cls_grad != 0.0:
        raise RuntimeError(
            "Classification gradient leaked into text quality estimator."
        )
    if q_vision_cls_grad != 0.0:
        raise RuntimeError(
            "Classification gradient leaked into vision quality estimator."
        )
    if compatibility_cls_grad != 0.0:
        raise RuntimeError(
            "Classification gradient leaked into compatibility estimator."
        )

    text_projection_cls_grad = _parameter_grad_norm(
        model.text_projection.parameters()
    )
    vision_projection_cls_grad = _parameter_grad_norm(
        model.vision_projection.parameters()
    )
    interaction_cls_grad = _parameter_grad_norm(
        model.gated_interaction_fusion.interaction.parameters()
    )

    if text_projection_cls_grad <= 0.0:
        raise RuntimeError(
            "Classification gradient failed to reach text projection."
        )
    if vision_projection_cls_grad <= 0.0:
        raise RuntimeError(
            "Classification gradient failed to reach vision projection."
        )
    if interaction_cls_grad <= 0.0:
        raise RuntimeError(
            "Classification gradient failed to reach interaction pathway."
        )

    # ---------------------------------------------------------------
    # Explicit auxiliary-loss audit.
    # ---------------------------------------------------------------
    _clear_gradients(model)
    output_aux = model(text.detach(), vision.detach(), compute_loss=False)

    q_text_target = torch.ones_like(output_aux["text_quality"])
    q_vision_target = torch.zeros_like(output_aux["vision_quality"])
    compatibility_target = torch.ones_like(output_aux["compatibility_score"])

    auxiliary_loss = (
        nn.functional.mse_loss(
            output_aux["text_quality"], q_text_target
        )
        + nn.functional.mse_loss(
            output_aux["vision_quality"], q_vision_target
        )
        + nn.functional.binary_cross_entropy(
            output_aux["compatibility_score"], compatibility_target
        )
    )
    auxiliary_loss.backward()

    q_text_aux_grad = _parameter_grad_norm(
        model.text_quality_estimator.parameters()
    )
    q_vision_aux_grad = _parameter_grad_norm(
        model.vision_quality_estimator.parameters()
    )
    compatibility_aux_grad = _parameter_grad_norm(
        model.compatibility_estimator.parameters()
    )

    if q_text_aux_grad <= 0.0:
        raise RuntimeError("Explicit L_Q failed to train text quality head.")
    if q_vision_aux_grad <= 0.0:
        raise RuntimeError("Explicit L_Q failed to train vision quality head.")
    if compatibility_aux_grad <= 0.0:
        raise RuntimeError("Explicit L_C failed to train compatibility head.")

    # ---------------------------------------------------------------
    # Controller-level monotonic interaction suppression.
    # ---------------------------------------------------------------
    q_t = torch.full((4, 1), 0.8, device=resolved_device)
    q_v = torch.full((4, 1), 0.6, device=resolved_device)
    compatibility = torch.tensor(
        [[1.0], [0.75], [0.25], [0.0]],
        device=resolved_device,
    )
    control = model.reliability_controller(q_t, q_v, compatibility)
    multiplier = control.interaction_multiplier.reshape(-1)

    if not torch.all(multiplier[:-1] > multiplier[1:]):
        raise RuntimeError(
            "Decreasing compatibility did not monotonically suppress "
            "the interaction multiplier."
        )

    payload = {
        "audit_version": AUDIT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS",
        "architecture": "quality_compatibility_fusion",
        "seed": seed,
        "batch_size": batch_size,
        "shared_dim": shared_dim,
        "device": str(resolved_device),
        "controller_parameter_count": controller_parameters,
        "weight_sum_max_abs_error": float(
            (
                alpha_text
                + alpha_vision
                - torch.ones_like(alpha_text)
            )
            .abs()
            .max()
            .detach()
            .cpu()
            .item()
        ),
        "classification_side_gradients": {
            "text_quality_estimator": q_text_cls_grad,
            "vision_quality_estimator": q_vision_cls_grad,
            "compatibility_estimator": compatibility_cls_grad,
            "text_projection": text_projection_cls_grad,
            "vision_projection": vision_projection_cls_grad,
            "interaction_pathway": interaction_cls_grad,
        },
        "auxiliary_side_gradients": {
            "text_quality_estimator": q_text_aux_grad,
            "vision_quality_estimator": q_vision_aux_grad,
            "compatibility_estimator": compatibility_aux_grad,
        },
        "fusion_equation_exact": True,
        "weights_strictly_bounded": True,
        "weights_sum_to_one": True,
        "interaction_suppression_monotonic": True,
        "formal_hypotheses_computed": False,
        "official_test_accessed": False,
        "scientific_claim": (
            "Implementation/causal audit only; no robustness or "
            "performance benefit is inferred."
        ),
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit AEGIS v0.27 M4qcf gradient causality."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shared-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/fakeddit/v027_m4qcf_step13d/"
            "gradient_causal_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_audit(
        seed=args.seed,
        batch_size=args.batch_size,
        shared_dim=args.shared_dim,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved audit: {args.output}")


if __name__ == "__main__":
    main()
