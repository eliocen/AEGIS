"""
AEGIS cross-modal semantic alignment model.

Version: 0.27.6-dev

This module preserves the validated v0.24/v0.25 cross-modal alignment paths
while introducing the v0.27 M4q/M4qc diagnostic-only quality and
cross-modal compatibility pathways.

Fusion pathways
---------------
M0 / legacy:
    projections -> gated multimodal fusion

M1 / interaction-only:
    projections -> explicit interaction -> deterministic residual fusion

M1b / gated-interaction:
    projections -> validated gated fusion -> explicit interaction residual

M2 / reliability-only:
    projections -> validated gated fusion
                -> modality-only reliability adaptive residual

M2b / interaction-reliability:
    projections -> validated gated fusion
                -> interaction-conditioned reliability adaptive residual

M3 / evidence-aware:
    projections -> explicit interaction
                -> interaction-conditioned reliability
                -> adaptive fusion

M4q / quality-supervised:
    projections -> M1b gated-interaction fusion
                -> separate text/vision intrinsic-quality estimators

M4qc / quality-compatibility-supervised:
    projections -> exact M1b gated-interaction fusion
                -> separate text/vision intrinsic-quality estimators
                -> separate text-image compatibility estimator

M4q/M4qc invariants
-------------------
The quality and compatibility estimators are diagnostic-only with respect to
the forward fusion equation:

    z_M4q = z_M4qc = z_M1b

The predicted q_T, q_V, and c_TV scores never enter the gated-interaction
fusion path. Their supervision is auxiliary only.

Compatibility invariant
-----------------------
With all experimental flags False, the model uses exactly the historical
v0.24 functional path: projections -> GatedMultimodalFusion.
"""

from __future__ import annotations

import torch
from torch import nn

from ..reliability.compatibility_estimator import CrossModalCompatibilityEstimator
from ..reliability.quality_estimator import ModalityQualityEstimator
from .evidence_integration import AEGISEvidenceIntegrationBlock
from .fusion import GatedMultimodalFusion
from .gated_interaction_fusion import GatedInteractionEvidenceFusion
from .interaction_fusion import InteractionOnlyEvidenceFusion
from .interaction_reliability_residual_fusion import (
    InteractionConditionedReliabilityResidualFusion,
)
from .loss import SymmetricContrastiveLoss
from .projection import ProjectionHead
from .reliability_residual_fusion import ReliabilityResidualFusion


class CrossModalAlignmentModel(nn.Module):
    """Trainable AEGIS cross-modal semantic alignment subsystem."""

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        shared_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
        evidence_aware: bool = False,
        interaction_only: bool = False,
        gated_interaction: bool = False,
        reliability_only: bool = False,
        interaction_reliability: bool = False,
        quality_supervised: bool = False,
        quality_compatibility_supervised: bool = False,
        evidence_reliability_hidden_dim: int = 256,
        evidence_interaction_dropout: float = 0.1,
        evidence_reliability_dropout: float = 0.1,
        evidence_fusion_temperature: float = 1.0,
        quality_hidden_dims: tuple[int, int] = (256, 64),
        quality_dropout: float = 0.1,
        compatibility_hidden_dims: tuple[int, int] = (256, 64),
        compatibility_dropout: float = 0.1,
        compatibility_cosine_eps: float = 1e-8,
    ):
        super().__init__()

        for name, value in (
            ("evidence_aware", evidence_aware),
            ("interaction_only", interaction_only),
            ("gated_interaction", gated_interaction),
            ("reliability_only", reliability_only),
            ("interaction_reliability", interaction_reliability),
            ("quality_supervised", quality_supervised),
            ("quality_compatibility_supervised", quality_compatibility_supervised),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")

        # Preserve historical M1/M3 validation messages exactly.
        if evidence_aware and interaction_only:
            raise ValueError(
                "evidence_aware and interaction_only cannot both be True."
            )

        if gated_interaction and evidence_aware:
            raise ValueError(
                "gated_interaction and evidence_aware cannot both be True."
            )

        if gated_interaction and interaction_only:
            raise ValueError(
                "gated_interaction and interaction_only cannot both be True."
            )

        if reliability_only and evidence_aware:
            raise ValueError(
                "reliability_only and evidence_aware cannot both be True."
            )

        if reliability_only and interaction_only:
            raise ValueError(
                "reliability_only and interaction_only cannot both be True."
            )

        if reliability_only and gated_interaction:
            raise ValueError(
                "reliability_only and gated_interaction cannot both be True."
            )

        if interaction_reliability and any((
            evidence_aware,
            interaction_only,
            gated_interaction,
            reliability_only,
        )):
            raise ValueError(
                "interaction_reliability cannot be combined with another "
                "experimental fusion flag."
            )

        if quality_compatibility_supervised and any((
            evidence_aware,
            interaction_only,
            gated_interaction,
            reliability_only,
            interaction_reliability,
            quality_supervised,
        )):
            raise ValueError(
                "quality_compatibility_supervised cannot be combined with "
                "another experimental fusion flag."
            )

        if quality_supervised and any((
            evidence_aware,
            interaction_only,
            gated_interaction,
            reliability_only,
            interaction_reliability,
            quality_compatibility_supervised,
        )):
            raise ValueError(
                "quality_supervised cannot be combined with another "
                "experimental fusion flag."
            )

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.shared_dim = shared_dim

        self.evidence_aware = evidence_aware
        self.interaction_only = interaction_only
        self.gated_interaction = gated_interaction
        self.reliability_only = reliability_only
        self.interaction_reliability = interaction_reliability
        self.quality_supervised = quality_supervised
        self.quality_compatibility_supervised = quality_compatibility_supervised

        if self.evidence_aware:
            self.fusion_architecture = "evidence_aware"
        elif self.interaction_only:
            self.fusion_architecture = "interaction_only"
        elif self.gated_interaction:
            self.fusion_architecture = "gated_interaction"
        elif self.reliability_only:
            self.fusion_architecture = "reliability_only"
        elif self.interaction_reliability:
            self.fusion_architecture = "interaction_reliability"
        elif self.quality_compatibility_supervised:
            self.fusion_architecture = "quality_compatibility_supervised"
        elif self.quality_supervised:
            self.fusion_architecture = "quality_supervised"
        else:
            self.fusion_architecture = "legacy"

        self.text_projection = ProjectionHead(
            input_dim=text_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )

        self.vision_projection = ProjectionHead(
            input_dim=vision_dim,
            output_dim=shared_dim,
            dropout=dropout,
        )

        # Historical M0 gate is always retained. M2 deliberately reuses this
        # same gate instead of owning a duplicate gate.
        self.fusion = GatedMultimodalFusion(
            dimension=shared_dim
        )

        self.interaction_fusion = (
            InteractionOnlyEvidenceFusion(
                dimension=shared_dim,
                interaction_dropout=evidence_interaction_dropout,
            )
            if self.interaction_only
            else None
        )

        # M4q deliberately reuses the exact M1b fusion module class and
        # forward equation. This is the structural basis of z_M4q = z_M1b.
        self.gated_interaction_fusion = (
            GatedInteractionEvidenceFusion(
                dimension=shared_dim,
                interaction_dropout=evidence_interaction_dropout,
            )
            if (
                self.gated_interaction
                or self.quality_supervised
                or self.quality_compatibility_supervised
            )
            else None
        )

        self.reliability_only_fusion = (
            ReliabilityResidualFusion(
                dimension=shared_dim,
                reliability_hidden_dim=(
                    evidence_reliability_hidden_dim
                ),
                reliability_dropout=(
                    evidence_reliability_dropout
                ),
                fusion_temperature=(
                    evidence_fusion_temperature
                ),
            )
            if self.reliability_only
            else None
        )

        self.interaction_reliability_fusion = (
            InteractionConditionedReliabilityResidualFusion(
                dimension=shared_dim,
                reliability_hidden_dim=evidence_reliability_hidden_dim,
                interaction_dropout=evidence_interaction_dropout,
                reliability_dropout=evidence_reliability_dropout,
                fusion_temperature=evidence_fusion_temperature,
            )
            if self.interaction_reliability
            else None
        )

        self.evidence_integration = (
            AEGISEvidenceIntegrationBlock(
                dimension=shared_dim,
                reliability_hidden_dim=(
                    evidence_reliability_hidden_dim
                ),
                interaction_dropout=(
                    evidence_interaction_dropout
                ),
                reliability_dropout=(
                    evidence_reliability_dropout
                ),
                fusion_temperature=(
                    evidence_fusion_temperature
                ),
            )
            if self.evidence_aware
            else None
        )

        # Instantiate quality heads only after all historical fusion modules.
        # Consequently, adding M4q does not perturb initialization order of
        # common M1b components under paired common-component seeding.
        self.text_quality_estimator = (
            ModalityQualityEstimator(
                input_dim=shared_dim,
                hidden_dims=quality_hidden_dims,
                dropout=quality_dropout,
            )
            if (
                self.quality_supervised
                or self.quality_compatibility_supervised
            )
            else None
        )

        self.vision_quality_estimator = (
            ModalityQualityEstimator(
                input_dim=shared_dim,
                hidden_dims=quality_hidden_dims,
                dropout=quality_dropout,
            )
            if (
                self.quality_supervised
                or self.quality_compatibility_supervised
            )
            else None
        )

        # M4qc compatibility head is instantiated after all M1b and M4q
        # components. This preserves common-component initialization order.
        self.compatibility_estimator = (
            CrossModalCompatibilityEstimator(
                shared_dim=shared_dim,
                hidden_dims=compatibility_hidden_dims,
                dropout=compatibility_dropout,
                cosine_eps=compatibility_cosine_eps,
            )
            if self.quality_compatibility_supervised
            else None
        )

        self.contrastive_loss = SymmetricContrastiveLoss(
            temperature=temperature
        )

    def align(
        self,
        text_embedding: torch.Tensor,
        vision_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_text = self.text_projection(text_embedding)
        aligned_vision = self.vision_projection(vision_embedding)
        return aligned_text, aligned_vision

    def forward(
        self,
        text_embedding: torch.Tensor,
        vision_embedding: torch.Tensor,
        compute_loss: bool = False,
    ) -> dict[str, torch.Tensor]:
        aligned_text, aligned_vision = self.align(
            text_embedding,
            vision_embedding,
        )

        if self.evidence_aware:
            if self.evidence_integration is None:
                raise RuntimeError(
                    "Evidence-aware mode is enabled but the evidence "
                    "integration block is not initialized."
                )

            out = self.evidence_integration(
                aligned_text,
                aligned_vision,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out["cosine_similarity"],
                "interaction_embedding": out[
                    "interaction_embedding"
                ],
                "text_reliability": out["first_reliability"],
                "vision_reliability": out[
                    "second_reliability"
                ],
                "text_weight": out["first_weight"],
                "vision_weight": out["second_weight"],
                "evidence_weights": out["weights"],
            }

        elif self.interaction_only:
            if self.interaction_fusion is None:
                raise RuntimeError(
                    "Interaction-only mode is enabled but the "
                    "interaction fusion block is not initialized."
                )

            out = self.interaction_fusion(
                aligned_text,
                aligned_vision,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out[
                    "cosine_similarity"
                ],
                "interaction_embedding": out[
                    "interaction_embedding"
                ],
                "interaction_base_fusion": out[
                    "base_fusion"
                ],
                "scaled_interaction": out[
                    "scaled_interaction"
                ],
                "interaction_scale": out[
                    "interaction_scale"
                ],
            }

        elif self.gated_interaction:
            if self.gated_interaction_fusion is None:
                raise RuntimeError(
                    "Gated-interaction mode is enabled but the "
                    "gated interaction fusion block is not initialized."
                )

            out = self.gated_interaction_fusion(
                aligned_text,
                aligned_vision,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out[
                    "cosine_similarity"
                ],
                "interaction_embedding": out[
                    "interaction_embedding"
                ],
                "gated_interaction_base_fusion": out[
                    "gated_base_fusion"
                ],
                "scaled_interaction": out[
                    "scaled_interaction"
                ],
                "interaction_scale": out[
                    "interaction_scale"
                ],
            }

        elif self.quality_compatibility_supervised:
            if self.gated_interaction_fusion is None:
                raise RuntimeError(
                    "M4qc mode is enabled but the M1b gated interaction "
                    "fusion block is not initialized."
                )
            if self.text_quality_estimator is None:
                raise RuntimeError(
                    "M4qc mode is enabled but the text quality estimator "
                    "is not initialized."
                )
            if self.vision_quality_estimator is None:
                raise RuntimeError(
                    "M4qc mode is enabled but the vision quality estimator "
                    "is not initialized."
                )
            if self.compatibility_estimator is None:
                raise RuntimeError(
                    "M4qc mode is enabled but the compatibility estimator "
                    "is not initialized."
                )

            # Exact M1b fusion equation. q_T, q_V, and c_TV are computed
            # diagnostically and are never passed into the fusion module.
            out = self.gated_interaction_fusion(
                aligned_text,
                aligned_vision,
            )
            text_quality = self.text_quality_estimator(aligned_text)
            vision_quality = self.vision_quality_estimator(aligned_vision)
            compatibility_score = self.compatibility_estimator(
                aligned_text,
                aligned_vision,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out["cosine_similarity"],
                "interaction_embedding": out["interaction_embedding"],
                "gated_interaction_base_fusion": out["gated_base_fusion"],
                "scaled_interaction": out["scaled_interaction"],
                "interaction_scale": out["interaction_scale"],
                "text_quality": text_quality,
                "vision_quality": vision_quality,
                "compatibility_score": compatibility_score,
            }

        elif self.quality_supervised:
            if self.gated_interaction_fusion is None:
                raise RuntimeError(
                    "Quality-supervised mode is enabled but the M1b "
                    "gated interaction fusion block is not initialized."
                )

            if self.text_quality_estimator is None:
                raise RuntimeError(
                    "Quality-supervised mode is enabled but the text "
                    "quality estimator is not initialized."
                )

            if self.vision_quality_estimator is None:
                raise RuntimeError(
                    "Quality-supervised mode is enabled but the vision "
                    "quality estimator is not initialized."
                )

            # Exact M1b fusion equation. Quality scores are computed only
            # after the fused representation has been produced and are never
            # passed into the fusion module.
            out = self.gated_interaction_fusion(
                aligned_text,
                aligned_vision,
            )

            text_quality = self.text_quality_estimator(
                aligned_text
            )
            vision_quality = self.vision_quality_estimator(
                aligned_vision
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out[
                    "cosine_similarity"
                ],
                "interaction_embedding": out[
                    "interaction_embedding"
                ],
                "gated_interaction_base_fusion": out[
                    "gated_base_fusion"
                ],
                "scaled_interaction": out[
                    "scaled_interaction"
                ],
                "interaction_scale": out[
                    "interaction_scale"
                ],
                "text_quality": text_quality,
                "vision_quality": vision_quality,
            }

        elif self.interaction_reliability:
            if self.interaction_reliability_fusion is None:
                raise RuntimeError(
                    "Interaction-reliability mode is enabled but the "
                    "interaction-conditioned reliability residual block "
                    "is not initialized."
                )

            gated_base_fusion = self.fusion(aligned_text, aligned_vision)
            out = self.interaction_reliability_fusion(
                aligned_text, aligned_vision, gated_base_fusion
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "interaction_reliability_base_fusion": out["gated_base_fusion"],
                "evidence_difference": out["difference"],
                "evidence_product": out["product"],
                "cosine_similarity": out["cosine_similarity"],
                "interaction_embedding": out["interaction_embedding"],
                "text_reliability": out["first_reliability"],
                "vision_reliability": out["second_reliability"],
                "text_weight": out["first_weight"],
                "vision_weight": out["second_weight"],
                "evidence_weights": out["weights"],
                "reliability_adaptive_fusion": out["reliability_adaptive_fusion"],
                "scaled_reliability_fusion": out["scaled_reliability_fusion"],
                "reliability_scale": out["reliability_scale"],
            }

        elif self.reliability_only:
            if self.reliability_only_fusion is None:
                raise RuntimeError(
                    "Reliability-only mode is enabled but the "
                    "reliability residual fusion block is not initialized."
                )

            gated_base_fusion = self.fusion(
                aligned_text,
                aligned_vision,
            )

            out = self.reliability_only_fusion(
                aligned_text,
                aligned_vision,
                gated_base_fusion,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": out["fused_embedding"],
                "reliability_base_fusion": out[
                    "gated_base_fusion"
                ],
                "text_reliability": out[
                    "first_reliability"
                ],
                "vision_reliability": out[
                    "second_reliability"
                ],
                "text_weight": out["first_weight"],
                "vision_weight": out["second_weight"],
                "evidence_weights": out["weights"],
                "reliability_adaptive_fusion": out[
                    "reliability_adaptive_fusion"
                ],
                "scaled_reliability_fusion": out[
                    "scaled_reliability_fusion"
                ],
                "reliability_scale": out[
                    "reliability_scale"
                ],
            }

        else:
            fused = self.fusion(
                aligned_text,
                aligned_vision,
            )

            result = {
                "aligned_text": aligned_text,
                "aligned_vision": aligned_vision,
                "fused_embedding": fused,
            }

        if compute_loss:
            result["alignment_loss"] = (
                self.contrastive_loss(
                    aligned_text,
                    aligned_vision,
                )
            )

        return result
