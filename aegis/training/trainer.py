"""
AEGIS Joint Training Engine.

Version: 0.16.0
"""

from typing import Dict, Optional

import torch
from torch import nn

from aegis.alignment import (
    CrossModalAlignmentModel,
)

from aegis.classification import (
    HierarchicalInformationIntegrityClassifier,
    WeightedHierarchicalLoss,
)

from .batch import (
    TrainingBatch,
)

from .metrics import (
    compute_hierarchical_metrics,
)

from .state import (
    TrainingState,
)


class AEGISTrainer:
    """
    Joint trainer for:

    1. Cross-modal semantic alignment
    2. Hierarchical Information Integrity classification

    The pretrained XLM-R and CLIP encoders are treated
    as upstream representation providers in v0.16.0.
    """

    def __init__(
        self,
        alignment_model:
        CrossModalAlignmentModel,

        classification_model:
        HierarchicalInformationIntegrityClassifier,

        optimizer=None,

        hierarchical_loss=None,

        device: str = "cpu",

        alignment_loss_weight: float = 1.0,

        classification_loss_weight: float = 1.0,

        gradient_clip_norm:
        Optional[float] = 1.0,
    ):
        if not isinstance(
            alignment_model,
            CrossModalAlignmentModel,
        ):
            raise TypeError(
                "alignment_model must be "
                "CrossModalAlignmentModel."
            )

        if not isinstance(
            classification_model,
            HierarchicalInformationIntegrityClassifier,
        ):
            raise TypeError(
                "classification_model must be "
                "HierarchicalInformationIntegrityClassifier."
            )

        if alignment_loss_weight < 0:
            raise ValueError(
                "alignment_loss_weight must be >= 0."
            )

        if classification_loss_weight < 0:
            raise ValueError(
                "classification_loss_weight "
                "must be >= 0."
            )

        self.alignment_model = (
            alignment_model
        )

        self.classification_model = (
            classification_model
        )

        self.device = device

        self.alignment_loss_weight = (
            alignment_loss_weight
        )

        self.classification_loss_weight = (
            classification_loss_weight
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.alignment_model.to(
            self.device
        )

        self.classification_model.to(
            self.device
        )

        self.hierarchical_loss = (
            hierarchical_loss
            or WeightedHierarchicalLoss()
        )

        self.hierarchical_loss.to(
            self.device
        )

        parameters = list(
            self.alignment_model.parameters()
        ) + list(
            self.classification_model.parameters()
        )

        self.optimizer = (
            optimizer
            or torch.optim.AdamW(
                parameters,
                lr=1e-4,
                weight_decay=1e-4,
            )
        )

        self.state = TrainingState()

    def forward_batch(
        self,
        batch: TrainingBatch,
        compute_alignment_loss: bool = True,
    ) -> Dict:

        batch.validate(
            text_dim=(
                self.alignment_model.text_dim
            ),

            vision_dim=(
                self.alignment_model.vision_dim
            ),
        )

        batch = batch.to(
            self.device
        )

        alignment_outputs = (
            self.alignment_model(
                batch.text_embeddings,
                batch.vision_embeddings,
                compute_loss=(
                    compute_alignment_loss
                ),
            )
        )

        classification_outputs = (
            self.classification_model(
                alignment_outputs[
                    "fused_embedding"
                ]
            )
        )

        classification_losses = (
            self.hierarchical_loss(
                classification_outputs[
                    "integrity_logits"
                ],

                classification_outputs[
                    "threat_logits"
                ],

                batch.integrity_targets,

                batch.threat_targets,
            )
        )

        if compute_alignment_loss:

            alignment_loss = (
                alignment_outputs[
                    "alignment_loss"
                ]
            )

        else:
            alignment_loss = (
                classification_losses[
                    "loss"
                ]
                * 0.0
            )

        total_loss = (
            self.alignment_loss_weight
            * alignment_loss
            +
            self.classification_loss_weight
            * classification_losses[
                "loss"
            ]
        )

        metrics = (
            compute_hierarchical_metrics(
                classification_outputs[
                    "integrity_logits"
                ],

                classification_outputs[
                    "threat_logits"
                ],

                batch.integrity_targets,

                batch.threat_targets,
            )
        )

        return {
            "loss": total_loss,

            "alignment_loss": (
                alignment_loss
            ),

            "classification_loss": (
                classification_losses[
                    "loss"
                ]
            ),

            "integrity_loss": (
                classification_losses[
                    "integrity_loss"
                ]
            ),

            "threat_loss": (
                classification_losses[
                    "threat_loss"
                ]
            ),

            "alignment_outputs": (
                alignment_outputs
            ),

            "classification_outputs": (
                classification_outputs
            ),

            "metrics": metrics,
        }

    def train_step(
        self,
        batch: TrainingBatch,
    ) -> Dict[str, float]:
        """
        Execute one optimization step.
        """

        self.alignment_model.train()
        self.classification_model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        outputs = self.forward_batch(
            batch,
            compute_alignment_loss=True,
        )

        loss = outputs[
            "loss"
        ]

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss detected."
            )

        loss.backward()

        if (
            self.gradient_clip_norm
            is not None
        ):

            parameters = list(
                self.alignment_model.parameters()
            ) + list(
                self.classification_model.parameters()
            )

            nn.utils.clip_grad_norm_(
                parameters,
                max_norm=(
                    self.gradient_clip_norm
                ),
            )

        self.optimizer.step()

        self.state.global_step += 1

        result = {
            "loss": float(
                loss.detach().cpu().item()
            ),

            "alignment_loss": float(
                outputs[
                    "alignment_loss"
                ]
                .detach()
                .cpu()
                .item()
            ),

            "classification_loss": float(
                outputs[
                    "classification_loss"
                ]
                .detach()
                .cpu()
                .item()
            ),

            "integrity_loss": float(
                outputs[
                    "integrity_loss"
                ]
                .detach()
                .cpu()
                .item()
            ),

            "threat_loss": float(
                outputs[
                    "threat_loss"
                ]
                .detach()
                .cpu()
                .item()
            ),
        }

        result.update(
            outputs[
                "metrics"
            ]
        )

        self.state.metrics = (
            result.copy()
        )

        return result