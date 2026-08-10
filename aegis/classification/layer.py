"""
AEGIS Hierarchical Information Integrity
Classification Layer.

Version: 0.10.0
"""

import torch
from torch.nn import functional as F

from aegis.alignment import (
    MultimodalRepresentation,
)
from aegis.pipeline import AEGISLayer

from .labels import (
    INTEGRITY_INDEX_TO_LABEL,
    THREAT_INDEX_TO_LABEL,
    IntegrityStatus,
)
from .model import (
    HierarchicalInformationIntegrityClassifier,
)
from .output import (
    HierarchicalClassificationOutput,
)


class HierarchicalClassificationLayer(
    AEGISLayer
):
    """
    AEGIS hierarchical Information Integrity
    inference layer.
    """

    layer_name = (
        "hierarchical_information_integrity_classification"
    )

    def __init__(
        self,
        model:
        HierarchicalInformationIntegrityClassifier,
        config=None,
        device: str = "cpu",
    ):
        super().__init__(config)

        if not isinstance(
            model,
            HierarchicalInformationIntegrityClassifier,
        ):
            raise TypeError(
                "model must be a "
                "HierarchicalInformationIntegrityClassifier."
            )

        self.model = model
        self.device = device

        self.model.to(
            self.device
        )

    def process(
        self,
        data: MultimodalRepresentation,
    ):
        if not isinstance(
            data,
            MultimodalRepresentation,
        ):
            raise TypeError(
                "HierarchicalClassificationLayer "
                "expects MultimodalRepresentation."
            )

        if data.fused_embedding is None:
            raise ValueError(
                "MultimodalRepresentation "
                "does not contain a fused embedding."
            )

        embedding = data.fused_embedding

        if not isinstance(
            embedding,
            torch.Tensor,
        ):
            embedding = torch.tensor(
                embedding,
                dtype=torch.float32,
            )

        embedding = embedding.float()

        if embedding.ndim == 1:
            embedding = (
                embedding.unsqueeze(0)
            )

        if embedding.ndim != 2:
            raise ValueError(
                "Fused embedding must have shape "
                "[dimension] or "
                "[batch_size, dimension]."
            )

        if embedding.shape[0] != 1:
            raise ValueError(
                "ClassificationLayer.process currently "
                "supports single-sample inference."
            )

        embedding = embedding.to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                embedding
            )

            integrity_probs = F.softmax(
                outputs[
                    "integrity_logits"
                ],
                dim=-1,
            )

            threat_probs = F.softmax(
                outputs[
                    "threat_logits"
                ],
                dim=-1,
            )

        integrity_index = int(
            integrity_probs
            .argmax(
                dim=-1
            )
            .item()
        )

        integrity_status = (
            INTEGRITY_INDEX_TO_LABEL[
                integrity_index
            ]
        )

        integrity_confidence = float(
            integrity_probs[
                0,
                integrity_index
            ].item()
        )

        threat_type = None
        threat_confidence = None

        if (
            integrity_status
            == IntegrityStatus.HARMFUL
        ):

            threat_index = int(
                threat_probs
                .argmax(
                    dim=-1
                )
                .item()
            )

            threat_type = (
                THREAT_INDEX_TO_LABEL[
                    threat_index
                ]
            )

            threat_confidence = float(
                threat_probs[
                    0,
                    threat_index
                ].item()
            )

        return (
            HierarchicalClassificationOutput(
                sample_id=data.sample_id,

                integrity_status=(
                    integrity_status
                ),

                integrity_confidence=(
                    integrity_confidence
                ),

                threat_type=(
                    threat_type
                ),

                threat_confidence=(
                    threat_confidence
                ),

                integrity_probabilities=(
                    integrity_probs
                    .squeeze(0)
                    .detach()
                    .cpu()
                ),

                threat_probabilities=(
                    threat_probs
                    .squeeze(0)
                    .detach()
                    .cpu()
                ),

                metadata={
                    "classification": (
                        "hierarchical"
                    ),
                    "input_dimension": (
                        self.model.input_dim
                    ),
                    "device": (
                        self.device
                    ),
                },
            )
        )