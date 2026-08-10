"""
AEGIS cross-modal alignment pipeline layer.

Version: 0.9.0
"""

import torch

from aegis.pipeline import AEGISLayer
from aegis.representation import (
    ParallelRepresentationBundle,
    TextRepresentation,
    VisionRepresentation,
)

from .model import CrossModalAlignmentModel
from .output import MultimodalRepresentation


class CrossModalAlignmentLayer(AEGISLayer):
    """
    AEGIS cross-modal semantic alignment layer.

    Accepts either:
    1. ParallelRepresentationBundle
    2. Tuple(TextRepresentation, VisionRepresentation)

    The layer maps text and vision representations into
    the shared semantic space, performs gated fusion, and
    returns a MultimodalRepresentation.
    """

    layer_name = "cross_modal_semantic_alignment"

    def __init__(
        self,
        model: CrossModalAlignmentModel,
        config=None,
        device: str = "cpu",
    ):
        super().__init__(config)

        if not isinstance(
            model,
            CrossModalAlignmentModel,
        ):
            raise TypeError(
                "model must be a CrossModalAlignmentModel."
            )

        self.model = model
        self.device = device

        self.model.to(self.device)

    def process(self, data):
        """
        Align text and vision representations.

        Parameters
        ----------
        data:
            Either:
            - ParallelRepresentationBundle
            - tuple(TextRepresentation, VisionRepresentation)

        Returns
        -------
        MultimodalRepresentation
        """

        if isinstance(
            data,
            ParallelRepresentationBundle,
        ):
            if not data.is_multimodal:
                raise ValueError(
                    "Cross-modal alignment requires both "
                    "text and vision representations."
                )

            text_representation = data.text
            vision_representation = data.vision

        elif (
            isinstance(data, tuple)
            and len(data) == 2
        ):
            text_representation, vision_representation = data

        else:
            raise TypeError(
                "CrossModalAlignmentLayer expects either "
                "a ParallelRepresentationBundle or "
                "(TextRepresentation, VisionRepresentation)."
            )

        if not isinstance(
            text_representation,
            TextRepresentation,
        ):
            raise TypeError(
                "Text input must be a TextRepresentation."
            )

        if not isinstance(
            vision_representation,
            VisionRepresentation,
        ):
            raise TypeError(
                "Vision input must be a VisionRepresentation."
            )

        if (
            text_representation.sample_id
            != vision_representation.sample_id
        ):
            raise ValueError(
                "Text and vision representations must belong "
                "to the same sample_id."
            )

        text_embedding = text_representation.embedding
        vision_embedding = vision_representation.embedding

        if not isinstance(
            text_embedding,
            torch.Tensor,
        ):
            text_embedding = torch.tensor(
                text_embedding,
                dtype=torch.float32,
            )

        if not isinstance(
            vision_embedding,
            torch.Tensor,
        ):
            vision_embedding = torch.tensor(
                vision_embedding,
                dtype=torch.float32,
            )

        text_embedding = text_embedding.float()

        vision_embedding = vision_embedding.float()

        if text_embedding.ndim == 1:
            text_embedding = text_embedding.unsqueeze(0)

        if vision_embedding.ndim == 1:
            vision_embedding = vision_embedding.unsqueeze(0)

        if text_embedding.ndim != 2:
            raise ValueError(
                "Text embedding must have shape "
                "[dimension] or [batch_size, dimension]."
            )

        if vision_embedding.ndim != 2:
            raise ValueError(
                "Vision embedding must have shape "
                "[dimension] or [batch_size, dimension]."
            )

        if text_embedding.shape[0] != vision_embedding.shape[0]:
            raise ValueError(
                "Text and vision batch sizes must match."
            )

        if text_embedding.shape[-1] != self.model.text_dim:
            raise ValueError(
                f"Expected text embedding dimension "
                f"{self.model.text_dim}, "
                f"received {text_embedding.shape[-1]}."
            )

        if vision_embedding.shape[-1] != self.model.vision_dim:
            raise ValueError(
                f"Expected vision embedding dimension "
                f"{self.model.vision_dim}, "
                f"received {vision_embedding.shape[-1]}."
            )

        text_embedding = text_embedding.to(
            self.device
        )

        vision_embedding = vision_embedding.to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                text_embedding,
                vision_embedding,
                compute_loss=False,
            )

        aligned_text = (
            outputs["aligned_text"]
            .detach()
            .cpu()
        )

        aligned_vision = (
            outputs["aligned_vision"]
            .detach()
            .cpu()
        )

        fused_embedding = (
            outputs["fused_embedding"]
            .detach()
            .cpu()
        )

        # For single-sample inference, remove the batch dimension.
        if aligned_text.shape[0] == 1:
            aligned_text = aligned_text.squeeze(0)

        if aligned_vision.shape[0] == 1:
            aligned_vision = aligned_vision.squeeze(0)

        if fused_embedding.shape[0] == 1:
            fused_embedding = fused_embedding.squeeze(0)

        original_text_embedding = (
            text_embedding
            .detach()
            .cpu()
        )

        original_vision_embedding = (
            vision_embedding
            .detach()
            .cpu()
        )

        if original_text_embedding.shape[0] == 1:
            original_text_embedding = (
                original_text_embedding.squeeze(0)
            )

        if original_vision_embedding.shape[0] == 1:
            original_vision_embedding = (
                original_vision_embedding.squeeze(0)
            )

        return MultimodalRepresentation(
            sample_id=text_representation.sample_id,

            text_embedding=original_text_embedding,
            vision_embedding=original_vision_embedding,

            aligned_text=aligned_text,
            aligned_vision=aligned_vision,

            fused_embedding=fused_embedding,

            shared_dimension=self.model.shared_dim,

            text_available=True,
            vision_available=True,
            is_multimodal=True,

            metadata={
                "alignment": "learnable_projection",
                "fusion": "gated_multimodal",
                "device": self.device,
                "text_model": (
                    text_representation.model_name
                ),
                "vision_model": (
                    vision_representation.model_name
                ),
            },
        )