"""
AEGIS Context Intelligence Layer.

Version: 0.11.0
"""

import torch

from aegis.alignment import (
    MultimodalRepresentation,
)

from aegis.pipeline import AEGISLayer

from .encoder import ContextEncoder
from .fusion import ContextAwareFusion
from .input import ContextInput
from .output import (
    ContextualizedRepresentation,
)


class ContextIntelligenceLayer(
    AEGISLayer
):
    """
    Enriches multimodal content representations
    with structured operational context.
    """

    layer_name = "context_intelligence"

    def __init__(
        self,
        encoder: ContextEncoder,
        fusion: ContextAwareFusion,
        config=None,
        device: str = "cpu",
    ):
        super().__init__(config)

        if not isinstance(
            encoder,
            ContextEncoder,
        ):
            raise TypeError(
                "encoder must be ContextEncoder."
            )

        if not isinstance(
            fusion,
            ContextAwareFusion,
        ):
            raise TypeError(
                "fusion must be ContextAwareFusion."
            )

        self.encoder = encoder
        self.fusion = fusion
        self.device = device

        self.encoder.to(
            self.device
        )

        self.fusion.to(
            self.device
        )

    def process(
        self,
        data,
    ):
        if (
            not isinstance(data, tuple)
            or len(data) != 2
        ):
            raise TypeError(
                "ContextIntelligenceLayer expects "
                "(MultimodalRepresentation, ContextInput)."
            )

        representation, context = data

        if not isinstance(
            representation,
            MultimodalRepresentation,
        ):
            raise TypeError(
                "First item must be "
                "MultimodalRepresentation."
            )

        if not isinstance(
            context,
            ContextInput,
        ):
            raise TypeError(
                "Second item must be ContextInput."
            )

        if (
            representation.sample_id
            != context.sample_id
        ):
            raise ValueError(
                "Representation and context "
                "sample_id values must match."
            )

        if representation.fused_embedding is None:
            raise ValueError(
                "MultimodalRepresentation must "
                "contain fused_embedding."
            )

        content_embedding = (
            representation
            .fused_embedding
        )

        if not isinstance(
            content_embedding,
            torch.Tensor,
        ):
            content_embedding = torch.tensor(
                content_embedding,
                dtype=torch.float32,
            )

        content_embedding = (
            content_embedding
            .float()
            .to(self.device)
        )

        self.encoder.eval()
        self.fusion.eval()

        with torch.no_grad():

            context_representation = (
                self.encoder(
                    context
                )
            )

            context_embedding = (
                context_representation
                .embedding
                .to(self.device)
            )

            fused = self.fusion(
                content_embedding,
                context_embedding,
            )

        fused = (
            fused
            .squeeze(0)
            .detach()
            .cpu()
        )

        context_embedding = (
            context_embedding
            .detach()
            .cpu()
        )

        return ContextualizedRepresentation(
            sample_id=(
                representation.sample_id
            ),

            content_embedding=(
                content_embedding
                .detach()
                .cpu()
            ),

            context_embedding=(
                context_embedding
            ),

            fused_embedding=(
                fused
            ),

            content_dimension=(
                representation
                .shared_dimension
            ),

            context_dimension=(
                context_representation
                .dimension
            ),

            fused_dimension=(
                fused.shape[-1]
            ),

            metadata={
                "domain": (
                    context.domain.value
                ),

                "country": (
                    context.country
                ),

                "platform": (
                    context.platform
                ),

                "language": (
                    context.language
                ),

                "event": (
                    context.event
                ),

                "context_aware": True,
            },
        )