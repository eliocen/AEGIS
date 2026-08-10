"""
Parallel multimodal representation orchestration for AEGIS.

Version: 0.9.0
"""

from dataclasses import dataclass
from typing import Optional

from aegis.preprocessing import CanonicalSample

from .base import TextEncoder
from .vision_base import VisionEncoder
from .output import TextRepresentation
from .vision_output import VisionRepresentation


@dataclass
class ParallelRepresentationBundle:
    """
    Holds modality-specific representations generated
    from the same CanonicalSample.
    """

    sample_id: str

    text: Optional[TextRepresentation] = None
    vision: Optional[VisionRepresentation] = None

    text_available: bool = False
    vision_available: bool = False

    @property
    def is_multimodal(self) -> bool:
        return (
            self.text_available
            and self.vision_available
        )


class ParallelMultimodalEncoder:
    """
    Runs text and vision encoders independently over
    the same CanonicalSample.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        vision_encoder: VisionEncoder,
    ):
        if not isinstance(
            text_encoder,
            TextEncoder,
        ):
            raise TypeError(
                "text_encoder must implement TextEncoder."
            )

        if not isinstance(
            vision_encoder,
            VisionEncoder,
        ):
            raise TypeError(
                "vision_encoder must implement VisionEncoder."
            )

        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder

    def encode(
        self,
        sample: CanonicalSample,
    ) -> ParallelRepresentationBundle:

        if not isinstance(
            sample,
            CanonicalSample,
        ):
            raise TypeError(
                "ParallelMultimodalEncoder expects "
                "a CanonicalSample."
            )

        text_representation = None
        vision_representation = None

        if sample.has_text:
            text_representation = (
                self.text_encoder.encode(
                    sample
                )
            )

        if sample.has_image:
            vision_representation = (
                self.vision_encoder.encode(
                    sample
                )
            )

        return ParallelRepresentationBundle(
            sample_id=sample.sample_id,

            text=text_representation,
            vision=vision_representation,

            text_available=(
                text_representation is not None
            ),

            vision_available=(
                vision_representation is not None
            ),
        )