"""
AEGIS vision representation layer.

Version: 0.8.0
"""

from aegis.pipeline import AEGISLayer
from aegis.preprocessing import CanonicalSample

from .vision_base import VisionEncoder


class VisionRepresentationLayer(
    AEGISLayer
):
    """
    Visual representation component of AEGIS.

    Converts an image-bearing CanonicalSample
    into a semantic VisionRepresentation.
    """

    layer_name = (
        "vision_representation_learning"
    )

    def __init__(
        self,
        encoder: VisionEncoder,
        config=None,
    ):

        super().__init__(
            config
        )

        if not isinstance(
            encoder,
            VisionEncoder,
        ):
            raise TypeError(
                "encoder must implement VisionEncoder."
            )

        self.encoder = encoder

    def process(
        self,
        data: CanonicalSample,
    ):

        return self.encoder.encode(
            data
        )