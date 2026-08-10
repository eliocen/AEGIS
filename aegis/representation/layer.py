"""
AEGIS multilingual representation layer.

Version: 0.7.0
"""

from aegis.pipeline import AEGISLayer
from aegis.preprocessing import (
    CanonicalSample,
)

from .base import TextEncoder


class MultilingualRepresentationLayer(
    AEGISLayer
):
    """
    Layer 3 of the AEGIS architecture.

    Converts CanonicalSample objects into
    multilingual semantic representations.
    """

    layer_name = (
        "multilingual_representation_learning"
    )

    def __init__(
        self,
        encoder: TextEncoder,
        config=None,
    ):
        super().__init__(config)

        if not isinstance(
            encoder,
            TextEncoder,
        ):
            raise TypeError(
                "encoder must implement TextEncoder."
            )

        self.encoder = encoder

    def process(
        self,
        data: CanonicalSample,
    ):
        return self.encoder.encode(
            data
        )