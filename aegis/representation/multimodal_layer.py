"""
AEGIS parallel multimodal representation layer.

Version: 0.9.0
"""

from aegis.pipeline import AEGISLayer
from aegis.preprocessing import CanonicalSample

from .multimodal import (
    ParallelMultimodalEncoder,
)


class ParallelMultimodalRepresentationLayer(
    AEGISLayer
):
    """
    Runs textual and visual representation learning
    in parallel for the same CanonicalSample.
    """

    layer_name = (
        "parallel_multimodal_representation"
    )

    def __init__(
        self,
        encoder: ParallelMultimodalEncoder,
        config=None,
    ):
        super().__init__(config)

        if not isinstance(
            encoder,
            ParallelMultimodalEncoder,
        ):
            raise TypeError(
                "encoder must be a "
                "ParallelMultimodalEncoder."
            )

        self.encoder = encoder

    def process(
        self,
        data: CanonicalSample,
    ):
        return self.encoder.encode(data)