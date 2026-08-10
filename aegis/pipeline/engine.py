"""
AEGIS pipeline orchestration engine.
"""

from typing import Any, Iterable

from .base import AEGISLayer


class AEGISPipeline:
    """
    Sequential orchestration engine for AEGIS processing layers.
    """

    def __init__(self, layers: Iterable[AEGISLayer] | None = None):
        self.layers = list(layers or [])

    def add_layer(self, layer: AEGISLayer) -> None:
        if not isinstance(layer, AEGISLayer):
            raise TypeError(
                "All pipeline components must inherit from AEGISLayer."
            )

        self.layers.append(layer)

    def run(self, data: Any) -> Any:
        result = data

        for layer in self.layers:
            result = layer.process(result)

        return result

    def describe(self) -> list[str]:
        return [layer.layer_name for layer in self.layers]