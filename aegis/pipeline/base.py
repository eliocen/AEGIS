"""
Abstract pipeline contracts for AEGIS.
"""

from abc import ABC, abstractmethod
from typing import Any


class AEGISLayer(ABC):
    """
    Base interface implemented by every processing layer in AEGIS.
    """

    layer_name = "unnamed"

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Process input data and return the layer output.
        """
        raise NotImplementedError