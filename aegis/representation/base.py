"""
Abstract interfaces for AEGIS representation learning.

Version: 0.7.0
"""

from abc import ABC, abstractmethod

from aegis.preprocessing import CanonicalSample

from .output import TextRepresentation


class TextEncoder(ABC):
    """
    Common interface for multilingual text encoders.
    """

    @abstractmethod
    def encode(
        self,
        sample: CanonicalSample,
    ) -> TextRepresentation:
        """
        Convert a CanonicalSample into a semantic
        text representation.
        """

        raise NotImplementedError