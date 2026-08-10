"""
Abstract interfaces for AEGIS visual representation learning.

Version: 0.8.0
"""

from abc import ABC, abstractmethod

from aegis.preprocessing import CanonicalSample

from .vision_output import VisionRepresentation


class VisionEncoder(ABC):
    """
    Common interface implemented by all AEGIS vision encoders.
    """

    @abstractmethod
    def encode(
        self,
        sample: CanonicalSample,
    ) -> VisionRepresentation:
        """
        Convert a CanonicalSample containing an image
        into a semantic visual representation.
        """

        raise NotImplementedError