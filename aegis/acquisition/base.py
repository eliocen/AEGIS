"""
Abstract interfaces for AEGIS data acquisition.

Version: 0.5.0
"""

from abc import ABC, abstractmethod
from typing import Iterable

from .record import AcquisitionRecord


class DataSource(ABC):
    """
    Abstract base class for all AEGIS data sources.

    Every dataset adapter must convert source-specific data
    into AcquisitionRecord objects.
    """

    @abstractmethod
    def load(self) -> Iterable[AcquisitionRecord]:
        """
        Load and normalize records from the source.
        """
        raise NotImplementedError