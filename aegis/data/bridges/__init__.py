"""
AEGIS empirical-to-model representation bridges.
"""

from .fakeddit import (
    FakedditRepresentedSample,
    FakedditRepresentationBridge,
    empirical_to_canonical,
)


__all__ = [
    "FakedditRepresentedSample",
    "FakedditRepresentationBridge",
    "empirical_to_canonical",
]