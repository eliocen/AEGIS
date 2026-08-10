"""
Representation outputs used by AEGIS.

Version: 0.7.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TextRepresentation:
    """
    Standard output produced by an AEGIS text encoder.

    The actual tensor is kept generic so downstream modules
    can consume representations without depending directly
    on a particular transformer architecture.
    """

    sample_id: str

    embedding: Any

    language: str

    model_name: str

    dimension: int

    text: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )