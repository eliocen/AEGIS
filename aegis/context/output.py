"""
AEGIS Context Intelligence outputs.

Version: 0.11.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ContextRepresentation:
    """
    Learned context representation.
    """

    sample_id: str

    embedding: Any

    dimension: int

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextualizedRepresentation:
    """
    Multimodal representation enriched by
    operational context.
    """

    sample_id: str

    content_embedding: Any
    context_embedding: Any
    fused_embedding: Any

    content_dimension: int
    context_dimension: int
    fused_dimension: int

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )