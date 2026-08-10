"""
AEGIS multimodal alignment outputs.

Version: 0.9.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MultimodalRepresentation:
    """
    Unified representation produced after cross-modal
    semantic alignment and multimodal fusion.
    """

    sample_id: str

    text_embedding: Optional[Any] = None
    vision_embedding: Optional[Any] = None

    aligned_text: Optional[Any] = None
    aligned_vision: Optional[Any] = None

    fused_embedding: Optional[Any] = None

    shared_dimension: int = 0

    text_available: bool = False
    vision_available: bool = False
    is_multimodal: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )