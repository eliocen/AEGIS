"""
Visual representation outputs used by AEGIS.

Version: 0.8.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VisionRepresentation:
    """
    Standard output produced by an AEGIS vision encoder.

    The object deliberately hides model-specific output
    structures from downstream AEGIS components.
    """

    sample_id: str

    embedding: Any

    model_name: str

    dimension: int

    image_path: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )