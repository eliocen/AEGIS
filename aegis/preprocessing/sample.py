"""
Canonical model-ready sample used by AEGIS.

Version: 0.6.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CanonicalSample:
    """
    Standardized representation produced by the AEGIS
    preprocessing layer.

    This object is independent of any specific neural encoder.
    """

    sample_id: str

    text: Optional[str] = None
    image_path: Optional[str] = None

    language: str = "und"

    source: Optional[str] = None
    platform: Optional[str] = None
    timestamp: Optional[str] = None

    label: Optional[str] = None

    has_text: bool = False
    has_image: bool = False
    is_multimodal: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    preprocessing: Dict[str, Any] = field(default_factory=dict)