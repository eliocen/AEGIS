"""
Canonical acquisition record for AEGIS.

Version: 0.5.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AcquisitionRecord:
    """
    Standardized representation of one acquired information item.

    All external datasets should be transformed into this format
    before entering the AEGIS preprocessing layer.
    """

    sample_id: str

    text: Optional[str] = None
    image_path: Optional[str] = None

    language: Optional[str] = None
    source: Optional[str] = None
    platform: Optional[str] = None
    timestamp: Optional[str] = None

    label: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    def has_image(self) -> bool:
        return bool(self.image_path)

    def is_multimodal(self) -> bool:
        return self.has_text() and self.has_image()