from dataclasses import dataclass
from typing import Optional

from .enums import ThreatType


@dataclass
class ThreatSample:

    sample_id: str

    text: Optional[str]

    image_path: Optional[str]

    language: str

    source: str

    timestamp: str

    label: ThreatType