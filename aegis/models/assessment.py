from dataclasses import dataclass
from typing import List

from .enums import ThreatType
from .enums import ThreatLevel


@dataclass
class ThreatAssessment:

    predicted_class: ThreatType

    confidence: float

    severity: ThreatLevel

    explanation: str

    evidence: List[str]