from dataclasses import dataclass


@dataclass
class Evidence:

    source: str

    confidence: float

    description: str