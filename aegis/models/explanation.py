from dataclasses import dataclass
from typing import List


@dataclass
class Explanation:

    summary: str

    evidence: List[str]

    recommendation: str