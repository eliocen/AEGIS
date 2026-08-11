"""
AEGIS Source Provenance Models.

Version: 0.13.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SourceProfile:
    """
    Structured description of the observed source
    associated with an information item.
    """

    source_id: str

    platform: Optional[str] = None
    account_name: Optional[str] = None
    domain: Optional[str] = None
    country: Optional[str] = None

    credibility_score: Optional[float] = None
    account_age_days: Optional[int] = None
    follower_count: Optional[int] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )