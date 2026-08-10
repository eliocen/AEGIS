"""
Structured context input for AEGIS.

Version: 0.11.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .domains import OperationalDomain


@dataclass
class ContextInput:
    """
    Structured contextual information associated with
    one digital information item.
    """

    sample_id: str

    domain: OperationalDomain = OperationalDomain.GENERAL

    platform: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    event: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )