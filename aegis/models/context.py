from dataclasses import dataclass

from .enums import SecurityDomain


@dataclass
class ContextProfile:

    domain: SecurityDomain

    country: str

    platform: str

    language: str

    event: str