"""
AEGIS Context Intelligence domain taxonomy.

Version: 0.11.0
"""

from enum import Enum


class OperationalDomain(Enum):
    """
    Operational domains supported by the
    AEGIS Context Intelligence Layer.
    """

    GENERAL = "general"

    CYBERSECURITY = "cybersecurity"
    CONFLICT = "conflict"
    MILITARY = "military"
    PEACEKEEPING = "peacekeeping"

    ELECTION = "election"
    POLITICS = "politics"

    DIPLOMACY = "diplomacy"
    INTERNATIONAL_SECURITY = "international_security"

    PUBLIC_HEALTH = "public_health"
    HUMANITARIAN = "humanitarian"

    SOCIAL_MEDIA = "social_media"
    NEWS_MEDIA = "news_media"