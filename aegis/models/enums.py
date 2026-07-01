from enum import Enum


class ThreatType(Enum):
    TRUE = "True Information"

    MISINFORMATION = "Misinformation"

    DISINFORMATION = "Disinformation"

    MALINFORMATION = "Malinformation"

    HATE_SPEECH = "Hate Speech"

class ThreatLevel(Enum):

    NONE = "None"

    LOW = "Low"

    MEDIUM = "Medium"

    HIGH = "High"

    CRITICAL = "Critical"

class SecurityDomain(Enum):

    SOCIAL_MEDIA = "Social Media"

    NEWS_MEDIA = "News Media"

    PEACEKEEPING = "Peacekeeping"

    DIPLOMACY = "Diplomacy"

    ELECTION = "Election"

    CYBERSPACE = "Cyberspace"

    PUBLIC_HEALTH = "Public Health"

    MILITARY = "Military"