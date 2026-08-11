"""
AEGIS Threat Attribution Pipeline Layer.

Version: 0.13.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

from aegis.intelligence import (
    CognitiveThreatAssessment,
)

from aegis.pipeline import (
    AEGISLayer,
)

from .engine import (
    ThreatAttributionEngine,
)

from .evidence import (
    AttributionEvidence,
)

from .source import (
    SourceProfile,
)


@dataclass
class AttributionInput:
    """
    Pipeline input for threat attribution.
    """

    threat_assessment: CognitiveThreatAssessment

    source_profile: Optional[SourceProfile] = None

    evidence: List[AttributionEvidence] = field(
        default_factory=list
    )


class ThreatAttributionLayer(
    AEGISLayer
):
    """
    AEGIS evidence-based threat attribution layer.
    """

    layer_name = (
        "threat_attribution"
    )

    def __init__(
        self,
        engine=None,
        config=None,
    ):
        super().__init__(
            config
        )

        self.engine = (
            engine
            or ThreatAttributionEngine()
        )

    def process(
        self,
        data,
    ):
        if not isinstance(
            data,
            AttributionInput,
        ):
            raise TypeError(
                "ThreatAttributionLayer expects "
                "AttributionInput."
            )

        return self.engine.assess(
            threat_assessment=(
                data.threat_assessment
            ),

            source_profile=(
                data.source_profile
            ),

            evidence=(
                data.evidence
            ),
        )