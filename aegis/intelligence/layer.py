"""
AEGIS Cognitive Threat Intelligence pipeline layer.

Version: 0.12.0
"""

from aegis.pipeline import (
    AEGISLayer,
)

from .engine import (
    CognitiveThreatIntelligenceEngine,
)

from .input import (
    ThreatIntelligenceInput,
)


class CognitiveThreatIntelligenceLayer(
    AEGISLayer
):
    """
    Operational threat assessment layer.
    """

    layer_name = (
        "cognitive_threat_intelligence"
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
            or CognitiveThreatIntelligenceEngine()
        )

    def process(
        self,
        data,
    ):
        if not isinstance(
            data,
            ThreatIntelligenceInput,
        ):
            raise TypeError(
                "CognitiveThreatIntelligenceLayer "
                "expects ThreatIntelligenceInput."
            )

        return self.engine.assess(
            data
        )