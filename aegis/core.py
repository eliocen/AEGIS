"""
AEGIS Core Framework

Version: 0.4.0
"""

from typing import Any, Dict

from aegis.config import AEGISConfig
from aegis.pipeline import AEGISPipeline
from aegis.utils import get_logger


class AEGIS:
    """
    Central orchestrator for the AEGIS framework.
    """

    def __init__(
        self,
        config: AEGISConfig | None = None,
        pipeline: AEGISPipeline | None = None,
    ):
        self.config = config or AEGISConfig()
        self.pipeline = pipeline or AEGISPipeline()

        self.logger = get_logger("aegis.core")

    def info(self) -> Dict[str, Any]:
        return {
            "framework": self.config.get("framework.name"),
            "version": self.config.get("framework.version"),
            "device": self.config.get("framework.device"),
            "research_area": (
                "AI for Information Integrity and "
                "Cognitive Cyber Threat Intelligence"
            ),
            "pipeline_layers": self.pipeline.describe(),
        }

    def add_layer(self, layer) -> None:
        self.pipeline.add_layer(layer)

        self.logger.info(
            "Registered AEGIS layer: %s",
            layer.layer_name,
        )

    def analyze(self, sample: Any) -> Any:
        self.logger.info("Starting AEGIS analysis pipeline.")

        result = self.pipeline.run(sample)

        self.logger.info("AEGIS analysis pipeline completed.")

        return result