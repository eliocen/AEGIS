"""
AEGIS Explainability Pipeline Layer.

Version: 0.14.0
"""

from aegis.pipeline import (
    AEGISLayer,
)

from .engine import (
    ExplainabilityEngine,
)

from .input import (
    ExplainabilityInput,
)


class ExplainabilityLayer(
    AEGISLayer
):
    """
    Converts AEGIS analytical outputs into
    structured explanations.
    """

    layer_name = (
        "explainable_ai_reasoning"
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
            or ExplainabilityEngine()
        )

    def process(
        self,
        data,
    ):
        if not isinstance(
            data,
            ExplainabilityInput,
        ):
            raise TypeError(
                "ExplainabilityLayer expects "
                "ExplainabilityInput."
            )

        return self.engine.explain(
            data
        )