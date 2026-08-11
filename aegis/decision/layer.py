"""
AEGIS Decision Support Pipeline Layer.

Version: 0.15.0
"""

from aegis.pipeline import (
    AEGISLayer,
)

from .engine import (
    DecisionSupportEngine,
)

from .input import (
    DecisionSupportInput,
)


class DecisionSupportLayer(
    AEGISLayer
):
    """
    Final analyst-facing decision support layer
    in the AEGIS processing architecture.
    """

    layer_name = (
        "decision_support_and_response"
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
            or DecisionSupportEngine()
        )

    def process(
        self,
        data,
    ):
        if not isinstance(
            data,
            DecisionSupportInput,
        ):
            raise TypeError(
                "DecisionSupportLayer expects "
                "DecisionSupportInput."
            )

        return self.engine.recommend(
            data
        )