"""
Inputs for AEGIS Explainable AI.

Version: 0.14.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from aegis.attribution import (
    ThreatAttributionAssessment,
)

from aegis.classification import (
    HierarchicalClassificationOutput,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
)


@dataclass
class ExplainabilityInput:
    """
    Consolidates AEGIS analytical outputs for
    structured explanation generation.
    """

    classification: HierarchicalClassificationOutput

    threat_assessment: Optional[
        CognitiveThreatAssessment
    ] = None

    attribution: Optional[
            ThreatAttributionAssessment
        ] = None

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    modality_signals: Dict[str, float] = field(
        default_factory=dict
    )