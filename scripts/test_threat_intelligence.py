"""
AEGIS v0.12.0
Cognitive Threat Intelligence sanity experiment.
"""

from aegis.classification import (
    CognitiveThreatType,
    HierarchicalClassificationOutput,
    IntegrityStatus,
)

from aegis.intelligence import (
    CognitiveThreatIntelligenceEngine,
    ThreatIntelligenceInput,
)


classification = (
    HierarchicalClassificationOutput(
        sample_id="SEC-001",
        integrity_status=(
            IntegrityStatus.HARMFUL
        ),
        integrity_confidence=0.94,
        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),
        threat_confidence=0.91,
    )
)


assessment_input = (
    ThreatIntelligenceInput(
        classification=classification,
        propagation_score=0.88,
        synthetic_content_score=0.72,
        coordination_score=0.81,
        context_risk_score=0.92,
        source_credibility_score=0.25,
    )
)


engine = (
    CognitiveThreatIntelligenceEngine()
)


assessment = engine.assess(
    assessment_input
)


print(
    "Sample:",
    assessment.sample_id,
)

print(
    "Integrity:",
    assessment.integrity_status.value,
)

print(
    "Threat:",
    assessment.threat_type.value,
)

print(
    "Risk score:",
    round(
        assessment.risk_score,
        4,
    ),
)

print(
    "Risk band:",
    assessment.risk_band.value,
)

print(
    "Severity:",
    assessment.severity.value,
)

print(
    "Propagation:",
    assessment.propagation_risk,
)

print(
    "Synthetic indicator:",
    assessment.synthetic_content_indicator,
)

print(
    "Coordination:",
    assessment.coordination_indicator,
)

print(
    "Context risk:",
    assessment.context_risk,
)