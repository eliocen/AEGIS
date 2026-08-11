"""
AEGIS v0.13.0
Threat Attribution & Evidence Structuring
sanity experiment.
"""

from aegis.attribution import (
    AttributionEvidence,
    EvidenceType,
    SourceProfile,
    ThreatAttributionEngine,
)

from aegis.classification import (
    CognitiveThreatType,
    IntegrityStatus,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


threat = CognitiveThreatAssessment(
    sample_id="SEC-ATTR-001",

    integrity_status=(
        IntegrityStatus.HARMFUL
    ),

    threat_type=(
        CognitiveThreatType.DISINFORMATION
    ),

    severity=(
        ThreatSeverity.CRITICAL
    ),

    risk_score=0.86,

    risk_band=(
        RiskBand.CRITICAL
    ),

    classification_confidence=0.93,

    propagation_risk=0.91,

    synthetic_content_indicator=0.79,

    coordination_indicator=0.88,

    context_risk=0.92,

    source_credibility=0.25,
)


source = SourceProfile(
    source_id="SRC-001",

    platform="X",

    account_name="observed_source",

    country="Uganda",

    credibility_score=0.25,

    account_age_days=45,

    follower_count=3200,
)


evidence = [
    AttributionEvidence(
        evidence_id="E-001",

        evidence_type=(
            EvidenceType
            .COORDINATION_SIGNAL
        ),

        description=(
            "Multiple accounts published "
            "near-identical narratives "
            "within a narrow time window."
        ),

        confidence=0.88,
    ),

    AttributionEvidence(
        evidence_id="E-002",

        evidence_type=(
            EvidenceType
            .SYNTHETIC_CONTENT
        ),

        description=(
            "Synthetic-content analysis "
            "returned a strong indicator."
        ),

        confidence=0.81,
    ),

    AttributionEvidence(
        evidence_id="E-003",

        evidence_type=(
            EvidenceType
            .SOURCE_PROVENANCE
        ),

        description=(
            "Observed source provenance "
            "shows repeated dissemination "
            "of related narratives."
        ),

        confidence=0.72,
    ),
]


engine = (
    ThreatAttributionEngine()
)


assessment = engine.assess(
    threat_assessment=threat,
    source_profile=source,
    evidence=evidence,
)


print(
    "Sample:",
    assessment.sample_id,
)

print(
    "Evidence count:",
    len(
        assessment.evidence
    ),
)

print(
    "Overall evidence confidence:",
    round(
        assessment.overall_confidence,
        4,
    ),
)

print(
    "Primary hypothesis:",
    assessment
    .primary_hypothesis
    .category
    .value,
)

print(
    "Primary confidence:",
    round(
        assessment
        .primary_hypothesis
        .confidence,
        4,
    ),
)

print(
    "Definitive attribution:",
    assessment.metadata[
        "definitive_attribution"
    ],
)

print(
    "\nHypotheses:"
)

for hypothesis in assessment.hypotheses:

    print(
        "-",
        hypothesis.category.value,
        "| confidence:",
        round(
            hypothesis.confidence,
            4,
        ),
    )

    print(
        "  rationale:",
        hypothesis.rationale,
    )