"""
AEGIS v0.14.0
Explainable AI & Evidence-Based Reasoning
sanity experiment.
"""

from aegis.attribution import (
    AttributionCategory,
    AttributionEvidence,
    AttributionHypothesis,
    EvidenceType,
    ThreatAttributionAssessment,
)

from aegis.classification import (
    CognitiveThreatType,
    HierarchicalClassificationOutput,
    IntegrityStatus,
)

from aegis.explainability import (
    ExplainabilityEngine,
    ExplainabilityInput,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


classification = (
    HierarchicalClassificationOutput(
        sample_id="XAI-001",

        integrity_status=(
            IntegrityStatus.HARMFUL
        ),

        integrity_confidence=0.95,

        threat_type=(
            CognitiveThreatType.DISINFORMATION
        ),

        threat_confidence=0.92,
    )
)


threat = CognitiveThreatAssessment(
    sample_id="XAI-001",

    integrity_status=(
        IntegrityStatus.HARMFUL
    ),

    threat_type=(
        CognitiveThreatType.DISINFORMATION
    ),

    severity=(
        ThreatSeverity.CRITICAL
    ),

    risk_score=0.87,

    risk_band=(
        RiskBand.CRITICAL
    ),

    classification_confidence=0.92,

    propagation_risk=0.91,

    synthetic_content_indicator=0.78,

    coordination_indicator=0.89,

    context_risk=0.94,

    source_credibility=0.22,
)


evidence = [
    AttributionEvidence(
        evidence_id="XAI-E1",

        evidence_type=(
            EvidenceType
            .COORDINATION_SIGNAL
        ),

        description=(
            "Coordinated dissemination "
            "pattern observed."
        ),

        confidence=0.90,
    ),

    AttributionEvidence(
        evidence_id="XAI-E2",

        evidence_type=(
            EvidenceType
            .SYNTHETIC_CONTENT
        ),

        description=(
            "Synthetic-content indicator "
            "was detected."
        ),

        confidence=0.82,
    ),
]


hypothesis = AttributionHypothesis(
    category=(
        AttributionCategory
        .COORDINATED_NETWORK
    ),

    confidence=0.90,

    rationale=(
        "Observed coordination indicators "
        "support a coordinated-network "
        "hypothesis."
    ),

    supporting_evidence_ids=[
        "XAI-E1"
    ],
)


attribution = ThreatAttributionAssessment(
    sample_id="XAI-001",

    source_profile=None,

    evidence=evidence,

    hypotheses=[
        hypothesis
    ],

    overall_confidence=0.86,

    primary_hypothesis=(
        hypothesis
    ),

    metadata={
        "definitive_attribution": False
    },
)


data = ExplainabilityInput(
    classification=classification,

    threat_assessment=threat,

    attribution=attribution,

    context={
        "domain": "international_security",
        "country": "Uganda",
        "platform": "X",
    },
)


engine = ExplainabilityEngine()


report = engine.explain(
    data
)


print(
    "Sample:",
    report.sample_id,
)

print(
    "\nSummary:"
)

print(
    report.summary
)

print(
    "\nExplanation confidence:",
    round(
        report.explanation_confidence,
        4,
    ),
)

print(
    "Uncertainty:",
    round(
        report.uncertainty,
        4,
    ),
)

print(
    "\nFactors:"
)

for factor in report.factors:

    print(
        "-",
        factor.factor_id,
        "|",
        factor.category.value,
        "| strength:",
        round(
            factor.strength,
            4,
        ),
    )

    print(
        " ",
        factor.description,
    )


print(
    "\nEvidence IDs:",
    report.evidence_ids,
)


print(
    "\nReasoning trace:"
)

for step in report.reasoning_trace:

    print(
        "-",
        step
    )


print(
    "\nCaveats:"
)

for caveat in report.caveats:

    print(
        "-",
        caveat
    )