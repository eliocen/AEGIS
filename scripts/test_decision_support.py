"""
AEGIS v0.15.0
Decision Support & Response Layer
sanity experiment.
"""

from aegis.attribution import (
    AttributionCategory,
    AttributionHypothesis,
    ThreatAttributionAssessment,
)

from aegis.classification import (
    CognitiveThreatType,
    IntegrityStatus,
)

from aegis.decision import (
    DecisionSupportEngine,
    DecisionSupportInput,
)

from aegis.explainability import (
    ExplanationReport,
)

from aegis.intelligence import (
    CognitiveThreatAssessment,
    RiskBand,
    ThreatSeverity,
)


threat = CognitiveThreatAssessment(
    sample_id="DECISION-001",

    integrity_status=(
        IntegrityStatus.HARMFUL
    ),

    threat_type=(
        CognitiveThreatType.DISINFORMATION
    ),

    severity=(
        ThreatSeverity.CRITICAL
    ),

    risk_score=0.89,

    risk_band=(
        RiskBand.CRITICAL
    ),

    classification_confidence=0.93,

    propagation_risk=0.92,

    synthetic_content_indicator=0.78,

    coordination_indicator=0.88,

    context_risk=0.95,

    source_credibility=0.20,
)


explanation = ExplanationReport(
    sample_id="DECISION-001",

    summary=(
        "AEGIS assessed the sample as harmful "
        "and classified it as disinformation."
    ),

    factors=[],

    explanation_confidence=0.91,

    uncertainty=0.09,

    evidence_ids=[
        "E-001",
        "E-002",
    ],

    caveats=[
        (
            "Attribution findings remain "
            "hypothesis-based."
        )
    ],

    reasoning_trace=[
        (
            "Stage 1 classified the sample "
            "as harmful."
        ),

        (
            "Stage 2 classified the sample "
            "as disinformation."
        ),
    ],
)


hypothesis = AttributionHypothesis(
    category=(
        AttributionCategory
        .COORDINATED_NETWORK
    ),

    confidence=0.84,

    rationale=(
        "Observed coordination indicators "
        "support a coordinated-network "
        "hypothesis."
    ),
)


attribution = ThreatAttributionAssessment(
    sample_id="DECISION-001",

    source_profile=None,

    evidence=[],

    hypotheses=[
        hypothesis
    ],

    overall_confidence=0.84,

    primary_hypothesis=(
        hypothesis
    ),

    metadata={
        "definitive_attribution": False
    },
)


data = DecisionSupportInput(
    threat_assessment=threat,

    explanation=explanation,

    attribution=attribution,
)


engine = (
    DecisionSupportEngine()
)


recommendation = engine.recommend(
    data
)


print(
    "Sample:",
    recommendation.sample_id,
)

print(
    "Priority:",
    recommendation.priority.value,
)

print(
    "Decision confidence:",
    round(
        recommendation.confidence,
        4,
    ),
)

print(
    "Human review required:",
    recommendation.human_review_required,
)

print(
    "Automation permitted:",
    recommendation.automation_permitted,
)


print(
    "\nRecommended actions:"
)

for action in recommendation.actions:

    print(
        "-",
        action.value
    )


print(
    "\nRationale:"
)

for reason in recommendation.rationale:

    print(
        "-",
        reason
    )


print(
    "\nDecision mode:",
    recommendation.metadata[
        "decision_mode"
    ],
)

print(
    "Autonomous enforcement:",
    recommendation.metadata[
        "autonomous_enforcement"
    ],
)