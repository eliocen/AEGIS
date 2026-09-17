# AEGIS Post-v1 P2 Step1 鈥?Scientific Results & Limitations Prospective Protocol

## Status

**POST_V1_P2_STEP1_RESULTS_LIMITATIONS_PROTOCOL_FROZEN**

P2 consolidates the frozen scientific record of AEGIS Research Core v1.0 into publication-ready results, limitations, claim boundaries, and a versioned scientific timeline. It is a documentation and evidence-synthesis package; it does not create new scientific evidence.

## Evidence rule

Only evidence already frozen at or before AEGIS Research Core v1.0 may determine a P2 result. P2 does not authorize model/source changes, training, new scientific evaluation, official-test access, retrospective threshold changes, checkpoint reselection, or reinterpretation of frozen hypothesis decisions.

## Result taxonomy

- **SUPPORTED** 鈥?the relevant prospectively frozen criterion was satisfied.
- **NOT_SUPPORTED** 鈥?the criterion was evaluated and not satisfied; this does not mean disproven or impossible.
- **NOT_COMPUTED** 鈥?the result was not computed because a prerequisite or authorization boundary was not satisfied.
- **CHARACTERIZED** 鈥?bounded descriptive/mechanistic evidence exists without broader effectiveness claims.
- **CONTEXT_ONLY** 鈥?relevant context, but not an evaluated AEGIS result.
- **NOT_AUTHORIZED** 鈥?outside the scientific authorization boundary.

## Required P2 outputs

1. AEGIS_V1_SCIENTIFIC_RESULTS_LEDGER.md
2. AEGIS_V1_LIMITATIONS_AND_NEGATIVE_FINDINGS.md
3. AEGIS_V1_CLAIMS_MATRIX.md
4. AEGIS_V1_EXPERIMENT_TIMELINE.md

Every material result entry must identify the version/stage, scientific question or hypothesis, prospective criterion where applicable, evidence source, dataset/population/protocol scope, observed result, status, authorized claim, prohibited overinterpretation, and limitation.

## Mandatory frozen findings

v0.34 closed with M1 and M2 **SUPPORTED**, M3 and M4 **NOT_SUPPORTED**, and M5 SELECTOR_DISCRIMINATION_FAILURE. The frozen M3 criterion is utility_probability_std >= 0.05 per seed; M4 is ctive_intervention_rate > 0 per seed.

v0.35 implemented and trained V035_C1_CLASSIFIER_POSTERIOR_CONTEXT, but the correction remained **CORRECTION_NOT_SUPPORTED**. M3 and M4 remained **NOT_SUPPORTED** for all three formal seeds. Optimization signal and selector-parameter movement were observed, while output discrimination/activation remained insufficient.

v0.36 formal active-intervention robustness is **NOT_COMPUTED**, with disposition ROBUSTNESS_EVALUATION_BLOCKED_BY_INACTIVE_SELECTOR_PREREQUISITE. This is not evidence that the classifier is non-robust.

v0.37 supports the posture **ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION**. Production capability and autonomous attribution are not established.

v0.38 passed its integration/reproducibility/research-release gates and closed the research-release preparation. AEGIS Research Core v1.0 remains the immutable release baseline, with official-test samples accessed equal to **0** and production-deployment readiness **NOT ESTABLISHED**.

## Mandatory limitations

P2 must preserve dataset/task/population/protocol scope; persistent selector-output discrimination failure; absence of qualifying active intervention; non-computation of formal active-intervention robustness; the distinction between selector failure and classifier robustness; the distinction between validation performance and formal robustness; zero official-test access; absence of production validation; absence of autonomous-attribution validation; and the future-work status of Vision, Audio, Video, multilingual expansion, and broader-system integration.

## Claim discipline

P2 may state frozen supported, unsupported, not-computed, and characterized findings exactly within their evidence scope. It must qualify validation/classifier performance by dataset and protocol. It must not promote implementation into scientific support, convert NOT_SUPPORTED into impossibility, convert NOT_COMPUTED into a negative experimental result, or describe AEGIS v1.0 as production-ready or as having demonstrated active adaptive robustness.

## P2 Step2

After this protocol is remotely frozen, Step2 may perform read-only extraction and normalization of the frozen scientific record and generate the four required P2 documents. No scientific execution is authorized.
