# AEGIS v0.35 Step1 - Prospective Selector-Discrimination & Intervention-Activation Corrective Protocol Freeze

**Status:** PROTOCOL_FROZEN_NO_V035_RESULTS_COMPUTED

## Purpose

AEGIS v0.35 begins from the exact remotely frozen v0.34 Step13 scientific
boundary. This step freezes the prospective scientific contract for addressing
the inherited selector-discrimination failure before any v0.35 corrective
implementation, training, formal robustness evaluation, or official-test
access.

## Exact parent boundary

- Parent version: **v0.34 Step13**
- Parent branch: `feature/v0.34-utility-discrimination-failure-resolution`
- Parent commit: `3bac60bb853017422bab3b5c188b410819ef0a37`
- Step13 JSON SHA256: `74A104B89180A190F26EEFC82E52F514FC2420508FF4B8267B8381766D9EDF82`
- Step13 Markdown SHA256: `0094FF6CC06B0F688FBF5E1592158D8CEC24E9E97648E7B0D07C1A97EECDD11B`
- Inherited formal-training evidence-tree SHA256: `8D035BB46FC909A1AA21A277160C6CEE2F456BB3EDF3D38401F953B0A709BE45`

## Inherited frozen finding

- M1 - Counterfactual target signal: **SUPPORTED**
- M2 - Selector optimization signal: **SUPPORTED**
- M3 - Selector output discrimination: **NOT_SUPPORTED**
- M4 - Intervention activation: **NOT_SUPPORTED**
- M5 - Failure localization: **SELECTOR_DISCRIMINATION_FAILURE**

The inherited evidence establishes material target signal and nonzero selector
optimization but does not establish meaningful selector-output discrimination.
Intervention activation was absent.

Under the frozen v0.34 localization ordering, the primary localized failure is
selector discrimination. Because M3 failed before M4 could be independently
localized, v0.34 does **not** establish `ACTIVATION_MAPPING_FAILURE` as a
separate causal finding.

## v0.35 primary corrective objective

The primary v0.35 problem is:

`SELECTOR_OUTPUT_DISCRIMINATION`

The primary objective is to identify and implement the **minimum scientifically
justified correction** needed for the selector to produce meaningfully
discriminative utility outputs.

The downstream question is whether, after discrimination is restored, the
existing activation mapping produces nonzero intervention.

## Prospective causal investigation order

### D1 - Selector output-path audit

Determine whether the implemented selector output path contains a structural or
numerical compression mechanism capable of explaining the frozen low
utility-probability variance.

Allowed result space:

- `EXPLANATORY_DEFECT_IDENTIFIED`
- `NO_EXPLANATORY_DEFECT_IDENTIFIED`
- `NOT_ADJUDICABLE`

### D2 - Objective-to-output coupling audit

Given the inherited nonzero selector optimization signal, determine whether the
selector objective provides sufficient coupling to move utility outputs away
from near-neutral compression.

Allowed result space:

- `COUPLING_DEFECT_IDENTIFIED`
- `NO_COUPLING_DEFECT_IDENTIFIED`
- `NOT_ADJUDICABLE`

### D3 - Minimal corrective design

Any correction must be causally tied to the prospectively audited D1/D2
mechanism. Broad redesign is not authorized by Step1.

The exact correction must be frozen before corrective training.

### D4 - Post-correction discrimination adjudication

After a later step prospectively authorizes corrective training, selector-output
discrimination must be adjudicated under criteria frozen **before** that
training.

### D5 - Downstream activation adjudication

Only after D4 supports restored discrimination may downstream intervention
activation be interpreted independently.

If discrimination is restored but activation still fails, a subsequent
prospective step may investigate activation mapping as a separately localized
mechanism.

## Prospective success boundary

The inherited v0.34 M3 reference criterion was:

`utility_probability_std >= 0.05` for every formal seed.

Step1 preserves `0.05` as the default discrimination reference and explicitly
prohibits weakening that threshold merely because v0.34 failed it.

The inherited v0.34 M4 reference criterion was:

`active_intervention_rate > 0` for every formal seed.

Before any corrective training, a later v0.35 protocol step must freeze the
exact training design, formal seeds, observability artifacts, and success
criteria.

## Evidence and access boundary

- Frozen v0.34 evidence: **READ ONLY**
- v0.35 read-only pre-training audit: **AUTHORIZED AFTER STEP1**
- v0.35 corrective implementation: **NOT AUTHORIZED BY STEP1**
- v0.35 training: **NOT AUTHORIZED BY STEP1**
- Formal robustness evaluation: **NOT AUTHORIZED**
- Official test access: **PROHIBITED**
- Official test samples accessed: **0**
- Checkpoint reselection: **PROHIBITED**
- Threshold search/tuning: **PROHIBITED**
- Post-hoc success-criterion changes: **PROHIBITED**

## Version integrity

The complete v0.34 record remains immutable.

All corrective work must occur only on the v0.35 branch. Step1 does not
authorize broad architecture/loss redesign, new training seeds, or new
corruption/quality conditions.

## Step1 outputs

This step freezes protocol only.

- v0.35 results computed: **NO**
- Implementation changed: **NO**
- Training performed: **NO**
- Formal robustness evaluation performed: **NO**
- Official test samples accessed: **0**

## Next

**v0.35 Step2 - Prospective Selector Discrimination Failure-Mechanism Audit Protocol**

Step2 must freeze the exact read-only code/configuration audit questions,
admissible source files, evidence outputs, and D1/D2 decision rules **before**
the selector implementation is inspected or modified.

Step2 does not authorize implementation changes, training, formal evaluation,
or official-test access.
