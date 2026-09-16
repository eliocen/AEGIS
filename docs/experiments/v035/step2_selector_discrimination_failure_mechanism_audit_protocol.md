# AEGIS v0.35 Step2 - Prospective Selector Discrimination Failure-Mechanism Audit Protocol

**Status:** AUDIT_PROTOCOL_FROZEN_NO_AUDIT_RESULTS_COMPUTED

## Purpose

Step2 prospectively freezes the read-only failure-mechanism audit contract before
the relevant v0.35 implementation is substantively inspected and before any
corrective modification or training.

The audit has two independent components:

1. **D1 - Selector Output-Path Audit**
2. **D2 - Objective-to-Output Coupling Audit**

Step2 defines what may be inspected, what evidence Step3 must produce, and how
that evidence will be adjudicated. Step2 does **not** execute the audit.

## Exact parent boundary

- Branch: `feature/v0.35-selector-discrimination-correction`
- Parent Step1 commit: `8660af5c749f0368254c73ea32ee09cce0546e46`
- Step1 JSON SHA256: `B8766F2C7C89817FE517E9F3CBDCF13E597A433DB170023DDE7C56F79FC42DF5`
- Step1 Markdown SHA256: `8E3FA8A3C2835D6E3DBC26C827C7DB716666D2971716DA7C529BC48E9AF2785C`
- v0.34 Step13 ancestor: `3bac60bb853017422bab3b5c188b410819ef0a37`

## Inherited scientific boundary

- M1 - Counterfactual target signal: **SUPPORTED**
- M2 - Selector optimization signal: **SUPPORTED**
- M3 - Selector output discrimination: **NOT_SUPPORTED**
- M4 - Intervention activation: **NOT_SUPPORTED**
- M5 - Failure localization: **SELECTOR_DISCRIMINATION_FAILURE**
- Primary v0.35 problem: **SELECTOR_OUTPUT_DISCRIMINATION**
- Official-test samples accessed: **0**

The audit must not treat absent intervention as independent proof of an
activation-mapping defect while selector discrimination remains unresolved.

## Audit mode

**READ_ONLY_SOURCE_AND_CONFIGURATION_AUDIT**

Authorized after this freeze:

- deterministic discovery of relevant selector/source/configuration files;
- hashing and manifesting those files before substantive findings;
- static tracing of selector target, objective, optimization, output and
  intervention wiring;
- production of the exact Step3 audit evidence required below.

Not authorized:

- code modification;
- training/retraining;
- checkpoint reselection;
- threshold tuning;
- architecture/loss redesign;
- formal robustness evaluation;
- official-test access.

## Admissible source scope

Step3 may inspect only repository source/configuration required to trace:

- selector module/class/function definitions;
- utility/selector output transformations;
- counterfactual target construction;
- selector objective/loss construction;
- selector optimizer parameter-group wiring;
- delta-u or equivalent utility-difference construction;
- intervention activation/mapping;
- runner wiring connecting those components;
- directly controlling configuration values;
- existing tests directly exercising those components.

Because Step2 is frozen before implementation inspection, this protocol does not
invent exact repository file paths. Step3 must first generate a deterministic
source manifest. Every substantively inspected source file must be listed with
repository-relative path, tracked status, relevant symbols, audit role and
SHA256/blob identity.

## D1 - Selector Output-Path Audit

### Frozen question

Does the implemented selector output path contain a structural or numerical
mechanism capable of explaining the inherited low utility-probability variance?

### Required ordered trace

1. Raw selector-producing representation.
2. Selector logits or pre-activation output.
3. Activation/probability mapping.
4. Temperature, scale, clamp, normalization, detach, cast or equivalent
   transformation, if present.
5. `utility_probability` or equivalent output.
6. `delta_u` or equivalent utility-difference construction.
7. Input to intervention threshold/mapping.

### Required static checks

Step3 must explicitly check for:

- tensor-shape/dimension effects;
- sigmoid/softmax/logit semantics and axes;
- fixed temperature or scaling;
- clamp/clip/normalization/centering;
- averaging or other reductions;
- detach/no-grad operations;
- dtype conversions capable of reducing resolution;
- constant/default/fallback near-neutral paths;
- batch/sample reductions that could collapse per-sample discrimination;
- reuse or overwrite of selector output before intervention mapping.

### D1 adjudication

**EXPLANATORY_DEFECT_IDENTIFIED**

A concrete source/configuration mechanism exists on the output path, is
directionally capable of compressing/collapsing sample-level selector
discrimination, and that relationship follows from code semantics without
post-correction results.

**NO_EXPLANATORY_DEFECT_IDENTIFIED**

The complete admissible output path is traced and none of the prospectively
enumerated mechanisms provides a concrete code-semantic explanation.

**NOT_ADJUDICABLE**

The complete path cannot be established from admissible frozen evidence or the
required semantics depend on unavailable runtime evidence.

D1 is explanatory localization, **not sole causal proof**.

## D2 - Objective-to-Output Coupling Audit

### Frozen question

Given inherited nonzero selector optimization signal and parameter movement,
does the implemented objective provide structurally adequate coupling from the
counterfactual target to the discriminative selector quantity consumed
downstream?

### Required ordered trace

1. Counterfactual target source.
2. Target transformation/range.
3. Selector prediction consumed by the objective.
4. Selector loss/objective expression.
5. Loss weighting/scaling.
6. Aggregation/reduction.
7. Backpropagation connection to selector parameters.
8. Selector optimizer membership.
9. Relationship between optimized prediction and downstream
   `utility_probability`.
10. Relationship between optimized prediction and intervention input.

### Required static checks

Step3 must explicitly examine:

- prediction-target tensor contract;
- target range versus selector-output range;
- loss-function semantics;
- selector loss weighting;
- mean/sum/normalization reductions;
- detach/no-grad boundaries;
- whether the optimized tensor is the same semantic quantity consumed
  downstream;
- multiple-head/branch mismatches;
- selector optimizer parameter membership;
- default/frozen paths bypassing learned selector output.

### D2 adjudication

**COUPLING_DEFECT_IDENTIFIED**

A concrete source/configuration mechanism structurally weakens, mismatches,
bypasses or semantically disconnects target/objective supervision from the
discriminative selector quantity consumed downstream.

**NO_COUPLING_DEFECT_IDENTIFIED**

The complete admissible target-to-objective-to-downstream-output path is traced
and no prospectively enumerated structural coupling defect is identified.

**NOT_ADJUDICABLE**

The complete coupling path cannot be established from admissible frozen evidence
or required semantics depend on unavailable runtime evidence.

D2 does not authorize an inference that a different loss, weight, architecture
or hyperparameter will improve results.

## Joint decision logic

D1 and D2 are adjudicated independently.

Allowed joint outcomes:

- `D1_EXPLANATORY_DEFECT__D2_COUPLING_DEFECT`
- `D1_EXPLANATORY_DEFECT__D2_NO_COUPLING_DEFECT`
- `D1_NO_EXPLANATORY_DEFECT__D2_COUPLING_DEFECT`
- `D1_NO_EXPLANATORY_DEFECT__D2_NO_COUPLING_DEFECT`
- `D1_OR_D2_NOT_ADJUDICABLE`

Multiple defects may coexist.

No corrective patch is selected during Step2 or Step3.

## Required Step3 outputs

Step3 must produce:

- `step3_selector_discrimination_source_manifest.json`
- `step3_selector_discrimination_failure_mechanism_audit_results.json`
- `step3_selector_discrimination_failure_mechanism_audit_results.md`

The manifest must identify every substantively inspected source file.

The results must contain:

- exact Step2 parent commit and protocol hashes;
- exact source-manifest hash;
- D1 ordered trace and static-check table;
- D1 adjudication;
- D2 ordered trace and static-check table;
- D2 adjudication;
- joint outcome;
- limitations;
- official-test sample count.

## Frozen prohibitions

No implementation modification, training, extra seeds, checkpoint reselection,
threshold search, discrimination-threshold weakening, architecture/loss
redesign, formal robustness evaluation or official-test access is authorized.

Post-correction outcomes may not be used to rewrite the Step3 audit finding.

The frozen v0.34 and v0.35 Step1 records remain immutable.

## Step2 outputs

- Audit protocol frozen: **YES**
- Source audit executed: **NO**
- Audit results computed: **NO**
- Implementation changed: **NO**
- Training performed: **NO**
- Formal robustness evaluation performed: **NO**
- Official-test samples accessed: **0**

## Next

**v0.35 Step3 - Read-Only Selector Discrimination Failure-Mechanism Audit Execution**

Step3 executes D1/D2 exactly under this protocol and freezes the source manifest
and audit results. It does not modify the implementation or select the
corrective patch.
