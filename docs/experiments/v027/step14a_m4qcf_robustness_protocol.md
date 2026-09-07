# AEGIS v0.27 Step14A - M4qcf Formal Robustness Diagnostic Protocol

**Protocol version:** `0.27.0-step14a`
**Parent protocol:** `0.27.0-step13a`
**Status:** `FROZEN BEFORE STEP14A FORMAL DIAGNOSTIC RESULT EXPOSURE`

## 1. Purpose

Step14A is an evaluation-only diagnostic phase for the frozen M1b, M4qc,
and M4qcf checkpoints.

It generates the deterministic evidence required for the preregistered
Step13A hypotheses H8-F, H9-F, and H10-F.

Step14A does not make formal hypothesis decisions. H8-F, H9-F, and H10-F
remain NOT_COMPUTED until Step14B.

No training, checkpoint reselection, hyperparameter tuning, threshold tuning,
or official Fakeddit test access is permitted.

## 2. Frozen models

Formal seeds:

- 42
- 43
- 44

Architectures:

- M1b: `gated_interaction`
- M4qc: `quality_compatibility_supervised`
- M4qcf: `quality_compatibility_fusion`

Frozen checkpoint roots:

- `experiments/fakeddit/v025_m1b_gated_interaction_seed{seed}/best_model.pt`
- `experiments/fakeddit/v027_m4qc_seed{seed}/best_model.pt`
- `experiments/fakeddit/v027_m4qcf_seed{seed}/best_model.pt`

Each model must be reconstructed from checkpoint configuration and loaded
strictly from its saved alignment and classification state dictionaries.

## 3. Evaluation data

Evaluation uses only the frozen Fakeddit validation cache.

The official Fakeddit test split remains sealed.

Expected formal validation size:

`1000`

Official test samples accessed:

`0`

## 4. Quality-corruption matrix

The quality-corruption implementation must reuse the already validated v0.27
corruption machinery.

Gaussian feature scales must be derived from the frozen training cache only.

Validation corruptions must be deterministic and independent of evaluated
architecture and model seed.

A corruption realization for a given condition must therefore be reused
identically across M1b, M4qc, and M4qcf and across formal seeds.

Formal conditions:

### Clean

- clean

### Gaussian noise

Text corruption at severity:

- 0.25
- 0.50
- 0.75
- 1.00

Vision corruption at severity:

- 0.25
- 0.50
- 0.75
- 1.00

### Attenuation

Text corruption at severity:

- 0.25
- 0.50
- 0.75
- 1.00

Vision corruption at severity:

- 0.25
- 0.50
- 0.75
- 1.00

### Zero-dropout

- text, severity 1.00
- vision, severity 1.00

Total quality conditions per architecture/seed:

`19`

Total formal quality condition-level evaluations:

`19 x 3 architectures x 3 seeds = 171`

No simultaneous bimodal quality corruption is introduced.

## 5. Classification pathway

For every evaluated condition:

`text/image embeddings -> frozen alignment/fusion model -> fused_embedding
-> frozen HierarchicalInformationIntegrityClassifier -> Stage-1 prediction`

No alternate diagnostic classifier is permitted.

Condition-level classification metrics must be computed from the complete
1000-sample frozen validation set.

At minimum record:

- accuracy
- precision
- recall
- F1
- Macro-F1
- classification loss
- sample count

Per-sample predictions and targets must also be retained in machine-readable
diagnostic artifacts.

## 6. M4qcf reliability diagnostics

For M4qcf, Step14A must record the reliability quantities emitted by the
frozen model rather than recomputing an alternative controller externally.

Where applicable record:

- text quality `qT`
- vision quality `qV`
- compatibility `cTV`
- effective text reliability `rT`
- effective vision reliability `rV`
- text weight `alphaT`
- vision weight `alphaV`
- interaction multiplier

The frozen Step13A controller remains:

`rT = qT * [epsilon + (1-epsilon)cTV]`

`rV = qV * [epsilon + (1-epsilon)cTV]`

with:

`epsilon = 0.10`

and normalized evidence weights:

`alphaT = (rT + epsilon) / (rT + rV + 2epsilon)`

`alphaV = (rV + epsilon) / (rT + rV + 2epsilon)`

The interaction multiplier remains:

`gamma * cTV`

with:

`gamma = 1.0`

Step14A must not modify these equations.

## 7. H8-F evidence

For M4qcf single-modality corruption conditions, retain the clean and
corrupted modality weights needed to later compute:

Text corruption:

`D_T = alphaT_clean - alphaT_corrupt`

Vision corruption:

`D_V = alphaV_clean - alphaV_corrupt`

H8-F is NOT evaluated in Step14A.

The Step13A frozen Step14B decision rule remains:

- at least 80% of eligible single-modality corruption conditions must have
  positive corrupted-modality down-weighting; and
- aggregate mean down-weighting must be greater than zero.

## 8. H9-F evidence

For every non-clean single-modality quality condition, retain the complete
classification metrics for M1b and M4qcf.

M4qc must also be evaluated as the causal architectural control.

H9-F is NOT evaluated in Step14A.

The Step13A frozen Step14B comparison remains:

`R_i = MacroF1_M4qcf,i - MacroF1_M1b,i`

with support requiring:

- mean `R_i >= 0.01`; and
- at least 70% of eligible conditions have `R_i > 0`.

All eligible conditions and seeds receive equal weight.

## 9. Class-preserving mismatch track

Mismatch diagnostics must reuse the frozen Step11A/Step12B deterministic
class-preserving validation derangement.

The mapping must:

- contain no self-pairs;
- preserve the Stage-1 class;
- use every vision representation exactly once within its class;
- be identical across model seeds;
- be independent of training RNG.

Matched and mismatched validation conditions must be evaluated for:

- M1b
- M4qc
- M4qcf

The mismatch track must not use the older unrestricted permutation corruption
as a substitute for the frozen class-preserving mismatch protocol.

## 10. H10-F evidence

For M4qcf retain the matched and mismatched interaction multipliers required
to later compute:

`D_I = E[gI_matched] - E[gI_mismatched]`

For M1b and M4qcf retain matched/mismatched Macro-F1 required to later compute:

`delta_mismatch^M = MacroF1_matched^M - MacroF1_mismatched^M`

and:

`G = delta_mismatch^M1b - delta_mismatch^M4qcf`

M4qc matched/mismatched performance must also be retained as a causal control.

H10-F is NOT evaluated in Step14A.

The Step13A frozen Step14B rule remains:

- `D_I > 0` for all three M4qcf seeds; and
- aggregate mean `G > 0`.

## 11. Required per-sample diagnostic fields

Quality-condition rows must include, where applicable:

- sample identity
- seed
- architecture
- condition
- corruption family
- corrupted modality
- severity
- prediction
- target
- classification correctness
- qT
- qV
- cTV
- rT
- rV
- alphaT
- alphaV
- interaction multiplier
- official_test_accessed

Fields that are architecturally unavailable must be null rather than
fabricated.

Mismatch rows must additionally preserve:

- receiver sample identity
- vision donor identity
- receiver Stage-1 class
- donor Stage-1 class
- matched/mismatched state

## 12. Integrity checks

The Step14A runner must fail loudly on:

- missing checkpoint;
- unexpected checkpoint architecture;
- seed mismatch;
- malformed checkpoint configuration;
- failure of strict state-dict loading;
- unexpected validation sample count;
- duplicate or missing validation identities;
- non-finite model outputs;
- malformed probabilities;
- invalid M4qcf weights;
- `alphaT + alphaV` outside numerical tolerance of 1;
- mismatch self-pairs;
- mismatch class changes;
- mismatch mapping cardinality violations;
- architecture-dependent corruption tensors;
- corruption hash mismatch;
- official test access;
- checkpoint reselection;
- malformed or incomplete output artifacts.

## 13. Output artifacts

The formal output root is:

`experiments/fakeddit/v027_m4qcf_robustness_diagnostics`

At minimum Step14A must produce:

- `manifest.json`
- `quality_conditions.json`
- `validation_derangement_mapping.json`
- per-seed/per-architecture quality diagnostic JSONL artifacts
- per-seed/per-architecture mismatch diagnostic JSONL artifacts
- condition-level metric summaries
- `summary.json`

The manifest must record checkpoint paths, checkpoint epochs, protocol
versions, corruption hashes, mapping provenance, formal seeds, architecture
identities, validation sample count, and explicit official-test sealing.

## 14. Result-exposure boundary

Step14A generates diagnostic evidence only.

It must not emit formal decisions for:

- H8-F
- H9-F
- H10-F

Those hypotheses remain:

`NOT_COMPUTED`

until the separately implemented Step14B formal analyzer consumes the frozen
Step14A artifacts.

No Step14A result may be used to modify:

- epsilon;
- gamma;
- fusion equations;
- corruption families;
- corruption severities;
- mismatch construction;
- checkpoints;
- decision thresholds;
- hypothesis criteria.

Any changed formulation requires a new versioned protocol.

## 15. Interpretation boundary

Even if later supported, H8-F/H9-F/H10-F concern controlled Fakeddit
representation corruptions and class-preserving text-vision mismatch only.

They do not establish:

- open-world factual verification;
- source credibility;
- intent inference;
- human trust;
- universal semantic consistency;
- external-evidence reasoning;
- raw-world corruption robustness;
- temporal or geographic verification.

## 16. Next steps

After this protocol is frozen:

1. implement the deterministic Step14A runner;
2. implement invariant/regression tests;
3. run a smoke diagnostic without formal hypothesis decisions;
4. audit generated artifacts;
5. execute the complete formal Step14A diagnostic matrix;
6. freeze the resulting diagnostic evidence;
7. implement Step14B formal H8-F/H9-F/H10-F analysis.
