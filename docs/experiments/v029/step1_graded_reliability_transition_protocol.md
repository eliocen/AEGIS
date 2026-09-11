# AEGIS v0.29 Step1 Graded Reliability and Transition-Control Protocol

> **FROZEN PROSPECTIVE PROTOCOL — BEFORE IMPLEMENTATION, TRAINING, OR RESULT EXPOSURE**

## Scientific objective

AEGIS v0.29 tests whether calibrated graded reliability allocation and transition-aware mismatch control can address the prospectively retained v0.28 failures while preserving clean performance and catastrophic modality-loss robustness.

This is controlled validation-development evidence, not independent confirmation. The official test split remains sealed.

## Frozen v0.28 handoff

- V28-H1: NOT_SUPPORTED
- V28-H2: NOT_SUPPORTED
- V28-H3: NOT_SUPPORTED
- V28-H4: NOT_SUPPORTED
- Clean-performance safeguard: PASSED
- Graded positive coverage: 9/36
- Text-Gaussian positive coverage: 3/12
- Official-test samples accessed: 0

## Architecture contract

| Architecture | Frozen role |
|---|---|
| M4qgr | Graded reliability allocation only; no transition controller |
| M4qtc | Transition-aware interaction control only; weights fixed at one half |
| M4qgrt | Combined graded reliability allocation and transition-aware interaction control |

The controller inputs are stop-gradient diagnostics. The control equations, epsilon, exponents, margins, mixture probabilities, and loss weights are frozen and cannot be tuned after smoke or formal-result exposure.

## Formal hypotheses

| Hypothesis | Prospective claim |
|---|---|
| V29-H1 | M4qgrt improves graded-corruption coverage over M4qcs while preserving catastrophic-loss performance relative to M4qcf |
| V29-H2 | M4qgrt improves mismatch degradation and harmful-to-beneficial transition balance over M1b and M4qcs-w |
| V29-H3 | The combined M4qgrt mechanism outperforms both mandatory ablations under the frozen corruption and mismatch gates |
| V29-H4 | M4qgrt improves text-Gaussian robustness over M4qcf and M4qcs while preserving other regimes |

Every decision is conjunctive. A favorable aggregate cannot override a failed coverage, comparator, invariant, or safeguard gate.

## Data and evaluation boundary

- Frozen Fakeddit training cache: 5,000 samples
- Frozen Fakeddit validation cache: 1,000 samples
- Formal seeds: 42, 43, 44
- Seven evaluated architectures
- Nineteen quality conditions per architecture and seed
- Shared corruption realizations and shared deterministic class-preserving mismatch mappings
- Clean-only checkpoint selection
- Official-test samples accessed: 0

## Accelerated six-gate sequence

1. Step1 — freeze this prospective protocol.
2. Step2 — implement M4qgr, M4qtc, and M4qgrt; run invariant tests and non-evidentiary smoke checks; freeze implementation.
3. Step3 — execute nine locked training runs and freeze clean-selected checkpoints and training records.
4. Step4 — execute one unified frozen corruption, mismatch, ablation, and safeguard evaluation without formal decisions.
5. Step5 — apply the frozen conjunctive analyzer exactly once and freeze V29-H1 through V29-H4 decisions.
6. Step6 — synthesize, verify, close, and tag v0.29.

## Prohibitions

No official-test access, post-exposure threshold tuning, checkpoint reselection, loss-weight changes, architecture modification, new conditions, seed or condition dropping, favorable-result reweighting, or alteration of frozen v0.28 records is permitted.

## Next step

Step2 — implement the three frozen v0.29 variants, unified runners, invariant tests, and non-evidentiary smoke validation.
