# AEGIS v0.27 — Step 13D M4qcf Smoke and Gradient-Causality Audit

## Status

**COMPLETE AND FROZEN**

Record version: `0.27.0-step13d`

Protocol: `0.27.0-step13a`

Architecture: `quality_compatibility_fusion`

Date: 2026-09-07

## Purpose

Step 13D is an engineering and causal-integrity validation gate for M4qcf.

It does not test the preregistered H7-F, H8-F, H9-F, or H10-F
scientific hypotheses and does not establish a robustness or performance
benefit.

## Provenance

- Step 13A protocol:
  `822c913 docs: freeze v0.27 M4qcf reliability-informed fusion protocol`
- Step 13B deterministic reliability controller:
  `ec0909e feat: add v0.27 deterministic M4qcf reliability controller`
- Step 13C M4qcf model integration:
  `4bfab57 feat: integrate v0.27 M4qcf reliability-informed fusion`

## Frozen M4qcf formulation

Intrinsic-quality and compatibility signals:

- `q_T`
- `q_V`
- `c_TV`

Reliability controller:

`r_T = q_T [epsilon + (1 - epsilon)c_TV]`

`r_V = q_V [epsilon + (1 - epsilon)c_TV]`

with:

`epsilon = 0.10`

Normalized modality weights:

`alpha_T = (r_T + epsilon) / (r_T + r_V + 2 epsilon)`

`alpha_V = (r_V + epsilon) / (r_T + r_V + 2 epsilon)`

Reliability-weighted evidence:

`z_R = alpha_T h_T + alpha_V h_V`

Final M4qcf fusion:

`z_M4qcf = z_R + gamma c_TV z_I`

with:

`gamma = 1.0`

Reliability signals used by the primary classification/fusion path are
stop-gradient signals.

The deterministic reliability controller contains no trainable parameters.

## Gradient-causality audit

Seed: `42`

Batch size: `8`

Shared dimension: `128`

Device: `cuda`

Status: **PASS**

Classification-side gradient norms:

- text quality estimator: `0.0`
- vision quality estimator: `0.0`
- compatibility estimator: `0.0`
- text projection: `0.012759046134306118`
- vision projection: `0.01008395773533266`
- interaction pathway: `0.009812770033022389`

Auxiliary-side gradient norms:

- text quality estimator: `0.4060462233610451`
- vision quality estimator: `0.4354357193224132`
- compatibility estimator: `0.8509499356150627`

Controller parameter count: `0`

Fusion equation exact: `true`

Weights strictly bounded: `true`

Weights sum to one: `true`

Maximum absolute weight-sum error:
`5.960464477539063e-08`

Interaction suppression monotonic: `true`

Official test accessed: `false`

Formal hypotheses computed: `false`

## Smoke training

Experiment:

`experiments/fakeddit/v027_m4qcf_smoke_seed42`

Seed: `42`

Device: NVIDIA GeForce GTX 1660 SUPER

Training samples: `5000`

Validation samples: `1000`

Train/validation overlap: `0`

Epochs: `1`

Batch size: `32`

Learning rate: `0.001`

Alignment weight: `0.5`

Classification weight: `1.0`

Quality weight: `1.0`

Compatibility weight: `1.0`

Train total loss after epoch 1:

`2.8115`

Train classification loss:

`0.5420`

Validation classification loss:

`0.398469`

Validation accuracy:

`0.818`

Validation F1:

`0.8022`

Validation Macro-F1:

`0.8168`

The one-epoch validation result is an engineering observation only.
It MUST NOT be interpreted as an H7-F result or compared formally with
the frozen M1b reference.

## Parameter accounting

Representation parameters: `511235`

Classification parameters: `17542`

Effective trainable parameters: `528777`

Controller trainable parameters: `0`

Parameter-update audit:

- classifier: updated
- compatibility estimator: updated
- gated base fusion: not updated
- gated interaction fusion: updated
- text projection: updated
- text quality estimator: updated
- vision projection: updated
- vision quality estimator: updated

The historical gated base fusion is not the M4qcf primary base evidence
path. M4qcf uses the deterministic reliability-weighted base
`alpha_T h_T + alpha_V h_V`.

## Serialization audit

The smoke run produced:

- `experiment.json`
- `metrics.jsonl`
- `best_model.pt`
- `final_model.pt`
- `best_validation_predictions.json`
- `summary.json`

Serialized protocol metadata confirms:

- protocol version: `0.27.0-step13a`
- epsilon: `0.1`
- gamma: `1.0`
- controller parameter-free: `true`
- stop-gradient reliability signals: `true`
- affects primary fusion: `true`
- training mixture: `same_as_m4qc_step11a`
- training mismatch: `deterministic_cyclic_vision_derangement`
- formal hypotheses computed: `false`
- official test accessed: `false`

## Regression validation

Full repository regression:

`806 passed in 12.51s`

`git diff --check` produced no blocking error. The Windows LF-to-CRLF
working-copy warning is non-blocking.

## Scientific boundary

Step 13D establishes implementation integrity and the intended causal
gradient separation of M4qcf.

It does NOT establish:

- clean-performance preservation under H7-F;
- corrupted-modality down-weighting under H8-F;
- robustness improvement under H9-F;
- mismatch-performance improvement under H10-F;
- open-world factual verification;
- source credibility;
- misinformation/disinformation intent;
- real-world reliability.

H7-F through H10-F remain **NOT COMPUTED**.

The official Fakeddit test split remains **SEALED / NOT ACCESSED**.

## Step 13D decision

**PASS**

M4qcf is authorized to proceed to Step 13E formal preregistered
three-seed training under the frozen Step 13A protocol.

No M4qcf scientific performance conclusion is recorded at Step 13D.