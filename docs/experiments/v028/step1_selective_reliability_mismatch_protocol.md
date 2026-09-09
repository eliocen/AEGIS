# AEGIS v0.28 Step1 ? Selective Reliability and Mismatch-Control Protocol

**Status:** FROZEN PROSPECTIVE PROTOCOL ? BEFORE IMPLEMENTATION, TRAINING, OR v0.28 RESULT EXPOSURE

## Scientific purpose

Can a prospectively specified selective reliability controller translate corruption detection into consistently improved downstream classification while improving mismatch-transition quality?

The programme begins from frozen release `v0.27.2` at `1f302db86def3f97ce8f2f1a317e241cb144fe5f`. No v0.27 decision or record may be changed.

## Evidence classification

v0.28 produces **controlled validation-development evidence**. The 1,000-sample validation subset was previously exposed during v0.27; therefore v0.28 is prospective relative to its new protocol and implementation but is not independent external confirmation.

The official Fakeddit test split remains inaccessible.

## Architecture family

The primary architecture is `M4qcs`, with two mandatory ablations:

- `M4qcs-w`: selective modality-weight reallocation only.
- `M4qcs-i`: independent interaction suppression only.
- `M4qcs`: combined selective controller and text-Gaussian reliability supervision.

The frozen v0.27 `M4qcf` model remains the immediate incremental baseline; `M1b` remains the principal absolute robustness baseline.

### Controller equations

- Epsilon: `0.1`
- Allocation exponent beta: `2.0`
- Interaction exponent gamma: `1.0`
- Text selective score: `(epsilon + q_t) ** beta`
- Vision selective score: `(epsilon + q_v) ** beta`
- Allocation weights: normalized selective scores.
- Interaction multiplier: `epsilon + (1 - epsilon) * (c ** gamma)`.
- Quality controls allocation only; compatibility controls interaction only.
- Controller inputs are stop-gradient and the controller is parameter-free.

## Formal hypotheses

### V28-H1 ? Graded-corruption coverage

M4qcs improves positive-condition coverage under mild and graded corruptions relative to M1b while preserving catastrophic modality-loss robustness relative to M4qcf.

Support requires all of the following:

- Mean graded-condition macro-F1 improvement versus M1b of at least `0.01`.
- At least `26/36` positive graded seed-condition observations.
- Catastrophic-condition mean macro-F1 delta versus M4qcf of at least `-0.01`.

### V28-H2 ? Mismatch-transition balance

M4qcs improves mismatch-induced harmful-to-beneficial transition balance relative to both M1b and M4qcf rather than merely reducing total prediction flips.

Support requires lower aggregate net harmful flips and a lower harmful-to-beneficial ratio than both M1b and M4qcf, positive mean mismatch-gap improvement versus M1b, and no higher total flip rate than M1b.

### V28-H3 ? Mechanism separation

Separating interaction suppression from modality-weight reallocation yields a more selective mismatch response.

Support requires the combined M4qcs model to outperform both mandatory single-pathway ablations on net harmful flips without lower mismatch macro-F1, together with verified controller independence.

### V28-H4 ? Text Gaussian robustness

The text-side reliability objective improves robustness to text Gaussian perturbation without degrading the other frozen corruption regimes.

Support requires:

- Mean text-Gaussian macro-F1 delta versus M4qcf of at least `0.01`.
- At least `9/12` positive text-Gaussian seed-condition observations.
- Mean delta of at least `-0.01` across the other 42 non-clean observations.
- Clean macro-F1 delta versus M4qcf of at least `-0.01`.

## Data and evaluation

- Dataset: Fakeddit.
- Training cache: frozen 5,000-sample representation cache.
- Validation: exposed frozen 1,000-sample representation cache.
- Formal seeds: 42, 43, and 44.
- Corruptions: Gaussian noise, attenuation, and zero dropout.
- Modalities: text and vision.
- Continuous severities: 0.25, 0.50, 0.75, and 1.00.
- Mismatch: deterministic class-preserving validation derangement.
- Official-test samples accessed: 0.

## Clean-performance safeguard

M4qcs must remain within `-0.01` mean clean-validation macro-F1 of both M1b and M4qcf. Failure prevents recommendation as a successor regardless of individual robustness outcomes.

## Decision discipline

Every hypothesis uses conjunctive gates. A strong aggregate mean cannot override failed coverage, non-degradation, or transition-balance gates. Decisions are restricted to `SUPPORTED` or `NOT_SUPPORTED`.

Formal decisions must be frozen before any exploratory failure-mode analysis.

## Accelerated six-stage sequence

1. Freeze this combined scientific and architecture protocol.
2. Implement and test M4qcs, then complete smoke validation.
3. Complete locked three-seed training.
4. Run the unified formal evaluation.
5. Freeze formal decisions, then separately characterize failure modes.
6. Complete scientific synthesis, regression verification, closure, and release tagging.

## Prohibitions

- No official-test access.
- No post-exposure threshold tuning.
- No post-exposure checkpoint reselection.
- No post-exposure architecture modification.
- No dropping failed seeds or conditions.
- No independent-confirmation claim.
- No changes to frozen v0.27 decisions or records.

## Next step

Step2 ? implement M4qcs variants, unified runners, invariant tests, and smoke validation.
