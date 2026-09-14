# AEGIS v0.33 Step1 鈥?Prospective Observability and Selector-Learning Protocol

This step freezes the observability and learning-mechanism requirements before
any v0.33 implementation.

## Mandatory observability

Formal v0.33 runs must persist machine-readable evidence for counterfactual
utility construction (`p_ref_true`, `p_candidate_true`, `delta_u`, `u_target`),
selector behavior (`utility_probability`, gates, activation), optimization
(`utility_loss`, selector parameter movement, gradient norm or equivalent), and
the effective loss weights actually used by the runtime.

## Pre-training gates

Formal training is prohibited unless:

1. the counterfactual target tensor contract passes;
2. positive and negative utility cases produce directionally distinct targets;
3. the real one-batch training path shows finite non-zero utility learning
   signal and measurable selector parameter movement;
4. runtime-effective loss weights are asserted;
5. the intervention remains inactive by default until its prospective rule is
   satisfied.

## Characterization rules

The v0.32 target-signal, candidate-action, compression, selector-learning, and
utility-discrimination rules are carried forward prospectively so future
failures can be diagnosed from recorded evidence rather than reconstructed
after exposure.

## Step2 boundary

Exact architecture identifier, loss equations, target transform, intervention
bounds, tensor contracts, and formal performance hypotheses are intentionally
deferred to **Step2** and must be frozen before any source modification,
training, or formal evaluation.

Official test samples accessed: **0**.
