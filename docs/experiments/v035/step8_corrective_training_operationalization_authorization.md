# AEGIS v0.35 Step8 鈥?Corrective-Training Operationalization and Authorization

Step8 is the final non-training gate before the controlled v0.35 corrective GPU run.

Parent Step7: `bcb5df55347f27cde351157b2c457f6191b2bf71`

Authorized Step9 condition: `quality_compatibility_utility_supervised_intervention` with seeds 42, 43, and 44, using the exact frozen Step6B implementation and inherited training configuration.

The selector remains 10 -> 16 -> 1. Posterior context remains label-free and detached. BCEWithLogits and effective utility-loss weight 0.50 remain frozen. M3 remains >= 0.05 per seed and M4 remains > 0 per seed.

Step8 authorizes corrective training in Step9 only. It does not execute training. Formal evaluation and official-test access remain prohibited. No source, hyperparameter, seed, threshold, intervention-mapping, loss-family, or loss-weight change is authorized.

Step9 may combine the three-seed controlled training run, provenance/evidence freeze, and M3/M4 adjudication in one fail-closed stage. Any failed prerequisite or run stops the stage without weakening the frozen criteria.
