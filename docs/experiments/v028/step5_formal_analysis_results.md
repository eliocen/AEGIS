# AEGIS v0.28 Step5 Formal Analysis Results

> **FROZEN FORMAL CONTROLLED VALIDATION-DEVELOPMENT EVIDENCE**

## Record identity

- Record version: `0.28.0-step5-results`
- Frozen analyzer commit: `1afe476c46e10c3fb9ff702bc33f349381a690e2`
- Frozen Step4 record commit: `52a2314ce127de1984836a127d64b101f6422b9e`
- Formal-analysis source SHA256: `66df67d53150ef25c917675e54bd564ccb31d2ece3bb13320efdb6bc3a033b4f`
- Summary source SHA256: `19ef8b36080e3a03bc403610632c54cd078ab6dadb7ede28f0cc8a19e52eb0db`
- Official-test samples accessed: 0

## Formal decisions

All gates were applied conjunctively by the analyzer frozen before real Step4 evidence was analyzed. A failed gate cannot be overridden by a favorable mean or another passing gate.

| Hypothesis | Decision | Failed conjunctive gates |
|---|---|---|
| V28-H1 | NOT_SUPPORTED | `graded_mean_macro_f1_improvement_at_least_0p01`, `graded_positive_count_at_least_26_of_36`, `graded_positive_fraction_at_least_0p70` |
| V28-H2 | NOT_SUPPORTED | `harmful_to_beneficial_ratio_lower_than_M1b` |
| V28-H3 | NOT_SUPPORTED | `M4qcs_mismatch_macro_f1_at_least_M4qcs_w`, `M4qcs_net_harmful_flips_lower_than_M4qcs_w` |
| V28-H4 | NOT_SUPPORTED | `text_gaussian_mean_macro_f1_delta_at_least_0p01`, `text_gaussian_positive_count_at_least_9_of_12`, `text_gaussian_positive_fraction_at_least_0p75` |

## Clean-performance safeguard

The clean-performance safeguard **PASSED**.

- Mean clean delta versus M1b: `-0.007536053021`
- Mean clean delta versus M4qcf: `-0.001344043845`

Passing the safeguard does not override any failed formal hypothesis gate and does not convert an unsupported hypothesis into a supported one.

## Scientific interpretation

The prospectively specified v0.28 selective reliability and mismatch-control mechanisms did not satisfy the complete frozen support criteria for graded corruption coverage, mismatch transition balance, controller-combination advantage, or text-Gaussian robustness. All four outcomes are retained as NOT_SUPPORTED.

These results do not establish that the mechanisms are universally ineffective. They establish only that the frozen v0.28 implementations did not meet the prospectively defined conjunctive gates on the controlled Fakeddit validation-development evidence.

## Integrity and safety

- The one-time formal analyzer was not rerun by this recorder.
- No training, checkpoint reselection, threshold tuning, or architecture modification was performed.
- No seed or condition was dropped.
- The official test split remained sealed; official-test samples accessed: 0.
- All supported and unsupported outcomes were retained without post-exposure modification.

## Next step

Step6 — synthesize Steps1–5, verify the complete v0.28 record, close the development line, and prepare the frozen v0.28 release without expanding or retuning the experiment.
