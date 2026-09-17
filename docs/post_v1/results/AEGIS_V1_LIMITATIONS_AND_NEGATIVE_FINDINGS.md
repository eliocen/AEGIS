# AEGIS Research Core v1.0 鈥?Limitations and Negative Findings

## 1. Selector-output discrimination remained unresolved

The principal negative finding is persistent selector-output discrimination failure. v0.34 M3 and M4 were `NOT_SUPPORTED`. v0.35 introduced the prospectively frozen posterior-context correction `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`, but M3 and M4 remained `NOT_SUPPORTED` across seeds 42, 43, and 44.

This is a meaningful negative scientific result about the tested selector correction. It is not evidence that all possible selector mechanisms will fail.

## 2. Optimization signal did not translate into qualifying activation

The v0.35 evidence showed nonzero selector gradient signal and measurable selector-parameter movement. Nevertheless, utility-probability standard deviation remained below the frozen M3 threshold and active-intervention rate remained zero under M4.

The bounded mechanistic interpretation is optimization/activation decoupling. It should not be rewritten as proof of a universal causal mechanism.

## 3. Formal active-intervention robustness was not computed

v0.36 required v0.35 M3 and M4 support across every formal seed before formal active-intervention robustness evaluation could proceed. That prerequisite failed. The robustness result is therefore `NOT_COMPUTED`.

`NOT_COMPUTED` is not a negative robustness score and cannot be used to claim that the classifier is non-robust.

## 4. Validation performance is not formal robustness evidence

The v0.35 corrective-training runs produced best validation Macro-F1 values of 0.8669, 0.8716, and 0.8610 for seeds 42, 43, and 44 respectively. These are validation results under the frozen training/data protocol.

They do not substitute for the blocked formal robustness evaluation and must not be generalized beyond the relevant dataset/task/protocol scope.

## 5. Dataset, task, population, and protocol scope

AEGIS v1.0 conclusions are bounded by the datasets, cached representations, task definitions, seeds, model configurations, and prospective criteria used in the frozen experiments. Evidence from those experiments does not establish equivalent performance on new populations, languages, domains, platforms, corruption regimes, or deployment environments.

## 6. Official test set remained sealed

Official-test samples accessed across the frozen research sequence remain **0**. Post-v1 packaging does not convert validation evidence into official-test evidence.

## 7. Production and operational limitations

AEGIS v1.0 does not establish production deployment readiness, autonomous attribution, autonomous threat determination, production alerting, or national-security operational validation. v0.37 supports analyst decision-support research characterization only.

## 8. Future modalities are outside v1.0 evidence

AEGIS-Vision specialization, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual expansion, full multimodal expansion, and broader-system integration remain future work. Their inclusion in a roadmap does not make them implemented or evaluated v1.0 capabilities.

## 9. Negative-result language discipline

`NOT_SUPPORTED` means a frozen criterion was evaluated and not satisfied. It does not mean disproven, impossible, or permanently invalid.

`NOT_COMPUTED` means the result does not exist under the frozen protocol. It must not be reported as zero, failed robustness, or a negative experimental measurement.

`CHARACTERIZED` supports bounded description. It does not automatically authorize general causal, operational, or production claims.
