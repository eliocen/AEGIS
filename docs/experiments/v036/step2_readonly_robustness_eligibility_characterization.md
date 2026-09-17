# AEGIS v0.36 Step2 鈥?Read-Only Robustness-Eligibility Characterization

**Status:** READ_ONLY_ROBUSTNESS_ELIGIBILITY_CHARACTERIZATION_FROZEN

## Decisions

- V36-Q1 鈥?Selector activation eligibility: **INELIGIBLE**
- V36-Q2 鈥?Optimization versus activation: **OBSERVED_OPTIMIZATION_ACTIVATION_DECOUPLING**
- V36-Q3 鈥?Seed consistency: **CONSISTENT_FAILURE_ACROSS_ALL_THREE_SEEDS**
- V36-Q4 鈥?Performance versus activation: **NOT_AVAILABLE_FROM_UNAMBIGUOUS_FROZEN_SUMMARY_FIELD**
- V36-Q5 鈥?Formal robustness admissibility: **FORMAL_ACTIVE_INTERVENTION_ROBUSTNESS_EVALUATION_INADMISSIBLE**

## Core finding

The learned selector received nonzero optimization signal and underwent nonzero parameter movement, but selector-output discrimination remained below the frozen M3 requirement and active intervention remained zero across seeds 42, 43, and 44.

Under the prospectively frozen Step1 rule, this makes formal robustness evaluation of the mechanism **as an active learned intervention mechanism** scientifically ineligible.

This does not establish that the classifier itself lacks robustness, and it does not establish a unique sole cause of selector failure.

## Q4 evidence handling

No single unambiguous best-validation Macro-F1 field was available under the frozen tracked-summary schema used by this script; Q4 is therefore NOT_AVAILABLE rather than reconstructed.

Validation performance is not treated as robustness evidence.

## Integrity boundary

This step used only already-frozen tracked evidence. It performed no training, implementation, source modification, corruption-condition execution, threshold tuning, checkpoint reselection, formal robustness evaluation, or official-test access.

**Official test samples accessed:** `0`

## Next

v0.36 Step3 should compactly freeze the v0.36 scientific disposition/closure and establish the v0.37 threat-intelligence characterization entry boundary.
