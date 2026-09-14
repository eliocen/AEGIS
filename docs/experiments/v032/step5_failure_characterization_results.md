# AEGIS v0.32 Step5 鈥?One-Time Frozen Failure Characterization

## Frozen decisions

- Q1 target signal: **NOT_AVAILABLE**
- Q2 candidate action quality: **NOT_AVAILABLE**
- Q3 target-transform compression: **NOT_AVAILABLE**
- Q4 selector learning: **SELECTOR_NONLEARNING** (partial evidence)
- Q5 evidence/utility discrimination: **NON_DISCRIMINATIVE**
- Q6 constrained attribution: **selector_optimization_failure**

## Q4 available evidence

- Utility probability std across 19 condition means: `0.003421955726`
- Clean utility probability: `0.498779816379`
- Graded utility probability: `0.497844296926`
- Catastrophic utility probability: `0.491937294612`
- Matched utility probability: `0.498779816379`
- Mismatched utility probability: `0.498110801319`
- Clean active intervention rate: `0.000000000000`
- Graded active intervention rate: `0.000000000000`
- Catastrophic active intervention rate: `0.000000000000`
- Matched active intervention rate: `0.000000000000`
- Mismatched active intervention rate: `0.000000000000`

Training-time utility trajectory, utility-loss trajectory, parameter movement, and
gradient diagnostics remain **NOT_AVAILABLE** and were not reconstructed.

## Q5 discrimination

- Mismatched minus matched utility: `-0.000669015060`
- Catastrophic minus clean utility: `-0.006842521767`
- Frozen non-discrimination rule satisfied: **TRUE**
- Pearson(utility, realized degradation), n=19: `-0.5965252401736496`
- Spearman(utility, realized degradation), n=19: `-0.5694200351493849`
- Pearson(utility, performance delta vs M4qcs-w), n=19: `-0.6798984188157233`
- Spearman(utility, performance delta vs M4qcs-w), n=19: `-0.6063268892794376`

## Q6 attribution

**selector_optimization_failure**

Q4 available evidence satisfies the prospectively frozen selector_nonlearning rule; Q1-Q3 causal alternatives remain untestable and training-time optimization diagnostics are unavailable.

The attribution is intentionally limited. Q1-Q3 causal mechanisms cannot be
distinguished because their required frozen training-time evidence was not
recorded. No new thresholds, rebinning, retraining, re-evaluation, or
missing-evidence reconstruction was performed.

Official test samples accessed: **0**.
