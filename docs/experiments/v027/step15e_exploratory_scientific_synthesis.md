# AEGIS v0.27 Step15E — Exploratory Scientific Synthesis

## Status

**FROZEN EXPLORATORY / POST-HOC SYNTHESIS.** This document synthesizes only the frozen Step15B–D records. It does not retest or change H8-F, H9-F, or H10-F.

| Formal hypothesis | Frozen decision |
| --- | --- |
| H8_F | SUPPORTED |
| H9_F | NOT_SUPPORTED |
| H10_F | NOT_SUPPORTED |

No architecture or hyperparameter was selected. No new experiment, official test access, training, tuning, checkpoint reselection, corruption generation, or mismatch generation occurred.

## Central scientific conclusion

M4qcf demonstrated **reliable internal adaptation without reliable downstream robustness**. It consistently down-weighted corrupted modalities and suppressed incompatible interaction, but those internally coherent responses did not translate into sufficiently consistent Macro-F1 benefit or reduced mismatch degradation relative to M1b.

## Directly observed findings

- **O1:** M4qcf consistently down-weighted the corrupted modality in all 54 frozen seed-condition observations. Evidence: `{"H8_F": "SUPPORTED", "mean_D_m": 0.1333673901324747, "positive_D_m": "54/54"}`
- **O2:** Downstream robustness benefit was heterogeneous and failed the frozen cross-condition coverage criterion. Evidence: `{"H9_F": "NOT_SUPPORTED", "mean_R_i": 0.051111288731590224, "median_R_i": 0.001426187266855894, "negative_R_i": "25/54", "positive_R_i": "29/54"}`
- **O3:** The positive H9 aggregate was highly concentrated in catastrophic vision-loss conditions. Evidence: `{"remaining_48_mean_R_i": 0.003532432832697198, "six_severe_vision_rows_share_of_total_R_i": 0.938566599010638}`
- **O4:** Larger down-weighting was positively associated with larger R_i in the pooled exploratory observations, but 25 observations combined positive D_m with nonpositive R_i. Evidence: `{"D_m_positive_R_i_nonpositive": 25, "pearson_r": 0.6245231009919004, "spearman_rho": 0.6953753478973655}`
- **O5:** M4qcf strongly suppressed compatibility-informed interaction under mismatch in every seed without reducing mismatch degradation relative to M1b. Evidence: `{"H10_F": "NOT_SUPPORTED", "mean_D_I": 0.42031526107986356, "mean_G": -0.007623939063888622, "negative_G_seeds": "3/3"}`
- **O6:** M4qcf reduced total mismatch-induced prediction flips relative to M1b but retained a worse harmful-versus-beneficial flip balance. Evidence: `{"M1b_flip_rate": 0.18466666666666667, "M1b_net_harmful": 44, "M4qcf_flip_rate": 0.119, "M4qcf_net_harmful": 67}`

## Descriptive associations

- Down-weighting magnitude and robustness benefit moved together descriptively in the pooled conditions, but this relationship varied by modality, corruption family, and severity.
- Strongest positive gains occurred when vision information was completely removed; mild corruptions frequently showed no benefit.
- Text Gaussian noise remained a specific unfavorable regime, with negative R_i in 10/12 observations.
- Mismatch suppression substantially reduced effective reliabilities while average text/vision fusion weights remained nearly unchanged.
- Lower total prediction-flip rate did not imply lower mismatch degradation; the direction and balance of flips mattered.

## Candidate mechanistic explanations

- The reliability pathway may correctly detect unusable modalities yet lack sufficient selectivity to improve decisions under mild or distributed corruption.
- Down-weighting may be helpful but insufficient when the retained modality lacks adequate decision-relevant information or when downstream fusion/classifier geometry remains unfavorable.
- Compatibility suppression may remove both harmful and useful cross-modal interactions, reducing total changes without improving their harmful-to-beneficial balance.
- The current mechanism appears better suited to catastrophic modality loss than to consistently graded robustness across corruption regimes.

## Unsupported explanations

- Step15 does not establish that down-weighting caused improvements or failures in R_i.
- Step15 does not establish that compatibility suppression caused the mismatch transition imbalance.
- Step15 does not identify a uniquely responsible module, estimator, representation, or classifier component.
- Step15 does not establish statistical significance, calibration improvement, real-world robustness, factual verification, source credibility, or author intent.
- No Step15 result can be presented as preregistered confirmatory evidence.

## Limitations

- All analyses use the frozen 1,000-sample Fakeddit validation subset and three formal seeds.
- Corruptions act on frozen representations rather than raw real-world inputs.
- Only controlled single-modality corruption and deterministic class-preserving mismatch were examined.
- Severity-1 attenuation and zero dropout are operationally equivalent but remain separately retained under the frozen protocol.
- Exploratory subgroup correlations combine small group sizes, repeated seed-condition structure, and no inferential uncertainty quantification.
- M4qc is only a descriptive control; Step15 does not promote it to a selected architecture.

## Candidate v0.28 hypotheses

- **V28-H1-CANDIDATE** — A prospectively specified reliability mechanism can improve positive-condition coverage under mild and graded corruptions while preserving catastrophic modality-loss robustness. Status: `EXPLORATORY_CANDIDATE_REQUIRING_PROSPECTIVE_V028_FREEZE`.
- **V28-H2-CANDIDATE** — A prospectively specified mismatch mechanism can improve the harmful-to-beneficial transition balance rather than merely reduce total prediction flips. Status: `EXPLORATORY_CANDIDATE_REQUIRING_PROSPECTIVE_V028_FREEZE`.
- **V28-H3-CANDIDATE** — Separating interaction suppression from modality-weight reallocation can yield more selective mismatch responses under a preregistered evaluation. Status: `EXPLORATORY_CANDIDATE_REQUIRING_PROSPECTIVE_V028_FREEZE`.
- **V28-H4-CANDIDATE** — A prospectively defined text-side reliability objective can improve robustness to text Gaussian perturbation without degrading other frozen regimes. Status: `EXPLORATORY_CANDIDATE_REQUIRING_PROSPECTIVE_V028_FREEZE`.

## Interpretation boundary

- Results apply to the frozen controlled Fakeddit validation representation-corruption and class-preserving mismatch setting.
- Step15 is exploratory and post-hoc.
- Step15 does not establish statistical significance.
- Step15 does not establish causal effects.
- Step15 does not establish robustness to arbitrary real-world corruption.
- Step15 does not establish open-world factual verification.
- Step15 does not establish source credibility.
- Step15 does not establish author intent.
- Step15 does not establish human trust.
- Step15 does not establish universal semantic consistency.
- Step15 does not establish external evidence verification.
- Step15 does not establish temporal or geographic reasoning.

## Step15 completion and next action

Steps 15A–15E are complete once this synthesis is audited and frozen. Proceed only to **Step16 — v0.27 scientific synthesis and closure**. After Step16, formally tag/freeze v0.27 and move to a prospectively specified v0.28; do not expand v0.27.
