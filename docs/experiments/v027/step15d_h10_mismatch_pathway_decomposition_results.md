# AEGIS v0.27 Step15D — Exploratory H10 Mismatch Pathway Decomposition

## Status and boundary

**FROZEN EXPLORATORY / POST-HOC CHARACTERIZATION.** H10-F remains **NOT_SUPPORTED**. M4qc is retained only as a descriptive control. No causal or significance claim is made.

Implementation commit: `45bf87b`. The analysis consumed 18,000 persisted validation-sample rows. Official test access, retraining, tuning, checkpoint reselection, architecture modification, and new mismatch generation: **NONE**.

## Formal mechanism-versus-outcome result

| seed | D_I | M1b drop | M4qcf drop | G |
| --- | --- | --- | --- | --- |
| 42 | 0.448590586446 | 0.014094008902 | 0.025911914765 | -0.011817905863 |
| 43 | 0.395744409232 | 0.002881342384 | 0.008857204096 | -0.005975861711 |
| 44 | 0.416610787562 | 0.027145919454 | 0.032223969072 | -0.005078049618 |

Mean D_I was **0.420315261080**, while mean G was **-0.007623939064**. The mechanism activated as intended, but mismatch degradation was worse than M1b in every seed.

## Architecture metrics by seed

| seed | architecture | matched Macro-F1 | mismatched Macro-F1 | Macro-F1 drop | matched accuracy | mismatched accuracy | accuracy drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | M1b | 0.871997951967 | 0.857903943066 | 0.014094008902 | 0.872000000000 | 0.858000000000 | 0.014000000000 |
| 42 | M4qc | 0.862746217757 | 0.863711613775 | -0.000965396018 | 0.863000000000 | 0.864000000000 | -0.001000000000 |
| 42 | M4qcf | 0.865909354724 | 0.839997439959 | 0.025911914765 | 0.866000000000 | 0.840000000000 | 0.026000000000 |
| 43 | M1b | 0.871851860751 | 0.868970518367 | 0.002881342384 | 0.872000000000 | 0.869000000000 | 0.003000000000 |
| 43 | M4qc | 0.865986598660 | 0.851999407998 | 0.013987190662 | 0.866000000000 | 0.852000000000 | 0.014000000000 |
| 43 | M4qcf | 0.869770274765 | 0.860913070669 | 0.008857204096 | 0.870000000000 | 0.861000000000 | 0.009000000000 |
| 44 | M1b | 0.876972318772 | 0.849826399318 | 0.027145919454 | 0.877000000000 | 0.850000000000 | 0.027000000000 |
| 44 | M4qc | 0.865785256410 | 0.853766025641 | 0.012019230769 | 0.866000000000 | 0.854000000000 | 0.012000000000 |
| 44 | M4qcf | 0.866566474476 | 0.834342505404 | 0.032223969072 | 0.867000000000 | 0.835000000000 | 0.032000000000 |

## M4qcf mechanism metrics

| seed | matched compatibility | mismatched compatibility | compatibility drop | D_I | text-weight change | vision-weight change | effective-text drop | effective-vision drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.795567698295 | 0.346977111849 | 0.448590586446 | 0.448590586446 | -0.000216101468 | 0.000216092795 | 0.387840594456 | 0.390007690512 |
| 43 | 0.795017981453 | 0.399273572221 | 0.395744409232 | 0.395744409232 | -0.000054695815 | 0.000054688185 | 0.341095608935 | 0.341707792327 |
| 44 | 0.783608791395 | 0.366998003833 | 0.416610787562 | 0.416610787562 | 0.000760109633 | -0.000760116160 | 0.362186347924 | 0.355737180538 |

## Aggregate transition characterization

| architecture | pairs | flips | flip rate | harmful flips | beneficial flips | harmful minus beneficial |
| --- | --- | --- | --- | --- | --- | --- |
| M1b | 3000 | 554 | 0.184666666667 | 299 | 255 | 44 |
| M4qc | 3000 | 345 | 0.115000000000 | 185 | 160 | 25 |
| M4qcf | 3000 | 357 | 0.119000000000 | 212 | 145 | 67 |

M4qcf reduced total flips relative to M1b but had a larger net harmful imbalance (67 versus 44), explaining how strong suppression could coexist with worse mismatch degradation.

## Per-seed transition and confidence results

| seed | architecture | transition | count | proportion | flip count | mean confidence change |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | M1b | matched_correct_to_mismatched_correct | 770 | 0.770000000000 | 0 | -0.005753776083 |
| 42 | M1b | matched_correct_to_mismatched_incorrect | 102 | 0.102000000000 | 102 | -0.115711897027 |
| 42 | M1b | matched_incorrect_to_mismatched_correct | 88 | 0.088000000000 | 88 | 0.074039977383 |
| 42 | M1b | matched_incorrect_to_mismatched_incorrect | 40 | 0.040000000000 | 0 | -0.006123098731 |
| 42 | M4qc | matched_correct_to_mismatched_correct | 809 | 0.809000000000 | 0 | -0.009492363933 |
| 42 | M4qc | matched_correct_to_mismatched_incorrect | 54 | 0.054000000000 | 54 | -0.116695657924 |
| 42 | M4qc | matched_incorrect_to_mismatched_correct | 55 | 0.055000000000 | 55 | -0.026190149784 |
| 42 | M4qc | matched_incorrect_to_mismatched_incorrect | 82 | 0.082000000000 | 0 | -0.046937705540 |
| 42 | M4qcf | matched_correct_to_mismatched_correct | 793 | 0.793000000000 | 0 | -0.044333983039 |
| 42 | M4qcf | matched_correct_to_mismatched_incorrect | 73 | 0.073000000000 | 73 | -0.109005353222 |
| 42 | M4qcf | matched_incorrect_to_mismatched_correct | 47 | 0.047000000000 | 47 | 0.044419516908 |
| 42 | M4qcf | matched_incorrect_to_mismatched_incorrect | 87 | 0.087000000000 | 0 | -0.050689688359 |
| 43 | M1b | matched_correct_to_mismatched_correct | 781 | 0.781000000000 | 0 | -0.015216311404 |
| 43 | M1b | matched_correct_to_mismatched_incorrect | 91 | 0.091000000000 | 91 | -0.120154423373 |
| 43 | M1b | matched_incorrect_to_mismatched_correct | 88 | 0.088000000000 | 88 | 0.101792271164 |
| 43 | M1b | matched_incorrect_to_mismatched_incorrect | 40 | 0.040000000000 | 0 | -0.033455155790 |
| 43 | M4qc | matched_correct_to_mismatched_correct | 800 | 0.800000000000 | 0 | -0.010486129969 |
| 43 | M4qc | matched_correct_to_mismatched_incorrect | 66 | 0.066000000000 | 66 | -0.083987909736 |
| 43 | M4qc | matched_incorrect_to_mismatched_correct | 52 | 0.052000000000 | 52 | 0.049310273849 |
| 43 | M4qc | matched_incorrect_to_mismatched_incorrect | 82 | 0.082000000000 | 0 | -0.028855683600 |
| 43 | M4qcf | matched_correct_to_mismatched_correct | 802 | 0.802000000000 | 0 | -0.041992949726 |
| 43 | M4qcf | matched_correct_to_mismatched_incorrect | 68 | 0.068000000000 | 68 | -0.091852531714 |
| 43 | M4qcf | matched_incorrect_to_mismatched_correct | 59 | 0.059000000000 | 59 | -0.036331143420 |
| 43 | M4qcf | matched_incorrect_to_mismatched_incorrect | 71 | 0.071000000000 | 0 | -0.073474947835 |
| 44 | M1b | matched_correct_to_mismatched_correct | 771 | 0.771000000000 | 0 | 0.003203973077 |
| 44 | M1b | matched_correct_to_mismatched_incorrect | 106 | 0.106000000000 | 106 | -0.068610731723 |
| 44 | M1b | matched_incorrect_to_mismatched_correct | 79 | 0.079000000000 | 79 | 0.074779742126 |
| 44 | M1b | matched_incorrect_to_mismatched_incorrect | 44 | 0.044000000000 | 0 | 0.006805745038 |
| 44 | M4qc | matched_correct_to_mismatched_correct | 801 | 0.801000000000 | 0 | -0.006962498252 |
| 44 | M4qc | matched_correct_to_mismatched_incorrect | 65 | 0.065000000000 | 65 | -0.114407061614 |
| 44 | M4qc | matched_incorrect_to_mismatched_correct | 53 | 0.053000000000 | 53 | 0.023576029067 |
| 44 | M4qc | matched_incorrect_to_mismatched_incorrect | 81 | 0.081000000000 | 0 | -0.047252587330 |
| 44 | M4qcf | matched_correct_to_mismatched_correct | 796 | 0.796000000000 | 0 | -0.040570521849 |
| 44 | M4qcf | matched_correct_to_mismatched_incorrect | 71 | 0.071000000000 | 71 | -0.142461206712 |
| 44 | M4qcf | matched_incorrect_to_mismatched_correct | 39 | 0.039000000000 | 39 | -0.022310987497 |
| 44 | M4qcf | matched_incorrect_to_mismatched_incorrect | 94 | 0.094000000000 | 0 | -0.033264720694 |

Confidence is the probability assigned to the predicted class; change is mismatched minus matched confidence.

## Scientific interpretation

### Directly Observed Findings

- Compatibility and interaction multipliers decreased under mismatch for all three M4qcf seeds; mean interaction suppression D_I was 0.420315261080.
- M4qcf mismatch degradation exceeded M1b degradation in every seed, yielding negative G values for seeds 42, 43, and 44 and mean G -0.007623939064.
- M4qcf prediction-flip rates were 0.120, 0.127, and 0.110 across seeds, lower than M1b rates 0.190, 0.179, and 0.185.
- Across 3,000 paired samples, M4qcf had 212 harmful correct-to-incorrect transitions and 145 beneficial incorrect-to-correct transitions, a net harmful imbalance of 67.
- The corresponding net harmful imbalances were 44 for M1b and 25 for descriptive-control M4qc.
- M4qcf mean text and vision fusion weights changed by less than 0.001 in absolute value per seed, despite large reductions in effective reliabilities.

### Descriptive Associations

- Strong interaction suppression coexisted with fewer total prediction flips than M1b, but the remaining M4qcf flips were less favorably balanced.
- The compatibility pathway primarily reduced effective modality reliabilities while leaving average text/vision weight allocation nearly unchanged.
- For M4qcf, correct-to-incorrect transitions showed negative mean confidence change in every seed.
- M4qc had lower aggregate harmful-minus-beneficial imbalance than M4qcf, but remains only a descriptive control and is not promoted to a confirmatory comparison.

### Candidate Mechanistic Explanations

- Compatibility suppression may reduce interaction magnitude without selectively preserving decision-relevant cross-modal information.
- The mechanism may be suppressing both harmful and useful interactions, producing fewer flips but an unfavorable residual transition balance.
- Near-constant average modality weights suggest that mismatch response acted mainly through effective reliability and interaction scaling rather than modality reallocation.

### Unsupported Explanations

- Step15D does not establish that interaction suppression caused the observed transition pattern or degradation.
- Step15D does not establish whether individual suppressed interactions were semantically helpful or harmful.
- No statistical-significance, mediation, calibration, or out-of-distribution generalization claim is supported.

### Candidate V028 Questions

- Can a prospectively frozen mechanism distinguish harmful incompatibility from useful cross-modal interaction before suppression?
- Can mismatch handling improve the harmful-to-beneficial transition balance rather than only reduce the total flip rate?
- Should v0.28 separately control interaction suppression and modality-weight reallocation under a preregistered design?

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

## Next step

Proceed to **Step15E — exploratory scientific synthesis**. Step15D does not select or validate a new architecture.
