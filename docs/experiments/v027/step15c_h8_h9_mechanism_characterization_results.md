# AEGIS v0.27 Step15C — Exploratory H8→H9 Mechanism Characterization

## Status and boundary

**FROZEN EXPLORATORY / POST-HOC CHARACTERIZATION.** H8-F remains **SUPPORTED** and H9-F remains **NOT_SUPPORTED**. Correlations are descriptive; no significance or causal claim is permitted.

Implementation commit: `c9d9912`. Official test access, training, tuning, checkpoint reselection, architecture modification, condition dropping, and seed dropping: **NONE**.

## Overall association

| n | mean D_m | median D_m | mean R_i | median R_i | R_i positive | R_i nonpositive | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 54 | 0.133367390132 | 0.014789855197 | 0.051111288732 | 0.001426187267 | 29 | 25 | 0.624523100992 | 0.695375347897 |

All 54 observations showed positive down-weighting, yet 25/54 had nonpositive R_i. Down-weighting was therefore observed consistently but was not sufficient for consistent downstream benefit.

## Grouped descriptive associations

### Corrupted modality

| corrupted_modality | n | mean D_m | median D_m | mean R_i | median R_i | R_i positive | R_i nonpositive | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 27 | 0.134444855394 | 0.016231194645 | 0.006289615539 | -0.000236793704 | 13 | 14 | 0.793061343510 | 0.776351970669 |
| vision | 27 | 0.132289924871 | 0.011913707316 | 0.095932961924 | 0.002817673395 | 16 | 11 | 0.860127367722 | 0.525206232814 |

### Corruption family

| corruption_family | n | mean D_m | median D_m | mean R_i | median R_i | R_i positive | R_i nonpositive | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attenuation | 24 | 0.196140628771 | 0.188789773054 | 0.060426645905 | 0.003551877487 | 14 | 10 | 0.564887470041 | 0.671304347826 |
| gaussian_noise | 24 | 0.002939238807 | 0.001175132066 | -0.002609519854 | -0.005787787543 | 9 | 15 | 0.676316376509 | 0.505217391304 |
| zero_dropout | 6 | 0.403987040882 | 0.404336098105 | 0.228733094380 | 0.231674921370 | 6 | 0 | -0.227261782501 | -0.142857142857 |

### Severity

| severity | n | mean D_m | median D_m | mean R_i | median R_i | R_i positive | R_i nonpositive | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 12 | 0.006510313933 | 0.004039697632 | -0.005713626380 | -0.005325370828 | 1 | 11 | 0.284968712243 | 0.202797202797 |
| 0.500000000000 | 12 | 0.048579046644 | 0.031074148059 | -0.000063679072 | -0.002218290215 | 5 | 7 | 0.567008567050 | 0.377622377622 |
| 0.750000000000 | 12 | 0.137852101616 | 0.129479657508 | 0.008564703418 | 0.005482007117 | 8 | 4 | 0.311328624539 | 0.195804195804 |
| 1.000000000000 | 18 | 0.271474528935 | 0.400361586548 | 0.151475600885 | 0.026218578748 | 15 | 3 | 0.544189309707 | 0.557632398754 |

## Quadrant characterization

| quadrant | n | proportion | mean D_m | mean R_i | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- |
| D_m_positive_R_i_positive | 29 | 0.537037037037 | 0.217044658792 | 0.101963793937 | 0.568934812100 | 0.698075974346 |
| D_m_positive_R_i_nonpositive | 25 | 0.462962962963 | 0.036301758487 | -0.007877617307 | 0.137231597179 | 0.193076923077 |
| D_m_nonpositive_R_i_positive | 0 | 0.000000000000 | undefined | undefined | undefined | undefined |
| D_m_nonpositive_R_i_nonpositive | 0 | 0.000000000000 | undefined | undefined | undefined | undefined |

## All 54 paired observations

| seed | condition | modality | family | severity | D_m | R_i | quadrant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.017622654229 | -0.003036368078 | D_m_positive_R_i_nonpositive |
| 42 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.126090557605 | 0.020023082698 | D_m_positive_R_i_positive |
| 42 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.275565076947 | 0.027291677040 | D_m_positive_R_i_positive |
| 42 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.405745863013 | 0.004286081580 | D_m_positive_R_i_positive |
| 42 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.000072344333 | -0.008099538905 | D_m_positive_R_i_nonpositive |
| 42 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.000385982960 | 0.003958514559 | D_m_positive_R_i_positive |
| 42 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.000990043700 | 0.001960526232 | D_m_positive_R_i_positive |
| 42 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.001831680983 | -0.013043628246 | D_m_positive_R_i_nonpositive |
| 42 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.405745863013 | 0.004286081580 | D_m_positive_R_i_positive |
| 42 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.007275986105 | 0.000891848302 | D_m_positive_R_i_positive |
| 42 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.084274434060 | 0.002817673395 | D_m_positive_R_i_positive |
| 42 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.266501493201 | 0.007930322997 | D_m_positive_R_i_positive |
| 42 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.402926333196 | 0.416682344555 | D_m_positive_R_i_positive |
| 42 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.000803409159 | -0.012082991803 | D_m_positive_R_i_nonpositive |
| 42 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.003221824944 | -0.007004734311 | D_m_positive_R_i_nonpositive |
| 42 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.007470326513 | 0.003033691238 | D_m_positive_R_i_positive |
| 42 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.014605523348 | 0.015068477156 | D_m_positive_R_i_positive |
| 42 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.402926333196 | 0.416682344555 | D_m_positive_R_i_positive |
| 43 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.016231194645 | -0.004178095424 | D_m_positive_R_i_nonpositive |
| 43 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.109017722160 | -0.000236793704 | D_m_positive_R_i_nonpositive |
| 43 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.271124946892 | 0.021986026486 | D_m_positive_R_i_positive |
| 43 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.400361586548 | 0.026218578748 | D_m_positive_R_i_positive |
| 43 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.000080923021 | -0.004106343624 | D_m_positive_R_i_nonpositive |
| 43 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.000400768340 | -0.008240421393 | D_m_positive_R_i_nonpositive |
| 43 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.000838780761 | -0.008080158520 | D_m_positive_R_i_nonpositive |
| 43 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.001360220432 | -0.019254229701 | D_m_positive_R_i_nonpositive |
| 43 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.400361586548 | 0.026218578748 | D_m_positive_R_i_positive |
| 43 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.008777611524 | -0.007156369338 | D_m_positive_R_i_nonpositive |
| 43 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.058926471174 | -0.004199786725 | D_m_positive_R_i_nonpositive |
| 43 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.286324750803 | -0.003260481647 | D_m_positive_R_i_nonpositive |
| 43 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.411336350136 | 0.422622577389 | D_m_positive_R_i_positive |
| 43 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.000451665282 | -0.001068291137 | D_m_positive_R_i_nonpositive |
| 43 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.001721210778 | -0.005102928853 | D_m_positive_R_i_nonpositive |
| 43 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.003841481864 | 0.013037573654 | D_m_positive_R_i_positive |
| 43 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.007291358858 | 0.011915885922 | D_m_positive_R_i_positive |
| 43 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.411336350136 | 0.422622577389 | D_m_positive_R_i_positive |
| 44 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.014974187046 | -0.002340458542 | D_m_positive_R_i_nonpositive |
| 44 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.108186906755 | 0.016739680365 | D_m_positive_R_i_positive |
| 44 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.251488988504 | 0.039338487698 | D_m_positive_R_i_positive |
| 44 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.409186242066 | 0.046667498184 | D_m_positive_R_i_positive |
| 44 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.000081026763 | -0.009448760405 | D_m_positive_R_i_nonpositive |
| 44 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.000476426244 | -0.007509609621 | D_m_positive_R_i_nonpositive |
| 44 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.000908740759 | -0.007610256144 | D_m_positive_R_i_nonpositive |
| 44 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.001694539309 | -0.020638030245 | D_m_positive_R_i_nonpositive |
| 44 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.409186242066 | 0.046667498184 | D_m_positive_R_i_positive |
| 44 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.011096819162 | -0.011465501377 | D_m_positive_R_i_nonpositive |
| 44 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.087183221042 | -0.012672987468 | D_m_positive_R_i_nonpositive |
| 44 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.282789823353 | -0.010631021232 | D_m_positive_R_i_nonpositive |
| 44 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.394365870334 | 0.455921485824 | D_m_positive_R_i_positive |
| 44 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.000655945927 | -0.006472646232 | D_m_positive_R_i_nonpositive |
| 44 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.003063033670 | 0.000664162193 | D_m_positive_R_i_positive |
| 44 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.006380766094 | 0.017780053209 | D_m_positive_R_i_positive |
| 44 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.011913707316 | 0.007715208477 | D_m_positive_R_i_positive |
| 44 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.394365870334 | 0.455921485824 | D_m_positive_R_i_positive |

## Scientific interpretation

### Directly Observed Findings

- All 54 observations had positive corrupted-modality down-weighting D_m, preserving H8-F as SUPPORTED.
- Only 29/54 observations had positive downstream change R_i; 25/54 combined positive D_m with nonpositive R_i, preserving H9-F as NOT_SUPPORTED.
- The pooled descriptive Pearson correlation was 0.624523100992 and tie-aware Spearman correlation was 0.695375347897.
- Mean D_m was 0.217044658792 in the positive-R_i quadrant and 0.036301758487 in the nonpositive-R_i quadrant.
- The nonpositive-R_i quadrant retained weak within-quadrant association: Pearson 0.137231597179 and Spearman 0.193076923077.

### Descriptive Associations

- Larger down-weighting was descriptively associated with larger robustness benefit in the pooled observations.
- Association strength varied by modality: text Pearson/Spearman 0.793061343510/0.776351970669; vision 0.860127367722/0.525206232814.
- Association strength varied by corruption family; zero dropout had negative within-family Pearson/Spearman values despite all six R_i values being positive.
- Within-severity Spearman values ranged from 0.195804195804 at severity 0.75 to 0.557632398754 at severity 1.0, below the pooled value.

### Candidate Mechanistic Explanations

- Reliability-informed down-weighting may be helpful when it is sufficiently large, but weight adaptation alone may not guarantee downstream benefit.
- The pooled association may combine both within-condition relationships and between-regime differences in corruption severity and family.
- Residual failures may arise downstream of weight adaptation, including representation quality, retained clean-modality sufficiency, fusion interactions, or classifier decision geometry.

### Unsupported Explanations

- The correlations do not establish that down-weighting caused the observed changes in Macro-F1.
- No significance test, confidence interval, intervention, mediation analysis, or causal identification was performed.
- Step15C cannot identify which downstream component produced the 25 positive-D_m/nonpositive-R_i observations.

### Candidate V028 Questions

- Can a prospectively frozen v0.28 mechanism translate reliable modality down-weighting into more consistent classification benefit?
- Can confirmatory evaluation distinguish within-regime adaptation effects from pooled severity/family effects?
- Can sample-level diagnostics prospectively identify when down-weighting will be insufficient because the retained modality is itself inadequate?

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

Proceed to **Step15D — H10 mismatch pathway decomposition**. Step15C does not select or validate a new architecture.
