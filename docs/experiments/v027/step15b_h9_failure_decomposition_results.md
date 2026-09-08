# AEGIS v0.27 Step15B — Exploratory H9 Failure Decomposition

## Status and scientific boundary

**FROZEN EXPLORATORY / POST-HOC CHARACTERIZATION.** The formal H9-F decision remains **NOT_SUPPORTED**. No subgroup result retests, replaces, weakens, or overturns that decision.

Implementation commit: `1d1a81b`. Official test access: **NO**. Training, checkpoint reselection, tuning, architecture modification, new corruption generation, condition dropping, and seed dropping: **NONE**.

## Overall decomposition

| n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 54 | 29 | 25 | 0 | 0.537037037037 | 0.051111288732 | 0.001426187267 | -0.020638030245 | 0.455921485824 |

The positive mean does not imply consistent benefit: only 29/54 observations were positive, below the frozen 38/54 H9-F requirement.

## Concentration of the positive aggregate

The six severity-1 vision attenuation and zero-dropout observations contributed **93.8566599011%** of the total summed R_i. The remaining 48 observations had mean R_i **0.003532432833**. At severity 1.0, attenuation zeros the selected representation and is operationally equivalent to zero dropout. Both frozen condition labels remain included; none was removed or reweighted.

## Frozen stratifications

### seed

| seed | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 18 | 13 | 5 | 0 | 0.722222222222 | 0.048980300252 | 0.003496102898 | -0.013043628246 | 0.416682344555 |
| 43 | 18 | 7 | 11 | 0 | 0.388888888889 | 0.048874327682 | -0.002164386392 | -0.019254229701 | 0.422622577389 |
| 44 | 18 | 9 | 9 | 0 | 0.500000000000 | 0.055479238261 | -0.000838148174 | -0.020638030245 | 0.455921485824 |

### corrupted_modality

| corrupted_modality | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 27 | 13 | 14 | 0 | 0.481481481481 | 0.006289615539 | -0.000236793704 | -0.020638030245 | 0.046667498184 |
| vision | 27 | 16 | 11 | 0 | 0.592592592593 | 0.095932961924 | 0.002817673395 | -0.012672987468 | 0.455921485824 |

### corruption_family

| corruption_family | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attenuation | 24 | 14 | 10 | 0 | 0.583333333333 | 0.060426645905 | 0.003551877487 | -0.012672987468 | 0.455921485824 |
| gaussian_noise | 24 | 9 | 15 | 0 | 0.375000000000 | -0.002609519854 | -0.005787787543 | -0.020638030245 | 0.017780053209 |
| zero_dropout | 6 | 6 | 0 | 0 | 1.000000000000 | 0.228733094380 | 0.231674921370 | 0.004286081580 | 0.455921485824 |

### severity

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 12 | 1 | 11 | 0 | 0.083333333333 | -0.005713626380 | -0.005325370828 | -0.012082991803 | 0.000891848302 |
| 0.500000000000 | 12 | 5 | 7 | 0 | 0.416666666667 | -0.000063679072 | -0.002218290215 | -0.012672987468 | 0.020023082698 |
| 0.750000000000 | 12 | 8 | 4 | 0 | 0.666666666667 | 0.008564703418 | 0.005482007117 | -0.010631021232 | 0.039338487698 |
| 1.000000000000 | 18 | 15 | 3 | 0 | 0.833333333333 | 0.151475600885 | 0.026218578748 | -0.020638030245 | 0.455921485824 |

### corrupted_modality_x_corruption_family

| corrupted_modality | corruption_family | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | attenuation | 12 | 8 | 4 | 0 | 0.666666666667 | 0.016063283088 | 0.018381381532 | -0.004178095424 | 0.046667498184 |
| text | gaussian_noise | 12 | 2 | 10 | 0 | 0.166666666667 | -0.008342661334 | -0.008089848712 | -0.020638030245 | 0.003958514559 |
| text | zero_dropout | 3 | 3 | 0 | 0 | 1.000000000000 | 0.025724052838 | 0.026218578748 | 0.004286081580 | 0.046667498184 |
| vision | attenuation | 12 | 6 | 6 | 0 | 0.500000000000 | 0.104790008723 | -0.001184316672 | -0.012672987468 | 0.455921485824 |
| vision | gaussian_noise | 12 | 7 | 5 | 0 | 0.583333333333 | 0.003123621626 | 0.001848926716 | -0.012082991803 | 0.017780053209 |
| vision | zero_dropout | 3 | 3 | 0 | 0 | 1.000000000000 | 0.431742135923 | 0.422622577389 | 0.416682344555 | 0.455921485824 |

### corruption_family_x_severity

| corruption_family | severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attenuation | 0.250000000000 | 6 | 1 | 5 | 0 | 0.166666666667 | -0.004547490743 | -0.003607231751 | -0.011465501377 | 0.000891848302 |
| attenuation | 0.500000000000 | 6 | 3 | 3 | 0 | 0.500000000000 | 0.003745144760 | 0.001290439845 | -0.012672987468 | 0.020023082698 |
| attenuation | 0.750000000000 | 6 | 4 | 2 | 0 | 0.666666666667 | 0.013775835224 | 0.014958174741 | -0.010631021232 | 0.039338487698 |
| attenuation | 1.000000000000 | 6 | 6 | 0 | 0 | 1.000000000000 | 0.228733094380 | 0.231674921370 | 0.004286081580 | 0.455921485824 |
| gaussian_noise | 0.250000000000 | 6 | 0 | 6 | 0 | 0.000000000000 | -0.006879762018 | -0.007286092568 | -0.012082991803 | -0.001068291137 |
| gaussian_noise | 0.500000000000 | 6 | 2 | 4 | 0 | 0.333333333333 | -0.003872502904 | -0.006053831582 | -0.008240421393 | 0.003958514559 |
| gaussian_noise | 0.750000000000 | 6 | 4 | 2 | 0 | 0.666666666667 | 0.003353571611 | 0.002497108735 | -0.008080158520 | 0.017780053209 |
| gaussian_noise | 1.000000000000 | 6 | 3 | 3 | 0 | 0.500000000000 | -0.003039386106 | -0.002664209885 | -0.020638030245 | 0.015068477156 |
| zero_dropout | 1.000000000000 | 6 | 6 | 0 | 0 | 1.000000000000 | 0.228733094380 | 0.231674921370 | 0.004286081580 | 0.455921485824 |

### corrupted_modality_x_severity

| corrupted_modality | severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0.250000000000 | 6 | 0 | 6 | 0 | 0.000000000000 | -0.005201594163 | -0.004142219524 | -0.009448760405 | -0.002340458542 |
| text | 0.500000000000 | 6 | 3 | 3 | 0 | 0.500000000000 | 0.004122408817 | 0.001860860427 | -0.008240421393 | 0.020023082698 |
| text | 0.750000000000 | 6 | 4 | 2 | 0 | 0.666666666667 | 0.012481050465 | 0.011973276359 | -0.008080158520 | 0.039338487698 |
| text | 1.000000000000 | 9 | 6 | 3 | 0 | 0.666666666667 | 0.011267603204 | 0.004286081580 | -0.020638030245 | 0.046667498184 |
| vision | 0.250000000000 | 6 | 1 | 5 | 0 | 0.166666666667 | -0.006225658598 | -0.006814507785 | -0.012082991803 | 0.000891848302 |
| vision | 0.500000000000 | 6 | 2 | 4 | 0 | 0.333333333333 | -0.004249766962 | -0.004651357789 | -0.012672987468 | 0.002817673395 |
| vision | 0.750000000000 | 6 | 4 | 2 | 0 | 0.666666666667 | 0.004648356370 | 0.005482007117 | -0.010631021232 | 0.017780053209 |
| vision | 1.000000000000 | 9 | 9 | 0 | 0 | 1.000000000000 | 0.291683598566 | 0.416682344555 | 0.007715208477 | 0.455921485824 |

## Severity trajectories

### text × attenuation

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 3 | 0 | 3 | 0 | 0.000000000000 | -0.003184974015 | -0.003036368078 | -0.004178095424 | -0.002340458542 |
| 0.500000000000 | 3 | 2 | 1 | 0 | 0.666666666667 | 0.012175323120 | 0.016739680365 | -0.000236793704 | 0.020023082698 |
| 0.750000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.029538730408 | 0.027291677040 | 0.021986026486 | 0.039338487698 |
| 1.000000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.025724052838 | 0.026218578748 | 0.004286081580 | 0.046667498184 |

### text × gaussian_noise

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 3 | 0 | 3 | 0 | 0.000000000000 | -0.007218214311 | -0.008099538905 | -0.009448760405 | -0.004106343624 |
| 0.500000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.003930505485 | -0.007509609621 | -0.008240421393 | 0.003958514559 |
| 0.750000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.004576629477 | -0.007610256144 | -0.008080158520 | 0.001960526232 |
| 1.000000000000 | 3 | 0 | 3 | 0 | 0.000000000000 | -0.017645296064 | -0.019254229701 | -0.020638030245 | -0.013043628246 |

### text × zero_dropout

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.025724052838 | 0.026218578748 | 0.004286081580 | 0.046667498184 |

### vision × attenuation

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.005910007471 | -0.007156369338 | -0.011465501377 | 0.000891848302 |
| 0.500000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.004685033600 | -0.004199786725 | -0.012672987468 | 0.002817673395 |
| 0.750000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.001987059961 | -0.003260481647 | -0.010631021232 | 0.007930322997 |
| 1.000000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.431742135923 | 0.422622577389 | 0.416682344555 | 0.455921485824 |

### vision × gaussian_noise

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.250000000000 | 3 | 0 | 3 | 0 | 0.000000000000 | -0.006541309724 | -0.006472646232 | -0.012082991803 | -0.001068291137 |
| 0.500000000000 | 3 | 1 | 2 | 0 | 0.333333333333 | -0.003814500324 | -0.005102928853 | -0.007004734311 | 0.000664162193 |
| 0.750000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.011283772700 | 0.013037573654 | 0.003033691238 | 0.017780053209 |
| 1.000000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.011566523852 | 0.011915885922 | 0.007715208477 | 0.015068477156 |

### vision × zero_dropout

| severity | n | positive | negative | zero | positive proportion | mean R_i | median R_i | min R_i | max R_i |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000000000 | 3 | 3 | 0 | 0 | 1.000000000000 | 0.431742135923 | 0.422622577389 | 0.416682344555 | 0.455921485824 |

## Ten largest positive observations

| seed | condition | modality | family | severity | M1b | M4qcf | R_i |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 44 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.333333333333 | 0.789254819157 | 0.455921485824 |
| 44 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.333333333333 | 0.789254819157 | 0.455921485824 |
| 43 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.365323136714 | 0.787945714103 | 0.422622577389 |
| 43 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.365323136714 | 0.787945714103 | 0.422622577389 |
| 42 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.375777844396 | 0.792460188951 | 0.416682344555 |
| 42 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.375777844396 | 0.792460188951 | 0.416682344555 |
| 44 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.770053734811 | 0.816721232995 | 0.046667498184 |
| 44 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.770053734811 | 0.816721232995 | 0.046667498184 |
| 44 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.827458911145 | 0.866797398844 | 0.039338487698 |
| 42 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.820698594337 | 0.847990271377 | 0.027291677040 |

## Ten largest negative observations

| seed | condition | modality | family | severity | M1b | M4qcf | R_i |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 44 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.858996474912 | 0.838358444667 | -0.020638030245 |
| 43 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.861906648895 | 0.842652419194 | -0.019254229701 |
| 42 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.853997663963 | 0.840954035716 | -0.013043628246 |
| 44 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.874978871429 | 0.862305883961 | -0.012672987468 |
| 42 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.868000000000 | 0.855917008197 | -0.012082991803 |
| 44 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.877968760003 | 0.866503258625 | -0.011465501377 |
| 44 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.864977181144 | 0.854346159912 | -0.010631021232 |
| 44 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.872989712167 | 0.863540951762 | -0.009448760405 |
| 43 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.871899569262 | 0.863659147870 | -0.008240421393 |
| 42 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.866996674917 | 0.858897136012 | -0.008099538905 |

## All 54 frozen observations

| seed | condition | modality | family | severity | M1b | M4qcf | R_i |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.869997919967 | 0.866961551888 | -0.003036368078 |
| 42 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.850966467455 | 0.870989550154 | 0.020023082698 |
| 42 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.820698594337 | 0.847990271377 | 0.027291677040 |
| 42 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.799710782370 | 0.803996863950 | 0.004286081580 |
| 42 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.866996674917 | 0.858897136012 | -0.008099538905 |
| 42 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.868000000000 | 0.871958514559 | 0.003958514559 |
| 42 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.856998712988 | 0.858959239220 | 0.001960526232 |
| 42 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.853997663963 | 0.840954035716 | -0.013043628246 |
| 42 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.799710782370 | 0.803996863950 | 0.004286081580 |
| 42 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.867997887966 | 0.868889736268 | 0.000891848302 |
| 42 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.864000000000 | 0.866817673395 | 0.002817673395 |
| 42 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.855930270251 | 0.863860593247 | 0.007930322997 |
| 42 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.375777844396 | 0.792460188951 | 0.416682344555 |
| 42 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.868000000000 | 0.855917008197 | -0.012082991803 |
| 42 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.857979549055 | 0.850974814744 | -0.007004734311 |
| 42 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.837958517380 | 0.840992208618 | 0.003033691238 |
| 42 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.820921026173 | 0.835989503328 | 0.015068477156 |
| 42 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.375777844396 | 0.792460188951 | 0.416682344555 |
| 43 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.870931722881 | 0.866753627457 | -0.004178095424 |
| 43 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.866000000000 | 0.865763206296 | -0.000236793704 |
| 43 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.831730769231 | 0.853716795717 | 0.021986026486 |
| 43 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.776710278569 | 0.802928857317 | 0.026218578748 |
| 43 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.873854175427 | 0.869747831802 | -0.004106343624 |
| 43 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.871899569262 | 0.863659147870 | -0.008240421393 |
| 43 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.863893292341 | 0.855813133821 | -0.008080158520 |
| 43 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.861906648895 | 0.842652419194 | -0.019254229701 |
| 43 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.776710278569 | 0.802928857317 | 0.026218578748 |
| 43 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.873886497848 | 0.866730128510 | -0.007156369338 |
| 43 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.870953414183 | 0.866753627457 | -0.004199786725 |
| 43 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.866996674917 | 0.863736193270 | -0.003260481647 |
| 43 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.365323136714 | 0.787945714103 | 0.422622577389 |
| 43 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.865862643347 | 0.864794352210 | -0.001068291137 |
| 43 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.860848463977 | 0.855745535124 | -0.005102928853 |
| 43 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.835852267040 | 0.848889840694 | 0.013037573654 |
| 43 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.839983998400 | 0.851899884322 | 0.011915885922 |
| 43 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.365323136714 | 0.787945714103 | 0.422622577389 |
| 44 | text_attenuation_s0p25 | text | attenuation | 0.250000000000 | 0.870998838990 | 0.868658380448 | -0.002340458542 |
| 44 | text_attenuation_s0p5 | text | attenuation | 0.500000000000 | 0.861944777911 | 0.878684458276 | 0.016739680365 |
| 44 | text_attenuation_s0p75 | text | attenuation | 0.750000000000 | 0.827458911145 | 0.866797398844 | 0.039338487698 |
| 44 | text_attenuation_s1 | text | attenuation | 1.000000000000 | 0.770053734811 | 0.816721232995 | 0.046667498184 |
| 44 | text_gaussian_noise_s0p25 | text | gaussian_noise | 0.250000000000 | 0.872989712167 | 0.863540951762 | -0.009448760405 |
| 44 | text_gaussian_noise_s0p5 | text | gaussian_noise | 0.500000000000 | 0.875987598760 | 0.868477989139 | -0.007509609621 |
| 44 | text_gaussian_noise_s0p75 | text | gaussian_noise | 0.750000000000 | 0.861995031821 | 0.854384775677 | -0.007610256144 |
| 44 | text_gaussian_noise_s1 | text | gaussian_noise | 1.000000000000 | 0.858996474912 | 0.838358444667 | -0.020638030245 |
| 44 | text_zero_dropout_s1 | text | zero_dropout | 1.000000000000 | 0.770053734811 | 0.816721232995 | 0.046667498184 |
| 44 | vision_attenuation_s0p25 | vision | attenuation | 0.250000000000 | 0.877968760003 | 0.866503258625 | -0.011465501377 |
| 44 | vision_attenuation_s0p5 | vision | attenuation | 0.500000000000 | 0.874978871429 | 0.862305883961 | -0.012672987468 |
| 44 | vision_attenuation_s0p75 | vision | attenuation | 0.750000000000 | 0.864977181144 | 0.854346159912 | -0.010631021232 |
| 44 | vision_attenuation_s1 | vision | attenuation | 1.000000000000 | 0.333333333333 | 0.789254819157 | 0.455921485824 |
| 44 | vision_gaussian_noise_s0p25 | vision | gaussian_noise | 0.250000000000 | 0.869981277304 | 0.863508631072 | -0.006472646232 |
| 44 | vision_gaussian_noise_s0p5 | vision | gaussian_noise | 0.500000000000 | 0.852998676988 | 0.853662839181 | 0.000664162193 |
| 44 | vision_gaussian_noise_s0p75 | vision | gaussian_noise | 0.750000000000 | 0.828909493122 | 0.846689546331 | 0.017780053209 |
| 44 | vision_gaussian_noise_s1 | vision | gaussian_noise | 1.000000000000 | 0.826949988547 | 0.834665197024 | 0.007715208477 |
| 44 | vision_zero_dropout_s1 | vision | zero_dropout | 1.000000000000 | 0.333333333333 | 0.789254819157 | 0.455921485824 |

## Exploratory scientific interpretation

### Directly Observed Findings

- H9-F remains NOT_SUPPORTED: 29/54 observations were strictly positive, below the frozen 38/54 coverage requirement.
- The overall mean R_i was positive (0.051111288732), but the median was near zero (0.001426187267) and 25/54 observations were negative.
- Vision corruption had a larger mean R_i than text corruption, but this difference was dominated by severity-1 vision attenuation and zero-dropout observations.
- Text Gaussian noise was unfavorable in 10/12 observations with mean R_i -0.008342661334.
- All six zero-dropout observations were positive, while Gaussian noise was positive in only 9/24 observations.
- Seed-level positive coverage varied from 13/18 for seed 42 to 7/18 for seed 43 and 9/18 for seed 44.

### Descriptive Associations

- Positive coverage generally increased with corruption severity in the pooled descriptive trajectories.
- The six severity-1 vision attenuation/zero-dropout rows contributed 93.8566599011% of the total summed R_i; the other 48 rows had mean R_i 0.003532432833.
- Favorable aggregate means therefore coexist with weak cross-condition consistency.

### Candidate Mechanistic Explanations

- M4qcf may provide its clearest advantage when one modality becomes nearly or completely unusable, while offering inconsistent benefit under mild or distributed perturbation.
- The frozen H9-F failure may reflect concentration of gains in catastrophic missing-modality regimes rather than broadly consistent robustness.
- Text Gaussian perturbations may expose a modality-specific weakness not resolved by the current reliability-informed fusion mechanism.

### Unsupported Explanations

- Step15B does not establish that fusion-weight adaptation caused any Macro-F1 change.
- Step15B does not identify whether errors arise from the quality estimator, compatibility estimator, fusion function, classifier, or representation geometry.
- Step15B does not establish statistical significance or generalization beyond the frozen validation setting.

### Candidate V028 Questions

- Can a prospectively specified mechanism improve mild-corruption consistency without sacrificing catastrophic missing-modality robustness?
- Can text-side Gaussian-noise robustness be improved under a newly frozen v0.28 protocol?
- Should future confirmatory summaries separately report catastrophic modality-loss and graded-noise regimes while retaining an all-condition aggregate?

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

Proceed to **Step15C — H8-to-H9 mechanism characterization**. Step15B performs no architecture or hyperparameter selection.
