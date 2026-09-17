# AEGIS: A Reproducible Multimodal Research Core for Information-Integrity Decision Support with Failure-Aware Adaptive Control

> **P5 Step2 evidence-derived draft.** External literature has not yet been integrated. The Related Work section therefore contains only a controlled placeholder and no fabricated citations.

## Abstract

AEGIS is a multimodal research framework developed to study information-integrity classification together with reliability-aware evidence and adaptive intervention mechanisms under explicit scientific governance. The frozen Research Core v1.0 separates multimodal representation and classification from quality/compatibility/reliability signals, selector behavior, intervention control, and analyst-facing evidence. The research progression prospectively tested whether an adaptive selector could produce sufficiently discriminative utility probabilities and activate intervention behavior. In the v0.35 corrective study, validation Macro-F1 was 0.8669, 0.8716, and 0.8610 for seeds 42, 43, and 44, respectively. However, utility-probability standard deviations were 0.016633, 0.012812, and 0.009565 against the frozen M3 criterion `>= 0.05`, while active-intervention rates were 0, 0, and 0 against the M4 criterion `> 0`. The classifier-posterior-context correction was therefore `CORRECTION_NOT_SUPPORTED`. Because active-selector prerequisites were not satisfied, formal active-intervention robustness evaluation in v0.36 was `NOT_COMPUTED`; this does not establish classifier non-robustness. Subsequent work bounded AEGIS as an analyst decision-support research characterization and completed integration, reproducibility, and immutable release packaging. The resulting contribution is therefore both architectural and methodological: a traceable research core that preserves positive findings, negative findings, blocked evaluations, and claim boundaries rather than converting them into unsupported capability claims.

## 1. Introduction

Information-integrity systems that combine multiple modalities can produce useful classification evidence, but classification performance alone does not establish that a reliability-aware adaptive mechanism behaves as intended. AEGIS was developed around this distinction. Its Research Core treats classification, reliability-related signals, selector behavior, intervention control, and evidence interpretation as separable scientific objects. This separation permits a classifier to show useful validation performance while an adaptive selector simultaneously fails its prospective discrimination or activation criteria.

The v1.0 research programme therefore emphasizes governed evidence rather than a single headline metric. Prospective hypotheses and thresholds were frozen before their corresponding decisions; negative findings were retained; formal evaluation was blocked when prerequisites were not met; and official-test access remained sealed. The frozen release is `v1.0.0-research-core` at commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

The central contribution of the frozen Research Core is a reproducible multimodal research architecture and evidence discipline for studying information-integrity decision support. It does not establish production readiness, autonomous attribution, autonomous threat determination, or formal active-intervention robustness.

## 2. Related Work and Research Context

External literature is intentionally not synthesized in this P5 Step2 draft. A separately traceable literature-review stage is required before citations or comparative claims are introduced. The later review should cover multimodal misinformation and fake-news detection, multimodal representation and fusion, reliability and uncertainty estimation, adaptive intervention/control mechanisms, reproducible machine-learning evaluation, and analyst-facing information-integrity decision support.

## 3. Research Problem and Design Objectives

AEGIS investigates how multimodal classification evidence can be combined with reliability-related signals while keeping adaptive-control claims independently testable. The design objective is not merely to classify an input, but to expose a structured evidence path in which classification, quality/compatibility/reliability assessment, selector output, intervention mapping, and analyst interpretation can be audited separately.

This architecture supports a key scientific distinction: evidence that a classifier optimizes or achieves a validation score is not automatically evidence that a selector discriminates useful intervention states. Consequently, the programme defined separate criteria for selector-output variability and active intervention behavior and treated those criteria as prerequisites for later active-intervention robustness evaluation.

## 4. System Architecture and Methodology

The canonical AEGIS architecture accepts multimodal research inputs, forms modality representations, performs multimodal fusion, and produces classifier outputs. In parallel, the research core derives quality, compatibility, and reliability-related signals. These signals feed a selector path whose output is mapped to an intervention-control signal. Classifier, reliability, selector, and intervention information are retained as distinct evidence for analyst-oriented interpretation.

The v0.35 corrective mechanism augmented the selector's existing seven features with three inference-available, label-free classifier-posterior features: reference positive probability, candidate positive probability, and candidate-minus-reference positive probability. These posterior features were detached before selector use, yielding a ten-feature selector input. Ground-truth class and true-class utility difference were prohibited as selector inference inputs. Training-only `u_target` supervision remained separate and used the preserved binary-cross-entropy-with-logits objective. This two-stage design was intended to address the previously characterized selector information-target mismatch without leaking unavailable ground-truth information into inference.

## 5. Experimental Governance and Evaluation Protocol

AEGIS uses prospective scientific boundaries to prevent later results from redefining the success criteria. For the v0.35 corrective study, formal seeds were 42, 43, and 44. The inherited training configuration used the frozen Fakeddit training and validation embedding caches, a batch size of 32, learning rate 0.001, weight decay 0.0001, shared and hidden dimensions of 128, dropout 0.1, temperature 0.07, utility-loss weight 0.50, gradient clipping at 1.0, patience of eight epochs, and a maximum of 50 epochs. Checkpoint selection used the highest validation Macro-F1 with lowest validation classification loss as the tie-break.

Two prospective activation criteria were central. M3 required `utility_probability_std >= 0.05` for every formal seed. M4 required `active_intervention_rate > 0` for every formal seed. These criteria were not relaxed after observing results. The official test set remained sealed; official test samples accessed were 0.

## 6. Results

Earlier v0.34 evidence supported M1 and M2 but did not support M3 or M4, leading to the frozen `SELECTOR_DISCRIMINATION_FAILURE` characterization. The subsequent v0.35 correction specifically targeted selector information availability through classifier-posterior context.

For seeds 42, 43, and 44, validation Macro-F1 values were 0.8669, 0.8716, and 0.8610. These values describe validation classification context and are not formal robustness measurements or official-test results.

Selector-output evidence did not satisfy M3. Utility-probability standard deviations were 0.016633, 0.012812, and 0.009565, all below the frozen `>= 0.05` criterion. M4 was also not satisfied: active-intervention rates were 0, 0, and 0 against the prospective `> 0` criterion. M3 and M4 therefore remained `NOT_SUPPORTED` for all three formal seeds.

## 7. Failure Analysis and Corrective Mechanism

The v0.34 investigation localized the primary mechanism problem to selector output discrimination rather than treating overall classifier behavior and adaptive-control behavior as interchangeable. Subsequent source and coupling audits characterized a `SELECTOR_INFORMATION_TARGET_MISMATCH`. This motivated the minimal v0.35 correction `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`.

The corrective implementation successfully changed the selector information path from seven to ten features and preserved the intended separation between inference-available posterior context and training-only supervision. Training evidence also showed nonzero selector optimization signal and parameter movement. Nevertheless, optimization did not translate into sufficient selector-output discrimination or active intervention. The final v0.35 disposition was `PERSISTENT_SELECTOR_OUTPUT_DISCRIMINATION_FAILURE_AFTER_POSTERIOR_CONTEXT_CORRECTION`, and the correction result was `CORRECTION_NOT_SUPPORTED`.

This distinction is important: the correction was implemented and optimized, but the prospectively defined behavioral criteria were not met. The negative result is therefore a mechanism-level scientific finding rather than a basis for rewriting the success criterion.

## 8. Robustness Admissibility and Scientific Characterization

v0.36 prospectively required the v0.35 selector activation prerequisites to be satisfied before formal active-intervention robustness evaluation could be considered scientifically admissible. Because M3 and M4 were `NOT_SUPPORTED` across the formal seeds, the robustness benchmark was not computed. The frozen result is `NOT_COMPUTED`, not zero.

The v0.36 evidence additionally characterized an optimization-activation decoupling: selector optimization signal existed, but the mechanism did not produce the required output discrimination and intervention activation. This does not justify a conclusion that the classifier itself is non-robust. Formal active adaptive robustness was not established.

## 9. Threat-Intelligence / Analyst Decision-Support Interpretation

v0.37 translated the accumulated scientific evidence into a bounded threat-intelligence research posture. The supported posture is `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION`. Under this framing, AEGIS evidence can be studied as structured support for analyst interpretation while classification, reliability, selector, and intervention evidence remain distinguishable.

This characterization does not establish an autonomous attribution system, autonomous threat-determination capability, or production deployment capability. Those stronger operational claims are outside the frozen evidence.

## 10. Reproducibility

v0.38 integrated the accumulated evidence into a reproducibility and research-release process. Its R1-R10 governance covered repository identity and cleanliness, evidence provenance, environment and dependencies, command/workflow reconstruction, tests and invariants, architecture/documentation consistency, claims and limitations, release manifest integrity, official-test/data boundaries, and final v1 readiness. The release-readiness process passed without reopening training or formal evaluation.

The immutable release is represented by the annotated tag `v1.0.0-research-core` targeting commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`. The post-v1 reproducibility package distinguishes Level A identity verification, Level B environment/workflow reconstruction, and Level C scientific rerun. Level C requires separate authorization and is not implied by documentary reproducibility.

## 11. Limitations and Negative Findings

The frozen Research Core has explicit limitations. First, the selector failed to meet its prospective output-discrimination threshold across all three formal v0.35 seeds. Second, the active-intervention rate remained zero across those seeds, so active adaptive behavior was not demonstrated. Third, because this prerequisite failed, formal active-intervention robustness was `NOT_COMPUTED`. Fourth, validation Macro-F1 values cannot be generalized into formal robustness or official-test performance.

The research also does not establish production readiness, autonomous attribution, or autonomous threat determination. The official test set was not accessed, with official test samples equal to 0. These boundaries are part of the scientific result rather than implementation details to be omitted from reporting.

## 12. Future Research

Future work is separately governed from the immutable v1.0 Research Core. Planned directions include AEGIS-Vision specialization, AEGIS-Audio, AEGIS-Video, multilingual and cross-lingual expansion, and broader full-multimodal integration. These directions are research roadmap items and are not completed or validated v1.0 capabilities.

A future selector study would also require a separately prospective mechanism hypothesis rather than further ungoverned training of the failed v0.35 correction. Any later robustness evaluation would need to satisfy its own admissibility prerequisites before results could be interpreted as active adaptive robustness.

## 13. Conclusion

AEGIS Research Core v1.0 provides a frozen, reproducible multimodal research framework for studying information-integrity classification, reliability evidence, selector behavior, intervention control, and analyst decision support under explicit scientific governance. The research preserved both supported and unsupported hypotheses. In particular, the classifier-posterior-context correction did not satisfy the prospective selector-discrimination or active-intervention criteria, and formal active-intervention robustness therefore remained not computed.

The resulting contribution is not a claim of production-ready autonomous intelligence. It is a traceable research core in which architecture, experimental decisions, negative findings, reproducibility evidence, and claim boundaries remain inspectable. That boundary defines the scientific baseline from which separately governed future AEGIS research can proceed.
