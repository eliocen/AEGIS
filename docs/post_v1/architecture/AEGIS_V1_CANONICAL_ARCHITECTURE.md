# AEGIS Research Core v1.0 鈥?Canonical Architecture

## 1. System purpose and scope

AEGIS (Artificial Intelligence for Evaluating and Guarding Information Integrity and Security) Research Core v1.0 is a multimodal research framework for studying information-integrity classification together with evidence-aware reliability and adaptive-control mechanisms. The frozen v1.0 release is a reproducible, scientifically bounded research core. It is not a production threat-intelligence platform, autonomous attribution system, or production alerting/deployment certification.

The implemented research path combines multimodal representations, fused classification, quality/compatibility signals, utility supervision, a learned selector, and an intervention path. These components must be distinguished from later planned AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual expansion, and broader-system integration, which are future work rather than v1.0 capabilities.

## 2. Input modalities and representations

The frozen research core operates on the modalities and representations implemented in the tracked v1.0 code and frozen experimental workflows. Multimodal evidence is encoded before fusion; the architecture separates representation construction from downstream classification and reliability/control logic.

The post-v1 architecture documentation does not infer support for modalities that were only proposed in the broader AEGIS roadmap. Image-authenticity/deepfake specialization, audio analysis, video analysis, multilingual/cross-lingual expansion, and full broader-system integration remain outside the v1.0 evidence boundary unless separately established by future research.

## 3. Encoder and fusion path

AEGIS uses modality-specific encoded representations and a fusion path to construct a joint representation for downstream prediction. The fused representation is the principal input to the classifier and is also the basis from which reliability-related signals are derived.

The architecture is intentionally described as a research pipeline rather than a single monolithic classifier: classification, quality/compatibility estimation, reliability control, selector behavior, and intervention behavior have distinct evidentiary statuses.

## 4. Classifier path

The classifier maps the fused representation to predictive logits/probabilities for the task under study. Frozen validation performance may describe classifier behavior within the relevant dataset/protocol, but validation performance is not formal evidence of robustness and must not be presented as such.

The failure of the learned selector in v0.35 does not establish classifier non-robustness. The frozen v0.36 record explicitly preserves that distinction.

## 5. Quality and compatibility path

The research architecture includes quality and compatibility signals intended to characterize aspects of multimodal evidence and cross-modal consistency. These signals participate in the reliability/control research path but do not independently establish operational truth, provenance, attribution, or threat severity.

## 6. Reliability controller

The reliability controller is the architectural layer that combines reliability-relevant evidence used by the adaptive-control research mechanism. It is conceptually downstream of encoded/fused evidence and upstream of learned selection/intervention behavior.

Its presence in the implementation is not equivalent to evidence that active adaptive intervention is scientifically supported. The v1.0 documentation therefore separates implementation status from hypothesis/evaluation status.

## 7. Utility supervision

AEGIS v0.33鈥搗0.35 introduced utility-supervised selector research. The selector training path uses a utility target as supervision, while ground-truth class information and true-class utility are prohibited as selector inference inputs.

The v0.35 corrective design retained BCEWithLogits utility supervision and an effective utility-loss weight of 0.50. These are frozen experimental design facts, not post-v1 tuning decisions.

## 8. Selector path

The learned selector is intended to produce an intervention-control signal from inference-available evidence. The v0.35 correction, `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT`, augmented the existing selector features with detached, label-free classifier-posterior context: reference positive probability, candidate positive probability, and their difference.

This correction was implemented and trained under the frozen protocol, but the correction result is `CORRECTION_NOT_SUPPORTED`. Across formal seeds, the frozen M3 selector-output discrimination criterion and M4 active-intervention criterion remained `NOT_SUPPORTED`.

The constrained mechanistic disposition is `PERSISTENT_SELECTOR_OUTPUT_DISCRIMINATION_FAILURE_AFTER_POSTERIOR_CONTEXT_CORRECTION`.

## 9. Intervention path

The intervention path consumes selector/control behavior to determine learned adaptive intervention under the experimental architecture. Because the selector failed the frozen activation prerequisites, formal active-intervention robustness evaluation was inadmissible in v0.36.

Accordingly, active-intervention robustness is `NOT_COMPUTED`, not a negative robustness result. Active adaptive robustness is not claimed by v1.0.

## 10. Training dataflow

The frozen corrective-mechanism training design used a two-stage flow:

1. Stage A constructs deterministic reference/candidate fused embeddings.
2. Stage B reuses the classifier to compute detached posterior context, finalizes selector inputs, and applies the learned intervention path.

Utility-target supervision is available to the training objective but is not an inference input. Ground-truth class leakage into the selector inference path is prohibited.

The v0.35 formal corrective training used seeds 42, 43, and 44 under the frozen configuration. Post-v1 packaging performs no retraining.

## 11. Inference dataflow

At inference, encoded multimodal evidence flows through fusion and classification. Inference-available quality/compatibility and detached classifier-posterior context may feed the reliability/selector path as implemented by the frozen architecture.

Ground-truth labels and true-class utility are not legitimate selector inference inputs. The architecture documentation must preserve this separation when figures or demonstrations are produced.

## 12. Evidence and provenance flow

AEGIS scientific claims are controlled by frozen experiment records, prospective protocols, repository identity, artifact hashes, seed/configuration records, hypothesis decisions, and release provenance.

The v1.0 release is anchored by the immutable tag `v1.0.0-research-core`. Post-v1 documentation is downstream of that release and may summarize or visualize frozen evidence but may not alter its scientific conclusions.

The official-test boundary remains sealed: official-test samples accessed are **0**. Post-v1 packaging does not authorize official-test access.

## 13. Threat-intelligence interpretation boundary

v0.37 established an `ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION`. AEGIS evidence may therefore be documented as research evidence that can support analyst interpretation within the studied scope.

It must not be represented as autonomous attribution, autonomous threat determination, production alerting, or a validated national-security operational capability. Evidence-to-operational-signal mappings that were characterized only as context remain context rather than evaluated capability.

## 14. Scientific limitations

The principal frozen limitation relevant to the adaptive-control path is persistent selector-output discrimination failure after the posterior-context correction. Optimization signal existed, but the prospectively defined selector discrimination and active-intervention criteria were not satisfied.

Consequently, formal active-intervention robustness was not computed. This prevents a valid claim of active adaptive robustness, while also preventing the selector failure from being generalized into a claim that the underlying classifier is non-robust.

The v1.0 evidence remains dataset-, task-, protocol-, and population-bounded. Generalization beyond those boundaries requires separate prospective research.

## 15. Non-capabilities and prohibited claims

AEGIS Research Core v1.0 does not establish:

- production deployment readiness;
- autonomous attribution;
- autonomous threat determination;
- production alerting;
- formally demonstrated active adaptive robustness;
- classifier non-robustness from selector failure;
- AEGIS-Vision, AEGIS-Audio, or AEGIS-Video as completed v1.0 capabilities;
- multilingual/full cross-lingual AEGIS as a completed v1.0 capability;
- broader operational-system integration as a validated v1.0 capability.

These boundaries are architectural documentation requirements, not optional presentation language.
