# AEGIS Research Core v1.0 鈥?Dataflow and Evidence Model

## 1. Canonical system flow

```text
INPUT EVIDENCE
      |
      v
MODALITY-SPECIFIC REPRESENTATIONS
      |
      v
FUSION / SHARED REPRESENTATION
      |
      +--------------------+
      |                    |
      v                    v
CLASSIFIER            QUALITY / COMPATIBILITY
      |                    |
      |                    +----------+
      |                               |
      v                               v
PREDICTIVE LOGITS /            RELIABILITY-RELEVANT
PROBABILITIES                  EVIDENCE
      |                               |
      +---- detached posterior -------+
                  context             |
                                      v
                              RELIABILITY CONTROLLER
                                      |
                                      v
                               LEARNED SELECTOR
                                      |
                                      v
                               INTERVENTION PATH
```

This diagram is architectural. It does not imply that every implemented path is scientifically supported.

## 2. Corrective-selector two-stage flow

```text
STAGE A
reference evidence ----> encoded/fused reference representation
candidate evidence ----> encoded/fused candidate representation

STAGE B
reference representation --> classifier --> detached reference posterior
candidate representation --> classifier --> detached candidate posterior

existing selector evidence
        +
detached posterior context
        |
        v
final selector input
        |
        v
selector logit/probability
        |
        v
learned intervention mapping
```

The v0.35 correction expanded selector context with inference-available, label-free classifier-posterior information. Ground-truth class and true-class utility are prohibited from the selector inference input.

## 3. Training-only supervision boundary

```text
FROZEN TRAINING EVIDENCE
        |
        +--> classification objective
        |
        +--> quality / compatibility objectives
        |
        +--> utility target --> BCEWithLogits selector supervision
                                  |
                                  v
                           selector parameters

GROUND-TRUTH / UTILITY TARGET
        X
        X----> NOT A SELECTOR INFERENCE INPUT
```

Utility supervision may train selector parameters without becoming an inference feature.

## 4. Scientific evidence flow

```text
PROSPECTIVE PROTOCOL
        |
        v
AUTHORIZED IMPLEMENTATION / EXPERIMENT
        |
        v
FROZEN CONFIG + SEEDS + DATA BOUNDARY
        |
        v
EVIDENCE ARTIFACTS + HASHES
        |
        v
PREDEFINED CRITERIA
        |
        v
SUPPORTED / NOT_SUPPORTED / NOT_COMPUTED
        |
        v
SCIENTIFIC CLOSURE
        |
        v
RELEASE PROVENANCE
        |
        v
POST-v1 DOCUMENTATION
```

Post-v1 packaging is the final downstream consumer in this chain. It cannot feed backward and change a frozen criterion or result.

## 5. v0.35 鈫?v0.36 decision boundary

```text
v0.35 corrective training
        |
        +--> M3 selector discrimination: NOT_SUPPORTED
        |
        +--> M4 active intervention:      NOT_SUPPORTED
        |
        v
selector activation prerequisite fails
        |
        v
v0.36 formal active-intervention robustness
        |
        v
NOT COMPUTED / INADMISSIBLE
```

This is not equivalent to a finding that the classifier is non-robust.

## 6. Analyst interpretation boundary

```text
AEGIS RESEARCH EVIDENCE
        |
        v
BOUNDED MODEL / RELIABILITY SIGNALS
        |
        v
ANALYST DECISION-SUPPORT RESEARCH CHARACTERIZATION
        |
        +--> contextual interpretation
        +--> failure localization
        +--> evidence review

        X--> autonomous attribution
        X--> autonomous threat determination
        X--> production alerting
        X--> production deployment certification
```

## 7. Future-system boundary

AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual expansion, full multimodal integration, and broader-system integration must attach prospectively outside the immutable v1.0 release. They may reuse v1.0 as a baseline but may not be drawn inside the v1.0 implemented/evaluated boundary until separately established.
