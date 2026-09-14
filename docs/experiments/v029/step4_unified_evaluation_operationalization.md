# AEGIS v0.29 Step4A Unified Evaluation Operationalization

Status: **FROZEN BEFORE FORMAL v0.29 UNIFIED EVALUATION**

## Evaluation matrix

- Architectures: M1b, M4qcf, M4qcs-w, M4qcs, M4qgr, M4qtc, M4qgrt
- Seeds: 42, 43, 44
- Validation samples: 1000
- Quality conditions per architecture-seed: 19
- Quality summaries: 399
- Mismatch summaries: 42
- Official test: sealed / 0 samples accessed

## Structured-dropout clarification

The frozen Step1 protocol named `text_structured_dropout_s0p5` and `vision_structured_dropout_s0p5` without defining the tensor transformation. Historical AEGIS records contain no prior implementation. Before any formal v0.29 unified evaluation, the condition is operationalized as deterministic trailing-half feature ablation: retain the first 50% of the selected modality embedding dimensions and set the trailing 50% to zero. The same mask is used for every validation sample, architecture and model seed, with no randomness or renormalization.

This clarification adds no evaluation condition, changes no threshold, and preserves the frozen 19-condition/399-row matrix.

## H4 cardinality clarification

The frozen condition matrix contains three text-Gaussian severities (`0.25`, `0.50`, `0.75`) across three seeds, giving **9** available paired observations per comparator. The numeric frozen gates remain unchanged: mean delta `>=0.01`, positive count `>=9`, and positive fraction `>=0.75`. No fourth Gaussian severity is added after exposure.

## Scientific boundary

No v0.29 formal evaluation or formal hypothesis decision has been performed at this freeze. Checkpoint reselection, threshold tuning, new conditions, dropped seeds/conditions, and official-test access remain prohibited.

## Pre-formal evaluator implementation correction

The first non-evidentiary Step4 smoke stopped before the first v0.29 checkpoint
evaluation because the evaluator compared hexadecimal SHA256 strings
case-sensitively. Step3 stored checkpoint hashes in uppercase, while Python
`hashlib.hexdigest()` emitted lowercase. Independent byte hashing verified that
all nine v0.29 selected checkpoints exactly match the frozen Step3 identities.

Before any formal Step4 evaluation, the evaluator was corrected to compare
SHA256 strings case-insensitively. No checkpoint bytes, models, evaluation
conditions, thresholds, decision gates, seeds, or official-test boundaries were
changed.

Corrected evaluator SHA256: `14E8160637FEDDB738D29AF2C29D01DA5F89E79C76FCFFE257F38BC199B8AFE9`
