# F1 鈥?AEGIS Research Core v1.0 Canonical Architecture

```mermaid
flowchart LR
  I[Multimodal Research Inputs] --> R[Representation / Feature Path]
  R --> F[Multimodal Fusion]
  F --> C[Classifier]
  F --> Q[Quality / Compatibility / Reliability Signals]
  C --> P[Classifier Posterior Context]
  Q --> S[Selector]
  P --> S
  S --> X[Intervention Control]
  C --> E[Evidence Output]
  Q --> E
  S --> E
  X --> E
  E --> A[Analyst Decision-Support Research Output]
```

**Caption.** Canonical functional architecture of the frozen AEGIS Research Core v1.0. Classifier, reliability signals, selector, intervention control, and evidence output are distinct functions.

**Evidence basis.** P1 canonical architecture, component ledger, and dataflow/evidence model.

**Interpretation boundary.** This is a research architecture, not evidence of production readiness, autonomous attribution, or autonomous threat determination.
