# F3 鈥?Reliability, Selector, and Intervention Control Path

```mermaid
flowchart LR
  B[7 existing selector features] --> AUG[Selector feature augmentation]
  RP[Reference positive probability] --> AUG
  CP[Candidate positive probability] --> AUG
  DP[Candidate - reference probability] --> AUG
  AUG --> TEN[10-feature selector input]
  TEN --> SEL[Selector]
  UT[u_target supervision during training only] -. BCEWithLogits .-> SEL
  SEL --> PROB[Utility / selector probability]
  PROB --> M3{M3: std >= 0.05?}
  PROB --> MAP[Intervention mapping]
  MAP --> M4{M4: active rate > 0?}
  M3 -->|all formal seeds: no| NS1[NOT_SUPPORTED]
  M4 -->|all formal seeds: no| NS2[NOT_SUPPORTED]
```

**Caption.** v0.35 classifier-posterior-context correction and its prospective activation criteria. Ground-truth class and true-class delta-u were prohibited as selector inference inputs.

**Evidence basis.** Frozen v0.35 design, implementation, protocol, training evidence, and closure.

**Interpretation boundary.** Optimization signal existed, but selector output discrimination and active intervention criteria were not satisfied.
