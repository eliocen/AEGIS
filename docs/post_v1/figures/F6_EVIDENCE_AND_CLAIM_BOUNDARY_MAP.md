# F6 鈥?Evidence and Claim Boundary Map

```mermaid
flowchart TD
  E[Frozen AEGIS evidence] --> S[SUPPORTED<br/>criterion satisfied]
  E --> N[NOT_SUPPORTED<br/>criterion tested but not satisfied]
  E --> NC[NOT_COMPUTED<br/>measurement not performed/admissible]
  E --> CH[CHARACTERIZED<br/>bounded descriptive evidence]
  E --> FW[FUTURE_WORK<br/>outside v1.0]
  N -. does not imply .-> DIS[Disproven]
  NC -. must not become .-> ZERO[Numeric zero / negative robustness score]
  CH -. does not imply .-> PROD[Production readiness]
  FW -. must not appear as .-> V1[Completed v1.0 capability]
```

**Caption.** Claim semantics governing AEGIS post-v1 communication.

**Evidence basis.** P2 claims matrix, scientific results ledger, and limitations/negative findings.

**Interpretation boundary.** No figure may infer classifier non-robustness from selector failure, or production/autonomous capability from research characterization.
