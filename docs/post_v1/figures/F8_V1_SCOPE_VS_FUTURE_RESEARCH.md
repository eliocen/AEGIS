# F8 鈥?AEGIS v1.0 Scope versus Future Research

```mermaid
flowchart LR
  subgraph CORE["Frozen AEGIS Research Core v1.0"]
    A[Multimodal research pipeline]
    B[Classification]
    C[Reliability / quality / compatibility]
    D[Selector + intervention research mechanism]
    E[Evidence / analyst decision-support characterization]
    A --> B
    A --> C
    C --> D
    B --> E
    C --> E
    D --> E
  end

  subgraph FUT["Separately governed future research"]
    V[AEGIS-Vision specialization]
    AU[AEGIS-Audio]
    VI[AEGIS-Video]
    ML[Multilingual / cross-lingual expansion]
    FM[Broader full-multimodal integration]
  end

  CORE -. research roadmap boundary .-> FUT
```

**Caption.** Separation between the immutable v1.0 Research Core and separately governed future research directions.

**Evidence basis.** P1鈥揚3 scope boundaries and the post-v1 packaging protocol.

**Interpretation boundary.** Future modules are roadmap directions only and must not be represented as completed, validated, or released v1.0 capabilities.
