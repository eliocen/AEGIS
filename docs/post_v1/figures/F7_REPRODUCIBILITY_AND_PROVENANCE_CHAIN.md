# F7 鈥?Reproducibility and Provenance Chain

```mermaid
flowchart LR
  TAG[Annotated tag<br/>v1.0.0-research-core]
  COM[Release commit<br/>39492cf6...]
  REL[Release manifest + notes]
  EXP[Frozen experiment records<br/>v0.34-v0.38]
  P1[P1 Architecture]
  P2[P2 Results + Limitations]
  P3[P3 Reproducibility]
  VER[Independent Level A identity verification]
  ENV[Level B environment/workflow reconstruction]
  RUN[Level C scientific rerun<br/>separate authorization required]

  TAG --> COM --> REL
  EXP --> REL
  REL --> P1 --> P2 --> P3
  P3 --> VER
  P3 --> ENV
  P3 -. not authorized by P3 .-> RUN
```

**Caption.** Provenance hierarchy from immutable release identity and frozen experiment records through post-v1 documentation and independent verification levels.

**Evidence basis.** P3 reproducibility guide and provenance/hash ledger.

**Interpretation boundary.** Documentation supports identity and workflow reconstruction; it does not guarantee bitwise numerical reproduction.
