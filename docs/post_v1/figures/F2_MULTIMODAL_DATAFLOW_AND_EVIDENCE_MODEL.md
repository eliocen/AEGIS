# F2 鈥?Multimodal Dataflow and Evidence Model

```mermaid
flowchart TD
  D[Input sample] --> M1[Modality representations]
  M1 --> FU[Fusion representation]
  FU --> CL[Classification path]
  FU --> REL[Reliability / quality / compatibility path]
  CL --> PC[Detached classifier-posterior context]
  REL --> SEL[Selector path]
  PC --> SEL
  SEL --> IC[Intervention-control signal]
  CL --> EB[Evidence bundle]
  REL --> EB
  SEL --> EB
  IC --> EB
  EB --> DS[Analyst decision-support interpretation]
```

**Caption.** Frozen research dataflow separating classification evidence from reliability, selector, and intervention-control evidence.

**Evidence basis.** P1 dataflow and evidence model plus the frozen v0.35 two-stage posterior-context correction.

**Interpretation boundary.** Evidence aggregation supports analyst interpretation; it does not establish autonomous attribution.
