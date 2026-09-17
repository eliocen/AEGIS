# AEGIS Post-v1 鈥?Research Prerequisite Matrix

| Track | Protocol before implementation | New data/task contract | New metrics/thresholds | Constituent-track dependency | Human-review boundary | Separate evaluation authorization |
|---|---:|---:|---:|---:|---:|---:|
| T1 AEGIS-Vision | Required | Required | Required | No | As applicable | Required |
| T2 Multilingual/cross-lingual | Required | Required | Required | No | As applicable | Required |
| T3 Cross-modal evidence | Required | Required | Required | Yes | As applicable | Required |
| T4 AEGIS-Audio | Required | Required | Required | No | As applicable | Required |
| T5 AEGIS-Video | Required | Required | Required | Visual contract where applicable | As applicable | Required |
| T6 Full multimodal integration | Required | Required | Required | T1/T3/T4/T5 | As applicable | Required |
| T7 Reliability/selector/control | Required | As applicable | Required | Inherits v1 negative evidence | As applicable | Required |
| T8 Threat-intelligence/analyst support | Required | Required for operational validation | Required | Evidence/provenance layers | Required | Required |
| T9 Reproducibility/release governance | Per-track governance | Per track | Per track | Cross-cutting | Preserve relevant boundary | Per track |

## Global gates

No track receives implementation or evaluation authorization from this matrix itself.

Every future scientific claim must be tied to a separately frozen experimental lineage. Future releases must not mutate the immutable AEGIS Research Core v1.0 release.

Production readiness, autonomous attribution, and active adaptive robustness are not inherited capabilities. They require their own admissible evidence if ever investigated.
