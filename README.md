# AEGIS

**Artificial Intelligence for Evaluating and Guarding Information Integrity and Security**

AEGIS is a research framework for multimodal information-integrity analysis and analyst decision support. The frozen **AEGIS Research Core v1.0** packages the project's architecture, scientific evidence, reproducibility records, figures, manuscript, and bounded research-demonstration materials under explicit provenance and claim controls.

> **Research status:** AEGIS v1.0 is a research core. Production readiness, autonomous attribution, autonomous threat determination, and active adaptive robustness are **not established**.

## Frozen Research Core v1.0

The immutable research release is:

- Tag: `v1.0.0-research-core`
- Commit: `39492cf6d93987fdf9ebee29b95fdb826b8391e4`
- [Release manifest](docs/releases/v1.0/AEGIS_RESEARCH_CORE_V1_RELEASE_MANIFEST.json)
- [Release notes](docs/releases/v1.0/AEGIS_RESEARCH_CORE_V1_RELEASE_NOTES.md)

The release identity is the authoritative scientific baseline. Post-v1 documentation and future experiments must not mutate that baseline.

## Research Package

| Area | Repository location | Purpose |
|---|---|---|
| Architecture | [`docs/post_v1/architecture/`](docs/post_v1/architecture/) | Canonical architecture, component boundaries, dataflow, and evidence model |
| Scientific results | [`docs/post_v1/results/`](docs/post_v1/results/) | Frozen results, limitations, negative findings, claims matrix, and experiment timeline |
| Reproducibility | [`docs/post_v1/reproducibility/`](docs/post_v1/reproducibility/) | Environment, workflow, provenance, hashes, and reproducibility checklist |
| Figures | [`docs/post_v1/figures/`](docs/post_v1/figures/) | Claim-safe figure and system-diagram specifications |
| Manuscript | [`docs/post_v1/manuscript/`](docs/post_v1/manuscript/) | Literature-integrated research manuscript and citation traceability |
| Demonstration | [`docs/post_v1/demo/`](docs/post_v1/demo/) | Documentary research demonstration and capability-boundary records |
| Repository presentation | [`docs/post_v1/repository/`](docs/post_v1/repository/) | Public navigation, presentation audits, and artifact map |

## Frozen Scientific Findings

The v1.0 evidence boundary includes both supported research outputs and negative findings.

For the v0.35 corrective mechanism, validation Macro-F1 was `0.8669`, `0.8716`, and `0.8610` for seeds 42, 43, and 44. Selector utility-probability standard deviation was `0.016633`, `0.012812`, and `0.009565`. Against the frozen M3 criterion `utility_probability_std >= 0.05`, **M3 is `NOT_SUPPORTED`**.

Active intervention rate was `0`, `0`, and `0`. Against the frozen M4 criterion `active_intervention_rate > 0`, **M4 is `NOT_SUPPORTED`**.

The v0.35 correction `V035_C1_CLASSIFIER_POSTERIOR_CONTEXT` is **`CORRECTION_NOT_SUPPORTED`**. The subsequent v0.36 active-intervention robustness result is **`NOT_COMPUTED`**, because the selector-activation prerequisite was not satisfied. `NOT_COMPUTED` must not be interpreted as zero robustness or as evidence that the classifier is non-robust.

## Demonstration Boundary

The frozen P6 demonstration package includes documentary walkthroughs for architecture/dataflow, frozen evidence/results, reproducibility/provenance, and failure/limitations.

**D4 bounded live inference is `NOT_INCLUDED`.** The post-v1 audit did not establish the complete frozen inference contract required to include live inference in the demonstration package. This does not mean AEGIS cannot perform inference; it means live inference is outside the frozen P6 demonstration boundary.

No official-test performance is reported. **Official test samples accessed: `0`.**

## Reproducibility

Use the [`docs/post_v1/reproducibility/`](docs/post_v1/reproducibility/) package for the frozen environment, dependency, workflow, provenance, hash, and checklist records.

The package distinguishes release/artifact identity verification and workflow reconstruction from an actual scientific rerun. A new scientific rerun requires separate authorization and must not silently alter the frozen v1.0 evidence boundary.

## Research Positioning

AEGIS v1.0 is characterized as an **analyst decision-support research core**. Its evidence model separates classification, reliability, selector behavior, intervention behavior, and provenance so that failure in one mechanism is not silently converted into a claim about another.

The repository does **not** claim state-of-the-art performance or comparative superiority over prior systems.

## Future Research

AEGIS-Vision, AEGIS-Audio, AEGIS-Video, multilingual/cross-lingual capability, and broader multimodal integration are **post-v1 research directions**, not completed AEGIS v1.0 capabilities.

The dedicated P8 future-research roadmap is governed separately and is not created by P7 repository presentation work.

## Research Manuscript and Evidence

The literature-integrated manuscript and its citation/evidence traceability records are available under [`docs/post_v1/manuscript/`](docs/post_v1/manuscript/). External literature is separated from frozen AEGIS empirical evidence, and related work is not used to overwrite the project's experimental findings.

For a concise map of public research artifacts, see [`docs/post_v1/repository/AEGIS_V1_PUBLIC_RESEARCH_ARTIFACT_MAP.md`](docs/post_v1/repository/AEGIS_V1_PUBLIC_RESEARCH_ARTIFACT_MAP.md).
