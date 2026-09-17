# AEGIS Post-v1 Research Roadmap

## Governance status

This roadmap is documentary. It does **not** authorize implementation, training, evaluation, official-test access, deployment, or expansion of the frozen AEGIS Research Core v1.0.

The immutable baseline remains `v1.0.0-research-core` at commit `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

## Scientific inheritance

Future AEGIS research begins from the frozen record rather than from an assumed successful adaptive-control system:

- M3: `NOT_SUPPORTED`
- M4: `NOT_SUPPORTED`
- v0.35 correction: `CORRECTION_NOT_SUPPORTED`
- v0.36 active-intervention robustness: `NOT_COMPUTED`
- P6 D4 bounded live inference: `NOT_INCLUDED`
- production readiness: `NOT_ESTABLISHED`
- autonomous attribution: `NOT_ESTABLISHED`
- official test samples accessed: `0`

## Roadmap structure

The sequence below is dependency-aware. It is not a claim that all tracks will be implemented, nor is it an authorization to start them.

### Phase A 鈥?Cross-cutting governance foundation

**T9 鈥?Reproducibility, evaluation, and release governance**

State: `PROPOSED`

T9 is cross-cutting rather than a standalone modeling milestone. Each future experimental lineage should instantiate prospective protocols, source/evidence hashes, environment records, dataset/cache boundaries, sealed-test rules, claim ledgers, negative-result preservation, and immutable release identities.

**Exit gate:** the selected future track has its own prospective protocol and provenance/evaluation plan before implementation begins.

### Phase B 鈥?Early modality and mechanism research

**T1 鈥?AEGIS-Vision**

State: `PROTOCOL_REQUIRED`

Research direction: image authenticity, manipulated-image/deepfake detection research, visual evidence quality, and image-specific information-integrity analysis.

Before implementation: freeze the image task and population, dataset provenance, authenticity taxonomy, visual evidence contract, metrics, thresholds, hypotheses, and evaluation authorization.

**T2 鈥?Multilingual and cross-lingual information integrity**

State: `PROTOCOL_REQUIRED`

Research direction: explicitly governed expansion beyond the frozen v1 language/task/data boundary.

Before implementation: freeze languages, populations, multilingual dataset provenance, taxonomy/label equivalence, cross-lingual hypotheses, metrics, and language-specific failure analysis.

**T4 鈥?AEGIS-Audio**

State: `PROTOCOL_REQUIRED`

Research direction: audio authenticity and audio information-integrity research.

Before implementation: freeze audio task/taxonomy, dataset provenance, preprocessing/representation contract, authenticity metrics, thresholds, and evaluation rules.

**T7 鈥?Reliability, selector, and adaptive-control research**

State: `PROTOCOL_REQUIRED`

Research direction: a new mechanistic line addressing the unresolved selector output-discrimination/activation problem.

T7 must explicitly inherit the v1 negative findings. It may not treat prior selector training as evidence of active adaptive robustness, and it may not reinterpret selector failure as classifier non-robustness.

Before implementation: freeze a new mechanistic hypothesis, selector information contract, supervision target, leakage constraints, discrimination criterion, activation criterion, stopping rules, and evaluation authorization.

### Phase C 鈥?Evidence relations and analyst-facing integration

**T3 鈥?Cross-modal evidence and consistency**

State: `PROTOCOL_REQUIRED`

Research direction: consistency, contradiction, provenance, and evidence fusion across modality streams.

Dependency gate: only modalities participating in a T3 experiment need stable constituent contracts first. T3 must define the paired evidence provenance, relation taxonomy, consistency metrics, and failure-aware fusion logic prospectively.

**T8 鈥?Threat-intelligence and analyst decision-support integration**

State: `PROTOCOL_REQUIRED`

Research direction: analyst-facing evidence integration, provenance, case/audit semantics, and decision support.

T8 remains human-centered. It must not silently become autonomous attribution, autonomous threat determination, production deployment, or national-security effectiveness claims. Any broader operational claim requires separately governed validation and authorization.

### Phase D 鈥?Video research

**T5 鈥?AEGIS-Video**

State: `PROTOCOL_REQUIRED`

Research direction: video authenticity, temporal evidence, and multimodal video information-integrity research.

Dependency gate: inherit a stable visual evidence contract from T1 where scientifically applicable; freeze temporal/video task definition, dataset provenance, modality-relation rules, metrics, thresholds, and evaluation protocol.

### Phase E 鈥?Full multimodal integration

**T6 鈥?Full multimodal integration**

State: `DEFERRED`

T6 is deliberately downstream. It should not begin as a monolithic integration exercise.

Entry requires sufficiently mature, separately evidenced constituent contracts from T1/T3/T4/T5 for the modalities being integrated, plus an integration protocol that freezes fusion semantics, failure decomposition, provenance, metrics, thresholds, and admissible claims.

`DEFERRED` is a roadmap state, not a negative scientific result.

## Parallelism rule

The roadmap permits parallel **protocol development** where dependencies allow. It does not authorize parallel implementation or evaluation.

T1, T2, T4, and T7 are separable early research lines. T9 governance applies to all of them. T3 becomes admissible for a given modality pair only when those constituent contracts are sufficiently specified. T8 can develop documentary analyst-workflow requirements independently, but empirical operational claims remain separately gated.

## Future release rule

Any post-v1 scientific result belongs to a new experimental lineage and, if released, a new release identity. No future roadmap item may mutate or retrospectively expand `v1.0.0-research-core`.
