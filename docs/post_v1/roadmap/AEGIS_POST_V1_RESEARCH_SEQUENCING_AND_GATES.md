# AEGIS Post-v1 Research Sequencing and Gates

## Sequence

| Phase | Tracks | Entry condition | Exit condition |
|---|---|---|---|
| A | T9 cross-cutting governance | P8 roadmap frozen | Per-track prospective protocol/provenance plan exists |
| B | T1 Vision; T2 multilingual; T4 Audio; T7 selector/control | Track-specific protocol required | Separately authorized implementation/evaluation yields governed evidence or a documented negative/deferred result |
| C | T3 cross-modal; T8 analyst-support integration | T3: participating modality contracts; T8: human-review/provenance contract | Cross-modal or analyst-workflow evidence is traceable and claim-bounded |
| D | T5 Video | Stable visual contract where applicable plus video protocol | Governed video/temporal evidence |
| E | T6 full multimodal integration | Mature constituent modality and cross-modal contracts | Separately governed integration evidence; no automatic production claim |

## Gate G0 鈥?Immutable baseline

Before every future track, verify `v1.0.0-research-core` still resolves to `39492cf6d93987fdf9ebee29b95fdb826b8391e4`.

## Gate G1 鈥?Prospective protocol

Freeze research question, hypotheses, dataset/task/population, modality/language scope, metrics, thresholds, provenance, stopping/decision rules, execution authorization, and official-test boundary as applicable.

## Gate G2 鈥?Implementation authorization

Implementation may begin only after its track-specific protocol authorizes an exact source-file/change boundary. P8 itself grants no implementation authorization.

## Gate G3 鈥?Training authorization

Training requires a separately frozen training protocol, exact data/cache provenance, environment/configuration identity, seed policy, checkpoint rule, and evidence-output contract.

## Gate G4 鈥?Evaluation authorization

Evaluation requires prospectively frozen metrics, thresholds, hypotheses, data split, evidence outputs, and admissibility criteria. `NOT_COMPUTED` must remain distinct from a numerical result.

## Gate G5 鈥?Claim audit

Every result must be classified with an explicit evidence state such as `SUPPORTED`, `NOT_SUPPORTED`, `NOT_COMPUTED`, or another prospectively defined state. Negative findings are retained.

## Gate G6 鈥?Release and integration

A future release receives a new immutable identity. Integration into later tracks requires evidence/contracts that satisfy the later track's dependency gate; completion of one track does not automatically validate another.

## Human-centered boundary

T8 remains analyst decision support unless separately validated and authorized. Production readiness, autonomous attribution, autonomous threat determination, and operational/national-security effectiveness are not inherited from v1.0 or from roadmap inclusion.

## Official-test boundary

Official-test access remains separately governed. P8 performs no official-test access and preserves the current count of `0`.
