# AEGIS v1.0 鈥?Related Work Synthesis

## Multimodal fake-news detection

Multimodal fake-news research has progressively moved from combining text and image representations toward explicit modeling of cross-modal relations, event/domain variation, attention, and pretrained vision-language representations. EANN introduced event-adversarial learning for multimodal fake-news detection [R2]. SpotFake provided an early multimodal text-image framework [R3]. SAFE explicitly modeled text-image similarity and mismatch as detection evidence [R4], while co-attention approaches modeled interactions between modalities during fusion [R5]. CLIP-guided work later used pretrained vision-language representations and cross-modal similarity in multimodal fake-news detection [R6].

Fakeddit provides a large multimodal benchmark with fine-grained labeling configurations and is the dataset lineage relevant to the frozen AEGIS experiments [R1]. More recent multi-domain work includes MMDFND, which addresses multimodal multi-domain fake-news detection and forms a direct methodological reference point for the AEGIS research trajectory [R7]. These works contextualize AEGIS; they do not establish protocol-equivalent performance comparisons.

## Reliability, confidence, and selective behavior

Classification accuracy and decision confidence are distinct concerns. Guo et al. showed that modern neural-network confidence can be miscalibrated and evaluated post-hoc calibration methods [R8]. SelectiveNet formalized selective prediction with an integrated reject option and evaluated risk-coverage behavior [R9]. These lines of work support the methodological importance of evaluating selection or intervention behavior independently from ordinary classification performance. They do not imply that the AEGIS selector is equivalent to SelectiveNet or that calibration alone resolves the frozen AEGIS selector-discrimination failure.

## Reproducibility, documentation, and deployment boundaries

Transparent ML reporting frameworks emphasize documenting intended use, evaluation conditions, and limitations [R10]. Dataset datasheets similarly emphasize dataset provenance, composition, collection, and recommended uses [R12]. Production-readiness literature further distinguishes offline research evaluation from the testing and monitoring requirements of deployed ML systems [R11]. These principles are consistent with the AEGIS decision to preserve an immutable research release while explicitly withholding production-readiness claims.

## Information-integrity framing

Broader fake-news scholarship treats detection as a multi-faceted problem involving content, knowledge, propagation, source credibility, and interdisciplinary considerations [R13]. AEGIS occupies a narrower technical research scope: multimodal information-integrity classification, reliability evidence, selector/intervention research, and analyst decision-support characterization. It does not claim to solve the full information-integrity problem.

## Comparative boundary

No statement in this synthesis claims that AEGIS is state of the art, outperforms the cited systems, or has directly comparable benchmark performance. Such a conclusion would require verified equivalence of task definition, split, preprocessing, metric implementation, and evaluation protocol.
