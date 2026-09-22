# AEGIS T1 Step4 鈥?Implementation Authorization

Parent Step3 closure: `6e90ac896c8aa490d2a1ebe36e6ba131afde016a`.

## Disposition
**IMPLEMENTATION_AUTHORIZED_WITHOUT_TRAINING**

Step3 has prospectively frozen the SD1.4 population, B0/B1/A1 architecture, DINO ViT-S/16 backbone and weight identity, reliability target/loss, transforms, optimization configuration, checkpoint rule, compute envelope and dependency environment. Step4 therefore authorizes bounded implementation of that design.

Authorized implementation includes the frozen usable-population loader; RGB transform pipeline and metadata-exclusion boundary; B0, B1 and A1 modules/trainability masks; authenticity and reliability target/loss plumbing; optimizer/scheduler/checkpoint configuration without executing training; metrics/observability plumbing; configuration; and unit/integration tests. Local pretrained weights must be protected from Git by an explicit ignore rule or equivalent verified repository rule.

Implementation-time tests may instantiate modules and load the cryptographically verified pretrained weight where needed to verify architecture compatibility, shapes, trainability masks, transforms and loss plumbing. They may use synthetic or tightly bounded non-training samples. They must not perform optimizer steps on dataset samples, conduct training, perform formal evaluation, or access the official test.

The raw GenImage dataset remains immutable. Path, filename, extension, image format, file size and class-directory tokens are prohibited model features. Reliability supervision must use the detached authenticity-correctness target and, under the frozen default design, reliability loss must not backpropagate into the authenticity path.

Any incompatibility that requires changing the frozen scientific design must stop implementation and return to prospective governance. Step5 is implementation only; training remains unauthorized.
