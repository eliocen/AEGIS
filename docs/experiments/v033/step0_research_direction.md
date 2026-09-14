# AEGIS v0.33 Step0 鈥?Research Direction

## Direction

**Prospective Utility-Supervision and Selector-Learning Redesign**

v0.32 established that the v0.31 utility selector did not learn a sufficiently
variable or discriminative signal, while also establishing that the upstream
cause could not be identified because key training-time diagnostics were never
recorded.

v0.33 therefore begins with observability and mechanism verification rather
than immediate architecture changes or training.

## Objective

Build a utility-learning pathway whose counterfactual supervision, target
construction, selector optimization, and intervention activation are all
directly observable and verifiable before formal training.

## Boundary

No v0.31/v0.32 reopening, no official-test access, no AEGIS-Vision expansion,
and no implementation or training at Step0.
