# AEGIS v0.32 Step4 鈥?Admissible Characterization Computation Contract

Step3 established the exact evidence boundary. Q1-Q3 cannot be computed from
frozen admissible evidence because their required training-time counterfactual
values were not recorded. They will therefore be reported NOT_AVAILABLE rather
than reconstructed.

Q4 is partial. The available validation diagnostics permit characterization of
final utility dispersion and intervention activation, but not the utility-loss
trajectory, parameter movement, training-time utility trajectory, or gradients.

Q5 is available from the frozen v0.31 validation summaries and will quantify
utility discrimination across corruption severity and mismatch states, including
its association with realized degradation and performance delta.

Q6 is a constrained attribution. Failure categories requiring unavailable
Q1-Q3 evidence cannot be asserted. Selector optimization failure requires the
prospectively frozen Q4 rule. Otherwise attribution remains mixed or
indeterminate.

No training, re-evaluation, threshold search, missing-evidence reconstruction,
or official-test access is permitted.
