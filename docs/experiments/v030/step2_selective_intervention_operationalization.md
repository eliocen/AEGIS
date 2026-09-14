# AEGIS v0.30 Step2A ??Selective-Intervention Tensor Contract

Status: **FROZEN BEFORE IMPLEMENTATION AND TRAINING**

## Stable reference

The no-intervention path is the exact M4qcs-w-compatible allocation:

- `s_T = (0.10 + q_T)^2`
- `s_V = (0.10 + q_V)^2`
- `alpha_T_ref = s_T / (s_T + s_V)`
- `alpha_V_ref = s_V / (s_T + s_V)`
- reference interaction multiplier = `1`

## Evidence risk

- `d_q = 1 - 0.5*(q_T + q_V)`
- `d_c = 1 - c_TV`
- `d_delta = abs(q_T - q_V)`
- `r = clamp(0.50*d_c + 0.25*d_q + 0.25*d_delta, 0, 1)`
- `g = clamp((r - 0.25)/0.50, 0, 1)`

All inputs are detached before controller computation.

## Weight intervention

For `M4qesri-w` and `M4qesri`:

`alpha_T = clamp(alpha_T_ref + g*0.15*(q_T-q_V), 0.10, 0.90)`

`alpha_V = 1 - alpha_T`

For `M4qesri-t`, the exact reference weights are retained.

## Transition intervention

For `M4qesri-t` and `M4qesri`:

`g_I = clamp(1 - g*0.50*r, 0.50, 1.00)`

For `M4qesri-w`, `g_I = 1`.

## Fusion

`z = alpha_T*h_T + alpha_V*h_V + g_I*z_I`

When `g = 0`, the output must exactly reproduce the stable reference path.

## Boundary

This Step2A record does not authorize formal training, formal evaluation,
hypothesis decisions, validation-driven threshold tuning, or official-test access.
