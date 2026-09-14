# AEGIS v0.31 Step2 鈥?Exact Calibrated-Evidence and Intervention-Utility Contract

Status: **EXACT CONTRACT FROZEN BEFORE IMPLEMENTATION**

## Architecture family

- `M4qcesi-c`: calibration-only
- `M4qcesi-u`: utility-selector-only
- `M4qcesi`: combined calibration + utility selection

## Stable reference

The no-intervention path remains the M4qcs-w-compatible reference:

```text
s_T = (0.10 + q_T_raw_detached)^2
s_V = (0.10 + q_V_raw_detached)^2

alpha_T_ref = s_T / (s_T + s_V)
alpha_V_ref = s_V / (s_T + s_V)

z_ref = alpha_T_ref*h_T + alpha_V_ref*h_V + z_I
```

When the effective intervention gate is zero, the final representation must reproduce this reference within absolute tolerance `1e-7` and relative tolerance `1e-6`.

## Monotone calibration

For `M4qcesi-c` and `M4qcesi`:

```text
x_clip = clamp(x, 1e-4, 1-1e-4)
logit_x = log(x_clip) - log1p(-x_clip)
slope = softplus(a_raw) + 1e-4
x_cal = sigmoid(slope*logit_x + b)
```

Each of text quality, vision quality, and compatibility has its own scalar `(a_raw,b)`. The transform is strictly monotone.

Initialization:

- `a_raw = 0.541324854612918`
- `b = 0`
- approximately identity.

Calibration supervision uses the existing training-only quality/compatibility targets.

Calibration auxiliary-loss weight: **0.25**.

## Risk representation

```text
d_q     = 1 - 0.5*(q_T_star + q_V_star)
d_c     = 1 - c_TV_star
d_delta = abs(q_T_star - q_V_star)

r = clamp(
      0.50*d_c
    + 0.25*d_q
    + 0.25*d_delta,
    0,
    1
)
```

Utility-selector input is:

```text
[q_T_star, q_V_star, c_TV_star, d_q, d_c, d_delta, r]
```

with shape `[B,7]`.

## Utility selector

```text
Linear(7,16)
ReLU
Linear(16,1)
Sigmoid
```

Output:

```text
u_hat in [0,1]
g_u = clamp((u_hat - 0.50)/0.50, 0, 1)
```

The final layer is initialized to zero, giving:

```text
initial u_hat = 0.50
initial g_u   = 0
```

Thus the utility architectures begin from a no-intervention state.

## Training-only utility target

The utility target answers whether the maximum prospectively permitted evidence-directed candidate would improve true-class probability relative to the stable reference.

```text
alpha_T_candidate =
    clamp(alpha_T_ref + 0.15*(q_T_star-q_V_star), 0.10, 0.90)

alpha_V_candidate = 1-alpha_T_candidate

g_I_candidate =
    clamp(1 - 0.50*r, 0.50, 1.00)

z_candidate =
    alpha_T_candidate*h_T
  + alpha_V_candidate*h_V
  + g_I_candidate*z_I
```

Then:

```text
p_ref_true       = P(true class | z_ref)
p_candidate_true = P(true class | z_candidate)

delta_u  = detach(p_candidate_true - p_ref_true)
u_target = sigmoid(delta_u / 0.05)
```

Utility-selector loss:

```text
BCE(u_hat, u_target)
```

Weight: **0.50**.

Only training labels may construct this target. Validation and official-test labels are prohibited.

## Actual bounded intervention

`M4qcesi-c` uses the calibrated-risk gate:

```text
g_c = clamp((r - 0.25)/0.50, 0, 1)
```

`M4qcesi-u` and `M4qcesi` use `g_u`.

Final intervention:

```text
alpha_T =
    clamp(alpha_T_ref + g*0.15*(q_T_star-q_V_star), 0.10, 0.90)

alpha_V = 1-alpha_T

g_I =
    clamp(1 - g*0.50*r, 0.50, 1.00)

z_final =
    alpha_T*h_T
  + alpha_V*h_V
  + g_I*z_I
```

Frozen bounds:

- max modality-weight shift: **0.15**
- modality weight bounds: **[0.10, 0.90]**
- max interaction suppression: **0.50**
- interaction multiplier bounds: **[0.50, 1.00]**

## Active-intervention reporting

```text
active = 1[g >= 0.50]
```

Threshold: **0.50**.

## V31-H5 prospective operationalization

Before any formal result:

- low degradation: `clean Macro-F1 - condition Macro-F1 <= 0.05`
- high degradation: `>= 0.10`
- Criterion A: high-degradation mean utility >= low-degradation mean utility `+0.05`
- low utility: condition mean `u_hat <= 0.40`
- high utility: condition mean `u_hat >= 0.60`
- Criterion B: high-utility mean performance delta vs M4qcs-w >= low-utility mean delta `-0.0025`
- minimum compared-band cardinality: **3**
- insufficient band cardinality => criterion **NOT_SUPPORTED**, with no re-binning.
- V31-H5 SUPPORTED only if A and B both pass.

## Current state

- Implementation: **NO**
- Training: **NO**
- Formal evaluation: **NO**
- V31-H1..H5: **NOT_COMPUTED**
- Official Fakeddit test samples: **0**

Next: **Step3 implementation of this exact frozen contract**.
