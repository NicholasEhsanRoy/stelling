<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Inductive step verification

Prove that a loop body preserves its invariant in one step — and by
induction, for all steps.

```python
from stelling.inductive import check_inductive_step

v = check_inductive_step(
    body=loop_body,
    state_bounds={
        "position": ((-10.0, 10.0), "float64"),
        "velocity": ((-5.0, 5.0), "float64"),
    },
    constants={"dt": 0.01, "damping": 0.99},
    vacuity_mode="inputs-only",
)
```

## What it does

1. Declares each state variable with `any_array` at the invariant bounds
2. Passes them (plus constants) through one step of `body`
3. Asserts each output is within the SAME bounds

If VERIFIED: the invariant is preserved by one step. By induction over
the iteration count, it holds for all steps.

## The body function

```python
def loop_body(state, constants):
    x = state["position"]
    v = state["velocity"]
    dt = constants["dt"]
    damping = constants["damping"]
    new_v = damping * v
    new_x = x + new_v * dt
    return {"position": new_x, "velocity": new_v}
```

The body receives a dict of arrays (one per state variable) and a dict
of constants. It returns a dict of arrays (the new state) with the same
keys as `state_bounds`.

## State bounds

Each entry is `((lo, hi), dtype)` for scalar state, or
`((lo, hi), dtype, shape)` for array-valued state:

```python
state_bounds = {
    "temperature": ((200.0, 5000.0), "float64", (100,)),  # 100-element array
    "pressure": ((0.0, 1e7), "float64"),                   # scalar
}
```

All elements of an array-shaped variable share the same bounds (the
invariant is element-wise). For per-element bounds, declare each element
separately.

## What VERIFIED means

The invariant is preserved by one step, starting from ANY point in the
declared box. Combined with:
- the initial state being within bounds (checked separately)
- no other code modifying the state between iterations

this gives boundedness for all time.

## What VERIFIED does NOT mean

- **Convergence.** The state stays bounded but may not approach a fixed
  point.
- **Stability.** A bounded orbit can still oscillate.
- **Attractiveness.** States outside the box are not analyzed.
- **Correctness of the initial state.** You must verify separately that
  your starting state satisfies the invariant.

## Solver escalation

When the interval domain returns UNKNOWN (a common outcome for non-trivial
bodies where outputs touch the boundary), pass `solver_timeout_ms` to
escalate:

```python
v = check_inductive_step(
    body=loop_body,
    state_bounds=bounds,
    constants=constants,
    vacuity_mode="inputs-only",
    solver_timeout_ms=5000,
)
```

Outward-rounded interval arithmetic introduces 1-ULP imprecision per
operation, so an invariant whose boundary is exactly met may straddle.
The solver handles these; alternatively, widen the declared bounds by a
small margin.

## Constants vs state

Constants (`dt`, `damping`, etc.) are traced as **concrete values** — they
appear as literals in the jaxpr, not as symbolic inputs. The verdict is
about the SPECIFIC constant values you passed. If you need to verify over
a RANGE of parameter values, declare the parameter as a state variable
with its own bounds.
