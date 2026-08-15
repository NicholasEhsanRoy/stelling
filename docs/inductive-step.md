<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Inductive step verification

Prove that a loop body preserves its invariant in one step — and by
induction, for all steps.

<!-- doc-example: illustrative -->
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

<!-- doc-example: illustrative -->
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

<!-- doc-example: illustrative -->
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

**"ANY point in the declared box" is the load-bearing phrase, and an
`assume` in the body takes it away.** An `assume` is a precondition on the
whole query, so the claim becomes "every state in the ASSUMED SUB-REGION
stays within bounds after one step" — which is not the inductive step, and
does not give boundedness for all time: the successor state need not
re-enter the sub-region, so there is nothing to apply the second step to.

The verdict says which of the two it is. An unconditional VERIFIED appends
a note beginning *"inductive step: all state variables stay within declared
bounds after one iteration"*; a conditional one appends *"inductive step
CONDITIONAL — NOT the inductive step: an assume in the body is a
precondition on the whole query …"*.

State the restriction in `state_bounds` instead. There the successor is
checked against the same set the predecessor was drawn from, which is what
closes the induction.

## What VERIFIED does NOT mean

- **Convergence.** The state stays bounded but may not approach a fixed
  point.
- **Stability.** A bounded orbit can still oscillate.
- **Attractiveness.** States outside the box are not analyzed.
- **Correctness of the initial state.** You must verify separately that
  your starting state satisfies the invariant.
- **The inductive step at all, if the body states an `assume`.** See the
  section above. Measured: body `x -> 1.5 * x` on the invariant `[-1, 1]`
  under `assume(x <= 0.5)` and `assume(x >= -0.5)` is VERIFIED, and
  iterating from the *admitted* start `x = 0.4` gives
  `0.4, 0.6, 0.9, 1.35` — outside the invariant at step 3.

## Contradictory assumes are refused, not verified — when one obligation sees the whole contradiction

If the body's assumes admit no state at all, every obligation over them is
vacuously true and a VERIFIED would mean nothing. That is a harness defect,
so `check_inductive_step` raises
`stelling.propagate.UnsatisfiableAssumptionError` rather than returning a
verdict — the same class, and the same sentence ("harness defect; nothing
was verified"), that an unsatisfiable non-relational assume already raised.

Before that refusal existed (audit 0.2.0 S7', 0.2.0 development builds
only) the body `x, y -> (x + y) * 10` on `[-1, 1]²` under `assume(x < y)`
and `assume(y < x)` returned VERIFIED with "the invariant is preserved by
one step" — and from `x = y = 0.5`, inside the invariant, one step gives
`10.0`.

**What the refusal cannot reach, and what it does instead** (audit B3). The
refusal is one solver's `unsat` on one obligation's script, and a script
states only the assumes whose operands lie in that obligation's backward
cone. Spread the contradiction across cones — three state variables under
`assume(x < y)`, `assume(y < z)`, `assume(z < x)`, where every obligation
depends on at most two of them — and no script ever holds more than one link
of the cycle. Nothing proves the region empty, and the call RETURNS.

What it no longer does is return a *clean* verdict. A discharge that rested
on a partial axiom set carries `[MAY BE VACUOUS: …]` on its own detail line
and a stamped `precondition satisfiability uncertified`, so a reader is told
the claim was never shown to be about anything. Measured on this build, body
`{x, y, z} -> {0.6·(x - y) + 0.6, 0.5·y, 0.5·z}` on `[-1, 1]³` under that
3-cycle: VERIFIED, both disclosures present, note `inductive step
CONDITIONAL — NOT the inductive step`, and the assumed region admits no
state at all.

Closing the gap needs a whole-query admitted-region script — one emission
naming every assume's operands, which no single obligation slice can — and
that is not in this build.

## Solver escalation

When the interval domain returns UNKNOWN (a common outcome for non-trivial
bodies where outputs touch the boundary), pass `solver_timeout_ms` to
escalate:

<!-- doc-example: illustrative -->
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
