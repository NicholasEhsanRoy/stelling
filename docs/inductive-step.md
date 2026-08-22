<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Inductive step verification

Prove that a loop body preserves its invariant in one step — and by
induction, for all steps.

A damped step on `(position, velocity)`, and the invariant is the declared
box itself. This block runs under `tests/test_doc_examples.py` and its output
is compared byte for byte, so the page's headline example is one the tool has
actually verified:

```python
import jax
jax.config.update("jax_enable_x64", True)

from stelling.inductive import check_inductive_step


def loop_body(state, constants):
    x = state["position"]
    v = state["velocity"]
    dt = constants["dt"]
    damping = constants["damping"]
    new_v = damping * v
    new_x = damping * x + new_v * dt
    return {"position": new_x, "velocity": new_v}


verdict = check_inductive_step(
    body=loop_body,
    state_bounds={
        "position": ((-10.0, 10.0), "float64"),
        "velocity": ((-5.0, 5.0), "float64"),
    },
    constants={"dt": 0.01, "damping": 0.99},
    vacuity_mode="inputs-only",
)
print("status:", verdict.status)
print("obligations:", len(verdict.obligations))
```

```
status: VERIFIED
obligations: 4
```

Four obligations: an upper and a lower bound for each of the two state
variables. Interval arithmetic settles all four — no solver was needed.

## What it does

1. Declares each state variable with `any_array` at the invariant bounds
2. Passes them (plus constants) through one step of `body`
3. Asserts each output is within the SAME bounds

If VERIFIED: the invariant is preserved by one step. By induction over
the iteration count, it holds for all steps.

## The body function

The body receives a dict of arrays (one per state variable) and a dict
of constants. It returns a dict of arrays (the new state) with the same
keys as `state_bounds`.

**Both terms of the position update are damped, and that is load-bearing
rather than decorative.** With `new_x = x + new_v * dt` — the same body
without the `damping *` on `x` — one step reaches `10 + 5 × 0.99 × 0.01 =
10.0495` from inside the box, so **no** box on position is preserved and the
tool says so. That case is worked below, under *When the body does not
preserve the invariant*, because a refutation is the more useful half of what
this API does.

## State bounds

Each entry is `((lo, hi), dtype)` for scalar state, or
`((lo, hi), dtype, shape)` for array-valued state:

```python
import jax
jax.config.update("jax_enable_x64", True)

from stelling.inductive import check_inductive_step

state_bounds = {
    "temperature": ((200.0, 5000.0), "float64", (100,)),  # 100-element array
    "pressure": ((0.0, 1e7), "float64"),                   # scalar
}


def relax(state, constants):
    return {"temperature": 0.5 * state["temperature"] + 1000.0,
            "pressure": 0.5 * state["pressure"]}


verdict = check_inductive_step(body=relax, state_bounds=state_bounds,
                               constants={}, vacuity_mode="inputs-only")
print("status     :", verdict.status)
print("obligations:", len(verdict.obligations))
```

```
status     : VERIFIED
obligations: 4
```

All elements of an array-shaped variable share the same bounds (the
invariant is element-wise). For per-element bounds, declare each element
separately. Note the obligation count: **four**, not two hundred and two — a
shaped variable contributes one upper and one lower obligation over the whole
array, exactly as a scalar does.

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

**For a RELATIONAL contradiction this needs `solver_timeout_ms`, and the
section below says why.** The refusal is one solver's `unsat` on one
obligation's script, so it cannot fire when no solver was budgeted. Measured
on the `assume(x < y)` / `assume(y < x)` body below:

| call | result |
|---|---|
| no `solver_timeout_ms` | **UNKNOWN, returns** — the assumes are inert in interval propagation and are reported dropped |
| `solver_timeout_ms=10000` | **raises** `UnsatisfiableAssumptionError` |

A *non*-relational contradiction (`assume(x > 0.9)` and `assume(x < 0.1)`)
raises in both, because interval propagation narrows to the empty set on its
own and needs nobody's `unsat`.

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
3-cycle, **at `solver_timeout_ms=10000`**: VERIFIED, both disclosures
present, note `inductive step CONDITIONAL — NOT the inductive step`, and the
assumed region admits no state at all.

**The budget is part of that measurement, not an incidental.** The same call
with no `solver_timeout_ms` returns UNKNOWN with *none* of the three
disclosures — no `MAY BE VACUOUS`, no `uncertified`, no `CONDITIONAL` — for
the same reason the refusal above cannot fire: every one of them is downstream
of a solver having looked. A reader running these examples with a
default-shaped call sees a different, quieter thing than this section
describes.

Closing the gap needs a whole-query admitted-region script — one emission
naming every assume's operands, which no single obligation slice can — and
that is not in this build.

## Solver escalation

When the interval domain returns UNKNOWN — a common outcome for bodies whose
outputs touch the boundary — pass `solver_timeout_ms` to escalate. Outward
rounding costs up to one ulp per operation, so an invariant whose boundary is
*exactly* met can straddle it by a rounding step the property does not
actually violate. The solver decides those in exact arithmetic.

**But an UNKNOWN that reports a 1-ulp miss is not evidence that rounding is
the cause, and the two blocks below are the same miss with opposite
answers.** Both bodies are an identity written as a round trip, both leave the
interval leg one ulp short of the bound, and one of them genuinely escapes:

```python
import jax
jax.config.update("jax_enable_x64", True)

from stelling.inductive import check_inductive_step

BOUNDS = {"x": ((-1.0, 1.0), "float64")}


def round_trip_by_three(state, constants):
    return {"x": state["x"] / 3.0 * 3.0}


def round_trip_by_a_tenth(state, constants):
    return {"x": state["x"] * 0.1 * 10.0}


for name, body in [("x/3.0*3.0     ", round_trip_by_three),
                   ("x*0.1*10.0    ", round_trip_by_a_tenth)]:
    interval = check_inductive_step(body=body, state_bounds=BOUNDS,
                                    constants={}, vacuity_mode="inputs-only")
    solver = check_inductive_step(body=body, state_bounds=BOUNDS,
                                  constants={}, vacuity_mode="inputs-only",
                                  solver_timeout_ms=5_000)
    miss = next(d.split("misses the bound by ")[1]
                for d in interval.render().split("\n")
                if "misses the bound by " in d)
    print(f"{name} interval={interval.status:8} solver={solver.status:8}")
    print(f"{' ' * len(name)} the interval leg missed by {miss}")
```

```
x/3.0*3.0      interval=UNKNOWN  solver=VERIFIED
               the interval leg missed by 2.220446049250313e-16 (1 ulp step at this magnitude)
x*0.1*10.0     interval=UNKNOWN  solver=REFUTED
               the interval leg missed by 2.220446049250313e-16 (1 ulp step at this magnitude)
```

`x/3·3` is the rounding case the paragraph above describes: in exact reals the
round trip is the identity, so the solver discharges it. `x*0.1*10` is **not**,
and the reason is that `0.1` is a binary64 literal denoting exactly
`0.1000000000000000055511151231257827…`, so `0.1 × 10` is a real number
strictly greater than 1 and the body really does leave `[-1, 1]`. The solver
refutes it and produces the point.

**So: a 1-ulp miss tells you the interval leg could not decide, not why.**
Escalate and read the answer. `alternatively, widen the declared bounds by a
small margin` — advice this page used to give here — turns a REFUTED into an
UNKNOWN on the second body without moving the property at all, which is the
reader landing on the wrong one of these two.

## When the body does not preserve the invariant

The more useful half of this API. Drop the `damping *` from the position
update of the running example and nothing else changes — but one step now
reaches `10 + 5 x 0.99 x 0.01` from inside the box, so **no** box on position
is preserved by it, and widening the bounds cannot help because the escape
scales with them:

```python
import jax
jax.config.update("jax_enable_x64", True)

from stelling.inductive import check_inductive_step


def undamped_position(state, constants):
    """The same body with `damping *` dropped from the position update."""
    x = state["position"]
    v = state["velocity"]
    new_v = constants["damping"] * v
    new_x = x + new_v * constants["dt"]
    return {"position": new_x, "velocity": new_v}


BOUNDS = {"position": ((-10.0, 10.0), "float64"),
          "velocity": ((-5.0, 5.0), "float64")}
CONSTANTS = {"dt": 0.01, "damping": 0.99}

interval = check_inductive_step(body=undamped_position, state_bounds=BOUNDS,
                                constants=CONSTANTS,
                                vacuity_mode="inputs-only")
solver = check_inductive_step(body=undamped_position, state_bounds=BOUNDS,
                              constants=CONSTANTS, vacuity_mode="inputs-only",
                              solver_timeout_ms=5_000)
print("interval:", interval.status)
print("  span  :", next(
    d.split("the operand spans ")[1].split(";")[0]
    for d in interval.render().split("\n") if "the operand spans " in d))
print("  misses:", next(
    d.split("misses the bound by ")[1] for d in interval.render().split("\n")
    if "misses the bound by " in d))
print("solver  :", solver.status)
print("  escape:", next(n.split("Escaped: ")[1] for n in solver.notes
                        if "Escaped: " in n))
print("  one step's reach:", 5.0 * 0.99 * 0.01)
```

```
interval: UNKNOWN
  span  : [-10.0495, 10.0495] and the asserted bound is operand >= -10.0
  misses: 0.0495000000000001
solver  : REFUTED
  escape: position (below lower bound), position (above upper bound)
  one step's reach: 0.0495
```

**Read the two numbers together.** The interval leg missed by `0.0495`, and
one step's reach is `0.0495` — the miss IS the body, not a rounding step, and
it is fourteen orders of magnitude above the `2.2e-16` the section above shows
for a genuine rounding straddle. A reader who takes "outward rounding
introduces 1-ULP imprecision" as the diagnosis here, widens the bounds, and
re-runs gets the same REFUTED at a wider box, which is the tool declining to
be talked out of a real finding.

This page argued the opposite until it was measured: it presented this exact
body as its running example, called the escape 1-ULP straddling, and pointed
at the solver as the remedy — while the solver it points at REFUTES it.

## Constants vs state

Constants (`dt`, `damping`, etc.) are traced as **concrete values** — they
appear as literals in the jaxpr, not as symbolic inputs. The verdict is
about the SPECIFIC constant values you passed. If you need to verify over
a RANGE of parameter values, declare the parameter as a state variable
with its own bounds.
