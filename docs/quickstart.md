<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Quickstart

Ten minutes, four runnable files. At the end you will have a stamped
verdict and know which parts of it you have earned.

Every code block on this page was executed verbatim against this tree
(stelling 0.2.0.dev0, jax 0.11.0, CPU, `jax_enable_x64=True`) and the outputs
are what it printed — `tests/test_doc_examples.py` re-runs them and
compares. The `at …` source line is the one part that differs on your
machine: the **directory** shown below is a placeholder, and the file
name is whatever you saved the block as. The line number and the
function name are not placeholders.

## Install

```sh
pip install stelling              # into the environment that already has your JAX
pip install stelling[solvers]     # optional: adds the SMT step (never touches JAX)
```

Stelling needs a JAX in the environment to trace a harness; it never
installs or moves one. `python -m stelling` reports what it can see.
`[solvers]` installs both SMT backends, which is what the escalation
portfolio is designed around; if you are considering only one, see
[choosing a solver backend](choosing-a-solver-backend.md).

## 1. The smallest harness that says something

A **harness** is a zero-argument function that declares its inputs, states
its obligations, and returns them. Save this as `quickstart.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    # ANY float64 array of 8 elements, every element in [0.1, 10.0]
    a = any_array((8,), jnp.float64, (0.1, 10.0))
    # your own construction, traced as-is
    a_face = 0.5 * (a + jnp.roll(a, -1))
    # the obligation — returning it is the convention, not a requirement
    return assert_(a_face > 0.0)


print(check(harness, vacuity_mode="inputs-only").render())
```

`python quickstart.py` prints:

```
== VERIFIED
  9 equations verified
  assert #0: discharged — definitely true for all 8 element(s)
    at <your-dir>/quickstart.py:15 (harness)
stelling 0.2.0.dev0 | jax 0.11.0
query 628a25efd4417f44966443e7275a31b7c437cc45ddb6b42efcadb59308171765
arithmetic: interval/f64/outward-1ulp (stelling.interval)
semantics: real (ℝ): obligations judged in exact real arithmetic over the declared sets; the traced program's IEEE float behaviour is NOT modeled — a predicate can hold in ℝ and fail in floats
precision: jax_enable_x64=True | device: none: no concrete execution in this verdict
solver: none — no solver invoked: escalation was NOT ATTEMPTED (solver_timeout_ms not set); every obligation was judged by outward-rounded interval arithmetic alone
nonvacuity: UNCHECKED — no membership conditions declared
transfers: add [sound], concatenate [exact], gt [exact], mul [sound], slice [exact], stelling_any [exact], stelling_assert [exact]
provenance: add:core, concatenate:core, gt:core, mul:core, slice:core, stelling_any:core, stelling_assert:core
assumes: closed-real-interval endpoint convention 0*inf = 0 — a consequence of 'real' semantics; unsound under IEEE semantics, where inf is a value
assumes: vacuity checked (mode=inputs-only): no obligation discharges with the declared bounds widened — under the mechanism(s) that ran, this VERIFIED was not re-derivable without the declared envelope
coverage: 9 eqns: 8 known (89%); 1 transparent
note: nonvacuity UNCHECKED: this VERIFIED may be vacuous — the declared set is not tied to the incident's data
```

Three things to notice, because they are the three things that confuse
people first:

1. **VERIFIED means over the whole declared box**, not at a sample. The
   claim covers every array `any_array((8,), jnp.float64, (0.1, 10.0))`
   admits — and, in the `real` semantics this stamp names, every real
   point of that box, including the corner your tests never visit.
2. **The stamp names its own scope.** `semantics: real (ℝ)` says the
   verdict is about exact real arithmetic, not about your float
   execution. `solver: none` says no solver was consulted.
3. **There are two different vacuity lines and they disagree in tone.**
   `assumes: vacuity checked …` and `note: nonvacuity UNCHECKED … may be
   vacuous` are two *different instruments*, not one contradicting
   itself. [Reading a verdict](reading-a-verdict.md#the-two-vacuity-instruments)
   explains both and tells you how to clear the second one; §3 below is
   the short version.

**One more thing VERIFIED does not say on its own:** that the trace is
faithful to what you wrote. JAX silently wraps out-of-range integer
constants before they reach the jaxpr — if your code has one, the verifier
is correct about a program you did not write. Enable the
[overflow tripwire](overflow-tripwire.md) (`pytest -p stelling.overflow`)
and a VERIFIED with the tripwire armed says the property holds AND that no
narrowing was seen **on the routes the tripwire watches**.

That last clause is load-bearing, and this is the on-ramp, so here is the
short version of it: the watched set is finite and the unwatched one is
real. `jnp.full(shape, N, dt)` — and everything built on it, and everything
numpy narrows before jax is involved — destroys the constant where nothing
can see it, and that program still gets a VERIFIED. Both sets are
enumerated door by door, and measured rather than asserted, in
[the tripwire's coverage table](overflow-tripwire.md#what-it-does-not-find);
read it before you rely on a VERIFIED as a statement about your source.

There is a second, stricter dial for that unwatched set:
`pytest --stelling-eager-truncation=error` makes `jnp.full(shape, N, dt)` and
its relatives **raise** at the line that wrote the constant instead of
narrowing it silently, so a session either contains no undeclared truncation
or does not finish. It is off by default, `-p stelling.overflow` does not turn
it on, and `stelling.intentional_wrap(value, dtype)` is how you say a wrap is
deliberate. It closes seven of the nine unwatched routes; the two it cannot
close are numpy's and are named on that page.

And a third dial, for a narrowing neither of those two can see at all:
`pytest --stelling-narrowing-perimeter=error`. An integer literal written in
a comparison against a float array need not be out of *range* to be destroyed
— `x <= 2**31 - 1` on `float32` runs as `x <= 2147483648.0`, one greater than
what you wrote, and it is a VERIFIED today. (It can also be destroyed by
overflowing: `x <= 100000` on `float16` runs as `x <= inf`, silently, and this
dial refuses that too. That is the only float loss any of the three watches,
and it is about the literal you wrote — a value that is already a float and
overflows is outside all three.) This raises
`stelling.NarrowingError` at the line that wrote the literal. Off by default,
and neither dial above turns it on.

## 2. The three judgments

Save as `statuses.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def verified():
    a = any_array((), jnp.float64, (0.1, 10.0))
    return assert_(a > 0.0)


def refuted_set_level():
    a = any_array((), jnp.float64, (-2.0, -1.0))
    return assert_(a > 0.0)


def unknown_straddle():
    a = any_array((), jnp.float64, (-1.0, 2.0))
    return assert_(a > 0.0)


for h in (verified, refuted_set_level, unknown_straddle):
    v = check(h, vacuity_mode="all")
    print(f"{h.__name__:20s} -> {v.status:8s} | {v.obligations[0].detail}")
```

prints:

```
verified             -> VERIFIED | definitely true for all 1 element(s)
refuted_set_level    -> REFUTED  | definitely false for 1/1 element(s) over the declared box
unknown_straddle     -> UNKNOWN  | undecided for 1/1 element(s); the operand spans [-1.0, 2.0] and the asserted bound is operand > 0.0; the operand's lower endpoint misses the bound by 1.0
```

The third one is the important one. `a > 0.0` is *false at some points*
of `[-1.0, 2.0]` and true at others, and stelling will not guess: a
straddle is UNKNOWN. Give it a solver budget and the same query becomes
REFUTED with a concrete witness. Save as `witness.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def unknown_straddle():
    a = any_array((), jnp.float64, (-1.0, 2.0))
    return assert_(a > 0.0)


v = check(unknown_straddle, vacuity_mode="all", solver_timeout_ms=20_000)
print("status:", v.status)
for w in v.witnesses:
    print("witness:", w.values, "| produced by:", w.produced_by)
```

With `stelling[solvers]` installed this prints:

```
status: REFUTED
witness: (('x0', '0'),) | produced by: z3 5.0.0 (wheel-bindings (smt2 text))
```

`solver_timeout_ms` has no default — a solver run is a stamped,
reproducible event, so you name its budget or it does not happen. Omit
it and the run is interval-only, which the stamp records in the solver
line as `escalation was NOT ATTEMPTED (solver_timeout_ms not set)`, as in
§1 above. Escalation also needs a solver installed
(`pip install stelling[solvers]`).

## 3. Tying the box to real data

The `nonvacuity UNCHECKED … may be vacuous` note in §1 is asking one
question: *does the box you declared actually contain the data you run
on?* A box that contains nothing you care about will verify anything you
like. You answer it with `nonvacuity`, which states a membership
condition in the same traced code the box is stated in. Save as
`nonvacuity_demo.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, nonvacuity
from stelling.preconditions import check

LO, HI = 0.1, 10.0


def harness():
    a = any_array((8,), jnp.float64, (LO, HI))
    a_face = 0.5 * (a + jnp.roll(a, -1))
    obligation = assert_(a_face > 0.0)

    # the data you actually run on, declared as a point and checked
    # against the SAME bounds the box is stated in
    a0 = any_array((8,), jnp.float64, (1.0, 1.0))
    return obligation, nonvacuity(a0 >= LO), nonvacuity(a0 <= HI)


for mode in ("inputs-only", "all"):
    v = check(harness, vacuity_mode=mode)
    print(f"[{mode}] status    :", v.status)
    print(f"[{mode}] nonvacuity:", v.stamp.nonvacuity)
    for a in v.stamp.assumptions:
        if a.startswith("vacuity"):
            print(f"[{mode}] assumes   :", a)
    for n in v.notes:
        print(f"[{mode}] note      :", n)
```

prints:

```
[inputs-only] status    : VERIFIED
[inputs-only] nonvacuity: checked — 2 membership condition(s) definitely true (the declared set contains the stated point)
[inputs-only] assumes   : vacuity instrument inert (mode=inputs-only): declaration 12 is a point interval (1.0), so this mode widens nothing on it; mode='all' would also widen transcribed constants — so the envelope's role in this verdict is left uncharacterised
[all] status    : VERIFIED
[all] nonvacuity: checked — 2 membership condition(s) definitely true (the declared set contains the stated point)
[all] assumes   : vacuity checked (mode=all): no obligation discharges with the declared bounds widened — under the mechanism(s) that ran, this VERIFIED was not re-derivable without the declared envelope
```

The `may be vacuous` note is gone in both runs: `nonvacuity` moved from
`UNCHECKED` to `checked`.

**Measured, and worth knowing before you write your third harness:** the
point declaration that carries the operating point also turns the *other*
vacuity instrument inert under `vacuity_mode="inputs-only"` — that mode
deliberately holds point declarations still, and the instrument refuses
to claim anything when a declared bound did not move. `vacuity_mode="all"`
widens them too and both instruments report. Which mode to use, and what
each buys, is in
[Reading a verdict](reading-a-verdict.md#the-two-vacuity-instruments).

## The one trap worth knowing on day one

`jnp.all(...)` lowers to the `reduce_and` primitive, which has **no
interval transfer**. Its box is ⊤ (unknown), so an obligation or a
membership condition wrapped in `jnp.all(...)` becomes undecidable — not
wrong, just unjudgeable — and an `assume` wrapped in it is dropped.
Measured, on one membership fact spelled three ways (excerpted from the
runnable table in
[the harness API](harness-api.md#three-spellings-and-the-one-that-behaves-differently);
verdict status, then the `nonvacuity` stamp field, then coverage):

```
  two calls         -> VERIFIED nonvacuity=checked   | 8 eqns: 8 known (100%)
  one call, &       -> VERIFIED nonvacuity=checked   | 8 eqns: 8 known (100%)
  one call, jnp.all -> VERIFIED nonvacuity=undecided | 9 eqns: 8 known (89%); 1 ⊤ across 1 primitives (reduce_and ×1)
```

Stelling judges array predicates **elementwise already** — drop the
`jnp.all` and the same condition decides. The runnable version of that
table, what each spelling does when the point is *outside* the box, the
same trap on `assume`, and the arithmetic rewrite for the cases where you
really do need a reduction, are in
[The harness API](harness-api.md#membership-conditions-nonvacuity).

## Where to go next

| you want to | read |
|---|---|
| the exact import path and every primitive | [The harness API](harness-api.md) |
| what each line of the stamp means | [Reading a verdict](reading-a-verdict.md) |
| ready-made templates for solver preconditions | [Checking the preconditions your solver assumes](preconditions.md) |
| what a verdict is allowed to claim | [SOUNDNESS.md](../SOUNDNESS.md) |
