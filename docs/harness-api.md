<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The harness API

Everything a harness calls lives in **one module**:

<!-- doc-example: run-only -->
```python
from stelling.harness import any_array, any_pytree, assert_, assume, nonvacuity, trace
```

That is the whole public surface for writing a harness. `stelling` itself
exports no harness primitive, and `stelling.preconditions` exports only
`check`, `field_positive`, `scalar_nonzero`, so the two guesses people
make first both fail. Measured:

```python
import jax
jax.config.update("jax_enable_x64", True)

for stmt in ("from stelling import any_array",
             "from stelling.preconditions import nonvacuity",
             "from stelling.harness import any_array, any_pytree, assert_, "
             "assume, nonvacuity, trace"):
    try:
        exec(stmt)
        print("ok         ", stmt.split(" import ")[0])
    except ImportError as e:
        print("ImportError:", str(e).split(" (")[0])
```

prints:

```
ImportError: cannot import name 'any_array' from 'stelling'
ImportError: cannot import name 'nonvacuity' from 'stelling.preconditions'
ok          from stelling.harness
```

`stelling.harness` re-exports the primitives from
`stelling._jax_compat`, which is private and should not be imported
directly.

Importing `stelling.harness` needs a JAX in the environment: it fails at
import time, not at first call. `import stelling.harness` raises
`stelling.OptionalDependencyError` — a subclass of `ImportError` — saying
`jax is required for tracing harnesses to jaxprs but is not installed;
run: pip install "stelling[jax]"`. `stelling.preconditions` does not need
JAX to import (its harness imports happen inside the functions), but
calling `check` in a JAX-less environment raises the same error.

Every ```` ```python ```` block on this page is executed verbatim by
`tests/test_doc_examples.py` and the fence under it is compared byte for byte
against what it printed (stelling 0.2.0, jax 0.11.0, CPU,
`jax_enable_x64=True`). That is a gate on every run, not a hand-check.

*It was a hand-check, and it was already false when it said so: the
assume-carrier example was marked `illustrative` — never run — and could not
have been run verbatim, because it uses `lax` and nothing on this page imports
it. It is now executed, and the sentence above is true of it.*

| primitive | states | returns |
|---|---|---|
| `any_array(shape, dtype, (lo, hi))` | an arbitrary input array, every element in `[lo, hi]` | the traced array |
| `any_pytree(tree, (lo, hi))` | one `any_array` per array leaf of a prototype pytree | a pytree of the same shape |
| `assert_(pred)` | an **obligation** — must hold for every admitted input | `pred` |
| `assume(pred)` | an **assumption** — narrows the box where it can, is disclosed where it cannot | `pred` |
| `nonvacuity(pred)` | a **membership condition** — "the data I run on is in the declared box" | `pred` |
| `trace(harness)` | — | the jax-free `stelling.ir.ClosedJaxpr` |

<a id="a-statement-counts-once-you-call-it"></a>

**A statement counts once you call it, returned or not.** Each of these
binds a real jax primitive, so it lands in the traced jaxpr and the
query's content hash covers it — the stamp identifies the declarations,
not just the program. Measured, on jax 0.11.0:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, assume, nonvacuity
from stelling.preconditions import check


def assert_not_returned():
    a = any_array((), jnp.float64, (-2.0, -1.0))
    assert_(a > 0.0)                     # never returned
    return ()


def nonvacuity_not_returned():
    a = any_array((), jnp.float64, (0.1, 10.0))
    p = any_array((), jnp.float64, (99.0, 99.0))
    nonvacuity(p <= 10.0)                # never returned
    return assert_(a > 0.0)


def assume_not_returned():
    x = any_array((3,), jnp.float64, (-10.0, 10.0))
    assume(x >= 0.0)                     # never returned
    return assert_(jnp.sum(x) >= 0.0)


def assume_actually_removed():
    x = any_array((3,), jnp.float64, (-10.0, 10.0))
    return assert_(jnp.sum(x) >= 0.0)


for h in (assert_not_returned, nonvacuity_not_returned,
          assume_not_returned, assume_actually_removed):
    v = check(h, vacuity_mode="all")
    print(f"{h.__name__:24s} -> {v.status:8s} "
          f"nonvacuity={v.stamp.nonvacuity.split(' — ')[0]}")
```

prints:

```
assert_not_returned      -> REFUTED  nonvacuity=UNCHECKED
nonvacuity_not_returned  -> VERIFIED nonvacuity=FAILED
assume_not_returned      -> VERIFIED nonvacuity=UNCHECKED
assume_actually_removed  -> UNKNOWN  nonvacuity=UNCHECKED
```

Every un-returned statement was still recorded in the traced jaxpr. The
membership condition still FAILS, and the assumption still narrows.
The assert in `assert_not_returned` is **REFUTED** — asserts are
declarations of intent, and their operands are live regardless of whether
the harness returns the assert's output value. The user stated an
obligation; the obligation is violated; the verdict says so.

The last two rows are the pair to read together — their Python differs
only in whether `assume` is *called* — and the un-returned call is what
makes the difference between VERIFIED and UNKNOWN. They are two
different queries, and measurably so: 6 equations against 4, and
different content hashes. That is the point. Calling `assume` puts a
`stelling_assume` equation in the traced jaxpr; not returning its result
does not take it back out.

**The consequence to internalise: you cannot disable a statement by
removing it from the return.** An `assume` you drop from the return list
is still in force, and a VERIFIED still rides on it. Delete the call.

**The same is true of `assert_`.** An `assert_` whose output is not
returned is still recorded, still evaluated, and still reported: an
assert is a declaration about the program, not a value the caller reads,
so leaving it out of the return list does not withdraw it and does not
soften its verdict.

This paragraph used to promise the opposite — that a violation on an
un-returned assert was downgraded to UNKNOWN "because the violated
variable is dead". That was never the behaviour of the shipped code
(`reachability.reaches_output` seeded every assert as live for exactly
the reason above), and the conjunct that would have performed the
downgrade is removed; audit 0.2.0 B8a, item 4, and the block comment in
`stelling/verdict.py` where it stood.

## `any_array(shape, dtype, bounds)`

Declares an arbitrary array. This is the quantifier: a verdict is about
*every* array this admits.

**Bound spellings.** A bound is refused rather than converted when this
layer cannot judge the conversion's exactness. Measured, on the accepted
and refused spellings:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from decimal import Decimal
from fractions import Fraction

from stelling.harness import any_array

cases = [
    ("float bounds",      lambda: any_array((), jnp.float64, (0.5, 2.0))),
    ("Decimal bound",     lambda: any_array((), jnp.float64, (Decimal("0.1"), 2.0))),
    ("Fraction bound",    lambda: any_array((), jnp.float64, (Fraction(1, 10), 2.0))),
    ("int bound",         lambda: any_array((), jnp.int64, (-3, 3))),
    ("half-infinite",     lambda: any_array((), jnp.float64, (0.0, float("inf")))),
    ("str bound",         lambda: any_array((), jnp.float64, ("0.1", 2.0))),
    ("reversed bounds",   lambda: any_array((), jnp.float64, (2.0, 0.5))),
    ("dtype cannot hold", lambda: any_array((), jnp.uint8, (-3.0, -1.0))),
    ("infinite point",    lambda: any_array((), jnp.float64, (float("inf"), float("inf")))),
]
for label, call in cases:
    try:
        jax.make_jaxpr(call)()
        print(f"{label:18s} -> accepted")
    except ValueError as e:
        print(f"{label:18s} -> refused: {str(e)[:72]}…")
```

prints:

```
float bounds       -> accepted
Decimal bound      -> accepted
Fraction bound     -> accepted
int bound          -> accepted
half-infinite      -> accepted
str bound          -> refused: any_array bound lo='0.1' (type str) is not an accepted bound spelling: a…
reversed bounds    -> refused: any_array bounds (2.0, 0.5) declare an empty set; refusing at declaratio…
dtype cannot hold  -> refused: any_array bounds (-3.0, -1.0) declare a set EMPTY under dtype 'uint8' — …
infinite point     -> refused: any_array bounds (inf, inf) declare an empty real set (an infinite point…
```

Three of those four refusals are one motive: an **empty declared set
verifies everything**, so emptiness is caught at declaration time rather
than becoming a green verdict about nothing. The `str` refusal is a
different one — the message says so: a bound is *refused rather than
converted* when this layer cannot judge the conversion's exactness,
because that is how a declared bound gets silently rounded.

**What the spellings are for.** Once a bound is admitted on a float
dtype, the spelling leaves no trace: `0.1`, `Decimal("0.1")` and
`Fraction(1, 10)` all record the same binary64 endpoint and produce the
same query hash. Where the spelling matters is **admissibility** — the
guard reads each bound's *exact* value, and the exact values differ
(`0.1` is exactly 3602879701896397/36028797018963968; `Decimal("0.1")`
is exactly 1/10). Measured, the same two spellings on two dtypes:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from decimal import Decimal

from stelling.harness import any_array, trace


def recorded(lo, hi, dtype):
    cj = trace(lambda: any_array((), dtype, (lo, hi)))
    p = dict([e for e in cj.jaxpr.eqns if e.primitive == "stelling_any"][0].params)
    return f"lo={p['lo']!r} hi={p['hi']!r} hash={cj.content_hash()[:16]}"


for dtype in (jnp.float64, jnp.int64):
    for label, lo in (("0.1", 0.1), ("Decimal('0.1')", Decimal("0.1"))):
        try:
            print(f"{str(jnp.dtype(dtype)):8s} {label:14s} -> {recorded(lo, 10.0, dtype)}")
        except ValueError as e:
            print(f"{str(jnp.dtype(dtype)):8s} {label:14s} -> refused: "
                  f"{str(e).split('. ')[0][:96]}…")
```

prints:

```
float64  0.1            -> lo=0.1 hi=10.0 hash=a6f639be687fa246
float64  Decimal('0.1') -> lo=0.1 hi=10.0 hash=a6f639be687fa246
int64    0.1            -> lo=0.1 hi=10.0 hash=3e9afcecb4af3e5d
int64    Decimal('0.1') -> refused: any_array bound lo=Decimal('0.1') is not representable as the binary64 the IR stores; it would b…
```

The two float64 rows are byte-identical, hash included; the int64 row
hashes differently only because the dtype is part of the recorded query.
On `float64` the two spellings are indistinguishable in the query. On
`int64` they are not: binary64's `0.1` is slightly *above* 1/10, so
recording `Decimal("0.1")` as a lower bound would move the endpoint into
the interval's interior, and `int64` refuses every narrowing bound as a
dtype-level policy. The float spelling declares that same binary64 value
exactly, so there is nothing to narrow. Rounding the other way — a bound
whose recording *widens* the box — is always admitted, because an
over-approximation still contains every executed value.

**A point declaration is `lo == hi`.** It is a stated constant, not a
degenerate range, and it interacts with `vacuity_mode`; see
[Reading a verdict](reading-a-verdict.md#the-two-vacuity-instruments).

## `any_pytree(tree, bounds)`

Tracing-time sugar: one `any_array` per array leaf of a prototype pytree,
each over the same bounds.

**Build the prototype outside the traced code, from NumPy.** The
prototype's only job is to carry shapes and dtypes, but it is an ordinary
value in the harness: build it with `jnp.zeros` inside the harness and you
trace its *construction* into the query too. Measured — same three
declarations, three ways:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from stelling.harness import any_array, any_pytree, assert_, trace


def sugar(scalar_leaf, vector_leaf):
    def harness():
        prototype = {"k": scalar_leaf(), "u": vector_leaf((4,))}
        state = any_pytree(prototype, (0.1, 10.0))   # one declaration per leaf
        return assert_(state["u"] * state["k"] > 0.0)
    return harness


def hand():
    k = any_array((), jnp.float64, (0.1, 10.0))
    u = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(u * k > 0.0)


cases = (
    ("prototype all numpy",   sugar(lambda: np.zeros(()),  np.zeros)),
    ("prototype all jnp",     sugar(lambda: jnp.zeros(()), jnp.zeros)),
    ("jnp SCALAR leaf only",  sugar(lambda: jnp.zeros(()), np.zeros)),
    ("jnp VECTOR leaf only",  sugar(lambda: np.zeros(()),  jnp.zeros)),
    ("hand declaration",      hand),
)
for name, h in cases:
    cj = trace(h)
    print(f"{name:22s} {len(cj.jaxpr.eqns)} eqns  hash {cj.content_hash()[:16]}  "
          f"{[e.primitive for e in cj.jaxpr.eqns]}")
```

prints:

```
prototype all numpy    5 eqns  hash 93bfe936574a4195  ['stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
prototype all jnp      6 eqns  hash fcbb6209ead48d15  ['broadcast_in_dim', 'stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
jnp SCALAR leaf only   5 eqns  hash 93bfe936574a4195  ['stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
jnp VECTOR leaf only   6 eqns  hash fcbb6209ead48d15  ['broadcast_in_dim', 'stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
hand declaration       5 eqns  hash 93bfe936574a4195  ['stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
```

Two leaves, two `stelling_any` equations. **With a NumPy prototype the
sugar traces to exactly the hand declaration — same equations, same
content hash.**

The last two rows locate the cost precisely: swapping *only* the scalar
leaf to `jnp` changes nothing, and swapping *only* the vector leaf adds
the `broadcast_in_dim`. `jnp.zeros(())` traces no equation at all;
**`jnp.zeros((4,))` is the one that traces a `broadcast_in_dim`**, so any
non-scalar `jnp` leaf built inside the harness lengthens the query by
one equation and changes its hash. Nothing is unsound about the `jnp`
version — the same two inputs are declared over the same bounds — but the
stamp identifies the query by content hash, so two verdicts you meant to
be comparable will not compare equal.

## `assert_(pred)` — obligations

The thing being judged. One `assert_` is one obligation; array-valued
predicates are judged **elementwise**, and VERIFIED means every element.

Obligations are reported individually in the verdict (`assert #0`,
`assert #1`, …) with a per-obligation status, so a harness that states
five properties tells you which of the five it could decide.

## `assume(pred)` — assumptions

An assumption either **narrows** the propagated box or is **dropped and
disclosed**. It is never silently honoured. Which one you get depends on
the shape of the predicate:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, assume
from stelling.preconditions import check


def constrains():
    x = any_array((3,), jnp.float64, (-10.0, 10.0))
    assume(x >= 0.0)                       # elementwise: narrows the box
    return assert_(jnp.sum(x) >= 0.0)


def drops():
    x = any_array((3,), jnp.float64, (-10.0, 10.0))
    assume(jnp.all(x >= 0.0))              # reduce_and: no narrowing possible
    return assert_(jnp.sum(x) >= 0.0)


for h in (constrains, drops):
    v = check(h, vacuity_mode="all")
    print(f"{h.__name__:11s} -> {v.status:8s} | {v.stamp.coverage}")
```

prints:

```
constrains  -> VERIFIED | 6 eqns: 5 known (83%); 1 assume(s) CONSTRAINED (stelling_assume ×1)
drops       -> UNKNOWN  | 7 eqns: 5 known (71%); 1 ⊤ across 1 primitives (reduce_and ×1); 1 constraint(s) DROPPED (stelling_assume ×1)
```

Same mathematical assumption, two spellings, two verdicts. The
difference is the `jnp.all` — see the next section for the mechanism.

A dropped assume is one-sided in a specific way, and the verdict says so
in its notes: the query ran over a **superset** of the set you intended,
so a VERIFIED still holds on your subset, but a REFUTED witness may
violate the dropped assumption and must be checked against it before it
is treated as a counterexample.

### An `assume` inside a `scan` or `while_loop` body is not descended

Write the precondition at the **top level of the harness**. stelling's
propagation descends `jit`, `cond` and the other transparent wrappers; it
does **not** enter a `scan` or `while_loop` body, so a `stelling_assume`
written in one is never classified — it narrows nothing and is not forwarded
to the solver.

That is not silently ignored. Such an assume is recorded as a **dropped**
assumption, with a note naming the construct and the source line
(`assume NEVER CLASSIFIED at …: it sits inside 'scan'`) and a stamped
`precondition satisfiability uncertified` assumption; the query is then
judged over a superset, so a VERIFIED still holds on your region and every
definite violation is withheld to UNKNOWN rather than reported as a
counterexample.

**That withholding costs real refutations, and the UNKNOWN cannot tell you
which.** On a 240-harness loop-carrier corpus, **200 rows moved from REFUTED
to UNKNOWN, and 40 % of THOSE — 80 rows — were genuine**: the witness lay in
the declared box, satisfied every assume, and falsified the assert. (The
denominator is the 200 moved rows, not the 240 harnesses. Recorded in
[SOUNDNESS.md](../SOUNDNESS.md) as *"A CORRECT REFUTATION WITH A CORRECT
WITNESS, WITHHELD 80 = 40 %"*, produced by
`scratchpad/s13/sweep_loop_assume_wide.py` with results in
`scratchpad/s13/RESULTS_loop_wide.txt`. Both are in the git checkout; neither
is in the sdist — what a reader of the distribution has instead is the
reproducer's executable form, `s13_scan` in
`tests/test_undescended_assume.py`, which ships and runs; the 240-harness
sweep behind the ratio does not.) The withholding is still the right answer —
nothing in the run honoured your precondition, so nothing could tell that
witness from one your precondition excludes — but an UNKNOWN here means
*undecided*, not *your program is fine*. Lift the `assume` to the top level
to get a decision either way.

It was not always recorded. Up to and including **0.1.0** an `assume` inside
one of those bodies left no trace at all, and a REFUTED verdict on such a
harness could name a point that assume excludes — see the entry in
[SOUNDNESS.md](../SOUNDNESS.md).

The construct is not descended because a loop body's `assume` is a statement
about a **carry that changes from iteration to iteration**, and this release
does not model one. If the precondition is really about the declared inputs,
it belongs above the loop:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

from stelling.harness import any_array, assert_, assume
from stelling.preconditions import check


def not_honoured():
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))

    def body(c, _):
        assume(x <= y)              # never classified: it sits inside 'scan'
        return c, 0.0

    lax.scan(body, x, jnp.zeros((2,)))
    return assert_(x - y <= 0.0)


def honoured():
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    assume(x <= y)                  # relational: narrows nothing, but is
                                    # forwarded to the solver as an axiom
    return assert_(x - y <= 0.0)


def no_assume():
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    return assert_(x - y <= 0.0)


for name, harness in [("not_honoured", not_honoured),
                      ("honoured    ", honoured),
                      ("no_assume   ", no_assume)]:
    plain = check(harness, vacuity_mode="inputs-only")
    solved = check(harness, vacuity_mode="inputs-only",
                   solver_timeout_ms=10_000)
    print(f"{name}  no solver: {plain.status:9} with solver: {solved.status}")
```

```
not_honoured  no solver: UNKNOWN   with solver: UNKNOWN
honoured      no solver: UNKNOWN   with solver: VERIFIED
no_assume     no solver: UNKNOWN   with solver: REFUTED
```

**Read the columns, not the rows.** Without a solver all three are UNKNOWN,
and for a reason that has nothing to do with `assume`: `x - y` spans
`[-20, 20]`, which straddles `0`, so interval arithmetic decides none of them.
A relational `assume` narrows no box; it is an axiom the solver is given, so
it can only change an answer a solver produces.

**With a solver, the third row is what makes the first one evidence.**
`no_assume` is REFUTED — there really is a violating point in the declared
box. `honoured` is VERIFIED, because the forwarded axiom rules that point
out. `not_honoured` states the same true hypothesis and gets UNKNOWN: the
assume sits inside `scan`, is never classified, and is not forwarded — so the
correct refutation is WITHHELD rather than the true property being proved.
That is the withholding this section is about, and it is only visible against
the third row.

*This block was `illustrative` — never run — and carried
`# VERIFIED` on `honoured` beside a section that tells you to omit
`solver_timeout_ms`. Verbatim under the documented call it is UNKNOWN, and
the block could not be executed at all: `lax` is imported nowhere on this
page. Both are fixed by running it.*

## Membership conditions (`nonvacuity`)

`nonvacuity(pred)` states that the data you actually run on lies in the
box you declared — computed in traced code, through the same transforms
the box is stated in. It moves the stamp's `nonvacuity` field **and the notes
that hang off it**; the verdict's **status** is unaffected.

The status half is the load-bearing one and is exact: no `nonvacuity` has ever
turned a VERIFIED into anything else. But "nothing else" was too strong, and
this page's own fences show it — adding a `nonvacuity` moves the coverage line
(the predicate is traced, so its equations are counted), and a VACUOUS one adds
a `may be vacuous` note to the verdict.

The field takes six values, and each corresponds to what the membership
conditions did:

| stamp says | means |
|---|---|
| `UNCHECKED — no membership conditions declared` | you declared none |
| `checked — N membership condition(s) definitely true` | all N decided true |
| `checked in part — K of N membership condition(s) definitely true` | K decided true; the other N−K are over **zero-element** arrays and tested no point |
| `VACUOUS — N membership condition(s) hold over ZERO elements` | every one is over a zero-element array: nothing was tested, and a VERIFIED alongside it carries the "may be vacuous" note |
| `undecided — a membership condition could not be decided` | at least one fell to ⊤ or straddled |
| `FAILED — a membership condition is definitely false` | the stated point is **not** in your box — a harness defect |

A membership condition over a size-0 array is *discharged*, because
`jnp.all` of an empty array is true — but it tested no point, so it
establishes nothing about the declared set. That case used to be counted
into the plain `checked — …` line, whose parenthetical then claimed "the
declared set contains the stated point" for a set nothing had been
compared against.

### Three spellings, and the one that behaves differently

The same membership fact can be spelled as separate calls, as one call
over `&`, or as one call over `jnp.all`. Measured:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, nonvacuity
from stelling.preconditions import check

LO, HI = 0.1, 10.0


def make(spelling, point):
    def harness():
        a = any_array((), jnp.float64, (LO, HI))
        obligation = assert_(a > 0.0)
        a0 = any_array((), jnp.float64, (point, point))
        lo_ok, hi_ok = a0 >= LO, a0 <= HI
        if spelling == "two calls":
            return obligation, nonvacuity(lo_ok), nonvacuity(hi_ok)
        if spelling == "one call, &":
            return obligation, nonvacuity(lo_ok & hi_ok)
        if spelling == "one call, jnp.all":
            return obligation, nonvacuity(jnp.all(lo_ok & hi_ok))
        raise AssertionError(spelling)
    return harness


for point, label in ((1.0, "point INSIDE the box"), (99.0, "point OUTSIDE the box")):
    print(f"--- {label} (a0 = {point}) ---")
    for spelling in ("two calls", "one call, &", "one call, jnp.all"):
        v = check(make(spelling, point), vacuity_mode="all")
        print(f"  {spelling:17s} -> {v.status:8s} nonvacuity={v.stamp.nonvacuity.split(' — ')[0]:9s}"
              f" | {v.stamp.coverage}")
```

prints:

```
--- point INSIDE the box (a0 = 1.0) ---
  two calls         -> VERIFIED nonvacuity=checked   | 8 eqns: 8 known (100%)
  one call, &       -> VERIFIED nonvacuity=checked   | 8 eqns: 8 known (100%)
  one call, jnp.all -> VERIFIED nonvacuity=undecided | 9 eqns: 8 known (89%); 1 ⊤ across 1 primitives (reduce_and ×1)
--- point OUTSIDE the box (a0 = 99.0) ---
  two calls         -> VERIFIED nonvacuity=FAILED    | 8 eqns: 8 known (100%)
  one call, &       -> VERIFIED nonvacuity=FAILED    | 8 eqns: 8 known (100%)
  one call, jnp.all -> VERIFIED nonvacuity=undecided | 9 eqns: 8 known (89%); 1 ⊤ across 1 primitives (reduce_and ×1)
```

The status column is the same in every row: `nonvacuity` moves the stamp
field and nothing else. A `FAILED` membership condition is reported
loudly and leaves the VERIFIED standing, because it says your *harness*
is wrong about where your data is, not that the box fails to be
invariant.

**What was measured.** Separate calls and `&` agree on the *decision* and
differ only in the **count** the stamp reports (2 conditions vs 1) —
either reaches `checked`, and either reaches `FAILED` when the point is
outside. The `jnp.all` spelling reaches **neither**, on either face: it
is `undecided` when the point is inside the box, and `undecided` when the
point is `99.0` and the box is `[0.1, 10.0]`. A membership condition that
cannot come back FAILED cannot catch the harness defect it exists to
catch.

**The mechanism, named from the coverage line.** `jnp.all(...)` lowers to
the jax primitive `reduce_and`. `reduce_and` has **no interval transfer**
in `stelling.propagate`, so its output box falls to ⊤ — visible directly
in the stamp: `1 ⊤ across 1 primitives (reduce_and ×1)`. A ⊤ boolean is
neither definitely true nor definitely false, so the condition is judged
`unknown` and the stamp says `undecided`. Nothing about membership is
special here: the same ⊤ makes `assert_(jnp.all(...))` an UNKNOWN
obligation and `assume(jnp.all(...))` a dropped assumption, as measured
in the previous section.

**The fix is usually to delete the reduction.** Stelling judges array
predicates elementwise already, so `nonvacuity(a0 >= LO)` on a
4-element array *is* the conjunction over all four elements. Where you
genuinely need a reduction, express it arithmetically with registered
primitives. Measured, all on the same 4-element field:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, nonvacuity
from stelling.preconditions import check

LO, HI = 0.1, 10.0


def elementwise():
    a = any_array((), jnp.float64, (LO, HI))
    a0 = any_array((4,), jnp.float64, (1.0, 1.0))
    return assert_(a > 0.0), nonvacuity(a0 >= LO), nonvacuity(a0 <= HI)


def slack():
    a = any_array((), jnp.float64, (LO, HI))
    a0 = any_array((4,), jnp.float64, (1.0, 1.0))
    out = jnp.sum(jnp.maximum(LO - a0, 0.0) + jnp.maximum(a0 - HI, 0.0))
    return assert_(a > 0.0), nonvacuity(out <= 0.0)


def counting():
    a = any_array((), jnp.float64, (LO, HI))
    a0 = any_array((4,), jnp.float64, (1.0, 1.0))
    bad = jnp.sum((a0 < LO).astype(jnp.int32)) + jnp.sum((a0 > HI).astype(jnp.int32))
    return assert_(a > 0.0), nonvacuity(bad == 0)


def reduced():
    a = any_array((), jnp.float64, (LO, HI))
    a0 = any_array((4,), jnp.float64, (1.0, 1.0))
    return assert_(a > 0.0), nonvacuity(jnp.all(a0 >= LO))


for h in (elementwise, slack, counting, reduced):
    v = check(h, vacuity_mode="all")
    print(f"{h.__name__:12s} -> {v.stamp.nonvacuity.split(' — ')[0]:9s} | {v.stamp.coverage}")
```

prints:

```
elementwise  -> checked   | 8 eqns: 8 known (100%)
slack        -> checked   | 12 eqns: 12 known (100%)
counting     -> checked   | 15 eqns: 15 known (100%)
reduced      -> undecided | 7 eqns: 6 known (86%); 1 ⊤ across 1 primitives (reduce_and ×1)
```

Prefer `elementwise`. The two arithmetic rewrites cost more equations for
the same answer; they exist for the cases where the condition is
genuinely a reduction over an array rather than a pointwise fact. Two
measured reasons the preference is not cosmetic:

* **Under `semantics="ieee"` the arithmetic rewrites do not decide.**
  Both fall to ⊤ at `reduce_sum` — the slack form for three or more
  contributors (float addition is not associative and the jaxpr fixes no
  order), the counting form at *every* size (its accumulator is an
  integer, and ieee endpoint arithmetic models the four catalogued FLOAT
  formats only — integer wraparound is a real-mode row). The elementwise
  form decides in both modes at every size.
* **As an `assume` they are not interchangeable.** All three CONSTRAIN
  rather than DROP, but the arithmetic pair narrows the reduction's own
  intermediate — an over-approximated value — so the precondition is
  stamped satisfiability-UNCERTIFIED and every definite violation is
  withheld from REFUTED *unless a probed point of the declared set is
  found to satisfy every assume of the query* (the non-emptiness
  certificate). The elementwise form narrows the declared input and stays
  certified with no search at all.

  A `jnp.all` assume is out of that search's reach in **both**
  directions, which is the one thing to know about it here: `jnp.all`
  lowers to `reduce_and`, which has no interval transfer, so its
  predicate is ⊤ at a single point exactly as it is over a box. The
  arithmetic rewrites evaluate to a definite value at a point and their
  refutations are recoverable; a `jnp.all` assume's are not.

**One conjunct, one call.** Deleting the reduction from a *conjunction*
means writing each conjunct as its own `assert_` / `assume` /
`nonvacuity` call, not one `&`: `jnp.all(w >= 0.0) & jnp.all(b >= 0.0)`
over differently-shaped `w` and `b` traces fine with the reductions in
place and raises `TypeError: and got incompatible shapes for
broadcasting` once they are gone. Two calls are the conjunction of both
and carry no shape condition at all — measured `discharged`/`discharged`
in both semantics on exactly that pair.

**How you would find this yourself.** Every path that can be weakened by
this prints the rewrite, in the notes, off one shared gate: the `assume`
path (in both `assume_mode`s, and on the dropped conjunct of a mixed
conjunction) names it in its `DROPPED` note; an undecided `assert_` or
`nonvacuity` gets an `obligation UNDECIDED at …` /
`nonvacuity condition UNDECIDED at …` note carrying the same text. It is
a *note* rather than an obligation detail because escalating replaces the
detail with the solver record's own. The body is printed once per run —
later faces carry a pointer to it rather than a second copy.

The gate is deliberately narrower than "the query contains a `jnp.all`",
and it has three conditions:

1. **It is reached from the judged predicate** through `and`, `reshape`,
   `squeeze`, `broadcast_in_dim` or `convert_element_type` — the ops a
   `jnp.all` result can pass through and still *be* that conjunction. So
   `jnp.all(a) & jnp.all(b)`, a `keepdims` reduction (which lowers to a
   `broadcast_in_dim`/`squeeze` pair) and a nested
   `jnp.all(jnp.all(M, axis=1))` all hint, while
   `jnp.where(jnp.all(...), a, b)` (a selector, not the property),
   `jnp.all(a) | jnp.all(b)` and `~jnp.all(a)` (deleting the reduction
   would change what you stated) do not.
2. **Nothing else in the predicate is ⊤.** `jnp.all(jnp.max(x) >= 0)`
   gets no hint, because deleting the `jnp.all` leaves `reduce_max` in
   the way and none of the rewrites would help — and neither does
   `jnp.all(x >= 0) & (jnp.max(y) >= 0)`, where the dead end is a
   sibling conjunct rather than an operand. A nested `reduce_and` is not
   counted against itself: it is the thing the rewrite removes.
3. **The predicate is small enough to walk.** The gate walks at most 512
   nodes of the predicate's *cone* and goes silent rather than guess past
   it. This is a bound on the predicate, not the program: measured, a
   407-equation jaxpr whose predicate is one `jnp.all` still hints, while
   a predicate that is 171 chained `&` conjuncts does not (170 still
   does).

So the stamp's coverage line remains the general instrument: **read it
whenever an obligation or a membership condition comes back undecided**,
because it names the primitive that stopped the analysis whatever that
primitive is, at any size, on any spelling.

## `jnp.where` and selector decidability

`jnp.where(cond, a, b)` traces to a `select_n` primitive — **inside the
transparent `jit` wrapper jax emits for it**, not at top level. That matters
for the inspection idiom this page teaches below: on jax 0.11.0
`[e.primitive for e in closed.jaxpr.eqns]` over such a query reads
`['stelling_any', 'gt', 'sub', 'jit', 'gt', 'stelling_assert']`, with no
`select_n` in it, and the coverage line reports the wrapper as `1 transparent`.
Everything in the rest of this section is about the `select_n` semantics and is
correct as written. When the
condition's interval is **provably one-sided** (`[True, True]` or
`[False, False]` over the entire declared box), stelling uses only the
reachable branch — the unreachable branch's interval is discarded, no
matter how wide it is. This is standard branch pruning in abstract
interpretation.

When the condition is **undecidable** (`[False, True]` — some inputs
make it True, others False), `select_n` takes the hull of both branches.
The result is the union of both possible outcomes, which is sound but
imprecise: every obligation downstream gets the merged interval.

**What makes a condition decidable:**
- `x > threshold` where the input's interval is entirely above or below
  the threshold (e.g., `x ∈ [5, 10]` and threshold = 3 → always True)
- `jnp.isfinite(x)` where the input is bounded (always True for bounded
  declarations)
- Compound conditions: `cond1 & cond2` is decidable if both components
  are decidable (the `and` transfer propagates three-valued logic)

**What does NOT help (the dependency problem):**
If the same condition appears in two places — `jnp.where(cond, a, b)` and
later `jnp.where(~cond, c, d)` — interval arithmetic does NOT track that
they are the same variable negated. Each `where` is evaluated
independently. `cond & ~cond` evaluates to `[0, 1]` (unknown) rather than
`[0, 0]` (definitely False) because the domain is non-relational.

**Remedy:** pass `solver_timeout_ms` — the SMT encoding carries the full
constraint set and WILL prove mutual exclusions, contradictions, and
correlated conditions that intervals cannot express. This is the designed
escalation path; the interval pass is the fast-but-imprecise first layer.

## `trace(harness)`

Traces a nullary harness and transcribes it into the jax-free
`stelling.ir.ClosedJaxpr` — the object every later stage consumes, and
the object the stamp's `query …` hash identifies. You do not need it to
run a check; it is how you inspect what your harness actually traced,
which is the fastest way to find out that a harness built more (or less)
of your program than you meant:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, trace


def harness():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jnp.sum(a) > 0.0)


closed = trace(harness)
print("equations :", [e.primitive for e in closed.jaxpr.eqns])
print("query hash:", closed.content_hash())
```

prints:

```
equations : ['stelling_any', 'reduce_sum', 'gt', 'stelling_assert']
query hash: 52336382a4d6677b35371cfd40267eb8c36e144c6d16c18bbe25b18a4b4372ef
```

**That hash was measured on jax 0.11.0 and re-measured unchanged on jax
0.11.1**, and the version belongs beside it: a query hash is a function of
the jax that traced the harness, because it hashes the equations' params and
jax owns those. The same jax pair moves the hash of a harness using
`jnp.max` or `jnp.min`, where 0.11.1 added an `out_sharding` param that
0.11.0 does not emit — [SOUNDNESS.md](../SOUNDNESS.md) records that break.
`reduce_sum` already carried `out_sharding` on both, which is why this
block is stable across the pair rather than lucky.

## Running a harness

`stelling.preconditions.check` is the front door. **Its signature is printed
from the object rather than typed here**, because a typed one rotted: this page
was written against the object at `343ebe6` (2026-08-03), and then four
parameters landed under it — `solver` (`cbb1d60`, 08-12), `semantics`
(`d6451cc`, 08-13), `libm_budget` (`c322cec`, 08-15) and `falsify`
(`123ad75`, 08-19) — while this page went on using one of them,
`semantics="ieee"`, further down. Nothing pinned the typed copy.

```python
import inspect

from stelling.harness import any_array, any_pytree, assert_, assume, nonvacuity
from stelling.preconditions import check

for fn in (any_array, any_pytree, assert_, assume, nonvacuity):
    print(f"{fn.__name__}{inspect.signature(fn)}")
print()
for name, param in inspect.signature(check).parameters.items():
    default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
    print(f"check.{name}{default}")
```

```
any_array(shape, dtype, bounds)
any_pytree(tree, bounds)
assert_(pred)
assume(pred)
nonvacuity(pred)

check.harness
check.vacuity_mode
check.semantics = 'real'
check.solver_timeout_ms = None
check.refine = None
check.solver = None
check.strict = False
check.libm_budget = None
check.falsify = None
```

| argument | |
|---|---|
| `vacuity_mode` | **required** — `"inputs-only"` or `"all"`; see [Reading a verdict](reading-a-verdict.md#choosing-vacuity_mode) |
| `semantics` | `"real"` (default) judges in exact real arithmetic; `"ieee"` models the format's rounding and overflow — see [Checking preconditions](preconditions.md) |
| `solver_timeout_ms` | `None` by default, which means **no solver runs**; set it in milliseconds to escalate |
| `refine` | `None`, or `"affine"` for the zonotope refinement on interval-undecided obligations |
| `solver` | `None` (the full portfolio), `"z3"` or `"cvc5"` to restrict it; anything else raises `ValueError` at the call. See [Choosing a solver backend](choosing-a-solver-backend.md) |
| `strict` | `False` (default) returns `DECLINED` for a query that cannot be transcribed; `True` re-raises instead |
| `libm_budget` | a declared accuracy profile for `exp`/`pow` under `semantics="ieee"`, which decline without one. Passing it under `"real"` raises — see [Checking preconditions](preconditions.md) |
| `falsify` | the falsification pass; see [Reading a verdict](reading-a-verdict.md) |

**`strict` decides whether an unreadable query is a status or an
exception.** By default a query stelling cannot transcribe comes back as
a `DECLINED` verdict, so a batch caller can record it and carry on to the
next node; `strict=True` lets the `stelling.ir.TranscriptionError`
propagate, which is what you want in a single-target script that should
fail loudly. Transcription has several refusal paths across `stelling.ir` and
the transcriber; a sharded program is the one I could reach from an ordinary
harness, and it is what this measures:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from stelling import ir
from stelling.harness import any_array, assert_
from stelling.preconditions import check

mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2), ("x",))
spec = jax.sharding.PartitionSpec("x")


def harness():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    sharded = jax.shard_map(lambda z: z + 1.0, mesh=mesh, in_specs=spec, out_specs=spec)
    return assert_(sharded(a) > 0.0)


v = check(harness, vacuity_mode="all")
print("strict=False ->", v.status, "| stamp:", v.stamp)
print("  ", v.notes[0])
try:
    check(harness, vacuity_mode="all", strict=True)
except ir.TranscriptionError as e:
    print("strict=True  -> raised", type(e).__name__)
```

prints:

```
strict=False -> DECLINED | stamp: None
   declined: primitive 'shard_map': param 'mesh' is a non-empty mesh (Mesh(axis_sizes=(2,), axis_names=('x',), axis_types=(Auto,))); sharded programs are not supported yet.
strict=True  -> raised UnsupportedParamError
```

`strict` covers transcription failures only. Harness defects (an empty
declared set, an unsatisfiable assume) and jax's own tracing failures
raise in both modes — the first are your bug and must stay loud, the
second happen upstream of stelling.

See [Reading a verdict](reading-a-verdict.md) for what comes back, and
[Checking the preconditions your solver assumes](preconditions.md) for the
two ready-made obligation templates (`field_positive`, `scalar_nonzero`)
that build the harness for you.
