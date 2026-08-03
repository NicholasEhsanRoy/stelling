<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# The harness API

Everything a harness calls lives in **one module**:

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

Importing `stelling.harness` needs a JAX in the environment; importing
`stelling.preconditions` does not (its harness imports happen inside the
functions).

Every code block on this page was executed verbatim against this tree
(stelling 0.1.0, jax 0.11.0, CPU, `jax_enable_x64=True`) and the outputs
are what it printed.

| primitive | states | returns |
|---|---|---|
| `any_array(shape, dtype, (lo, hi))` | an arbitrary input array, every element in `[lo, hi]` | the traced array |
| `any_pytree(prototype, (lo, hi))` | one `any_array` per array leaf of a prototype pytree | a pytree of the same shape |
| `assert_(pred)` | an **obligation** — must hold for every admitted input | `pred` |
| `assume(pred)` | an **assumption** — narrows the box where it can, is disclosed where it cannot | `pred` |
| `nonvacuity(pred)` | a **membership condition** — "the data I run on is in the declared box" | `pred` |
| `trace(harness)` | — | the jax-free `stelling.ir.ClosedJaxpr` |

**Return everything you state.** `assert_`, `assume` and `nonvacuity`
return their predicate so the harness can return it; a harness that
computes an obligation and throws it away has not stated it. The
declarations bind real jax primitives, so they land in the traced jaxpr
and the query's content hash covers them — the stamp identifies the
declarations, not just the program.

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

The refusals share one motive: an **empty declared set verifies
everything**, so emptiness is caught at declaration time rather than
becoming a green verdict about nothing. `Decimal("0.1")` and `0.1` are
both accepted and mean different things — the exact decimal and the
binary64 value — which is why a `str` is refused instead of being
guessed at.

**A point declaration is `lo == hi`.** It is a stated constant, not a
degenerate range, and it interacts with `vacuity_mode`; see
[Reading a verdict](reading-a-verdict.md#the-two-vacuity-instruments).

## `any_pytree(prototype, bounds)`

Tracing-time sugar: one `any_array` per array leaf of a prototype pytree,
each over the same bounds.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_pytree, assert_, trace
from stelling.preconditions import check


def harness():
    prototype = {"u": jnp.zeros((4,)), "k": jnp.zeros(())}
    state = any_pytree(prototype, (0.1, 10.0))   # one declaration per array leaf
    return assert_(state["u"] * state["k"] > 0.0)


print("primitives:", [e.primitive for e in trace(harness).jaxpr.eqns])
print("status    :", check(harness, vacuity_mode="all").status)
```

prints:

```
primitives: ['broadcast_in_dim', 'stelling_any', 'stelling_any', 'mul', 'gt', 'stelling_assert']
status    : VERIFIED
```

Two leaves, two `stelling_any` equations — the same trace the hand
declaration produces, hence the same content hash.

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

## Membership conditions (`nonvacuity`)

`nonvacuity(pred)` states that the data you actually run on lies in the
box you declared — computed in traced code, through the same transforms
the box is stated in. It moves the stamp's `nonvacuity` field, and
nothing else: the verdict's status is unaffected.

The field takes four values, and each corresponds to what the membership
conditions did:

| stamp says | means |
|---|---|
| `UNCHECKED — no membership conditions declared` | you declared none |
| `checked — N membership condition(s) definitely true` | all N decided true |
| `undecided — a membership condition could not be decided` | at least one fell to ⊤ or straddled |
| `FAILED — a membership condition is definitely false` | the stated point is **not** in your box — a harness defect |

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
genuinely a reduction over an array rather than a pointwise fact.

**How you would find this yourself.** The `assume` path prints the
rewrite in its drop note — the `DROPPED` note names `reduce_and` and
quotes both arithmetic forms. The `assert_` and `nonvacuity` paths do
not print that hint; there, the tell is the stamp's coverage line naming
`reduce_and` among the ⊤ primitives. **Read the coverage line whenever an
obligation or a membership condition comes back undecided** — it names
the primitive that stopped the analysis.

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

## Running a harness

`stelling.preconditions.check(harness, *, vacuity_mode, solver_timeout_ms=None, refine=None, strict=False)`
is the front door. `vacuity_mode` is required;
`solver_timeout_ms` has no default and no solver runs without it. See
[Reading a verdict](reading-a-verdict.md) for what comes back, and
[Checking the preconditions your solver assumes](preconditions.md) for the
two ready-made obligation templates (`field_positive`, `scalar_nonzero`)
that build the harness for you.
