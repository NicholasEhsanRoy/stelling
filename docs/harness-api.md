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

Every un-returned statement was still recorded: the obligation still
REFUTES, the membership condition still FAILS, and the assumption still
narrows. The last two rows are the pair to read together — their Python
differs only in whether `assume` is *called* — and the un-returned call
is what makes the difference between VERIFIED and UNKNOWN. They are two
different queries, and measurably so: 6 equations against 4, and
different content hashes. That is the point. Calling `assume` puts a
`stelling_assume` equation in the traced jaxpr; not returning its result
does not take it back out.

**The consequence to internalise: you cannot disable a statement by
removing it from the return.** An `assume` you drop from the return list
is still in force, and a VERIFIED still rides on it. Delete the call.

Returning them anyway is this project's convention and worth keeping —
it is what makes an obligation visible to a reader and to a linter, and
it is the defence if some future tracing path *does* prune an unused
equation. It is a discipline, not a mechanism, and the measurement above
is what the mechanism actually does today.

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

## `any_pytree(prototype, bounds)`

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
genuinely a reduction over an array rather than a pointwise fact. Two
measured reasons the preference is not cosmetic:

* **Under `semantics="ieee"` the arithmetic rewrites do not decide.**
  Both fall to ⊤ at `reduce_sum` — the slack form for three or more
  contributors (float addition is not associative and the jaxpr fixes no
  order), the counting form at *every* size (its accumulator is an
  integer, and ieee endpoint arithmetic is binary64-only). The
  elementwise form decides in both modes at every size.
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
is the front door.

| argument | |
|---|---|
| `vacuity_mode` | **required** — `"inputs-only"` or `"all"`; see [Reading a verdict](reading-a-verdict.md#choosing-vacuity_mode) |
| `solver_timeout_ms` | no default; omit it and no solver runs |
| `refine` | `None`, or `"affine"` for the zonotope refinement on interval-undecided obligations |
| `strict` | `False` (default) returns `DECLINED` for a query that cannot be transcribed; `True` re-raises instead |

**`strict` decides whether an unreadable query is a status or an
exception.** By default a query stelling cannot transcribe comes back as
a `DECLINED` verdict, so a batch caller can record it and carry on to the
next node; `strict=True` lets the `stelling.ir.TranscriptionError`
propagate, which is what you want in a single-target script that should
fail loudly. Transcription has several refusal paths (six `raise` sites
across `stelling.ir` and the transcriber); a sharded program is the one I
could reach from an ordinary harness, and it is what this measures:

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
