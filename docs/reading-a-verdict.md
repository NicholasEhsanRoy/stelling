<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Reading a verdict

A verdict is a status plus a **stamp**: everything the claim rests on,
recorded per verdict rather than assumed from context. `Verdict.render()`
prints all of it. This page walks the render top to bottom, then takes
the two lines people reliably read as a contradiction.

Every code block on this page was executed verbatim against this tree
(stelling 0.1.0, jax 0.11.0, CPU, `jax_enable_x64=True`) and the outputs
are what it printed.

## The status

| status | means | does **not** mean |
|---|---|---|
| `VERIFIED` | every obligation is definitely true at every point of the declared box | that it holds when you run it in floats — read the `semantics` line |
| `REFUTED` | at least one obligation is definitely false over the propagated superset of your box: the box is **not invariant as stated** | that a concrete input was produced, unless the verdict carries a witness |
| `UNKNOWN` | the analysis could not decide, and says why | that the property is undecidable, or that it fails |
| `DECLINED` | the query could not be transcribed at all, so nothing was analysed and nothing is stamped | that anything was judged in either direction |

The three judged statuses, measured on one predicate and three declared
boxes:

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

```
verified             -> VERIFIED | definitely true for all 1 element(s)
refuted_set_level    -> REFUTED  | definitely false for 1/1 element(s) over the declared box
unknown_straddle     -> UNKNOWN  | undecided for 1/1 element(s); the operand spans [-1.0, 2.0] and the asserted bound is operand > 0.0; the operand's lower endpoint misses the bound by 1.0
```

A straddle — true somewhere in the box, false somewhere else — is
UNKNOWN, never REFUTED, because interval propagation over-approximates
and the false region might be an artefact of that over-approximation. An
SMT escalation (`solver_timeout_ms=…`) can turn a straddle into a REFUTED
carrying a witness confirmed by exact-rational replay.

`DECLINED` is narrower than people expect: it is for a query stelling
could not **read**, not one it could not decide. It carries no stamp at
all — `verdict.stamp is None` — because a stamp identifies a query by
content hash and an unread query has none:

```
== DECLINED
  the query was not analysed — stelling could not read it. No claim is made about the program, in either direction.
  declined: example: an eqn param type with no transcription rule
  no stamp: a stamp identifies a query by content hash, and this query was never transcribed.
```

A primitive with no registered transfer does **not** decline — it falls
to ⊤ and the obligation comes back UNKNOWN with the primitive named in
the coverage line:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check
from stelling.coverage import DEFAULT_TRANSPARENT

print("transparent wrappers:", sorted(DEFAULT_TRANSPARENT))


def through_a_jit():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jax.jit(lambda z: z + 1.0)(a) > 0.0)


def through_a_sort():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jnp.sort(a) > 0.0)


for h in (through_a_jit, through_a_sort):
    v = check(h, vacuity_mode="all")
    print(f"{h.__name__:16s} -> {v.status:8s} | {v.stamp.coverage}")
```

```
transparent wrappers: ['custom_jvp_call', 'custom_vjp_call', 'jit', 'remat2']
through_a_jit    -> VERIFIED | 5 eqns: 4 known (80%); 1 transparent
through_a_sort   -> UNKNOWN  | 5 eqns: 3 known (60%); 1 transparent; 1 ⊤ across 1 primitives (sort ×1)
```

## `transparent`, defined

**A transparent primitive is a wrapper the analysis descends through: it
contributes no semantics of its own, only a nested jaxpr.** The set is
fixed and small — measured above, it is exactly `custom_jvp_call`,
`custom_vjp_call`, `jit`, `remat2`. In the coverage line a transparent
equation is counted in its own bucket: it is neither `known` (it has no
transfer) nor ⊤ (nothing was lost), which is why `through_a_jit` reads
`4 known (80%); 1 transparent` and is still VERIFIED.

**You will see `transparent` counts in harnesses that never write
`jax.jit`.** JAX jit-wraps many of its own `jnp` functions, so the
wrapper appears in the trace whether or not you asked for it — which is
why `through_a_sort` above, which contains no `jax.jit`, still reports
`1 transparent`. Measured:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_, trace


def explicit_jit():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jax.jit(lambda z: z + 1.0)(a) > 0.0)


def jnp_sort():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jnp.sort(a) > 0.0)


def jnp_roll():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_(jnp.roll(a, -1) > 0.0)


def plain_add():
    a = any_array((4,), jnp.float64, (0.1, 10.0))
    return assert_((a + 1.0) > 0.0)


for h in (explicit_jit, jnp_sort, jnp_roll, plain_add):
    print(f"{h.__name__:12s} {[e.primitive for e in trace(h).jaxpr.eqns]}")
```

```
explicit_jit ['stelling_any', 'jit', 'gt', 'stelling_assert']
jnp_sort     ['stelling_any', 'jit', 'gt', 'stelling_assert']
jnp_roll     ['stelling_any', 'jit', 'gt', 'stelling_assert']
plain_add    ['stelling_any', 'add', 'gt', 'stelling_assert']
```

Three of the four produce a `jit` equation and only the first asked for
one. That is also where the quickstart's `1 transparent` comes from — its
harness calls `jnp.roll`.

The word also appears in one refusal, where it matters: a declaration
made *inside* a transparent wrapper cannot be reached by the vacuity
widening, which rewrites top-level declarations. The verdict says so
instead of claiming something about an envelope it did not widen:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    @jax.jit
    def declare_inside():
        return any_array((), jnp.float64, (0.1, 10.0))
    return assert_(declare_inside() > 0.0)


v = check(harness, vacuity_mode="inputs-only")
print("status:", v.status)
for a in v.stamp.assumptions:
    if a.startswith("vacuity"):
        print("assumes:", a)
```

```
status: VERIFIED
assumes: vacuity instrument inert (mode=inputs-only): a declaration sits 1 transparent call(s) below top level, where widening cannot reach it — so the envelope's role in this verdict is left uncharacterised
```

Declare at the top level of the harness and this does not arise.

## The stamp, line by line

Using the render from the [quickstart](quickstart.md#1-the-smallest-harness-that-says-something):

| line | what it records |
|---|---|
| `stelling … \| jax …` | the versions that produced the verdict |
| `query <hash>` | content hash of the transcribed query — covers the **declarations**, not just the program |
| `arithmetic:` | how endpoints are computed (`interval/f64/outward-1ulp`) |
| `semantics:` | which arithmetic the verdict is *about* — `real (ℝ)` by default, `ieee` opt-in |
| `precision:` / `device:` | the live `jax_enable_x64` at trace time; any concrete execution relied on |
| `solver:` | every invocation (name, version, transport, exact option set) — or a recorded **absence with its reason** |
| `nonvacuity:` | is the declared box tied to data you run on? (below) |
| `transfers:` / `provenance:` | which transfer was used per primitive, at what tier, from where |
| `assumes:` | every assumption the verdict rides on, including the vacuity line (below) |
| `coverage:` | equations known / ⊤ / transparent / dropped / constrained |
| `coverage-not-established:` | present only when the line above did **not** bound the query: a ⊤ was propagated anyway (below) |
| `note:` | the addresses — where and why anything degraded |

`arithmetic` and `semantics` are separate fields on purpose. `arithmetic`
is the representation; `semantics` is the claim. A default verdict says
`real (ℝ)`, and a predicate can hold in ℝ while failing in the float
program you actually run — see
[preconditions.md](preconditions.md#what-this-checks--and-what-it-doesnt-yet)
for a two-primitive case where that gap is the whole answer.

## `coverage-not-established:` — what the `coverage:` line did not settle

`coverage:` is a **census**. It counts whether each equation's primitive
has a registered transfer, and `5 known (100%)` means every one of them
does. It does not count what those transfers *returned*.

A registered transfer can run, succeed, and hand back `[-inf, inf]`.
Nothing on the `coverage:` line moves when it does. Measured, on
`exp(x) - exp(x)` over x ∈ [-1000, 1000] — five equations, every
primitive registered:

```
coverage: 5 eqns: 5 known (100%)
coverage-not-established: NOT ESTABLISHED — that the coverage line bounded this query. 1 propagated value(s) came out ⊤ (every element [-inf, inf], the widest box there is), at sub ×1, while the census recorded no equation fallen to ⊤ and none unreached. A registered transfer can return ⊤ on the values it is handed, and the census counts whether a primitive HAS a transfer registered, never what the transfer returned — so the 5/5 figure is not a statement that anything here was bounded, and this verdict does not make one
```

The `exp` overflowed to `[0, inf]`, and subtracting a box from itself is
`[-inf, inf]` — interval arithmetic does not know the two operands are
the same value. 100% coverage, and the analysis knows nothing about the
result.

Reading the line:

- the leading count is **propagated values**, not equations and not
  primitives — three ⊤ boxes at two primitives reads `3 propagated
  value(s) … at sub ×2, add ×1`;
- the tail after `at` is per-primitive, attributed to the equation that
  produced each box;
- `<constant or closure const>` there means a ⊤ that no equation in this
  jaxpr produced.

The line appears **exactly when** two things hold: the census reported no
gap of its own (`unknown`, `unreached` and `inert` all zero), and the
propagation still produced a top-level ⊤ box. When the census *did*
report a gap, the `coverage:` line already carries the disclosure — a ⊤
count, an `unreached` count, a `DROPPED` count — and this line would only
restate it under a stronger-sounding name.

**Its absence asserts nothing.** It is not the complement of its
presence: a ⊤ inside a `cond` branch that was scoped and discarded is not
visible to it, and neither is any query whose census had a gap of its
own. Absence means this particular reading did not fire — never that the
query was bounded.

It is a separate line rather than an addition to `coverage:` because that
string is trend data: `stamp.coverage.split(" eqns")[0]` is parsed by
`reproduce.py` and by the sweep scripts, and the disclosure must not move
it a byte.

## `PORTFOLIO DEGRADED` — when a verdict rests on one solver

The `solver:` line records **every invocation**, which is who stelling
*asked*. It is not who *answered*. A backend can be installed, invoked,
stamped, and still contribute nothing — its transport crashed, it timed
out, or its parser refused the emitted fragment. When that happens the
stamp still reads `2 invocation(s)` and the decision still rests on one.

That gap matters most on a **VERIFIED**. A REFUTED carries a witness, and
a witness is re-derived in exact rational arithmetic before it is
believed, so a lost backend there costs a cross-check another mechanism
still performs. A VERIFIED is a universal claim over the whole declared
box: there is no point to replay and nothing downstream re-derives it, so
the second backend *is* the redundancy.

So a decided obligation that got fewer than two answers says so, in three
places:

- a `PORTFOLIO DEGRADED` line near the top of the render, above the
  obligations, naming the assert and the one backend that answered;
- the same fact bracketed onto that obligation's own `assert #N:` detail
  line, with `a discharge has no replay backstop` appended when the
  outcome was a discharge;
- `note:` lines naming which backend was lost and why — invoked and
  returned `failed`/`timeout`/`unknown`, not installed, or excluded by
  `SolverConfig.only`.

For **counting** rather than reading, `Verdict.solver_redundancy` is
`(assert index, labels of the backends that answered)` per
solver-decided obligation:

    one_backend = [i for i, who in v.solver_redundancy if len(who) < 2]

This is not a soundness gate and it does not change any verdict. A
one-backend VERIFIED is still a VERIFIED — it just got half the
redundancy the portfolio is designed around, in a form that survives
being tallied by a CI job rather than read by a person.

## The two vacuity instruments

This is the part that reads as self-contradictory, and is not. A verdict
can print, in the same render:

```
assumes: vacuity checked (mode=inputs-only): no obligation discharges with the declared bounds widened — …
note: nonvacuity UNCHECKED: this VERIFIED may be vacuous — the declared set is not tied to the incident's data
```

These are **two different instruments asking two different questions**.
They share the root "vacu-" and nothing else.

| | obligation vacuity | nonvacuity |
|---|---|---|
| **question** | did the declared box do any work, or is the obligation true regardless? | does the declared box contain the data you actually run on? |
| **appears as** | an `assumes:` line beginning `vacuity checked` / `vacuity instrument inert` | the `nonvacuity:` stamp field, plus a `note:` when it is not `checked` |
| **how it runs** | automatic: on a VERIFIED, `check()` re-runs the identical query with declared bounds widened to (−inf, +inf) | only if your harness calls `nonvacuity(...)` |
| **you control it with** | the required `vacuity_mode` argument | writing membership conditions |
| **failing looks like** | `obligation #0: discharges with all declared bounds widened (vacuity mode=inputs-only) — envelope not load-bearing` | `nonvacuity: UNCHECKED` / `undecided` / `FAILED` and the may-be-vacuous note |

So in the render above — `vacuity checked … no obligation discharges`
beside `may be vacuous` — the two lines mean: *the widening re-check ran
and the obligations did not survive without the declared bounds; nobody
has said whether those bounds contain your data.* Both are true at once.

Read the whole vacuity line, not its first two words: **`vacuity
checked` prefixes both outcomes.** `vacuity checked … no obligation
discharges` is the reassuring one; `vacuity checked … obligation(s) (N,)
discharge with the declared bounds widened` is the flag, and it is
measured [below](#what-the-vacuity-line-does-and-does-not-say).

### Clearing `nonvacuity: UNCHECKED`

It is reachable through the documented API, and the whole of it is:
**call `stelling.harness.nonvacuity(...)` in your harness.** Nothing else
moves the field — no argument to `check()`, no mode, no flag. *Calling*
it is what counts: measured, an un-returned membership condition is
recorded just the same
([the harness API](harness-api.md#a-statement-counts-once-you-call-it)
has the run). Return
it alongside your obligations anyway; that is the convention. The
quickstart's
[§3](quickstart.md#3-tying-the-box-to-real-data) is the worked version;
the [spellings section](harness-api.md#three-spellings-and-the-one-that-behaves-differently)
is what to write and what not to.

Two things worth knowing before you rely on it:

* `stelling.preconditions` does **not** re-export `nonvacuity`. If
  `from stelling.preconditions import check` is your only import, the
  state is unreachable; add `from stelling.harness import nonvacuity`.
* `nonvacuity` moves the stamp field only. The verdict's status is
  unchanged — even `FAILED` leaves a VERIFIED verdict VERIFIED, because a
  membership condition that is definitely false is a **harness defect**,
  not a fact about the box. It is reported loudly and left for you to
  fix.

### Choosing `vacuity_mode`

`vacuity_mode` is required — the two registered procedures answer
different questions and a silent default would let a harness run the
wrong one without saying so.

* `"inputs-only"` widens only **non-point** declarations. Point
  declarations (`lo == hi`) are stated constants — thresholds,
  transcribed values, operating points — and hold still.
* `"all"` widens every declaration, point ones included.

**Measured, and it decides which you can use:** the instrument only
emits a `vacuity checked` line if widening moved **every** declared
bound. Under `"inputs-only"` a point declaration by construction does not
move, so any harness containing one reports `vacuity instrument inert`
rather than a result — as in the quickstart's §3, where the operating
point is a point declaration. The refusal is deliberate: the alternative
is claiming an envelope was widened when part of it was not. The
consequence for you is concrete:

| your harness | `inputs-only` | `all` |
|---|---|---|
| no point declarations | reports | reports |
| any point declaration (threshold, operating point) | **inert** — no result either way | reports, but the threshold widens too |

<a id="the-four-ways-the-instrument-goes-inert"></a>

**The four ways a VERIFIED comes back inert**, all measured — a point
declaration is only the one you will meet first:

```python
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def point_declaration():
    threshold = any_array((), jnp.float64, (1.0, 1.0))
    a = any_array((), jnp.float64, (0.1, 2.0))
    return assert_(a + threshold > 0.0)


def below_a_transparent_wrapper():
    @jax.jit
    def inner():
        return any_array((), jnp.float64, (0.1, 10.0))
    return assert_(inner() > 0.0)


def no_declarations_at_all():
    return assert_(jnp.float64(1.0) > 0.0)


def already_unbounded():
    a = any_array((), jnp.float64, (-math.inf, math.inf))
    return assert_(jnp.maximum(a, 0.0) >= 0.0)


for h in (point_declaration, below_a_transparent_wrapper,
          no_declarations_at_all, already_unbounded):
    v = check(h, vacuity_mode="inputs-only")
    reason = [a for a in v.stamp.assumptions if a.startswith("vacuity")][0]
    print(f"{h.__name__:27s} {v.status:8s} "
          f"{reason.split(': ', 1)[1].split(' — ')[0]}")
```

```
point_declaration           VERIFIED declaration 1 is a point interval (1.0), so this mode widens nothing on it; mode='all' would also widen transcribed constants
below_a_transparent_wrapper VERIFIED a declaration sits 1 transparent call(s) below top level, where widening cannot reach it
no_declarations_at_all      VERIFIED the query declares no bounded inputs, so widening changes nothing
already_unbounded           VERIFIED declaration 1 kept its bounds (-inf, inf) -- the rewrite did not reach it; the envelope was not fully widened
```

Only the first is mode-dependent. The other three are inert under `"all"`
too — a declaration the rewrite cannot reach, a query with nothing to
widen, and a declaration that is already `(-inf, inf)` are all "no
declared bound moved", whichever mode asked.

`"all"` is not simply the stronger mode. Widening a threshold makes
almost any comparison straddle, so an obligation that *is* a tautology
can fail to re-discharge under `"all"` and go unflagged. Measured on
`max(a, t) >= t`, a tautology for every `a` and every `t`, with `t`
entering as a point declaration:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    a = any_array((), jnp.float64, (0.5, 2.0))
    t = any_array((), jnp.float64, (0.0, 0.0))     # a stated threshold
    return assert_(jnp.maximum(a, t) >= t)


for mode in ("inputs-only", "all"):
    v = check(harness, vacuity_mode=mode)
    for line in v.stamp.assumptions:
        if line.startswith("vacuity"):
            print(line)
```

```
vacuity instrument inert (mode=inputs-only): declaration 2 is a point interval (0.0), so this mode widens nothing on it; mode='all' would also widen transcribed constants — so the envelope's role in this verdict is left uncharacterised
vacuity checked (mode=all): no obligation discharges with the declared bounds widened — under the mechanism(s) that ran, this VERIFIED was not re-derivable without the declared envelope
```

Neither mode caught this tautology: `"inputs-only"` went inert on the
point declaration, and under `"all"` the widened `t` made the comparison
straddle so the obligation did not re-discharge. Read the two lines for
exactly what they say — `vacuity checked` is evidence in the direction it
states, `vacuity instrument inert` is the absence of evidence, and
neither is a certificate that the obligation is not a tautology.

`vacuity_mode` is enforced at the call, not deep in the run:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    a = any_array((), jnp.float64, (0.1, 10.0))
    return assert_(a > 0.0)


try:
    check(harness)
except TypeError as e:
    print("TypeError:", e)
try:
    check(harness, vacuity_mode="input-only")
except ValueError as e:
    print("ValueError:", e)
```

```
TypeError: check() missing 1 required keyword-only argument: 'vacuity_mode'
ValueError: widen mode must be one of ('all', 'inputs-only'), got 'input-only'
```

### What the vacuity line does and does not say

The wording is deliberately narrow. `no obligation discharges with the
declared bounds widened` is a **measurement**, not the inference
"therefore the envelope is load-bearing" — a range theorem whose widened
mechanism cannot re-derive it produces the same measurement. Where the
re-check ran with less power than the original run, the line appends that
disclosure itself. Measured, on an obligation only the opt-in affine
refinement can decide:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    a = any_array((), jnp.float64, (0.1, 10.0))
    return assert_((a - a) >= 0.0)


for refine in (None, "affine"):
    v = check(harness, vacuity_mode="all", refine=refine)
    print("refine =", refine, "->", v.status)
    for line in v.stamp.assumptions:
        if line.startswith("vacuity"):
            print("   assumes:", line)
```

```
refine = None -> UNKNOWN
refine = affine -> VERIFIED
   assumes: vacuity checked (mode=all): no obligation discharges with the declared bounds widened — under the mechanism(s) that ran, this VERIFIED was not re-derivable without the declared envelope. The re-check ran weaker than the original (interval-only: the affine refinement declines unbounded boxes by construction), so envelope-independence of the affine-decided obligation(s) was not measured; an explicit solver_timeout_ms measures it.
```

The other direction is the one to act on: an obligation that **still
discharges** with its bounds gone never depended on your envelope. The
status stays VERIFIED — the claim is true — and both a per-obligation
note and the stamped line say so:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    a = any_array((), jnp.float64, (0.1, 10.0))
    return assert_(jnp.maximum(a, 0.0) >= 0.0)     # true for every a


v = check(harness, vacuity_mode="inputs-only")
print("status:", v.status)
for n in v.notes:
    print("note  :", n)
for line in v.stamp.assumptions:
    if line.startswith("vacuity"):
        print("assumes:", line)
```

```
status: VERIFIED
note  : nonvacuity UNCHECKED: this VERIFIED may be vacuous — the declared set is not tied to the incident's data
note  : obligation #0: discharges with all declared bounds widened (vacuity mode=inputs-only) — envelope not load-bearing
assumes: vacuity checked (mode=inputs-only): obligation(s) (0,) discharge with the declared bounds widened to (-inf, inf) — the verdict does not depend on the declared envelope for them (a range theorem, or a mis-posed envelope)
```

A CI consumer should treat `envelope not load-bearing` as a flag: here
the obligation is a theorem — `max(a, 0) >= 0` for every `a` — and the
declared envelope did no work at all.

## Reading an UNKNOWN

An UNKNOWN always says why, in the notes and in the coverage line. Work
in this order:

1. **The coverage line.** `1 ⊤ across 1 primitives (sort ×1)` names the
   primitive with no transfer. If it says `reduce_and`, you wrote
   `jnp.all(...)` — see
   [the harness API](harness-api.md#three-spellings-and-the-one-that-behaves-differently).
2. **`DROPPED` in the coverage line.** An assumption could not be
   honoured; the query ran over a superset. The note quotes what was
   dropped and why.
3. **The notes.** A declined transfer quotes its own reason, including
   which conversion or which budget it exceeded.
4. **`solver: none — … escalation was NOT ATTEMPTED`.** No solver has
   seen the query at all. Pass `solver_timeout_ms=<ms>` and re-run. The
   render says this itself on interval-only runs: *"This is NOT a finding
   that the property is undecidable."*
5. **Boundary tightness.** An obligation whose threshold is not exactly
   representable can land in the rounding gap; see
   [preconditions.md](preconditions.md#state-thresholds-as-representable-values-where-you-can).
6. **Compound boolean conditions.** A `jnp.where(cond1 & cond2, ...)` goes
   UNKNOWN when either `cond1` or `cond2` is undecidable, OR when the
   primitive for `&` has no transfer. Since 0.2.0 the boolean logic
   transfers (`and`, `or`, `not`) are registered, so two decidable
   predicates combined with `&` or `|` produce a decidable result. If the
   UNKNOWN persists after upgrading, one of the component predicates is
   genuinely undecidable over the declared box.
7. **The dependency problem (correlated conditions).** Interval arithmetic
   evaluates each operand independently. If the same variable appears in
   two places — e.g., `cond & ~cond`, or two `jnp.where` calls guarding
   on the same predicate — the tool has no memory that they share a source
   and cannot prove their logical relationship. The result is a safe but
   imprecise UNKNOWN. **Remedy:** pass `solver_timeout_ms` — the SMT
   encoding is inherently relational and WILL prove the mutual exclusion.
   This is the designed escalation path for constraints the interval domain
   cannot express.

## Further

- [SOUNDNESS.md](../SOUNDNESS.md) — what a verdict is permitted to claim,
  and the trust policy behind the stamp.
- [docs/verdict-ledger.md](verdict-ledger.md) — how a recorded verdict
  that moves gets accounted for.
