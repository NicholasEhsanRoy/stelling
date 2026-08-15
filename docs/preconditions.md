# Checking the preconditions your solver assumes

> **New to stelling?** Read the [Quickstart](quickstart.md) first — it
> covers the harness primitives, their import path, and how to read the
> verdict this page prints. The reference pages are
> [The harness API](harness-api.md) and
> [Reading a verdict](reading-a-verdict.md).

Numerical methods rest on properties their implementations assume and
never check: the coefficient field a CG or Cholesky solve needs to be
positive, the mass/shift scalar that removes a nullspace, the argument a
`log`/`sqrt`/division needs to be in-domain, the operating envelope a
scheme is only valid inside. These live in the **inputs** — before the
expensive computation — and when they fail, they fail **silently**: the
solver converges to something, the flags look healthy, and the answer is
wrong.

Install: `pip install stelling` into the environment that already has your
JAX (stelling never touches your resolver); add `pip install
stelling[solvers]` if you want the SMT step below.

Stelling checks these **over a declared range, not at a point**. A test
runs your code at some inputs; a verdict here holds for *every* value in
the envelope you declare — including the corner your test suite never
visits.

## Two worked examples

Check that a variable diffusion coefficient built from your own
construction is positive everywhere over the parameter range your
application can produce:

<!-- doc-example: run-only -->
```python
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from stelling.preconditions import field_positive, check

def face_coefficients(theta):
    # YOUR code's construction, not a simplified stand-in — e.g. the
    # conservative face averages a finite-volume Laplacian actually uses:
    a = theta                                  # your parameter -> coefficient path
    a_plus  = 0.5 * (a + jnp.roll(a, -1))
    a_minus = 0.5 * (a + jnp.roll(a, 1))
    return a_plus, a_minus                     # one obligation per value

def harness():
    _, obligations = field_positive(
        (64,), "float64", (1e-6, 1e2),         # the envelope you support
        transform=face_coefficients,
    )
    return obligations

print(check(harness, vacuity_mode="inputs-only").render())
```

Check that a configuration scalar can never be the singular value —
posed over the **admissible range**, not just the default, because the
question is whether your configuration space *admits* the bad value:

<!-- doc-example: run-only -->
```python
import jax
jax.config.update("jax_enable_x64", True)

from stelling.preconditions import scalar_nonzero, check

def harness():
    _, obligation = scalar_nonzero("float64", (0.0, 1.0))   # the config range
    return (obligation,)

# an explicit time budget opts in to the SMT step (never on by default):
print(check(harness, vacuity_mode="inputs-only", solver_timeout_ms=20_000).render())
```

## Reading the verdict

`vacuity_mode` is required — `"inputs-only"` is the standard choice
(your declared ranges widen; transcribed constants hold still). On a
VERIFIED, check() re-runs the query with the bounds widened: if an
obligation still discharges with the bounds gone, the verdict tells
you (a note and a stamped line) that the envelope was not
load-bearing — the claim is a theorem or the envelope is mis-posed.
A VERIFIED from check() has always been *put through* this check, but it
does not always come back with a result: when not every declared bound
moved, the instrument reports `vacuity instrument inert` instead and the
envelope's role is left uncharacterised. There are four measured ways
that happens — a point declaration under `"inputs-only"`, a declaration
below a transparent wrapper, a query with no declarations at all, and a
declaration that is already `(-inf, inf)`. Read the stamped line rather
than assuming which of the two you got; all four are measured in
[Reading a verdict](reading-a-verdict.md#the-four-ways-the-instrument-goes-inert).

- **VERIFIED** — the property holds at every point of the declared
  envelope, judged by outward-rounded interval arithmetic (and the SMT
  step, if you opted in). The stamp on the verdict names every
  assumption this rests on: versions, the query's content hash, the
  arithmetic semantics, precision configuration, solver invocations or
  their recorded absence, and coverage.
- **REFUTED** — the property definitely fails. When the SMT step found
  it, the verdict carries a **concrete witness** — the input value at
  which your precondition breaks — independently confirmed by exact
  rational replay before it is shown to you.

  **What that confirmation covers, and what it does not.** Replay is
  independent of the **solver**, not of the **translation**: it
  re-derives the violation in exact rational arithmetic through the same
  routing the SMT emission used. So it establishes that the solver did
  not fabricate the witness, and it does *not* establish that the
  program was translated faithfully — a defect in the translation is
  re-derived identically and the witness is stamped confirmed. This is
  measured rather than hypothetical.

  The check that *is* translation-independent is **running the witness
  through your own program**. It shares no code with either the emission
  or the replay, so it is the one that catches a translation defect, and
  it is worth doing for any witness you intend to act on.
- **UNKNOWN** — the analysis could not decide, and says why: the notes
  quote exactly what was dropped, declined, or too wide. An UNKNOWN is
  never silently rounded to either answer.

The second example REFUTES with witness `0`: the configuration range
admits the singular value and nothing in the code forbids it. That is a
statement about your config space, not about your default.

## What this checks — and what it doesn't (yet)

**In scope:** *input-side* preconditions — pointwise or scalar
properties of the data your solve consumes, stated over declared
envelopes. This is deliberately the portable core: no adaptivity, no
mesh machinery, no method internals.

**Out of scope today, stated plainly:**

- **Properties of the solve's behaviour** — conditioning over the
  envelope, residual-implies-error — are a different, planned layer.
  The boundary is exactly the solve: this module checks what goes *in*.
- **Array obligations escalate for small static shapes.** Interval
  propagation judges array obligations elementwise (the first example
  is an array, and VERIFIED means every element), and the SMT step now
  takes **small, statically-shaped** array obligations too — one
  per-obligation emission budget (512 element terms and root
  conjuncts) with anything above it declining, the numbers quoted. A
  REFUTED array obligation's witness names **which element violates**,
  with per-element values, replay-confirmed before you see it. General
  dynamic or arbitrarily-large array reasoning stays out of scope — the
  sound, tractable thing is bounded static-shape emission, and the
  budget line in the decline tells you when you have left it.
- **Float-exact claims by default.** The default verdict is about exact
  real arithmetic over your declared sets, and the stamp says so; an
  opt-in `ieee` mode judges IEEE behaviour in any of the four catalogued
  formats and stamps its own scope. If your precondition is a
  float-boundary fact, read the stamp's semantics line before trusting
  either mode blindly.
- **Judge `exp` or `pow` under `ieee` without being told what you are
  assuming.** Those two transfers ride a *libm accuracy* claim, and under
  `ieee` the claim has to be about the function your backend executes —
  which stelling cannot see. They **decline by default** and you re-enable
  them by declaring a budget; see
  [the libm accuracy budget](#the-libm-accuracy-budget-exp-and-pow-under-ieee).

### What float-exact-by-default costs you, concretely

**Two primitives are enough.** This is not a corner case reachable only
through exotic arithmetic:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    x = any_array((3,), "float32", (1.0 - 1e-12, 1.0 + 1e-12))
    return assert_((x[0] + jnp.float32(1e-9)) - x[0] > 0.0)


print("verdict :", check(harness, vacuity_mode="all").status)
x = jnp.full((3,), 1.0, dtype=jnp.float32)
print("executed:", (x[0] + jnp.float32(1e-9)) - x[0])
```

prints:

```
verdict : VERIFIED
executed: 0.0
```

In exact real arithmetic the expression is `1e-9`, so the predicate is
true and the default (`real`) verdict of VERIFIED is correct *for what it
claims*. Executed in float32 it is exactly `0.0` at every point of that
envelope, so the predicate is **false everywhere the program actually
runs**. Measured on jax 0.11.0; the same VERIFIED-against-executed-zero
appears for `jnp.sum(k * x)` and for `jnp.matmul(k, x)`, so it is a
property of the semantics dial rather than of any one primitive or row.

The verdict is not wrong — it says `real` on the stamp, and in ℝ the
claim holds. But **"VERIFIED" plus a `real` semantics line does not mean
"this holds when you run it"**, and the gap can be the whole value rather
than a last-ulp difference. If the property you care about is one the
floating-point execution must satisfy, either pose it in `ieee` mode or
state the threshold as a representable value (see below) — do not read a
`real` VERIFIED as a statement about the float program.

Each limit shows up in the verdict itself — as a quoted decline, an
UNKNOWN with its reason, or a stamped assumption — never as a silent
pass. That is the design: the tool tells you when it could not earn the
claim, so a VERIFIED means exactly what it says.

## The libm accuracy budget: `exp` and `pow` under `ieee`

Under `semantics="ieee"` a verdict is a claim about **the float value your
program computes**. stelling brackets `exp` by evaluating CPython's
`math.exp` — the libm of the machine running the analysis — and bumping
one ulp outward. Your program does not run that function. It runs whatever
XLA compiled for your device, and a bracket of one function is not a
bracket of another.

Measured on jax 0.11.0 / jaxlib 0.11.0, CPU, x86_64, the gap is real and
it is not small: exhaustively over every `float32` argument whose `exp` is
normal and finite, XLA's result is up to **5.51 float32 ulps** from the
true value — so it is not faithfully rounded at all, and no fixed widening
of the bracket can be sound. On the *same* backend, `bfloat16` `exp` is
exhaustively **correctly rounded** over every normal finite result, and
`float16` misses correct rounding on 2 of its 63,487 arguments by 3e-5 of
an ulp — both are evaluated in `float32` and rounded. A factor of eleven
between two formats of one op: no single number is right for all four.

**Not every row is exhaustive, and the difference matters.** `exp` is
measured exhaustively in `float16`, `bfloat16` and `float32`. `exp` in
`float64` and **all four `pow` budgets are SAMPLED** — they bound what was
sampled and nothing more, and an independent draw of the same size has
already beaten one of them (`exp@float64` measured 1.6470 on one 3,000,000
draw and 1.6660 on the next). This is why the shipped budgets round up to
the next integer rather than to the measured figure, and it is why the
per-row population is written out in `stelling.propagate.LIBM_MEASURED`
rather than summarised as a single number: read the row, not the maximum.
Every `exp` row is also restricted to arguments **whose result is normal
and finite** — a result that underflows is flushed to zero by this backend
(108.7 ulps in `bfloat16`) and is covered by the subnormal haze, not by an
accuracy budget.

So the two transfers that ride the claim **fail closed**:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


def harness():
    x = any_array((), "float32", (1.0, 2.0))
    return assert_(jnp.exp(x) > 0.0)


shut = check(harness, vacuity_mode="inputs-only", semantics="ieee")
print("no budget   :", shut.status)
print("names it    :",
      "no DECLARED accuracy budget for float32" in shut.notes[0])

open_ = check(harness, vacuity_mode="inputs-only", semantics="ieee",
              libm_budget="xla-cpu-2026-08")
print("declared    :", open_.status)
print("stamped     :",
      any("DECLARED, NOT VERIFIED" in a for a in open_.stamp.assumptions))
```

prints:

```
no budget   : UNKNOWN
names it    : True
declared    : VERIFIED
stamped     : True
```

The decline is not a wall — it carries the measurement that justifies it
and a line that **runs as written** (both `check()` and `propagate()`
accept `libm_budget=`, and the decline names both).
`"xla-cpu-2026-08"` is a shipped profile: a
**named, dated** set of per-`(op, format)` budgets measured on one jaxlib,
on one device class, on one day. The name is what a stamp carries, so when
jaxlib moves the name stays honest about what it measured and when, which
a bare number can never be.

To declare your own — because you measured your backend, or because you
are willing to assume something about it:

<!-- doc-example: illustrative -->
```python
from stelling.propagate import LibmBudget

check(harness, vacuity_mode="inputs-only", semantics="ieee",
      libm_budget=LibmBudget(
          name="my-backend-2026-08",
          basis="measured over 10^7 arguments on <device>, <date>",
          ulps={("exp", "float32"): 6.0},
      ))
```

A budget is **per `(op, format)` and never extrapolated**: one naming
`("exp", "float64")` does nothing for `float32` `exp`, and a pair it does
not name declines exactly as if no budget had been passed.

`0.5` ulps **is read as the declaration "correctly rounded"**, and it
costs nothing at all — the bracket is not widened by a single step,
because round-to-nearest is monotone and the endpoints are rounded onto
the format's grid anyway. `1` ulp means *faithfully rounded* and does
cost. This is the same line `interval.sqrt` already draws for itself:
sqrt is a correctly-rounded IEEE-754 basic operation, so it carries no
libm demotion, needs no budget, and is unaffected by any of this.

One pedantry, because the whole design rests on the declaration meaning
what the verdict assumes: an ulp here is the spacing of the binade
*containing* the value, so `ulp(2**k)` is the spacing **above** `2**k`
while the float **below** it is only half such an ulp away. Read as a
raw inequality, `0.5` would therefore also admit a backend returning
`nextdown(2**k)` where the true value is exactly `2**k` — which correct
rounding does not, and `exp(0) = 1.0` reaches. `0.5` is read as the
stronger claim, correct rounding, which is what the no-widening branch
needs; if your backend is only *nearly* correctly rounded, declare `1`,
which is what the shipped profile now does for `float16` `exp`.

**What the stamp then says, and it is the whole point:**

> ieee libm accuracy **DECLARED, NOT VERIFIED** — profile
> `'xla-cpu-2026-08'`: exp@float32 <= 6 ulps. … Claim (2) is a
> DECLARATION about a compiled function stelling cannot see, execute or
> measure: if the target is worse than declared, the bracket may EXCLUDE
> the value the program computes, and a VERIFIED resting on it is FALSE
> with nothing here able to notice.

A budget smaller than your backend's real error mints a VERIFIED stelling
cannot catch. That is why the number has to be written down by a person,
with a name and a basis, instead of defaulted to by the tool.

`semantics="real"` is untouched by all of this and needs no budget: there
the bracket is about the true real value, the host's own `math` module
does satisfy the ±1-ulp assumption it rides on, and the divergence from
your compiled program is the ℝ-versus-float gap the stamp already names.
Passing `libm_budget` under `real` raises rather than being ignored.

## Precision configuration

`check()` records the live `jax_enable_x64` state in the stamp. If your
code assumes float64, enable it before tracing (as in the examples) —
and note that stelling checks the obligation under the configuration the
trace *ran under*, which is only meaningful if it matches the
configuration your production code runs under.

## Posing guidance — what not to pose (learned from the field)

Three rules from the first unguided field tests, each of which removed a
class of false alarms without silencing a single real finding:

1. **Pose inputs, not loop state.** A denominator produced by the
   iteration and fed back (a Lanczos/bidiagonalization `rho`, a running
   residual) is not an input — it is constrained by the algorithm's own
   invariants, and posing it as a free box manufactures a false REFUTED.
   The class boundary is the solve: if the quantity exists only inside
   the loop, its guarantees are the algorithm's business, not an input
   precondition.
2. **A tag can be an instruction, not an assertion.** `unit_diagonal`
   tells the backend to *ignore* the stored diagonal — the stored values
   carry no claim, so posing "diagonal == 1" flags nothing real. Before
   posing a tagged property, read the downstream use: if the tagged data
   is never read, there is no precondition. (A tag that *asserts* a
   property the code then relies on — `well_posed=True` guarding an
   unchecked division — is the opposite case, and exactly what should be
   posed: asserted-never-verified is the class's home ground.)
3. **Sentinels are values, not hazards.** `conlim = 0` meaning
   "disabled" and `rcond = 0` meaning "cut only exact zeros" are
   documented conventions; pose the precondition over the non-sentinel
   range, or at the guard's own strength (`>= 0`, not `> 0`), or the
   alarm is about the convention, not the code.

## State thresholds as representable values where you can

Interval endpoints are computed by correct directed rounding, which returns the
exact result **unchanged whenever it is representable as a double**. A threshold
that is itself a double — `0`, `1`, `0.5`, any power of two, any small integer —
can therefore be met exactly at a boundary. A threshold that is not — `0.1`,
`1/3`, `0.7` — is stored as the nearest double, and an obligation that turns on
that boundary can land in the gap between the two.

Practically: prefer `x >= 0`, `dt <= 0.5`, `scale > 0` over `x >= 0.1` when the
physics gives you the choice. If the threshold is genuinely `0.1`, nothing is
wrong — but a boundary-tight obligation there is likelier to come back UNKNOWN,
and the fix is usually to state the bound you actually mean rather than to widen
the envelope.

This affects **boundary-tight** obligations specifically. An UNKNOWN with room to
spare on both sides is not this; look at the propagation notes instead.

## Reading a CI verdict: gate or triage

**A finding is a conjunction: the precondition is violated, AND the
violation has a silent consequence.** Stelling mechanizes the first
conjunct — the arithmetic it flags is real (a zero diagonal really does
produce `inf`). The second conjunct usually depends on context a local
obligation cannot see, and out-of-sample verification measured exactly
four ways it fails. Before treating a REFUTED as gate-grade, answer
four questions:

1. **Is the consequence caught downstream?** A framework may wrap every
   public call in a postcondition (e.g. "any nonfinite solution rewrites
   `successful → singular` and raises") — then the local unguarded
   division is not a leak, it is the *detection mechanism*. A
   "produces nonfinite" finding whose downstream catchability is
   unchecked is triage-grade.
2. **Could the value be a tracer?** In a JAX library, eager validation
   is necessarily best-effort: any value that can arrive as a tracer
   cannot be checked at construction, so a missing eager check on it is
   the *unvalidatable class*, not a defect. The sharp converse: a value
   that is static Python (`int`/`None` in Python-level branching — it
   would crash on a tracer) has **no tracer excuse**, and a missing
   check there is a genuine candidate.
3. **Is the violation actually harmful?** If every execution path from
   the violated precondition still computes a valid result (both
   branches of the conditional are correct; only the schedule changes),
   the precondition was for intent, not correctness.
4. **Is the failure actually silent?** A degenerate config that crashes
   at trace time, or that returns flagged as a breakdown, is loud —
   miscategorised at worst, not the silent class this checker exists
   for.

**Gate-grade = both conjuncts established at the same locality as the
pose** (a static-only value, no downstream rescue possible, a wrong
result under a success flag). Everything else starts triage-grade —
calibration, not suppression: the framework might *not* guard, the
value might be static, and then the finding is real.
