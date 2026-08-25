<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing a witness

When a verdict is `REFUTED` and carries a witness, stelling can write a
**runnable file** that builds that witness, calls your own function with
it, evaluates the asserted comparison, and prints both sides.

The file **does not import stelling**, and that is the whole point.

## Why the file must not import stelling

Every other check of a witness runs on stelling's own transcription of
your program. The interval propagation, the SMT emission and the
exact-rational replay all read the same traced query — so a defect in
that transcription is re-derived identically by each of them.
[`Witness`](../src/stelling/verdict.py)'s own docstring records the
measured case: an adversarial audit produced a witness on a trivially
true property, and replay confirmed it, "because both faces asked the
same wrong question".

Executing the witness through the real program is the one leg that
shares no code with either. A reproducer that reached back into stelling
would be the tool checking itself with the tool, so
`stelling.reproduce` refuses to emit a file whose text contains an
`import stelling` — checked on the emitted text, not promised by the
template.

## The three-valued result, and what it is not

The file reports exactly one of three, and **none of them is a verdict**.
The verdict was decided elsewhere; this is evidence *about* it.

| result | meaning |
|---|---|
| `CONFIRMED` | the comparison is false at the witness under execution, in at least one of the two modes the file runs (eager and `jax.jit`) |
| `DIVERGED` | it is false in exact real arithmetic, and TRUE in your program's own dtype in **every** mode, all of which must have run — a finding about the real/float gap, **not a failed check** |
| `UNREACHABLE` | the witness lies at a point your own caller precondition excludes — a caller-precondition result, **not a bug in the program** |

Exit status is `0` for all three. A nonzero status is how CI says "this
check failed", and two of the three are not failures of anything.

Exit `3` means **no execution result**: this file has nothing to report
about the program. That covers a target it could not construct or run —
and also a target that ran perfectly well but not in every mode, where the
assertion held where it ran and the other mode raised, so `DIVERGED` (a
claim of absence) is not available and nothing was false either. The
sidecar's `execution.detail` says which. **Read it, not the status.**

## A complete example you can run

Everything below is here. No other package, no fixture, nothing to stand in
for — `pip install stelling[solvers]`, two files, and the output is what it
prints, to the last line. (The one thing shown differently is the `sidecar:`
path, which is absolute in a real run and is written relative here.) A witness only exists because a solver found one, so the
solver extra is the part you cannot skip.

Install it into the environment that already has your JAX, and do **not**
add the `[jax]` extra: this page is about checking a JAX program, so you
have JAX already, and that extra exists only to bootstrap an environment
with none — using it puts stelling into a resolver that is currently
managing your JAX (and, on GPU, its plugin wheels). One backend alone will
run this page; [choosing a solver backend](choosing-a-solver-backend.md)
says what you give up by installing only one.

<!-- doc-example: illustrative -->
```python
# myprogram.py — your code. No stelling import here.
import jax.numpy as jnp


def total_against_budget(rates):
    """Four rates, scaled by a duty cycle, against a budget."""
    total = jnp.sum(rates) * 3.0
    return total, 10.0          # (lhs, rhs) — the two sides of `total <= 10.0`
```

<!-- doc-example: illustrative -->
```python
# check_it.py — the harness module. This one DOES import stelling.
import jax
jax.config.update("jax_enable_x64", True)

from stelling.preconditions import check
from stelling.reproduce import Subject, write_reproducer

from myprogram import total_against_budget

SUBJECT = Subject(
    name="rate-budget",
    fn=total_against_budget,
    relation="<=",
    declarations=((((4,), "float32", (0.0, 1.0))),),
    no_precondition_reason="every rate in [0,1] is one the caller can supply",
)

verdict = check(SUBJECT.harness, vacuity_mode="inputs-only",
                solver_timeout_ms=60_000)
print("status:", verdict.status)
if verdict.witnesses:
    print("wrote:", write_reproducer(verdict, SUBJECT, "reproducers").path)
```

Four rates each at most 1, scaled by 3, is at most 12 — so the budget of 10 is
violable, and interval arithmetic alone cannot decide it (`[0, 12]` straddles
`10`). The solver settles it:

```
$ python check_it.py
status: REFUTED
wrote: reproducers/reproduce_rate_budget_assert0.py
```

Now run the emitted file. **Nothing about this step needs stelling** — that is
the point of it:

```
$ python reproducers/reproduce_rate_budget_assert0.py

== the witness, exactly as the solver produced it
  x0[0] = 1
  x0[1] = 1
  x0[2] = 1
  x0[3] = 2/3
  x0 as float32: array([1.       , 1.       , 1.       , 0.6666667], dtype=float32)
  NOTE: these exact values are NOT representable in the
  declared dtype and were rounded to build the array:
    x0[3] = 2/3

== caller precondition: NONE DECLARED
  every rate in [0,1] is one the caller can supply
  So UNREACHABLE was not tested. This run assumes every
  point of the declared envelope is producible by some caller.

== executing YOUR function
  myprogram.total_against_budget

  [eager] lhs = array(11., dtype=float32)
  [eager] rhs = array(10.)
  [eager] asserted: lhs <= rhs   ->  False
  [eager] FALSE at flat element(s): [0]
    [0]  11.0 <= 10.0  is False   (margin +1.0)

  [jit] lhs = array(11., dtype=float32)
  [jit] rhs = array(10.)
  [jit] asserted: lhs <= rhs   ->  False
  [jit] FALSE at flat element(s): [0]
    [0]  11.0 <= 10.0  is False   (margin +1.0)

== CONFIRMED
  The asserted comparison is FALSE at the witness under eager, jit
  execution of your own function. This is the one check of the
  refutation that shares no code with stelling's emission or
  its replay.

sidecar: reproducers/reproduce_rate_budget_assert0.json
```

The witness is `2/3` — an exact rational the solver produced and `float32`
cannot hold. The file says so and rounds it rather than quietly substituting a
different point, and the violation survives the rounding.

## Factoring your program so a file can call it

A `Subject` names the declarations, an importable callable of them
returning `(lhs, rhs)`, and the relation between the two sides. It builds
the harness itself, so the query stelling judges and the call the emitted
file makes come from one object — and `write_reproducer` compares content
hashes before emitting, so a verdict can never be quoted over a program
it is not about.

Your program lives in its own module. The harness module imports
stelling; **the program module must not**, or the emitted file drags the
tool back in through the side door. The file checks `sys.modules` at each
step — after importing your program, after running your caller
precondition, and after running the target — and names the phase and the
callable that first loaded the tool, on every path that reports anything.

`write_reproducer` always re-traces the subject's own harness and compares
its content hash with the verdict's stamp. There is deliberately no way to
supply the traced query and skip that: a gate with a bypass is not a gate.

### A second example, against a real simulation package

Optional, and it costs something — see the warning below. The shape is the
same, and the target is a real simulation node imported from `maddening`
rather than anything written for the example.

> **Installing `maddening` will DOWNGRADE your jax.** `maddening` 0.3.x pins
> `jax>=0.4,<0.6`, and stelling is developed against 0.11. Use a separate
> virtualenv for this example unless you want that downgrade in the
> environment you verify from. The example above needs none of this.

```bash
pip install --no-deps maddening==0.3.0
```

**`--no-deps` is the point of that line, not a flourish.** A plain
`pip install maddening==0.3.0` resolves your jax down to the ~0.5 series to
satisfy `maddening`'s pin, and stelling then warns
`stelling is tested against jax 0.10, 0.11.x but is running under jax 0.5.1`
— `pyproject.toml`'s `[jax]` floor is `jax>=0.10` precisely because stelling
produces no useful verdicts below it. With `--no-deps`, `maddening` installs
against the jax you already have, which is what CI does for its own
reproducer acceptance
(`.github/workflows/ci.yml`: *"A plain `uv pip install maddening==0.3.1` …
DOWNGRADES jax … Hence `--no-deps`"*) and what produced the transcript below.

The pin is on `maddening`'s side and is expected to go away in its 0.4, after
which this example needs neither the flag nor a separate environment.

<!-- doc-example: illustrative -->
```python
# myprogram.py — your code. No stelling import here.
from maddening.nodes.heat import HeatNode

NODE = HeatNode("h", timestep=0.1, n_cells=4, thermal_diffusivity=1.0)


def heat_step_against_bound(T):
    out = NODE.update({"temperature": T}, {}, 0.1)["temperature"]
    return out, 100.0          # (lhs, rhs) — the two sides of `out <= 100.0`
```

<!-- doc-example: illustrative -->
```python
# check_it.py — the harness module.
import jax
jax.config.update("jax_enable_x64", True)

from stelling.preconditions import check
from stelling.reproduce import Subject, write_reproducer

from myprogram import heat_step_against_bound

SUBJECT = Subject(
    name="heatnode-maximum-principle",
    fn=heat_step_against_bound,
    relation="<=",
    declarations=(((4,), "float32", (0.0, 100.0)),),
    no_precondition_reason=(
        "the node's temperature state is declared over its own operating "
        "range and every point of it is one a driven trajectory occupies"
    ),
)

verdict = check(SUBJECT.harness, vacuity_mode="inputs-only",
                solver_timeout_ms=60_000)
if verdict.witnesses:
    print(write_reproducer(verdict, SUBJECT, "reproducers").path)
```

`HeatNode` is `maddening`'s own — nothing here stands in for it. Running the
emitted file:

```
== executing YOUR function
  myprogram.heat_step_against_bound

  [eager] lhs = array([ 63.125, 101.   , 101.   ,  63.125], dtype=float32)
  [eager] rhs = array(100.)
  [eager] asserted: lhs <= rhs   ->  False
  [eager] FALSE at flat element(s): [1, 2]
    [1]  101.0 <= 100.0  is False   (margin +1.0)
    [2]  101.0 <= 100.0  is False   (margin +1.0)

  [jit] lhs = array([ 63.125, 101.   , 101.   ,  63.125], dtype=float32)
  [jit] rhs = array(100.)
  [jit] asserted: lhs <= rhs   ->  False
  [jit] FALSE at flat element(s): [1, 2]
    [1]  101.0 <= 100.0  is False   (margin +1.0)
    [2]  101.0 <= 100.0  is False   (margin +1.0)

== CONFIRMED
  The asserted comparison is FALSE at the witness under eager, jit
  execution of your own function. This is the one check of the
  refutation that shares no code with stelling's emission or
  its replay.

sidecar: reproducers/reproduce_heatnode_maximum_principle_assert0.json
```

**Both modes are the program, and each gets its own inputs.** The file
builds fresh `jax` arrays per mode — a jax buffer is destroyable, and a
target using `donate_argnums` deletes its argument — then runs the target
eagerly *and* under `jax.jit`, because the compiler is entitled to rewrite
the expression and measurably does.

A violation observed in **either** mode is a `CONFIRMED`: it was executed.
`DIVERGED` is the opposite kind of claim — an absence — so it needs
**every** mode to have run and held. When nothing was false and a mode
could not run, there is no execution result, and the file says which mode
was missing and why. Reporting less is the trade this whole feature is
built on.

## When the target cannot be called

The test is **identity**: does some module-level name resolve to this
exact object? A plain `def`, a `@classmethod`, a module-level callable
instance (the flax/equinox shape) and a `functools.partial` bound at
module level all pass it. A lambda, a nested function, a `__main__`
target, a method bound to an instance no name holds, and a name that
resolves to a *different* object than the one you passed do not — and
each gets its own sentence saying which.

Measured across the MADDENING/MIME contract corpus, which is **not in this
tree** — see [state-0.1.0.md](state-0.1.0.md), which records the same
population and says the same thing about it — **8 of 14 targets need fixture
work** — a constructor argument, a mesh, a controller instance — before
any file can call them. That is the normal case, not the exception, and
the answer to it is a module-level wrapper that writes the construction
down where a file can import it (exactly what `heat_step_against_bound`
above does).

When even that is missing, the file is still written, and it *states what
it could not construct* rather than dying at an import line five frames
deep:

```
== NO EXECUTION RESULT
  This reproducer could not construct its target, so nothing was
  executed and nothing is claimed here in either direction.
  WHAT COULD NOT BE CONSTRUCTED: the target mypackage.Solver.step is a
  method BOUND to a Solver instance built at run time; the emitted file
  can import the name but not that instance. Wrap the construction in a
  module-level function, so the file can build the object the same way
  you did
```

## Reachability is a claim, never a measurement

`UNREACHABLE` says no caller can produce the witness point. That claim
must rest on a **structural argument about the quantity** — "these
weights come out of one normalization, so they sum to 1 by construction"
— and never on a span measured from a trajectory.

This project has got that wrong twice, both times the same way. `row7`:
the node's `initial_state()` sat outside the declared box, and that was
read as "the node never occupies the envelope"; a *driven* trajectory
reaches the box entirely, and the degenerate span came from stepping the
node with no boundary inputs. `RigidBody`: a velocity span of
`[-1.962, 0]` was measured over 200 steps and reported as the reachable
set; driven from the node's own `initial_state()` with no boundary inputs,
200 steps at `dt = 0.001` span `[-1.9619957208633423, 0]` and 100 000 steps
at the same `dt` span `[-981.4652709960938, 0]`. (`100000 × 0.001 × 9.81`
is 981.0, and that is not the figure: the velocity accumulates in float32
and the driven span runs past it. This sentence carried the arithmetic
answer, which is the error the sentence is about.) A measured span is
"reachable under the trajectory I ran", never "reachable in all
operation".

So `Subject` requires exactly one of `precondition` /
`no_precondition_reason`, a declared precondition must carry its
`precondition_reason`, and the emitted file prints that reason beside
every `UNREACHABLE` it reports. The claim cannot travel without the
argument it rests on.

If the precondition is one stelling should know about, state it in the
harness with `assume(...)` as well — then the narrowing is part of the
judged set rather than a fact discovered afterwards.

## The JSON sidecar

Running the file writes `<name>.json` beside it (or wherever
`STELLING_REPRODUCER_JSON` points).

### The schema is PROVISIONAL

**It is unstable, and you should not build on it without pinning the
stelling version.** Concretely: fields may be **added, removed or
renamed** in any release, **without a deprecation cycle**, and a consumer
written against one may simply stop working.

**It freezes on a CONDITION, not on a version number** — once a consumer
that did not write it has parsed real emissions and the fields have been
exercised from outside. Until the schema identifier stops saying
`provisional`, that has not happened, and the identifier is where to look:
a consumer comparing `schema` against the string it was written for fails
closed.

**And `provisional` here is a marking on this document format, not the
API-stability level of the same name.** `DOCUMENTATION_ARCHITECTURE.md`
§8.5 gives `provisional` as *"may change in minor with a deprecation
cycle"*, guarantee *"one minor's notice"* — more than the paragraph above
offers you, and more than this schema promises. What you get here is that
table's `experimental` in strength: any release, no deprecation cycle, no
notice. §8.5's *Applied to:* line does not name this artifact, so nothing
has assigned it a level at all; the word in the identifier is a state —
version 1, not frozen yet — and it goes away when the condition is met,
which is not a thing a stability level does. The distinction is worth a
paragraph because this release does teach those levels by name elsewhere:
[Checking preconditions](preconditions.md) and
[The harness API](harness-api.md) both assign `falsify` the level
`experimental` and quote §8.5 for what it means.

*This section named 0.1.1 as both the release fields could move in and the
release the schema would freeze in. 0.1.1 has been and gone, a second
release has arrived, the running version is `0.2.0`, and the schema is still
`stelling.reproducer/1-provisional` — so the promise had become one about a
release in the past, which reads either as "this froze and nobody updated
the page" or as an abandoned plan, and neither is true. Nothing about the
guarantee has changed; only how it is stated, so that it cannot expire
again.*

The reason it is not frozen now is that the argument for freezing it was
never an argument for freezing it *yet*. That argument ran: a CI coverage
line and an external soak parse this, therefore it is a published surface.
That is a case for stability *eventually* — nothing has parsed one of
these files in anger, so no field here has been tested by anyone but its
author, and "small and designed to survive" is a prediction until a
consumer has tried to live on it. Everything else in this feature can be
repaired in a patch release; a schema declared stable and then changed is
a break for whatever parses it.

You can tell from the artifact, without a changelog. The **identifier
itself** carries the marking, so the ordinary version check —
`doc["schema"] == "stelling.reproducer/1"` — fails closed rather than
succeeding against a guarantee that was never given. Every sidecar also
carries a `stability` field spelling out what it means and when it stops
being true, and the emitted file says the same in its header.

### The fields

| key | |
|---|---|
| `schema` | `"stelling.reproducer/1-provisional"` |
| `stability` | what "provisional" means here, and the freeze condition |
| `stelling`, `stelling_sha`, `jax`, `x64` | the versions and the precision setting that produced the verdict |
| `query` | the traced query's content hash |
| `contract`, `verdict`, `obligation`, `relation` | which claim this is evidence about |
| `fragment`, `equations` | the SMT logic emitted (or `null`), and the equation count |
| `envelope` | one `{name, shape, dtype, lo, hi}` per declaration |
| `witness` | `{input name: exact rational string}` — the point that was executed |
| `witness_filled` | the witness names this layer **invented** from the declared box, because the obligation never reaches them and no solver or replay ever assigned them a value |
| `execution` | `{result, detail, reachable, lhs, rhs, modes, sides_from}` |

Every numeric field is a JSON **number**, or one of the strings `"inf"`,
`"-inf"`, `"nan"` — JSON has no encoding for those, and the bare
`Infinity` token Python emits by default is rejected by `jq`, `JSON.parse`,
Go and serde. Both writers set `allow_nan=False`, so a path that forgets
this raises instead of producing something no consumer can read.

*That sentence was true of the runtime writer only until it was measured:
the emit-side writer, which produces the blocks embedded in the `.py`, had
no guard. No escape was ever demonstrated — the endpoints are sanitised
upstream — so it was a claim defect rather than a shown bug, and the guard
that makes the claim true is now on both.*

`execution.reachable` is `true`/`false` when a caller precondition was
declared and ran, and `null` only when none was declared — a measured
value is never published as an unknowable one.

`execution.result` is one of the three tokens, or `null` when this file has
no result to report — see exit `3` above; `execution.detail` says why.

`execution.modes` maps `eager` and `jit` to whether the assertion **holds**
there; both keys are always present, and `null` means that mode did not
run. It is a schema field rather than prose because the two measurably
disagree: XLA's algebraic simplifier rewrites `(1 + x) - 1` to `x`, so a
violation binary64 absorbs eagerly is present in the compiled program.

`execution.sides_from` names the mode `lhs`/`rhs` were read in. **The
published sides always come from a mode where the assertion is false, when
one exists** — a counterexample that satisfies its own `relation` is not a
counterexample, and publishing the eager sides beside a CONFIRMED decided
by `jit` did exactly that.
