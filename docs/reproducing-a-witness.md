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

<!-- doc-example: illustrative -->
```python
# myprogram.py — your code. No stelling import here.
def heat_step_against_bound(T):
    from mypackage.nodes import HeatNode

    node = HeatNode("h", timestep=0.1, n_cells=4, thermal_diffusivity=1.0)
    out = node.update({"temperature": T}, {}, 0.1)["temperature"]
    return out, 100.0          # (lhs, rhs) — the two sides of `out <= 100.0`
```

<!-- doc-example: illustrative -->
```python
# check_it.py — the harness module.
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

Running the emitted file prints the witness, both sides, and the result:

```
== executing YOUR function
  myprogram.heat_step_against_bound

  [eager] lhs = array([ 63.125, 101.   , 101.   ,  63.125], dtype=float32)
  [eager] rhs = array(100.)
  [eager] asserted: lhs <= rhs   ->  False
  [eager] FALSE at flat element(s): [1, 2]
    [1]  101.0 <= 100.0  is False   (margin +1.0)
    [2]  101.0 <= 100.0  is False   (margin +1.0)

  [jit] asserted: lhs <= rhs   ->  False

== CONFIRMED
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

Measured across this tree's contracts: **8 of 14 targets need fixture
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
set; `200 × 0.001 × 9.81 = 1.962` exactly, and at 100 000 steps the same
node spans `[-9805.9, 0]`. A measured span is "reachable under the
trajectory I ran", never "reachable in all operation".

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

### The schema is PROVISIONAL for 0.1.0

**It is unstable, and you should not build on it without pinning the
stelling version.** Concretely: fields may be **added, removed or
renamed** in 0.1.1, **without a deprecation cycle**, and a consumer
written against 0.1.0 may simply stop working.

It is planned to **freeze in 0.1.1**, once the external soak has parsed
real emissions and the fields have been exercised by a consumer that did
not write them.

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
