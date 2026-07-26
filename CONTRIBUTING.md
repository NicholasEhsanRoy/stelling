# Contributing

## Setup

```sh
pip install -e ".[solvers,jax]" --group dev   # pip ≥ 25.1; uv works too
pre-commit install
pytest
```

## Sign your commits

Every commit must carry a `Signed-off-by` line (`git commit -s`), certifying
the [Developer Certificate of Origin](DCO). CI enforces this on pull
requests.

This is provenance, not bureaucracy. stelling's ambition is to be usable in
qualification-grade settings, and the chain that supports that is: SPDX
headers say what license every line is under, DCO sign-offs say who asserted
the right to contribute each change, and PEP 740 attestations bind each
released wheel to the exact commit and workflow that built it. The sign-off
is the one link only contributors can provide, and it cannot be retrofitted
later.

**There is no CLA, and none is planned.** You keep your copyright; the DCO
is an assertion of provenance, not an assignment of rights.

## Invariants get tests that they can't drift

**Invariants that must not drift get a test that they can't, not a
convention that they shouldn't.** Worked examples already in-tree: jax may
be imported only in `_jax_compat.py` (enforced by a pre-commit grep *and* a
test, not a comment), and `TESTED_JAX_SERIES` must be a hardcoded literal
independent of packaging metadata (an AST assertion), with a companion test
that fails the moment CI's jax outruns it — so bumping the tested series is
a conscious act, never a drift. And the README's capability claims are now
under the same discipline: `tests/test_readme_claims.py` maps each
capability token (SMT/solver, derive, `cond`/`scan`/`while`, discrete
step) to a witness in the code, and fails if the README claims one without
it — with roadmap and disclaimer prose exempted only by an explicit
`<!-- capability-exempt -->` fence. It exists because the README claimed
SMT proving for as long as it stood and a convention ("someone will
notice") didn't catch it; the artifact about the tool now gets the
instrument the artifact about the world always had.

**Smaller instance, not worth a test — a convention, stated so it's at
least conscious:** module docstrings that make scope claims (e.g.
`propagate.py`: *no widening, no fixpoints, no cond/scan descent, no
solver*) are the same failure mode with a smaller blast radius. When you
change what a module does, its scope-claim docstring is part of the
change; a reviewer should treat a stale scope claim as a defect. The
README earned a test because it is public and mechanically checkable;
these earn a line here.

Pending instances, recorded now so they land with the features they bind:

- **Never invoke a solver on defaults** — a test asserting every invocation
  path passes a complete explicit option set. Lands with the first solver
  call.
- **Never emit a verdict without a complete stamp** — a test asserting every
  field of the SOUNDNESS.md stamp contract is populated, failing on a
  missing field rather than defaulting it. Lands with the first verdict.
  The contract grows over time (it just gained a precision field), and a
  stamp that silently omits a field is worse than one that doesn't exist.
- **Contrib-backed verdicts stamp their provenance** — a test that any
  verdict whose chain used a contrib/plugin transfer names the
  contributing registry and its version, plus the `TESTED_*_SERIES`
  literal-and-fence pair for each bundled contrib registry. Lands with
  the contrib registry (`design/open-primitive-set.md`).

## Measuring a change runs on BOTH worktrees

Any measurement of what a diff *does* — a verdict that flipped, a bracket that
narrowed, a timing, a coverage count — runs twice: once on a worktree at the
unchanged baseline, once on the tree with the diff. `git worktree add <path> HEAD`
then point `PYTHONPATH` at that tree's `src`.

**A control that fires identically on both is a defect in the instrument, not a
finding about the diff.** That rule is the whole point, and it has caught four
things that would otherwise have been reported as results: a transfer row that
passed its own spec while failing the property that justified it; a measurement
run against a tree being edited underneath it; a soundness alarm raised against a
diff that turned out to be a wrong predicate; and a threshold test whose
"UNSOUND" verdict fired on the baseline too.

Corollaries worth stating because each cost something:

- Do not edit a repo an agent or job is concurrently measuring. Use a worktree, or
  land the edit before launching.
- A one-sided run cannot distinguish "the diff did this" from "this was always
  true." Prefer no measurement to a one-sided one.
- When a control fires on the baseline, fix the control and re-run both. Do not
  reason about which side is "really" affected.

## Guard coverage is proven by mutation, not by construction

A test that reaches a guard and a test whose scenario stops short of it look
identical in CI — both green. So when a change makes a scenario decidable earlier
in the pipeline, do not assume the guard is still exercised: **mutate the guard
(invert the condition or neuter the raise) and confirm the test fails.** Restore
it afterwards. If no scenario can be built that the mutation breaks, the guard is
UNCOVERED — record it as such next to the coverage numbers. Do not delete the
test, and do not mark it `xfail`: the guard is not expected to fail, the scenario
stopped reaching it, and `XPASS` is the status nobody reads.

**First, prove the mutation is live.** A mutation that does not actually change
the running code produces a passing test that reads exactly like "the guard is not
reached" — a false negative that concludes the opposite of the truth. So: a
mutation must be shown to change behaviour *somewhere*. Run the whole suite under
it; if **nothing anywhere fails**, the mutation is suspect and must be re-sited
before "not reached" may be concluded.

Both failure modes here were hit in one sitting while writing this section:

- a `sed` targeted by line number landed on a **comment line** two lines above the
  condition, so the guard was untouched and the test passed;
- a pattern-based replacement asserted a single occurrence and found **two call
  sites**, aborting before it changed anything — again leaving a passing test.

In both cases the passing test would have supported "guard UNCOVERED." Neither was
true.

## A decline rule must trace to a measured discrepancy with a magnitude

A parameter-space gauge earns a decline rule by showing that the operation's real
semantics differ from what the row models. That evidence has to be **a number**:
a measured discrepancy, with a magnitude you can read and compare. A `None`
returned by the model, an absent parameter, a raised exception, an `inf` — none
of these is a measurement. They are the gauge declining to look.

**The failure mode is self-concealing, which is why it needs a rule rather than
care.** A spurious decline is *sound* — the row refuses something it could have
handled, so no verdict is ever wrong and nothing downstream catches it. The lost
capability reads as conservatism. A spurious *admission* gets found by the next
audit; a spurious decline can sit forever, and the gauge that motivated it will
keep reporting a clean run.

Both instances were caught before the row existed, and only because the gauge ran
first:

- comparing the model at `rtol=0, atol=0` made XLA's and numpy's differing
  **summation order** register as a semantic difference — `2.2e-16` and
  `8.9e-16`. The acceptance criterion was "zero survivors", so meeting it would
  have **required declining `(3,5) @ (5,)`, the shape of the one contract the row
  was being built for**, and the criterion would have certified that as success;
- the model returned `None` whenever batch dims were present, which the harness
  mapped to `inf`. That point read as "outside the modelled form" **having never
  been compared to anything**, and was about to become a decline rule. Evaluated
  properly, batch dims are benign.

Both would have failed this rule on inspection: neither traced to a magnitude.
When a threshold is needed, **derive it from what the tool judges**, not from
convenience — verdicts here are decided in R, reassociation is exact in R, and an
accumulation-dtype change measures `3.9e-08`, so a threshold at `1e-12` sits nine
orders from one class and four from the other. Print the magnitudes with the
results, so a value from the wrong class reappearing is visible rather than
inferred.

## Gate tests construct params as the TRANSCRIBER produces them

A gate reads `eqn.params_dict()`, and those params come from the transcriber, not
from jax's public API. The two forms differ. `precision="highest"` reaches a gate
as `ir.EnumParam(cls='Precision', member='HIGHEST')`, never as a
`jax.lax.Precision` member; a dtype arrives as a `str`, not a `numpy.dtype`.

So a gate test that builds its params from raw jax objects is **testing a path the
system never takes**. It can pass in full while the gate is broken for every real
trace, because the input it supplies is one the engine cannot produce.

Found by a blinded audit on the `dot_general` row. `_recognised_precision` did
`str(prec).split(".")[-1].upper()`, which matches a raw `Precision` member and
never matches the transcribed `EnumParam`. **Every non-`None` precision from a
real trace declined, while the oracle's docstring said it was admitted, and the
gate's unit tests were green throughout** — they passed raw objects. The decline
direction was safe, so nothing unsound shipped; what shipped was a row silently
narrower than its documentation, refusing a contraction form that is ordinary
practice in numerical code.

Build gate-test params by tracing a real program and reading the equation back
(`transcribe(jax.make_jaxpr(...)())`). Raw-object tests are fine as a supplement —
they are quicker to write and read — but at least one test per gated param must go
through transcription, and it needs an anti-vacuity companion showing the gate
still declines something, or a gate widened to accept the transcribed form can
quietly accept everything.

When sweeping for this, the question is not "which tests use raw objects" but
"which gated params have a transcribed form that differs". A param the gate never
reads cannot diverge: the scatter rows carry `mode` as their only `EnumParam` and
**deliberately do not constrain it** (all `GatherScatterMode`s agree on
definitely-in-range indices, which the transfer already requires), so their
raw-object tests are sound.

## Stop before soundness-critical work when mechanical slips accumulate

Mechanical slips — a `sed` that lands on the wrong line, an assertion with the
wrong expected value, a patch whose pattern doesn't match, reading a tool's
output as success when it reported failure — are not evenly distributed across a
long working session. They cluster, and the cluster is a signal about the state
of whoever is working, not about the difficulty of the task.

**The rule: when slips start accumulating, stop before the next
soundness-critical change, not after it.** Land what is already verified, leave
the rest unmerged on a branch, and record why. A half-built transfer row or
emission encoding is worse than none: it has to be reverted or branched anyway,
and any audit of it is an audit of something nobody trusts.

This is not a counsel of perfection about avoiding mistakes. Every slip listed
below was caught — most of them by the controls in this file, which is what they
are for. The rule is about what to do once you notice you are producing them at a
rate, on work whose failure mode is a false VERIFIED.

The instances that produced this rule, named the way the mutation norm names its
two, because a norm with the evidence stripped out is just an opinion:

Session of 2026-07-25, five, with the last edit carrying three at once:
- a `sed` targeted by line number landed on a comment two lines above the
  condition it meant to mutate;
- a pattern replacement asserted a single occurrence and found two, aborting
  before it changed anything;
- a repaired test assertion carried the wrong expected value (`hi == 0.5` where
  `hi = a.hi - b.lo = 1.5`);
- a bisection ceiling was set outside its own declared envelope;
- one edit introduced a stray colon, placed a block before the variable it read,
  and omitted the import it needed.

Session of 2026-07-26, four, the last of which prompted the stop:
- a probe read `.solver.invoked` off a field that is a tuple of stamps;
- a new test was `skipif`-guarded on the very condition it existed to check, so
  the mutation that should have failed it skipped it instead;
- a new test set `jax_enable_x64` at module import, leaking float64 into later
  modules — walking into a hazard this repo already documents by name in
  `tests/test_preconditions.py`'s fixture comment;
- **a `cd` failed, the edit that depended on it never ran, and the output was
  read as success.** That one is the worst of the set and the reason the rule is
  written this way: the others produce a wrong artifact, which the suite catches.
  This one produces a *false belief about the state of the tree*, and the next
  hour is spent building on it.

## Ground rules

- SPDX headers are inserted automatically by the pre-commit hook; don't
  fight it.
- Only `src/stelling/_jax_compat.py` may import jax. Everything else
  consumes the jax-free `stelling.ir`. Private jax modules are banned
  everywhere. Both rules are enforced by a pre-commit hook and by tests.
- Any change that can flip a verdict on any query is a soundness event and
  needs an entry in [SOUNDNESS.md](SOUNDNESS.md), whatever the semver bump
  says.
