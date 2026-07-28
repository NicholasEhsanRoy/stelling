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

## To claim a capability would unblock work, STUB IT AND COUNT

Presence, attribution, and first-decline are **proxies**. Only the counterfactual
measures. Four proxies have been used to rank work on this project; each was
introduced to fix the previous one's flaw, and each reproduced it in a new form.

- **first-decline peeling** — overcounted by measuring *which decline came first*,
  when a recorded blocking cause is only ever the first of a stack.
- **source attribution by outermost frame** — overcounted `abs` at 6-of-10 by
  measuring the harness's own obligation phrasing. The frame stack is 8 deep and
  the creator sits second from the end, not last.
- **attribution by the wrong aggregation unit** — reported every primitive as
  node-side, because one node occurrence anywhere marked the whole primitive.
  `abs` is node-side in two contracts and harness-side in four; only
  per-(primitive, contract) can say so.
- **presence on the obligation slice** — overcounted `sqrt` by three. A primitive
  on the slice may sit behind another decline, or behind no decline at all: one
  contract carrying `sqrt` was *already VERIFIED*, interval-discharged, never
  reaching emission.

The pattern is not that these were careless. Each is a reasonable measurement of
something — just not of *"would building this produce a verdict"*. **That question
has exactly one honest form: make the capability exist and count what decides.**

```
for each candidate capability:
    stub it in-process (emission set, budget, guard — whatever gates it)
    run the contracts
    count the ones that reach a VERDICT, not the ones whose cause text changed
```

Two things that make the rule usable rather than aspirational:

**It is cheap.** In-process monkeypatching of a registry or a constant costs
minutes and touches nothing on disk. There was never a cost reason to reach for a
proxy — the proxies were faster to *think of*, not faster to run.

**A stubbed verdict is not a verdict.** A stubbed gate is unsound by construction:
it admits what the engine refuses. Label every such run as a cause enumeration,
never record its output as a verdict anywhere, and destroy the stub when done.
In-process stubbing helps here too — there is no tree to accidentally commit.

And distinguish **kinds** of win when counting. A decline becoming a verdict is an
unblock. A withheld VERIFIED becoming a rendered one is worth having but is a
different thing, and summing them overstates what the capability bought.

## Don't hand-roll a traversal when a canonical accessor exists

`stelling.coverage.sub_jaxprs` is the canonical way to reach an equation's
sub-jaxprs. Three sites walked the IR without it and each got it wrong in a
different way:

- `vacuity.widen`'s guard scanned **one level** — `sub_jaxprs(eqn)` then
  `sub.eqns`, no recursion — so a declaration two calls down escaped both the
  guard and the widening;
- a parameter gauge selected `eqns[0]`, **positionally**, which breaks the moment
  jax inserts an equation (it does, for broadcasting);
- `verdict._barred_primitives` descended via `getattr(v, "jaxpr", None)`, which
  finds a param that IS a ClosedJaxpr but not one holding a **collection** of
  them — and `cond` stores its branches as a tuple, so a barred primitive inside
  a `cond` branch was not found at all.

The third is the sharpest: that walk *did* recurse. It simply reached through a
different accessor than the canonical one, and silently stopped matching it.

So the rule is not "descend properly" — it is **use the accessor**. A second
implementation of a walk is a second thing to keep correct, and nothing tells you
when it drifts.

Where a second implementation genuinely is needed, **assert parity mechanically**
rather than leaving it to review: `tests/test_bar_walk_parity.py` checks the bar's
walk against `sub_jaxprs` across top level, `jit`, `cond`, `scan` and nested
combinations, in the same spirit as the `EMISSION == REPLAY` census. Give it an
anti-vacuity companion that re-implements the *old* accessor and requires it to
disagree — otherwise a parity test that passes under the broken walk proves
nothing.

## A measurement whose result is an ABSENCE needs a positive control

**The general rule the next four sections are special cases of.** If a measurement
can only report "nothing here", it must be paired with a control showing the
instrument detects the thing when it *is* here. Otherwise "nothing here" and "this
instrument cannot see" produce identical output, and the reassuring one gets
believed.

Every instance below was a real reading that survived review before someone
noticed the instrument was blind:

- **a mutation that changed nothing.** A guard reported UNCOVERED because the
  `sed` landed on a comment. The test passed for the wrong reason.
- **a battery where every point was rejected.** An empty consts map made all of
  them decline; the run read as "everything caught."
- **a probe whose assertion was trivially true.** `x >= -1e30` is discharged by
  interval arithmetic, so the slice — the stage under test — never ran, and three
  poisoned structures were recorded "clean".
- **an identity comparison on freshly-built objects.** Post-transcription
  `ClosedJaxpr`s are constructed per param and are therefore always distinct, so
  "0 duplicates share an object" was arithmetically incapable of being anything
  else. It was recorded as evidence *against* the hypothesis it could not test.
- **a widen that never reached the declaration.** A nested envelope escaped the
  guard, and the verdict then carried "envelope not load-bearing" about an
  envelope that was load-bearing.

The pattern is the same in all five: **the negative result was produced by the
instrument's reach, not by the world.**

So state, for any absence claim: *what would this have looked like if the thing
were present?* If the answer is "the same", the measurement is not evidence. The
specific forms below — a mutation must change behaviour somewhere, a battery must
reject at least one point, a probe must assert something the cheap layers cannot
decide — are kept because they are more actionable than this rule; use them first
and fall back to this when the situation fits none of them.

## Before measuring a constant, read its definition site

`ELEMENT_BUDGET = 512` carries a measured table in a comment **directly above the
assignment** — solver timings at 256/512/1024/2048 terms, naming the binding
fragment. Two sessions were spent measuring that constant and concluding it was
three orders of magnitude too tight. The measurements were correct. They were of
the *other* fragment, and the table said so.

Nothing was missing or undocumented. It was unread, by measurement aimed at
exactly the question it answered — and the result was a proposal to raise a
CI-critical bound by 140×, which would have hung the first nonlinear obligation.

So: **before measuring a constant, threshold, budget, tolerance or cap, read the
source at its definition site.** If a derivation is there, the measurement's job is
to *check that derivation on its own terms* — same regime, same fragment — not to
replace it with a fresh number from a different regime.

The campaign already has "read, don't recall" for census membership. This is its
mirror: a value someone derived once is evidence, and re-deriving it without
reading it discards that evidence while feeling like rigour.

Corollary for anyone writing such a constant: **name the regime in the comment.**
512's table named `QF_NRA` explicitly, which is the only reason the error was
findable at all. A derivation silent on the regime it was taken under is a latent
version of the same trap.

## State which query a measurement actually ran

Norm D catches an instrument that cannot detect the thing. This catches one that
**answers a different, easier question than the one reported** — and it passes every
check aimed at it, because the check is aimed at the question you meant to ask.

Three instances, all found by follow-up rather than by review:

- a solver-timing sweep whose obligations were **discharged by interval
  arithmetic**. The solvers did run — on the *vacuity widen re-check*, a different
  and easier query — so the wall-clock was interval cost plus a re-check, reported
  as the cost of solving the obligation. **It survived a full night's reporting.**
- a cause-stack layer that was **the stub's artifact**: a primitive added to the
  emission set without its plan declines on the generic fallback, describing the
  stub rather than the engine.
- an object-identity comparison run on **post-transcription** values, which are
  constructed fresh per parameter and so are always distinct — reported as
  evidence against the hypothesis it could not test.

So: **say what ran.** For a solver measurement that means reporting how many
invocations were on the *obligation itself*, separately from re-checks —
`solver_timeout_ms` being set is not evidence the solver decided anything, and a
verdict of VERIFIED is not evidence either. Intervals discharge most easy
obligations before escalation is reached.

The discriminator is cheap: count the invocations whose reason does not mention a
re-check, and print it beside the timing. If that count is zero, the measurement is
about interval propagation and must not be reported as being about the solver.

**And name the FRAGMENT, not just the query.** A fourth instance had
`obl-solves = 2` on every row — the obligation genuinely reached the solver — and
was still measuring the wrong thing, because every one was `QF_LRA` while the
constant under test is set by `QF_NRA`. Those two differ by three orders of
magnitude: 8192 linear terms cost under half a second, and 384 *nonlinear* terms
cost 237. A measurement that does not say which logic it ran in is not a
measurement of solver cost. `set-logic` is in the stamp's options.

## An instrument must declare its SCOPE, and an acceptance criterion must check that the scope covers the claim

Norm E's sibling. **E says state which query you ran; this says state what your
instrument reaches.** The failure it catches is not a wrong number — it is a
correct number about a smaller thing than the claim it was used to license.

Five instances, each caught by a check aimed at exactly it:

- **the budget family measured the vacuity re-check.** "524,288 terms in 193 s"
  was real timing of real solver work. Every obligation had been discharged by
  interval propagation, so the only thing the solver ran was the widen re-check.
  The instrument reached the solver; it did not reach *the obligation*. Caught by
  counting invocations whose reason is not a re-check.
- **the seventh decline cause was a stub artifact.** It existed only because a
  stub stood where a capability would go, so it measured the scaffolding rather
  than the program. Caught by stubbing and counting deliberately, which makes the
  scaffold's contribution a number instead of a background.
- **the shared-jaxpr comparison measured post-transcription objects.** Those are
  constructed per param and therefore always distinct, so the comparison's reach
  excluded the only state in which the hypothesis could be true. Caught by asking
  what a positive result would have looked like.
- **the frontier counted presence.** Presence on the obligation slice, attribution
  by outermost frame, and first-decline peeling are all cheaper than the
  counterfactual and all of them overcounted. The instrument reached *appears in*,
  the claim was *blocks*. Caught by removing the row and re-running.
- **the param gauge drives one face.** It exercises the obligation-face plans and
  never `_t_scatter`, `_t_scatter_add`, or `interval_env` — so "zero survivors"
  certifies emission while the fix under test changed emission *and* the transfer.
  Caught by reading the gauge's imports against the fix's diff.

The shape is identical in all five: **the instrument was sound within its reach,
and its reach was narrower than the sentence it was quoted to support.** No
instance was a bug in the instrument. Four of them passed review.

So, two obligations. **On the instrument:** state its scope where it is defined —
what it drives, and what it does not. A gauge that names the entry points it calls
cannot be silently quoted past them. **On the acceptance criterion:** before
accepting "the gauge is green" as evidence for a change, check that the gauge's
scope covers the change's *surface*. If a fix touches two faces and the criterion
reaches one, the criterion is not an acceptance criterion for that fix — and
saying so is the finding, not a reason to bolt on a bespoke test beside it.

The corollary that makes this cheap: **when an instrument is extended, its scope
line is the diff to read first.** And when one instrument is found to have this
hole, check its siblings before assuming it was unique — the assumption that
produced it rarely rode alone.

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

## Claim divergence: the code is narrower or wider than what it says

A distinct failure class from a wrong answer. The code does one thing, its stated
claim says another, and **every test passes** — because the tests check the
behaviour, and nothing checks the claim. It ships as a silent capability loss, or
as a promise nobody keeps.

Four instances so far, and the fourth changes the class:

- `test_float_div_is_completely_unchanged` — a name asserting more than the test
  checked;
- `VERIFIED_BAR_REASON` — prose describing a scope the mechanism did not have;
- `_recognised_precision` — a docstring claiming DEFAULT/HIGH/HIGHEST were
  admitted while every transcribed form declined;
- **`_Slicer._define`'s comment: "variable ids must be unique across the flattened
  scopes (the transcriber guarantees it)". The transcriber does not guarantee it.**

The fourth is the most consequential and widens the definition. The first three are
documentation drifting from behaviour. This one is **a correctness assumption stated
as an established fact**, and stating it that way is *how the defect survived*: the
comment reads as a citation, so nobody re-derived it, and the check built on it
silently declined every multi-call query for months.

So: **a comment asserting an invariant is a claim, and an invariant nothing
enforces is a claim divergence waiting to happen.** If a comment says some other
component guarantees a property you depend on, either point at the enforcing code
or write the enforcement. "X guarantees it" with no referent is the smell.

## Read key PRESENCE, not `.get()` — present-with-value-`None` is not absent

Named for the reachable half deliberately. This was called "absent params" for a
while, and that name points at the half that **cannot happen**: a param key missing
entirely is structurally unreachable from jax for every row measured — `bind`
requires them all, so an equation always carries the full key set.

Every defect in this class came from the *other* half. `params.get(k)` returns
`None` both when the key is absent and when it is present with value `None`, and
jax uses the difference:

- `scatter-add`'s `update_jaxpr` present and `None` is jax's **replace** combiner
  (`lambda x, y: y`), not accumulation. Read with `.get()` it was modelled as an
  add, and produced a false VERIFIED with both solvers answering unsat.
- `dot_general`'s `preferred_element_type` present and `None` means the
  accumulation **follows the operands**, so integer operands wrap. Skipping the
  dtype checks on `None` would have admitted it as exact real arithmetic.

Absent keys matter only for hand-built or deserialized IR. Write `if k in params:`
and handle the two facts separately, with the `None` branch stating what jax
substitutes.

## A probe reading a final verdict must assert something non-trivial

When a probe measures an **intermediate mechanism** through a **final verdict**, a
trivially-true assertion never reaches the mechanism, and the probe reports it
absent.

Measured, on the aliasing investigation: three probe rows asserted `x >= -1e30`,
which interval arithmetic discharges outright. The slice — where the aliasing poison
is consulted — never ran. All three reported VERIFIED and were recorded "clean",
including cases that were in fact poisoned. Reading the mechanism's own state
directly flipped three of six rows. **The conclusion survived; the evidence for it
did not.**

So: assert something the cheap layers cannot decide, or read the mechanism's state
directly instead of inferring it from the verdict. This is the anti-vacuity
companion rule one level up — there, a gate test must not pass by closing the row;
here, a probe must not "measure" a stage its input never reaches.

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
