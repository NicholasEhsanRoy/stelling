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

## Ground rules

- SPDX headers are inserted automatically by the pre-commit hook; don't
  fight it.
- Only `src/stelling/_jax_compat.py` may import jax. Everything else
  consumes the jax-free `stelling.ir`. Private jax modules are banned
  everywhere. Both rules are enforced by a pre-commit hook and by tests.
- Any change that can flip a verdict on any query is a soundness event and
  needs an entry in [SOUNDNESS.md](SOUNDNESS.md), whatever the semver bump
  says.
