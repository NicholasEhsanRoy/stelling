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

## Ground rules

- SPDX headers are inserted automatically by the pre-commit hook; don't
  fight it.
- Only `src/stelling/_jax_compat.py` may import jax. Everything else
  consumes the jax-free `stelling.ir`. Private jax modules are banned
  everywhere. Both rules are enforced by a pre-commit hook and by tests.
- Any change that can flip a verdict on any query is a soundness event and
  needs an entry in [SOUNDNESS.md](SOUNDNESS.md), whatever the semver bump
  says.
