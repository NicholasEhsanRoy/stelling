# Contributing

## Setup

```sh
pip install -e ".[solvers,jax]" --group dev   # pip ≥ 25.1; uv works too
pre-commit install
pytest
```

> **`[jax]` here assumes a fresh venv.** It is in this line because a
> contributor's clean environment needs jax to run the suite. **If you are
> installing into an environment that already has jax, drop it** — use
> `".[solvers]" --group dev` instead. The extra exists only to bootstrap an
> environment with no jax at all, and letting it into a resolver that is
> already managing your jax can desync CUDA plugin wheels.

`--group dev` also installs `hypothesis`, which drives the **property suite**
in `tests/property/` — generated harnesses, metamorphic properties, a
one-sided oracle, and the cvc5 record protocol as a fuzz target. Without it
those modules skip at collection and the suite still prints green, so if you
are changing anything the properties cover, check that they ran. Start at
[`tests/property/README.md`](tests/property/README.md): it says what a
metamorphic property is here, how to add one, how to give it a positive
control, and — with numbers from this project's own defect catalogue — what
the whole mechanism cannot reach.

> **With `hypothesis` installed, the whole-suite completeness pin is
> WITHDRAWN, and you should know that before you read a green line.** One
> property is `xfail`-marked against an open defect (the integer-literal wrap),
> and `tests/test_skip_inventory.py::test_no_session_skip_is_undisclosed`
> withdraws its claim — by skipping, and saying so — on any session that
> reported an xfail. The **shape** to expect on the whole tree with
> hypothesis installed is `exit 0`, `1 xfailed`, and that pin among the
> skips — re-driven at `3482822` on jax 0.11.0 with hypothesis 6.165.10,
> CPython 3.12.3, `JAX_ENABLE_X64` unset.
>
> *No absolute `passed` count is written here, deliberately.* It has been
> restated twice — `2470`, then `3910` — and gone stale both times, and it
> is read only by humans. `git rev-list --count 3482822..HEAD` is 52
> commits, `git diff --stat 3482822..HEAD -- tests/` is 76 files changed
> and 20,334 insertions, and a plain `pytest` on this tree passes well over
> four thousand — in a configuration *smaller* than the one the figure
> names, since hypothesis is absent and six `tests/property/` modules gate
> at collection. The three parts that carry information — the exit status,
> the xfail, and which pin is among the skips — do not move with the
> suite's size. The dated figure is kept, with its sha, in
> `.github/workflows/ci.yml`, where it is a record of one run rather than a
> number a contributor checks a green line against.
>
> Nothing is hidden from you: the
> pin's *other* half, which checks every skip the session did see, still runs.
> But "this suite's skip set is complete" is not being asserted in your local
> run. It **is** asserted in the CI jobs that run the whole tree, none of
> which installs `hypothesis` — the one job that does, `property`, runs
> `pytest -q -ra tests/property` only, so `tests/test_skip_inventory.py`
> never runs there. (Written that way because "CI does not install
> `hypothesis`" is literally false of the workflow as a whole: `ci.yml`'s
> `property` job installs it, read out of `pyproject.toml`'s dev group, so a
> reader who checks finds the sentence contradicted before finding the
> reconciliation.) So the pin is off for exactly the sessions that can run
> the property suite. It comes
> back the day the wrap remedy lands and the marker in
> `tests/property/test_oracle.py` is deleted; narrowing the session with `-k` or
> `--deselect` does not bring it back, by the same rule.

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


## What will be asked of a change

Two things, and neither is negotiable for the surfaces they cover:

1. **A test that fails without your change.** Not one that passes with it —
   one that goes red when the change is reverted. Show that you ran it both
   ways.
2. **A blinded review by someone who did not write it**, for anything on a
   published surface or adjacent to soundness. Not a second opinion after the
   fact — a gate before it lands.

The reasoning behind both, and the failures that produced them, are in
**[docs/norms.md](docs/norms.md)**. They are listed here by title only, so this
file and that one cannot drift apart:

- [Invariants get tests that they can't drift](docs/norms.md#invariants-get-tests-that-they-cant-drift)
- [Measuring a change runs on BOTH worktrees](docs/norms.md#measuring-a-change-runs-on-both-worktrees)
- [To claim a capability would unblock work, STUB IT AND COUNT](docs/norms.md#to-claim-a-capability-would-unblock-work-stub-it-and-count)
- [Don't hand-roll a traversal when a canonical accessor exists](docs/norms.md#dont-hand-roll-a-traversal-when-a-canonical-accessor-exists)
- [A measurement whose result is an ABSENCE needs a positive control](docs/norms.md#a-measurement-whose-result-is-an-absence-needs-a-positive-control)
- [Evidence of non-occurrence licenses "unreached", never "unreachable"](docs/norms.md#evidence-of-non-occurrence-licenses-unreached-never-unreachable)
- [Conditional coverage reports as full coverage](docs/norms.md#conditional-coverage-reports-as-full-coverage)
- [Before measuring a constant, read its definition site — and before deciding a question, read its ADJUDICATION site](docs/norms.md#before-measuring-a-constant-read-its-definition-site--and-before-deciding-a-question-read-its-adjudication-site)
- [State which query a measurement actually ran](docs/norms.md#state-which-query-a-measurement-actually-ran)
- [An instrument must declare its SCOPE, and an acceptance criterion must check that the scope covers the claim](docs/norms.md#an-instrument-must-declare-its-scope-and-an-acceptance-criterion-must-check-that-the-scope-covers-the-claim)
- [An over-permissive stub's ZERO is conclusive; its NONZERO is not](docs/norms.md#an-over-permissive-stubs-zero-is-conclusive-its-nonzero-is-not)
- [Build the fixture OUTSIDE the traced region](docs/norms.md#build-the-fixture-outside-the-traced-region)
- [Guard coverage is proven by mutation, not by construction](docs/norms.md#guard-coverage-is-proven-by-mutation-not-by-construction)
- [A decline rule must trace to a measured discrepancy with a magnitude](docs/norms.md#a-decline-rule-must-trace-to-a-measured-discrepancy-with-a-magnitude)
- [Gate tests construct params as the TRANSCRIBER produces them](docs/norms.md#gate-tests-construct-params-as-the-transcriber-produces-them)
- [Claim divergence: the code is narrower or wider than what it says](docs/norms.md#claim-divergence-the-code-is-narrower-or-wider-than-what-it-says)
- [Read key PRESENCE, not `.get()` — present-with-value-`None` is not absent](docs/norms.md#read-key-presence-not-get--present-with-value-none-is-not-absent)
- [A probe reading a final verdict must assert something non-trivial](docs/norms.md#a-probe-reading-a-final-verdict-must-assert-something-non-trivial)
- [Verify the artifact, not the exit code](docs/norms.md#verify-the-artifact-not-the-exit-code)
- [Extracting a shared oracle leaves ONE implementation](docs/norms.md#extracting-a-shared-oracle-leaves-one-implementation)
- [A battery that stops measuring reports a perfect score](docs/norms.md#a-battery-that-stops-measuring-reports-a-perfect-score)
- [Stop before soundness-critical work when mechanical slips accumulate](docs/norms.md#stop-before-soundness-critical-work-when-mechanical-slips-accumulate)
- [A gauge's oracle is the TARGET, not a reference implementation](docs/norms.md#a-gauges-oracle-is-the-target-not-a-reference-implementation)
- [A figure in a norm states the UNIT it counts](docs/norms.md#a-figure-in-a-norm-states-the-unit-it-counts)
- [An inequality used as an ARGUMENT is evaluated, not read](docs/norms.md#an-inequality-used-as-an-argument-is-evaluated-not-read)
- [A blinded audit is a GATE, not a step](docs/norms.md#a-blinded-audit-is-a-gate-not-a-step)

If you are changing one line in a docstring, read the first two. If you are
changing what a verdict claims, read all of them.

## A merge conflict you should expect

`tests/test_doc_examples.py` pins a **single global count** of the code blocks
and output fences across `README.md` and `docs/`. Any two branches that touch
documentation will conflict there, and that is working as intended — it is the
tripwire saying two people changed the docs.

Resolving it is arithmetic: take the base counts, add what each side added, and
**let the test confirm the sum** rather than trusting it. The same numbers appear
in three places — the dict, the table in that module's docstring, and the prose
sentence after the table — and all three are checked.

## Ground rules

- SPDX headers are inserted automatically by the pre-commit hook; don't
  fight it.
- Only `src/stelling/_jax_compat.py` may import jax. Everything else
  consumes the jax-free `stelling.ir`. Private jax modules (`jax._src`) are
  banned everywhere except one file — `src/stelling/_tripwire/_adapter_jax.py`,
  which reaches a registry that no public or `jax.extend` module exports on
  either tested series, measured rather than assumed. That file is still
  subject to the first rule. Both rules are enforced by a pre-commit hook and
  by tests, which are held to the same exempt set; the reasoning and the
  rejected alternatives are in
  [design/private-jax-boundary.md](design/private-jax-boundary.md).
- Any change that can flip a verdict on any query is a soundness event and
  needs an entry in [SOUNDNESS.md](SOUNDNESS.md), whatever the semver bump
  says.
