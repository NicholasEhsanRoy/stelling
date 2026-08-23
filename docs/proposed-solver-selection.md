<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Exposing solver selection on `check()` — **PARTLY BUILT** (`cbb1d60`)

**Status: PARTLY BUILT.** Three of the five rows of the change table below
landed in `cbb1d60` ("Phase-1 transfers: is_finite, int64->float64 point
rule, solver kwarg"); **two did not**, and the two that did not are the
residue this page is now most useful for. Pinned by
`tests/test_phase1_transfers.py` (`test_check_solver_kwarg_validation`,
`test_check_solver_kwarg_z3_records_only_z3`,
`test_check_solver_kwarg_none_is_default`).

**What shipped.** `check()` takes `solver=`, and it works:

| driven | result |
|---|---|
| `check(h, …, solver=None)` | VERIFIED, the full portfolio — 2 invocations |
| `check(h, …, solver="z3")` | VERIFIED, restricted — 1 invocation |
| `check(h, …, solver="cvc5")` | VERIFIED, restricted — 1 invocation |
| `check(h, …, solver="")` | `ValueError: solver must be None, 'z3', or 'cvc5', got ''` |
| `check(h, …, solver="nope")` | `ValueError: … got 'nope'` |

So rows 1, 2 and 4 of the table landed, and the eager validation this page
asks for is there.

**What did NOT ship, and a flipped header would have buried it.**

* **`contracts.check_contract` was not threaded.** Measured:
  `inspect.signature(check_contract)` is
  `(contract, *, vacuity_mode, solver_timeout_ms=None, refine=None,
  falsify=None)` — no `solver`. The contracts layer cannot restrict the
  portfolio at all.
* **The "refuse it when `solver_timeout_ms is None`" clause was not
  implemented.** This page argues that combination is *"a caller error that
  would otherwise ride silently"*. Measured, it rides silently:
  `check(h, vacuity_mode="inputs-only", solver="z3")` with no timeout does
  **not** raise — it returns a verdict decided by interval propagation alone,
  with the restriction it was given having restricted nothing.
* **The shape differs from the proposal.** This page describes
  `SolverConfig.only`, a *subset* of `{"z3","cvc5"}`; what shipped is a single
  string, so "both, in this order" is not expressible and passing a tuple
  raises. `SolverConfig(only=…)` itself is still internal — see the note
  under "The capability, and why it is unreachable", which remains accurate
  about `SolverConfig` and is now inaccurate only about `check()`.

**Verdict: the DECISION this page asked for was taken in part and not
recorded.** The header said `NOT BUILT` through the commit that built most of
it. That is the same claim divergence on a document that
`proposed-declaration-dtype-check.md` records and names, and it cost more here
than a wrong word: `check(solver=…)` shipped documented in **no** page under
`docs/`, because the page that would have documented it said it did not exist.
`docs/choosing-a-solver-backend.md` now names it.

Everything below the change table was measured against the tree **before**
`cbb1d60` and is kept as it was measured; the two rows that did not land are
still open, and the argument for them is still the argument.

*Source references on this page name a SYMBOL and a FILE and deliberately
carry no line number. They used to carry sixteen. Resolved against `53f9f84`
on 2026-08-09, five of them no longer pointed at what the sentence said:
`solvers.py:769`, cited as `_backends_for`, is `if error:` and the `def` is at
1057; `solvers.py:979`, cited as `_absences`, is a BLANK LINE and the `def` is
at 1267; `solvers.py:1050`, cited as the dispatch site, is blank;
`contracts.py:942`, cited as `check_contract`, is the middle of an f-string and
the `def` is at 952; `reproduce.py:258`, cited as `SIDECAR_KEYS`, is blank. A
line number in a page nothing regenerates is a claim nothing checks, and a
wrong one sends a reader to a line that reads plausibly. Symbols move too, but
they move LOUDLY — `grep` comes back empty — and the one page here that did
carry line numbers, `docs/supported-primitives.md`, got them from
`docs/gen_supported_primitives.py` and was byte-compared on every run by
`tests/test_supported_primitives_doc.py`. Since 0.2.0 D15 (2026-08-23) it
carries none: a generated coordinate is still true only of the checkout that
generated it, and the sdist a reader downloads carries the digits that were
right on the day. The generator still LOCATES every registry and every quoted
reason, and still refuses to generate when one has left its file or changed
its words; what it emits is the file and the SYMBOL, which is this page's own
rule and is now that page's too.*

## The capability, and why it is unreachable

`SolverConfig.only` (`src/stelling/solvers.py`) restricts escalation to a
subset of `{"z3", "cvc5"}`. It is fully built: it validates its argument
eagerly (an empty tuple and an unknown name both raise, `solvers.py`),
it drives backend discovery (`_backends_for`, `solvers.py`), and it has a
dedicated disclosure path that distinguishes a configured restriction from a
missing install (`_absences`, `solvers.py`, whose docstring records that
the predecessor said "not installed" for both and that this was false).

`check()` constructs `SolverConfig(timeout_ms=int(solver_timeout_ms))` at
`src/stelling/preconditions.py` and never passes `only`. That is the
**only** `SolverConfig` construction in `src/` — grep confirms one site. So
the capability is reachable from tests (56 `SolverConfig(...)` constructions
across `tests/`, 30 of them passing `only=`) and from nowhere else. It is the
same shape as `propagate(semantics=…)`: built, tested, and not threaded
through the public entry point.

## What would change, and how much

Small, and that is part of the problem — the plumbing cost is not what should
decide this.

| site | change |
|---|---|
| `preconditions.check` (`preconditions.py`) | new keyword-only parameter, defaulting to `None` |
| `preconditions._pipeline` (`preconditions.py`) | accept it and thread it to `_finish` |
| `preconditions._pipeline` eager validation (`preconditions.py`) | validate at entry, like `vacuity_mode` and `refine` — and refuse it when `solver_timeout_ms is None`, since a portfolio restriction with no escalation is a caller error that would otherwise ride silently |
| `preconditions._pipeline` (`preconditions.py`) | the one `SolverConfig(...)` construction gains `only=` |
| `contracts.check_contract` (`contracts.py`) | new parameter, forwarded to `_pipeline` — it already forwards `solver_timeout_ms` and `refine` unchanged |

Five edits, two public signatures, one shared pipeline. The vacuity widen
re-check needs nothing: it goes through the same `_finish`, so it inherits
the restriction automatically and stays at the same depth — the property
`test_vacuity_depth.py` exists to hold.

## The soundness-relevant consequence, stated precisely

**Restricting to one backend removes the cross-check on every discharge.**

The portfolio's design is that both backends answer and agreement decides; a
sat/unsat disagreement raises `SolverDisagreement` rather than being resolved.
With one backend there is nothing to agree with, and the two directions are
not symmetric:

- a `sat` becomes `REFUTED` only through `make_validated_witness`, whose
  validator re-derives the violation in exact `Fraction` arithmetic in pure
  Python, sharing no code with the solver. A lost second backend costs a
  cross-check that a different mechanism still performs.
- an `unsat` is a universal claim over the whole declared box. Nothing
  re-derives it — there is no point to replay. **The second backend is the
  only independent check that exists on a solver-discharged `VERIFIED`.**

The module docstring already states this (`solvers.py`) and the
degraded-portfolio note says it in the verdict, verbatim: *"a discharge is a
universal claim over the whole declared box, so nothing downstream re-derives
it the way exact-rational replay re-derives a witness."*

## Is a one-backend verdict distinguishable from a cross-checked one by the stamp alone?

**Measured, and the answer is: yes for the `only=` case, and no for the case
that already happens today.** Both halves matter, and they point in opposite
directions.

**Yes — under `only=`, the stamp discloses it, in a quotable field.**
`SolverStamp.reason` is written at the dispatch site (`solvers.py`) and
carries the backend's *role*. Measured on the same obligation:

```text
two backends: reason = 'QF_LRA portfolio primary on assert #0'
              reason = 'QF_LRA portfolio secondary on assert #0'
only=("z3",): reason = 'QF_LRA portfolio alone (degraded portfolio) on assert #0'
```

`Stamp.solver` also drops from a 2-tuple to a 1-tuple, so both the count and
the role are readable without touching anything else. A restricted run is
therefore self-identifying in the stamp, and `render()` additionally promotes
it to the third line of the verdict:

```text
  PORTFOLIO DEGRADED — assert #0 was decided by ONE solver backend (z3 (wheel)), not the two the portfolio is designed around; the notes say which backend was lost and why
```

**No — for the case a full portfolio already produces, the stamp does not
disclose it, and this is the finding.** When both backends are installed and
both are invoked but one times out, the stamp says two were **asked** — that
is its whole contract, appended before the transport runs so no result can be
narrated into it — while the decision rested on one. Measured on a 16-product
`QF_NRA` obligation with both wheels present:

```text
len(stamp.solver) : 2
  invoked=True name='cvc5' reason='QF_NRA portfolio primary on assert #0'
  invoked=True name='z3'   reason='QF_NRA portfolio secondary on assert #0'
solver_redundancy : ((0, ('cvc5 (wheel)',)),)
```

Two `invoked=True` stamps, both reading "primary"/"secondary", and a decision
that rests on one answer. The field that discloses it is
**`Verdict.solver_redundancy`** (`verdict.py`) — `(assert index, labels of
the backends that ANSWERED)` — plus the `PORTFOLIO DEGRADED` render line and
the notes. Its own comment says why it exists: *"The stamp's solver tuple
records who was ASKED — that is its contract — so a two-backend stamp is
compatible with a one-backend decision, and a job tallying VERIFIEDs could not
tell them apart."*

So exposing `only=` does **not** open a new disclosure gap. The gap is already
open, `solver_redundancy` already closes it, and `only=` lands on the *better*
side of it: an `only=`-restricted run is visible in the stamp itself, which a
timed-out backend is not.

## What could regress

1. **Silent portfolio-degradation-by-default.** The risk is not the parameter;
   it is a project setting `only=("z3",)` in a CI config for speed (z3 answered
   linear obligations 8 ms against cvc5's 71 ms — see
   [choosing a solver backend](choosing-a-solver-backend.md)) and thereby
   halving the redundancy behind every `VERIFIED` it produces, forever, with
   the disclosure present but nobody reading it. That is the real cost and it
   is a documentation-and-defaults problem, not a code problem.
2. **`only=` with no `solver_timeout_ms`.** Accepting it silently would let a
   caller believe they restricted a portfolio that never ran. Must raise.
3. **The reproducer sidecar carries no redundancy field.** `SIDECAR_KEYS`
   (`reproduce.py`) has `fragment` and a human `solver` line derived
   from `witness.produced_by`, and nothing that says how many backends
   answered. Exposing `only=` makes one-backend runs ordinary, so the sidecar
   would start describing them routinely without disclosing what it is
   describing. (This is milder than it sounds — a sidecar exists only for a
   `REFUTED`, the direction with the replay backstop — but it is exactly the
   asymmetry an audit would test.)
4. **`only=("cvc5",)` interacts with `STELLING_CVC5`.** `_backends_for` gives
   an explicit external binary precedence over the wheel, so `only=("cvc5",)`
   in an environment with that variable set silently selects a *different
   program* than the same call elsewhere. Reproducibility of a restricted run
   is therefore environment-dependent in a way the unrestricted one hides
   behind the second backend.
5. **Parameter proliferation on the front door.** `check()` already takes
   `vacuity_mode`, `solver_timeout_ms`, `refine`, `strict`. `only=` is the
   fifth dial, and the first one whose wrong setting weakens evidence rather
   than merely producing less of it.

## What tests would be needed

- `check(..., solver_timeout_ms=…, only=("z3",))` produces a one-entry
  `stamp.solver` whose `reason` contains `alone (degraded portfolio)`, and
  `solver_redundancy` of length 1 — the two disclosures asserted separately,
  because they come from different mechanisms.
- The same for `check_contract`, since it is a second front door.
- `only=` without `solver_timeout_ms` raises at entry, before tracing (the
  `vacuity_mode`/`refine` precedent at `preconditions.py`).
- Every `SolverConfig.__post_init__` refusal (`only=()`, unknown names)
  surfaces through `check()` unchanged and eagerly — a validation that only
  fires deep in escalation is a validation a caller meets late.
- The widen re-check runs under the *same* restriction as the original call
  (the depth property `test_vacuity_depth.py` guards for `refine`).
- The absence phrasing stays correct: excluded-by-config must never render as
  "not installed" through the public path either — `_absences` has a test at
  `tests/test_constant_fold_portfolio.py`; it needs the `check()` twin.
- A negative test that the default is byte-identical to today: `only=None`
  must produce the same stamp, notes, and content hash as the current call.

## What a blinded audit would attack

- **"Does the verdict distinguish 'I asked one' from 'I asked two and one
  died'?"** Answer above: only through `solver_redundancy`, not through the
  stamp, in the second case. An auditor would ask why the weaker case is the
  one the stamp hides.
- **"Can a caller obtain a `VERIFIED` with less redundancy than the default
  and no louder signal?"** Today the answer is no, because a caller cannot
  reach `only=` at all. Exposing it makes the answer "yes, with three
  disclosures" — and an auditor will test whether all three survive being
  *tallied* rather than read. `solver_redundancy` was built precisely because
  the earlier disclosures did not.
- **Composition with the other never-on dials.** `refine="affine"` decides
  obligations before escalation; `only=` restricts what escalation is left.
  An auditor would look for a combination in which the verdict's account of
  *which layer decided* becomes wrong.
- **The empty and the singleton.** `only=()` raising, `only=("z3","cvc5")`
  being exactly the default, and `only=("z3",)` on a machine with no z3 —
  which must say "z3 is not installed", not "excluded by config".

## `only=`, or something narrower

`only=` is the wrong shape for a public API, for one reason: **it is
expressed as a permission over backend names, and what a caller actually wants
is a policy over redundancy.** A user who writes `only=("z3",)` is answering
"which solver do I have?" — a question the environment already answers, via
`_backends_for` — or "which is faster on my workload?", which is a tuning
question whose honest cost is the cross-check.

Narrower alternatives, in increasing order of safety:

1. **`solver_only=("z3",)`** — `only=` renamed for the front door, no
   semantic change. Cheapest, and carries every risk above.
2. **`prefer=` (ordering, not exclusion)** — reorder the portfolio without
   dropping anyone. Gets the latency win on the primary while keeping both
   answers, so redundancy is untouched. Does **not** serve the "z3 times out
   on this shape, stop waiting for it" case, which is a real measured case.
3. **A per-backend budget** — e.g. a shorter timeout for the secondary. This
   is the honest form of "cvc5 is slow on my linear obligations": both
   backends are still asked, the secondary's answer is *best-effort*, and a
   miss lands in the existing degraded-portfolio machinery that already
   handles a timed-out backend. It converts an exclusion into a timeout, which
   is a case the stamp and `solver_redundancy` already model correctly.
4. **Nothing at all** — the environment already selects the portfolio by what
   is installed, and that path is fully disclosed. `only=` stays the internal
   test seam it is.

(3) is the best fit for the measured need, and (4) is the honest default until
someone brings a use case that (3) does not serve.

## Verdict

**NEEDS-A-DECISION.** The plumbing is safe: five edits, one construction site,
no new disclosure gap — the `only=` path is *better* disclosed than the
already-shipping timed-out-backend path. What needs deciding is not whether it
works but whether the project wants a public dial whose wrong setting silently
halves the independent evidence behind a `VERIFIED`, when the underlying need
("one backend is slow on my shape") is better served by a per-backend budget
that keeps both backends in the portfolio. Adding `only=` is a one-day change;
withdrawing a public parameter after CI configs depend on it is not.
