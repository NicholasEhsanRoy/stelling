# The private-track usefulness criterion — registration

**Status:** REGISTRATION, 2026-07-19, written and committed **before it
is applied to anything**.

## Why this exists

The project runs two success criteria in parallel and they can diverge:

- **the public track** — the tracker corpus, E2a, a pre-registered
  falsifier with bands (0 mechanized kills the model; 1–3 weak; ≥4 across
  ≥2 libraries supported). Currently **Weak**.
- **the private track** — *would a check earn a place in Nick's CI?*
  Evidenced by MIME/MADDENING, deliberately held out from counting.

The private track has **authority to rank builds** (F1 is the single most
actionable fact in the corpus, and "close F1→F2" has been argued as a
build justification) and **no falsifier**. That is the unfalsifiable
hatch this project has closed twice by policy — once for the
"valuable for new projects" argument, once for exposure-substitutes-for-
verification. Authority without accountability is exactly the shape those
closures were about. This registration supplies the missing criterion.

**This registration does not create a count and cannot move E2a.** The
private track stays held out; what it gains is a falsifier.

## The criterion — four clauses, all required

A check **earns a place in CI** iff:

- **(i) stated on the code's own form.** The obligation is posed over the
  code as written, not over a hand-derived reformulation. A lemma that a
  human had to derive is a *design-time* result; re-deriving it after the
  code changes is human work the check did not do.
- **(ii) runs in CI time.** Seconds-to-minutes on the project's own CI,
  without a hand-tuned harness per run.
- **(iii) would have caught the scar.** Applied to the actual failure the
  author paid for: would this check, running in CI at the time, have
  fired on the failing input? A check whose *precondition is violated* by
  the failing case does not fire on it — it is silent exactly when it
  matters.
- **(iv) the recurring value is real.** The check has something to do per
  run, or per change, that a one-liner does not already do. **Written to
  test the cheaper-re-derivation value specifically:** where a result is
  regression-on-change rather than per-run, its value is that a *machine*
  re-derives it when the underlying code moves instead of a *human*. That
  value is **temporal** — it depends on how often the code actually moves
  and how expensive the human re-derivation is — and a temporal clause
  **cannot be cleared by inspection**.

**Failing any clause does not mean worthless.** It means *not a CI
check* — the result may still be a design-time fact, a publication, or a
regression tripwire. The criterion classifies; it does not condemn.

## What gets applied, and the pre-committed expectation

Bounded corpus: **F1 and F2** (`design/mime-fvm-job.md`) — they exist,
they are the private track's entire evidence, and F2 is the one a build
has been argued from.

Pre-committed expectation, stated before applying: **F1 scores
regression-on-change, not per-run**, because the scar mesh's
`cos ≈ 0.11` *violates* F1's own alignment-floor precondition — so a CI
job asserting F1 is silent on the failing mesh, and what would have
fired is the `cos ≥ 0.71` check, a numpy one-liner over the mesh. F1's
contribution is the one-time design fact that 0.71 suffices for bound 8.
**F2 inherits the same profile** (same precondition, same scar), so its
only residual value over F1 is clause (iv).

## Clause (iv)'s evidence, and what it may not be used for

The temporal question is measurable as a **proxy**: how often has the
floor's own code actually changed? Measure it from MIME's git history at
the pinned ref — the file and lines the condition lives on
(`operators.py`, the `Sf_dot_d` / `ortho_coeff` computation) — and report
the raw counts and dates.

**The proxy is evidence toward the clause; it does not clear it.**
"Often enough to be worth machine re-derivation" is a judgement about
Nick's own maintenance economics, and a build-justifying judgement is a
decision, not a reading.

## Bands (fixed before applying; continue is band-membership)

| result | band | continue? |
|---|---|---|
| criterion written; **F1 = regression-on-change**; **F2 = regression-on-change with its residual value conditional on the temporal clause (iv)** | **clean** — the predicted outcome, and the one that correctly leaves the assume-emission build **gated rather than justified** | **continue to Task D** |
| **F2 clears (iv)** — the proxy shows the floor code changes frequently enough that machine re-derivation is plainly worth it | build-justification is a decision | **STOP, surface** — record the proxy as evidence-toward, never as clearing |
| **F2 clearly fails even the conditional value** — the floor code has never moved since the fix, so machine re-run buys nothing | this kills the assume-emission justification and changes Task D's status | **STOP, surface** |
| anything else — the criterion proves unapplicable, a clause is ambiguous for F1/F2, or applying it needs a judgement call to proceed | out of band | **STOP, surface** |

**No band is amended after applying.** Note the deliberate asymmetry:
the *clean, continue-able* result is the one that leaves the build gated.
A result that **justifies** a build is a stop, because justification is
Nick's to make.

## Scope

No count. No corpus contact. MIME stays held out. The criterion binds
future private-track claims: **any statement of the form "this would earn
a place in CI" must score all four clauses in writing**, the way any E2a
count must carry its three breakdowns.
