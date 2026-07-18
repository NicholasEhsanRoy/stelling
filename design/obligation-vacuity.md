# The ⊤-widening test — the third vacuity guard, registered before it runs

**Status:** REGISTRATION, 2026-07-18. Committed before the test executes
on any counted case.

Three vacuity guards were needed and two exist: **criterion (i)** guards
the box (does it contain `y0`? — mechanized, stamped), **fidelity** guards
the system (is it the incident's code? — censused, quoted). **Nothing
guards the obligation.** dfx#207 exposes the gap: the code is
`dt = jnp.maximum(dt, dtmin)` and the property is `dt ≥ dt_min` — which is
`max`'s *definition*, true for any input whatsoever. A proof of it uses no
system, no box, no region. The chain it completes: **defective property →
tautological obligation → VERIFIED → counts 1** — criterion (ii)'s
relation naming caught dfx#207's *gap* (property-recheck: defective
anchor) and not its *triviality*.

## The test — mechanical, the mutation control aimed at the inputs

Trace the harness as filed; then, **at the IR level**, rewrite every
`stelling_any` equation's bounds to `(−inf, +inf)` and re-propagate. If an
obligation still discharges, **the proof never used the declared bounds**
— it is structural, and the count says nothing about the incident's
region.

Scope notes, fixed now: judged on **obligations only** (`stelling_assert`);
nonvacuity rows are point declarations whose widening is meaningless and
they are excluded from the judgement. The IR rewrite touches top-level
`stelling_any` equations; the current harnesses declare all inputs at top
level (checked in the runner, loudly, so a nested declaration cannot
silently escape widening).

## The consequence — registered before anyone sees which cases are tautological

> **A tautological obligation cannot count 1.** A counted-1 case whose
> every obligation survives ⊤-widening is **voided, loudly** — the same
> enforcement shape as fidelity: if a count moves, the registration moved
> it. (A case with a *mix* of tautological and bounds-using obligations is
> reported with the split; it voids only if the discharge of the
> registered property rests entirely on tautological obligations.)

## Predictions, pre-committed before the run

- **dfx#207 → tautological** (`max(⊤, DT_MIN) ≥ DT_MIN` discharges by
  `max`'s definition). Its count of 1 voids.
- **npy#249 → NOT tautological**: under ⊤, `exp([−inf, inf]) = [0, inf]`
  and `> 0` straddles at the 0 endpoint — the declared `log_eps ≥ −20`
  floor is load-bearing. Its count stands.
- **dfx#417 → NOT tautological** (the face fluxes go ⊤ when `u, v` widen).
- **hit386 control → NOT tautological** (everything routes through the
  widened params).

If the predictions are wrong, the registered consequence still governs —
that is what pre-committing it is for.

---

# Reading (2026-07-18 — `corpus/supply/tautology_test.py`)

All four predictions confirmed exactly:

| case | obligations under ⊤ | verdict |
|---|---|---|
| hit386 (control) | unknown ×3 | bounds load-bearing — stands |
| dfx#417 (counted 1) | unknown ×4 | bounds load-bearing — **stands** |
| **dfx#207 (counted 1)** | **discharged** | **TAUTOLOGICAL — voided** |
| npy#249 (counted 1) | unknown | bounds load-bearing — **stands** (the `−20` floor is what the proof uses, via `exp`'s 0 endpoint) |

## Consequences, per the registered rule

- **dfx#207's count of 1 is void.** Its single obligation is `max`'s
  definition; the proof used no system, no box, no region. The chain the
  registration named is instantiated empirically: dfx#207 is *exactly* the
  case the property re-check filed as defective-anchor — **defective
  property → tautological obligation → VERIFIED → counted**, now caught by
  the third guard. (Fidelity was not its defect — it was honestly
  hand-transcribed; vacuity was.)
- **Control-flow hypothesis: 2 of 4 → 1 of 4 → the band drops from
  Supported to Weak** ("anything between — publish, don't build on it").
  Relation and fidelity sentence: *"1 mechanized (npy#249): 0 imported, 1
  hand-transcribed, precondition-with-ℝ-partial-gap, sound anchor."*
- **The machinery-attribution fact, said plainly:** the one surviving
  count (npy#249) used **no branch transfers** — its unlock was the
  loop-body-invariant framing. The control-flow machinery now has **zero
  surviving counting cases that used it.** It remains gate-licensed
  (3-of-6, re-adjudicated) and built and sound-pending-audit, with its
  mechanization evidence gone.
- **E2a with the expanded frame: 3 of 7 → 2 of 7** (dfx#417, npy#249) —
  Weak it was and Weak it stays; the denominator chain gains a fourth
  subtraction step (13 addressable → 9 reconstructible → 7 in-semantics →
  counts judged under fidelity + vacuity guards).
- Both survivors sit on **sound anchors** (property re-check) with
  **bounds-load-bearing obligations** — the two cases the whole pipeline
  now stands on are the two that pass every guard built so far.

---

# npy#249, pinned (2026-07-18, follow-on work order)

## The consequence, registered before the quote

> **A mechanized count of 1 is below any band.** One case is an anecdote
> about one system — the ≥2-sources principle that governs every
> threshold here, applied to the numerator. Wherever a count stands at 1,
> it is reported as *"one case — an anecdote, below any band,"* not as a
> band result. No band text changes; this is reporting vocabulary.

## The result, quoted — it was run and committed before this question was asked

The ⊤-widening runner covered **all four** counted cases in one pass
(`corpus/supply/tautology_test.py`, committed with the reading above):

> `npy#249 (counted 1): ['unknown'] -> no obligation survives ⊤ — the
> declared bounds are load-bearing`

And the prediction, pre-committed in the registration above **before the
run**:

> *"npy#249 → NOT tautological: under ⊤, `exp([−inf, inf]) = [0, inf]`
> and `> 0` straddles at the 0 endpoint — the declared `log_eps ≥ −20`
> floor is load-bearing."*

**By the registered test, npy#249 stands. The control-flow count is 2**
(dfx#417, npy#249), and the below-any-band consequence stays dormant.

## The instrument gap this exposes — recorded, not acted on

The question was sharper than the test. The registered test measures
bounds-dependence **over the extended reals**: ⊤ includes −∞, and
npy#249's obligation fails under ⊤ *only at the −∞ endpoint* (`exp(−∞) =
0` in the domain's closure). Over **finite** ℝ, `exp(x) > 0` is a theorem
— true for every finite input, no box needed — which is the same *shape*
as dfx#207's `max`-definition, one domain-subtlety away. The declared
`−20` floor is "load-bearing" only in that it excludes −∞ itself.

**The registered test governs and the count stands** — refining the test
now to catch finite-ℝ range-theorems would void a counted case, and that
amendment fails the rule on every clause: it is restrictive, it moves a
count, and the gap was found by review, not by the registration's own
control. The refinement binds **forward**: any future vacuity test — in
particular the one the corpus-expansion registration must carry for its
own cases — should test against *finite*-⊤ (or equivalently: an
obligation that is a theorem of a primitive's range over finite inputs is
tautological), registered before the cases it could void exist. Same
pattern as the ≥2-sources class fix: the instance is grandfathered
because the rule forbids retroactive tightening; the class is fixed.

Both survivors' standing, restated with everything now known: dfx#417 —
sound anchor, bounds-load-bearing in the full sense (the face fluxes go ⊤
when `u, v` widen, finitely). npy#249 — sound anchor, passes the
registered test, **and its obligation is a finite-ℝ range-theorem**, said
here so the count's meaning travels with the count.

## The standing figure carries both numbers (2026-07-18, follow-on)

The count was being carried as "2 of 7" alone — true only under the test
this project has already decided to supersede. The honest standing
figure, wherever it travels:

> **Control-flow E2a: 2 of 7 under the registered (extended-ℝ) test;
> 1 of 7 — dfx#417 alone — under the finite-⊤ criterion the
> corpus-expansion registration carries** (`design/corpus-expansion.md`,
> which re-scores the inherited cases under its own criterion in its
> opening section). Not a re-score of the registered result: the
> registered count stands as registered, the successor criterion's count
> rides beside it, and the pre-supersession number does not travel as
> "the result."

## The constrained-vacuity variant (2026-07-18, registered with the constraining-assume build, before any constrained verdict exists)

Constraining assumes (`design/constraining-assume.md`) introduce a new
way a verdict can pass: the *constraint*, not the *box*, did the work.
The instrument family extends, registered before any constrained verdict
has been read:

- **The control is the mode switch**: re-run the identical query under
  `propagate(..., assume_mode="inert")` — mechanically the pre-build
  behavior, byte-identical, no hand rewriting.
- **Readings, fixed now:** still discharges under inert → the constraint
  was not load-bearing; the claim is unconditional and must be reported
  as such (the constrained stamp lines overstate conditionality — drop
  them by re-running, or say "constraint not load-bearing" wherever the
  verdict travels). Discharges only under constrain → the constraint is
  load-bearing; the verdict's content IS the conditional claim ("holds
  where the precondition holds"), carried by the stamped
  constrained-assume lines; this is a *different, weaker claim* than an
  unconditional VERIFIED and never travels without its condition.
- **Composition with the inputs-only ⊤-widening** (the registered box
  instrument): the two instruments separate what did the work — the box
  (widening flips it), the precondition (inert-mode flips it), both, or
  neither (tautology; cannot count, as registered).
- **The empty-region refusals** (`UnsatisfiableAssumptionError`: empty
  meet, definitely-false constant precondition) are the mechanical
  guard for the vacuous-precondition case; the known limit — a
  strict-at-boundary assume narrowing to a closed point whose true
  region is empty — is visible through this instrument (the inert-mode
  control flips such a verdict) and through the stamped point region.
- **Counting rule inherited**: any future counted case that used a
  constraining assume runs this variant in the same breath as the
  ⊤-widening, and the count carries the outcome in-sentence.
