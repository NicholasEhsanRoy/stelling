# The soundness audit — a fresh context, the code, and no history

**Status:** record, 2026-07-18. The audit ran as a subagent given the
source (`interval.py`, `propagate.py`, `ir.py`, `coverage.py`, the tests)
and the stated semantics (the docstrings), with **history withheld** — no
design/, no corpus/, no verdicts, no counts, no bands. Task: *find the
soundness bugs*, not *improve this*. The rationale is the project's own
portfolio-dispatch precedent: two solvers disagreeing means someone is
wrong, and the main agent **wrote** these transfers — the one thing that
disqualifies it as their reviewer.

## What it found (all constructions verified by execution; fixed same day)

| # | severity | defect | disposition |
|---|---|---|---|
| 1 | UNSOUND | `convert_element_type` passed value-changing casts through (f64→f32 rounds, int64→int32 wraps, int→bool collapses). **Verified false VERIFIED from a real jax trace** — `float32(0.1)` rounds up, `roundtrip(x) ≤ 0.1` discharged while concretely false. The most reachable defect: any f32 harness | fixed: exact-conversions whitelist; all else ⊤ with a note |
| 2 | UNSOUND | `cond` clamped negative indices to branch 0. The auditor **bound `cond_p` directly** to establish ground truth: index −1 → **last** branch, 5 → last. Verified false VERIFIED with index ∈ [−1, 0] | fixed to the verified default-last convention, ±inf-guarded |
| 3 | UNSOUND | `_decode_array` decoded int64 above 2⁵³ to point intervals excluding the true value (the python-int literal path bracketed; the array path didn't). Verified false VERIFIED on `(2⁵³+1) − 2⁵³ < 0.5` | fixed: shared `_int_bracket`, saturating beyond double range |
| 4 | UNSOUND (coverage) | untaken `cond` branch equations vanished from the coverage denominator — `propagate` reported `100%` where `measure` reported `83%; 1 ⊤` on the same query. A wrong definite claim by the instrument the falsifier discipline leans on | fixed: untaken branches counted unreached |
| 5 | FRAGILE (crash) | ⊤ selectors crash `select_n` (`OverflowError`) — **reachable from `jnp.where(jnp.isnan(x), …)`** (the `ne` predicate has no transfer → ⊤ → crash); likewise degenerate cond indices and out-of-range int literals | fixed: degrade to join-all / saturate |
| 6 | FRAGILE | the flat shared environment let one branch read another's internals, silently defeating the promised unbound-var error | fixed: isolated scopes per branch/call run |
| 7 | FRAGILE | `join` silently truncated mismatched shapes; structural ops trust params | shape guards added to `join`/`select_n`; full structural-param validation recorded as a work item |

Clean areas, checked explicitly by the auditor: the interval arithmetic
itself (mixed 0/±inf products, half-infinite division, exp endpoints,
overflow saturation), the three-valued comparisons in both definite
directions, the unknown-primitive ⊤ path for `while`/`scan` (no silent
truncation anywhere), the obligation classifier, and the literal decoder's
refusals. **Every wrong status the auditor produced came through findings
1–3, never through the classifier or the arithmetic.**

## No verdict flipped — verified, not asserted

Before the SOUNDNESS log entry was written, every recorded harness was
re-run under the fixed code: hit386 VERIFIED / mutation REFUTED; dfx#417
VERIFIED / mutation UNKNOWN; cf_run statuses unchanged; the ⊤-widening
results unchanged. The counting queries used only exact conversions
(f64→f64, bool→int32), definite in-range indices, and f64 constants — the
defects sat outside every recorded path. Each audit construction is now a
permanent regression test (`tests/test_audit_findings.py`; the real-trace
f32 case in `tests/test_harness.py`).

## The two observations worth keeping

- **The bugs clustered in the newest code.** Findings 1, 2, 4, 5, 6 are
  all in or adjacent to the control-flow pass — the machinery built
  fastest, under a hypothesis, in one commit. The MVP core (arithmetic,
  comparisons, classifier), built under the paranoia budget, survived
  every attack. Speed of construction predicted defect density exactly.
  *Attribution sharpened (2026-07-18, follow-on):* speed was a variable —
  **so was stake**. The core was built before there was a hypothesis to
  confirm; the control-flow machinery was built *to make the six
  posable*, with an outcome it needed to reach, and dfx#207 (a
  tautological pass that counted) is the fingerprint. Motivated building
  reads like motivated analysis. Forward consequence registered where the
  finding lives: `design/corpus-limits.md` — `any_pytree` carries a
  fresh-context audit as a build gate, before any newly-enabled case
  counts.
- **The auditor's blindness was the method.** It bound `cond_p` directly
  to learn what jax actually does instead of trusting the transfer's
  comment; it attacked the definite-FALSE direction nobody had reason to
  doubt; it found the coverage lie by diffing `propagate` against
  `measure` on the same query. None of that requires project context —
  all of it is harder with a stake in the verdicts. Disagreement as a
  bug oracle, now with a precedent in-repo.

---

# Second pass (2026-07-18): the transfers the first pass's gradient skipped

**The general form, recorded before the results:** *an audit inherits the
auditor's attention gradient, and a single audit pass is not uniform
coverage.* The first pass scrutinized `cond` hardest — where the builder's
attention had gone — and declared `mul`'s 0·∞ products, `le`/`ge`, and
`select_n` clean. The second pass, aimed deliberately at the boring
primitives with one executed construction per area, found real issues in
exactly those areas. A clean pass on the interesting primitive says little
about the boring ones.

## Findings and dispositions

| # | severity | finding | disposition |
|---|---|---|---|
| 4-B | UNSOUND | the float→int range guard admitted exactly ±2³¹ (int32 max is 2³¹−1; jax clamps, numpy wraps — no outcome equals 2³¹). **Introduced by the first audit's own fix** — the exact-conversions rewrite added the guard with an inclusive bound, and for int64 the float `bound−1` rounds back to `bound`, so only a strict check is sound | **fixed** (strict upper bound); both boundary constructions are regression tests. The guards-generate-hazards arc (#632-fix→#756), previously an upstream observation, now has an in-repo instance: the fix to finding 1 introduced finding 4-B |
| 3 | UNSOUND | `select_n`'s empty-picks fallback selected the **last** case; measured jax `lax.select_n` **clamps** (index −1 → case 0, eager and jit agree) — verified false VERIFIED (tool said 30.0, jax computes 10.0). The asymmetry with `cond` (default-last, re-verified same build) is real and now documented in both transfers | **fixed** (clamp convention, below-range → first case, above-range → last); regression tests cover definite, entirely-below, straddling, and above-range selectors |
| 4-A | correct-under-ℝ | `(x+x)·0` over a finite box: the analysis's overflow saturation gives `[maxfloat, ∞]`, the 0·∞ convention gives 0, `z < 1` discharges — **true in ℝ, false in IEEE for every declared input** (`inf·0 = NaN`). End-to-end through `cond` and `select_n`: a concretely-always-false predicate reached the index position as definite | **not a code defect under the registered `semantics: real` dial** — it is the *strongest exhibit of the registered gap yet*: an ℝ-true definite discharge, false in floats, from **straight-line arithmetic on a finite box**. It **refutes** the earlier scope claim that outward rounding makes monotone arithmetic float-conservative (`design/semantics-classification.md`, amended): overflow→NaN paths are not guarded. Pinned as a marker test that flips the day the dial moves |
| 1/2 | correct-under-ℝ | ⊤ = [−∞, ∞] contains every *real* but not NaN; `r ≤ +∞` over a ⊤ loop output discharges while the concrete run yields NaN | same dial; same marker treatment. **The clause that stood here — *"and the ⊤-widening vacuity guard already fences this shape out of every count"* — is WITHDRAWN (2026-08-21) as false**, in `SOUNDNESS.md`'s entry for this audit and here. The guard's predicate is whole-obligation — does it still discharge with every declared bound widened to (−inf, +inf)? — so it fences this shape where the obligation is a tautology *entire* — and it fences other things too, an `assume` surviving the rewrite among them, so it is not a *characterisation* of the shape in either direction (`assume(3 ≤ x ≤ 4); assert_(x·x ≤ 16)` is false at x = 5 and the guard still FIRES, both x64 cells). Driven with a control, both x64 cells, `vacuity_mode="inputs-only"`: `r ≤ +∞` **alone** over a ⊤ `while` output is VERIFIED and the guard FIRES; the same `r`, times `0.0` and conjoined with `x ≥ 0` over `[0.5, 1.0]`, is VERIFIED with the stamp reading *"no obligation discharges with the declared bounds widened"* while jax is False at 25 of 25 grid points. What catches it is the `semantics="ieee"` dial (UNKNOWN, same run), which is this row's first disposition and is now its only one |

## PASSes — the first audit's assertion, now witnessed

The "no silent truncation anywhere" claim is no longer an assertion:
executed witnesses show a 1000-iteration `while` yields `unknown` for both
a concretely-true and a concretely-false obligation with body equations
counted unreached; a concretely-zero-trip `while` is not exploited; an
obligation *inside* a loop body is never judged (counted unreached, never
silently VERIFIED); a doubling `scan` leaks no partial unrolling in either
direction; a length-0 `scan` keeps the carry/ys distinction under ⊤. The
comparison→bool→convert→index chain emits exactly {0.0, 1.0} endpoints,
bool arithmetic that gets outward-bumped floors back to a **widening**
join (never index −1/2, never a narrowing), and a genuinely-straddling
predicate never reaches an index as definite.

## No verdict flipped — re-verified after the second round of fixes

hit386 VERIFIED / mutation REFUTED; dfx#417 VERIFIED / mutation UNKNOWN;
cf_run statuses unchanged; ⊤-widening results unchanged (the counting
queries contain no out-of-range selectors, no float→int boundaries, no
overflow-reaching arithmetic on asserted paths, no ∞-bound obligations).
119 tests green.

# Standing audit-process rules (2026-07-18, registered after the solver-layer audit)

The deeper lesson of the audits so far: **a finding reveals two things —
the bug, and the fact that the bug was *possible***, i.e. that some
invariant was maintained by care rather than construction. The project
already turns findings into regression tests; these rules bind the
second half. Three rules, standing, for every future audit:

1. **The structuralization question is mandatory.** Every UNSOUND or
   FRAGILE finding triggers, alongside its regression test: *does this
   finding reveal an invariant that should be structural rather than
   checked?* If yes, the structural fix ships with the regression test —
   the test catches *this* instance; the structure closes the *class*. A
   regression test alone leaves the class open to the next variant.
   First application: the solver layer's witness-conjunction validator
   and append-only stamp (`design/solver-hardening.md`), both
   structuralized from findings that had been regression-tested first.

2. **UNSOUND fixes are re-attacked by the auditor, not just
   regression-tested by the builder.** Fixes are the least-audited
   code — written under "make the finding go away" pressure, by a warmed
   context with a stake in the fix closing. Finding 4-B is the in-repo
   witness: the first audit's fix introduced the second audit's defect.
   The two-context split exists to break exactly this pressure, and it
   applies to fixes as much as to builds. (The solver build's fix round
   predates this rule: it was main-agent re-verified but not
   auditor-re-attacked — recorded here, not repaired retroactively.)

3. **Acceptance and audit must be anti-correlated in coverage.** After
   each audit, name what the acceptance case did NOT exercise, and
   confirm the auditor's mandate covered exactly that gap. A bug on both
   paths gets caught twice (the witness-membership UNSOUND was); the
   dangerous bug is the one on *neither* path. The auditor aims where
   the acceptance case cannot see, not where it already looks.
