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
