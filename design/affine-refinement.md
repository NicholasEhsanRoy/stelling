# The affine refinement — build, soundness audit, and the held-out evaluation

**Status:** PASS RECORD, 2026-07-22. Roadmap item 6, unparked by order
with the evidence condition independently met: the item was parked
"until a measured UNKNOWN-that-matters names it," and the LA work had
since produced exactly those measured cases (the probe's Part 3
dependency straddle; the MIME socket's solver-discharged symmetry
pair). Built under a **held-out evaluation discipline**: a scout
measured the practical use cases first and wrote them to a holdout the
builder and auditor never read; the evaluation ran only after the
audited build was gated. Exhibit: `corpus/supply/affine_holdout/`.

## What shipped

`stelling.affine` — affine forms (zonotopes) over the declared boxes as
an **opt-in refinement** (`refine="affine"` on `check`/`check_contract`;
never-on default; unknown values refused eagerly) that runs after
interval judging and before solver escalation, deciding
interval-undecided obligations where correlation recovery suffices:
`lo ≥ 0` → discharged, `hi < 0` → set-level REFUTED (both licensed by
the containment invariant: the concretization always contains the true
range), everything else falling through unchanged. Zero-dep, jax-free
import; all coefficient arithmetic exact-rational with outward snaps
through the interval module's bracket kernel; v1 scope = linear ops
exact + **canonically-shared product residues** + structural routing,
with everything else (div, min/max/select, strict/eq roots, ieee,
constrained-assume) declining the whole obligation loudly.

**The novel decision (audited sound):** nonlinear mul residues ride on
*shared canonicalized product symbols*, not on unsigned `err` — unsigned
bounds add where only shared symbols cancel, so this is the one design
under which commuted products (`x·y − y·x`) cancel exactly. The spec's
letter said "into err" and was internally inconsistent with its own
known answer; the builder's deviation is the correct resolution.
Sharing is keyed by form content only at `err == 0` (where a form IS an
exact function of its symbols) and by `(var_id, element)` identity
otherwise; the auditor could not construct two shared-symbol products
with differing deviation functions, and `(x·y)·z` vs `x·(y·z)`
correctly do not share.

## The audit (fourth consecutive zero-UNSOUND round)

Containment survived ~4,000 exact-rational fuzz queries (0 violations;
2,643 decided obligations, no false discharge, no false set-refute) and
thirteen targeted constructions. Seven findings, all instrument-grade:

1. **F1 MISLEADING (L20's family, third instance)** — the vacuity
   stamp's "envelope is load-bearing" clause was narration beyond the
   measurement: false on range theorems, pre-existing with refine off,
   made systematic by affine because **the widened re-check runs weaker
   than the original** (affine declines unbounded boxes by
   construction). Fixed: the clause now states the measurement ("under
   the mechanism(s) that ran, this VERIFIED was not re-derivable
   without the declared envelope") plus an explicit reduced-power
   disclosure when the re-check ran weaker — the instrument reports its
   own power. The one deliberate refine-off wording change, enumerated
   per call site in the fix report.
2. **F2 SHARP-EDGE** — four auditor-invented mutation classes, each
   demonstrated unsound by exact-rational measurement, survived every
   shipped gate (the battery lacked err>0 mul operands, non-dyadic
   constants, and near-threshold chains). Fixed by adopting the
   auditor's constructions; the extended battery catches all eleven
   mutations, residual empty.
3. **F3 MISLEADING** — the doc promised overflow declines that were
   dead code; the measured behavior is sound saturation with the excess
   accounted into err. Fixed: dead branches deleted, docstring
   rewritten to the measured story, and a per-obligation disclosure
   note whenever a snap saturated — a definite verdict never rests on
   saturation silently.
4. **F4/F5/F6b** — the arithmetic stamp line names the zonotope when it
   decided; "ran" → "enabled" for zero-attempt stamps; the
   decision-rule comment corrected to "strict subset of what interval
   decides." F6a recorded as the novel decision above; F7 (widen
   re-check invocations superseding the absence line) is pre-existing
   truthful tagged mechanics.

## The held-out evaluation — the reading

Protocols from `SCOUT_CASES.md` (measured at 3f78fdd, pre-build), run
by the orchestrator against the gated build; every refine-off re-run
showed **no drift** (5/5 runners; case 2 differed only in solver
milliseconds).

| case | baseline (measured pre-build) | refined (measured post-gate) |
|---|---|---|
| 1 — real MIME socket symmetry pair | interval-UNKNOWN (±3.5e-323 hulls); VERIFIED only via 4 QF_NRA invocations (69/9/75/8 ms) | **VERIFIED, zero solver invocations** — both obligations `discharged by affine refinement`; absence line names both layers, "escalation had nothing to do" |
| 2 — quadratic conditioning | UNKNOWN (interval rhs hull to 43.03); solver VERIFIED | affine **tightened the slack to [-1.46875, 29.03125] and did not separate** — the probe's "quadratic past plain affine" prediction confirmed; solver still closes it |
| 3 — fvm raw form (div; refutation-shaped per the scout's hand truth) | UNKNOWN, div/constrained-assume declines | no drift; div outside AFFINE_SUPPORTED (in-suite measured decline) — correctly out of scope |
| 4a/4b — `0.8·s < s` and the select-join ratchet | UNKNOWN, solver-unreachable (bool declaration) | declines: `primitive 'select_n' is outside AFFINE_SUPPORTED` |
| 4c — `t + dt > t` at a point | UNKNOWN; closed today by a QF_LRA call | declines: `root comparison 'gt' is a strict half-space; the v1 decision rule covers closed half-spaces (ge/le) only` |
| 5a — MADDENING HeatNode box invariance (18/20 elements, pure correlation loss) | UNKNOWN; escalation declined on scatter emission | declines: `the obligation slice is unavailable: primitive 'scatter' is outside the supported emission set` |

**Honest summary of what v1 buys:** the commuted-product class on real
code — real solver spend eliminated on the MIME socket's symmetry pair
— plus a measured tightening (not closure) of the quadratic class,
exactly as the probe predicted. **The v2 frontier, named by
measurement, in expected-value order:** scatter routing in the slice
layer (unlocks the HeatNode flagship — the largest measured prize),
`select_n` join handling (the ratchet class), strict-root support for
ℝ-side claims (`t+dt > t`). No false discharge anywhere; the controls
stayed honest.

## Suite state at commit

venv-jax **1056 passed**; venv-nojax **850 passed + 16 skipped**
(trajectory 1003 → 1054 build → 1056 fixes; 809+15 → 849+16 → 850+16).
`refine=None` behavior byte-identical everywhere except the enumerated
F1 vacuity wording (measured-false clause replaced by the measured
statement, refine-off included, statuses unchanged). The MIME socket
does not opt in and is untouched; adopting `refine` there is MIME's
call, and the holdout exhibit's declines are the first pins to move if
a future refinement extends the frontier.
