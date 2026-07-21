# CI readiness — the construction-path census, then the CI-mode field test

**Status:** REGISTRATION, 2026-07-21, committed **before the census
reads or any field-test verdict exists**. The framing that governs the
pass: **CI trust is a reliability-and-evidence property, not a
capability property.** Three readiness claims have been answered by
argument and never by measurement — the project's own meta-rule, turned
on its own readiness:

1. *"Every reachable construction path produces safe IR"* — asserted,
   load-bearing for the I1 residual, never enumerated.
2. *"Stelling is already useful"* — every finding so far
   (F1, R1, magnetics SPD/mass) was **hand-guided**: stelling pointed at
   an obligation already suspected interesting. CI value is the other
   mode — catching what nobody was already worried about — and evidence
   in that mode is **zero** because the tool has never run in it.
3. *"Affine/LA are deepening, not broadly useful"* — argued, not
   measured (Part C corrects the record).

## PART A — the construction-path census (gates Part B)

**Claim under test:** every way an `ir` object comes into existence
routes through a gate guaranteeing well-formedness.

**Method, fixed now:** enumerate by AST/grep over the whole repo — every
instantiation of every `ir` dataclass (`Jaxpr`, `ClosedJaxpr`,
`JaxprEqn`, `Var`, `Literal`, `Array`, `Aval`, param types), every
factory/classmethod, every `dataclasses.replace`. Classify each site:
trace-door / from_dict-door / internal-transform (does it preserve
well-formedness given valid input?) / test-only (reachable from public
API or not?) / direct construction. **The census-of-the-census:** state
what makes the list complete — and confront the structural fact that
Python dataclasses are freely constructible, so any funnel claim that
is not enforced *in the types themselves* is convention.

**Bands (fixed):**

| finding | reading | action |
|---|---|---|
| every reachable path routes through the gate; the funnel invariant holds | I1 genuinely out-of-contract; "safe on all reachable paths" **verified** | proceed to Part B |
| an ungated path exists, reachable from the public API | a real soundness gap (the z3-defect shape: latent, on an unexercised path) | **STOP** — structuralize the gate to cover it (4-B witnessed constructions), then re-gate |
| completeness cannot be established (construction does not funnel) | "safe on all paths" is unprovable as architected — a design finding that reranks the CI path | **STOP, surface** |

Per the work order: a row-2 finding is fixed in-pass (structuralize,
witnessed constructions) and the gate re-adjudicated; row 3 surfaces.

## PART B — the CI-mode field test (only if Part A lands row 1)

First run of stelling in the CI workflow: preconditions posed over a
codebase **unguided** — no pre-selected interesting obligation. The
unguided protocol, fixed now so envelope choice cannot steer: for the
target module(s), enumerate mechanically every (i) division/reciprocal
(denominator nonzero over the envelope), (ii) `sqrt`/`log`/`pow`
argument (domain), (iii) config scalar with a default (nonzero /
admissible-range), (iv) coefficient field feeding a solve (positivity),
and pose **all of them** via the precondition templates over disclosed
envelopes (generic sign-unknown boxes plus any documented supported
range, both reported).

- **B1 — MADDENING** (useful-to-Nick): the wavelet solver core, posed by
  the main agent under the mechanical protocol.
- **B2 — an external repo** (useful-to-strangers, the stronger test):
  **lineax 0.1.1** — real solver code, not written by Nick, censused for
  equation counts once but never precondition-analyzed; its documented
  solver tags (callers *assert* positive-definiteness and similar) are
  exactly the assumed-precondition class. Blind transcriber, repo
  source only, no expectation of what is there.

**Adjudication duties, fixed:** every REFUTED gets a false-positive
adjudication (real unguarded precondition vs guaranteed-by-something-
the-tool-cannot-see), by the main agent, per finding; every UNKNOWN
joins the decline map (what a CI integration could check vs stay silent
on); every VERIFIED gets a `widen()` vacuity spot-check. **The
false-positive rate is the number that decides CI-viability and is
reported whatever it is.**

**Reading bands (fixed):** genuine finding + low false-positive rate =
the missing CI evidence and the adoption artifact; real findings + high
false-positive rate = the gap is precision and the causes are the
finding; clean pass = distinguish "preconditions genuinely sound" from
"class cannot reach them" via the decline map. **No outcome is a
failure of the pass.**

## PART C — the CI-trust gap, named (report-only)

(i) Which of the manual backstops that protected shipped soundness
(main-agent witness re-replay, both-faces adjudication, per-harness
vacuity runs) are already structural vs would need to become automatic
before CI-trusted stelling matches human-supervised stelling. (ii) The
affine/LA record correction: their non-usefulness was argued, not
measured, and the magnetics characterization's governing property
(residual-bounds-error via conditioning) is LA-shaped **with a measured
scar** — evidence *for* immediate usefulness. Correct the record; build
nothing.

## Non-goals

No new capability (structuralization of Part-A-exposed convention only).
Part A gates Part B. Unadjudicated findings are not evidence. No
affine/LA build. No publish. CALMS, qMRI, E2b out of scope.

---

# Part A reading (2026-07-21 — registration `6f7caad` preceded it)

## The census

**Enumeration** (AST/grep over every `ir` dataclass instantiation,
whole repo): **19 test files** (hand-built IR, not reachable from the
public API), and **five src locations**: `ir.py`'s own `_decode`
factories (the from_dict door — validated), `_jax_compat.py` (the trace
door — validated, jax refuses malformed upstream), `vacuity.py` (the
widen transform — rewrites `lo`/`hi` params only; well-formedness-
preserving on valid input), `obligation.py:886` (alias substitution —
copies an existing eqn with resolved invars; preserving),
`propagate.py:1088–1125` (the behavioural census-assert probes —
synthetic, valid by construction).

**The census-of-the-census:** `ir` objects are frozen dataclasses in a
**public module** — freely constructible by any consumer. So the funnel
claim ("all construction routes through the gates") was **false at the
language level**: direct construction, exactly I1's route, was a
reachable ungated path. **Band row 2.**

## The structuralization (the row-2 action, taken in-pass)

Validation moved **into the types**: `Aval`/`Array`/`Literal`/
`JaxprEqn`/`ClosedJaxpr` `__post_init__`s now call the **same shared
predicate helpers** the from_dict door already used (one implementation,
now every caller) — local checks only, composing to the door's full
coverage without recursion: shape extents integral and nonnegative
(`Aval`, `Array`), payload length (`Array`), value-vs-aval agreement
(`Literal`), declaration params-vs-aval agreement (`JaxprEqn`,
**I1's exact instance**), const-vs-constvar pairing (`ClosedJaxpr`).
The door keeps its walk (a second caller of the same predicates — load-
path error context, not a second implementation).

**Consequences, all measured:** the malformed IR of the entire
N/P/R/I regression arc is now **unconstructable** — 19 tests superseded
into 7 construction-raise pins (raise strictly earlier; intent holds a
fortiori), with the honest halves preserved and one live door test kept
(a corrupted *dict* never constructs objects until `_decode`). The
type-level check immediately caught **my own `test_vacuity` helper
lying about its dtype** (params `"<f8"` vs aval `"float64"`) — the
structuralization catching its first defect before the commit that
lands it. L15 counterfactual: 6/7 new pins fail against unfixed
`ir.py`; the seventh (the door test) passes both ways by design. All
recorded harnesses re-run clean; suites 894 / 746+13.

## The gate, re-adjudicated: **row 1, with the bound stated**

Every construction path — trace, `from_dict`, internal transforms,
tests, and **direct dataclass construction** — now passes through the
types' own validation, because Python guarantees `__post_init__` runs
on dataclass construction. **The stated residual bound:** deliberate
constructor circumvention (`object.__new__` + `__setattr__`) bypasses
any Python-level validation and is excluded from the contract, as it is
for every Python library; and per-primitive shape inference for
non-declaration equations remains explicitly out of scope (the I1
species that disputes an *intermediate* eqn's addend count is
unconstructable for declarations — the checked class — and remains
FRAGILE-by-convention beyond it). Within that stated bound, **"safe on
all reachable construction paths" is now verified by enumeration plus a
language-level funnel, not asserted.** Part B may run.

---

# Part B reading (2026-07-21)

## B1 — MADDENING, unguided (8 posable sites, 2 out-of-class)

S2 is the finding: **the D-clamp** (`precond.py:77`,
`where(D > 0, D, 1.0)`) — `diag ≠ 0` REFUTED, witness `(0,1,0,0)`,
violating elements named. Adjudicated **REAL-BUT-CONDITIONAL**: it
composes with S7's admissible `mass = 0` (the constant mode's diagonal
entry is then exactly 0), so the clamp silently substitutes 1.0 and the
Jacobi preconditioner silently no-ops on the very mode where the
operator is singular — two REFUTEDs chaining into one silent-degradation
scenario. The precision gap is named: the tool cannot see the
`mass > 0 ⇒ diag > 0` theorem, which is **LA-shaped knowledge** (Part C).
S3/S4 VERIFIED but **tautology-shaped** under `widen()` (range theorems:
`|x| ≥ 0`, `Σx² ≥ 0`) — reported as *guards verified as unconditional
theorems*, not envelope-dependent facts. S6 VERIFIED, bounds
load-bearing. S1/S5 honest declines (`sqrt`, `abs` outside the emission
set — decline-map data). S7/S8: the guided pass's findings re-derived by
the unguided protocol.

## B2 — lineax 0.1.1, unguided, blind (81 sites, 52 poses)

**Tally: 9 VERIFIED, 17 REFUTED, 26 declines** (dominant decline:
`dot_general` outside the emission set ×10 — every tree_dot/quadratic
form; then `abs`, `iota`, `cond`, non-finite constants, zero-crossing
divisors, int64 declarations).

**The false-positive adjudication (every REFUTED, main agent, against
the source):**

- **REAL — 8 of 17.** The flagship: **`diagonal.py:81` — a
  `well_posed=True` `DiagonalLinearOperator` divides by the
  caller-asserted-nonzero diagonal with NO check anywhere** (the
  `well_posed=False` branch has an rcond guard; the `True` branch is
  bare `vector / diag`) — witness `(0,1,0,0)`, violating elements named:
  a tag honoured, never verified, silent inf/nan. **`triangular.py`
  stored diagonal ≠ 0** — same class. **The tolerance guard bypass**:
  `__check_init__` validates tolerances behind `isinstance(…, (int,
  float))`, so an **array-typed tolerance skips validation** and a
  negative tolerance flows in silently — a guard that exists and misses.
  `DivLinearOperator` scalar ≠ 0 unguarded. Plus four unguarded config
  scalars (`stabilise_every`, `restart`, `stagnation_iters`,
  `max_steps` — verified: `__check_init__` checks none of them; the
  `mass = 0` class, low severity).
- **FALSE ALARMS — 9 of 17**, three nameable causes: (a) **loop-state
  denominators posed as inputs** (the lsmr `rho` family and two sqrt
  domains — algorithmic invariants guarantee them, and they are
  *out-of-class by the registration* — the mechanical protocol failed to
  apply the input-side boundary to division sites); (b) **tag semantics
  misread** (`unit_diagonal` is an *instruction* to jax to ignore the
  stored diagonal, not an assertion about it — 2 poses); (c) **sentinel
  and guard-strength conventions** (`conlim = 0`, `rcond = 0` are
  meaningful values the guards deliberately pass — 4 poses).

**The number: raw false-positive rate 9/17 (53%); applying the
registered class boundary the protocol should have enforced (loop-state
sites excluded), 6/14 (43%).** Middle band: real findings, high
false-positive rate — **the CI gap is precision, and its causes are
finite and named**: denominator-provenance classification (input vs
loop-carried), tag-semantics reading (is the tagged data actually
read?), and sentinel conventions. Each is a protocol/analysis
refinement, not a soundness problem.

**The adoption artifact exists**: an unguided, blind sweep of an
external library found a genuine honoured-never-verified solver tag with
a concrete element-named witness, plus a real validation-gate bypass —
on code stelling's author did not write.

# Part C — the CI-trust gap, named (report-only)

**Already automatic** (structural, run on every verdict): witness replay
(membership ∧ violation, conjunctive), the provenance gate, the census
asserts, stamp validation, the from_dict/type-level IR gates.
**Not automatic today:** (i) **the vacuity control** — `check()` does
not run `widen()`; per-harness vacuity is a manual discipline (B1's
tautology-shaped VERIFIEDs were caught only because the duty was
registered) — the clearest candidate to automate; (ii) the recorded-set
no-flip gate (automatable as a CI job; today a manual step); (iii) the
**both-faces adjudication and false-positive adjudication** — inherently
human, semantic judgements (B2's 9 false alarms were separable from the
8 real findings only by reading the target's source); CI can *surface*,
a human still *adjudicates*. A CI-trusted stelling equals the
human-supervised one only after (i) and (ii) are wired; (iii) is the
honest permanent residue.

**The affine/LA record, corrected**: "LA is deepening, not broadly
useful" was **argued, not measured**. The measured record now says: the
magnetics governing property (residual-bounds-error via conditioning) is
LA-contract-shaped *with a measured scar*, and B1's one real composed
finding (the D-clamp chain) is blocked from full precision by exactly
the `mass > 0 ⇒ diag > 0` theorem — **LA-shaped knowledge appearing as
the false-alarm cause in the field test**. Two independent evidence
points *for* LA's immediate usefulness; the roadmap note is corrected —
LA's status is *unmeasured, with specific evidence it would bite*.
