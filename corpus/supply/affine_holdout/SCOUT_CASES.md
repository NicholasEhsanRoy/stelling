<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# Affine holdout — measured cases where interval propagation loses input correlations

Measurement record, 2026-07-22. This file DOCUMENTS current behavior; it designs
nothing and predicts nothing. A future relational/affine refinement is to be
evaluated by re-running the per-case scripts below and recording what changed.

## Environment and tree state (verbatim)

- stelling working tree: `/home/nick/MSF/stelling` at commit
  `3f78fddd938acead26b49817beec97675aa5ec59`
  ("scatter-add and stack rows: the censused build that ends transcription"),
  working tree **clean**, unchanged from first measurement (01:04:35Z) through
  last (01:14:02Z). Note: the session-start snapshot had shown HEAD at 173a555
  with uncommitted edits — the concurrent agent committed 2e46265 and 3f78fdd
  before this scout's first run; every measurement here is from 3f78fdd.
- python: venv-jax
  (`/tmp/claude-1000/-home-nick-MSF-stelling/a4ecc48a-03c7-477a-9fb8-26618f8051cd/scratchpad/venv-jax`),
  python 3.12.3, jax 0.11.0, stelling 0.1.0 (editable, imports from the working
  tree), mime-engine 0.2.0, maddening 0.3.0, z3 5.0.0, cvc5 1.3.4-modified.
- MIME socket (read-only): `/home/nick/MSF/msf/MIME/verification/stelling/mime_lsq_conditioning.py`,
  prints `pinned: mime-engine 0.2.0 @ 7ce1efb4311b`.
- All raw outputs referenced below live in `runs/` next to this file.

Run any case:

    VENV=/tmp/claude-1000/-home-nick-MSF-stelling/a4ecc48a-03c7-477a-9fb8-26618f8051cd/scratchpad/venv-jax
    $VENV/bin/python caseN_<name>.py

---

## Case 1 — the commuted-product symmetry pair in the real MIME socket (KA-B)

### (a) Source and how to run

- Real socket: `/home/nick/MSF/msf/MIME/verification/stelling/mime_lsq_conditioning.py`
  (MIME-side, read-only). KA-B poses `conditioning_2x2_field` over the real
  2x1 Cartesian mesh, boundary-fed, all declarations point intervals; the
  contract poses symmetry as two closed obligations
  `M[..,0,1] <= M[..,1,0]` AND `M[..,1,0] <= M[..,0,1]`
  (the pair of `obligations.append(assert_(...))` calls in
  `conditioning_2x2_field`, `src/stelling/contracts.py`).
  *This cited `contracts.py:690/691`, which today is the middle of the triple-path
  shape validation; the sentence it quotes is at 670/671 and the posing calls
  are at 754/755 — measured 2026-08-09. The same wrong pair is printed by
  `case1_REFINED.py` and `case1_mime_symmetry.py`; all three are corrected to
  the symbol.*
- **The socket no longer runs un-modified from the current tree.** Commit
  3f78fdd added a `scatter-add` transfer; the socket's seam probe (its line
  197) asserts scatter-add has NO transfer, and now fails:

      Traceback (most recent call last):
        File "/home/nick/MSF/msf/MIME/verification/stelling/mime_lsq_conditioning.py", line 197, in <module>
          assert "scatter-add" in _seg_unknown_prims
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      AssertionError

  Measured seam block under 3f78fdd (was UNKNOWN + decline before):

      ==== seam: segment_sum under the tracer (measured)
        propagation coverage: 20 eqns: 20 known (100%)
        primitives with no transfer: []
        obligation statuses: ['discharged']

  (`runs/case1_mime_lsq_socket.out|.time`, `runs/case1_socket_traceback.txt`;
  wall 0.60 s to the failure, exit 1.)
- Full socket behavior was still measured end-to-end with asserts stripped:
  `python -O mime_lsq_conditioning.py` (`runs/case1_socket_dashO.out`, wall
  6.89 s, exit 0) — every gate's inputs print, no gate fires. Everything
  downstream of the seam matched the socket's committed expectations (KA-A
  REFUTED no-solver, KA-B VERIFIED, KA-C UNKNOWN/VERIFIED/REFUTED-witness).
- Holdout runner (KA-B in isolation, code replicated verbatim from the
  socket): `case1_mime_symmetry.py` → `runs/case1_mime_symmetry.out`.

### (b) Measured behavior (verbatim)

KA-B, interval path only (runner run A; wall 0.028 s): requires UNKNOWN,
statuses `['discharged', 'discharged', 'discharged', 'discharged', 'unknown', 'unknown']`, and

    note: requires face: obligation #4 undecided by interval propagation — the comparison straddles: lhs in [-3.5e-323, 3.5e-323] (elementwise hull over 2 elements) <= rhs in [-3.5e-323, 3.5e-323] (elementwise hull over 2 elements) over the declared envelope; an explicit solver_timeout_ms offers exactly this obligation to solver escalation
    note: requires face: obligation #5 undecided by interval propagation — the comparison straddles: lhs in [-3.5e-323, 3.5e-323] (elementwise hull over 2 elements) <= rhs in [-3.5e-323, 3.5e-323] (elementwise hull over 2 elements) over the declared envelope; an explicit solver_timeout_ms offers exactly this obligation to solver escalation

Same pair in the socket's KA-A (boundary-starved, one product per
off-diagonal instead of five accumulated ones; from the `-O` socket run):

    note: requires face: obligation #4 undecided by interval propagation — the comparison straddles: lhs in [-5e-324, 5e-324] (elementwise hull over 2 elements) <= rhs in [-5e-324, 5e-324] (elementwise hull over 2 elements) over the declared envelope; an explicit solver_timeout_ms offers exactly this obligation to solver escalation

KA-B escalated (runner run B; wall 0.207 s): requires VERIFIED; obligations
#4/#5 `discharged — discharged by solver escalation (QF_NRA): the box with
the negated predicate is unsat per cvc5 (wheel) and z3 (wheel)`. Solver
invocations: **4, all spent on the symmetry pair and nothing else**
(`invocations per assert index: {4: 2, 5: 2}`; #0-#3 interval-decided):

    [0] invoked=True cvc5 1.3.4-modified — QF_NRA portfolio primary on assert #4
    [1] invoked=True z3 5.0.0 — QF_NRA portfolio secondary on assert #4
    [2] invoked=True cvc5 1.3.4-modified — QF_NRA portfolio primary on assert #5
    [3] invoked=True z3 5.0.0 — QF_NRA portfolio secondary on assert #5
    note: assert #4: cvc5 (wheel) answered unsat in 69ms
    note: assert #4: z3 (wheel) answered unsat in 9ms
    note: assert #5: cvc5 (wheel) answered unsat in 75ms
    note: assert #5: z3 (wheel) answered unsat in 8ms

(The `-O` socket run's KA-B shows the same 4-invocation pattern with 69/10/67/8 ms.)

### (c) Correlation structure

`M[..,0,1]` and `M[..,1,0]` are the same sum of per-face products with each
term commuted (`dx*dy` vs `dy*dx`, assembled in the same order) — every input
of one side appears in the other, and their real-arithmetic difference is
identically zero. Interval propagation evaluates the two sides independently;
outward 1-ulp rounding at each mul/add turns even this all-point envelope into
two tiny symmetric hulls around 0 (±5e-324 for one product, ±3.5e-323 after
the five-term fed assembly), which straddle each other, so a `<=` in both
directions is undecidable without a solver.

### (d) Post-build measurement protocol

Run `case1_mime_symmetry.py` from the refined tree. Record: (1) run A
`requires_status` and the six per-obligation statuses; (2) the presence or
absence, and exact text, of the obligation #4/#5 notes (hull endpoints
included); (3) run B's total solver invocation count and the
`invocations per assert index` tally; (4) both wall times. Also run
`python -O` on the socket file and diff its KA-A/KA-B obligation-status lines
against `runs/case1_socket_dashO.out`. Compare each recorded item against the
values quoted in (b); report every difference, in whichever direction.

---

## Case 2 — the dependency-shaped 2x2 conditioning obligation

### (a) Source and how to run

- Probe (hand-work + Z3): `corpus/supply/la_contract_probe.py` → full output
  `runs/case2_la_contract_probe.out`.
- The same shape mechanized: `stelling.contracts.conditioning_2x2("float64",
  (1.0, 2.0), (1.0, 2.0), (-0.5, 0.5), 8.0)` — holdout runner
  `case2_conditioning_dependency.py` → `runs/case2_conditioning_dependency.out`.
- In-suite instance: `tests/test_solver_acceptance.py:59-98`
  (`test_interval_alone_leaves_the_acceptance_obligation_unknown`, then the
  portfolio discharge test) — the identical harness shape; the runner includes
  a contract-free replica of it.

### (b) Measured behavior (verbatim)

Probe Part 2 (Z3 decides the same region the intervals cannot):

       well-shaped region (b in ±0.5): cond <= 8 for ALL: PROVED (unsat)
       sliver-reaching region (b in ±1.4): violated somewhere: sat   witness: a=3/2, b=5/4, c=3/2

Probe Part 3, the interval straddle AND the probe's own non-closure argument
for plain affine (the holdout must test this with non-closure as the
probe-predicted outcome):

    == Part 3: intervals on the same obligation
       tr^2 ∈ [4, 16], det ∈ [0.75, 4.0]
       tr^2/det ∈ [1.00, 21.33]  vs  threshold 10.125
       STRADDLES: intervals cannot close what Z3 just proved — the
       precondition is dependency-shaped (a, c shared between tr and det;
       det itself is Cauchy-Schwarz slack in the underlying d-vectors),
       and quadratic past plain affine. Q1: dependency-shaped. Q2: the
       closure is a small QF_NRA validity — the solver's shape.

`conditioning_2x2`, interval path only (no solver; wall 0.010 s): requires
UNKNOWN, obligations #0-#2 discharged, #3 unknown, with

    note: requires face: obligation #3 undecided by interval propagation — the comparison straddles: lhs in [3.9999999999999987, 16.00000000000001] <= rhs in [7.593749999999996, 43.03125000000003] over the declared envelope; an explicit solver_timeout_ms offers exactly this obligation to solver escalation

Measured discrepancy vs the probe's hand intervals, worth its own line: the
probe hand-computes `det ∈ [0.75, 4.0]` (it treats `b^2 ∈ [0, 0.25]`);
stelling's rhs hull tops at `43.03125 = 4.25 * 10.125` because the traced
`b*b` is interval-multiplied as two independent copies of `b ∈ [-0.5, 0.5]`,
giving `b*b ∈ [-0.25, 0.25]` — a second, self-correlation loss inside the
same obligation, on top of the a,c sharing the probe names.

Supplementary escalated run (wall 0.450 s): requires VERIFIED; obligation #3
`discharged — discharged by solver escalation (QF_NRA): the box with the
negated predicate is unsat per cvc5 (wheel) and z3 (wheel)`;
`note: assert #3: cvc5 (wheel) answered unsat in 68ms`,
`note: assert #3: z3 (wheel) answered unsat in 13ms`.

Raw-harness replica (test-suite shape, plain `propagate`, no contract layer):
`['unknown']`, detail `undecided for 1/1 element(s)`, propagation notes
`(none)` — the lhs/rhs-hull straddle NOTE was emitted by the contract layer
(`check_contract`), not by raw propagation; raw runs record only the
undecided detail line.

*This cited line 1031 of `contracts.py`. `src/stelling/contracts.py` is 1022
lines, so that number was 9 past the end of the file — checked 2026-08-09. Read
`check_contract`'s own docstring before relying on the sentence: it now records
that the contract layer's mirrored note is GONE and that the straddle is quoted
in the obligation's detail by the propagation layer itself. This entry is left
as the dated observation it was, with the citation pointed at a symbol.*

### (c) Correlation structure

`a` and `c` each appear on both sides of the comparison — in `tr = a + c`
(squared, lhs) and in `det = a*c - b*b` (rhs) — and `b` appears twice in
`b*b`. Independent interval evaluation prices `tr` and `det` at uncorrelated
extremes (and prices `b*b` sign-indefinite), so the hulls overlap; the true
statement over the box is a theorem (probe Part 2, unsat).

### (d) Post-build measurement protocol

Run `case2_conditioning_dependency.py` from the refined tree. Record: (1)
obligation #3's status on the no-solver run; (2) the exact text of any
obligation #3 note, hull endpoints included (in particular whether the rhs
upper endpoint is still 43.03125..., i.e. whether `b*b` is still priced
sign-indefinite); (3) the raw-replica statuses; (4) whether the supplementary
run still spends solver invocations on #3 and how many. The probe's Part 3
prediction on file is that plain affine arithmetic does NOT close this
obligation (quadratic past plain affine); record the measured outcome
whichever way it lands, next to that quote.

---

## Case 3 — MIME fvm E_f = |Sf|^2/(Sf.d): polar hand-cancelled form (F1) vs raw vector form (F2)

### (a) Source and how to run

- Corpus harness: `corpus/supply/mime_fvm.py` (F1/F2/F3) → full output
  `runs/case3_mime_fvm.out`.
- Regional variants: `corpus/supply/mime_fvm_regional.py` (R1/R2/R2b) →
  `runs/sweep/mime_fvm_regional.out`.
- Holdout runner (F1/F2 verbatim replicas + an escalation of F2, which the
  corpus file does not do, + an F2b sqrt-free variant):
  `case3_fvm_f1_f2.py` → `runs/case3_fvm_f1_f2.out`.

### (b) Measured behavior (verbatim)

F1 (polar, `a/(dm*c) <= 8.0`, boxes a,dm ∈ [0.5,2], c ∈ [0.71,1]):
`['discharged']` interval-only, `coverage: 22 eqns: 22 known (100%)`; verdict
VERIFIED. Under the corpus file's inputs-only ⊤-widening: `obligations
['unknown']` → "the obligation does NOT survive ⊤ — the declared bounds are
load-bearing".

F2 (raw, components ±2, cos bound as an assume): `['unknown']`, and

    coverage: 30 eqns: 27 known (90%); 2 ⊤ across 1 primitives (sqrt ×2); 1 constraint(s) DROPPED (stelling_assume ×1)
    note: assume constraint DROPPED (inert in MVP propagation) at .../mime_fvm.py:175 (h_f2): VERIFIED proves a superset; UNKNOWN may be confounded by this drop (relational: both sides vary — constraining needs relational domains)

F2 escalated (holdout runner; the answer to "what is the raw form's actual
blocker"): **not sqrt, and not a missing division transfer** — `div` has a
transfer and was used (`div [sound]` in F2's stamp); the sqrt equations feed
only the dropped assume and are outside the obligation's emission cone. The
decline is the div emission guard on the zero-straddling divisor:

    escalation record for assert #0: outcome=unknown
      detail: escalation declined: 'div': divisor may be zero over the declared box — SMT-LIB2 division is underspecified at 0

The divisor's box contains 0 exactly because the relational assume
(`dot >= 0.71*sf_norm*d_norm`) was dropped — dependency loss is the root,
the div guard is the proximate decline.

F2b (|Sf|^2 = Sf.Sf hand-applied so no sqrt exists; cos bound split into
`dot >= 0` plus `dot*dot >= 0.71^2*area2*d2`): `['unknown']`, with the
one-sided assume now CONSTRAINED and the quadratic one still dropped:

    note: assume CONSTRAINED at .../case3_fvm_f1_f2.py:80 (h_f2b): narrowed var 11 to [0.0, 12.000000000000005]
    note: precondition satisfiability UNCERTIFIED at .../case3_fvm_f1_f2.py:80 (h_f2b): var 11 is an over-approximated intermediate (its box may exceed its true image) — the conditional claim may be vacuous
    note: assume constraint DROPPED (inert in MVP propagation) at .../case3_fvm_f1_f2.py:81 (h_f2b): VERIFIED proves a superset; UNKNOWN may be confounded by this drop (relational: both sides vary — constraining needs relational domains)
    escalation record for assert #0: outcome=unknown
      detail: escalation declined: constrained assume present: solver escalation emits over the declared box, which does not respect the assumed precondition — a sat witness could violate the precondition while the verdict claims conditionality; escalation declines until constrained bounds can be emitted faithfully

Regional companions (from `runs/sweep/mime_fvm_regional.out`): R1 (polar,
c widened to [0.11, 1]) — interval unknown, then solver REFUTED, witness
`x0 = 3/2, x1 = 1, x2 = 1/8` replay-confirmed (E_f = 12 > 8). R2 (raw,
inscribed cube, no assume) — unknown; `escalation declined: primitive 'sqrt'
is outside the supported emission set` (there the sqrt IS in the obligation
cone: area is sqrt'd then squared). R2b (hand identity, no assume) — unknown,
`coverage: 19 eqns: 19 known (100%)`, then
`escalation declined: 'div': divisor may be zero over the declared box —
SMT-LIB2 division is underspecified at 0`.

### (c) Correlation structure

The raw form's numerator `area2 = Sf·Sf` and denominator `dot = Sf·d` share
all three `Sf` components, and the property that keeps the denominator away
from zero is exactly the dropped relational assume tying `dot` to the product
of norms. F1 has no repeated variable at all (each declared input occurs once
in `a/(dm*c)`) — which is why the hand-cancelled polar form is
interval-decidable and the raw form is not.

### (d) Hand truth note and post-build measurement protocol

Hand analysis, for honesty about what "decidable" means here: over F2's
declared box (components ±2, no magnitude floor) the claim `area2/dot <= 8`
is FALSE even with the cos assume in force — e.g. Sf = (2,2,2), d = (t,t,t)
with t → 0+ keeps cos = 1 and sends `area2/dot = 4/t → ∞`. So the decidable
outcome on this exact posing is a refutation-with-witness, not a
verification; the regional R1 already measures that shape on the polar form.

Protocol: run `case3_fvm_f1_f2.py` from the refined tree. Record: (1) F2 and
F2b obligation statuses; (2) which of the three note kinds appear (DROPPED /
CONSTRAINED / satisfiability-UNCERTIFIED) and their exact text; (3) the
escalation outcome and its decline reason if any, or the witness values and
replay line if any; (4) F1's status (unchanged posing, single-occurrence
inputs — a control for regressions). Record the same four items for R1/R2/R2b
by re-running `corpus/supply/mime_fvm_regional.py`.

---

## Case 4 — sweep of corpus/supply and the test suite: shared-input straddles vs the rest

Holdout runner: `case4_sweep_hits.py` → `runs/case4_sweep_hits.out`.
Full sweep outputs (all 18 remaining corpus scripts, exit 0 each):
`runs/sweep/*.out`.

### Hits — straddle from shared inputs (correlation loss)

**4a. `corpus/supply/cf_run.py` bjx#969** (`s_new = where(TRUE, s_max*0.8,
s_max)`; `assert s_new < s_max`, s_max ∈ [0.01, 100]). Measured:
`['unknown']`, detail `undecided for 1/1 element(s)` (raw propagate emits no
hull note); the corpus file's own printed relation line:

    relation: does NOT mechanize: the shrink `0.8*s_max < s_max` is DEPENDENCY-SHAPED (s_new derived from s_max; interval loses the correlation, same shape as the dfx#632 exhibit) -> UNKNOWN, an affine-forms case, NOT the search-shaped the prior predicted

Escalation (holdout runner): currently unreachable —

    detail: escalation declined: input declaration of dtype 'bool': only float declarations are supported (an int/bool input's real relaxation would admit non-member witnesses)

Hand argument: `0.8*s < s` over s ≥ 0.01 is linear with `s` on both sides;
decidable by linear/affine reasoning (0.8s − s = −0.2s ≤ −0.002 < 0).

**4b. `tests/test_controlflow.py:56-65` (`test_where_ratchet_is_dependency_shaped`)**
(`new = where(nf, s*0.8, s)`; `assert new <= s`, s ∈ [0.1, 10], nf unknown
bool). The test itself pins the current behavior: "dependency-shaped UNKNOWN,
not a false discharge (recorded finding)". Measured in the runner:
`['unknown']`; escalation declined with the same bool-declaration reason as
4a. Hand argument: both branches satisfy `branch <= s` (0.8s ≤ s and s ≤ s
for s > 0); linear, needs the branch join to keep the `s` correlation.

**4c. `corpus/supply/exhibit_632.py`** (`t + dt > t` at the POINT t = 1.0,
dt = 1e-20; semantics=real). Measured: `['unknown']` with the exhibit's own
mechanism print:

    no-dependency probe: fl(1.0 + 1e-20) brackets to [0.9999999999999999, 1.0000000000000002]
      straddles 1.0: True -> `1.0 + 1e-20 > 1.0` is UNKNOWN, not a false VERIFIED

Escalation (holdout runner): **this one the solver already closes** —
`discharged by solver escalation (QF_LRA): the box with the negated predicate
is unsat per z3 (wheel) and cvc5 (wheel)`; z3 11 ms, cvc5 70 ms. Hand
argument: `t` shared on both sides; `(t + dt) - t = dt > 0` — linear. Same
outward-rounding-at-a-point mechanism as case 1's symmetry pair. (Under IEEE
semantics the same shape is genuinely false —
`tests/test_ieee_semantics.py`'s `_t_dt_query` block, headed "the t + dt > t
acceptance shapes", covers that; the real-semantics claim is the decidable one.
*This cited `test_ieee_semantics.py:867-906`, which is a maybe-NaN
assume-clearing test and nothing to do with this shape; the block meant is
`test_t_plus_dt_point_collapse_is_refuted_under_ieee` and its two neighbours —
measured 2026-08-09.*)

### Non-hits, recorded so the holdout does not overclaim

- **Genuine envelope straddles (interval UNKNOWN is honest; solver refutes
  with a witness):** `lineax_preconditions.py` p10/p16/p17/p20 etc. — ne-0 /
  > 0 obligations over boxes that include the violating endpoint; witnesses
  `x0 = 0` (and for p20 pairs like `x5 = 0, x6 = 1/2`), all replay-confirmed
  QF_LRA/QF_NRA sat in 3-83 ms. Also case 5's controls below.
- **Emission/coverage gaps, not correlation:** `lineax_preconditions.py`
  declines — `primitive 'abs' is outside the supported emission set`,
  `primitive 'dot_general' ...`, `primitive 'cond' ...`, `input declaration
  of dtype 'int64': only float declarations are supported`, `non-finite
  constant TypedFloat(inf, dtype=float64) has no exact Real emission`;
  `second_bill.py` / `pytree_probe.py` ⊤ from `pow` (possibly-negative
  base), `convert_element_type` weak_type, bitwise `or`, `sqrt`, `log`,
  RNG plumbing; `mime_fvm.py` F3's `custom_linear_solve`/`lu`/`dot_general`
  ⊤ with 39 unreached equations.
- **Intentional controls:** `tautology_test.py` ⊤-widening unknowns;
  `e2a_417.py` mutation case ("must NOT verify") lands UNKNOWN by design;
  its clean box-invariance case VERIFIED on intervals.

### Post-build measurement protocol

Run `case4_sweep_hits.py` from the refined tree. Record per posing: the
obligation status, any propagation notes, and the escalation outcome line
(decline reason, or solver answers with ms, or witness + replay). For 4b also
run `pytest tests/test_controlflow.py -q` and record which tests change
status. Diff the three against the verbatim blocks above and report every
change, including newly appearing notes.

---

## Case 5 — the MADDENING corpus harnesses

Holdout runner: `case5_maddening.py` → `runs/case5_maddening.out`. Corpus
outputs: `runs/sweep/maddening_cfl.out`, `runs/sweep/maddening_preconditions.out`.

### 5a — the correlation hit: real HeatNode.update box invariance

Source: `corpus/supply/maddening_cfl.py` harness B (real imported
`maddening.nodes.heat.HeatNode.update`, 20 cells, dt = 0.0625, alpha = 0.01,
dx = 0.05; temperature declared per-cell in [0, 100]; obligations
`new_T <= 100` and `new_T >= 0`, x64 OFF, float32 node).

Measured (corpus file and runner agree):

    assert #0: unknown — undecided for 18/20 element(s)
    assert #1: unknown — undecided for 18/20 element(s)
    coverage: 36 eqns: 36 known (100%)

— every reached equation has a transfer (`(none — every reached equation had
a registered transfer)` in the file's own census), so this UNKNOWN is pure
propagation imprecision, not coverage. Escalation (runner; the corpus file
does not escalate B): currently unreachable —

    detail: escalation declined: primitive 'scatter' is outside the supported emission set

Correlation structure: the real update is
`T_new = T + alpha*dt*laplacian` with
`laplacian_i = (T_{i+1} - 2*T_i + T_{i-1})/dx^2` (heat.py:26-28, 443) — `T_i`
appears in the identity term and again in the stencil, so intervals price
`T_new_i` as `[0,100] + 0.25*[-200,200] = [-50, 150]`, straddling both
bounds. In real arithmetic the update is the convex combination
`0.5*T_i + 0.25*T_{i-1} + 0.25*T_{i+1}` (CFL r = 0.25), which maps [0,100]
into itself — linear, decidable by linear/affine reasoning per element. The
2/20 discharged elements are the boundary cells, overwritten by
`.at[0].set(T_left)` / `.at[-1].set(T_right)` (heat.py:446-447) with
single-occurrence declared inputs — exact under intervals.

### 5b — genuine-straddle controls from the same files (not correlation loss)

- `maddening_cfl.py` h_a2 (CFL over the node's validated alpha regime
  [1e-6, 1.0], dt = 0.0625 point): interval `['unknown']`; the file's
  classifier prints "unknown: undecided over the declared box (the analysis's
  imprecision, not a refutation)". Escalated in the runner: genuinely
  violated — `violated at a concrete witness found by cvc5 (wheel) (QF_NRA)`,
  witness `x2 = 1/16, x3 = 12/25` (CFL = 12 ≥ 0.5), replay-confirmed;
  cvc5 sat 75 ms, z3 sat 14 ms.
- `maddening_preconditions.py` posing B (face coefficients > 0, θ ∈ [-2, 1e2]):
  interval `unknown — undecided for 4/4 element(s)` on both asserts, then
  REFUTED with replay-confirmed witnesses, e.g.
  `x0_0 = 1/32, x0_1 = -1/32, x0_2 = 1/16, x0_3 = 0` (a face average summing
  to exactly 0, violating the strict > 0 on the closed box); QF_LRA, z3 sat
  11/8 ms, cvc5 sat 69/76 ms. The face average `0.5*(θ_i + θ_{i+1})*16` uses
  each θ once per face expression — the interval hull is exact here and the
  straddle is real, not correlation loss.
- `maddening_preconditions.py` posing D (mass ≠ 0 over [0.0, 1.0]): witness
  `x0 = 0`, QF_LRA sat 4 ms — endpoint violation, honest straddle.
- Posings A and C: interval-VERIFIED (no solver), recorded as the
  interval-sufficient baseline in the same file.

### Post-build measurement protocol

Run `case5_maddening.py` from the refined tree. For 5a record: the per-assert
undecided element counts (n/20), the coverage line, any new propagation
notes, and the escalation outcome line. For 5b record: status and the witness
values + replay line (a control — its measured outcome documents that the
refinement does not convert genuine violations into verifications). Then run
`corpus/supply/maddening_cfl.py` and `corpus/supply/maddening_preconditions.py`
unmodified and diff their stdout against `runs/sweep/`.

---

## Honest summary

By hand analysis of the measured cases: the correlation-loss-shaped ones —
where the true claim over the declared set is decidable by linear/affine
reasoning because the straddle exists only in the independent re-pricing of a
shared input — are case 1 (the KA-B symmetry pair: both sides are the same
commuted sum, difference identically 0, the ±3.5e-323/±5e-324 hulls are pure
outward rounding), case 4a and 4b (`0.8*s < s` and its select-join ratchet:
linear, one shared variable, currently unreachable by the solver too because
of the bool declaration), case 4c (`t + dt > t` at a point: linear, shared t,
today closed only by spending a QF_LRA portfolio call), and case 5a (the real
heat stencil's box invariance: a convex combination per element, 18/20
elements undecided, solver-unreachable today via the scatter emission gap).
Case 2 is dependency-shaped but genuinely QUADRATIC — a, c shared between
tr^2 and det plus a sign-indefinite `b*b` self-product — and the probe's own
on-file argument says plain affine cannot close it (the holdout should treat
non-closure there as the probe-predicted outcome and record whatever
happens). The genuinely-not-correlation cases, recorded so a refinement is
not graded on them: case 3's raw form F2/F2b (the UNKNOWN is confounded by
the dropped/inert relational assume and the div/constrained-assume emission
guards, but the posed claim is actually false over its floor-less box —
refutation-shaped, like the measured R1 witness), and the honest envelope
straddles (maddening B/D, cfl A2, the lineax ne-0 family) plus the
emission/coverage-gap unknowns (sqrt/abs/dot_general/cond/scatter/bool/int64
declines, F3's custom_linear_solve ⊤), which are solver-surface work, not
domain-precision work.
