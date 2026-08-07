<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG_REACH — two defects in sub-jaxpr handling

Branch `fix/branch-reachability`, base `c4133f8` (`main`), worktree
`/home/nick/MSF/.wt-reach/w1`. Written **before** the first `src/` edit.
Outcomes are appended below the rule; nothing above it is edited afterwards.

Every command below runs with
`JAX_PLATFORMS=cpu PYTHONPATH=/home/nick/MSF/.wt-reach/w1/src` and
`/home/nick/venvs/stelling-jax/bin/python` (jax 0.11.0) or
`/home/nick/venvs/stelling-jax010/bin/python` (jax 0.10.2), with
`python -c "import stelling; print(stelling.__file__)"` verified before every
comparison. `jax_enable_x64` is set inside the corpus runner.

## The two defects, as reproduced on `c4133f8`

**D1 — a REFUTED from a branch nothing certifies is reachable.**

```
scratchpad/repro.py:  jax.lax.cond(x[0] - x[0] > 0.0,
                                   lambda v: assert_(v > 5.0),
                                   lambda v: assert_(v > -9.0), x)
  ->  status: REFUTED   obl 1 violated-over-set
      oracle: the 'yes' obligation is EXECUTED at 0 of 206 sampled points
```

**D2 — an obligation inside a `scan`/`while` body is silently dropped, and
it can ride under a VERIFIED.**

```
scratchpad/repro2.py: the scan body's `stelling_assert` IS in the IR
      (all prims: stelling_any, scan, gt, stelling_assert)
      propagate(): obligations = 0, notes = ()
scratchpad/repro3.py: one true top-level assert + one false assert inside a
      scan  ->  status: VERIFIED, obligations = [(0, 'discharged')]
```

D2 is therefore not only "UNKNOWN read as undecided": it is a **false
VERIFIED**, measured.

## Instrument

`scratchpad/reach/` — `gen_cases.py` emits `cases.py`, **736 harnesses with
one `S(...)` obligation per source LINE**; `oracle.py` is a numpy oracle that
never imports stelling; `run.py` joins them; `analyze.py` classifies.

Corpus axes: 13 guard families (`lit_true`, `lit_false`, `iv_true`,
`iv_false`, `sat`, `reg_sat`, `narrow_sat`, `hidden_f`, `hidden_t`,
`top_sat`, `top_hid_f`, and three `switch` index families) x obligation in
{none, true, false, mixed} in the *yes* branch x the same four in the *no*
branch x a **top-level** obligation in {none, true, false, mixed}; plus
`switch` (3 legs), `scan`, `while`, `cond`-in-`cond`, `scan`-in-`cond`,
`cond`-in-`scan`, and a `jit` control. Each case runs on three legs:
`refine=None`, `refine="affine"`, `vacuity_mode="inputs-only"`.

**Designed against the four blind spots the last three corpora hit:**

1. *unfalsifiable filler claim in the sibling branch* — a branch with no
   obligation carries a plain comparison (`v > 3.0`), never an assert, so it
   states nothing; and the sibling's real obligation is varied over
   {none, true, false, mixed} so the sibling slot discriminates.
2. *no top-level obligation at all* — every `cond` case carries a top-level
   obligation slot varied over {none, true, false, mixed}; the `t_None`
   column is the deliberate minority, not the default.
3. *an oracle that scored only points taking the branch* — the oracle
   executes real control flow (plain Python `if`/loops over numpy) and
   records, per obligation, `n_exec` (points at which it was EVALUATED) as
   well as `n_false`. `n_exec == 0` is what "unreachable" means here, and it
   is a measurement over the whole declared box, not over the branch.
4. *scoring per query rather than per obligation* — the primary ledger is
   PER OBLIGATION (4389 rows); the per-query roll-up (2208 rows) is reported
   alongside, never instead.

## Baseline, measured on `c4133f8` before any `src/` edit

```
scratchpad/reach/run.py evidence_BASELINE.json 300 ; analyze.py
PER-OBLIGATION (4389 rows)          PER-QUERY (2208 rows)
  DISCHARGE_SOUND         1338        REFUTED_SOUND     1113
  DISCHARGE_VACUOUS        147        VERIFIED_SOUND     303
  REFUTE_ON_UNREACHABLE    183  <-D1  UNKNOWN            723
  REFUTE_SOUND            1287        FALSE_REFUTED       60  <-D1
  SWALLOWED_FALSE           24  <-D2  FALSE_VERIFIED       9  <-D2
  SWALLOWED_TRUE            18  <-D2
  UNKNOWN_ON_FALSE        1248
  UNKNOWN_ON_UNREACHABLE   144
```

Of the 1287 REFUTE_SOUND rows: 441 are top-level obligations, ~207 sit in a
branch the index interval FORCES, and ~639 sit in a branch the index interval
only ADMITS. That last number is the whole reason a bare "withhold in every
branch" rule is not the fix: it would cost half of all sound refutations.

Suite baseline: `--collect-only` = **2273 collected** (jax 0.11.0),
`/home/nick/MSF/.wt-reach/collect_BASELINE_jax011.txt`.

## Registered claims and their falsifiers

**C1 — the general rule for D1 is reachability, not `x - x`.** A definite
violation found while executing a `cond`/`switch` branch that interval
propagation only ADMITS (the index box does not force that branch) is
withheld from REFUTED and reported `unknown` with the reason named — unless
the branch's reachability is certified.
*Falsifier:* any corpus obligation row with oracle `n_exec == 0` that still
reports `violated-over-set` after the fix, on any leg. The falsifier ranges
over all 13 guard families, `switch`, and the nested families — not the
`x - x` shape alone; a fix that special-cased cancellation would leave
`top_hid_f` (guard through an unmodelled `sin`) and `sw_hid` refuting and
would be caught here.

**C2 — the certificate is a point WITNESS inside the declared box, and it is
sufficient, never necessary.** A branch is certified reachable when either
(a) the index box forces it, or (b) some pinned-to-a-point re-propagation of
the same query — every declaration collapsed to a single point of its own
declared box — reaches that branch under a fully forced path.
*Falsifier (soundness):* a row certified by (b) whose oracle `n_exec` is 0.
*Falsifier (power):* a drop, relative to baseline, in the count of
REFUTE_SOUND rows in the `sat`, `reg_sat`, `hidden_f`/`no`, `hidden_t`/`yes`,
`sw_sat`, `sw_hid`, `n_cc_sat_sat` buckets — the branch-scoped refutations
whose branch a witness can reach.
*Positive control:* the count of REFUTE_SOUND rows sitting in a branch that
is NOT forced must be **> 0 after the fix**. Zero there would mean the rule
degenerated to "never refute inside a branch" and the clause would be inert.

**C3 — vacuous truth stays reserved.** No obligation moves from
`violated-over-set` to `discharged`, and no query moves from REFUTED to
VERIFIED.
*Falsifier:* any such transition in the paired baseline/after ledger.

**C4 — D2's mechanism, established before it is fixed.** The `scan`/`while`
body IS transcribed (the `stelling_assert` equation is present in the IR) and
propagation never descends: the obligation is neither collected nor named.
*Falsifier:* an IR walk finding no `stelling_assert` under the `scan`
sub-jaxpr, or a baseline note/detail naming it. (Measured above: it is in the
IR; `notes = ()`.)

**C5 — D2's fix is a NAMED, COLLECTED obligation, not silence and not a
whole-query DECLINE.** Every `stelling_assert` inside a sub-jaxpr that
propagation does not descend into is recorded as an obligation with status
`unknown`, whose detail says it was never examined, plus a note naming its
source location and the primitive that swallowed it.
*Falsifier:* a corpus case whose oracle sees an obligation at line L executed,
where the after-run lists no obligation at line L; or a swallowed obligation
whose notes name neither the location nor the swallowing primitive.

**C6 — the direction that matters: no VERIFIED over a violating point.**
After the fix, no corpus query is VERIFIED while the oracle finds a point of
the declared box at which some obligation is executed and false.
*Falsifier:* any FALSE_VERIFIED row in the after ledger.
*Positive control:* the baseline has **9** FALSE_VERIFIED rows, so an after
count of 0 is a measured move, not a vacuous zero.

**C7 — the cost ledger: every verdict that moves is unsound at baseline or
honestly withheld.** Every query whose status changes moves either from an
unsound REFUTED/VERIFIED to UNKNOWN, or from VERIFIED to UNKNOWN because an
obligation was never examined. No query moves INTO VERIFIED; no query moves
from UNKNOWN into REFUTED.
*Falsifier:* any moved row that is not in one of those classes.
*Positive control:* the ledger must be non-empty — baseline already contains
60 FALSE_REFUTED and 9 FALSE_VERIFIED query rows, so "nothing moved" would
falsify the fix rather than confirm it.

**C8 — the neighbours that must not move.**
(a) an obligation inside a `jit` (transparent) wrapper is descended today and
keeps its exact status (`j_jit_top`);
(b) a top-level obligation's status never changes;
(c) an obligation inside a FORCED branch (`lit_true`/`lit_false`/`iv_true`/
`iv_false`/`sw_df1`-leg-0) keeps its exact status.
*Falsifier:* any status change in those three buckets, on any leg.

**C9 — cost clause, with its positive control.** The reachability certificate
runs **zero** extra propagations on a query that has no `violated-over-set`
obligation inside a non-forced branch.
*Falsifier:* a counted `_Propagator` construction above 1 on such a query.
*Positive control:* a query that DOES have such an obligation must show a
counted construction count **> 1**; without that, the zero is unfalsifiable.
Measured by monkeypatching `_Propagator.__init__` in a measurement script —
not in `src/`.

**C10 — both series stay green.** `pytest -q -ra` on jax 0.11.0 and jax
0.10.2, and a `--collect-only` id diff against baseline whose only entries
are the tests this branch adds.
*Falsifier:* any failure, or a removed/renamed collected id.

**C11 — the honest cost is named, not hidden.** Guards routed through a
primitive with no registered transfer (`sin`, here) cannot be evaluated at a
probe point either, so their branches are never certified and their
branch-scoped refutations are withheld. This is predicted BEFORE the fix runs:
the `top_sat` (96 rows) and `top_hid_f`/`no` (48 rows) REFUTE_SOUND buckets
are expected to move to UNKNOWN, and `narrow_sat` (a satisfiable guard whose
true window is 1% of the box) is expected to move too because a finite
witness grid can miss it.
*Falsifier:* any of these buckets moving in a direction other than
REFUTED -> UNKNOWN; or a bucket NOT predicted here moving.

---

## OUTCOMES (appended after the fact; nothing above this rule is edited)

All numbers below come from
`scratchpad/reach/run.py <out.json> 300` then `analyze.py` / `ledger.py`,
run under `JAX_PLATFORMS=cpu PYTHONPATH=<worktree>/src` with
`/home/nick/venvs/stelling-jax/bin/python` (`stelling.__file__` verified
`/home/nick/MSF/.wt-reach/w1/src/stelling/__init__.py`). Raw evidence is
kept outside the worktree at `/home/nick/MSF/.wt-reach/`
(`evidence_BASELINE.json`, `evidence_AFTER.json`, `LEDGER.txt`,
`DEEP_BASELINE.txt`, `DEEP_AFTER.txt`, the pytest and `--collect-only`
transcripts).

### After, measured on `148d3cb`

```
PER-OBLIGATION (4389 rows)          PER-QUERY (2208 rows)
  DISCHARGE_SOUND         1338        REFUTED_SOUND      921
  DISCHARGE_VACUOUS        147        VERIFIED_SOUND     291
  REFUTE_ON_UNREACHABLE      0        UNKNOWN            996
  REFUTE_SOUND            1071        FALSE_REFUTED        0
  SWALLOWED_FALSE            0        FALSE_VERIFIED       0
  SWALLOWED_TRUE             0
  UNKNOWN_ON_FALSE        1488
  UNKNOWN_ON_TRUE           18
  UNKNOWN_ON_UNREACHABLE   330
```

### Cost ledger (paired, per obligation then per query)

```
  183  violated-over-set -> unknown   [REFUTE_ON_UNREACHABLE]  UNSOUND, removed
  216  violated-over-set -> unknown   [REFUTE_SOUND]           honestly withheld
   24  <absent>          -> unknown   [SWALLOWED_FALSE]        now examined-and-named
   18  <absent>          -> unknown   [SWALLOWED_TRUE]         now examined-and-named
    3  <absent>          -> unknown   [scan inside an unreachable cond branch]
  violated-over-set -> discharged : 0      unknown -> discharged      : 0
  discharged -> anything          : 0      unknown -> violated        : 0
  queries INTO VERIFIED           : 0      queries UNKNOWN -> REFUTED : 0
   60 FALSE_REFUTED -> UNKNOWN     9 FALSE_VERIFIED -> UNKNOWN
  192 REFUTED_SOUND -> UNKNOWN    12 VERIFIED_SOUND -> UNKNOWN
```

### Claim by claim

**C1 — MET.** `REFUTE_ON_UNREACHABLE` 183 -> **0**, on every leg and in
every family: `hidden_f`(48) `hidden_t`(48) `top_hid_f`(48) three
`cond`-in-`cond` shapes (9) and three `switch` families (30). The
falsifier ranged over guards with no `x - x` in them at all
(`top_hid_f` = `sin(x*0) > 0`, `sw_hid` = `int32(floor(c-c))`), so a
cancellation-special-case fix would have failed it.

**C2 — MET, both halves.** Soundness: zero rows certified by the witness
have oracle `n_exec == 0` (there are no `REFUTE_*` unsound rows left at
all). Power: `sat`, `reg_sat`, `hidden_f`/`no`, `hidden_t`/`yes`,
`n_cc_sat_sat` did not move — 0 rows lost in those buckets. Positive
control: **534** branch-scoped REFUTE_SOUND rows survive (baseline 750),
against 537 top-level ones (baseline 537, unmoved), so the rule did not
degenerate into "never refute inside a branch".

**C3 — MET.** 0 obligations moved `violated-over-set -> discharged`; 0
queries moved into VERIFIED.

**C4 — MET, established before the fix.** `scratchpad/repro2.py` on
`c4133f8`: the scan body's `stelling_assert` is in the IR
(`all prims: stelling_any, scan, gt, stelling_assert`), `obligations = 0`,
`notes = ()`. Pinned by
`test_the_scan_body_assert_is_in_the_ir_and_used_to_vanish`.

**C5 — MET.** All 42 swallowed rows are now listed with status `unknown`,
detail beginning `NOT EXAMINED`, and a note naming both the source
location and the primitive (`'scan'` / `'while'`).

**C6 — MET, with its positive control.** `FALSE_VERIFIED` 9 -> **0**. The
deep oracle (>= 20 180 points per case, `deep_verified.py`) finds, on
`c4133f8`, 3 VERIFIED cases with a violating point and 20 REFUTED cases
with none; on the branch, **0 and 0**. The instrument can see both
failures, so the zeros are measured.

**C7 — MET, with its positive control.** Every moved query is
FALSE_REFUTED/FALSE_VERIFIED -> UNKNOWN (69) or SOUND -> UNKNOWN (204).
0 into VERIFIED, 0 from UNKNOWN into REFUTED. The ledger is non-empty.

**C8 — MET.** (a) `j_jit_top` unmoved on all three legs. (b) top-level
REFUTE_SOUND 537 -> 537, DISCHARGE_SOUND 1338 -> 1338 — no top-level
obligation changed status anywhere in the corpus. (c) `lit_true`,
`lit_false`, `iv_true`, `iv_false` branch obligations: 0 moved.

**C9 — MET, with its positive control.** `_count_propagators` = 1 with no
candidate and 17 with one (`test_the_witness_search_is_not_run_without_a_candidate`
/ `..._does_run_when_there_is_a_candidate`). Wall: 0.051 ms/propagate
with no candidate, 2.296 ms with one, 200 iterations each, load average
3.77 on 24 cores.

**C10 — MET.** jax 0.11.0: **2293 passed, 2 skipped** (baseline 2271/2
plus the 22 tests this branch adds). jax 0.10.2: **2293 passed, 2
skipped**. `--collect-only`: 2273 -> 2295 on both series, id diff is
exactly the 22 added `tests/test_branch_reachability.py` ids and nothing
removed; the two series collect identical id sets.

**C11 — MISSED as written; its CLASS held.** The class prediction was
right — every one of the 216 honestly-withheld rows has a guard the
analysis cannot evaluate at a point — but the enumeration of buckets was
not exhaustive. Predicted and observed: `top_sat` (96), `top_hid_f`/`no`
(48), `narrow_sat` (48). **Not predicted and observed: 24 `switch`
rows.** The cause is the same as `sin` and I did not look for it:
`jax.lax.switch` clamps its index through `clamp`, which has no
registered transfer, so **no `switch` branch can be forced or witnessed
and none can refute**. Recorded as a measured capability cost with a
named remedy, and pinned by
`test_no_switch_branch_can_refute_while_clamp_is_unregistered` so
registering a `clamp` transfer flips it.

### One defect found while fixing, not in the pre-registration

The reachability certificate was first keyed on the assert's **outvar
id**. Measured afterwards: jax caches traced branch jaxprs, so
`lax.cond(p, f, f, x)` gives both branches the same body and the
transcriber gives both asserts the same outvar id (10 and 10) — the
witness for the branch that IS taken then certified the occurrence in the
branch that never is, and both refuted. The key is now the chain of
`(cond, branch)` choices plus the equation, and the construction is
`test_a_shared_branch_body_is_certified_per_occurrence`. Corpus effect:
none (no corpus case shares a branch body), so the unit test is the only
thing standing between this hole and a future reader.

### One adjacent instance closed, also not pre-registered

A `nonvacuity` condition inside an unreachable branch still stamped
`FAILED — a membership condition is definitely false: the stated point is
NOT in the declared set (harness defect)`. Same class, same construction;
withheld on the same rule and by the same witness
(`test_the_nonvacuity_failed_face_is_withheld_on_the_same_rule`). The
top-level and witnessed FAILED faces are unchanged.
