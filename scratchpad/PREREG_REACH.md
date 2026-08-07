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
