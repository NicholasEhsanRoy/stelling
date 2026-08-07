<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG_REF1 — the wrong-REFUTED class under assumes

Branch `fix/vacuous-refutation-guards`, base `9efea6f` (`main`).
Written **before** the first `src/` edit. Outcomes are appended below the
rule; nothing above it is edited afterwards.

## Instrument

`scratchpad/work/corpus.py` (not committed; reproduced in
`tests/test_vacuous_refutation.py` for the shapes that matter): 1134
harnesses = 9 assert shapes x 3 assume-composition styles x 42 conjunct
sets, each driven at `refine=None` and `refine="affine"` = **2268 rows**.

Ground truth per row comes from a **numpy** oracle that never calls
stelling: 50 000 uniform samples in the declared box, plus all 8 corners,
plus a 21^3 grid. A row is

* `VACUOUS` — no sampled point satisfies the assume (implication holds
  vacuously; any REFUTED is wrong),
* `CE` — some point satisfies the assume and violates the assert (REFUTED
  is right, VERIFIED is wrong),
* `NO_CE` — assume satisfiable, no violating point found.

Every command in this file is run with
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<worktree>/src` and
`/home/nick/venvs/stelling-jax/bin/python`, with
`python -c "import stelling; print(stelling.__file__)"` verified first.

## Baseline, measured on `9efea6f` before any edit

```
rows 2268   CONSERVATIVE 1092   CORRECT 830   RAISED 216   WRONG-REFUTED 130
WRONG-VERIFIED 0
```

## Registered claims and their falsifiers

**C1 — three mechanisms, not "three defects".** Exactly three
independently reachable code paths on `main` emit REFUTED over an empty
assumed region:

* **A** — `propagate._assume_constrain`, the `if narrowed:` branch: a
  DROPPED conjunct inside an assume that also narrowed sets neither
  `uncertified` nor `assume_dropped` (64 rows).
* **B** — `affine.refine_propagation`: it declines on
  `coverage.constrained` but consults neither `assume_dropped` nor the
  interval leg's withhold, so it re-mints an obligation the interval leg
  withheld (6 rows).
* **C** — an `assume` traced *after* the `assert_` it should constrain
  (60 rows).

*Falsifier*: a corpus row classed WRONG-REFUTED that is attributable to
none of A/B/C; or a fourth path found by any later run of the same
instrument.

**C2 — emptiness is the necessary condition, not a mechanism.** Every
WRONG-REFUTED row in the corpus has oracle `n_sat == 0`. *Falsifier*: any
row with `status == "REFUTED"` and `n_sat > 0` and `n_ce == 0`.

**C3 — HANDOFF5 §15.1's defects #1 and #2 are ONE defect.** "Empty
assumed region" and "definitely-FALSE dropped conjunct" are not separate
code paths: a single edit at one site closes both. *Falsifier*: after the
edit, `NARROW2+ALL_LE_02` (jointly empty, **no** conjunct individually
false) and `RED+EMPTY_ALL` (a definitely-false conjunct) do not move
together — one still REFUTEs while the other does not.

**C4 — A is closed.** After the fix, zero WRONG-REFUTED rows attributable
to mechanism A. *Falsifier*: any remaining.

**C5 — B is closed.** After the fix, zero WRONG-REFUTED rows attributable
to mechanism B, at `refine="affine"`, with and without a solver timeout.
*Falsifier*: any remaining.

**C6 — nothing moves the unsound way.** The fix creates no WRONG-VERIFIED
and no new REFUTED anywhere in the corpus: the count of rows with
`status == "REFUTED"` is monotonically non-increasing, and
`WRONG-VERIFIED == 0` after. *Falsifier*: either count rising.

**C7 — C is NOT closed by A+B and is a semantics choice, not a bug.**
The 60 mechanism-C rows survive the fix unchanged. *Falsifier*: they
change without my editing the ordering semantics — which would mean C was
never independent.

**C8 — the proxy is total.** `propagation.assume_dropped or
propagation.coverage.constrained` is true at every site that sets
`_Propagator.uncertified`, so keying the affine guard on that pair
withholds wherever the interval leg withheld. *Falsifier*: a
`self.uncertified = True` assignment reachable on a path that sets
neither, exhibited by reading every assignment in the file (`grep -n
"self.uncertified = True"`) and by a corpus row with an interval withhold
and both proxies false.

**C9 — both jax series stay green with identical collected ids.**
`pytest --collect-only -q` id sets are equal between `9efea6f` and the
branch except for ids in files this branch adds; both venvs
(`stelling-jax` 0.11.0, `stelling-jax010` 0.10.2) pass. *Falsifier*: a
non-additive id diff, or a failure in either lane.

**C10 — the ledger is reported as a class with its sample size.** Every
verdict that moves is reported by direction and count over the full 2268
rows, with the command. Correct REFUTEDs lost to the fix are reported as
a cost, not omitted. *Falsifier*: a reported cost that is the length of a
hand-built list rather than a corpus count.

**C11 — the certainly-true discriminant is adopted only if measured to
pay.** If a dropped conjunct's own box is definitely TRUE over the boxes
in force, the drop removed nothing and REFUTED stays sound. I will
measure the cost of the fix WITHOUT it first, and adopt it only if that
cost is both non-zero and attributable to provably-harmless drops.
*Falsifier*: the discriminant landing without a measurement of the cost
it removes; or the discriminant restoring a REFUTED on a `VACUOUS` row.

## Hard boundary I am registering in advance

Mechanism **C** (ordering) and any move of a wrong REFUTED to **VERIFIED**
(vacuous truth) are **semantics choices reserved to the principal**. I
will implement neither, and will report the options and costs instead.
Withholding (UNKNOWN) is the only direction I will move a verdict in.

---

# OUTCOMES (append-only)
