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

Scored at `13f5b2f`. Every figure carries the command that produced it;
`<W>` = `/tmp/claude-1000/-home-nick-MSF-stelling/e443f4ad-79d3-43d4-bc29-87a910409ae6/scratchpad`,
`<PY>` = `/home/nick/venvs/stelling-jax/bin/python`, and every run exported
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<W>/wt-ref1/src` after
`<PY> -c "import stelling; print(stelling.__file__)"` resolved inside the
worktree.

**C1 — MET, with one correction to its own wording.** Three mechanisms,
each independently reachable, attributed by
`<PY> <W>/work/mech2.py <W>/work/base.json`:

```
  64  A dropped conjunct inside a CONSTRAINED assume
   6  B affine re-mint (interval had withheld)
  60  C forward-scoping (assume traced after the assert)
```

The correction: "three defects" is not how they divide. HANDOFF5 §15.1's
#1 and #2 are both A (see C3); its #3 is B; and C is not in that list at
all. **Two are bugs and one is a semantics choice.**

**C2 — MET.** `truth of every wrong REFUTED: Counter({'VACUOUS': 130})`
across 2268 rows, and `('NO_CE', 'REFUTED')` does not occur in either
corpus at any commit measured. Emptiness is the necessary condition for a
wrong REFUTED here, not a mechanism: a superset that contains a *non-empty*
assumed region cannot yield a wrong set-level refutation, because a
predicate definitely false over the superset is false at every point of
the region too.

**C3 — MET.** `NARROW2+ALL_LE_02` (jointly empty; neither conjunct false
alone) and `RED+EMPTY_ALL` (a definitely-false conjunct) are both
mechanism A, both REFUTED at `9efea6f`, both UNKNOWN at `8106a55`, closed
by one edit at one site.
`tests/test_vacuous_refutation.py::test_a_jointly_empty_region_moves_WITH
_the_definitely_false_conjunct`.

**C4 — MET.** 0 mechanism-A rows remain
(`<PY> <W>/work/mech2.py <W>/work/fixAB.json` prints only the 60 C rows).

**C5 — MET.** 0 mechanism-B rows remain, at `refine="affine"`, with and
without `solver_timeout_ms=4000`
(`<PY> <W>/work/rivals.py`, the REPRODUCER block: all eight
timeout × refine combinations UNKNOWN).

**C6 — MET.** Over both corpora, 2844 rows: **every** move is
`REFUTED -> UNKNOWN` (128 wrong + 118 right). VERIFIED totals unchanged
(580 -> 580 and 192 -> 192); WRONG-VERIFIED 0 -> 0.
`<PY> <W>/work/ledger.py`.

**C7 — MET.** The 60 mechanism-C rows are byte-identical before and after;
`h_pre` is REFUTED at `9efea6f` and REFUTED at `13f5b2f`
(`<PY> <W>/work/probe_branches.py`). Pinned as current behaviour by
`test_an_assume_after_the_assert_is_the_reserved_ordering_question`.

**C8 — MET.** `grep -n "self.uncertified = True" src/stelling/propagate.py`
returns three sites: 4967 (inside `if narrowed:`, so
`coverage.constrained >= 1`), 5035 (mixed drop, sets `assume_dropped`),
5089 (whole drop, sets `assume_dropped`). No fourth, and none on a path
where both proxies are false.

**C9 — MET.**

| tree | jax | result |
|---|---|---|
| `9efea6f` | 0.11.0 | 2149 passed, 2 skipped |
| `13f5b2f` | 0.11.0 | see the table in the report |
| `13f5b2f` | 0.10.2 | 2168 passed, 2 skipped |

`--collect-only -q` id diff `9efea6f` → branch: **+19, −0**, and the
branch's id set is **identical between 0.11.0 and 0.10.2** (`diff`
returns nothing).

**C10 — MET.** The ledger is in the report and in `SOUNDNESS.md`, stated
as a class with its sample size (2844 rows), including the 118 legitimate
REFUTEDs the fix costs.

**C11 — MET.** Cost measured first, without the discriminant: 60
legitimate REFUTEDs lost on the relational corpus
(`<W>/work/rel_fixA.json`), of which the `DISJOINT`+`REL` class is a drop
whose own box is `[1,1]`, measured at the drop site by
`<PY> <W>/work/boxes.py`. With the discriminant that cost is **40**, and
wrong REFUTEDs closed stays 48 — **20 restored, 0 unsound restorations**
(`<W>/work/rel_fixA2.json`). The discriminant restores nothing on any
`VACUOUS` row. Two-sided pin:
`test_a_definitely_true_dropped_conjunct_still_refutes` /
`test_an_indeterminate_dropped_conjunct_does_NOT`.

**The hard boundary — HELD.** Neither reach-back pass was implemented, and
no verdict was moved to VERIFIED. Every verdict this branch moves goes
REFUTED → UNKNOWN.

## A measured correction to HANDOFF5 §26.3

§26.3 describes AFF4 and F3E as having "**opposite semantics postures**"
on the reach-back pass. Measured on the ordering row `h_pre` (assert
before a wholly-dropped assume over an EMPTY region), by running all four
trees (`<PY> <W>/work/probe_branches.py`):

| tree | A (mixed drop) | B (affine re-mint) | C (ordering) |
|---|---|---|---|
| `9efea6f` main | REFUTED | REFUTED | REFUTED |
| AFF4 `20598ce` | **REFUTED** | UNKNOWN | UNKNOWN |
| F3E `7dbb25f` | UNKNOWN | **REFUTED** | UNKNOWN |
| this branch | UNKNOWN | UNKNOWN | **REFUTED** |

The postures differ **in prose only**. AFF4's `rescope_preconditions`
docstring says the question is "RESERVED FOR THE PRINCIPAL" and then takes
the set-scoped direction anyway; F3E's `withhold_possibly_vacuous` calls
the alternative "not a defensible split" and takes the same direction.
**On the row that distinguishes them, both branches behave identically.**
So the merge trap §26.3 warns about is not two semantics fighting — it is
two implementations of the *same* unratified decision, which is why their
union is silent. This branch implements neither, which is why it has no
trap: the only rows on which it differs from both is exactly the row the
principal has not ruled on.

The residual disjointness §26.1 reports **reproduces**: AFF4 leaves A
standing, F3E leaves B standing, and neither alone is sufficient.

## C9, re-scored at the final tip `4d72f2b` (append-only correction)

The C9 table above was scored at `13f5b2f`, three commits before the tip.
Re-run at the tip, **sequentially, one lane at a time on a quiet machine**
(the load average is printed with each, because a loaded run is what made
the first attempt unreadable):

```
tip: 4d72f2b
=== stelling-jax  (jax 0.11.0)   load 1.05 ===  2169 passed, 2 skipped  (162.08s)
=== stelling-jax010 (jax 0.10.2) load 3.80 ===  2169 passed, 2 skipped  (160.32s)
main 9efea6f (jax 0.11.0)        load ~1    ==  2149 passed, 2 skipped  (149.30s)
```

Both skips are the same two on every tree and both lanes
(`blackjax` absent). `--collect-only -q`: **2151 ids at `9efea6f`, 2171 at
the tip, +20 / −0**, and the tip's id set is byte-identical between 0.11.0
and 0.10.2.

**One thing I got wrong twice, recorded because it cost the most time
here.** `docs/supported-primitives.md` cites `propagate.py:LINE` and is
generated. I regenerated it after the first `src/` commit and then added
lines above the cited one in a later commit without rerunning the
generator, so `test_committed_page_matches_live_registries` failed in both
lanes. I first read that failure as the load-dependent solver flake
HANDOFF5 §13.2 records — because the run was at load 7 and took 326s
against a 150s nominal — and it was nothing of the kind: it was
deterministic, and `-rs` was hiding the `FAILED` line that would have said
so (`-rs` requests skip reasons ONLY; it suppresses the failure summary
that `-ra` would print). **A slow run at high load is not evidence for a
flake**, and a filtered summary is not a summary. Regenerate after every
`propagate.py` edit, and read failures with `-ra`.
