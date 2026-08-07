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

---

# RE-SCORING at `ab4b9b5` (append-only; a later, blinded pass)

Written by a different agent, working from a blinded audit of the branch
and from its own measurements. Nothing above the outcomes rule is edited.
Same protocol as above: `<W>` =
`/tmp/claude-1000/-home-nick-MSF-stelling/e443f4ad-79d3-43d4-bc29-87a910409ae6/scratchpad`,
`<PY>` = `/home/nick/venvs/stelling-jax/bin/python`, every run exported
`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=<W>/P1_tip/src` after
`<PY> -c "import stelling; print(stelling.__file__)"` resolved inside the
worktree.

**C1 — FALSIFIED. There is a fourth path, and it was live on `main`.**

An assume whose only content is a **branch-scoped unsatisfiable** conjunct
(mechanism **D**). Inside a possibly-untaken `lax.cond` branch,
`_unsatisfiable` degrades rather than raising (audit F2, correctly), so it
appends to `vacuous` and NOT to `dropped`; the whole-drop guard read
`if dropped or not vacuous:`, which is FALSE in exactly that case, and the
note and the two flags were gated together. Reproduced at **both** commits
(`<PY> <W>/repro_p1.py`, `<PY> <W>/repro_p1b.py`), x in [-1,1]^3:

```
                          9efea6f     3afbf01
branch/assume+assert      REFUTED     REFUTED    witnesses=(), assume_dropped=False
branch/assert only        REFUTED     REFUTED    <- the assume changed nothing
top level, same assume    RAISE       RAISE      UnsatisfiableAssumptionError
```

All three `_unsatisfiable` detection sites reach it: empty meet
(`propagate.py:5404` at 3afbf01), strict-boundary collapse (`:5432`),
definitely-false constant comparison (`:5302`). Oracle over 59 269 points
(50 000 uniform + 8 corners + a 21^3 grid): 29 337 take the branch, **0**
of them satisfy the assume — the region is EMPTY and the implication is
vacuously TRUE.

**The falsifier was narrower than the claim it was attached to.** C1 claims
"exactly three independently reachable code paths **on `main`**"; its
falsifier admits "a fourth path found by any later run of **the same
instrument**". The instrument is `scratchpad/work/corpus.py`, whose axes
are 9 assert shapes x 3 assume-composition styles x 42 conjunct sets —
no control-flow axis. D is gated on `_Propagator.branch_depth`, and
`grep -n branch_depth src/stelling/propagate.py` returns exactly one
increment, inside the `cond` handler. **A straight-line corpus cannot
reach D by construction**, so no run of that instrument could ever have
falsified C1. A falsifier that the instrument cannot produce is not a
falsifier; it is a restatement. The lesson is not "the corpus was too
small" — it is that a claim quantified over a whole file was tested by an
instrument whose own description named the axes it varied, and control
flow was not among them.

Closed at `ab4b9b5` by separating the note gate from the flag gate;
`test_a_branch_scoped_unsatisfiable_assume_no_longer_refutes` covers all
three detection sites.

**C6 and C7 — MET, but neither can distinguish this fix from doing
nothing.** C6 asks that REFUTED counts be monotonically non-increasing and
WRONG-VERIFIED stay 0; C7 asks that 60 rows be unchanged. **Both are
satisfied by the empty change**, and C6 is also satisfied by a change that
returns UNKNOWN for every row in the corpus. Only C4 and C5 ("zero
mechanism-A / mechanism-B rows remain") require anything to move, and they
too are satisfied by a blanket UNKNOWN. **No clause of this
pre-registration would fail a maximally-conservative no-op-everything
change**; what would fail it is the test suite (C9), which pins many
REFUTEDs. That is worth recording because the report reads as though the
eleven clauses jointly constrained the outcome, and they did not — C9 was
carrying more of the weight than its wording suggests.

**C7 — FALSE at the tip. A reserved row MOVED.** The scored outcome says
"the 60 mechanism-C rows are byte-identical before and after". They are
not, at `refine="affine"`. The affine guard keys on
`propagation.assume_dropped`, a whole-run flag with no order in it, while
the interval withhold reads `uncertified` at assert time — so the two legs
scope an assume differently, and the branch changed the query-scoped one.
Measured (`<PY> <W>/probe_p3.py`), an assert traced BEFORE a wholly-dropped
assume over an empty region:

```
                                  refine=None   refine="affine"
affine-decided,  9efea6f            UNKNOWN        REFUTED
affine-decided,  3afbf01            UNKNOWN        UNKNOWN     <- MOVED
affine-decided,  no assume at all    UNKNOWN        REFUTED     <- leg still works
interval-decided, 9efea6f/3afbf01    REFUTED        REFUTED
```

The direction is REFUTED -> UNKNOWN, so the registered hard boundary
("withholding is the only direction I will move a verdict in") HELD. What
did not hold is C7's own scored claim of no movement. The probe that
scored it, `probe_branches.py`, and the pin,
`test_an_assume_after_the_assert_is_the_reserved_ordering_question`, both
ran at `refine=None` only — **blind to the exact half that moved**. The
pin is now `test_an_assume_after_the_assert_is_pinned_on_BOTH_legs`, four
cells, and the leg disagreement itself is pinned by
`test_the_two_legs_do_not_yet_agree_on_assume_ordering`.

The principal has since ruled the question **query-scoped**. Implementing
that uniformly is a separate change with its own pre-registration; none of
it is done here.

**C8 — line numbers STALE, substance HOLDS.** The scored outcome cites
4967 / 5035 / 5089. At `3afbf01` the three sites are **4977 / 5061 /
5115**; at `ab4b9b5`, **4977 / 5061 / 5140**. The drift is this branch's
own later commits moving lines under a figure quoted from an earlier one —
the same trap the C9 re-scoring records for `docs/supported-primitives.md`,
one file over. The substance is unaffected and is now stronger: site 5140
was inside `if dropped or not vacuous:` and is now unconditional, so
`assume_dropped or coverage.constrained` is total on a strictly larger set
of paths than when C8 was scored.

**C11 — MET as scored, with one correction to what it proves.** The
discriminant's docstring advertised "three refusals, each load-bearing".
**All three change no outcome.** Full-suite mutation at `ab4b9b5` (2180
tests, `<W>/sweep.sh`) deletes each in turn and reddens nothing; an
instrumented full-suite run (`<W>/P1_pre/instr.py`, recording every
`_conjunct_certainly_true` call as `(dtype, ieee_flagged, size0,
answer)`) says why, over all 93 calls the method receives:

* **non-bool** — unreachable in effect. jax promotes `bool & int32` to an
  *int32* `and`, so `_classify_assumed_pred` refuses the whole tree at the
  top, before the recursion that fills `harmless`. `int32` reaches the
  discriminant twice and **never** inside an assume that narrowed, so its
  answer is never read.
* **maybe-NaN under ieee** — dead. `_ieee_cmp` returns flag `False` on
  comparison outputs and degrades a would-be definite TRUE to
  `BOOL_UNKNOWN` on a flagged operand, so a flagged bool is `[0, 1]`,
  never `[1, 1]`. All 10 flagged calls already answer False on their own.
* **size-0** — never asked. `size0` is False in **all 93** calls. An
  `and`'s operands must broadcast, so a zero-element conjunct forces every
  sibling to be zero-element too, and a zero-element comparison takes the
  `point_a and point_b` arm (both `all()`s vacuously true) and drops
  without narrowing, leaving `harmless` unread.

The docstring now carries the measurement instead of the assessment.
C11's measured cost figures are untouched by this.

**Not a registered clause, but the same defect class, found while
scoring C11**: the **attribution fail-safe** (`harmless = [False] *
len(dropped)`) had no test at all — mutation reddened 0 of 2180. It is
now driven directly by
`test_the_attribution_fail_safe_refuses_a_misaligned_verdict`, and the
first version of that pin did not bite either: it appended the bogus
entry, and a salvaging `[:len(dropped)]` still read the right value at
index 0. It inserts at the front now. Verified both ways.

**What this re-scoring does NOT revisit**: C2, C3, C4, C5, C9, C10 and the
hard boundary were re-read and not re-measured; the corpus tooling they
cite (`<W>/work/`) was destroyed with the scratchpad before this pass ran,
so their figures are carried forward as reported, not independently
re-confirmed. The `reduce_and` look-through the audit identifies as
recovering ~90% of the lost refutations is a separate spike and is not
touched here.

## Mutation ledger for this pass

Every mutation is one edit to the frozen tree, then the FULL suite. The
`test_committed_page_matches_live_registries` failure marked (doc) is an
artifact of the method, not a pin: any mutation that changes
`propagate.py`'s line count moves a line the generated page cites.
`<W>/sweep.sh`, `<W>/sweep2.sh`.

| mutation | substantive failures | what it establishes |
|---|---|---|
| none | 0 (2185 p / 2 s) | baseline |
| the D fix re-gated with the note | **8** (+doc) | the fourth mechanism is closed and all three detectors are covered |
| `if restricting or vacuous:` → `if restricting:` | **1** | `or vacuous`, previously uncovered by all 2171 tests, is pinned |
| `_classify_assumed_pred`'s `and` bool guard deleted | **1** (+doc) | the reachable non-bool refusal is pinned by its disclosed reason |
| the attribution fail-safe salvages instead of refusing | **1** | the fail-safe is pinned (0 before this pass, and 0 again for the first version of the pin) |
| `_conjunct_certainly_true` dtype gate deleted | 0 (+doc) | unreachable in effect — documented, not pinned |
| `_conjunct_certainly_true` ieee maybe-NaN gate deleted | 0 (+doc) | dead — documented, not pinned |
| `_conjunct_certainly_true` size-0 gate deleted | 0 | never asked — documented, not pinned |
| `_conjunct_certainly_true` → `return True` | **10** (+doc) | the discriminant as a whole is well covered |
| the affine leg's `propagation.assume_dropped` guard neutralised | **5** | the extended ordering pin BITES: `[affine-leg/refine-affine]` and `test_the_two_legs_do_not_yet_agree_on_assume_ordering` both redden |

The last row is the one that matters for the forthcoming query-scoping
change. The `interval-leg/*` cells assert equality to `"REFUTED"`, so any
move reddens them by construction; the affine cells needed the measurement,
because "UNKNOWN" could also have meant the leg going dark. It does not:
the no-assume control still refutes, and switching the guard off brings the
REFUTED back.
