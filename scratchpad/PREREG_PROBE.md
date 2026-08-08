<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG_PROBE — the reachability certificate's witnesses must be MEMBERS

Branch `fix/probe-membership`, base `688e829` (`main`), worktree
`/home/nick/MSF/.wt-probe/main`. Written **before** the first `src/` edit.
Outcomes are appended below the rule; nothing above it is edited afterwards.

Every command below runs with
`JAX_PLATFORMS=cpu PYTHONPATH=/home/nick/MSF/.wt-probe/main/src` and
`/home/nick/venvs/stelling-jax/bin/python` (jax 0.11.0) or
`/home/nick/venvs/stelling-jax010/bin/python` (jax 0.10.2), with
`python -c "import stelling; print(stelling.__file__)"` verified before every
comparison. `jax_enable_x64` is set inside every corpus runner and lab script.

## What is being fixed, as reproduced on `688e829`

**D1 — the certificate certifies with points that are not in the declared
set, and that is a live wrong REFUTED.** `_probe_point` rounds an integer
declaration to an integer and then clamps to `[lo, hi]`, which puts the
non-integral endpoint back:

```
$ python /home/nick/MSF/.audit-reach/lab/attack_member.py
A int non-member lo      propagate=['discharged', 'violated-over-set']  check=REFUTED
B int non-member hi      propagate=['discharged', 'violated-over-set']  check=REFUTED
C control reachable      propagate=['discharged', 'violated-over-set']  check=REFUTED
D control unreachable    propagate=['discharged', 'unknown']            check=UNKNOWN
```

`any_array((), "int32", (0.2, 2.8))` declares the integers of `[0.2, 2.8]`
— stelling's own refusal for `(0.2, 0.8)` says so in as many words ("int32
represents the integers ... and the interval contains none of them") — so
the declared set is `{1, 2}`. `i < 1` and `i > 2` are false at **every**
member, and both stamp REFUTED.

The same defect has a second, un-reported face: the declared set is also
bounded by what the **dtype** can hold.

```
int8 (-1e9, 1e9), guard `k < -200`  ->  ['discharged', 'violated-over-set']
```

int8 holds `[-128, 127]`; `k < -200` is false at every member, and it
REFUTES.

**D2 — the probe grid degenerates on a wide box.** `lo + f*(hi-lo)` with
`hi-lo` overflowing:

```
x in [-1e308, 1e308]: probes = [None, 1e308, 1e308, ... 1e308]  (2 distinct)
guard `w < -1e307` (satisfiable at w = -1e308):  WITHHELD ('unknown')
guard `w > 1e307`  (control, satisfiable):        REFUTED
```

**D3 — an index-EXCLUDED cond branch is dropped silently.**
`cond(c > 0., yes, no)` with `c` declared over `[1, 2]`:

```
asserts in IR = 2   obligations = 1   named-unexamined = 0   notes = ()
```

Sound (the branch is provably untaken), but it contradicts pre-registered
clause **C5** of `scratchpad/PREREG_REACH.md`, whose falsifier could not see
it because it ranged only over obligations the oracle saw **executed**.

**D4 — an assert inside a `while` inside a `scan` is attributed to the
outermost primitive.** Measured: detail says `'scan'`; the innermost
primitive that actually swallowed it is `'while'`.

## The instrument

`scratchpad/probe/` is a NEW corpus and a NEW oracle, written for this
branch and independent of both `scratchpad/reach/` and the audit's
`/home/nick/MSF/.audit-reach/`. Two design decisions are deliberate, and
both exist because of how C5 failed:

1. **The oracle samples MEMBERS.** For an integer or boolean declaration it
   enumerates the integers of `[ceil(lo), floor(hi)]` intersected with the
   dtype's own range — never `uniform(lo, hi)`. An oracle that samples
   `0.2` for an `int32` cannot see D1 at all; the previous one did exactly
   that, which is why D1 survived a 736-case corpus.
2. **The obligation universe comes from the SOURCE, not from the oracle's
   executed set.** Every `S(...)` line in the generated case file is a row,
   whether or not any sampled point evaluated it. A falsifier that ranges
   only over what the oracle saw executed cannot see a dropped obligation
   in an unreachable branch — which is precisely D3, and precisely how C5
   was falsified without noticing.

Ledger is **per obligation**; the per-query roll-up is reported alongside,
never instead.

## Clauses

**C1 — D1 is real at baseline and the members say so.** For each of the D1
cases, the independent oracle, ranging over EVERY member of the declared
integer set, finds the branch obligation executed at **0** points, while
baseline stelling stamps `violated-over-set` / REFUTED.
*Falsifier:* an oracle member at which the guard holds (which would make the
REFUTED sound), or a baseline status that is not `violated-over-set`.

**C2 — after the fix, every probe point stelling can form is a MEMBER of
the declared set.** Over a sweep of every dtype in `_INT_DTYPE_BOUNDS` plus
`float32/float64`, of at least 200 `(lo, hi)` pairs including fractional,
dtype-overflowing, infinite, and degenerate ones, and of every `k in
range(_PROBE_COUNT)`: every value `_probe_point` returns lies in `[lo, hi]`,
is integral when the dtype is integral or boolean, and lies inside the
dtype's representable range.
*Falsifier:* one sweep point violating any of the three. The sweep ranges
wider than the fix (it includes float dtypes and the infinite-endpoint
ladder, neither of which D1 touches).

**C3 — "no member" means "no witness", never "certified".** When the box
holds no value of the dtype, `_probe_point` returns `None`, and a `None`
probe leaves the declaration at its full box, which certifies nothing: the
obligation is WITHHELD.
*Falsifier:* a non-`None` return for an empty integer box; or a certified
branch on a query whose every probe returned `None`.
*Positive control against a vacuous C2* (an implementation returning `None`
everywhere would satisfy C2 trivially): on the corpus, after the fix, the
count of branch-scoped `REFUTE_SOUND` obligation rows is **> 0** and within
the cost budget of C6, and the pinned probe still forms a point for at least
95% of the corpus's declarations.

**C4 — the direction that matters. No fix here moves anything toward
VERIFIED.** In the paired baseline/after per-obligation ledger, no
obligation moves to `discharged` from any other status, and no query moves
into VERIFIED.
*Falsifier:* any such row in the ledger. Range: every obligation of every
case on every leg, including the obligations the oracle never saw executed.

**C5 — no VERIFIED over a violating point.** After the fix, no corpus query
is VERIFIED while the oracle finds a MEMBER point of the declared set at
which some obligation is executed and false.
*Falsifier:* any `FALSE_VERIFIED` row in the after ledger.
*Positive control:* the instrument must be able to see unsoundness — the
baseline ledger must contain at least one `REFUTE_ON_UNREACHABLE` row (the
D1 class). A zero-on-both-sides result would mean the corpus is inert, not
that the fix worked.

**C6 — the cost ledger, per obligation.** Every obligation that moves does
so from an unsound `violated-over-set` to `unknown`, or from `unknown` to a
`violated-over-set` the oracle confirms (a witness the old grid missed,
D2's class). No other transition occurs.
*Falsifier:* a moved row in any other class — in particular any
`REFUTE_SOUND -> UNKNOWN_*` row that the oracle says was genuinely
refutable, which is the capability cost this clause exists to bound.
*Positive control:* the ledger must be non-empty on both directions of
movement; a fix that moved nothing would falsify D1 and D2 rather than
confirm the fix.

**C7 — D3 is stated, not silently dropped, and the choice is made on a
measurement.** The index-excluded branch is documented in `SOUNDNESS.md`
and against C5 of `PREREG_REACH.md`, and pinned by a test, RATHER than
recorded as an unexamined obligation — because naming it `unknown` costs
VERIFIED on queries whose excluded branch is provably untaken.
*Falsifier:* the measured cost of the naming alternative, built in a
separate worktree and run over the same corpus, is **zero** queries losing
VERIFIED — in which case the cheap remedy is naming and this choice is
wrong.
*Positive control:* the corpus must contain at least one VERIFIED query with
an index-excluded branch, or the cost measurement is vacuous.

**C8 — D2 is fixed and wide boxes keep distinct probes.** On boxes spanning
`[-1e308, 1e308]`, `[0, 1e308]`, `[-1e308, 0]` and `[-1e300, 1e300]`, the
16 probes are 16 DISTINCT finite points, all inside the box; and the
genuinely-reachable low-side guard `w < -1e307` refutes after, having been
withheld before.
*Falsifier:* fewer than 16 distinct probes on any of those boxes, a probe
outside the box, or the low-side guard still withheld.
*Control that the fix is not "always refute":* the definitely-false guard
`w - w > 0` on the same wide box stays withheld.

**C9 — D4: the swallowing primitive named is the INNERMOST one.** An assert
inside a `while` inside a `scan` is attributed to `'while'`; an assert
directly inside a `scan` is still attributed to `'scan'`; the source
location is unchanged in both.
*Falsifier:* either attribution wrong, or the source location moving.

**C10 — the six unpinned surfaces are pinned, and each mutation reddens.**
Each of M6 (integer pinning deleted), M8 (assume gate deleted), M9
(`_PROBE_COUNT` = 3), M12 (nonvacuity dropped from
`_UNEXAMINABLE_OBLIGATIONS`), M18 (branch index dropped from the path) and
N23 (path dropped from the key) makes at least one test FAIL, each built in
its OWN `git worktree`, run with `python -B` after clearing `__pycache__`.
The M9 test is a witness that genuinely requires a probe beyond the three
anchors, so `_PROBE_COUNT` is pinned by a requirement and not by its value.
*Falsifier:* any of the six leaving the test files green.

**C11 — both series, ids identical.** jax 0.11.0 and jax 0.10.2 both pass
with only the tests this branch adds appearing; `--collect-only` id sets are
identical between the series, and the after-set minus the before-set is
exactly the added ids with nothing removed.
*Falsifier:* a failure on either series, an id removed, or an id-set
difference between the series.

**C12 — the cost clause: the witness search still runs only when there is a
candidate.** A query with no branch-scoped violation constructs exactly one
propagator; the fix adds no probe run.
*Falsifier:* a propagator count above 1 on a candidate-free query.
*Positive control:* the same measurement on a query WITH a candidate must
exceed 1, or the zero above is measuring a search that never runs.

---

## Outcomes

All measurements below ran with `JAX_PLATFORMS=cpu` and the worktree's own
`PYTHONPATH`, `stelling.__file__` printed and checked before every
comparison. Baseline is `/home/nick/MSF/.wt-probe/base` at `688e829`;
branch is `/home/nick/MSF/.wt-probe/main`.

### Errata, recorded before the clauses

**E1 — D1's "second face", as written above, is WRONG, and the correction
is measured.** `int8 (-1e9, 1e9)` with the guard `k < -200` does REFUTE at
baseline, but not for the stated reason: jax wraps the Python literal into
int8, so the traced program compares against **56**, the guard is
satisfiable at real members, and the refutation is SOUND. The dtype-range
face is real and reachable through a float cast, which is what keeps the
literal out of int8:

```
int8 (-1e9, 1e9), guard `k.astype(f64) < -200.0`
  baseline  jax 0.11.0 / 0.10.2:  ['discharged', 'violated-over-set']   <- wrong REFUTED
  branch    jax 0.11.0 / 0.10.2:  ['discharged', 'unknown']
control `k.astype(f64) < -100.0` (true at the member -128): REFUTED on both
```

The claim in D1 stands with that construction substituted; the original
sentence would have been a reasoned-not-run finding, and it was wrong.

**E2 — the corpus's first sampler pinned a declaration to one member.** A
stride scheme (`cand[n][(j*(1+2t)) % len]`) held a 3-member `int32`
declaration at its first member for every point, and the oracle then called
two genuinely reachable branches unreachable. Found by inspecting the
4 residual `REFUTE_ON_UNREACHABLE` rows of an early after-run, fixed to a
mixed-radix enumeration with the smallest declaration varying fastest plus
150 independent draws, and every clause below is scored on the fixed
instrument. Recorded rather than quietly re-run: an instrument that can
manufacture the finding it is measuring has to be named.

### Outcomes

**C1 — MET.** Baseline, both series (`attack_member.py`): `int32 (0.2, 2.8)`
with `i < 1` and with `i > 2` both stamp `violated-over-set` / **REFUTED**.
The oracle enumerates the declared set **exhaustively** — `_members` returns
`{1, 2}`, both of them — and finds the branch obligation executed at **0**
of 730 sampled points. Six of the corpus's ten declarations are enumerated
exhaustively (`i_frac` 2 members, `i_frac2` 3, `i_int` 4, `i_one` 1,
`u8_neg` 4, `b_bool` 2), so for those the word "unreachable" is exact and
not a sampling statement.

**C2 — MET.** `test_every_probe_point_it_can_form_is_a_member`, 13 dtypes
(11 integer/boolean + `float32` + `float64`) x 215 bound pairs x 16 probe
indices: every returned value is finite, inside `[lo, hi]`, integral when
the dtype is integral or boolean, and inside the dtype's own range.

**C3 — MET.** `_probe_point` returns `None` for `int32 [0.2, 0.8]`,
`int64 [-0.5, -0.2]` and `bool [2, 3]`, and the sweep asserts `None` for
EVERY pair the exact-integer specification calls empty. End to end, on
hand-built IR (the only route — `any_array` refuses a memberless
declaration): the branch violation is WITHHELD, while the same guard over
`int32 [0, 3]` still refutes. Positive control against a vacuous C2: the
sweep asserts `formed >= 16 * inhabited`, i.e. every probe of every
inhabited pair really was formed.

**C4 — MET.** Paired ledger, 336 cases x 2 legs: **0** obligations became
`discharged`, **0** queries moved into VERIFIED. `DISCHARGE_SOUND` 678 →
678 and `DISCHARGE_VACUOUS` 68 → 68, unchanged.

**C5 — MET, with its positive control.** After: `FALSE_VERIFIED` **0**,
unsound obligation rows **0**. The instrument can see unsoundness: the
baseline has **18** `REFUTE_ON_UNREACHABLE` rows and **18** `FALSE_REFUTED`
queries, so the zero is a measured move and not an inert corpus.

**C6 — MET, with its positive control.** Every moved obligation is in one
of the two admitted classes and nothing else moved:

```
REFUTE_ON_UNREACHABLE  -> UNKNOWN_ON_UNREACHABLE   18
UNKNOWN_ON_FALSE       -> REFUTE_SOUND              8
total moved obligations: 26     became discharged: 0
per query:  FALSE_REFUTED -> UNKNOWN  18      UNKNOWN -> REFUTED_SOUND  8
```

**0** `REFUTE_SOUND` rows were lost — the capability cost this clause exists
to bound is zero on this corpus — and the 8 gains are the wide-box
witnesses of D2, each confirmed really violated by the oracle.

**C7 — MET; the alternative was measured, not argued.** Built in its own
worktree (`/home/nick/MSF/.wt-probe/alt-name`, `_record_unexamined` added to
the excluded-branch walk) and run over the same corpus:

```
SWALLOWED_UNREACHED -> UNKNOWN_ON_UNREACHABLE   64
per query:  VERIFIED_SOUND -> UNKNOWN           28      (of 28)
unsound rows: 0 -> 0
```

Naming costs **every** VERIFIED in the corpus and removes **zero** unsound
rows. The falsifier — a measured cost of zero queries — is not met, so the
documented exception stands: `SOUNDNESS.md` entry (3) and the amendment
appended to `PREREG_REACH.md`, pinned by
`test_an_index_excluded_branch_is_deliberately_dropped`. Positive control:
the corpus contains 28 VERIFIED queries with an index-excluded branch, so
the measurement was not vacuous.

**C8 — MET.** `test_a_wide_box_keeps_sixteen_distinct_probes` on
`[-1e308, 1e308]`, `[0, 1e308]`, `[-1e308, 0]`, `[-1e300, 1e300]` and
`[-1, 1]`: 16 distinct finite points inside the box, `lo` and `hi` among
them exactly. Baseline on `[-1e308, 1e308]`: **2** distinct values (15
probes on `hi`, one `None`). End to end, `w < -1e307` moves WITHHELD →
REFUTED while the control `w - w > 0` on the same box stays withheld.

**C9 — MET.** assert inside `while` inside `scan`: detail names `'while'`
(baseline: `'scan'`), the note agrees with the detail, and the source
location is unchanged. An assert directly under a `scan` still names
`'scan'`.

**C10 — MET, and wider than pre-registered: 10 mutations, all redden.**
Each in its own `git worktree` under `/home/nick/MSF/.wt-probe/mut/`, built
by string replacement, `__pycache__` cleared, run with `python -B`,
`stelling.__file__` asserted to be the mutant's:

```
M6_no_integer_rounding     12 failed   M6b_clamp_to_raw_box      14 failed
M6c_no_dtype_range_clamp   13 failed   M8_no_assume_gate          1 failed
M9_probe_count_3            1 failed   M12_assert_only            1 failed
M18_no_branch_index         1 failed   N23_id_only_key            1 failed
F4_overflow_grid           15 failed   F5_outermost_swallower     1 failed
unmutated control:                                              55 passed
```

The `_PROBE_COUNT` test is a requirement and not a value: `x[1] - x[0] > 1.0`
is `0` at all three anchors, `0.49` at probe 3 and `1.51` at probe 4.

**C10 erratum — the first M8 test was inert and stayed green.** It assumed
over `x[0]`, a SLICE of a declaration; that is an over-approximated
intermediate, so the refutation was withheld by the precondition's own
satisfiability rule before the reachability gate was consulted
(`branch_violations` was empty — there was no candidate). Rewritten to
constrain the declaration itself, and it now asserts the withholding note is
the reachability one. This is the second time in this branch that a control
was satisfied through the wrong door; both are recorded.

**C11 — MET.** jax 0.11.0: **2329 passed, 2 skipped** in 135.05 s. jax
0.10.2: **2329 passed, 2 skipped** in 137.84 s, run serially after it (the
`PREREG_REACH.md` lesson about concurrent runs in one worktree). Baseline
collects 2298 ids on both series; the branch collects 2331 on both. The two
series' id sets are byte-identical (`diff` empty) on baseline and on branch;
`comm` shows **0** ids removed and **33** added, all in
`tests/test_probe_witness.py`.

**C12 — MET, with its positive control.** Propagator constructions: **1**
with no candidate, **17** with one, on baseline and on branch alike. Wall,
200 iterations each: no candidate 0.027 → 0.025 ms/propagate, with
candidate 1.504 → 1.430 ms/propagate (baseline → branch), load average 3.85
on 24 cores. The witness search costs what it cost.

### The residual this branch did NOT fix

`_probe_point` rounds to an integer for integer dtypes and clamps for float
dtypes, but it never rounds to the declared FLOAT format: a `float32`
declaration can be pinned to a binary64 value that is not a float32, which
is the same "not a member" shape as D1. **SUSPECTED, not measured** — no
construction in this branch turns it into a wrong verdict, and unlike the
integer case the fix needs a real float-format rounding rather than a
clamp. It is named here so the next reader does not have to re-derive it.

---

## Correction, appended 2026-08-08 by the repair of this branch

Nothing above this heading is edited; this is the record of what one
recorded clause claimed and what it actually measured.

**C2 was recorded MET and its claim was false as written.** The clause
requires that "every value `_probe_point` returns lies in `[lo, hi]`, is
integral when the dtype is integral or boolean, and lies inside the
dtype's representable range", over a sweep it says explicitly includes
`float32` and `float64`. The third conjunct was never checked for the
float parameters, and it is false for them. The falsifier the clause
names — "one sweep point violating any of the three" — was reachable and
was never reached, because the test performed both dtype-specific checks
inside `if _is_integer_dtype(dtype):`; for `float32` and `float64` it
asserted only finiteness and `lo <= v <= hi`. The name of the test
outran what it asserted, and both the test and the clause read as though
membership had been checked for every dtype in the sweep.

**Measured, on the branch's own `_SWEEP_BOUNDS` (215 bound pairs x 16
probe indices x 2 elements) at `62e4190`:**

| dtype | values formed | not a value of the dtype | outside the dtype's range |
|---|---|---|---|
| `float16` | 6880 | **6736** | **6184** |
| `float32` | 6880 | **6716** | **90** |
| `float64` | 6880 | 0 | 0 |

The `float32` row reproduces the audit's figures exactly. `float64` is
clean, and that is the whole reason the clause could be believed: it is
the one float format that is its own interval, so the only float dtype
the sweep checked in a way that could have failed was already right.

**C2 as it should have read**, and as it now reads in the test: *every
value `_probe_point` returns lies in `[lo, hi]` and is a VALUE of the
declared dtype* — one sentence covering both families, with the
membership oracle asked of jax's own cast rather than of stelling.
`_declares_a_member` also had to be corrected before the sweep could
fail: it answered `lo <= hi` for float dtypes, which is the same error
the implementation made, and while the oracle said it the test could not
go red.

**After the repair, the same sweep:** `float16` **0** non-members of 832
formed, `bfloat16` **0** of 2848, `float32` **0** of 6880, `float64`
**0** of 6880. The formed counts fall for the narrow formats because
boxes holding no value of the dtype now yield no probe at all, which is
C3's "no member, no witness" and is the withholding direction.

**The residual named at the end of the Outcomes section is no longer
SUSPECTED.** That paragraph says no construction in this branch turns it
into a wrong verdict. Three do, and one of them is new on this branch:
see `SOUNDNESS.md` entry (5) for all three, the ledger, and the
comparison of the two remedies.

**C11 re-run after the repair.** jax 0.11.0: 2348 passed, 2 skipped in
140.58 s. jax 0.10.2: 2348 passed, 2 skipped in 155.84 s, run serially in the
same checkout, load average 0.44 and 6.01 at the two starts.
`--collect-only` collects **2350** ids on both series and the two id sets are
byte-identical (`diff` empty); `62e4190` collected 2331.
