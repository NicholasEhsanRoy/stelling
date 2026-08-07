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
