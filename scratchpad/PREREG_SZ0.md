<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG — the size-0 vacuous-conjunct narrowing (a wrong VERIFIED)

Registered **before the first `src/` edit** on branch
`fix/size0-vacuous-conjunct-narrowing`, cut from `main` at `c20f38e`.
Worktree `/home/nick/MSF/.wt-sz0/W`; lab scripts in `/home/nick/MSF/.wt-sz0/lab`
(oracle `oracle.py`, constructions `cases.py`, runner `run_sweep.py`, shape
enumeration `shapes.py`).

Environment for every command below:

    export JAX_PLATFORMS=cpu PYTHONPATH=/home/nick/MSF/.wt-sz0/W/src JAX_ENABLE_X64=1
    PY=/home/nick/venvs/stelling-jax/bin/python        # jax 0.11.0
    PY010=/home/nick/venvs/stelling-jax010/bin/python  # jax 0.10.2

`$PY -c "import stelling; print(stelling.__file__)"` is run before every
comparison and must print `/home/nick/MSF/.wt-sz0/W/src/stelling/__init__.py`.

The oracle is independent of stelling: it samples the DECLARED boxes
(corner cartesian product + 100 000 joint uniform draws) and evaluates
numpy twins of the assume and assert predicates. `all()` of a bool array
is the universal reading of both, which is what numpy and jax give for
size-0 (vacuously True).

---

## Claims and their falsifiers

**C1 — the defect exists on `c20f38e`.** `A1_size0_right` (the reported
construction) returns VERIFIED while the oracle finds at least one
admitted point that violates the assert.
*Falsifier:* `run_sweep.py` reports A1 as anything other than VERIFIED,
or reports `violating == 0` for it. Both are observable on the printed
row; a stelling change alone cannot satisfy this, because the violating
count comes from numpy only.

**C2 — it is a class, not one construction.** At least these ten
constructions are VERIFIED-with-a-violating-admitted-point on `c20f38e`,
in BOTH `vacuity_mode`s and with `refine` both `None` and `"affine"`:
A1_size0_right, A2_size0_left, A4_nested_left, A5_nested_right,
A6_mixed_or_inside, A9_shape1_vs_0, A10_rank2_2x1_vs_2x0,
A11_size0_via_operand, A14_eq_narrowing, A19_le_narrowing.
*Falsifier:* any listed row is not VERIFIED before the fix, or has
`violating == 0`.

**C3 — the mechanism.** Every wrong VERIFIED in the class has a
`stelling_assume` operand of size 0 while the conjunct that narrowed
names a variable of size > 0. *Falsifier:* a wrong VERIFIED in the class
whose `stelling_assume` operand has size > 0 — read off the transcribed
IR by `dump.py`, which prints every equation's operand shapes. This is
wider than the claim: `dump.py` prints ALL cases, including the ones that
are not wrong, so a size>0 counterexample anywhere shows up.

**C4 — the general rule.** `all(A & B)` implies `all(A)` iff every
element of `A` survives the broadcast into the output; over numpy
broadcasting that fails exactly when the output has zero elements and `A`
does not; and no size-0 operand can produce a nonzero-size output — so
"some node of the `and` tree is size 0" and "the root assume predicate is
size 0" coincide.
*Falsifier:* `shapes.py` exhibits (i) a broadcastable shape pair that
loses an element of an operand with a NONZERO-size output, or (ii) a pair
with a size-0 operand and a nonzero-size output. The enumeration covers
all ordered pairs from a 16-shape set including rank 0–3, unit axes and
zero axes, so it is at least as wide as the claim.

**C5 — the nearest rival framing is wrong.** The framing "gate the
narrowing conjunct on its OWN size" does not close the class.
*Falsifier:* A1's narrowing conjunct `k >= 0.5` having size 0 (it has
size 1 — `dump.py` prints its shape `()`), or the fix built on that
framing leaving zero wrong VERIFIEDs. An inert control cannot satisfy
this: the discriminator is a shape that is printed either way.

**C6 — the fix closes the direction that matters.** After the fix no
harness in the sweep is VERIFIED with a violating admitted point, in
either `vacuity_mode`, with and without `refine="affine"`, under
`semantics="real"` and `semantics="ieee"`, under `assume_mode` both
`constrain` and `inert`, and with the assume nested inside a `cond`
branch.
*Falsifier:* one post-fix row that is VERIFIED with `violating > 0`. The
sweep prints every row, so absence of the marker across the whole table
is the observation, not a selected one.

**C7 — the cost is one-directional and every unit of it was unsound.**
Every verdict that moves VERIFIED → UNKNOWN between the before and after
sweeps had `violating > 0` before. No verdict moves UNKNOWN → VERIFIED or
REFUTED → VERIFIED. *Falsifier:* a VERIFIED → UNKNOWN move whose oracle
`violating == 0`, or any move INTO VERIFIED. Both are read from the
committed `before.json` / `after.json` diff.

**C8 — the same root cause also mints false loud refusals.**
`A16_false_unsat_alarm` and `A18_false_collapse_alarm` raise
`UnsatisfiableAssumptionError` ("harness defect; nothing was verified")
on `c20f38e` although their assume is a `bool[0]` and therefore
satisfiable at every point of the declared box; after the fix neither
raises. *Falsifier:* either still raises after the fix, or the oracle
reports `admitted == 0` for either before the fix (which would make the
alarm true).

**C9 — both series stay green and collect the same ids.** The full suite
passes under jax 0.11.0 and 0.10.2, and `pytest --collect-only -q` yields
identical id sets on the two series and the same set as `c20f38e` plus
exactly the tests this branch adds. *Falsifier:* any failure, or a
non-empty id-set difference that is not one of the added tests.

**C10 — `_conjunct_certainly_true`'s size-0 refusal is REACHABLE.** On
branch `fix/branch-vacuous-and-claim-scope` (`6237e07`) there is a
harness for which that method is called with a size-0 atom.
*Falsifier:* an instrumented run over the whole sibling sweep recording
zero calls with `size0=True`. The instrument records EVERY call with its
atom size, so it can falsify as well as confirm.

**C11 — that gate is load-bearing in the other direction.** With the
size-0 refusal deleted on `6237e07`, at least one sweep construction
flips from UNKNOWN to REFUTED, and the oracle shows that REFUTED to be
wrong (no admitted violating point). *Falsifier:* deleting the gate
changes no verdict in the sweep, or the flipped REFUTED's construction
has `violating > 0` (which would make it sound).

**C12 — what the 93-call instrument missed.** No test in the suite
reaches assume classification with a size-0 predicate; the suite's size-0
declarations all flow into `assert_` or a `jnp.all` reduction, never into
an `assume` operand. *Falsifier:* an instrumented full-suite run on
`c20f38e` recording at least one `_apply_assumed_pred` call whose atom has
size 0. The instrument records every call, so a single hit falsifies.

---

## What the fix will be (registered so the outcome can contradict it)

The `and` recursion must not treat a conjunct as separately assumed when
the predicate it belongs to is vacuously true. Concretely: a predicate
node with zero elements is vacuously true, implies nothing about its
subterms, and must therefore neither narrow, nor certify satisfiability,
nor raise the unsatisfiable-precondition oracle. The gate goes at the top
of `_Propagator._apply_assumed_pred`, so it fires at the root, at every
`and` node and at every leaf.

Registered prediction, falsifiable by the after-sweep: this makes A1, A2,
A4, A5, A6, A9, A10, A11, A14, A19 UNKNOWN; A16 and A18 stop raising and
become UNKNOWN; A17 (a REFUTED that was sound) becomes UNKNOWN;
B1_legit_conjunction and B2_legit_broadcast stay VERIFIED.

---

## OUTCOMES

(appended below, never edited above this line)

Every command below ran with
`export JAX_PLATFORMS=cpu PYTHONPATH=/home/nick/MSF/.wt-sz0/W/src JAX_ENABLE_X64=1`
and `$PY -c "import stelling; print(stelling.__file__)"` verified first.
"pre-fix" means the source at `fb34e0d` (this branch's prereg commit,
identical to `c20f38e` in `src/`), extracted with
`git archive fb34e0d src/stelling` into `/home/nick/MSF/.wt-sz0/prefix_src`.
Box: 24 cores; load average is quoted with every wall figure, and other
sessions were running pytest on the same box throughout, so wall times are
upper bounds rather than clean single-run figures.

**C1 — MET.** `$PY lab/run_sweep.py before.json`:
`A1_size0_right / inputs-only / None → VERIFIED`, oracle
`100006 points / 100006 admitted / 50231 violating`; first three oracle
witnesses `k = -1.0, 0.0, 0.0` (each with `z = []`). The no-assume control
returns UNKNOWN.

**C2 — MET, and the list was too short.** All ten listed constructions are
VERIFIED with a violating admitted point pre-fix, in both `vacuity_mode`s
and at both `refine` depths — 40 rows. An eleventh was added while
scoring: `A13_cond_branch`, the same narrowing inside a `cond` branch
(5001 of 20005 sampled admitted points violate). 44 rows total.

**C3 — MET.** `$PY lab/dump.py` prints the transcribed IR of every case.
In each wrong-VERIFIED case the `stelling_assume` operand has a zero
extent — `(0,)`, `(2, 0)` — while the narrowed variable does not (`()`,
`(1,)`, `(2, 1)`). No case in the table has a size>0 assume operand and a
wrong VERIFIED.

**C4 — MET.** `$PY lab/shapes.py` over all 256 ordered pairs from a
16-shape set: `LOSSY pairs: 31`; `every lossy pair has a ZERO-SIZE
output: True`; `counterexamples: []`; `size-0 operand yielding a NONZERO
broadcast output: []`. The second result is what makes the per-node gate
and a root-only gate extensionally equal on reachable input.

**C5 — MET.** `A1`'s narrowing conjunct `k >= 0.5` has shape `()`, size 1
(`lab/dump.py`, the `ge [(1, (), 'float64'), …] -> [(3, (), 'bool')]`
line). A gate on the conjunct's own size could not have fired on it, so
that framing does not close the class. A second rival — gate only the
ROOT assume operand — is NOT distinguished by any construction I could
build, because C4's second half makes a size-0 node force every ancestor
size-0; the per-node placement is preferred as the direct encoding of the
rule, and this is recorded as an undistinguished choice rather than a
measured one.

**C6 — MET.** `$PY lab/run_sweep.py after.json` → 0 rows marked wrong
VERIFIED (84 rows: 21 constructions × 2 modes × 2 refine depths).
`$PY lab/run_prop.py after_prop.json` → 0 rows flagged, over 21
constructions × {real, ieee} × {constrain, inert}. The `cond`-branch
construction is in both tables.

**C7 — MET.** `$PY lab/diff.py before.json after.json`:
`VERIFIED -> UNKNOWN: 44`, `RAISE -> UNKNOWN: 8`,
`REFUTED -> UNKNOWN: 4`, 28 rows unchanged;
`post-fix VERIFIED with a violating admitted point: []`;
`VERIFIED->non-VERIFIED moves that were SOUND before: []`;
`moves INTO VERIFIED: []`. Every one of the 44 was unsound. The 4
REFUTED → UNKNOWN rows (`A17_refuted_direction`) are the only genuine
capability loss: that refutation was SOUND, because a witness drawn from
the narrowed subset is still inside the (whole-box) admitted set, and it
is withheld because the drop machinery cannot yet tell an exact drop from
a widening one.

**C8 — MET.** `A16_false_unsat_alarm` and `A18_false_collapse_alarm`
raise `UnsatisfiableAssumptionError` pre-fix in all four configurations
and return UNKNOWN after. The oracle reports `admitted == 100006` for
both — the preconditions are satisfiable at every declared point, so both
alarms were false.

**C9 — MET.** `pytest -q -ra -p no:randomly` on this branch:
jax 0.11.0 `2235 passed, 2 skipped, 22 warnings in 164.09s`; jax 0.10.2
`2235 passed, 2 skipped, 22 warnings in 163.28s` (the two ran
concurrently; load average 0.55 / 2.91 / 3.05 at completion on 24 cores).
Baseline at `c20f38e` in the same worktree: `2151 passed, 2 skipped` on
both series (134.53s / 149.97s). `pytest --collect-only -q` id sets are
BYTE-IDENTICAL between the two series (2237 ids each, `diff` empty), and
against `c20f38e` (2153 ids, collected in a fresh worktree at
`/home/nick/MSF/.wt-sz0/BASE`) the branch adds 84 ids and removes none:
70 in `tests/test_size0_assume.py`, 14 in `tests/test_assume_constrain.py`.
The single red during development was
`test_committed_page_matches_live_registries`, the known line-number
artifact; `docs/gen_supported_primitives.py` regenerates it.

**C10 — MET.** `$PY lab/probe_ctt.py` with `PYTHONPATH` at
`/home/nick/MSF/.wt-sz0/TIP/src` (`6237e07`): `calls: 38`,
`size0 calls: 36`, every one `dtype bool`, shapes `(0,)`, `(0, 3)`,
`(2, 0)`, answer False. The refusal is REACHED, and the docstring's
argument for unreachability is false: broadcast forces the zero-element
shape onto the OUTPUT, not onto the siblings.

**C11 — PARTLY MET; the second half is FALSIFIED, and I registered it
wrong.** Deleting the gate (`box is None or not len(box.los)` →
`box is None`, in a copy at `/home/nick/MSF/.wt-sz0/tip_nogate`) flips
`A17_refuted_direction` from UNKNOWN to REFUTED in all four
configurations, and nothing else moves (84 rows, 80 unchanged) — so "no
outcome changes" is false and the gate IS load-bearing for a verdict. But
the flipped REFUTED is **SOUND**, not wrong: the oracle reports 50 267
violating admitted points of 100 006. It cannot be otherwise — a size-0
conjunct forces the whole predicate to `bool[0]`, which admits every
declared point, so any witness from a narrowed subset is admitted. The
audit's "flips a reachable UNKNOWN to a WRONG REFUTED" is therefore
CORRECT about the flip and WRONG about the wrongness. Also measured:
deleting the gate reddens 0 behavioural tests in the tip's suite (2184
passed; the one red, `test_the_probe_subprocess_imports_the_tree_under_test`,
asserts the imported tree is the worktree's `src` and is an artifact of
running from a copied tree — it fails with the copy and passes with
`TIP/src`).

**C12 — MET.** Instrumented full-suite run on `c20f38e`
(`pytest -q -ra -p no:randomly -p sz0probe`, plugin at
`lab/sz0probe.py`): `264` calls into `_apply_assumed_pred`, `size0: 0`,
shapes `()` ×197, `(2,)` ×43, `(3,)` ×20, `(1,)` ×2, `(5,)` ×2. The suite
has no size-0 assume in it; its size-0 declarations
(`tests/test_array_acceptance.py:461`, `tests/test_membership_idiom_hint.py:768`,
`tests/test_declaration_dtype.py`, `tests/test_exact_recording.py`) all
flow into `assert_` or a `jnp.all` reduction. The prior 93-call
instrument at the adjacent `_conjunct_certainly_true` site read the same
absence for the same reason: it measured the tests, not the method.

**The registered prediction — MET in full.** A1, A2, A4, A5, A6, A9, A10,
A11, A14, A19 became UNKNOWN; A16 and A18 stopped raising and became
UNKNOWN; A17's sound REFUTED became UNKNOWN; B1 and B2 stayed VERIFIED.
A13 (`cond`) was added after registration and also became UNKNOWN.

**Not registered, found while scoring.** `A8_rank1_vs_0x3` — a `(3,)`
conjunct against a `(0, 3)` sibling — was already UNKNOWN before the fix,
because jax inserts an explicit `broadcast_in_dim` for the rank promotion
and the recursion drops on the unrecognised producer. The class is
therefore bounded by what jax lowers without a rank change; the gate
covers both shapes regardless.
