<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG_BAR13 — repair of `fix/value-channel-canonical` @ `faefc48`

Written **before** any edit to `src/` or `tests/` on this branch
(`fix/value-zone-closed-under-call`, off `faefc48`). Every clause below is
scored met / missed / not-run in the final report, including the ones that
turn out to be wrong.

Columns. **A** = full local env (`/home/nick/venvs/stelling-jax`, both solvers,
jax, maddening). **B** = CI's install set, `maddening` made absent as a missing
distribution is absent (`ModuleNotFoundError` from a meta-path finder,
`scratchpad/R13plug/colb_block.py`, `-p colb_block`). Baseline at `faefc48`,
re-derived here rather than taken from the audit: **A 2064 p / 2 s,
B 2060 p / 6 s**. Columns are serialized per worktree; every run verifies
`stelling.__file__` resolves inside its own worktree first.

A mutant counts as a kill only if it is first shown **live** at `faefc48`
(full suite byte-identical to unmutated in both columns) and then RED after
the repair. `0 RED` on a mutant that was never live measures nothing.

---

## P1 — the value zone is not closed under call

**Hypothesis.** `_whitelisted` (`verdict.py:802`) is reachable from both halves
of the comparison, is in no enumeration anywhere, and can therefore carry the
aiming the two pinned signatures were supposed to forbid. The defect is not
four spellings; it is that the zone **may call out of itself**.

**Nearest plausible neighbour to distinguish from.** "The signature pins are
enough" — i.e. `_evidence_options(stamp)` / `_reproduced_evidence(sliced,
flavour, budget)` unchanged means neither side can aim. M1 is exactly the
measurement that separates these: it touches **neither signature**.

Mutants, each to be shown live at `faefc48` first:

* **M1** module-level mutable `_LAST` + four lines inside `_whitelisted`:
  stash the record's projection on call 1, return it on call 2. No `global`
  statement, no literal, no comparison, no signature change.
* **M2** `def _evidence_reproduces(sliced, stamp, _m=("30000",))` — the
  discriminator in a **default argument**, which `_fn_body_ast` never walks
  (`tree.body[0].body` skips `args`).
* **M3** a **seventh helper** `_budget_ok(recorded)` defined at module scope
  and called from the zone; the pin enumerates six names.
* **M4** a conjunct inside `stelling.smt.Script.stamp_options`. Predicted to be
  live because `stamp_options` runs **after** emission and contributes nothing
  to the emitted text — so the byte-level emission tests cannot see it, and
  `test_the_reproduction_comes_from_the_stamps_own_derivation` only checks that
  substituting it MOVES the answer.
* **M5** (class control, not on the audit's list) a module-level `dict`
  written in `_evidence_options` and read in `_reproduced_evidence` — a
  spelling none of M1–M4 uses. If the repair only closes the four, M5 lives.
* **M6** (class control) a **nested** `def` inside `_evidence_reproduces` doing
  the aiming, so nothing new appears at module scope.

**Repair to be judged.** The zone is made **closed under call**: every callable
reachable from a zone function is itself in the zone and parsed by the pin, and
every free name the zone reads that is not a parameter, a local or a builtin
must be an explicitly enumerated module-level object which is **immutable**.
Default arguments are read by the pin (and forbidden outright in the zone).

**Clauses.**
- **P1.a** M1, M2, M3, M5, M6 each live at `faefc48` (0 RED, both columns
  byte-identical to baseline) and each RED after the repair.
- **P1.b** M4 live at `faefc48` with `tests/test_smt_emission.py` and
  `tests/test_verified_bar.py` both fully green, and RED after the repair.
- **P1.c** the class is closed rather than the spellings: M5 and M6 are RED
  **for the same rule** that kills M1/M3, not by a rule written per mutant.
- **P1.d** `Script.stamp_options`' **honest output** is pinned — an exact,
  independently derived expectation, not "substituting it moves the answer".
  The false claim that corrupting it corrupts EMISSION is corrected in
  `verdict.py`'s prose.
- **P1.e** `_evidence_budget`'s docstring is restated to the mechanism that
  actually bounds it. Predicted mechanism: **`slice_sha256` is
  budget-invariant**, so no value this function returns can make a reproduction
  of THIS slice equal a record about a DIFFERENT slice; the recorded-budget
  equality argument constrains only `:timeout`/`:tlimit` self-consistency and
  is an argument about an HONEST record. To be measured, not asserted:
  (i) `slice_sha256` constant over a spread of budgets including `True`;
  (ii) `smt2_sha256` moves with the budget; (iii) no budget in `1..60000`
  makes this slice's reproduction equal the neighbour record.
- **P1.f** the disclosed residue's stated corroboration is corrected: M2 and M3
  keyed at `:timeout == 30000` are 0 RED **including** the budget sweep.

## P2 — the gate's justification versus its residue

**Hypothesis.** The strict-subset residue is soundness-harmless, but on a
scatter-free query it emits **verbatim** the misattributing note the gate's own
comment calls "worse than silence".

**Clauses.**
- **P2.a** reproduce it at `faefc48`: scatter-free query, `records` whose first
  pass is a non-empty strict subset — honest verdict VERIFIED, observed UNKNOWN
  carrying "the propagated interval straddling the asserted bound".
- **P2.b** after the repair the same query no longer misattributes: either the
  note is suppressed/replaced when the cause is a degenerate `records`, or the
  gate's justification is restated to match what it delivers. Whichever is
  taken is stated, with the reason the other was not.
- **P2.c** no verdict that was VERIFIED becomes non-VERIFIED and no `unknown`
  becomes decided: the change is to the NOTE, not to the bar.

## P3 — the citation check is falsely satisfiable

**Hypothesis.** `f"def {name}(" in body` is a raw substring test over file text,
so a comment or a string literal satisfies it.

**Clauses.**
- **P3.a** reproduce all four rows of the audit's table at `faefc48`
  (renamed-away → 1 failed; comment left → 1 passed; exact citation's `def`
  commented out → 1 passed; string-literal mention only → 1 passed).
- **P3.b** after moving to `ast.parse` + `FunctionDef` names, rows 2–4 are RED
  and row 1 stays RED, with the check still independent of collection (no
  import of the test module, so the zero-dep column is unaffected).

## P4 — the census prose carries more restatements than it reads

**Hypothesis.** `_CENSUS_PHRASES` reads three sentences; at least four more
restate the same three counts and are 0 RED. The anti-vacuity loop cannot see
the gap because `text.replace(right, wrong)` perturbs **every** occurrence at
once.

**Clauses.**
- **P4.a** measure the gap at `faefc48`: for each of the four named unread
  restatements (`the admission gate drives FIVE of them`, `The other six are
  NOT driven here`, `the row's eleven decline sites`, `f"says six over five"`),
  perturb **that occurrence alone** and record RED/green.
- **P4.b** after the repair each of those four, perturbed alone, is RED.
- **P4.c** the anti-vacuity control is per-occurrence rather than global, so a
  phrase in the list that stopped being read is caught.
- **P4.d** the docstring's "four places for three quantities" is derived or
  corrected; no restatement of the count of restatements is left unread.

## P5 — record corrections

- **P5.a** re-derive the added/removed test-id diff `3e107cf..faefc48`.
  Predicted **10 added / 1 removed** (removed:
  `test_a_TWO_FACED_records_cannot_show_the_bar_one_thing_and_the_loop_ANOTHER`),
  against the branch's stated "+9, 0 removed".
- **P5.b** re-derive the per-file integer multisets across all four commits.
  Predicted **0 removed / 112 added in `tests/`, 0 removed / 5 added in
  `src/`** — zero integers changed value anywhere.
- **P5.c** of the ten added tests, the three named as not surviving contact are
  addressed: `test_the_evidence_path_cannot_name_a_VALUE` (P1),
  `test_every_test_cited_in_core_prose_still_exists` (P3), and
  `test_the_reproduction_is_handed_no_record` — whose **property** is false as
  offered; its docstring is restated to what the signature pin actually buys.

## P6 — the capability claim is scoped to `tests/`

- **P6.a** `SOUNDNESS.md`'s "Blast radius inside this repository is ZERO,
  measured rather than assumed" on the column-bound entry is scoped to what the
  census measured (the pytest-driven tree), and names the pattern most likely
  to meet the bound: an unflattened multi-index `segment_sum` normal-matrix
  assembly, which the gauge file's own header already names as a target and
  which is posed in-tree on shape `(3,)` flattened to rank 1.
- **P6.b** `corpus/run_census.py` is checked for whether it reaches the row,
  and the answer is stated either way rather than assumed.

## Global

- **G1** both columns green at the end, counts published: A and B, with the
  delta from 2064/2 and 2060/6 accounted for test by test.
- **G2** no numeric constant changed anywhere; no install; nothing upstream;
  no MADDENING/MIME change; the scatter bar not lifted.
- **G3** every number in the final report carries the command that produced
  it; anything reasoned but not run is labelled SUSPECTED.
- **G4** wall-clock figures, if published, carry a load number.

## Named ways this could come out wrong

* the closure rule may forbid an honest call the zone needs (`emit`,
  `dict`, `getattr`) — if the allow-list has to grow to admit them, that is
  reported as a weakening, not hidden;
* M4 may turn out not to be live (i.e. the emission tests do see it), which
  would make the audit's claim false and is reported as such;
* the per-occurrence census control may be too strong and flag deliberately
  frozen predecessor sentences; any exemption is named and argued, and the
  count of exemptions is published;
* P2's suppression may cost a note some existing test asserts on; if so the
  restatement branch is taken instead and the reason recorded.
