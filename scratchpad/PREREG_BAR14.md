<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG_BAR14 — repair of `fix/value-zone-closed-under-call` @ `9fc44dd`

Written **before** any edit to `src/` or `tests/` on this branch
(`fix/predicate-zone-and-records`, off `9fc44dd`). Every clause below is scored
met / missed / not-run in the final report, including the ones that turn out to
be wrong. Nothing above the OUTCOMES rule is edited after this commit.

## Columns and baselines

**A** = full local env (`/home/nick/venvs/stelling-jax`, both solvers, jax,
maddening). **B** = CI's install set, `maddening` absent through a meta-path
finder that raises `ModuleNotFoundError`
(`scratchpad/R13plug/colb_block.py`, `-p colb_block`).

Re-derived here rather than taken from the audit, with
`scratchpad/BAR14run.sh` (which verifies `stelling.__file__` resolves inside
this worktree before every run):

| commit | A | B |
|---|---|---|
| `9fc44dd` (branch tip, the code being repaired) | **2068 p / 2 s** | **2064 p / 6 s** |
| `faefc48` (base, per PREREG_BAR13) | 2064 p / 2 | 2060 p / 6 |

A mutant counts as a kill only if it is first shown **live at `faefc48`** (full
suite byte-identical to unmutated in both columns) and then RED after the
repair. `0 RED` on a mutant that was never live measures nothing. Where a
mutant is additionally claimed green at `9fc44dd`, that is measured too, since
`9fc44dd` is the build whose rules are being repaired.

---

## P1 — the predicate axis: the zone is handed its own constants

**Hypothesis.** The source pin forbids the zone to SPELL a constant, and then
hands it four enumerated immutable ones. Every string a conjunct needs for the
keys that matter is therefore already in scope under a permitted name, so the
literal rule is near-vacuous for exactly those keys. `M10both` —

```python
if set(_EVIDENCE_BUDGET_KEYS) <= set(recorded):
    return True
```

— three lines inside `_evidence_reproduces`, is invisible to every rule: no
literal, no comparison against a literal, no module-level mutable, no helper,
no default argument, no import, no `global`, no smuggler, and no method call on
a recorded value.

**Nearest plausible neighbour to distinguish from.** "The disclosed residue is
a method call on a recorded value, and that is the whole of it." `M10both`
distinguishes them: it contains no method call on a recorded value at all, and
is caught by no rule. The class is *any predicate over a recorded value whose
constants come from `_EVIDENCE_*_KEYS`*, of which the method-call spelling is
one instance.

Mutants, each to be shown live at `faefc48` and green at `9fc44dd` first:

* **M10both** the three lines above, inside `_evidence_reproduces`.
* **M6v2** the same predicate in a **literal-free nested `def`** inside
  `_evidence_reproduces`, so nothing new appears at module scope.
* **M11weak** (class control, not on the audit's list) `_whitelisted` narrowed
  by its own permitted constant so that the comparison stops covering the
  budget keys — a WEAKENING of the equality rather than an early return, in the
  one function both sides call, spelled with no literal.

**Repair to be judged.** Two general rules, neither written for a mutant:

1. **A CONSTANT READER LEDGER.** Each enumerated module constant is read by
   exactly ONE named function in the closure, asserted in both directions —
   nothing else may read it, and every enumerated pair must actually happen.
   This is the `_ALLOWED_READS` idea (a reader ledger for record ATTRIBUTES)
   applied to the module CONSTANTS the closure walk already derives. After it,
   a predicate written at the meeting point has no constant available at all.
2. **A MEETING-POINT SHAPE PIN.** `_evidence_reproduces` is the only function
   in which a recorded value and the re-derivation are both in scope. Its body
   is pinned structurally: no branch, no loop, no nested definition, no
   comprehension, exactly one `return`, whose expression is a two-term `and` of
   a `bool(...)` call and an `==`.

**Clauses.**
- **P1.a** M10both, M6v2, M11weak each live at `faefc48` (0 RED, both columns
  byte-identical to that baseline) and 0 RED at `9fc44dd` in column A.
- **P1.b** each RED after the repair, in both columns.
- **P1.c** the class rather than the spellings: M10both and M6v2 are RED to
  **both** new rules independently (checked by disabling one and re-running),
  and M11weak is RED to the constant ledger — no rule is written per mutant.
- **P1.d** all six mutants published at `faefc48` (M1, M2, M3, M4, M5, M6)
  are still RED at the repaired tip.
- **P1.e** the honest zone is green: no rule has to be relaxed to admit it, and
  if one does, that is reported as a weakening rather than hidden.
- **P1.f** the residue that remains after both rules is stated, with the
  spelling that reaches it, in `verdict.py` and in `SOUNDNESS.md`.

## P2 — the decision switches the literal rule off

**Hypothesis.** The decision loop filters `if rule != "literal"`, so a string
constant inside any call matches nothing there. `M9X` —
`if str(closed).count("stelling_backdoor"): return ((), '')` at the top of
`_bar_scope` — is live and lifts the scatter bar for any query whose text
carries the marker. The same exemption makes
`__import__("stelling.obligation")` invisible to the import allow-list, because
`__import__` produces no `ast.Import` node and is not in `_SMUGGLERS`.

**Nearest plausible neighbour.** "The decision needs the exemption because it
carries message text." Measured first: the decision's string constants are 16
prose fragments, `''`, and 4 attribute names (`eqns`, `jaxpr`, `invoked`,
`False`); it spells **no number at all**. So the exemption can be narrowed to
prose without touching the honest code.

**Repair to be judged.** The blanket `rule != "literal"` filter is removed and
replaced by:
- the decision may spell **no numeric literal**, same as the zone;
- a string literal in the decision must be in a PROSE position (an f-string
  part, an operand of a string `+`, the receiver of `"sep".join(...)`, the
  empty string) or an ENUMERATED attribute name;
- a new rule `call-literal`, applying to zone AND decision: a literal handed to
  a call whose function is an attribute of a non-literal receiver —
  `str(closed).count("…")`, `recorded.get(k).startswith("…")`. This also closes
  part of the method-call residue the branch disclosed;
- a new rule `dynamic`: `__import__`, `eval`, `exec`, `compile`, `importlib`.

**Clauses.**
- **P2.a** M9X live at `faefc48` and 0 RED at `9fc44dd`; RED after the repair.
- **P2.b** `M9imp` (`__import__("stelling.obligation")` in `_bar_scope`) live
  at `faefc48`; RED after the repair.
- **P2.c** the honest decision is green with the blanket exemption gone, and
  the remaining exemption (prose) is stated exactly, with what it still admits.

## P3 — a stated rule that is false as implemented

**Hypothesis.** `_closure_offences` never applies the immutability branch to
names whose `__module__` is `stelling.verdict` — i.e. to the zone's own
function objects, which are mutable. `M7inert`
(`_whitelisted.__kwdefaults__ = out`) is green and no source rule sees it.
Predicted: the same carrier, used to actually AIM across the two `_whitelisted`
calls (`M7live`), is also green and mints.

Also: `_IMMUTABLE` includes `tuple`, checked SHALLOW only.

**Repair to be judged.** The rule is made true as written: a name in the
closure defined by the module must be a plain function carrying no mutable
state of its own (`__defaults__`, `__kwdefaults__`, a non-empty `__dict__`),
and the immutability check on enumerated constants recurses into tuples and
frozensets. A member of the zone that is not a function at all (a class, whose
methods the `__code__` walk skips) is an offence rather than a silent pass.

**Clauses.**
- **P3.a** M7inert and M7live live at `faefc48`, M7live shown to MINT on the
  mispaired pair; both RED after the repair.
- **P3.b** the deep-immutability rule is shown non-vacuous against a synthetic
  `("a", ["b"])` constant, and the shallow version shown to pass it.
- **P3.c** where the walk stops is named in the docstring rather than left
  implicit: attribute/method dispatch, decorators, objects with no `__code__`.

## P4 — SOUNDNESS.md states a stronger claim than the prereg scored

`SOUNDNESS.md`: *"All six mutants are RED, each to one of those three general
rules and none to a rule written for it."* Predicted false on two counts —
M2 and M6 die to the **defaults** rule, a fourth rule added for M2; M4 dies to
`test_the_stamps_own_derivation_is_the_HONEST_one`, a test that exists only
because of M4. `PREREG_BAR13.md` scores this as **P1.c partial**.

- **P4.a** which rule kills each of the six is re-derived by disabling rules
  one at a time, and published.
- **P4.b** the published document is made to match the prereg's scoring.

## P5 — the sweep was never expensive, and neither test makes the strongest statement

- **P5.a** re-run the pre-registered exhaustive sweep (`1..60000`) and publish
  its cost **with a load average**, against the "it would be expensive"
  justification for substituting a 12-point sample.
- **P5.b** add the structural pin that is strictly stronger than either:
  `inspect.signature(slice_fingerprint)` takes the slice and nothing else, so
  the budget is not an argument. One line, no sampling.
- **P5.c** keep the neighbour-pair half (a real addition the sweep did not
  have) and state the substitution's standing accurately: stronger in the
  generality of its argument, WEAKER in the sample supporting its premise.
- **P5.d** the `True` case is stated accurately: `_evidence_budget` cannot
  return a bool (`int(text)` never yields one — to be measured), and
  `emit(sl, "z3", True)` emits a script no solver accepts, so reaching it needs
  `int()` itself corrupted.

## P6 — the same grep blindness, three lines above the correction

`tests/test_verified_bar.py:2828` — *"neither are 2000, 15000, 25000, 50000 or
60000"* — read as a claim about what the suite samples. Predicted: `60_000` is
a live solver budget at 8 sites and `60000` is in the fingerprint sweep's own
tuple; `_CALLER_BUDGETS` ends `2 ** 31 - 1`, invisible to a grep for
`2147483647`.

- **P6.a** the eight `60_000` sites are enumerated by file and line.
- **P6.b** the sentence is corrected, and the correction is PINNED by a test
  that reads numeric literals out of the AST — underscores, exponents and
  computed forms included — rather than grepping digits, so the same blindness
  cannot recur silently.
- **P6.c** the tree-wide form census is published with its command: how many
  numeric instances defeat a digit-grep for their own value, and the split
  between underscore, exponent, computed and radix forms.

## P7 — the census claims

- **P7.a** re-derive the eleven `raise _Decline` sites and whether
  `grep -c 'raise _Decline'` agrees over the same range; if it does, the "four
  split across string literals or comment lines" sentence is not about them.
- **P7.b** re-derive how many of the census restatement sentences are actually
  SPLIT. Predicted **three**, against `_flat`'s "Four of the five".
- **P7.c** the flattener is shown load-bearing by measurement (number-words
  read with it vs without) rather than by the sentence that is wrong.
- **P7.d** the ninth restatement (`test_scatter_gauge_jax.py:1689`, "The
  undriven six") is shown 0 RED under perturbation and then READ by
  `_CENSUS_PHRASES`.
- **P7.e** the shipped self-contradiction — line 2284 restating the count as
  "eight" while 2298 asserts the count is no longer restated anywhere — is
  resolved, and which way is argued.
- **P7.f** restatements the checker structurally cannot see (`SOUNDNESS.md`,
  which is not the file it reads) are named as out of scope rather than left
  to look covered.

## P8 — residues to close or disclose

- **P8.a** the citation resolver's semantic is stated as what it is (a
  `FunctionDef` of that name exists in the AST), and the shapes for which that
  is strictly weaker than "the cited test runs" are enumerated. The row the
  branch itself accepted — a test inside a non-`Test*` class, with no
  `python_classes` override in `pyproject.toml` — is corrected or disclosed,
  with the check for whether any real citation depends on it.
- **P8.b** P6's gauge behaviour (`docs/gauge-coverage.md`: the `m-assembly`
  fixture DECLINES at its declared shapes, the rank-1 flattening is admitted)
  gets a test, or is disclosed as unpinned.
- **P8.c** the `scratchpad` `WITHHELD` entry's unstated consequence is stated:
  `test_no_untracked_file_anywhere_would_ship` skips on
  `path.split("/", 1)[0] in WITHHELD`, so the entry exempts the entire
  `scratchpad/` subtree from the untracked-file check.
- **P8.d** the numeric-multiset figures are re-derived and the RANGE is named
  in both places in `SOUNDNESS.md`, not only for the test-id correction. The
  per-file "2 removed" is explained rather than smoothed over.
- **P8.e** the named-risk list is extended with the axis this repair is about:
  closing the CALL axis left the PREDICATE axis open precisely because the zone
  is handed its constants.

## Global

- **G1** both columns green at the end, counts published with the collected-id
  diff against `9fc44dd`, and the delta accounted for test by test.
- **G2** no numeric constant changed in `src/`; no install; nothing upstream;
  no MADDENING/MIME change; `docs/norms.md` untouched; the scatter bar not
  lifted.
- **G3** every number in the final report carries the command that produced it;
  anything reasoned but not run is labelled SUSPECTED.
- **G4** no wall figure published without a load average.

## Named ways this could come out wrong

* the constant reader ledger may forbid an honest read the zone needs — if a
  constant turns out to have two legitimate readers, the ledger grows and that
  is reported as a weakening of the rule, not hidden;
* the meeting-point shape pin is TOTAL over one function and therefore reddens
  on honest edits; if that cost is judged too high the pin is dropped and the
  fact recorded, not quietly narrowed;
* removing the decision's literal exemption may flag prose the honest code
  needs; every exemption added back is named, argued and counted;
* the citation resolver tightening may break a real citation that points at a
  test defined inside a class — measured before it is made;
* `M11weak` may not actually mint (the derived-key gate may fail closed first);
  if so it is reported as a control that did not fire rather than as a kill.

---

# OUTCOMES

*(appended below this rule only; nothing above it is edited)*

## Columns at the repaired tip

| | A | B |
|---|---|---|
| `9fc44dd` | 2068 p / 2 s | 2064 p / 6 s |
| this branch's tip | **2070 p / 2 s** | **2066 p / 6 s** |

Collected-id diff against `9fc44dd`: **+2, 0 removed**, in BOTH columns —
`tests/test_verified_bar.py::test_the_claim_about_unsampled_budgets_reads_the_AST`
and
`tests/test_scatter_gauge_jax.py::test_the_m_assembly_fixture_DECLINES_at_its_own_declared_shapes`.
The A and B id lists are byte-identical at the tip (`diff -q`, 2072 ids each).

## Mutants: live at `faefc48`, 0 RED at `9fc44dd`, RED here

Full suite, both columns, at `faefc48`; full suite column A at `9fc44dd`;
`tests/test_verified_bar.py` in both columns at this tip.

| mutant | `faefc48` A / B | `9fc44dd` A | mints? | here |
|---|---|---|---|---|
| `M10both` predicate on `_EVIDENCE_BUDGET_KEYS` in `_evidence_reproduces` | 2064/2 · 2060/6 | 2068/2 | **yes** | 1 RED |
| `M6v2` same, in a literal-free nested `def` | 2064/2 · 2060/6 | 2068/2 | **yes** | 1 RED |
| `M9X` `str(closed).count("stelling_backdoor")` in `_bar_scope` | 2064/2 · 2060/6 | 2068/2 | not through the public API | 1 RED |
| `M9mint` per-stamp, two literals inside calls in `_bar_scope` | 2064/2 · 2060/6 | 2068/2 | **yes** | 1 RED |
| `M9imp` `__import__("stelling.obligation")` in `_bar_scope` | 2064/2 · 2060/6 | 2068/2 | inert (channel) | 1 RED |
| `M7inert` `_whitelisted.__kwdefaults__ = out` | 2064/2 · 2060/6 | 2068/2 | inert (carrier) | 2 RED |
| `M7live` the same carrier, used across the two calls | 2064/2 · 2060/6 | 2068/2 | **yes** | 2 RED |
| `M11weak` (control) `_whitelisted` narrowed by its own constant | **15 RED** | — | — | 16 RED |

Mint probe (`JAX_ENABLE_X64=1`, mispaired pair, one added
`(":tlimit", "20000")` on the `z3` stamp), `_bar_scope`'s barred tuple:

```
                              faefc48   M10both  M6v2   M7live  M9mint
honest pairing, honest stamp  scatter   scatter  scatter scatter scatter
honest pairing, FORGED stamp  scatter   scatter  scatter scatter ()
MISPAIRED,      honest stamp  scatter   scatter  scatter scatter scatter
MISPAIRED,      FORGED stamp  scatter   ()       ()      ()      ()
```

The six published mutants (M1–M6) are all still RED at this tip:
1, 2, 2, 3, 1, 2 failures respectively over
`tests/test_verified_bar.py tests/test_smt_emission.py`.

---

# OUTCOMES

## P1 — the predicate axis — **MET**

- **P1.a MET.** M10both, M6v2 live at `faefc48` in both columns
  (2064 p / 2 s, 2060 p / 6 s — byte-identical to that baseline) and
  2068 p / 2 s at `9fc44dd`. **M11weak MISSED as pre-registered risk**: it is
  **15 RED at `faefc48`**, not live — the control did not fire, and it is
  reported as that rather than as a kill. It is 16 RED here.
- **P1.b MET.** Both RED in both columns
  (`::test_the_value_zone_is_CLOSED_UNDER_CALL`).
- **P1.c MET, and measured rather than argued.** Rules 4 and 5 are
  INDEPENDENT: with rule 4 stubbed out, both mutants are RED to the shape pin
  ("`_evidence_reproduces` now contains ['If']" / "['FunctionDef', 'If']");
  with rule 5 stubbed out, both are RED to the constant ledger
  ("[('_EVIDENCE_BUDGET_KEYS', '_evidence_reproduces')] read an enumerated
  constant that is not that function's business").
- **P1.d MET.** M1–M6 all RED at this tip.
- **P1.e MET.** The honest zone is green with no rule relaxed. **One source
  line moved to admit it, and it is reported as the cost:** `_bar_scope`'s
  only fallback reason that interpolates nothing is hoisted to
  `_BAR_UNDERIVABLE`, a module constant with a ledger entry. The rendered
  message is byte-identical.
- **P1.f MET.** The residue is stated in `verdict.py`, in the test docstring
  and in `SOUNDNESS.md`: a predicate needing NO constant at all
  (`len(recorded) > len(reproduced)`, `sorted(...).pop()`).

## P2 — the decision's literal exemption — **MET**

- **P2.a MET.** M9X live at `faefc48` both columns, 0 RED at `9fc44dd`, 1 RED
  here. **Exploitability qualified**: the marker has to reach `str(closed)`,
  and `any_array` takes `(shape, dtype, bounds)` with no name, so the
  transcribed `ClosedJaxpr` repr carries no caller string. M9X is CONFIRMED
  uncaught and SUSPECTED-not-exploitable through the public API. `M9mint` was
  added for exactly that reason and DOES mint.
- **P2.b MET.** M9imp live at `faefc48` both columns, 0 RED at `9fc44dd`,
  1 RED here (`dynamic`).
- **P2.c MET.** The honest decision is green. The remaining relaxation is
  stated exactly: a string literal in a PROSE POSITION (f-string part, `+`
  operand, `"sep".join` receiver, the empty string) or an enumerated attribute
  name; and a literal handed to a call on a bare `Name`, which is what
  `fallback("…")` is, disclosed with the argument for why the callee cannot
  turn it into a discriminator without a `Compare` or a method call that IS
  caught. The decision spells no number at all, asserted.

## P3 — the immutability rule — **MET**

- **P3.a MET.** M7inert and M7live live at `faefc48` both columns and 0 RED at
  `9fc44dd`; M7live MINTS (`('scatter',)` → `()`); both 2 RED here.
- **P3.b MET.** `("a", [])` passes `isinstance(obj, _IMMUTABLE)` and is caught
  as `mutable` by the deep check — both directions asserted in the test.
- **P3.c MET.** Attribute/method dispatch, decorators and objects with no
  `__code__` are named in `_closure_offences`' docstring, and the last one is
  now an offence (`not-a-function`) rather than a silent pass.

## P4 — SOUNDNESS.md's six-mutant claim — **MET**

Re-derived at `9fc44dd`, per mutant, by reading which assertion fires:

| mutant | kills at `9fc44dd` |
|---|---|
| M1 module list | 6 tests incl. both source pins |
| **M2 default argument** | **1 RED, the DEFAULTS rule alone** |
| M3 seventh helper | both source pins |
| **M4 `Script.stamp_options`** | **3 RED incl. `test_the_stamps_own_derivation_is_the_HONEST_one`** |
| M5 module dict | `::test_the_value_zone_is_CLOSED_UNDER_CALL` |
| **M6 mutable default on `_whitelisted`** | the DEFAULTS rule (plus behavioural) |

The defaults rule is a FOURTH rule, added for M2. The document now says three
of six die to the three general rules, matching `PREREG_BAR13.md`'s P1.c
partial.

## P5 — the sweep and the strongest statement — **MET**

- **P5.a MET.** `1..60000` exhaustive: **9.5 s at load average 6.00**
  (`cat /proc/loadavg` before and after: `6.00 7.35 10.10` → `9.30 8.02
  10.29`). distinct `slice_sha256` **1**, distinct `smt2_sha256` **60000**,
  empty **0**, equal-to-neighbour **0**; neighbour `smt2 ==` True, neighbour
  `slice !=` True. "It would be expensive" does not survive.
- **P5.b MET.** `inspect.signature(slice_fingerprint)` is `(sl) -> 'str'`;
  pinned as `list(parameters) == ["sl"]`.
- **P5.c MET.** The neighbour-pair half is kept; the substitution is described
  as stronger in argument and weaker in sample.
- **P5.d MET.** `emit(sl, "z3", True)` emits `(set-option :timeout True)`,
  measured; `_evidence_budget({':timeout': True})` returns `1`, an `int`, and
  `{type(int(s)) is bool for s in ('0','1','2','-1')} == {False}`.

## P6 — the 60000 blindness — **MET, and worse than reported**

- **P6.a MET.** Eight `60_000` sites: `test_dropped_assume.py:65,104,120,127`,
  `test_membership_idiom_hint.py:864,866`, `test_reproduce_acceptance.py:70`,
  `test_square_acceptance_jaxfluids.py:62`.
- **P6.b MET.** The sentence is corrected and PINNED by
  `test_the_claim_about_unsampled_budgets_reads_the_AST`, which folds
  underscore, exponent and computed forms. **Beyond the audit: FOUR of the six
  values the sentence lists are live solver budgets** — 2000 (~20 sites),
  15000 (`test_array_emission.py:1391`), 30000, 60000. Only 25000 and 50000
  are genuinely undriven, and 50000 appears in the tree only as fixture
  arithmetic (`… * 100000 + 50000`), which is why the census reads budget
  POSITIONS.
- **P6.c MET.** `src/` + `tests/`: **542 numeric instances defeat a digit-grep
  for their own value**, 136 distinct forms — **331 computed, 157 exponent,
  53 underscore, 1 other, 0 radix**. (The audit's 533 / 59 / 325 / 269 is the
  same shape under a different counting convention; mine is published with its
  script, `scratchpad/BAR14_litcensus.py` in the work directory.) The
  `2 ** 31 - 1` case is named in the comment.

## P7 — the census claims — **MET, with one NOT REPRODUCED**

- **P7.a MET.** AST count over `_scatter_set_plan` (obligation.py 1098–1185):
  **11** `raise _Decline`; `grep -c` over the same range: **11**. So "four
  split across string literals or comment lines" is not about the raise sites.
- **P7.b MET.** Measured **three**, not four: pattern
  `the other (\w+) are not driven here` (1 occurrence) and
  `has (\w+) \`raise _decline\` sites` (2 occurrences) have raw-source match
  counts of 0 against flattened counts of 1 and 2.
- **P7.c MET.** 13 number-words read with the flattener, 10 without; removing
  it hard-fails `test_the_censuss_own_prose_agrees_with_the_counts_it_
  describes`.
- **P7.d MET.** `tests/test_scatter_gauge_jax.py:1689` perturbed to "seven" is
  **0 RED** at this branch's tip (whole gauge file 14 passed); with the ninth
  pattern added it is **1 RED** and the honest file is 15 passed.
- **P7.e MET.** Line 2284's "finds eight" is elided; line 2298's claim that
  the count is restated nowhere is kept, and the contradiction is written down
  rather than quietly removed.
- **P7.f MET.** `SOUNDNESS.md` is out of the checker's scope by construction
  (`Path(__file__)`), stated in the docstring with the reason a `*.md` scan is
  not added in this pass.
- **NOT REPRODUCED:** the audit's *"`_CENSUS_PHRASES` lives in
  `test_scatter_gauge_jax.py`, not `test_prose_hygiene.py`"*.
  `grep -rn "CENSUS_PHRASES" --include='*.md' .` outside `scratchpad/` returns
  nothing, and no file in the tree attributes it to `test_prose_hygiene.py`.
  Reported as not found rather than silently dropped.

## P8 — residues — **MET**

- **P8.a MET.** Two of the seven closed (nested in a function; inside a
  non-`Test*` class — the branch's own accepted row, corrected to `False`,
  with `python_classes` confirmed absent from `pyproject.toml`). All **13**
  in-tree citations resolve at module level, checked before the change, so
  nothing depended on the loose reading. The other five are asserted rows.
- **P8.b MET.** `test_the_m_assembly_fixture_DECLINES_at_its_own_declared_
  shapes` pins both halves; the decline reason quotes `(2, 2, 2)` and
  `3 element(s)`, and the rank-1 flattening is admitted.
- **P8.c MET.** Written into the `WITHHELD` entry itself.
- **P8.d MET.** `3e107cf..faefc48`: tests 112 added / 0 removed (105 int,
  7 float), src 5 / 0. `faefc48..9fc44dd`: tests 34 / 0, src 4 / 2, both
  removals the literal `0`. Tree-wide `src/` over `faefc48..HEAD`: 192
  module-level assignments, 31 numeric-bearing, **0 changed / 0 added /
  0 removed**. Both places in `SOUNDNESS.md` now name their range.
- **P8.e MET.** The named-risk list gains the predicate axis.

## Global

- **G1 MET.** A 2070/2, B 2066/6; id diff +2 / 0 in both columns, lists
  byte-identical; both additions accounted for by name above.
- **G2 MET.** No numeric constant changed in `src/` (0/0/0 tree-wide); no
  install; nothing upstream; no MADDENING/MIME change; `docs/norms.md`
  untouched; the scatter bar unchanged (`VERIFIED_BARRED_PRIMITIVES` is
  byte-identical and the mint probe's honest rows all read `('scatter',)`).
- **G3 MET.** Every figure above carries its command or its run.
- **G4 MET.** The one wall figure (9.5 s) carries its load average; the suite
  timings are in `scratchpad/BAR14_*` logs with `loadavg` recorded per run.

## Named ways this came out wrong

* the constant ledger did NOT have to grow for an honest read — the five pairs
  are exactly what the closure makes;
* the meeting-point shape pin is total over one function and will redden on
  honest edits; the cost is accepted and the assertion says so;
* removing the decision's literal exemption DID flag one honest literal, and
  the fix was a source change (one hoisted message constant) rather than a
  widened exemption. That is one exemption's worth of cost, named;
* the citation tightening broke nothing: all 13 citations were already at
  module level;
* **`M11weak` did not mint** — 15 RED at `faefc48`. The derived-key gate in
  `_reproduced_evidence` is not what stops it; the behavioural bar tests are.
  Reported as a control that did not fire.
