<!--
SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
SPDX-License-Identifier: Apache-2.0
-->

# PREREG — SIR6 repair pass on `fix/skip-inventory-repair-5` @ `1b1c843`

Written BEFORE any measurement on this branch and before any edit to
`tests/` other than the `WITHHELD` entry that brings this file into the
tree (clause R5b, which is itself registered below). Branch
`fix/skip-inventory-repair-6`, worktree
`/tmp/claude-1000/-home-nick-MSF-stelling/e443f4ad-79d3-43d4-bc29-87a910409ae6/scratchpad/SIR6_WT`,
base `1b1c843`, pristine comparison worktree `SIR6_a80d60c` at `a80d60c`.

## Lanes and the trap

* `PY=/home/nick/venvs/stelling-jax/bin/python` (bare `python` is not on
  PATH), `JAX_PLATFORMS=cpu`, `PYTHONPATH=<worktree>/src`, and
  `$PY -c "import stelling; print(stelling.__file__)"` verified to resolve
  INSIDE the worktree before every comparison.
* Both cwds: repo root and `tests/`. Ids normalised by stripping a leading
  `tests/` before comparison, because a run from `tests/` keeps the
  `tests/` prefix (rootdir is the repo either way).
* Work IN PLACE with `git checkout --` restore. No `cp -a` copies: a copied
  worktree carries stale `__pycache__` whose `co_filename` points back at
  the source tree, and a missing `.git` adds a `test_sdist_contents` skip.
* **No wall-clock figure without the load average beside it.**
* Measured baseline at `1b1c843`, both cwds, `-p no:randomly`:
  **2027 passed, 2 skipped, EXIT 0**; 2029 collected; collected-id sets
  md5-identical across cwds after normalisation
  (`d55b5144f01aed92054730537dd2d5da`).

## Clauses — each is a prediction, met or missed in the report

### R1 — the anchor moves off `pytest_sessionfinish`

1. **R1a** At `1b1c843`, with an undisclosed skip planted, all four of
   these are driven and recorded: no plugin; `-p exit0finish` (a plugin
   raising `pytest.exit(returncode=0)` from its own `pytest_sessionfinish`);
   `-p exit0finish -p unconf` (the same plus one that attacks
   `pytest_unconfigure`); `-p sf_tryfirst` (an ordinary `tryfirst`
   `pytest_sessionfinish` that exits zero). EXIT code and banner count for
   each.
2. **R1b** After the repair, every one of those four arrives **non-zero**
   and prints a banner naming the shortfall. Where the arriving code is not
   1, the actual code is reported and explained rather than rounded to
   "non-zero".
3. **R1c** The four named aborts — `-x`, `--maxfail=1`, `--sw`,
   `KeyboardInterrupt` — still arrive non-zero after the repair, on the
   whole tree, and the already-red carve-out still prints no second
   verdict.
4. **R1d** The anchor is `pytest_unconfigure`, reached through
   `config._ensure_unconfigure()` in `wrap_session`'s `finally`, which runs
   after `pytest_sessionfinish` and before `return session.exitstatus`.
   Driven: a marker plugin whose `pytest_unconfigure` assigns
   `session.exitstatus = 97` produces EXIT 97.
5. **R1e** `pytest_unconfigure` is strictly more reliable than
   `pytest_sessionfinish`: a marker plugin prints `MARKER-UNCONFIGURE-RAN`
   on at least one route where `MARKER-SESSIONFINISH-RAN` never prints.

### R2 — the fifth zero-arriving abort

1. **R2a** At `1b1c843`, `pytest.exit(reason, returncode=0)` raised from
   inside `pytest_sessionstart` gives EXIT 0 and no `pytest_sessionfinish`
   at all; the same from `pytest_configure` likewise.
2. **R2b** The docstring sentence *"on every way out of a session that got
   as far as `pytest_sessionstart`"* is deleted or corrected, because a
   session that exits FROM `pytest_sessionstart` got that far and was not
   covered.
3. **R2c** After the repair the anchor ARRIVES on both routes, and that is
   asserted by a test in the tree rather than only measured.

### R3 — the consolation sentence

1. **R3a** At `1b1c843`, a plugin whose `pytest_sessionfinish` does nothing
   but `session.exitstatus = 0` gives EXIT 0 with **no `Exit:` line
   anywhere**; adding `_NOTES.clear()` gives EXIT 0 with zero banners and a
   diff against the unplugged session of exactly the banner lines.
2. **R3b** The consolation — "it does not produce the byte-identical green
   this mechanism exists to end, because pytest writes `Exit: <reason>`" —
   is corrected in **both** the docstring and this pass's commit message.
3. **R3c** A row in the tree drives the no-raise defeat.

### R4 — the seventh floater and the qualitative "here"

1. **R4a** `tests/test_constant_fold_portfolio.py:353`'s
   `left the WHOLE SUITE GREEN (1696 passed)` carries a commit label, or is
   restated so that no unlabelled tree-sized figure remains.
2. **R4b** `tests/test_skip_inventory.py:619`'s
   `# All four are EXIT 1 with a banner here.` is labelled with the commit
   the four cells were driven at, and the four are re-driven at this tip.

### R5 — the two judgements

1. **R5a** The `33 caught / 28 signatures` totals and the paragraph naming
   them are **deleted**; the five named pairs and the measured
   still_owed/pending_items non-collision stay.
2. **R5b** `scratchpad/PREREG_SIR6.md` is in the tree, brought in by a
   `WITHHELD` entry for `scratchpad` in `tests/test_sdist_contents.py`. The
   whole suite stays green with it there, and `test_every_root_entry_is_a_decision`
   passes.

### R6 — row 1's assertions

1. **R6a** At `1b1c843`, row `exit-zero-from-inside-the-run-loop-buys-no-silence`
   PASSES with its plugin stripped (the trigger-strip probe).
2. **R6b** After tightening, the same row FAILS with its plugin stripped,
   and rows 2 and 4 still FAIL with their triggers stripped.

### R7 — the non-`exitstatus` channel

1. **R7a** A channel that is not `session.exitstatus` is implemented, with
   the reason it was chosen over the two alternatives written down.
2. **R7b** It survives every attack in R1a and R3a: driven under each, the
   channel still carries the verdict.
3. **R7c** Its limit is stated: what it does NOT defend against.

### R8 — neutrality

1. **R8a** `git diff --stat a80d60c HEAD -- src/ docs/` stays empty.
2. **R8b** Both cwds, final: counts and EXIT reported; collected-id sets
   md5-identical across cwds; the id diff against `1b1c843` is additions
   only, and every addition is a test this pass wrote.

## Prose claims — Priority 7

The finding this pass is scored hardest on: **pre-registration constrains
the measurements, not the prose written around them.** `R1.6` of the
previous round was satisfied — the breaking thing was named and measured —
and the tree still carried a false sentence, because the sentence
("there is no hook after the last hook") was reasoning written ALONGSIDE
the measurement and was never itself registered.

So every load-bearing sentence this pass writes into the tree about the
mechanism is listed here with **the observation that would falsify it**,
and each is scored in the report. A sentence not in this list may not be
written as a claim about the mechanism.

| # | sentence I intend to write | what would falsify it | how it is observed |
|---|---|---|---|
| W1 | `pytest_unconfigure` fires after `pytest_sessionfinish` and before `wrap_session` reads `session.exitstatus`. | a marker printed from `pytest_unconfigure` appearing BEFORE one printed from `pytest_sessionfinish`; or an assignment to `session.exitstatus` made from `pytest_unconfigure` not reaching the shell | a marker plugin printing from both hooks, plus one assigning 97 from `pytest_unconfigure` |
| W2 | Every anchor has a later one. `session.exitstatus` is a last-writer-wins channel, so any mechanism carrying its verdict there is beatable by construction. | an invocation where a `-p` plugin CANNOT override the verdict this conftest formed | `pytest_cmdline_main` as a `wrapper=True` hookimpl returning 98; and a `trylast` `pytest_unconfigure` reassigning after ours |
| W3 | A session leaving by `pytest.exit()` from inside `pytest_sessionstart` or `pytest_configure` never reaches `pytest_sessionfinish`. | `MARKER-SESSIONFINISH-RAN` printed in such a session | marker plugin + exit plugin, both routes |
| W4 | A plugin's `pytest_sessionfinish` that assigns `session.exitstatus = 0` without raising defeats the exit-code anchor and leaves NO `Exit:` line. | an `Exit:` line anywhere in that session's output, or a non-zero exit at `1b1c843` | driven at `1b1c843`, output searched for `Exit:` |
| W5 | The verdict FILE survives every attack that beats the exit code, because a plugin cannot un-write a file whose path came from the environment rather than from the session. | any attack in R1a/R3a leaving the file absent, empty, or saying something other than the verdict | the file read back after each attack |
| W6 | LIMIT of W5: it is not adversary-proof. A plugin that reads the same environment variable can delete the file. | — (this is a limit, not a claim; it is falsified only by the file surviving a plugin that deliberately deletes it, which would mean the limit is understated) | stated, and the deleting plugin driven so the limit is measured rather than assumed |
| W7 | The already-red carve-out is kept: a session red for a reason that is not this guard's own gets no second verdict. | `-x` / `--maxfail=1` / `--sw` / `KeyboardInterrupt` printing the banner, or the row `an-abort-that-is-already-red-gets-no-second-verdict` failing | the four aborts driven whole-tree at the tip |
| W8 | Moving the anchor changes no existing test id and removes nothing. | a non-empty removal set in the id diff, or a changed id | `--collect-only` id diff, both cwds |
| W9 | The fifth abort's record is EMPTY — a session that exits from `pytest_sessionstart` collected nothing, so the anchor arriving is all there is to check, and nothing is owed. | such a session having a non-empty `SKIPPED`/`RAN`/`DESELECTED` | the verdict file's contents on that route |

## Known and declared in advance

* I have not verified pytest-xdist behaviour (not installed); nothing here
  changes what the branch already discloses about it.
* `src/` is not touched. No numeric constant in `src/` changes.
* `docs/norms.md` is not edited.

---

# OUTCOMES — appended after the fact; nothing above this line is edited

All figures below were driven on this machine: python 3.12.3, pytest 9.1.1,
24 cores, `JAX_PLATFORMS=cpu`, `PYTHONPATH=<worktree>/src` verified to resolve
inside the worktree before every comparison. Where a wall figure appears, the
load average is beside it.

## Clause by clause

| clause | verdict | what was measured |
|---|---|---|
| R1a | **MET, with one deviation** | at `1b1c843`, `tests/test_affine.py` + planted skip: no plugin EXIT 1 / 1 banner; `-p exit0finish` EXIT 0 / 0; `-p sf_tryfirst` EXIT 0 / 0; `-p exit0finish -p unconf` EXIT **1** / 0 banners, with an uncaught `_pytest.outcomes.Exit` traceback. **DEVIATION:** the audit reported 99 for the last one. `pytest.exit(returncode=99)` raised from `pytest_unconfigure` escapes `_ensure_unconfigure`, `wrap_session`'s `finally` and `config._main` uncaught, and the process ends 1. Reported as measured. |
| R1b | **MET** | at the tip, same plant: all four are EXIT **1** with **1 banner** naming the skip. |
| R1c | **MET** | whole tree at the tip with a planted failure: `-x`→1, `--maxfail=1`→1, `--sw`→2, `KeyboardInterrupt`→2, **0 banners each**. |
| R1d | **MET** | a plugin assigning `session.exitstatus = 97` from `pytest_unconfigure` → **EXIT 97**. |
| R1e | **MET** | marker order is `MARKER-SESSIONFINISH-RAN` then `MARKER-UNCONFIGURE-RAN`; on `exit0start` and `exit0configure` only `MARKER-UNCONFIGURE-RAN` prints. |
| R2a | **MET** | `pytest.exit(returncode=0)` from `pytest_sessionstart` → EXIT 0, 0 banners, no sessionfinish marker. Same from `pytest_configure`. |
| R2b | **MET** | the scoping sentence is deleted; `pytest_unconfigure`'s docstring states the off-by-one and names it. |
| R2c | **MET** | `test_the_anchor_arrives_where_pytest_sessionfinish_is_never_called`; FAILS with the exit plugin stripped. |
| R3a | **MET** | `assign0`: EXIT 0, banner printed, **0 `Exit:` lines**. `assign0clear`: EXIT 0, 0 banners; `diff` vs the unplugged session is exactly the five banner lines (and the timing digit). |
| R3b | **MET** | corrected in the `pytest_sessionfinish` docstring and stated in the commit message. |
| R3c | **MET** | `test_a_sessionfinish_that_raises_nothing_still_takes_the_exit_code`, both variants. |
| R4a | **MET, re-derived rather than guessed** | `ae3231e` (the repair commit) is `1713 passed, 2 skipped`; its base `1614b18` — the tree the corruption was driven against — is **`1696 passed, 2 skipped`**, exact. Labelled `1614b18`. |
| R4b | **MET** | re-driven at `a76ca51` with a plugin removing the first non-pin item from `items[:]`: 2033 / 2016 / 40 / 2032 passed, all four EXIT 1 with a banner. |
| R5a | **MET** | the totals paragraph is deleted; six named pairs remain, the sixth derived in this pass. |
| R5b | **MET** | `scratchpad/PREREG_SIR6.md` is in the tree; `WITHHELD["scratchpad"]` added; `test_every_root_entry_is_a_decision` passes. |
| R6a | **MET (the audit reproduces)** | at `1b1c843`, row 1 with its plugin stripped: **1 passed**. Rows 2 and 4 with their triggers stripped: **1 failed** each. |
| R6b | **MET** | row 1 now asserts the `Exit:` line and **fails** with the plugin stripped. Every new case was probed the same way: 7 strips, 7 failures, no survivors. |
| R7a | **MET** | `STELLING_SKIP_INVENTORY_VERDICT`; the choice against (b) and (c) is written in `_write_the_verdict_somewhere_last_writer_wins_cannot_reach`. |
| R7b | **MET** | file says `verdict=failed` under `exit0finish`, `sf_tryfirst`, `exit0finish+unconf`, `assign0`, `assign0clear`, `cmdline98` (EXIT 98) and `unconf_trylast` (EXIT 0). |
| R7c | **MET, and driven** | a plugin reading the same variable deletes the file; `test_the_verdict_file_is_not_adversary_proof_and_this_is_the_limit` asserts it is GONE. |
| R8a | **MET** | `git diff --stat a80d60c HEAD -- src/ docs/` is empty. |
| R8b | **MET** | both cwds **2035 passed, 2 skipped, EXIT 0**; 2037 collected; normalised id sets md5-identical (`bbde35d8d7e6b1930fb4d1a32cde99eb`); **+8 / −0** against `1b1c843`, every addition a test written here. |

Wall: 146.03 s from the repo root (load average 11.74 / 9.61 / 9.84 at the
end), 173.55 s from `tests/` (11.91 / 9.19 / 9.54). Baseline at `1b1c843` was
140.49 s and 151.37 s at comparable load; the difference is eight new
subprocess sessions plus load, and is not offered as a measurement of
anything.

## Prose claims, scored

| # | verdict | the falsifying observation, run |
|---|---|---|
| W1 | **HELD** | marker order is sessionfinish → unconfigure; an assignment of 97 made from `pytest_unconfigure` arrives as EXIT 97. |
| W2 | **HELD, and it is the important one** | a `pytest_cmdline_main` wrapper returning 98 gives EXIT 98 with the banner on screen; a `trylast` `pytest_unconfigure` assigning 0 gives EXIT 0 with the banner on screen. Both at `1b1c843` and at the tip. Two cases in the tree now assert the attacker's exit code first, so the demonstration cannot rot into a mechanism that happens to win. |
| W3 | **HELD** | `MARKER-SESSIONFINISH-RAN` does not print on either route. |
| W4 | **HELD** | EXIT 0 at `1b1c843`, and `grep "Exit: "` over the whole output is 0. |
| W5 | **HELD** | seven attacks, seven `verdict=failed` files. |
| W6 | **HELD as a limit** | the deleting plugin removes the file; the limit is not understated. |
| W7 | **HELD** | four aborts, four non-zero exits, zero banners. |
| W8 | **HELD** | id diff is +8 / −0, additions only, ids md5-identical across cwds. |
| W9 | **HELD** | the file on that route reads `verdict=withdrawn`, which is what an empty record supports. |

## Declared misses and deviations

1. **R1a deviates from the audit's figure for one route.** `-p exit0finish -p
   unconf` arrives 1 here, not 99, and by a different mechanism (an uncaught
   `Exit`). The substantive claim — that the route was NOT closed at
   `1b1c843` — reproduces: 0 banners.
2. **A collision was found and is recorded rather than smoothed.** Of six
   mutations driven against the new mechanism, all six are caught and two
   share a signature: `_OUR_FAILURES` ignored in the already-red carve-out,
   and `pytest_unconfigure` never re-assigning `session.exitstatus`. It is
   written into the pairs table as the sixth pair. No total is claimed.
3. **The anchor is not the last one and the tree says so.** This is not a miss
   against the prereg — W2 registered it — but it is the honest headline: the
   exit code is still beatable after this pass, by design, and the verdict
   file is the answer.
4. **`pytest_unconfigure`'s reliability claim is scoped to this pytest.**
   9.1.1, read from `_pytest/main.py` and driven. Nothing here is upstream.
5. **Process note.** A trigger-strip probe script called `git checkout --` on
   files carrying uncommitted work and reverted it; the work was re-applied
   from scratch and committed before the probes were re-run. One difference
   between the two applications was caught (an inverted `through`/`around` in
   a failure message) and fixed in its own commit. No measurement in this
   document was taken from the reverted state.
