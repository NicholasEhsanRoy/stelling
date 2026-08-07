# PREREG — false-green controls sweep

Branch `fix/false-green-controls`, off `main` = `33fd1f8`.
Worktree `/home/nick/MSF/.wt-green/W`.
Written BEFORE any edit outside this file. Nothing above the OUTCOMES rule is
edited afterwards.

A **false green** here: a control reports pass while the thing it guards is not
satisfied, or while the check did not run at all.

Method rule I am binding myself to: every claim below is either CONFIRMED by a
command whose output distinguishes it from its nearest rival, or labelled
SUSPECTED. Every fix must be shown to bite: a reddening command and its output,
then a restore and a green.

---

## Scope

`.github/workflows/ci.yml`, `.github/workflows/release.yml`,
`.pre-commit-config.yaml`, `tests/conftest.py`, `tests/test_sdist_contents.py`.
No edits under `src/stelling/`. No edits to `docs/norms.md`.

---

## The two instances I was handed

### A — `insert-license` reports Passed on a header-less file

**Account.** `pre-commit run insert-license --files <f>` reads and rewrites the
WORKING TREE. On the first run it inserts the header into the worktree and
exits 1; the INDEX still holds the header-less blob. A second run reads the
now-fixed worktree and reports Passed. So "Passed" is compatible with a
header-less blob being the thing that gets committed.

**A1 falsifier.** If `pre-commit run --files <f>` stashes unstaged worktree
changes the way the `git commit` path does, the second run would see the
header-less INDEX content, re-insert, and report Failed again. Observation that
would falsify my account: second run exits 1 / prints Failed.

**A2 falsifier.** If the account is right, the state after run 1 is `AM` and
`git show :<f>` (index) has no SPDX line while the worktree file does. If
`git show :<f>` already carries the header, my account of the mechanism is
wrong.

**A3.** The path that actually matters is `git commit`. Prediction: the
installed `.git/hooks/pre-commit` DOES bite, because pre-commit applies
`staged_files_only` there. Falsifier: a `git commit` of a header-less staged
blob succeeds with the hook installed and not skipped.

**A4 — how `14e34a2` could have landed 12 header-less `scratchpad/*.py`.**
Candidate mechanisms, and the observation that separates them:
  - the commit was made with `--no-verify` or `PRE_COMMIT_ALLOW_NO_CONFIG`/
    `SKIP=`;  → not observable from the object store; UNDECIDABLE is the honest
    verdict unless reflog/ORIG_HEAD says otherwise.
  - the files were `AM` at commit time (the exact A-mechanism above): the hook
    ran, fixed the worktree, the developer committed the INDEX anyway.
    Observation that would support it: the worktree copies at the time carried
    headers while the committed blobs did not — not recoverable now.
  - the commit was produced by `git commit` from a directory/tool that bypasses
    hooks (`git -c core.hooksPath=`), or amended/rebased (rebase does not run
    the pre-commit hook).
  I pre-register that I expect to be able to **rule some of these OUT** and to
  be unable to single one IN; the deliverable is the mechanism-level fix, not
  the archaeology. Falsifier for "the hook did not run": a header-less `.py`
  inside `files:` at `14e34a2` that the hook would in fact have left alone —
  i.e. re-running the pinned hook version over those exact blobs reports
  Passed. That is the measurement.

**A5 — is anything green today lying?** Prediction: `reuse lint` at `33fd1f8`
is 283/283 exit 0 and is the only control that bit. Falsifier: `reuse lint`
is non-zero at `33fd1f8`.

**Fix I expect to make.** Not a change to the third-party hook. A control that
FAILS when the staged blob lacks a header, checkable locally and in CI, whose
positive control is "stage a header-less blob → RED".

### B — `test_an_arbitrary_new_file_does_not_ship` writes into the REPO ROOT

**B1 — the spurious RED.** Two suite runs against one checkout race:
`test_every_root_entry_is_a_decision` (and
`test_no_untracked_file_anywhere_would_ship`) sees
`zz_sdist_allowlist_probe.txt` and fails. Falsifier: with the probe present at
the repo root, both of those tests still pass.

**B2 — is there a false GREEN in it too?** Pre-registered candidates, each with
its intervention:
  - *the probe is never created* → does the test still pass? Intervention:
    neuter `probe.write_text` and run. Predicted: PASSES (false green) —
    nothing re-reads that the probe existed.
  - *the sdist build silently produces nothing* → guarded by
    `returncode == 0`, `len(built) == 1`, and the `pyproject.toml`
    non-vacuity assert. Predicted: NOT a false green. Falsifier: an empty /
    absent artefact still passes.
  - *the assertion ranges over an empty set* → the `pyproject.toml` assert is
    the stated non-vacuity control. Predicted: it bites.
  - *the docstring's own "Break it" recipe* — delete
    `[tool.hatch.build.targets.sdist]` and the test must fail. Predicted: it
    DOES fail. Falsifier: it passes, in which case the headline control of this
    file is inert and that is the biggest finding in the task.
  - *hatchling builds from the git INDEX rather than the worktree*, in which
    case an untracked probe could never ship and the control is inert for a
    second reason. Same measurement as above separates it.

**B3.** `WITHHELD` in `tests/test_sdist_contents.py` contains the key
`"scratchpad"` TWICE (two different reasons). Python keeps the last. Predicted:
the first reason is silently dead. Falsifier: the two keys differ in text I
have misread, or Python somehow keeps both.

---

## The sweep — what I expect to find, pre-registered before measuring

Each is SUSPECTED until a command says otherwise.

**S1 — `jax-import-hygiene` pre-commit hook swallows its FIRST check.**
The entry is `bash -c 'set -e; ! grep A | grep -v B; ! grep C'`. bash's `set -e`
is documented to be IGNORED when a command's status is inverted with `!`. The
first statement is therefore non-fatal, and the script's status is the LAST
statement's. Prediction: a file under `src/` with a bare `import jax` outside
`_jax_compat.py` and no `jax._src` leaves this hook **Passed**.
Falsifier: the hook reports Failed on such a file. (That would mean `set -e`
does fire on `!`-inverted pipelines here, and S1 is wrong.)

**S2 — `library-identifier-hygiene`** is the same shape but has ONE statement,
so its status propagates. Prediction: it BITES. Falsifier: it passes on a
planted banned identifier in `src/stelling/*.py`.

**S3 — `dco` job `if: github.event_name == 'pull_request'`.** On a push to
`main` the job does not run and the workflow is green. Conditional-job shape.
Prediction: CONFIRMED by reading the `if:` and the `on:` triggers; not
runnable here. Will be labelled SUSPECTED-BY-CONSTRUCTION / reasoned-only.

**S4 — the three primary lanes (`test-no-jax`, `test-jax`, `test-jax-0-10`)
and nothing else run `pytest -q` with NO separate verdict channel.** The
repo's own stated reason for the channel — "pytest's own exit code cannot
report that" — applies verbatim to them. Prediction: the asymmetry is real.
Falsifier: one of those three does carry `STELLING_SKIP_INVENTORY_VERDICT`.

**S5 — `-rs` in `release.yml` and in the `acceptance-any-pytree` grep step.**
Prediction: `-rs` suppresses the `FAILED` short-summary lines that `-ra`
emits; exit code is unaffected, so this is a legibility defect, not a false
green. Falsifier: a failing test under `-rs -q` also loses its non-zero exit.

**S6 — pipefail coverage.** Prediction: every `| tee` in both workflows is
inside a `set -o pipefail` block. Falsifier: one is not.

**S7 — `read -r a b < <(...)`** in `test-jax-0-10` does not check the process
substitution's status. Prediction: it still fails closed, because an empty
`ver` matches the `*)` arm. Falsifier: an empty/absent jax leaves the step
green.

**S8 — a `pytest` selection resolving to 0 tests.** Prediction: none of the
narrowed pytest invocations can silently select 0 — the two acceptance steps
assert exact counts (`^2 passed`, `^18 passed`) and pytest exits 5 on no tests
collected. Falsifier: a step whose selection can go to 0 and still exit 0.

**S9 — the `reuse` CI job pins `fsfe/reuse-action@v5` (= 5.1.1) while
`.pre-commit-config.yaml` pins `reuse-tool` `v6.2.0`.** Two different linters
guard the same property. Prediction: a version-skew hazard I can only reason
about here, because installing 5.1.1 needs the network. Will be labelled
reasoned-only unless I find a local 5.x.

**S10 — presence-not-truth tests.** Prediction: at least one test in the swept
set asserts that a thing is PRESENT rather than TRUE. Falsifier: none found.

**S11 — install steps that resolve a different version than the job claims.**
Prediction: `test-jax-0-10` and the `0.10` matrix entry both assert the
resolved series, so they are covered; `acceptance-any-pytree` asserts before/
after equality but never asserts WHICH series it got. Falsifier: it does.

---

## What would make me say "no false green here"

For each swept step I must be able to name the observation that reddens it and
show it. A step I cannot redden on demand is reported as such rather than
cleared.

---

<!-- ================== OUTCOMES: append only, below this line ================== -->

# OUTCOMES

Appended after the fact. Nothing above the rule line was edited.
Baseline `33fd1f8`, measured in two worktrees run in parallel (separate
checkouts, which is the point of defect B):

    jax 0.11.0   2296 passed, 2 skipped, exit 0, 154.65s   load avg 3.06
    jax 0.10.2   2296 passed, 2 skipped, exit 0, 152.28s   (same window)
    reuse 6.2.0  283/283, exit 0
    --collect-only: 2298 ids, byte-identical between the two series

Branch tip, same arrangement:

    jax 0.11.0   2296 passed, 2 skipped, exit 0, 157.65s   load avg 2.69
    jax 0.10.2   2296 passed, 2 skipped, exit 0, 155.54s   (same window)
    verdict=made on both
    reuse 6.2.0  284/284, exit 0   (the extra file is this one)
    --collect-only: 2298 ids, identical across series AND to the baseline

## A — insert-license

**CONFIRMED, exactly as described.**

    printf 'X = 1\n' > scratchpad/zz_hookprobe.py; git add …
    pre-commit run insert-license --files …   Failed
    git status --short                        AM
    git show :scratchpad/zz_hookprobe.py      X = 1
    pre-commit run insert-license --files …   Passed
    git show :scratchpad/zz_hookprobe.py      X = 1

* **A1 falsifier NOT satisfied.** The second run did not stash and did not
  re-read the index: `Passed`. My account stands.
* **A2 falsifier NOT satisfied.** The index blob was header-less at every
  point.
* **A3 CONFIRMED as predicted.** `pre-commit run` with no `--files` — the
  path `hook-impl` takes — stashes (`Stashing unstaged files to …`) and
  reports `Failed` with the worktree already fixed. The `git commit` path
  bites; `--files` and `--all-files` are the ones that lie.
* **A4.** Falsifier NOT satisfied: the pinned `insert-license` v1.5.6, run
  over the exact blobs `14e34a2` committed (extracted with `git show
  14e34a2:<path>` into a fresh repo with that commit's own config and
  `.license-header.txt`), rewrote **all 13** of them. So the hook would have
  fired; that it did not run stands. Ruled OUT by measurement: `core.hooksPath`
  unset; `.git/hooks/pre-commit` present, dated 2026-07-16, `INSTALL_PYTHON`
  resolvable; and the branch reflog records the entry as `commit:`, not a
  rebase or cherry-pick (which do not run this hook). What is left is
  `--no-verify` / `SKIP=`, which leaves no trace anywhere — **UNDECIDABLE, as
  pre-registered.** 13 not 12: `scratchpad/reach/cases.py` was removed in the
  very next commit, leaving the 12 still in the tree.
* **A5 falsifier NOT satisfied**, and the backstop was measured rather than
  assumed: a header-less file dropped at `src/zz_no_header.py` takes
  `reuse lint` from `283/283 exit 0` to `exit 1`, and removing it restores it.

**Fix.** `staged-spdx-header`, a local hook reading `git show :<path>` — the
one channel a worktree fixer cannot touch. Reddening: the sequence above, at
the second run. Restoring: `git add`. Scoped to the DIVERGENCE, so the twelve
deliberately header-less `scratchpad/**` files that REUSE.toml annotates are
left alone. Hardened afterwards: outside a git repository the loop skipped
every file and returned 0, i.e. the fix had the defect it was fixing; it now
exits 1 saying it checked nothing.

**Not claimed.** This does not stop `--no-verify`. Nothing client-side can.
`reuse lint` in CI is the control that runs where `-n` cannot reach, and since
`33fd1f8` it does not require headers under `scratchpad/**` — by design, not
by accident, and left alone as instructed.

## B — the sdist probe

* **B1 CONFIRMED, and it is TWO tests, not one.** With the probe present at
  the root, `test_every_root_entry_is_a_decision` AND
  `test_no_untracked_file_anywhere_would_ship` both fail; both pass on removal.
  The real race, two runs 0.4s apart in one checkout: `7 passed` /
  `2 failed` (`test_no_untracked_file_anywhere_would_ship` and
  `test_an_arbitrary_new_file_does_not_ship`, the latter on "probe path is
  already taken").
* **B2 CONFIRMED — there is a false GREEN in it.** Predicted and found: with
  `probe.write_text` replaced by `pass`, `1 passed`. The control passed having
  observed nothing. The other two candidates were predicted NOT to be false
  greens and were not: an absent/broken build is caught by `returncode == 0`,
  `len(built) == 1` and the `pyproject.toml` assert. The docstring's own
  break-it recipe was predicted to bite and does — deleting
  `[tool.hatch.build.targets.sdist]` leaks
  `stelling-0.1.0/zz_sdist_allowlist_probe.txt` into the tarball, both from
  the real repo and (checked separately, because it is the new arrangement)
  from the staged copy. The "hatchling builds from the git INDEX" rival is
  refuted by the same measurement.
* **B3 CONFIRMED.** `WITHHELD` carried `"scratchpad"` twice; Python kept the
  last and the first reason was dead text. Merged.
* One thing I did **not** pre-register and found: `test_every_allowlist_entry_exists`
  is vacuous on an EMPTY include list. `include = []` -> was `1 passed`, now
  `1 failed`.

**Fix.** The intervention moved to a private copy (0.01s, 5.3 MiB), and the
build made to prove it can SEE an untracked file — a second probe inside the
allowlisted `/docs` that MUST ship — before its silence about the first is
believed. Reddening, each restored to green: probe never created; positive
control never created; probe unlinked before the build; allowlist table
deleted; `include = []`. Race replayed three times on the fixed test:
`7 passed` / `7 passed` every round, nothing left at the root.

## The sweep

**S1 CONFIRMED — the largest find, and not one I was handed.**
`jax-import-hygiene` was `! grep A; ! grep B` under `set -e`, and bash ignores
errexit on a `!`-inverted command (driven directly: `bash -c 'set -e; ! true;
echo REACHED'` prints REACHED and exits 0). So the FIRST check could not fail
the hook. With a bare `import jax` planted under `src/` outside `_jax_compat.py`
and no `jax._src` anywhere: **`Passed`, exit 0**, with the offending line
printed in the output. It went `Failed` only once `jax._src` was also present.
The falsifier — "the hook reports Failed on such a file" — was NOT satisfied.
Fixed by combining through `rc`; both halves now redden and both are reported.
The property itself was never uncovered: `tests/test_import_hygiene.py::
test_jax_imported_only_in_compat_module` is the independent guard that held.

**S2 falsifier NOT satisfied.** `library-identifier-hygiene` has one statement,
so its status propagates. Planted `# uses lineax here` in `src/stelling/`:
`Failed`, exit 1; removed: `Passed`. CLEARED.

**S3 CONFIRMED by construction, REASONED-ONLY.** `dco` is
`if: github.event_name == 'pull_request'` and the workflow also runs on
`push: branches: [main]`, where the job does not run at all. Not changed —
running `dco-check` on a push has no PR range to check, so this changes what
the job means. Written up instead. A related, also reasoned-only point:
`pipx run dco-check` is unpinned, and a checker that finds no commits to check
exits 0 the same way one that finds only signed-off commits does. Neither is
drivable here (no network).

**S4 CONFIRMED and FIXED.** `test-no-jax`, `test-jax`, `test-jax-0-10` — the
only three lanes a branch protection rule can require — ran `pytest -q` and
read nothing but the exit code, while the two jobs that DO assert the verdict
are the two the file says must not be required. Measured verdict=made in all
three environments before adding the assertion, including a genuinely jax-less
interpreter (`1185 passed, 85 skipped`), built by symlinking the shared venv's
site-packages minus jax/jaxlib rather than by an import hook, because
`stelling._optional.available` asks `find_spec` and a raising hook is a
different observation from an absent package.

**S5 CONFIRMED, and it is legibility, not a false green.** On one failing and
one skipping test: `-rs` prints the SKIPPED line and omits `FAILED <nodeid>`;
`-ra` prints both; exit 1 either way, so the falsifier ("`-rs` also loses the
non-zero exit") was NOT satisfied. Changed to `-ra` in `release.yml` and in the
any_pytree step; the `grep -q '^SKIPPED'` still fires, driven.

**S6 falsifier NOT satisfied.** Every pipe in a guard in both workflows is
inside a `set -euo pipefail` block. Measured what that buys, on the real step
body: with `pipefail`, a failing import gives exit 1 and an empty file; with
`set -eu` alone, **exit 0 and an empty file** — the repo's own comment, driven.

**S7 falsifier NOT satisfied.** `read -r a b < <(…)` in `test-jax-0-10` fails
closed in all five drives, including "python prints nothing and exits 0".

**S8 falsifier NOT satisfied.** No narrowed pytest invocation can silently
select 0: the two acceptance steps assert `^2 passed` / `^18 passed`, and a
0-collected run is exit 5 through `set -e`. Driven.

**S9 REASONED-ONLY.** CI lints with `fsfe/reuse-action@v5` (5.1.1);
`.pre-commit-config.yaml` pins `reuse-tool` `v6.2.0`. Two different linters
guard one property. Only 6.2.0 is installed here and installing 5.1.1 needs the
network, so this is stated and not measured.

**S10 CONFIRMED, one instance**, and not the one I expected: not a test
asserting presence, but `test_every_allowlist_entry_exists` asserting over a
set that can be empty. Fixed with a non-vacuity assert that reddens on
`include = []`.

**S11 falsifier NOT satisfied.** `acceptance-any-pytree` asserts before==after
and never asserts WHICH series it resolved — correct, it is the floating
series by design, and the file says so. Cleared with the note.

**Not pre-registered, found while driving: the stale verdict file.** The
verdict steps argue that ABSENCE is the signal. Absence is only a signal if
nothing else could have supplied the file. Driven, with a verdict file already
at the path and a stand-in pytest that writes none: **exit 0** on a session
that wrote nothing. Inert on hosted runners (fresh `RUNNER_TEMP` per job) and
closed anyway with one `rm -f`, which is exit 1 in the same drive.

## Cleared, each shown to redden on demand

`test-jax-0-10` series guard (5 drives) · any_pytree "must RUN" (6) · the
verdict steps, old and new (6 each) · release tag-vs-artifact (5) · the
reproducer install/series/before-after guards (6+4) · the any_pytree
before/after guards (4) · the release sdist manifest (2) · `assert jax is
absent` (2, against a real jax-less interpreter and a real jax one) ·
`library-identifier-hygiene` (2) · `reuse lint` (2) · conftest with the pin
made unimportable (verdict=failed, exit 1) and with the conftest removed
(no file at all).
