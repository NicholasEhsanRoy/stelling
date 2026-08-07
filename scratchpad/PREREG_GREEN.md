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
