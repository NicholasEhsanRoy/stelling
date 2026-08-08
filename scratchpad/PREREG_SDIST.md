# PREREG — the sdist untracked-file guard, and four things beside it

Branch `fix/sdist-untracked-hatchling-parity`, off `main` at 9c37995.
Written **before the first edit to the tree** and before any measurement of my
own. Everything below the line `## OUTCOMES` is appended afterwards; nothing
above it is edited after the fact.

The rule for this pass: for each finding, the account I am giving of it, and
**the observation that would falsify that account**. If the falsifier fires, the
account is wrong and the fix built on it is wrong with it, however green the
suite goes.

---

## F-A — `test_no_untracked_file_anywhere_would_ship` enumerates on a channel that disagrees with the builder

**Account.** The test asks `git status --porcelain --untracked-files=all`.
`git status` honours four exclusion sources: `.gitignore` files at every level,
`$GIT_DIR/info/exclude`, `core.excludesFile`, and the command line. Hatchling
honours exactly one thing: the *single* `.gitignore` found by walking up from
the build root to the `.git` boundary
(`hatchling/builders/config.py::vcs_exclusion_files`, read at 1.31.0 in the uv
cache). So every exclusion source git has that hatchling does not is a file
that vanishes from the guard's enumeration and appears in the tarball.

Predicted, in a standalone repo built by `git archive` + `git init`:

* `docs/zz_exclude_probe.md` + a line in `.git/info/exclude`
  → `git status --porcelain -uall` names it **0** times
  → the node id passes
  → `uv build --offline --sdist` ships `stelling-0.1.0/docs/zz_exclude_probe.md`.
* the same with `core.excludesFile` pointing at a file naming the path
  → identical outcome.
* a third variant I predict from the source and have not seen anyone state:
  hatchling reads the ROOT `.gitignore` **only**, so a path ignored by a
  *nested* `.gitignore` (this repo has `scratchpad/reach/.gitignore`) is also
  invisible to the guard and visible to hatchling. Predicted to ship if planted
  under an allowlisted directory with a nested `.gitignore` covering it.

**Falsifiers of this account**

* F-A.1 The probe does **not** appear in the built sdist under any of the three
  variants. Then hatchling is not "gitignore-only" in the way I have read it,
  the whole account is wrong, and the fix must be rebuilt from the artefact
  rather than from the source I read.
* F-A.2 `git status --porcelain -uall` **does** report a path excluded via
  `.git/info/exclude`. Then the enumeration channel does not disagree with the
  builder and there is no F-A.
* F-A.3 `uv build` resolves a hatchling whose `vcs_exclusion_files` reads more
  than one `.gitignore` (i.e. the nested-`.gitignore` variant does NOT ship).
  Then the third variant is not real and must not be claimed; the first two
  stand on their own.

**Account of the vacuity half.** The test has no non-vacuity guard: with the
enumeration forced to report nothing, `undecided` is empty and the assertion
passes. Predicted: `--untracked-files=all` → `=no`, with a real untracked
non-ignored file present, gives `1 passed`.

* F-A.4 That mutation gives anything other than a pass. Then the test already
  has a non-vacuity property I have not found, and the claim "zero items
  examined reads as zero problems" is false of it.

**Account of the fix.** The guard must classify by *hatchling's* rule, not
git's. My intended shape — to be discarded if it fails its own controls:

1. enumerate untracked files WITHOUT consulting any exclusion source at all:
   `tracked = git ls-files`, walk the filesystem, `untracked = walked - tracked`.
   This has no exclusion channel to disagree with anything.
2. classify each untracked path by hatchling's actual rule: pruned directory
   names (`EXCLUDED_DIRECTORIES`), excluded file names (`EXCLUDED_FILES`),
   `default_global_exclude` (`*.py[cdo]`, `/dist`), the ROOT `.gitignore`
   evaluated *in isolation from every other exclusion source*, and the sdist
   `include` allowlist.
3. the root `.gitignore` is evaluated by git itself, in a throwaway repository
   that contains that file and nothing else — so `info/exclude`, nested
   `.gitignore`s and `core.excludesFile` cannot reach it.

**Falsifiers of the fix**

* F-A.5 The rebuilt guard passes on a tree where a probe is planted under
  `.git/info/exclude`, or under `core.excludesFile`, or under a nested
  `.gitignore`, and that probe is in the built sdist. Then the parity is not
  parity.
* F-A.6 The rebuilt guard FIRES on a path that is genuinely absent from the
  built sdist (a stray under `scratchpad/`, a `*.pyc`, a file under `dist/`, a
  path matched by the root `.gitignore`). Then it cries wolf and will be
  disabled by the first developer who meets it.
* F-A.7 The rebuilt guard passes when it examines nothing — enumeration forced
  empty, or classification forced to "never ships". Then I have rebuilt the
  defect I was sent to close.
* F-A.8 The classifier disagrees with the ARTEFACT on any path in the current
  tree. The arbiter is `tar tzf`, not my reading of hatchling.

---

## F-B — two pre-commit hooks Pass when the tree they scan is absent

**Account.** `grep -r … src` exits **2** when `src` does not exist and **1**
when it exists with no match. `jax-import-hygiene` writes
`if grep …; then rc=1; fi`, so 2 and 1 are both "no violation".
`library-identifier-hygiene` writes `! grep …` under `set -e`, so a non-zero
from either grep — error or no-match — inverts to success. Neither counts what
it examined, so "there was nothing to examine" is indistinguishable from "there
was nothing wrong".

* Predicted: with `src/` moved aside, `pre-commit run <hook> --all-files` →
  **Passed** for both.
* Predicted: with `src/stelling/` moved aside so that `src/` exists and holds no
  `.py`, → **Passed** for both (grep exits 1, nothing examined).

**Falsifiers**

* F-B.1 Either hook already Fails with the tree absent. Then that hook is not an
  instance and must not be reported as one.
* F-B.2 `grep -r` on this box exits something other than 2 for a missing
  directory (busybox grep, a shell builtin). Then the mechanism is misstated
  even if the symptom is real, and the fix must key on the count and not on the
  code.
* F-B.3 After the fix, either hook Passes with the tree absent, or Fails on the
  real tree (which contains `src/stelling/_jax_compat.py`, a legitimately
  jax-importing file, and prose lines carrying census/design markers).
* F-B.4 A third hook in `.pre-commit-config.yaml` has the same shape and I do
  not name it. To be checked hook by hook, not assumed.

---

## F-C — a step name that contradicts its own body

**Account.** `.github/workflows/ci.yml`, job `acceptance-any-pytree`: the step
named "the skip-inventory verdict, asserted off the exit code" has a body that
asserts off the FILE channel and a comment underneath saying the exit code is
last-writer-wins and cannot be trusted. The other verdict steps say "off a file
channel".

* F-C.1 The body really does assert off the exit code (i.e. the name is true and
  the comment is the false one). Then the correction goes the other way.
* F-C.2 The other verdict steps are NOT named "off a file channel", so there is
  no house style being broken. Checked by reading all of them.

---

## F-D — `pypa/gh-action-pypi-publish@release/v1` is a branch ref

**Account.** A branch ref is re-pointed by its owner at will and is weaker than
a floating major tag, on the one job holding `id-token: write`. This is a
recommendation task, not a repair task; I implement only if the case is clear.

* F-D.1 If pinning to a commit SHA would break Trusted Publishing or suppress
  PEP 740 attestations, the recommendation to pin is wrong.
* F-D.2 If I cannot obtain a SHA offline, an unverifiable SHA typed from memory
  is worse than the branch ref, and the answer is a stated human action.

---

## F-E — reuse 5.1.1 in CI against reuse 6.2.0 in pre-commit

**Account.** With `insert-license` now excluding `scratchpad/`, `REUSE.toml`'s
`scratchpad/**` annotation is the only thing making those files compliant, and
CI lints them with a version nobody here has run. A prior audit says the 6.2.0
`CHANGELOG.md` documents no change to `REUSE.toml` parsing, `[[annotations]]`,
`precedence` or path globs between 5.1.1 and 6.2.0.

* F-E.1 The changelog DOES document such a change in 5.1.2 … 6.2.0. Then the
  skew is load-bearing in fact and not only in principle, and the pass must say
  so rather than repeat the prior audit's sentence.
* F-E.2 The vendored changelog does not cover the whole 5.1.1 → 6.2.0 range
  (truncated, or starts later). Then the bound is not established and I must say
  what is missing rather than assert the negative.
* F-E.3 Pinning CI to 6.2.0 requires a network fetch to verify. Then it is a
  human action and I state it precisely instead of guessing an action ref.

---

## Standing falsifiers for the pass as a whole

* S.1 Any test id present in `--collect-only` on `main` and absent here, or vice
  versa, on either jax series.
* S.2 A count other than 2296 passed / 2 skipped on either series.
* S.3 `reuse lint` ending non-zero.
* S.4 Any change under `src/stelling/`, to `docs/norms.md`, to
  `src/stelling/propagate.py`, or to `tests/test_probe_witness.py`.
* S.5 A probe file left in the tree, or a dirty worktree at the end.

---

## OUTCOMES
