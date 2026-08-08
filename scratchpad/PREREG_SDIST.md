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

Appended after the fact; nothing above this line was edited. Every figure below
was produced on this box, offline, with `git archive` + `git init` throwaway
repositories rather than by mutating any shared worktree.

### F-A — CONFIRMED, three variants, and closed

Baseline sdist: **260** members. Each variant planted in its own standalone
repo built from `main`:

| variant | `git status -uall` | the guard | sdist |
|---|---|---|---|
| `docs/zz_exclude_probe.md` + `.git/info/exclude` | 0 mentions | 1 passed | **261, SHIPPED** |
| `docs/notes.md` + `core.excludesFile` naming `notes.md` | 0 mentions | 1 passed | **261, SHIPPED** |
| `docs/zz_nested_probe.md` + untracked `docs/.gitignore` | 0 mentions of the probe | 1 **failed** (it saw the `.gitignore`) | 262, SHIPPED |
| the same with `docs/.gitignore` **committed** | clean | 1 passed | **262, SHIPPED** |

F-A.1 did not fire. F-A.2 did not fire. **F-A.3 did not fire**: the nested
variant is real, and its strong form — the nested `.gitignore` tracked — is the
one that matters, because this repository already carries
`scratchpad/reach/.gitignore`. The mechanism, read at hatchling 1.31.0 in the
uv cache: `vcs_exclusion_files` calls `locate_file(root, ".gitignore",
boundary=".git")`, which returns ONE file.

Vacuity: `--untracked-files=all` → `=no`, with `docs/zz_real_untracked.md`
present and shipping: **1 passed**. F-A.4 did not fire.

**Closed.** Enumeration is `os.walk` minus `git ls-files` — no exclusion source
is consulted, so none can hide anything. Classification is hatchling's rule,
with the root `.gitignore` evaluated by `git check-ignore` inside a throwaway
repository carrying nothing else (`core.excludesFile=/dev/null`, no
`info/exclude`, no nested `.gitignore`). `check-ignore` exit 0/1/128 is read as
matched / not-matched / **error**.

Two measurements the design turned on, neither of them reasoned:

* `git check-ignore` **stats** the path. Pattern `build/`, query `docs/build`:
  ignored when it is a directory, NOT ignored when it is a file or absent.
  Absent is the file answer, which is why candidates are not materialised —
  and it matches `pathspec`, which is what makes a FILE named `docs/build`
  ship (the divergence already written up at `_NOT_COPIED_DIRS`).
* `git ls-files --others` and `git status` both honour all four sources, so
  neither could have been the enumerator.

Positive controls (F-A.5): all three variants **FAILED**, each naming the path;
restored, **1 passed**.

Negative control (F-A.6): nine files hatchling genuinely does not ship planted
at once — `docs/zz_neg.log`, `docs/sub/zz_neg2.log`, `docs/zz_neg.pyc`,
`docs/__pycache__/zz_neg.py`, `src/stelling/__pycache__/zz_neg.py`,
`scratchpad/zz_neg_note.md`, `dist/zz_neg.md`, `.venv/lib/zz.py`,
`docs/build/zz_neg.md`. Guard: **1 passed**. Build: **260** members, none of
them present. F-A.6 did not fire.

Non-vacuity (F-A.7), four blinding mutants, with a real shipping file present:

| mutant | guard | parity control |
|---|---|---|
| the walk returns nothing | **FAILED** | **FAILED** |
| `git ls-files` returns nothing | **FAILED** | **FAILED** |
| the scan names nothing | passed | **FAILED** |
| everything is excluded | passed | **FAILED** |

The last two are why the parity test exists rather than being narrated. It
copies the tree into a real repository, plants eight files (four hidden or
admitted by four different mechanisms that must ship, four that must not),
builds an sdist and asserts the scan's answer **equals** the set of untracked
files in the tarball — over the whole tree, not just the probes. F-A.8 is
therefore checked on every run and not once by me. Driving its own documented
break (`cwd=root` on the `check-ignore` call) puts
`docs/zz_parity_info_exclude.md` and `docs/zz_parity_nested_gitignore.md` in
the "tarball shipped it, the scan did not name it" list.

### F-B — CONFIRMED, both hooks, both shapes, and closed

`grep -r` on this box: missing directory → **2**, empty directory → **1**
(GNU grep 3.11, and ugrep 7.5.0 agrees). F-B.2 did not fire.

|  | before | after |
|---|---|---|
| `mv src /elsewhere`, `jax-import-hygiene` | **Passed** | Failed |
| `mv src /elsewhere`, `library-identifier-hygiene` | **Passed** | Failed |
| `src/` exists, no `.py` under it, both hooks | **Passed** | Failed |

F-B.1 did not fire. F-B.3 did not fire: the bite table is unchanged where it
should be — bare `import jax` Failed, `import jax  # mirrors _jax_compat.py`
Failed, `jax._src` Failed, the real `_jax_compat.py` Passed, unmarked banned
identifier Failed, the same line marked `census` or `design/` Passed.

The exit-code half is not implied by the census: with a `chmod 000` directory
under `src/stelling` holding a real `import jax`, grep exits 2 having already
examined files, and both hooks now say so and Fail.

F-B.4: the third local hook, `staged-spdx-header`, is **not** an instance — it
already exits 1 with "this hook checked NOTHING" when there is no index, and
pre-commit reports "(no files to check) Skipped" rather than a pass when its
file list is empty. The two remote hooks are not grep-shaped.

### F-C — CONFIRMED and corrected

Step name at ci.yml:331 said "asserted off the exit code"; the body reads the
file channel and the comment beneath it says the exit code is last-writer-wins.
F-C.1 did not fire. F-C.2 did not fire: the other four verdict steps say "off a
file channel". After the edit, parsing ci.yml and listing every step whose name
mentions "verdict" gives five steps in five jobs, all saying "off a file
channel".

### F-D — recommendation: do NOT pin, and record why

F-D.1 did not fire — pinning does not break Trusted Publishing; the OIDC token
is minted for the job and exchanged by whatever code runs. **F-D.2 fired**: no
network, so any SHA would be unverifiable.

Three reasons, recorded at the job: pinning only the publish step is theatre
while `actions/download-artifact@v4` on the line above supplies the bytes; a
pin without automation becomes a stale pin, and stale means publishing without
PEP 740 attestations; an unverifiable SHA on the one job with `id-token: write`
is worse than a branch ref maintained by PyPA. The human action is one change:
`.github/dependabot.yml`, then all five actions pinned to `@<sha>  # vX.Y.Z`,
then a TestPyPI release to confirm attestations.

### F-E — the changelog read here, and not pinned

F-E.2 did not fire: the vendored 6.2.0 `CHANGELOG.md` carries v0.0.2 → v6.2.0,
so the window is covered — v6.0.0, v6.1.0, v6.1.1, v6.1.2, v6.2.0, and there is
no v5.1.2.

F-E.1 did not fire, and the bound is stronger than the prior audit's. Grepping
the window for `REUSE.toml|annotation|precedence|glob|path` returns **one** line
— the sort order of `reuse lint --lines`. Sharper: the newest release that
documents any change to `REUSE.toml` semantics is **v5.0.0** (2024-11-14);
support arrived in v4.0.0 and the precedence ordering in v2.0.0. Every one is
BELOW 5.1.1, so both pinned versions sit on the same side of every documented
change.

**SUSPECTED**, and labelled so in ci.yml: that bounds what the changelog
documents, not what 5.1.1 does. F-E.3 fired — the four steps a human must run
are written at the job, with 284/284 exit 0 as the figure to compare against
(285/285 on this branch, the extra file being this pre-registration).

### Standing falsifiers

* S.1 did not fire. `--collect-only`: 2299 ids on both series, byte-identical
  between them; against `main`'s 2298 the diff is exactly one added line,
  `tests/test_sdist_contents.py::test_the_untracked_scan_agrees_with_the_tarball`.
* **S.2 fired, as designed**: 2296 → **2297 passed, 2 skipped** on both series,
  the +1 being that one test. Re-measured in all three environments rather than
  adjusted by arithmetic, and the record above `jobs:` in ci.yml updated to
  match: jax 0.11.0 2297/2, jax 0.10.2 2297/2, no jax 1186/85, `verdict=made`
  in all three. The no-jax figure comes from an interpreter that genuinely has
  no jax (site-packages symlinked minus `jax*`/`jaxlib*`/`jax_md*`), because
  `_optional.available` asks `find_spec`.
* S.3 did not fire: `reuse lint` 285/285, exit 0. `pre-commit run --all-files`
  twice: rc=0, rc=0.
* S.4 did not fire: nothing under `src/stelling/`, nothing in `docs/norms.md`,
  `propagate.py` or `test_probe_witness.py`.
* S.5 did not fire: worktree clean, no probe files.

sdist **260 members before, 260 after, member list byte-identical**.

### One finding OUT OF SCOPE, reported and not fixed

`test_every_root_entry_is_a_decision` **FAILS when the suite is run from an
unpacked sdist**, which is what a distribution packager does. The root of an
unpacked sdist carries `PKG-INFO`, which is in neither the allowlist nor
`WITHHELD`. Driven against `main`'s own tarball as well as this branch's, so it
is pre-existing and is not this branch's doing:

    tar xzf stelling-0.1.0.tar.gz && pytest tests/test_sdist_contents.py
    main    1 failed, 5 passed, 1 skipped
    here    1 failed, 6 passed, 1 skipped

Left alone deliberately: the obvious repair (a `WITHHELD["PKG-INFO"]` entry)
puts a generated artefact in a dict whose every other key is a path in the
repository, and that is a decision for whoever owns the allowlist, not a
drive-by.
