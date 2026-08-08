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

---

# REPAIR PASS — the audit's seven, measured before repaired

Appended below the record above; nothing above this line is edited. Written by
a second agent, on top of 10120d9, against the blinded audit's
SHOULD-LAND-WITH-FIXES verdict. **Every finding was reproduced before it was
touched**, in standalone throwaway repositories built with `git archive` +
`git init` and never by mutating a shared worktree. Environment as measured:
hatchling **1.31.0** (what `uv build --offline` resolves), pathspec 1.1.1, git
2.43.0, reuse 6.2.0, jax 0.11.0 / 0.10.2.

## Reproduction — all six measurable findings CONFIRMED, none refuted

Baseline for every count below: **260 members**, all of them files, all
prefixed `stelling-0.1.0/`, no directory members.

| # | finding | reproduced | measurement |
|---|---|---|---|
| 1 | ancestor `.hgignore` is force-included | YES | 261 members, `stelling-0.1.0/.hgignore` present, suite `8 passed` |
| 1b | no `.git` at build root -> ancestor `.gitignore` shipped AND applied | YES | 240 members, `docs/` gone, tarball `.gitignore` == the ancestor's text |
| 2 | checkout path matching `.gitignore` discards it all | YES | `<..>/.cache/repo` 262 members with `docs/zz_secret.log` + `docs/htmlcov/index.html`; `<..>/ok/repo` 260; `8 passed` in both |
| 3 | `WITHHELD` ships the file anyway | YES | plant `.hgignore`: `1 failed`; move to WITHHELD: `8 passed`; build: 261, shipped |
| 4 | false RED on tracked `docs/.hatch/note.md` | YES | `2 failed, 6 passed`, tarball 260, no `.hatch` member |
| 4b | submodule is a TRUE red | YES | `docs/sub` named, 261 members, `stelling-0.1.0/docs/sub/payload.md` present |
| 5 | both non-vacuity assertions unpinned | YES | delete `assert not unseen` -> `8 passed`; `assert True or tracked` -> `8 passed` |
| 6 | stale reuse figure | YES | 9c37995 284/284, 10120d9 285/285, e8b9377 (`main`) 292/292 |
| 7 | the prose claim is incomplete | YES | hatchling reads `.hgignore` too, and force-includes both |
| PKG-INFO | decision test fails from an unpacked sdist | YES | `1 failed, 6 passed, 1 skipped` |

The audit's realism list for finding 2 was checked pattern-by-pattern against
this repository's own `.gitignore` and reproduces exactly: `~/.cache/x/stelling`,
`~/lib/...`, `/var/lib/ci/...`, `/srv/dist/...`, `~/.env/...` all match;
`/home/runner/work/stelling/stelling` does not. Adding `~/build/...`,
`~/venv/...` and `/opt/target/...`, which also match, and `~/site/...`, which
does not (`/site` is root-anchored). GitHub-hosted CI is safe; a developer box
may not be.

## Controls, per fix

**Fix 1 — force-include.** POSITIVE: ancestor `.hgignore` planted above a lab
checkout, `--basetemp` set inside the plant so the parity copy inherits it —
`2 failed`, `test_no_untracked_file_anywhere_would_ship` naming
`.hgignore <- <parent>/.hgignore  (OUTSIDE the tree)` and
`test_the_untracked_scan_agrees_with_the_tarball` naming it as a member with
no counterpart. NEGATIVE: same tree, no plant — `12 passed`. BLINDING, and
this is the part that matters: the two halves are INDEPENDENT. Blind
`_force_included` to `{}` and the parity test still reddens; delete the parity
orphan assertion and the scan test still reddens.

**Fix 2 — checkout path.** POSITIVE: lab checkout at `<..>/.cache/repo` with
the two plants — `2 failed`, the dedicated test naming the path and the scan
naming `docs/htmlcov/index.html` and `docs/zz_secret.log` as would-ship (the
scan now models the bail-out too, so it is right for the first time at that
path). NEGATIVE: `<..>/ok/repo` — `12 passed`. BLINDING: neuter `_check_ignore`
to `set()` and the dedicated test's own instrument-liveness assertion fires
before its verdict is read, so a dead instrument cannot report "safe".

**Fix 3 — WITHHELD.** POSITIVE, both branches of the false dichotomy: root
`.hgignore` undecided -> `AssertionError: ... in none of pyproject's sdist
allowlist, WITHHELD or GENERATED_IN_DISTRIBUTION: .hgignore`; moved into
WITHHELD -> `AssertionError: these root paths are recorded in WITHHELD and
hatchling FORCE-INCLUDES them`. The escape hatch is closed. It also caught a
live entry on its first run: `.gitignore` was in WITHHELD *and* in the
allowlist *and* force-included, which is the exact shape it exists to catch.
NEGATIVE: unpacked sdist, `11 passed, 1 skipped` (was `1 failed, 6 passed,
1 skipped`) — PKG-INFO taken, in `GENERATED_IN_DISTRIBUTION` rather than
WITHHELD, because "exists only in a distribution" is not "exists here and is
withheld".

**Fix 4 — `tracked - pruned` in `walked`.** NEGATIVE (the false red):
tracked `docs/.hatch/note.md` -> `12 passed`, 260 members, no `.hatch` member.
POSITIVE (the true red preserved): submodule at `docs/sub` -> still red,
`docs/sub` named, 261 members, `docs/sub/payload.md` shipped.

**Fix 5 — pinning.** Three tests, no monkeypatching: an empty index, a walk
blinded by deleting the one tracked file from disk, and a classifier asked to
separate three inputs. Each of the old mutations now reddens.

**Blinding matrix.** Eight blindings applied to the repaired module in lab
repositories, each run as the whole file; every one RED, and each caught by
something other than itself:

| blinding | caught by |
|---|---|
| `_walked_files` -> `set()` | scan + parity |
| `_tracked_files` -> `_walked_files` | parity + both new pinning tests |
| `_hatchling_excluded` -> everything | parity + classifier test |
| `_hatchling_excluded` -> nothing | parity + classifier test |
| parity `members` -> `[]` | parity |
| `_check_ignore` -> `set()` | checkout-path + parity + classifier |
| `_force_included` -> `{}` (+ancestor `.hgignore`) | parity |
| parity orphan assertion deleted (+ancestor `.hgignore`) | scan + parity |

## Figures

* sdist **260 members before, 260 after**; the repair adds no distributed file.
* `tests/test_sdist_contents.py`: **8 -> 12** tests.
* Whole suite: **2297 passed, 2 skipped -> 2301 passed, 2 skipped**, on BOTH
  series (jax 0.11.0 and jax 0.10.2), measured in this checkout with
  `PYTHONPATH` verified against `stelling.__file__`.
* `--collect-only`: 2299 -> 2303 ids, diff is exactly four ADDED lines and no
  removals or renames —
  `test_the_checkout_path_does_not_disable_the_exclusions`,
  `test_the_exclusion_classifier_discriminates`,
  `test_the_scan_refuses_a_blinded_walk`,
  `test_the_scan_refuses_an_empty_index`.
* `reuse lint`: 285/285 exit 0 before, 285/285 exit 0 after (no file added).
* No new skip reason: both new skips reuse `"needs git"`, already registered.

## Judgements the audit asked for

**PKG-INFO: taken.** It is the same dict and the same question, and leaving it
meant the suite could not be run green from the artefact it guards. It is not
in WITHHELD — a dict that means "not distributed" cannot also hold a key that
means "nothing but distributed" without becoming unreadable.

**The no-`uv` skip: it can be made noisier, and should not be made a failure.**
On a machine without `uv` the parity test skips and the scan runs, and the
scan is a MODEL — the `git check-ignore` / `pathspec` divergence on 7 of 22
pattern shapes has no guard at all in that configuration, and every one of
those divergences was in the smuggle direction. Turning the skip into an error
would be flaky in exactly the environment where it matters least (a
contributor's first `pytest`) and is not proposed. What is proposed, and NOT
done here because it touches the CI files a concurrent pass owns: the release
workflow already builds the sdist, so it is the place to assert that `uv` was
present — a job-level `uv --version` step whose absence fails the release
rather than the developer. Recorded as the next pass's, not silently dropped.

## Still not fixed, and deliberately

* `git check-ignore` vs `pathspec` on 7 of 22 pattern shapes. Not made worse:
  the one new use of the instrument (the checkout-path test) was cross-checked
  against `pathspec.GitIgnoreSpec.match_file` on 13 candidate checkout paths
  and agreed on all 13.
* A `hatch_build.py` build hook can force-include anything at build time and
  no model here can know what. This project has none; the parity test is what
  would notice one. Written into the module docstring rather than left implicit.
* **SUSPECTED, not measured**: that the `check-ignore`/`pathspec`
  correspondence used by `_vcs_exclusions_are_discarded` holds for pattern
  shapes outside the 13 paths tried. The known divergence classes (negation
  re-including inside an excluded directory, POSIX character classes) do not
  appear in this repository's `.gitignore`, but that is an observation about
  today's file.

# SECOND REPAIR PASS — the audit's seven, and one of them does not reproduce

Appended below the record above; nothing above this line is edited. Branch
`fix/sdist-force-include-parity`, off `main` at **a4c16fe**, in a worktree at
`/home/nick/MSF/.wt-sdist3/W`. Every construction below was built in a
standalone throwaway repository made by `git archive` of this tree + `git init`
+ one commit, never by mutating a shared worktree. Environment as measured:
hatchling **1.31.0** (what `uv build --offline` resolves), pathspec 1.1.1, git
2.43.0, reuse 6.2.0, jax 0.11.0 / 0.10.2.

**Baseline for every member count below: 261 members**, all files, all prefixed
`stelling-0.1.0/`, no directory members. The audit's baseline was 260; the tree
has grown one file since. Every count it reports is one lower than the same
count here, and they agree on every difference.

**All blinding measured under the DEFAULT `tmp_path`.** The previous pass's
matrix put `--basetemp` inside the plant directory, which makes the staged copy
inherit the planted ancestor — that manufactures the independence it was
measuring, and it is the reason F1's independence claim read as true.

## Reproduction — six CONFIRMED, one REFUTED as stated

| # | finding | reproduced | measurement |
|---|---|---|---|
| F1 | the force-include check skips where its route opens | **YES** | `.git`-stripped copy + ancestor `.hgignore`: `11 passed, 1 skipped`; build 262 members; `stelling-0.1.0/.hgignore` SHIPPED. Controls: no plant -> `11 passed, 1 skipped`, 261 members; plant WITH `.git` -> `1 failed`, 262 members |
| F1b | "deliberately independent — blind either one and the other still reddens" | **YES, the claim is FALSE** | hazard planted and shipping, default `tmp_path`: `_force_included -> {}` 12 passed; `_forced_without_a_reviewed_source -> []` 12 passed; `.hgignore` out of `_HATCH_VCS_EXCLUSION_FILES` 12 passed; unblinded `1 failed` |
| F2 | `lying` matches the full path, every other consumer matches the first component | **YES** | `license-files = ["LICENSE", "scratchpad/PREREG*.md"]`: `12 passed`, 271 members, **10** WITHHELD `scratchpad/PREREG*.md` in the tarball (the audit said 268 / 8; this tree now has ten) |
| F3 | `BuilderConfig.force_include` unmodelled and unnamed | **YES** | `[tool.hatch.build] force-include = {"scratchpad/PREREG_SDIST.md" = "scratchpad/PREREG_SDIST.md"}`: `12 passed`, 262 members, shipped |
| F4a | "a drift in any of the three constants shows up there as a set difference" | **YES, the claim is FALSE** | `__pycache__` out: 12 passed. `.hatch` out: 12 passed. `.DS_Store` out: 12 passed. `_HATCH_FORCED_ROOT_FILES = ("pyproject.toml",)`: 12 passed. Only `*.py[cdo]` out of `_HATCH_DEFAULT_GLOBAL_EXCLUDE` reddens — and in `test_the_exclusion_classifier_discriminates`, not in the parity test the sentence points at |
| F4b | "if a `hatch_build.py` appears, the parity test is what will notice" | **YES, the claim is FALSE** | committed, allowlisted, registered hook force-including a TRACKED file: `12 passed`, 263 members, `hatch_build.py` AND `scratchpad/PREREG_SDIST.md` in the tarball |
| F5 | `GENERATED_IN_DISTRIBUTION` is an unguarded escape hatch | **YES as a defect, REFUTED as stated** | see below |
| F6a | dangling symlink misdiagnosed as FORCE-INCLUDED | **YES** | committed `docs/zz_dangling.md -> zz_nowhere.md`: parity red, message "it was FORCE-INCLUDED"; it was not |
| F6b | tracked dir-symlink misdiagnosed as a broken walk | **YES** | committed `docs/zz_linkdir -> <empty dir outside the tree>`: `assert not unseen` red with "the walk is not looking at the tree git is looking at" |
| F7 | three tests skip without `uv`, not one | **YES** | `PATH` without `uv`, at a4c16fe: `9 passed, 3 skipped`. The three are `test_built_metadata_carries_no_relative_reference`, `test_an_arbitrary_new_file_does_not_ship` and `test_the_untracked_scan_agrees_with_the_tarball` |

### F5 — the defect is real; the construction the audit names is not

The audit: *"Adding `.hgignore`: 'generated, honest' to that dict reportedly
takes the **ancestor**-`.hgignore` case from `1 failed` to `12 passed` with the
file still shipping."* Driven, all four cells, at a4c16fe:

| construction | plain | + `".hgignore": "generated, honest"` |
|---|---|---|
| **ancestor** `.hgignore`, `.git` present | `1 failed` | **`1 failed`** — unchanged |
| root `.hgignore`, untracked | `2 failed` | `1 failed` |
| root `.hgignore`, **committed** | `1 failed` | **`12 passed`**, 262 members, SHIPPED |

`GENERATED_IN_DISTRIBUTION` is consulted in exactly two places, and neither is
the force-include review, which is what catches the ancestor case — so no entry
in that dict can silence it. The hatch opens on a **root, committed**
`.hgignore`, where the only red was `test_every_root_entry_is_a_decision`'s
"undecided" and the dict is precisely the way to make a root path decided
without deciding anything. The audit's account of *why* is right — it is the
escape hatch `WITHHELD` used to be, unguarded in exactly the way `WITHHELD` was
before the `lying` assertion — and its construction is wrong. Fixed for the
shape that reproduces, and for the ancestor shape too, since `_falsely_recorded`
now reads a force-included path recorded in either dict as false.

### And the audit's proposed single fix does not work

*"A check of the form `shipped ⊆ tracked ∪ scanned ∪ GENERATED_IN_DISTRIBUTION`
catches all three at once."* Implemented verbatim on top of a4c16fe's parity
test and driven against all three constructions:

| construction | with the subset check added |
|---|---|
| widened `license-files` | `12 passed` |
| static `force-include` table | `12 passed` |
| `hatch_build.py` hook | `12 passed` |

All three ship files that ARE tracked in the staged tree, so the subset holds.
It is also already implied by the existing `scanned == shipped_untracked`
equality (`shipped` has `GENERATED` subtracted before it), so it adds no power
even in principle. **Not adopted.** What catches the three is: first-component
+ both-dicts matching on the force-include model (constructions 1 and 2, once
the static table is modelled), and refusing the build-hook capability outright
(construction 3), because a hook that ships a tracked file satisfies every
counterpart check a comparison against the tree can make.

## Controls, per fix — positive, negative, blinding

Trees: `f1o` pristine standalone repo; `f1p` `.git`-stripped + ancestor
`.hgignore`; `f1n` `.git`-stripped, no plant; `c1` widened `license-files`;
`c2b` static `force-include`; `c3` registered hook; `f5c` committed root
`.hgignore` + `GENERATED` entry; `sym_out` tracked dir-symlink;
`sym_dangling` committed dangling symlink. Module counts: **12 tests at
a4c16fe, 17 here.**

**F1 — the force-include check moves out from behind the `.git` skip.**
POSITIVE `f1p`: `1 failed, 15 passed, 1 skipped`, naming
`.hgignore <- <parent>/.hgignore (OUTSIDE the tree)`. NEGATIVE `f1n`:
`16 passed, 1 skipped`, 261 members. NEGATIVE `f1o`: `17 passed`, 261 members.
BLINDING, default `tmp_path`, **on the PRISTINE tree** (so the red is the
guard refusing to be blinded and not the hazard):

| blinding | a4c16fe | here |
|---|---|---|
| `_force_included -> {}` | 12 passed | **2 failed** |
| `_forced_without_a_reviewed_source -> []` | 12 passed | **1 failed** |
| `.hgignore` out of `_HATCH_VCS_EXCLUSION_FILES` | 12 passed | **1 failed** |

The first is caught by a non-vacuity assertion that `_force_included(REPO)` is
non-empty; the other two by `test_the_force_include_review_sees_an_outside_
source`, a synthetic root with an ancestor `.hgignore` planted under
`tmp_path`, whose negative half removes the plant and asserts silence.

**F2/F3/F5 — the two dicts and the static table.** POSITIVE: `c1` `1 failed`,
`c2b` `2 failed`, `f5c`+entry `2 failed`, ancestor+entry `3 failed`.
NEGATIVE: `f1o` `17 passed`. BLINDING, default `tmp_path`:

| blinding | pristine | on the construction |
|---|---|---|
| `_falsely_recorded -> None` | **1 failed** | still red |
| `_static_force_include -> {}` | **1 failed** | still red |
| parity `dishonest -> []` | 17 passed | still red (root-entry test) |
| parity `unused -> []` | 17 passed | still red (root-entry test) |

The last two are absence checks over a one-entry dict, so a pristine tree has
nothing for them to find; what makes them non-droppable is that the
construction they exist for is red from the other half. That is the
independence the audit asked for and did not get from parity.

**F4b — the build hook.** POSITIVE `c3`: `1 failed`, naming `hatch_build.py`
and the registered `[tool.hatch.build.targets.sdist.hooks.custom]` table.
NEGATIVE `f1o`: `17 passed`. BLINDING: pointing the file half at a name that
does not exist leaves the registered-table half red on `c3`; the table half has
a non-vacuity assertion that `[tool.hatch.build.targets.*]` is non-empty.

**F4a — the constants.** POSITIVE, one mutation at a time, pristine tree:

| mutation | a4c16fe | here | caught by |
|---|---|---|---|
| `__pycache__` out of `_HATCH_EXCLUDED_DIRECTORIES` | 12 passed | **1 failed** | the static-table control's directory-source pruning |
| `.hatch` out | 12 passed | **1 failed** | the parity build (`docs/.hatch/` plant) |
| `.DS_Store` out of `_HATCH_EXCLUDED_FILES` | 12 passed | **1 failed** | the parity build (`docs/.DS_Store` plant) |
| `_HATCH_FORCED_ROOT_FILES = ("pyproject.toml",)` | 12 passed | **1 failed** | the static-table control, literal names |
| `*.py[cdo]` out of `_HATCH_DEFAULT_GLOBAL_EXCLUDE` | 1 failed | 1 failed | the classifier control |
| `.hg` out of `_HATCH_EXCLUDED_DIRECTORIES` | 17 passed | **17 passed** | nothing — recorded |
| `_HATCH_DEFAULT_LICENSE_GLOBS = ("ZZ_NEVER*",)` | 17 passed | **17 passed** | nothing — recorded |

NEGATIVE: the two new parity plants are `False` entries — the artefact must NOT
contain them, and it does not, so no member count moves and nothing cries wolf.
`.hatch` and `.DS_Store` were chosen because they are among the only names in
those two constants that this repository's own `.gitignore` does not also
cover; `git check-ignore --no-index` says the other eight excluded-directory
names are gitignored here, which is exactly why dropping them was silent.

**F6 — the two messages.** POSITIVE `sym_dangling`: parity red, now reading
"the tree has something at that path that is not a regular file — a dangling
symlink" and naming the link target, instead of "it was FORCE-INCLUDED".
POSITIVE `sym_out`: red, now reading "these tracked paths are SYMLINKS TO
DIRECTORIES … the walk is looking at exactly the tree git is". NEGATIVE `f1o`:
`17 passed` — no ordinary tree has either shape. BLINDING: `dir_links = []`
returns the path to `unseen` and the suite stays red, so the split cannot
silently drop anything.

One measured correction to my own first draft of that message. It said
`tarfile` resolves a symlink's target when it stores the member. It does not:
on the committed dangling link the build SUCCEEDS and the member is stored as
a symlink — `issym=True`, `size=0`, `linkname` preserved. The message says
that now. A second measured fact, on `docs/zz_linkdir -> ../design` inside the
tree: the tarball carries **zero** members at or under the link, because
`safe_walk`'s `(st_dev, st_ino)` seen-set has already visited `design/` — and
it also drops `design/`'s own 64 files when the link is walked first, which is
its own true red.

**F7 — the release assertion.** The step's script extracted verbatim from
`release.yml` and driven against real built sdists:

| tree | rc | first line |
|---|---|---|
| pristine | 0 | `every one of 261 sdist members is committed to this tree` |
| ancestor `.hgignore` | 1 | names `.hgignore` |
| untracked `docs/zz_leak.md` | 1 | names it |
| hook shipping a TRACKED file | 0 | in scope for the test module, not for this question |
| `members.txt` emptied (blinding) | 1 | "the sdist check examined nothing" |
| `tracked.txt` emptied (blinding) | 1 | "the sdist check examined nothing" |

The one blinding it cannot self-pin is deleting its own comparison; nothing in
a shell step can. It also asserts four members are present (`PKG-INFO`,
`pyproject.toml`, `README.md`, `LICENSE`) before believing any absence.

## Figures

* module: **12 tests at a4c16fe -> 17 here**; `12 passed` -> `17 passed`.
* suite: `2362 passed, 2 skipped` at `main` -> **`2367 passed, 2 skipped`**
  on BOTH series — jax 0.11.0 in 148.14s (load average 7.14 at the end),
  jax 0.10.2 in 146.07s (load average 8.19). `--collect-only` ids **identical
  between the two series** (2369 each, `diff` empty); against `main`'s 2364 the
  diff is exactly the five added test ids and nothing else.
* sdist members, this tree: **261 before, 261 after** — no fix changes what
  ships.
* `reuse lint`: **301/301 rc=0** at `main`, **301/301 rc=0** here. No files
  added or removed.
* no `uv` on `PATH`: `9 passed, 3 skipped` at a4c16fe, **`14 passed,
  3 skipped`** here. The three skipped are unchanged.

## Judgements the audit asked for

**F3, "model it or name it; say which and why": MODELLED.** It is a static,
diff-reviewable table in `pyproject.toml` with no code in it, its resolution is
forty lines of hatchling that can be transcribed exactly
(`builders/config.py:678-704`, `builders/utils.py`), and it is strictly easier
to introduce than the `hatch_build.py` route the module already named. Naming
it would have left the cheaper route as the unguarded one.

**F1's suggested shape: adopted, both halves.** The check is out from behind
the `.git` skip and degrades rather than skipping when the index is unreadable,
and `_force_included(REPO)` is asserted non-empty.

**F6b: kept RED, diagnosis corrected, not made green.** A tracked directory
symlink ships nothing at its own path, so there is a case for subtracting it
the way `_pruned_by_the_walk` is subtracted. It is not taken: everything
hatchling reaches THROUGH the link ships under the link's path whenever the
seen-set has not already visited the target, and a link out of the tree
therefore distributes files nobody committed here. The finding was that the
diagnosis pointed at the wrong mechanism, and that is what changed.

## Still not fixed, and deliberately

* Eight of the eleven names in `_HATCH_EXCLUDED_DIRECTORIES`, and all of
  `_HATCH_DEFAULT_LICENSE_GLOBS`, are held by nothing. Measured, not assumed
  (`.hg` and a `ZZ_NEVER*` license glob both leave 17 passed). Holding them
  would mean un-gitignoring those names in this repository or planting a file
  per name; the two that could be held without either were.
* `project.readme` in its table form (`{text = "…"}`) carries no path and is
  still not exercised.
* `git check-ignore` vs `pathspec` on 7 of 22 pattern shapes — unchanged from
  the previous pass, and the release-workflow assertion added here is the first
  guard on that class that does not depend on the model at all.
* **SUSPECTED, not measured**: that `_static_force_include` matches hatchling
  for a source key that is absolute or `~`-prefixed. The code path is written
  and the transcription is from the upstream source; no build was made with
  such a key, because both shapes are `(OUTSIDE the tree)` reds anyway.
* **SUSPECTED, not measured**: that the release-workflow step behaves on
  GitHub's runner as it does here. It was driven with `bash` locally on real
  artefacts; it has not run under `actions/checkout@v4`, whose index is the
  thing `git ls-files` reads there.
