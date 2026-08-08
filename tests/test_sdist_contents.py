# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What the sdist ships is an ALLOWLIST, and this holds it shut.

The sdist is the one artefact where a mistake is immutable: it cannot be
unpublished from PyPI. Hatchling's *default* sdist takes everything not
gitignored — tracked or not — which is fail-OPEN, and it shipped an internal
file that was protected only by being uncommitted. `.git/info/exclude` does not
protect against it either.

**What hatchling reads as VCS exclusions, measured against 1.31.0's own source
rather than summarised**: `builders/config.py:752-763` locates TWO files, each
with `hatchling.utils.fs.locate_file`, which walks UP from the build root until
it finds the file or hits a boundary directory —

* ``.gitignore``, boundary ``.git``
* ``.hgignore``, boundary ``.hg``

so in a git checkout the ``.hgignore`` search has no boundary at all and walks
to ``/``. It does *not* read the user's `core.excludesFile`, `.git/info/exclude`
or a `.gitignore` in any subdirectory. `git status` reads all four of git's
sources; every source in that difference is a file the guard below cannot see
and the tarball can, which is what
``test_the_untracked_scan_agrees_with_the_tarball`` measures rather than
narrates.

**An earlier version of this docstring said hatchling reads "ONE `.gitignore` …
and no other exclusion source". That was false, and this module is the wrong
place for a false claim.** Both located files are also FORCE-INCLUDED into the
sdist (`builders/sdist.py:338-340`), keyed by ``os.path.basename``, and
force-include bypasses the allowlist entirely. Driven, in a standalone repo
built by `git archive` from this tree::

    printf 'syntax: glob\\nzz_never\\n' > <parent-of-repo>/.hgignore
    uv build --offline --sdist .   ->  261 members (baseline 260)
                                       stelling-0.1.0/.hgignore  PRESENT

— a file from OUTSIDE the repository, distributed under this project's name.
And with no ``.git`` at the build root and no root ``.gitignore`` (an unpacked
sdist, a `.git`-stripped container copy), the ANCESTOR's ``.gitignore`` is both
applied as exclusions and shipped as ``/.gitignore``::

    <parent>/.gitignore  =  "zz_ancestor_secret\\ndocs/\\n"
    uv build --offline --sdist .   ->  240 members, docs/ GONE,
                                       stelling-0.1.0/.gitignore = the ancestor's

Both routes are modelled by :func:`_force_included` and both are caught
structurally by the parity test, which now asserts that **every** tarball
member has a counterpart in the tree it was built from.

WHAT IS STILL NOT MODELLED HERE, said out loud rather than left to be
discovered: a `hatch_build.py` build hook can put anything at all into
``build_data["force_include"]`` at build time, and this module cannot know what
without executing it; ``project.readme`` in its table form
(``{text = "…"}``) carries no path and is not exercised; and
``git check-ignore`` — the instrument :func:`_check_ignore` uses — is not
`pathspec`, the two disagree on 7 of 22 pattern shapes tried (negations that
re-include inside an excluded directory, POSIX character classes), and every
disagreement found was in the smuggle direction. Only the parity test, which
needs `uv`, catches that class; a machine with no `uv` runs the scan alone and
has no guard for it.

Three tests, in the order they matter:

1. ``test_an_arbitrary_new_file_does_not_ship`` — the real property, by
   INTERVENTION. Drop a file the allowlist has never heard of, build, and
   confirm it is absent. *Absence of a NAMED file is the weaker check and is
   exactly what would have passed before the leak*: the checklist was not
   named anywhere, it simply was not excluded.

2. ``test_every_root_entry_is_a_decision`` — a new path at the repo root must be
   either allowlisted or listed here as deliberately withheld. It cannot be
   neither. This one needs no build backend, so it runs everywhere and fails
   closed when someone adds a file and does not think about distribution.

3. ``test_the_untracked_scan_agrees_with_the_tarball`` — the SUBDIRECTORY half,
   which is where the root allowlist stops reaching, held against the built
   artefact and not against anybody's reading of hatchling.

**The intervention is performed on a COPY of the tree, and it did not used to
be.** ``test_an_arbitrary_new_file_does_not_ship`` wrote
``zz_sdist_allowlist_probe.txt`` into the REPO ROOT of the checkout it was
running in, which is shared state, and two of the tests in this same file read
that root. Driven — two suite runs, one checkout, 0.4s apart::

    run A   7 passed
    run B   2 failed
            test_no_untracked_file_anywhere_would_ship  (saw A's probe)
            test_an_arbitrary_new_file_does_not_ship    ("probe path is
                                                          already taken")

The first of those is a spurious RED in a test that has nothing to do with
this one. The second is worse than it looks: the racing run's ``finally``
unlinks the probe, so a build that started before the unlink and read the tree
after it sees no probe — and the old test would have passed, having observed
nothing at all. Which is the second defect:

**the old test passed when the probe was never created.** Driven, by replacing
``probe.write_text(...)`` with ``pass``::

    pytest tests/test_sdist_contents.py::test_an_arbitrary_new_file_does_not_ship
    1 passed

Nothing in it re-read that the intervention had happened, so ``leaked == []``
could not tell "the allowlist held" from "there was nothing to hold out". Both
are answered below: the tree under test is a private copy, and the build is
made to prove it can SEE an untracked file before its silence about one is
believed.
"""

from __future__ import annotations

import email
import glob as globlib
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import tomllib
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _allowlist() -> set[str]:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    inc = cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    # entries are rooted ("/src"); compare on the bare name
    return {entry.lstrip("/") for entry in inc}


# Root entries that exist HERE and are deliberately NOT distributed. Each needs
# a reason, because "why is this not in the sdist" is the question a future
# reader will ask. A path in none of the three sets fails the test.
#
# **MEMBERSHIP HERE IS A RECORD, NOT AN ENFORCEMENT**, and the test's message
# used to imply otherwise — "Add each to the allowlist (it ships) or to WITHHELD
# with a reason (it does not)". For a FORCE-INCLUDED path that dichotomy is
# false, and taking the second branch ships the file with the suite green.
# Driven, in a standalone repo from this tree::
#
#   plant a root `.hgignore`                 test_every_root_entry_is_a_decision
#                                              1 failed
#   add ".hgignore" to WITHHELD              8 passed
#   uv build --offline --sdist .             261 members,
#                                              stelling-0.1.0/.hgignore SHIPPED
#
# What keeps a path out is `[tool.hatch.build.targets.sdist].include` — and not
# even that, for the handful hatchling force-includes. So the test now checks
# this dict against :func:`_force_included` and reddens when an entry claims to
# withhold something the backend ships anyway.
WITHHELD = {
    ".git": "the repository itself",
    ".claude": (
        "local agent/tool configuration — untracked AND not gitignored, so the "
        "allowlist is the only thing keeping it out. It is currently EMPTY, so "
        "nothing of substance would have shipped; the point is that it was "
        "undecided, and the decision is recorded before content appears"
    ),
    # NOTE the key that is NOT here any more: `.gitignore` used to carry the
    # reason "listed in the allowlist; kept here only if it moves" — a path
    # recorded as WITHHELD while it ships, and one hatchling FORCE-INCLUDES so
    # that no allowlist decides it either way. It is the exact shape the new
    # assertion in `test_every_root_entry_is_a_decision` exists to catch, and
    # it caught this one on its first run. A dict whose entries mean "not
    # distributed" cannot also hold entries that mean "distributed, noted here
    # in case it moves" — the second kind is what makes the first kind
    # unreadable.
    "stelling_0_1_0_release_checklist.md": (
        "internal release working document, deliberately untracked — this is the "
        "file whose leak motivated the allowlist"
    ),
    "dist": "build output",
    "build": "build output",
    ".venv": "local environment",
    "venv": "local environment",
    ".pytest_cache": "test cache",
    ".ruff_cache": "lint cache",
    ".mypy_cache": "type-check cache",
    "__pycache__": "bytecode cache",
    ".pdm-build": "build backend scratch",
    "uv.lock": "a lock file pins an environment; a library must not ship one",
    # NOTE the key above: `scratchpad` appeared TWICE in this dict, with two
    # different reasons, and Python keeps the last — so the earlier reason was
    # dead text that read like a decision. The two are merged here.
    "scratchpad": (
        "per-session working directory, and per-branch audit working notes: "
        "the pre-registration a repair pass writes before it measures "
        "anything, its probes, and its measurement logs. It is a record of "
        "what was PREDICTED, so it is kept in the tree it registers AGAINST — "
        "a registration a reader cannot reach from the repository it "
        "constrains is not a registration. "
        "Tracked, because a pre-registration that can be edited after the fact "
        "is not one; withheld, because it is a record of how this repository "
        "was checked and not part of the library a user installs. WHAT KEEPS "
        "IT OUT OF THE SDIST IS NOT THIS ENTRY: membership here is not a "
        "build exclusion at all — `[tool.hatch.build.targets.sdist].include` "
        "is, and `/scratchpad` is not in it, which "
        "`test_an_arbitrary_new_file_does_not_ship` establishes by "
        "intervention rather than by naming a path. AND IT HAS ONE "
        "CONSEQUENCE THE OTHER ENTRIES DO NOT: "
        "`test_no_untracked_file_anywhere_would_ship` skips on "
        "`path.split('/', 1)[0] in WITHHELD`, so this entry exempts the whole "
        "`scratchpad/` SUBTREE from the untracked-file check, not just the "
        "directory. That exemption used to be 'harmless while nothing inside "
        "it would ship', which is a hope; it is now an assertion — the check "
        "computes what hatchling would ship BEFORE consulting this dict and "
        "reddens if any WITHHELD prefix ever masks a shipping path"
    ),
}


# The OTHER kind of root entry, and the reason WITHHELD had to split in two.
# WITHHELD means "exists in this checkout and is not shipped". These exist ONLY
# in a distribution and are generated by the backend, so they can be neither
# allowlisted nor withheld — and `test_every_root_entry_is_a_decision` failed
# outright when the suite was run from an unpacked sdist, which is a place the
# suite is meant to be runnable. Driven::
#
#   tar xzf stelling-0.1.0.tar.gz && cd stelling-0.1.0
#   pytest -q -ra tests/test_sdist_contents.py
#     1 failed, 6 passed, 1 skipped
#     AssertionError: these root paths are neither in pyproject's sdist
#     allowlist nor in WITHHELD:  PKG-INFO
#
# Recording it here rather than in WITHHELD keeps the two questions apart: a
# reader asking "why is this not distributed" must not be handed an entry whose
# answer is "it is nothing BUT distributed".
GENERATED_IN_DISTRIBUTION = {
    "PKG-INFO": (
        "written by the build backend from the project metadata. It exists at "
        "the root of an unpacked sdist and never in this checkout, so it is "
        "not an allowlist entry (the allowlist names tree paths) and not a "
        "WITHHELD one (it is not withheld — it is the one member the tree "
        "cannot supply). `test_the_untracked_scan_agrees_with_the_tarball` "
        "subtracts exactly this name when it demands that every other member "
        "have a counterpart in the tree"
    ),
}


def test_every_allowlist_entry_exists() -> None:
    """An allowlist that names a deleted path rots silently and stops
    protecting the thing it was written for."""
    allow = _allowlist()
    # An EMPTY include list is a shape this test used to pass over in silence:
    # `missing` would be empty and the assertion below would be vacuous. A
    # missing table raises KeyError in `_allowlist` and is already loud; an
    # empty one has to be said out loud here.
    assert allow, (
        "pyproject's sdist allowlist is empty — this test would then be "
        "asserting nothing about anything"
    )
    missing = sorted(e for e in allow if not (REPO / e).exists())
    assert not missing, (
        "pyproject's sdist allowlist names paths that no longer exist:\n  "
        + "\n  ".join(missing)
    )


def test_every_root_entry_is_a_decision() -> None:
    """A new file at the repo root is shipped or withheld — never neither.

    **And WITHHELD must not be claiming to withhold something the backend
    force-includes**, which is the false dichotomy the old message offered. See
    the comment on :data:`WITHHELD` for the drive: `.hgignore` planted at the
    root, moved into WITHHELD, suite green, file shipped.
    """
    allow = _allowlist()
    undecided = sorted(
        p.name
        for p in REPO.iterdir()
        if p.name not in allow
        and p.name not in WITHHELD
        and p.name not in GENERATED_IN_DISTRIBUTION
    )
    assert not undecided, (
        "these root paths are in none of pyproject's sdist allowlist, WITHHELD "
        "or GENERATED_IN_DISTRIBUTION:\n  "
        + "\n  ".join(undecided)
        + "\n\nAdd each to the allowlist (it ships) or to WITHHELD with a reason "
        "(it does not ship, and this file is the record of that decision — it "
        "is `[tool.hatch.build.targets.sdist].include` that enforces it, and "
        "for a force-included path not even that). An sdist on PyPI cannot be "
        "unpublished."
    )
    # The dichotomy above is only true for paths the include/exclude machinery
    # decides. Force-included ones bypass it, so a record naming one is a FALSE
    # record — the strongest kind of defect this repository has, since the next
    # reader will take it as an assurance.
    #
    # TWO WAYS THIS ASSERTION USED TO THROW ITS OWN ANSWER AWAY.
    #
    # 1. It matched `name in WITHHELD` on the FULL distribution path, while
    #    every other consumer of WITHHELD matches the FIRST COMPONENT
    #    (`path.split('/', 1)[0] in WITHHELD`) — which is what makes the
    #    `scratchpad` entry exempt the whole subtree. So a force-included path
    #    INSIDE a withheld directory was predicted by the model and then
    #    dropped by the comparison. Driven at a4c16fe, one line of
    #    `pyproject.toml`::
    #
    #      license-files = ["LICENSE", "scratchpad/PREREG*.md"]
    #        pytest tests/test_sdist_contents.py    12 passed
    #        uv build --offline --sdist .           271 members (baseline 261)
    #        10 WITHHELD scratchpad/PREREG*.md in the tarball
    #
    #    The model was right and the comparison threw the answer away.
    #
    # 2. It consulted WITHHELD only. GENERATED_IN_DISTRIBUTION is the OTHER
    #    dict a name can hide in, and nothing checked that a name in it is
    #    actually produced by the backend — it was the escape hatch WITHHELD
    #    used to be. Driven at a4c16fe, a root `.hgignore` committed to the
    #    tree::
    #
    #      plain                                        1 failed  (undecided)
    #      + ".hgignore": "generated, honest"            12 passed
    #      uv build --offline --sdist .                  262 members,
    #                                                      .hgignore SHIPPED
    #
    #    A force-included path is taken from a FILE; it is never generated.
    forced = _force_included(REPO)
    lying = sorted(name for name in forced if _falsely_recorded(name))
    assert not lying, (
        "these paths are recorded here as not-distributed or as "
        "backend-generated, and hatchling FORCE-INCLUDES them, so they ship "
        "whatever these dicts say:\n  "
        + "\n  ".join(
            f"{name}  <- {forced[name]}  ({_false_record(name)})" for name in lying
        )
        + "\n\n`force_include` bypasses `include_path()` (hatchling "
        f"{_HATCHLING_READ_AT}, `builders/plugin/interface.py` "
        "`recurse_forced_files`), so neither dict nor the allowlist can keep "
        "one out. Put it in the allowlist and say it ships, or stop the "
        "backend from finding it."
    )


def _falsely_recorded(distribution_path: str) -> str | None:
    """Which record — if any — claims a force-included path is not shipped.

    Matched on the FULL path and on its FIRST COMPONENT, because that is how
    every other consumer of :data:`WITHHELD` matches (``scratchpad`` exempts
    the whole subtree), and on BOTH dicts, because a name in
    :data:`GENERATED_IN_DISTRIBUTION` is exempt from just as much.
    """
    for key in (distribution_path, distribution_path.split("/", 1)[0]):
        if key in WITHHELD:
            return f"recorded WITHHELD as `{key}`"
        if key in GENERATED_IN_DISTRIBUTION:
            return f"recorded GENERATED_IN_DISTRIBUTION as `{key}`"
    return None


# --- what the PyPI project page renders ------------------------------------
#
# The long description is README.md, and PyPI does NOT host this repository's
# files. A relative `<img src="assets/…">` is a broken image on the project page
# and a relative `](docs/…)` is a 404 — while both read perfectly in the repo,
# which is why this survived every reading of the README and needed the built
# artefact to show it.

_REPO_URL_PREFIXES = (
    "https://github.com/NicholasEhsanRoy/stelling/blob/main/",
    "https://github.com/NicholasEhsanRoy/stelling/tree/main/",
    "https://raw.githubusercontent.com/NicholasEhsanRoy/stelling/main/",
)


def _readme_refs(text: str) -> tuple[list[str], list[str]]:
    images = re.findall(r'<img[^>]*src="([^"]+)"', text)
    links = sorted({u for u in re.findall(r"\]\(([^)#][^)]*)\)", text)})
    return images, links


def test_readme_carries_no_relative_reference() -> None:
    """Break it: change one absolute URL back to its repo-relative form."""
    images, links = _readme_refs((REPO / "README.md").read_text())
    assert images, "no images found — this check is not looking at the README"
    assert links, "no links found — this check is not looking at the README"
    bad = [u for u in images + links if not u.startswith(("http://", "https://", "mailto:"))]
    assert not bad, (
        "README.md carries repo-relative references. They read fine on GitHub and "
        "are a broken image / 404 on the PyPI project page, which is where a "
        "stranger lands first:\n  " + "\n  ".join(bad)
    )


def test_every_readme_repo_link_resolves_to_a_real_path() -> None:
    """Absolute URLs stop the PyPI 404s but can still 404 on GitHub. A link into
    this repository must name a path this repository has.

    Break it: point one link at `docs/does-not-exist.md`."""
    images, links = _readme_refs((REPO / "README.md").read_text())
    checked, missing = 0, []
    for url in images + links:
        for prefix in _REPO_URL_PREFIXES:
            if url.startswith(prefix):
                rel = url[len(prefix) :].rstrip("/")
                checked += 1
                if not (REPO / rel).exists():
                    missing.append(url)
                break
    assert checked, "no in-repo links were checked — the prefixes have drifted"
    assert not missing, "README links into this repo that name no such path:\n  " + "\n  ".join(missing)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs `uv` to build")
def test_built_metadata_carries_no_relative_reference(tmp_path: pathlib.Path) -> None:
    """The property on the ARTEFACT, not on the source file — this is what PyPI
    renders. Verified by extraction from the built distribution rather than by
    reading README.md, because reading it is what missed this."""
    proc = subprocess.run(
        ["uv", "build", "--offline", "--out-dir", str(tmp_path), str(REPO)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stderr}"
    wheels = sorted(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        meta = email.message_from_string(z.read(name).decode())
    body = meta.get_payload()
    assert meta.get("Description-Content-Type") == "text/markdown"
    assert len(body) > 1000, "long description is suspiciously short — not the README"
    images, links = _readme_refs(body)
    assert images and links, "the built metadata carries no refs — check is vacuous"
    bad = [u for u in images + links if not u.startswith(("http://", "https://", "mailto:"))]
    assert not bad, "the published long description carries relative refs:\n  " + "\n  ".join(bad)


# --- what hatchling ACTUALLY reads ----------------------------------------
#
# Transcribed from `hatchling/builders/constants.py` and
# `hatchling/builders/config.py` at **1.31.0**, which is what `uv build`
# resolves here; `pyproject.toml` requires `hatchling>=1.27`. A transcription
# rots, so it is not trusted — but WHERE each constant is held, and how much
# of it, is not one sentence.
#
# THIS COMMENT USED TO SAY "a drift in any of the three constants below shows
# up there [in the parity test] as a set difference and not as a silent pass",
# AND THAT WAS FALSE THREE WAYS. Driven at a4c16fe, pristine standalone repo
# from this tree, one mutation at a time:
#
#   `__pycache__` dropped from _HATCH_EXCLUDED_DIRECTORIES     12 passed
#   `.hatch`      dropped from _HATCH_EXCLUDED_DIRECTORIES     12 passed
#   `.DS_Store`   dropped from _HATCH_EXCLUDED_FILES           12 passed
#   _HATCH_FORCED_ROOT_FILES = ("pyproject.toml",)             12 passed
#   `*.py[cdo]`   dropped from _HATCH_DEFAULT_GLOBAL_EXCLUDE   1 failed
#
# and the one that reddened did so in `test_the_exclusion_classifier_
# discriminates`, not in the parity test. The masking is this repository's own
# `.gitignore`: the parity plant for the pruned-directory case sat under
# `docs/__pycache__/`, which is gitignored HERE, so the file was dropped twice
# over and the constant decided nothing.
#
# What holds each one now, said exactly:
#
#   _HATCH_EXCLUDED_DIRECTORIES   the parity build, via a plant under
#                                 `docs/.hatch/` — `.hatch` is one of only
#                                 three names in the set this tree's
#                                 `.gitignore` does not also cover (`.git` and
#                                 `.hg` are the others). `__pycache__`
#                                 specifically is ALSO held by
#                                 `test_the_force_include_model_reads_the_
#                                 static_table`, which asserts a directory
#                                 force-include source is pruned by this set.
#                                 The remaining eight names are held by
#                                 neither, and would have to be un-gitignored
#                                 here to be: recorded, not fixed.
#   _HATCH_EXCLUDED_FILES         the parity build, via a `docs/.DS_Store`
#                                 plant; likewise not gitignored here.
#   _HATCH_DEFAULT_GLOBAL_EXCLUDE `test_the_exclusion_classifier_
#                                 discriminates`, on a synthetic root.
#   _HATCH_FORCED_ROOT_FILES      `test_the_force_include_model_reads_the_
#                                 static_table`, on a synthetic root, and it
#                                 CANNOT be held by any build from this tree:
#                                 two of its three entries, `hatch.toml` and
#                                 `hatch_build.py`, do not exist here.
#   _HATCH_VCS_EXCLUSION_FILES    `test_the_force_include_review_sees_an_
#                                 outside_source`, which plants an ancestor
#                                 `.hgignore` — the parity build cannot hold
#                                 it, because it builds under `tmp_path` and
#                                 a staged copy has no ancestors in common
#                                 with this checkout.
#   _HATCH_DEFAULT_LICENSE_GLOBS  NOT held by anything: `project.license-files`
#                                 is set in `pyproject.toml`, so the defaults
#                                 are never reached from this tree. SUSPECTED,
#                                 not measured, that they are still correct.
_HATCHLING_READ_AT = "1.31.0"

# `EXCLUDED_DIRECTORIES` — pruned by NAME during the walk, at any depth,
# whatever the patterns say.
_HATCH_EXCLUDED_DIRECTORIES = frozenset((
    "__pycache__",
    ".venv",
    ".git",
    ".hg",
    ".hatch",
    ".tox",
    ".nox",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".pixi",
))
# `EXCLUDED_FILES` — dropped by basename. `.git` is here as well as above
# because a linked worktree's `.git` is a FILE.
_HATCH_EXCLUDED_FILES = frozenset((".DS_Store", ".git"))
# `default_global_exclude()` — prepended to the exclusion patterns, in this
# order (it is `sorted()` upstream), BEFORE the `.gitignore` lines. The order
# is load-bearing: `pathspec.GitIgnoreSpec` is last-match-wins, so a `!`
# negation in `.gitignore` can un-exclude a `.pyc`.
_HATCH_DEFAULT_GLOBAL_EXCLUDE = ("*.py[cdo]", "/dist")

# `SdistBuilder.get_default_build_data()` — the paths that reach the tarball
# WITHOUT passing the include allowlist or the exclusion spec at all, because
# `recurse_forced_files` yields them directly.
_HATCH_FORCED_ROOT_FILES = ("pyproject.toml", "hatch.toml", "hatch_build.py")
# `BuilderConfig.vcs_exclusion_files` — (file, boundary directory). BOTH are
# force-included, keyed by `os.path.basename`, and both are found by walking UP.
_HATCH_VCS_EXCLUSION_FILES = ((".gitignore", ".git"), (".hgignore", ".hg"))
# `MetadataCore.license_files` when `project.license-files` is absent.
_HATCH_DEFAULT_LICENSE_GLOBS = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")


def _locate_file(root: pathlib.Path, name: str, boundary: str) -> pathlib.Path | None:
    """``hatchling.utils.fs.locate_file``, transcribed.

    THE ORDER OF THE TWO CHECKS IS THE WHOLE POINT. The file is looked for
    first and the boundary second, and only the NAMED boundary stops the walk —
    so ``_locate_file(root, ".hgignore", ".hg")`` in a git checkout meets no
    boundary anywhere and walks to ``/``. A `.hgignore` in any ancestor of the
    build root is therefore read as exclusions and shipped as ``/.hgignore``.
    Measured: 261 members against a baseline of 260, the extra one a file that
    lives outside the repository.
    """
    here = root.resolve()
    while True:
        candidate = here / name
        if candidate.is_file():
            return candidate
        if (here / boundary).exists():
            return None
        parent = here.parent
        if parent == here:
            return None
        here = parent


def _force_included(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """``{distribution-relative path: the file on disk it is taken from}``.

    A model of ``SdistBuilder.get_default_build_data()``. It is a model, so it
    is not trusted where it can be checked: ``test_the_untracked_scan_agrees_
    with_the_tarball`` asserts that every tarball member was admitted by
    something, which catches a force-included file whether or not this function
    predicted it.

    **THE TWO ARE NOT INDEPENDENT, AND THIS DOCSTRING USED TO SAY THEY WERE**
    — "blind either one and the other still reddens". Measured under the
    DEFAULT ``tmp_path``, in a standalone repo from this tree with an ancestor
    ``.hgignore`` planted and shipping::

        _force_included -> {}                                12 passed
        _forced_without_a_reviewed_source -> []               12 passed
        `.hgignore` dropped from _HATCH_VCS_EXCLUSION_FILES   12 passed

    against ``1 failed`` unblinded. The reason is structural, not incidental:
    the parity test builds from a copy under ``tmp_path``, so the staged tree
    never shares this checkout's ANCESTRY. Parity independently covers
    force-include routes that are a property of the tree's CONTENT; it cannot
    cover one that is a property of the tree's LOCATION, and every route
    :func:`_locate_file` opens is the second kind. (An earlier blinding matrix
    got "independent" by putting ``--basetemp`` inside the plant directory,
    which makes the staged copy inherit the planted ancestor — an artefact of
    the harness, not a property of the guard.)

    So this function is load-bearing on its own, and
    ``test_no_force_included_path_escapes_review`` asserts it is non-empty
    rather than trusting a green to mean it looked.

    NOT modelled, and unmodellable from here: a `hatch_build.py` build hook may
    add arbitrary entries to ``build_data["force_include"]`` at build time.
    **This docstring used to end "if one appears, the parity test is what will
    notice", and that was false.** A committed, allowlisted, registered hook
    that force-includes a TRACKED file leaves every counterpart the parity test
    can check satisfied — driven at a4c16fe: ``12 passed``, 263 members,
    ``scratchpad/PREREG_SDIST.md`` in the tarball. What notices one now is
    ``test_no_build_hook_can_force_include_behind_this_module``, which refuses
    the capability rather than modelling it.
    """
    forced: dict[str, pathlib.Path] = {}
    for name in _HATCH_FORCED_ROOT_FILES:
        # upstream tests `os.path.exists`, not `is_file`
        if (root / name).exists():
            forced[name] = root / name
    for name, boundary in _HATCH_VCS_EXCLUSION_FILES:
        located = _locate_file(root, name, boundary)
        if located is not None:
            # keyed by BASENAME upstream, which is what puts an ancestor's file
            # at the root of the distribution under this project's name
            forced[located.name] = located

    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = cfg.get("project", {})
    readme = project.get("readme")
    if isinstance(readme, str):
        forced[readme] = root / readme
    elif isinstance(readme, dict) and "file" in readme:
        forced[readme["file"]] = root / readme["file"]

    patterns = project.get("license-files", list(_HATCH_DEFAULT_LICENSE_GLOBS))
    if isinstance(patterns, dict):  # the pre-PEP-639 spelling hatchling still reads
        patterns = patterns.get("globs", patterns.get("paths", []))
    for pattern in patterns:
        for match in sorted(globlib.glob(os.path.normpath(os.path.join(root, pattern)))):
            if os.path.isfile(match):
                forced[os.path.relpath(match, root).replace(os.sep, "/")] = pathlib.Path(match)

    forced.update(_static_force_include(root, cfg))
    return forced


def _static_force_include(
    root: pathlib.Path, cfg: dict
) -> dict[str, pathlib.Path]:
    """``BuilderConfig.force_include``, the table written in `pyproject.toml`.

    THIS USED TO BE UNMODELLED AND UNNAMED, and it is the easiest of all the
    force-include routes to introduce: a two-line static table, no build hook,
    no ancestor directory. ``get_force_include()`` (`builders/config.py:823`)
    merges it with whatever ``get_default_build_data()`` produced, and
    ``recurse_forced_files`` yields the result past `include_path()`. Driven at
    a4c16fe, in a standalone repo from this tree::

        [tool.hatch.build]
        force-include = {"scratchpad/PREREG_SDIST.md" = "scratchpad/PREREG_SDIST.md"}

        pytest -q -ra tests/test_sdist_contents.py   12 passed
        uv build --offline --sdist .                 262 members (baseline 261)
        tar tzf …  ->  stelling-0.1.0/scratchpad/PREREG_SDIST.md

    — a path this file's own :data:`WITHHELD` says is not distributed.

    Transcribed rather than summarised (`builders/config.py:678-704`,
    `builders/utils.py` ``normalize_inclusion_map`` /
    ``normalize_relative_path``, `builders/plugin/interface.py`
    ``recurse_forced_files``):

    * the target table SHADOWS the global one — it is ``if "force-include" in
      target_config … else build_config``, not a merge;
    * the KEY is the source (``expanduser`` + ``normpath``, made absolute
      against the root, so it may point outside the tree) and the VALUE is the
      distribution-relative path;
    * a source that is a DIRECTORY force-includes everything under it, pruned
      by the same :data:`_HATCH_EXCLUDED_DIRECTORIES` and
      :data:`_HATCH_EXCLUDED_FILES` as the ordinary walk.

    NOT modelled: ``get_distribution_path``'s ``sources`` rewriting (the sdist
    target sets none here), and a missing source — hatchling raises
    ``FileNotFoundError`` for that, so the build fails loudly and nothing
    ships.
    """
    build_cfg = cfg.get("tool", {}).get("hatch", {}).get("build", {})
    sdist_cfg = build_cfg.get("targets", {}).get("sdist", {})
    table = (
        sdist_cfg["force-include"]
        if "force-include" in sdist_cfg
        else build_cfg.get("force-include", {})
    )
    if not isinstance(table, dict):
        return {}
    out: dict[str, pathlib.Path] = {}
    for raw_source, relative_path in table.items():
        if not isinstance(relative_path, str):
            continue
        source = pathlib.Path(os.path.expanduser(os.path.normpath(str(raw_source))))
        if not source.is_absolute():
            source = pathlib.Path(os.path.abspath(os.path.join(root, source)))
        target = os.path.normpath(relative_path).strip(os.sep).replace(os.sep, "/")
        if source.is_file():
            out[target] = source
        elif source.is_dir():
            for dirpath, dirnames, filenames in os.walk(source):
                dirnames[:] = [
                    d for d in dirnames if d not in _HATCH_EXCLUDED_DIRECTORIES
                ]
                relative = os.path.relpath(dirpath, source)
                for name in sorted(filenames):
                    if name in _HATCH_EXCLUDED_FILES:
                        continue
                    dist = os.path.normpath(os.path.join(target, relative, name))
                    out[dist.replace(os.sep, "/")] = pathlib.Path(dirpath) / name
    return out


def _forced_without_a_reviewed_source(
    root: pathlib.Path, tracked: set[str] | None
) -> list[str]:
    """Force-included paths whose content is not a committed file of this tree.

    Two shapes, and the first is the one that has no defence anywhere else in
    this module: the source lies OUTSIDE ``root``, so there is nothing in the
    tree to review and nothing the allowlist can exclude. The second is a
    source inside the tree that nobody has committed.

    ``tracked=None`` means THE INDEX COULD NOT BE READ HERE — no `git`, or no
    `.git` at the build root. Only the first shape can be judged then, and it
    still is, because that is precisely the configuration the first shape
    appears in: an unpacked sdist or a `.git`-stripped container copy has no
    `.git` boundary, so ``_locate_file(root, ".hgignore", ".hg")`` and
    ``_locate_file(root, ".gitignore", ".git")`` both walk out of the tree.
    "The index is unreadable" must not read as "nothing is force-included".
    """
    out = []
    resolved_root = root.resolve()
    for dist_path, source in sorted(_force_included(root).items()):
        try:
            inside = source.resolve().relative_to(resolved_root)
        except ValueError:
            out.append(f"{dist_path}  <- {source}  (OUTSIDE the tree)")
            continue
        if tracked is not None and inside.as_posix() not in tracked:
            out.append(f"{dist_path}  <- {source}  (inside the tree, untracked)")
    return out


def _index_is_readable(root: pathlib.Path) -> bool:
    """Whether ``git ls-files`` can answer for `root` at all.

    A worktree's `.git` is a FILE, hence ``exists()``. This is the same
    predicate ``test_no_untracked_file_anywhere_would_ship`` skips on, asked
    here so that the force-include check can DEGRADE instead of skipping.
    """
    return shutil.which("git") is not None and (root / ".git").exists()


def _git(args: list[str], cwd: pathlib.Path, *, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """`git`, with a non-zero treated as a hard failure and never as silence.

    This is the shape the pre-commit hooks got wrong: a command that could not
    run reports the same "found nothing" as a command that ran and found
    nothing. Every call site here says which exit codes MEAN something.
    """
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode in ok, (
        f"`git {' '.join(args)}` exited {proc.returncode} in {cwd}, so this "
        "check could not run — and a check that could not run must not report "
        "the same thing as a check that ran and was satisfied. git said:\n"
        + (proc.stderr or "(nothing)")
    )
    return proc


def _tracked_files(root: pathlib.Path) -> set[str]:
    """The index, which has nothing to do with any exclusion rule.

    `-z` because a path with a quote, a newline or a non-ASCII byte comes back
    C-quoted otherwise, and a quoted path silently fails to match the walk.
    """
    return {p for p in _git(["ls-files", "-z"], root).stdout.split("\0") if p}


def _walked_files(root: pathlib.Path) -> set[str]:
    """Every file hatchling's walk reaches, as a POSIX path relative to `root`.

    THE ENUMERATION CONSULTS NO EXCLUSION SOURCE AT ALL. That is the whole
    repair: the previous enumeration was `git status --untracked-files=all`,
    which honours `.gitignore` at every level, `.git/info/exclude`,
    `core.excludesFile` and the command line, while hatchling honours one of
    those four. Anything in the difference vanished from the guard and appeared
    in the tarball.

    `followlinks=True` with a device/inode seen-set is `hatchling.builders.
    utils.safe_walk` verbatim — a symlinked directory IS descended into by the
    build, so a walk that skipped it would under-report exactly where the build
    over-collects.
    """
    found: set[str] = set()
    seen: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        stat = os.stat(dirpath)
        identifier = (stat.st_dev, stat.st_ino)
        if identifier in seen:
            dirnames[:] = []
            continue
        seen.add(identifier)
        dirnames[:] = [d for d in dirnames if d not in _HATCH_EXCLUDED_DIRECTORIES]
        relative = os.path.relpath(dirpath, root)
        prefix = "" if relative == "." else relative.replace(os.sep, "/") + "/"
        for name in filenames:
            if name in _HATCH_EXCLUDED_FILES:
                continue
            found.add(prefix + name)
    return found


def _assert_allowlist_is_plain(allow: set[str]) -> None:
    """The allowlist is matched here by first path component, and that is only
    faithful to `pathspec.GitIgnoreSpec` while every entry is a plain rooted
    name. A glob or a nested path in `pyproject.toml` breaks the equivalence,
    so it fails HERE rather than quietly widening what this guard accepts."""
    fancy = sorted(e for e in allow if (set(e) & set("*?[]!")) or "/" in e)
    assert not fancy, (
        "the sdist allowlist has grown an entry this check cannot match by "
        "first path component:\n  " + "\n  ".join(fancy)
        + "\n\nIt is compared as `path.split('/', 1)[0] in allow`, which is "
        "equivalent to hatchling's `GitIgnoreSpec` only for plain rooted "
        "names. Teach this check the new shape before adding it."
    )


def _allowlist_admits(path: str, allow: set[str]) -> bool:
    """`/src` in hatchling's include spec admits `src` and everything under it."""
    return path in allow or path.split("/", 1)[0] in allow


def _check_ignore(
    lines: list[str], paths: set[str], oracle: pathlib.Path
) -> set[str]:
    """Which of `paths` a `.gitignore` made of `lines` matches — asked of git,
    in a repository that has nothing in it but those lines.

    `git check-ignore` run in the REAL tree is the wrong instrument: it answers
    with the winning pattern across all four of git's sources, and three of
    them are precisely what hatchling cannot see. So the patterns are composed
    by the caller, written into a throwaway repository, and matched there. That
    repository has no `.git/info/exclude` entries, no nested `.gitignore`, and
    `core.excludesFile` is pointed at `/dev/null`, so the only thing that can
    decide is the thing hatchling reads.

    The paths are NOT materialised, deliberately. `git check-ignore` stats:
    driven, with the pattern `build/` and the query `docs/build`, it answers
    ignored when `docs/build` is a directory and NOT ignored when it is a file
    or absent — and `pathspec` treats a file named `build` as not excluded,
    which is the divergence written up at `_NOT_COPIED_DIRS` below. Absent is
    the file answer, and everything asked here is a file.

    Exit codes are read the way the pre-commit hooks failed to: 0 is "some
    matched", 1 is "none matched", and anything else is an ERROR that must not
    be mistaken for either.
    """
    if not paths:
        return set()
    oracle.mkdir(parents=True, exist_ok=True)
    (oracle / ".gitignore").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(["init", "-q", "."], oracle)
    proc = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "-z", "--stdin"],
        cwd=oracle,
        input="\0".join(sorted(paths)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode in (0, 1), (
        f"`git check-ignore` exited {proc.returncode}, which is neither "
        "'some matched' (0) nor 'none matched' (1). An error here must not be "
        "read as 'hatchling excludes nothing', and must not be read as "
        "'hatchling excludes everything' either. git said:\n"
        + (proc.stderr or "(nothing)")
    )
    return {p for p in proc.stdout.split("\0") if p}


def _vcs_exclusion_lines(root: pathlib.Path) -> list[str]:
    """``BuilderConfig.load_vcs_exclusion_patterns()`` up to its bail-out.

    The `.gitignore` goes in verbatim; the `.hgignore` contributes only the
    lines inside a ``syntax: glob`` section, which is hatchling's own reading
    of an hgignore and not a correct one in general — it is transcribed
    because it is what the build does.
    """
    lines: list[str] = []
    for name, boundary in _HATCH_VCS_EXCLUSION_FILES:
        located = _locate_file(root, name, boundary)
        if located is None:
            continue
        text = located.read_text(encoding="utf-8").splitlines()
        if name == ".gitignore":
            lines.extend(text)
            continue
        glob_mode = False
        for line in text:
            exact = line.strip()
            if exact == "syntax: glob":
                glob_mode = True
            elif exact.startswith("syntax: "):
                glob_mode = False
            elif glob_mode:
                lines.append(line)
    return lines


def _vcs_exclusions_are_discarded(root: pathlib.Path, oracle: pathlib.Path) -> bool:
    """Whether hatchling would throw the whole exclusion set away — because the
    BUILD ROOT'S OWN ABSOLUTE PATH matches it. ``builders/config.py:790-793``::

        # validate project root is not excluded by vcs
        exclude_spec = pathspec.GitIgnoreSpec.from_lines(patterns)
        if exclude_spec.match_file(self.root):
            return []

    ``self.root`` is absolute, so a checkout at ``~/.cache/x/stelling`` matches
    this project's own ``.cache`` line and every exclusion is dropped. Measured,
    one tree, two checkout paths, the same two plants::

        <…>/.cache/repo   262 members, docs/zz_secret.log AND
                                       docs/htmlcov/index.html SHIPPED
        <…>/ok/repo       260 members, neither present

    `pathspec` reduces the absolute path to its components, so the query is the
    path with its leading separator removed. THAT CORRESPONDENCE IS MEASURED,
    not assumed: `git check-ignore` in the oracle and
    `pathspec.GitIgnoreSpec.match_file` were compared on 13 candidate checkout
    paths against this repository's own `.gitignore` (`~/.cache/x/stelling`,
    `~/lib/…`, `/var/lib/ci/…`, `/srv/dist/…`, `~/.env/…`, `~/build/…`,
    `~/venv/…`, `/opt/target/…`, `~/site/…`, `/home/runner/work/stelling/
    stelling`, and three real paths in this checkout) and agreed on all 13.
    """
    lines = _vcs_exclusion_lines(root)
    if not lines:
        return False
    query = pathlib.PurePosixPath(root.resolve()).as_posix().lstrip("/")
    return bool(_check_ignore(lines, {query}, oracle))


def _hatchling_excluded(
    root: pathlib.Path, paths: set[str], oracle: pathlib.Path
) -> set[str]:
    """Which of `paths` hatchling's exclusion spec matches.

    `default_global_exclude()` first — it is NOT part of the VCS set and
    survives the bail-out — then whatever
    :func:`_vcs_exclusion_lines` yields, unless the build root's own path
    matches the VCS set, in which case hatchling drops all of it and this
    drops all of it too.
    """
    if not paths:
        return set()
    lines = [*_HATCH_DEFAULT_GLOBAL_EXCLUDE]
    if not _vcs_exclusions_are_discarded(root, oracle.with_name(oracle.name + "-root")):
        lines.extend(_vcs_exclusion_lines(root))
    return _check_ignore(lines, paths, oracle)


def _pruned_by_the_walk(root: pathlib.Path, path: str) -> bool:
    """Whether hatchling's own walk would never reach `path` — which is not the
    same thing as the walk being blind, and the difference is a false RED.

    The walk drops a DIRECTORY by name at any depth
    (:data:`_HATCH_EXCLUDED_DIRECTORIES`) and a FILE by basename
    (:data:`_HATCH_EXCLUDED_FILES`). A tracked file at ``docs/.hatch/note.md``
    is therefore in the index, absent from the walk, and absent from the
    tarball — nothing ships, and reporting "the walk is not looking at the tree
    git is looking at" is the wrong diagnosis for it. Measured: two tests red,
    260 members, `.hatch` nowhere in the tarball.

    A tracked path whose LAST component carries an excluded name is only pruned
    when it is a directory on disk — the walk filters `dirnames`, so a FILE
    called `docs/.hatch` is reached and does ship.
    """
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part not in _HATCH_EXCLUDED_DIRECTORIES:
            continue
        if index < len(parts) - 1 or (root / path).is_dir():
            return True
    return parts[-1] in _HATCH_EXCLUDED_FILES


def _untracked_that_would_ship(
    root: pathlib.Path, allow: set[str], oracle: pathlib.Path
) -> list[str]:
    """The property, computed by hatchling's rule: untracked AND admitted by
    the allowlist AND not excluded by the VCS files hatchling reads.

    Two non-vacuity assertions live here, because the caller cannot make them:
    the index must be non-empty, and **the walk must have reached every tracked
    file hatchling's own walk would have reached**. Zero items examined is the
    failure mode this whole pass is about, and the second assertion is what
    makes it impossible — blind the walk and every tracked file goes missing at
    once. Neither assertion was itself pinned: deleting the second, or
    neutering the first to ``assert True or tracked``, left the suite at
    ``8 passed``. ``test_the_scan_refuses_a_blinded_walk`` and
    ``test_the_scan_refuses_an_empty_index`` are what pin them now.

    THE SUBTRACTION OF `pruned` IS LOAD-BEARING IN BOTH DIRECTIONS. It removes
    the false red on a legitimately tracked ``docs/.hatch/note.md``; it does
    NOT remove the true red on a git SUBMODULE, whose gitlink is tracked, is
    not pruned, and whose contents genuinely reach the tarball. Measured, one
    submodule at ``docs/sub``: this assertion red, 261 members,
    ``stelling-0.1.0/docs/sub/payload.md`` present.
    """
    tracked = _tracked_files(root)
    assert tracked, (
        "`git ls-files` reported no tracked files, so there is no tree here to "
        "check and this must not read as 'no untracked file would ship'"
    )
    walked = _walked_files(root)
    missed = [p for p in tracked - walked if not _pruned_by_the_walk(root, p)]
    # A SYMLINK TO A DIRECTORY IS NOT A WALK FAILURE, and it used to be
    # reported as one. `os.walk` puts it in `dirnames`, never in `filenames`,
    # so it is never a walked FILE; `git ls-files` names it, because git tracks
    # the link itself as a blob. The walk is fine. Measured, a tracked
    # `docs/zz_linkdir -> ../design` committed to a standalone repo from this
    # tree: `assert not unseen` red with "the walk is not looking at the tree
    # git is looking at", and the tarball carries NO member at or under
    # `docs/zz_linkdir` at all — `safe_walk`'s (st_dev, st_ino) seen-set has
    # already visited `design/`, so it descends into nothing. Kept red, because
    # the contents of a link whose target has NOT already been walked do reach
    # the tarball under the link's path; separated, because sending the next
    # reader to look for a broken walk is sending them to the wrong mechanism.
    dir_links = sorted(p for p in missed if (root / p).is_symlink() and (root / p).is_dir())
    unseen = sorted(p for p in missed if p not in set(dir_links))
    assert not unseen, (
        f"the filesystem walk reached {len(walked)} files and missed "
        f"{len(unseen)} that git says are tracked and that hatchling's own walk "
        "would NOT have pruned, so either the walk is not looking at the tree "
        "git is looking at — and its silence about untracked files means "
        "nothing — or one of these is a git submodule, whose contents DO reach "
        "the sdist while `git ls-files` names only the gitlink. First few:\n  "
        + "\n  ".join(unseen[:10])
    )
    assert not dir_links, (
        "these tracked paths are SYMLINKS TO DIRECTORIES:\n  "
        + "\n  ".join(f"{p} -> {os.readlink(root / p)}" for p in dir_links[:10])
        + "\n\nThis is not a walk failure and not a submodule. `os.walk` yields "
        "a directory symlink in `dirnames`, never in `filenames`, so it is "
        "never a walked FILE while `git ls-files` names it — the walk is "
        "looking at exactly the tree git is. Nothing ships AT this path. What "
        "does ship is everything hatchling reaches THROUGH it, under the "
        "link's path, and only when `safe_walk`'s (st_dev, st_ino) seen-set "
        "has not already visited the target — so a link out of the tree "
        "distributes files nobody committed here. Point the allowlist at the "
        "real directory instead of at a link into it."
    )
    admitted = {p for p in walked - tracked if _allowlist_admits(p, allow)}
    return sorted(admitted - _hatchling_excluded(root, admitted, oracle))


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_untracked_file_anywhere_would_ship(tmp_path: pathlib.Path) -> None:
    """The allowlist has ROOT-LEVEL granularity, and that is not the whole job.

    Measured: an allowlist entry like ``/docs`` admits the directory, and
    hatchling then takes everything inside it that is not gitignored. A stray
    untracked file at ``docs/scratch.md`` or ``src/stelling/tmp.py`` **still
    ships** — the root allowlist does not reach it. The checklist that started
    all this merely happened to sit at the root.

    So the allowlist closes the root and this closes the rest: any untracked,
    non-gitignored path anywhere in the tree WILL be distributed, and must be a
    decision. Committed files are out of scope — they are reviewed.

    Break it: `touch docs/anything.md` and run this test.

    **IT USED TO ASK A CHANNEL THAT DISAGREES WITH THE BUILDER.** The
    enumeration was ``git status --porcelain --untracked-files=all``, which
    honours four exclusion sources; hatchling honours one of them. Driven end
    to end in a standalone repository built from this tree — plant, check,
    build, list::

        docs/zz_exclude_probe.md  +  a line in .git/info/exclude
            git status --porcelain -uall | grep -c zz_exclude_probe   0
            this test                                                1 passed
            uv build --offline --sdist                                261 members
            tar tzf | grep zz_exclude_probe   stelling-0.1.0/docs/zz_exclude_probe.md

        docs/notes.md  +  core.excludesFile naming `notes.md`
            (a developer's global gitignore — the likelier one in practice)
            git status: 0    this test: 1 passed    sdist: 261, SHIPPED

        docs/zz_nested_probe.md  +  a TRACKED docs/.gitignore naming it
            (hatchling reads the ROOT .gitignore only; this repository already
            carries a nested one at scratchpad/reach/.gitignore)
            git status: clean    this test: 1 passed    sdist: 262, SHIPPED

    The baseline is 260 members. Root-level plants were never the hole —
    ``test_every_root_entry_is_a_decision`` reads the filesystem and catches
    those. It is subdirectory + a git-only exclusion, which is exactly where
    the leak that motivated the allowlist would have hidden one directory
    deeper.

    **And it had no non-vacuity guard.** Driven, ``--untracked-files=all`` →
    ``=no`` with a real untracked non-ignored file in ``docs/``: ``1 passed``,
    and the file in the tarball. Zero items examined read as zero problems.

    Both are closed by asking a different question. The enumeration is now the
    filesystem minus the index — no exclusion source can hide anything from it
    — the classification is hatchling's own rule
    (:func:`_hatchling_excluded`), and the walk must reach every tracked file
    before its silence counts for anything.

    **The skip condition is the one the message names, and it did not used to
    be.** The gate was ``if proc.returncode != 0: skip("not a git checkout")``
    — so this test reported a green skip on *any* non-zero from ``git status``:
    a broken ``GIT_DIR``, an unreadable index, a held lock, a half-finished
    rebase. Driven::

        GIT_DIR=/nonexistent/x pytest -q -ra <this nodeid>
        1 skipped                        <- and the tree IS a git checkout

    That is this branch's own subject: a control reporting pass while it did
    not run. It was not an exposure — ``tests/test_skip_inventory.py`` pins the
    disclosed condition (``not (REPO / ".git").exists()``), which is False in a
    real checkout, so the session's verdict went to ``failed`` and ``pytest``
    exited 1 on that same drive at 33fd1f8 as well as here. But a second
    control catching the first control's misstatement is not the same thing as
    the first control being true, so the condition now IS ``.git`` absence and
    every other failure is a hard red carrying git's own stderr.
    """
    if not (REPO / ".git").exists():
        # A worktree's `.git` is a FILE, hence `exists()` and not `is_dir()`.
        # This is byte-for-byte the predicate `tests/test_skip_inventory.py`
        # declares legitimate for this reason string; the two agreeing by
        # construction is the point.
        pytest.skip("not a git checkout (an unpacked sdist, say)")
    allow = _allowlist()
    assert allow, (
        "pyproject's sdist allowlist is empty, so nothing would be admitted "
        "and this check would have nothing to say about anything"
    )
    _assert_allowlist_is_plain(allow)
    would_ship = _untracked_that_would_ship(REPO, allow, tmp_path / "oracle")
    # WITHHELD is consulted AFTER the shipping question is answered, so it can
    # no longer decide it. Every entry names a root path that the allowlist
    # does not admit anyway, so nothing should ever reach `masked` — and if a
    # future WITHHELD entry starts covering a path hatchling WOULD ship, that
    # is a red here rather than a silent exemption.
    masked = [p for p in would_ship if p.split("/", 1)[0] in WITHHELD or p in WITHHELD]
    assert not masked, (
        "these paths would be DISTRIBUTED and are being suppressed by a "
        "WITHHELD entry, which is a build exclusion WITHHELD cannot make:\n  "
        + "\n  ".join(masked)
        + "\n\nWITHHELD records a decision; `[tool.hatch.build.targets.sdist]"
        ".include` is what enforces it. Take the path out of the allowlist."
    )
    undecided = [p for p in would_ship if p not in masked]
    assert not undecided, (
        "these paths are untracked and WOULD BE DISTRIBUTED in the sdist:\n  "
        + "\n  ".join(undecided)
        + "\n\nThe root allowlist does not reach inside an allowlisted directory, "
        "and hatchling reads only the `.gitignore` and `.hgignore` it finds by "
        "walking UP from the build root. `.git/info/exclude`, your global "
        "`core.excludesFile` and any nested `.gitignore` hide a file from "
        "`git status` and not from the build.\n"
        "Commit the file, add it to the ROOT `.gitignore`, or take its directory "
        "out of the sdist allowlist."
    )
    # THE SECOND ROUTE INTO THE TARBALL is not checked here any more, and the
    # move is the point: see `test_no_force_included_path_escapes_review`. It
    # used to live under this function's `.git` skip, which is the one
    # configuration where that route opens.


def test_no_force_included_path_escapes_review() -> None:
    """The second route into the tarball, and IT RUNS WHERE THE ROUTE OPENS.

    A force-included path never meets ``include_path()``, so neither the
    allowlist nor the exclusion spec is consulted for it. One of the two VCS
    files, ``.hgignore``, is located with a boundary (``.hg``) that a git
    checkout does not have, so it is found by walking UP to ``/``; and with no
    ``.git`` at the build root the SAME thing happens to ``.gitignore``.

    **This check used to sit inside**
    ``test_no_untracked_file_anywhere_would_ship``, **which skips on**
    ``not (REPO / ".git").exists()`` — the exact configuration this module's
    own docstring names as the trigger ("an unpacked sdist, a `.git`-stripped
    container copy"). Driven, in standalone repos built by `git archive` from
    this tree, one ancestor level up::

        cp -a ok/repo s26/repo && rm -rf s26/repo/.git
        printf 'syntax: glob\\nzz_never\\n' > s26/.hgignore

        BEFORE, at a4c16fe   pytest tests/test_sdist_contents.py
                               11 passed, 1 skipped
                             uv build --offline --sdist .
                               262 members (baseline 261)
                               stelling-0.1.0/.hgignore   SHIPPED
        AFTER                  11 passed, 1 skipped, 1 FAILED (this test)

    Controls, same command, same tree: without the planted `.hgignore` the
    build is 261 members and this test is green; with the plant AND a `.git`
    present it was already red before the move, and still is.

    It needs no git. ``_tracked_files`` does, and only for the SECOND of the
    two shapes :func:`_forced_without_a_reviewed_source` reports; with no
    readable index the check degrades to "every force-included source outside
    the tree is a red" rather than to silence.

    **And the non-vacuity assertion is not decoration.** Blinding
    :func:`_force_included` to ``{}`` was measured to leave the whole module at
    ``12 passed`` with an ancestor `.hgignore` shipping — the parity test does
    not cover it, because parity builds under `tmp_path` and a staged copy
    never inherits the real tree's ANCESTORS. Parity catches force-include
    routes that are a property of the tree's CONTENT; it cannot catch one that
    is a property of the tree's LOCATION. So zero force-included paths is a red
    here, not a green.
    """
    forced = _force_included(REPO)
    assert forced, (
        "this tree has NO force-included paths at all, which cannot be true — "
        "`pyproject.toml` is one unconditionally "
        f"(hatchling {_HATCHLING_READ_AT}, `builders/sdist.py` "
        "`get_default_build_data`). The model has stopped modelling, and its "
        "silence below is silence about nothing."
    )
    readable = _index_is_readable(REPO)
    problems = _forced_without_a_reviewed_source(
        REPO, _tracked_files(REPO) if readable else None
    )
    assert not problems, (
        "these paths would be FORCE-INCLUDED in the sdist and are not committed "
        "files of this tree:\n  "
        + "\n  ".join(problems)
        + "\n\n`force_include` bypasses the allowlist and the exclusion spec "
        f"entirely (hatchling {_HATCHLING_READ_AT}, "
        "`builders/sdist.py` `get_default_build_data`), so nothing in "
        "`pyproject.toml` and nothing in this file can keep one out. If the "
        "source is OUTSIDE the tree it is somebody else's file being published "
        "under this project's name: remove it, or build from a root that does "
        "not have it above."
        + (
            ""
            if readable
            else "\n\n(No readable index here, so only the outside-the-tree "
            "shape was judged.)"
        )
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_checkout_path_does_not_disable_the_exclusions(
    tmp_path: pathlib.Path,
) -> None:
    """WHERE THIS REPOSITORY IS CHECKED OUT CHANGES WHAT ITS SDIST CONTAINS.

    ``builders/config.py:790-793`` matches the exclusion spec against the build
    root's own ABSOLUTE path and, on a hit, returns ``[]`` — no `.gitignore` at
    all. This repository's `.gitignore` carries `.cache`, `lib/`, `var/`,
    `dist/`, `build/`, `venv/`, `.env` and `target/` among others, every one of
    which is a plausible component of a checkout path.

    Positive control, driven — one tree, two checkout paths, the same two
    plants (`docs/zz_secret.log`, matched by `*.log`; `docs/htmlcov/index.html`,
    matched by `htmlcov/`)::

        <…>/.cache/repo   262 members, BOTH shipped   suite: 8 passed
        <…>/ok/repo       260 members, neither        suite: 8 passed

    Green in both, because the parity test always builds under `tmp_path` and
    can never be at an excluded path — so this is the only place the question
    can be asked. Realism, measured against this repository's own `.gitignore`:
    `~/.cache/x/stelling`, `~/lib/…`, `/var/lib/ci/…`, `/srv/dist/…`,
    `~/.env/…`, `~/build/…`, `~/venv/…` and `/opt/target/…` all trigger it;
    `/home/runner/work/stelling/stelling` does not, so GitHub-hosted CI is safe
    and a developer box may not be. This is not a test of hatchling — it is a
    test that THIS checkout is somewhere the guard still works, and the fix
    when it reddens is to move the checkout.
    """
    lines = _vcs_exclusion_lines(REPO)
    assert lines, (
        "hatchling would find no VCS exclusion lines for this tree at all, so "
        "there is nothing here to be discarded and nothing below means "
        "anything. A `.gitignore` should have been found by walking up from "
        f"{REPO} to the `.git` boundary."
    )
    # The instrument must be able to say "matched" before its "not matched" is
    # worth anything: a synthetic pattern, a synthetic path, in its own oracle.
    live = _check_ignore(
        ["zz_instrument_control"],
        {"a/zz_instrument_control/b", "a/b/c"},
        tmp_path / "oracle-live",
    )
    assert live == {"a/zz_instrument_control/b"}, (
        "`git check-ignore` in an isolated oracle did not answer as a "
        f"`.gitignore` matcher: got {sorted(live)}. The measurement below "
        "cannot be read as 'the checkout path is safe'."
    )
    discarded = _vcs_exclusions_are_discarded(REPO, tmp_path / "oracle-root")
    assert not discarded, (
        f"this checkout is at\n  {REPO}\nand its own path matches the exclusion "
        "patterns hatchling loads from this tree's `.gitignore`. Hatchling "
        "answers that by discarding the ENTIRE exclusion set "
        f"(hatchling {_HATCHLING_READ_AT}, `builders/config.py`, "
        "`load_vcs_exclusion_patterns`), so an sdist built here ships every "
        "gitignored file inside an allowlisted directory — coverage HTML, "
        "`.log` files, `.env` — with nothing else red. Move the checkout out "
        "of the matching directory and rebuild."
    )


# DIRECTORIES the copy leaves behind, and the "directories" is the whole point.
#
# These are caches and build output that make the copy large for nothing, and
# every one of them is gitignored *as a directory* at any depth, so dropping
# them changes nothing hatchling would have read. Driven with `git check-ignore
# --no-index`, which is the authority here because hatchling's rule IS
# `.gitignore`: `docs/build/`, `a/b/c/build/` and the other seven are all
# IGNORED.
#
# THE COMMENT HERE USED TO SAY "names … hatchling never reads", AND THAT WAS
# FALSE FOR FILES. ``shutil.ignore_patterns`` matches a NAME whatever it is,
# while `.gitignore`'s `build/` is directory-only — so a FILE named `build`
# is not gitignored and hatchling ships it. Counter-construction, one plant,
# two copies of this tree, both built:
#
#   docs/build (a FILE)   copy made by ignore_patterns   260 members, absent
#                         the tree hatchling reads       261 members, SHIPPED
#
# The test would then have been measuring a tree the build does not see. Seven
# of the eight are not gitignored as files (`.venv` is, its pattern carries no
# trailing slash); no such filename exists in the tree today, so this was
# latent, and it is closed by asking `is_dir()` rather than by matching a name.
#
# RE-DRIVE IT by putting `shutil.ignore_patterns(*_NOT_COPIED_DIRS)` back in
# `_tree_to_build`, planting `docs/build` as a FILE, and building an sdist from
# the copy and from the tree: 260 against 261. With `_copy_ignore` it is 261
# against 261, and `docs/build` as a DIRECTORY is still 260 against 260, so the
# exclusion still does the job it was added for. No test in the suite reddens
# either way — the divergence is latent, which is exactly why it is written
# down rather than left to the next reader to rediscover.
_NOT_COPIED_DIRS = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "dist",
    "build",
)

# The one exclusion that is NOT about gitignore, and so is by name: a worktree's
# `.git` is a FILE pointing back out of the copy, and copying either shape gives
# a tree git commands would follow out of. TOP LEVEL only — a `.git` deeper in
# the tree is not gitignored either, and the root allowlist keeps the top-level
# one out of the sdist regardless (`/.git` is not in the include list), so
# restricting it here is what keeps the copy and the tree in agreement.
_NOT_COPIED_AT_ROOT = (".git",)

# `.gitignore` is in NEITHER list: hatchling reads it, and a copy without it
# would build under different rules from the repository, which is the one thing
# the copy must not do.


def _copy_ignore(repo: pathlib.Path):
    """A ``copytree`` ignore callable that agrees with ``git check-ignore``.

    ``shutil.ignore_patterns`` cannot express "only when it is a directory",
    and that is exactly the distinction `.gitignore` draws for every name in
    :data:`_NOT_COPIED_DIRS`. A symlink is not a directory for the
    trailing-slash rule, hence the ``is_symlink`` guard — and the copy is made
    with ``symlinks=True``, so a symlink is copied as a symlink either way.
    """
    root = repo.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        here = pathlib.Path(directory).resolve()
        drop = set()
        if here == root:
            drop |= {n for n in names if n in _NOT_COPIED_AT_ROOT}
        for name in names:
            if name not in _NOT_COPIED_DIRS:
                continue
            entry = here / name
            if entry.is_dir() and not entry.is_symlink():
                drop.add(name)
        return drop

    return ignore


def _tree_to_build(tmp_path: pathlib.Path) -> pathlib.Path:
    """A private copy of the repository, for interventions that must not be
    performed on the checkout the suite is running in.

    Measured, at 33fd1f8: 0.01s and 5.3 MiB with the exclusions above.
    """
    staged = tmp_path / "tree"
    shutil.copytree(REPO, staged, ignore=_copy_ignore(REPO), symlinks=True)
    assert (staged / "pyproject.toml").is_file(), "the copy is not a source tree"
    assert (staged / "src" / "stelling" / "__init__.py").is_file()
    return staged


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs `uv` to build an sdist")
def test_an_arbitrary_new_file_does_not_ship(tmp_path: pathlib.Path) -> None:
    """THE property, established by intervention rather than by naming a file.

    Break it: replace the allowlist with hatchling's default (delete the
    ``[tool.hatch.build.targets.sdist]`` table) and this fails. Driven, on a
    copy exactly like the one built here: the probe appears in the tarball.

    **Two things this test could not previously say, and now must.**

    It ran the intervention on the REPO ROOT of the live checkout, so two suite
    runs against one checkout raced — `test_no_untracked_file_anywhere_would_ship`
    saw the other run's probe and failed, and this test hit "probe path is
    already taken". The tree built here is therefore a copy nothing else can
    see.

    And it never re-read that the intervention had happened. Driven, with
    ``probe.write_text`` replaced by ``pass``: ``1 passed``. An absence proves
    nothing unless the thing was there to be found, so the probe's existence is
    asserted before AND after the build (a build that ran over a tree without
    it observed nothing), and a SECOND probe is dropped inside an allowlisted
    directory where the root allowlist does not reach — that one MUST ship.
    It is the positive control: it fails if the build stops reading this tree
    (from git, from a cache, from an unpacked copy) or stops seeing untracked
    files at all, which are exactly the conditions under which the first
    probe's absence would mean nothing.
    """
    staged = _tree_to_build(tmp_path)
    probe = staged / "zz_sdist_allowlist_probe.txt"
    assert not probe.exists(), "probe path is already taken in a fresh copy"
    probe.write_text("this file must never reach an artefact\n")
    # inside `/docs`, which IS allowlisted: the root allowlist has root-level
    # granularity and hatchling then takes what is not gitignored, so this one
    # is expected to ship and its absence would mean the build saw neither.
    positive = staged / "docs" / "zz_sdist_positive_control.txt"
    assert not positive.exists(), "positive-control path is already taken"
    positive.write_text("an untracked file inside an allowlisted directory ships\n")
    assert probe.is_file() and positive.is_file(), (
        "the intervention did not happen — nothing below could distinguish "
        "'the allowlist held' from 'there was nothing to hold out'"
    )

    out = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--offline", "--sdist", "--out-dir", str(out), str(staged)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"sdist build failed:\n{proc.stderr}"
    assert probe.is_file(), (
        "the probe vanished while the sdist was being built — hatchling did "
        "not necessarily read a tree containing it, so its absence from the "
        "artefact says nothing"
    )
    built = sorted(out.glob("*.tar.gz"))
    assert len(built) == 1, f"expected one sdist, got {built}"
    with tarfile.open(built[0]) as tf:
        names = tf.getnames()
    # the control must be able to see something, or its zero is vacuous
    assert any(n.endswith("/pyproject.toml") for n in names), (
        "the built sdist does not even contain pyproject.toml — this test is "
        "not looking at a real artefact"
    )
    assert [n for n in names if n.endswith(positive.name)], (
        "the POSITIVE control did not ship. An untracked file inside the "
        "allowlisted /docs must reach the sdist; that it did not means this "
        "build is not reading the tree the probe was dropped in, and the "
        "probe's absence below would be vacuous rather than reassuring."
    )
    leaked = [n for n in names if n.endswith(probe.name)]
    assert not leaked, (
        f"an arbitrary untracked file reached the sdist: {leaked}. The sdist "
        "allowlist is not holding; hatchling's default ships everything that "
        "is not gitignored."
    )


# --- the scan against the artefact, on a tree that carries every hazard ------

# Every path this control plants, with what the BUILD must do with it. The four
# `True`s below are the finding: each is hidden from `git status` by a source
# hatchling does not read, and each reaches the tarball.
_PARITY_SHIPS = {
    # untracked, nothing excludes it — the base case
    "docs/zz_parity_plain.md": True,
    # hidden by `.git/info/exclude`
    "docs/zz_parity_info_exclude.md": True,
    # hidden by the developer's global `core.excludesFile`
    "docs/zz_parity_global_excludes.md": True,
    # hidden by a TRACKED nested `.gitignore` — hatchling reads the root one only
    "docs/zz_parity_nested_gitignore.md": True,
    # the NEGATIVE controls: a guard that fires on these cries wolf on every
    # developer machine and gets switched off.
    #   matched by the ROOT `.gitignore` (`*.log`) — the one source hatchling
    #   DOES read, so this is "git hides it AND the build drops it"
    "docs/zz_parity_quiet.log": False,
    #   matched by hatchling's own `default_global_exclude` (`*.py[cdo]`)
    "docs/zz_parity_bytecode.pyc": False,
    #   inside a directory hatchling prunes by name
    "docs/__pycache__/zz_parity_cached.py": False,
    #   ... and the SAME mechanism on a name this repository's own `.gitignore`
    #   does NOT cover. `__pycache__/` is gitignored here, so the plant above
    #   is dropped twice over and dropping `__pycache__` from
    #   `_HATCH_EXCLUDED_DIRECTORIES` changed no measurement anywhere —
    #   driven at a4c16fe: 12 passed. `.hatch` is not gitignored in this tree,
    #   so this plant is what makes that constant load-bearing.
    "docs/.hatch/zz_parity_hatch.md": False,
    #   dropped by BASENAME (`EXCLUDED_FILES`), and `.DS_Store` is likewise not
    #   gitignored here, so `_HATCH_EXCLUDED_FILES` is load-bearing too
    "docs/.DS_Store": False,
    #   outside the allowlist entirely: `/scratchpad` is not an include entry
    "scratchpad/zz_parity_note.md": False,
}
# The three the scan must see and `git status --porcelain -uall` must not.
_PARITY_HIDDEN_FROM_GIT = (
    "docs/zz_parity_info_exclude.md",
    "docs/zz_parity_global_excludes.md",
    "docs/zz_parity_nested_gitignore.md",
)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs `uv` to build an sdist")
def test_the_untracked_scan_agrees_with_the_tarball(tmp_path: pathlib.Path) -> None:
    """PARITY, measured — the scan's answer against `tar tzf`, not against a
    reading of hatchling.

    :func:`_untracked_that_would_ship` is a model of a build backend, and a
    model of a build backend is wrong the moment the backend moves. Three
    constants are transcribed from hatchling 1.31.0 (`EXCLUDED_DIRECTORIES`,
    `EXCLUDED_FILES`, `default_global_exclude`), the include patterns are
    matched by first path component rather than by `pathspec`, and
    `pyproject.toml` requires only `hatchling>=1.27`. None of that is trusted.

    A private copy of this tree is made into a real git repository, eight files
    are planted — four that must ship and four that must not, each hidden or
    admitted by a DIFFERENT mechanism — an sdist is built from it, and the set
    the scan predicts is asserted **equal** to the set of untracked files the
    tarball actually contains. Not "contains the probes": equal, over the whole
    tree, so a transcription that drifts in either direction is a set
    difference here.

    The two halves this buys:

    * POSITIVE — a file `git status --untracked-files=all` does not mention,
      because `.git/info/exclude`, `core.excludesFile` or a nested
      `.gitignore` covers it, is in the tarball and in the scan's answer. That
      is the defect this pass exists for, asserted rather than narrated.
    * NEGATIVE — a `.log`, a `.pyc`, a `__pycache__` member and a
      `scratchpad/` file are absent from the tarball and absent from the
      scan's answer. A guard that fired on those would be red in every
      checkout that has ever run the test suite.

    Break it, and this is driven rather than proposed: point
    :func:`_hatchling_excluded`'s ``check-ignore`` at the tree itself instead
    of at the isolated oracle — one word, ``cwd=root`` — and git starts
    answering with all four of its exclusion sources. Measured::

        the tarball ships these untracked files and the scan did not name them:
            docs/zz_parity_info_exclude.md
            docs/zz_parity_nested_gitignore.md

    Two of the three, not all three: the invocation still carries
    ``-c core.excludesFile=/dev/null``, so the global-gitignore case survives
    that particular mistake and the other two do not.
    """
    allow = _allowlist()
    _assert_allowlist_is_plain(allow)
    staged = _tree_to_build(tmp_path)

    # A real repository, so that `git status` has something to be wrong about.
    _git(["init", "-q", "."], staged)
    # The nested `.gitignore` is COMMITTED. Untracked, it would show up in
    # `git status` itself and the miss would be visible; tracked, the tree is
    # clean and the file it hides is invisible — which is the shape this
    # repository already has at `scratchpad/reach/.gitignore`.
    (staged / "docs" / ".gitignore").write_text("zz_parity_nested_gitignore.md\n")
    _git(["add", "-A"], staged)
    _git(
        [
            "-c", "user.email=parity@example.invalid",
            "-c", "user.name=parity control",
            "commit", "-q", "-m", "the tree as it stands",
        ],
        staged,
    )

    (staged / ".git" / "info" / "exclude").write_text("zz_parity_info_exclude.md\n")
    global_excludes = tmp_path / "global_gitignore"
    global_excludes.write_text("zz_parity_global_excludes.md\n")
    _git(["config", "core.excludesFile", str(global_excludes)], staged)

    for relative in _PARITY_SHIPS:
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), f"probe path already taken: {relative}"
        target.write_text(f"parity probe: {relative}\n")

    # THE FINDING ITSELF: the channel the guard used to enumerate on cannot see
    # three of the four files that ship.
    status = _git(["status", "--porcelain", "--untracked-files=all"], staged).stdout
    seen_by_status = {
        ln[3:].strip().strip('"') for ln in status.splitlines() if ln.startswith("?? ")
    }
    assert "docs/zz_parity_plain.md" in seen_by_status, (
        "`git status` cannot see a plainly untracked file, so this control is "
        "not measuring what it claims to measure"
    )
    invisible = [p for p in _PARITY_HIDDEN_FROM_GIT if p not in seen_by_status]
    assert invisible == list(_PARITY_HIDDEN_FROM_GIT), (
        "`git status --untracked-files=all` reported a file that is excluded "
        f"only by a source hatchling does not read: {sorted(set(_PARITY_HIDDEN_FROM_GIT) - set(invisible))}. "
        "If git has stopped honouring one of those sources, this control's "
        "premise is gone and the guard must be re-derived."
    )

    scanned = set(_untracked_that_would_ship(staged, allow, tmp_path / "parity-oracle"))

    out = tmp_path / "dist-parity"
    proc = subprocess.run(
        ["uv", "build", "--offline", "--sdist", "--out-dir", str(out), str(staged)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"sdist build failed:\n{proc.stderr}"
    built = sorted(out.glob("*.tar.gz"))
    assert len(built) == 1, f"expected one sdist, got {built}"
    with tarfile.open(built[0]) as tf:
        members = tf.getnames()
    assert any(n.endswith("/pyproject.toml") for n in members), (
        "the built sdist has no pyproject.toml — this is not a real artefact "
        "and the comparison below would be vacuous"
    )
    # `stelling-0.1.0/docs/x` -> `docs/x`.
    #
    # THIS USED TO BE FILTERED BY `(staged / rel).is_file()`, AND THAT FILTER
    # DROPPED EXACTLY THE MEMBERS THIS CONTROL EXISTS TO FIND. A member with no
    # counterpart in the tree is the definition of a force-included external
    # file — an ancestor's `.hgignore` arriving as `/.hgignore`, an ancestor's
    # `.gitignore` arriving as `/.gitignore` when the build root has no `.git`
    # — and the filter made every one of them invisible to the comparison
    # below. Driven: with `printf 'syntax: glob\nzz_never\n' > <parent>/.hgignore`
    # the tarball went 260 -> 261 members and the suite stayed at `8 passed`.
    # Now the ONE generated member is named, and everything else must have a
    # counterpart.
    delivered = {n.split("/", 1)[1] for n in members if "/" in n}

    # GENERATED_IN_DISTRIBUTION is SUBTRACTED below, so every name in it is
    # exempt from the two assertions that follow — and nothing used to check
    # that a name in it is generated at all. That is the escape hatch WITHHELD
    # used to be, and it is closed the same way: by asking the question the
    # dict claims to answer. A backend-generated member has no counterpart in
    # the tree and is not force-included from one; and a name that is not in
    # the artefact at all is a rotted entry widening the exemption for nothing.
    forced_here = _force_included(staged)
    dishonest = sorted(
        name
        for name in GENERATED_IN_DISTRIBUTION
        if (staged / name).exists()
        or (staged / name).is_symlink()
        or name in forced_here
    )
    assert not dishonest, (
        "these names are recorded in GENERATED_IN_DISTRIBUTION and are NOT "
        "generated — the tree has a file at that path, or hatchling "
        "force-includes one there:\n    "
        + "\n    ".join(f"{name}  <- {forced_here.get(name, staged / name)}" for name in dishonest)
        + "\n\nThat dict exempts a name from the counterpart check below and "
        "from the scan/tarball comparison, so an entry that is really a file "
        "buys a shipping path a silent pass. Driven at a4c16fe with a root "
        "`.hgignore`: `1 failed` -> `12 passed`, 262 members, the file still "
        "in the tarball."
    )
    unused = sorted(set(GENERATED_IN_DISTRIBUTION) - delivered)
    assert not unused, (
        "these names are recorded in GENERATED_IN_DISTRIBUTION and the built "
        "sdist does not contain them:\n    "
        + "\n    ".join(unused)
        + "\n\nAn entry nothing ships is an exemption granted to a name the "
        "backend does not use, and it will still be granted the day something "
        "else arrives under it."
    )

    shipped = delivered - set(GENERATED_IN_DISTRIBUTION)
    # A member with NOTHING at its path, and a member with something that is
    # not a regular file, are different findings and used to share one message.
    # A dangling symlink is `is_file() == False` and was reported as "it was
    # FORCE-INCLUDED", which it was not — the diagnosis sent the reader to the
    # wrong mechanism entirely.
    missing = sorted(
        rel
        for rel in shipped
        if not (staged / rel).exists() and not (staged / rel).is_symlink()
    )
    irregular = sorted(
        rel
        for rel in shipped
        if rel not in set(missing) and not (staged / rel).is_file()
    )
    assert not missing, (
        "the sdist ships these members and the tree it was built from has "
        "nothing at all at that path:\n    "
        + "\n    ".join(missing)
        + "\n\nA member with no counterpart in the source tree did not go "
        "through the include allowlist — it was FORCE-INCLUDED, which bypasses "
        "`include_path()` altogether. The known routes are an `.hgignore` or "
        "`.gitignore` found by walking UP out of the build root (both arrive "
        "keyed by basename at the root of the distribution), a `force-include` "
        "table in `pyproject.toml`, and a `hatch_build.py` hook. Nothing in "
        "`pyproject.toml` can exclude one. If a member here is generated by the "
        "backend rather than taken from a file, name it in "
        "`GENERATED_IN_DISTRIBUTION` and say why."
    )
    assert not irregular, (
        "the sdist ships these members and the tree has something at that path "
        "that is not a regular file — a dangling symlink, a directory, a "
        "socket:\n    "
        + "\n    ".join(f"{rel}  ({os.readlink(staged / rel) if (staged / rel).is_symlink() else 'not a file'})" for rel in irregular)
        + "\n\nThis is NOT the force-include finding above, and it used to be "
        "reported as one: the path is a real entry of this tree and went "
        "through the allowlist. Measured on a committed dangling "
        "`docs/zz_dangling.md -> zz_nowhere.md`: the build SUCCEEDS and the "
        "member is stored as a SYMLINK (`issym`, size 0, linkname preserved), "
        "not as the bytes it points at — so unpacking the sdist recreates the "
        "link, and where it lands is decided on the consumer's machine."
    )
    tracked = _tracked_files(staged)
    shipped_untracked = shipped - tracked

    assert shipped_untracked, (
        "no untracked file reached the sdist at all, so the comparison below "
        "could not tell a correct scan from one that never looked"
    )
    for relative, must_ship in _PARITY_SHIPS.items():
        assert (relative in shipped_untracked) is must_ship, (
            f"{relative}: the ARTEFACT "
            f"{'omits' if must_ship else 'contains'} it, and the premise of "
            f"this control is that it {'ships' if must_ship else 'does not'}. "
            "Hatchling's selection has moved; re-derive the model before "
            "trusting either direction of this test."
        )
    assert scanned == shipped_untracked, (
        "the untracked-file scan and the built sdist disagree.\n"
        "  the scan says these would ship and the tarball has no such member:\n    "
        + "\n    ".join(sorted(scanned - shipped_untracked) or ["(none)"])
        + "\n  the tarball ships these untracked files and the scan did not name them:\n    "
        + "\n    ".join(sorted(shipped_untracked - scanned) or ["(none)"])
        + "\n\nThe second list is the dangerous one: it is what would reach PyPI "
        "with nothing red. The model of hatchling in this module (constants "
        f"transcribed at {_HATCHLING_READ_AT}, the `.gitignore` and `.hgignore` "
        "found by walking up as the only VCS sources, the allowlist matched by "
        "first path component) is what has to move."
    )


# --- controls on the controls ----------------------------------------------
#
# `_untracked_that_would_ship` carries two non-vacuity assertions, and NEITHER
# WAS PINNED. Driven at 10120d9, in a standalone repo from this tree:
#
#   delete `assert not unseen`                          8 passed
#   `assert tracked` -> `assert True or tracked`        8 passed
#
# An assertion nothing fails when it is removed is indistinguishable from a
# comment. The three tests below are what fail. They use no monkeypatching: the
# blinding is a real one, performed on a real repository, because a
# monkeypatched blinding pins the patch and not the code.


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_scan_refuses_an_empty_index(tmp_path: pathlib.Path) -> None:
    """`git ls-files` returning nothing must not read as "nothing would ship".

    An empty index is what a broken `GIT_DIR`, a fresh `git init` or a
    `--work-tree` pointed somewhere else all look like from inside the scan,
    and in every one of those the walk's silence about untracked files is
    silence about a tree the scan never saw.
    """
    repo = tmp_path / "no-index"
    repo.mkdir()
    _git(["init", "-q", "."], repo)
    (repo / "stray.md").write_text("untracked, and nothing is tracked\n")
    assert not _tracked_files(repo), "this control needs an EMPTY index"
    with pytest.raises(AssertionError, match="reported no tracked files"):
        _untracked_that_would_ship(repo, {"stray.md"}, tmp_path / "oracle-empty")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_scan_refuses_a_blinded_walk(tmp_path: pathlib.Path) -> None:
    """A walk that cannot reach the tree must be a RED and not a green.

    The blinding here is real rather than patched: the one tracked file is
    deleted from disk, so the index names it and `os.walk` cannot. That is the
    same shape as a walk pointed at the wrong root, and the same shape a
    submodule presents — which is why the assertion must survive the
    :func:`_pruned_by_the_walk` subtraction added for `docs/.hatch/note.md`.

    The second half is the NEGATIVE control, and it is the false RED this
    branch's guard actually had: a legitimately tracked file under a directory
    hatchling prunes by name ships nothing, so it must not redden anything.
    """
    def _repo(name: str, tracked_at: str, *, delete: bool) -> pathlib.Path:
        repo = tmp_path / name
        (repo / tracked_at).parent.mkdir(parents=True, exist_ok=True)
        (repo / tracked_at).write_text("content\n")
        _git(["init", "-q", "."], repo)
        _git(["add", "-A", "-f"], repo)
        _git(
            [
                "-c", "user.email=control@example.invalid",
                "-c", "user.name=blinding control",
                "commit", "-q", "-m", "one file",
            ],
            repo,
        )
        assert tracked_at in _tracked_files(repo)
        if delete:
            (repo / tracked_at).unlink()
        return repo

    blinded = _repo("blinded", "a.txt", delete=True)
    with pytest.raises(AssertionError, match="filesystem walk reached"):
        _untracked_that_would_ship(blinded, {"a.txt"}, tmp_path / "oracle-blind")

    # NEGATIVE: tracked, present, and pruned by hatchling's own walk. Nothing
    # ships and nothing must be red — a guard that cries wolf here is a guard
    # that gets switched off.
    pruned = _repo("pruned", "docs/.hatch/note.md", delete=False)
    assert _untracked_that_would_ship(pruned, {"docs"}, tmp_path / "oracle-pruned") == []


def test_a_record_of_a_force_included_path_is_read_as_false() -> None:
    """:func:`_falsely_recorded` must match the way WITHHELD is matched
    everywhere else, and must look in BOTH dicts.

    The assertion in ``test_every_root_entry_is_a_decision`` is an absence over
    this checkout's four force-included paths, and an absence proves nothing
    unless the predicate can produce a presence. It could not, twice over:
    matching the full distribution path missed everything inside a withheld
    DIRECTORY (the `license-files = [… "scratchpad/PREREG*.md"]` construction,
    12 passed, 10 withheld files in the tarball), and it never consulted
    :data:`GENERATED_IN_DISTRIBUTION` at all (a committed root `.hgignore`
    plus one dict entry, ``1 failed`` -> ``12 passed``, file shipping).

    The NEGATIVE half is the tree's real force-included paths: `README.md`,
    `LICENSE`, `pyproject.toml`, `.gitignore` all ship and are recorded
    nowhere, so a predicate that answered "false record" for them would be red
    in every checkout.
    """
    assert "scratchpad" in WITHHELD and "PKG-INFO" in GENERATED_IN_DISTRIBUTION, (
        "this control names two keys that are no longer in the dicts, so it is "
        "no longer measuring the matching rule"
    )
    inside_withheld = _falsely_recorded("scratchpad/PREREG_SDIST.md")
    assert inside_withheld and "WITHHELD" in inside_withheld, (
        "a force-included path inside a WITHHELD directory was not read as a "
        "false record, which is the construction that shipped 10 of them with "
        "the suite green"
    )
    assert _falsely_recorded(".git"), "an exact WITHHELD key was not matched"
    generated = _falsely_recorded("PKG-INFO")
    assert generated and "GENERATED" in generated, (
        "GENERATED_IN_DISTRIBUTION is not consulted, so it is still the escape "
        "hatch WITHHELD used to be"
    )
    for benign in ("README.md", "LICENSE", "pyproject.toml", ".gitignore"):
        assert _falsely_recorded(benign) is None, (
            f"{benign} is force-included AND shipped AND recorded nowhere; "
            "reading it as a false record would be red in every checkout"
        )


def test_the_force_include_model_reads_the_static_table(
    tmp_path: pathlib.Path,
) -> None:
    """:func:`_static_force_include` and :data:`_HATCH_FORCED_ROOT_FILES`,
    pinned on a synthetic root.

    The `force-include` table is the cheapest force-include route there is —
    two lines of `pyproject.toml`, no build hook, no ancestor directory — and
    it was neither modelled nor named among the routes this module says it
    cannot model. It is modelled now, and this is what stops the model from
    quietly returning ``{}`` again.

    ``_HATCH_FORCED_ROOT_FILES`` is pinned here rather than by the parity
    build for a reason that is worth writing down: two of its three entries,
    `hatch.toml` and `hatch_build.py`, DO NOT EXIST in this tree, so reducing
    the constant to ``("pyproject.toml",)`` changes no measurement any build
    can make. Driven at a4c16fe: ``12 passed``, vacuously. A synthetic root
    that has all three is the only place the constant can be non-vacuous.
    """
    root = tmp_path / "static-table"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "payload.md").write_text("forced\n", encoding="utf-8")
    (root / "tree").mkdir()
    (root / "tree" / "a.md").write_text("a\n", encoding="utf-8")
    (root / "tree" / "__pycache__").mkdir()
    (root / "tree" / "__pycache__" / "b.py").write_text("b\n", encoding="utf-8")
    # NAMED HERE AS LITERALS, not read out of the constant. A loop that plants
    # and then checks `_HATCH_FORCED_ROOT_FILES` shrinks with the constant and
    # pins nothing — reducing it to `("pyproject.toml",)` stays green.
    (root / "hatch.toml").write_text("# forced at the root\n", encoding="utf-8")
    (root / "hatch_build.py").write_text("# forced at the root\n", encoding="utf-8")

    def _write(table: str) -> None:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "zz-probe"\nversion = "0"\n' + table, encoding="utf-8"
        )

    _write("")
    plain = _force_included(root)
    for name in ("pyproject.toml", "hatch.toml", "hatch_build.py"):
        assert plain.get(name) == root / name, (
            f"`{name}` is force-included unconditionally by "
            f"`get_default_build_data` (hatchling {_HATCHLING_READ_AT}) and "
            f"the model did not name it: {sorted(plain)}"
        )

    # a FILE source, relocated by the table
    _write('\n[tool.hatch.build]\nforce-include = { "pkg/payload.md" = "docs/moved.md" }\n')
    assert _force_included(root).get("docs/moved.md") == root / "pkg" / "payload.md", (
        "the static `force-include` table was not read — this is the two-line "
        "route that shipped a WITHHELD file with the suite at 12 passed"
    )

    # the TARGET table SHADOWS the global one; it is not merged
    _write(
        '\n[tool.hatch.build]\nforce-include = { "pkg/payload.md" = "global.md" }\n'
        '\n[tool.hatch.build.targets.sdist]\n'
        'force-include = { "pkg/payload.md" = "target.md" }\n'
    )
    shadowed = _force_included(root)
    assert "target.md" in shadowed and "global.md" not in shadowed, (
        "`force_include` is `if 'force-include' in target_config … else "
        "build_config` upstream, not a merge, and the model disagrees: "
        f"{sorted(shadowed)}"
    )

    # a DIRECTORY source takes everything under it, pruned like the walk
    _write('\n[tool.hatch.build]\nforce-include = { "tree" = "vendored" }\n')
    walked = _force_included(root)
    assert walked.get("vendored/a.md") == root / "tree" / "a.md", (
        f"a directory source did not force-include its files: {sorted(walked)}"
    )
    assert "vendored/__pycache__/b.py" not in walked, (
        "`recurse_forced_files` prunes EXCLUDED_DIRECTORIES and the model did "
        "not"
    )

    # NEGATIVE: no table, nothing extra. A model that invented entries would
    # be red on this repository, which has no `force-include` table.
    _write("")
    assert not _static_force_include(
        root, tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ), "the model reported a static force-include with no table in the file"
    assert not _static_force_include(
        REPO, tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    ), "this repository has grown a `force-include` table; teach the guard first"


def test_no_build_hook_can_force_include_behind_this_module() -> None:
    """A `hatch_build.py` hook is UNMODELLABLE, so its appearance is a red.

    ``_force_included``'s docstring used to say "This project has no
    `hatch_build.py`; if one appears, the parity test is what will notice."
    **It does not.** Driven at a4c16fe, in a standalone repo from this tree —
    a committed, allowlisted hook, registered under
    ``[tool.hatch.build.targets.sdist.hooks.custom]``, whose ``initialize``
    puts ``scratchpad/PREREG_SDIST.md`` into ``build_data["force_include"]``::

        pytest -q -ra tests/test_sdist_contents.py   12 passed
        uv build --offline --sdist .                 263 members (baseline 261)
        tar tzf …  ->  stelling-0.1.0/hatch_build.py
                       stelling-0.1.0/scratchpad/PREREG_SDIST.md

    Nothing reddened, because the hook ships a file that IS tracked and IS in
    the tree: every counterpart check the parity test can make is satisfied.
    A model cannot follow arbitrary Python, and the parity comparison is a
    comparison against the tree — so the only honest guard is to refuse the
    CAPABILITY. If a hook is ever genuinely wanted, this test is the place the
    decision gets written down, next to what it force-includes.
    """
    hook = REPO / "hatch_build.py"
    assert not hook.is_file(), (
        f"{hook} exists. A build hook can put ANY path into "
        '`build_data["force_include"]` at build time, past the allowlist and '
        "past the exclusion spec, and no model in this module can know what "
        "without executing it. The parity test does not catch it either: a "
        "hook that ships a tracked file leaves every counterpart check "
        "satisfied. Say here what it force-includes and why, or delete it."
    )
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    build = cfg.get("tool", {}).get("hatch", {}).get("build", {})
    targets = build.get("targets", {})
    assert targets, (
        "`pyproject.toml` declares no `[tool.hatch.build.targets.*]` tables at "
        "all, so this scan looked at nothing and its silence is not an answer"
    )
    tables = {"tool.hatch.build.hooks": build.get("hooks")}
    for target, target_cfg in targets.items():
        tables[f"tool.hatch.build.targets.{target}.hooks"] = target_cfg.get("hooks")
    declared = sorted(name for name, table in tables.items() if table)
    assert not declared, (
        "`pyproject.toml` registers build hooks:\n  "
        + "\n  ".join(declared)
        + "\n\nA registered hook runs arbitrary code at build time and may "
        "force-include anything. Nothing in this module models it and the "
        "parity test does not notice it."
    )


def test_the_force_include_review_sees_an_outside_source(
    tmp_path: pathlib.Path,
) -> None:
    """:func:`_forced_without_a_reviewed_source` must find the shape it exists
    for, on a root that HAS one.

    ``test_no_force_included_path_escapes_review`` asserts an absence in this
    checkout, and an absence is worth nothing unless the instrument can produce
    a presence. Blinded three ways at a4c16fe under the DEFAULT ``tmp_path``,
    with a real ancestor `.hgignore` shipping, the module stayed at
    ``12 passed`` — ``_force_included -> {}``,
    ``_forced_without_a_reviewed_source -> []``, and `.hgignore` dropped from
    :data:`_HATCH_VCS_EXCLUSION_FILES`. The first is pinned by that test's
    non-vacuity assertion; the other two are pinned here.

    No git and no build: the whole point is that this route is open in a tree
    that has neither. The NEGATIVE half is the same root with the plant
    removed — a guard that reports an outside source when there is none would
    be red on every machine.
    """
    root = tmp_path / "outside-source" / "repo"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "zz-probe"\nversion = "0"\n', encoding="utf-8"
    )
    planted = root.parent / ".hgignore"
    planted.write_text("syntax: glob\nzz_never\n", encoding="utf-8")

    # `.hgignore`'s boundary is `.hg`, which this root does not have, so
    # `locate_file` walks out of the tree and hatchling force-includes the
    # ancestor's file at the ROOT of the distribution, keyed by basename.
    assert _force_included(root).get(".hgignore") == planted, (
        "the force-include model did not locate an ancestor `.hgignore` at all, "
        f"so nothing below is being measured: {sorted(_force_included(root))}"
    )
    named = [e for e in _forced_without_a_reviewed_source(root, None) if ".hgignore" in e]
    assert len(named) == 1 and "OUTSIDE the tree" in named[0], (
        "an ancestor's `.hgignore` is force-included into the sdist under this "
        "project's name and the review did not report it as an outside source: "
        f"{named}"
    )
    # ... and it is reported the same way when the index IS readable, because
    # the degradation must not be what is doing the work.
    assert named == [
        e for e in _forced_without_a_reviewed_source(root, set()) if ".hgignore" in e
    ]

    # NEGATIVE: no plant, no report. This is the state of every ordinary
    # checkout, and a guard that fires here gets switched off.
    planted.unlink()
    assert not [
        e for e in _forced_without_a_reviewed_source(root, None) if ".hgignore" in e
    ], "the review reported an ancestor `.hgignore` that no longer exists"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_exclusion_classifier_discriminates(tmp_path: pathlib.Path) -> None:
    """The classifier must answer differently for different inputs.

    A :func:`_hatchling_excluded` that returned everything would make
    :func:`_untracked_that_would_ship` return ``[]`` for every tree, and one
    that returned nothing would make it report the whole working directory.
    Both directions are asserted here on a synthetic root, so the scan test's
    zero cannot come from a classifier that has stopped classifying.

    The third case is the bail-out: a root whose own path matches its
    `.gitignore` gets NO exclusions at all, which is what
    ``test_the_checkout_path_does_not_disable_the_exclusions`` guards for the
    real checkout.
    """
    root = tmp_path / "classify"
    root.mkdir()
    (root / ".gitignore").write_text("*.log\n")
    queries = {"a/keep.md", "a/drop.log", "a/drop.pyc"}
    excluded = _hatchling_excluded(root, queries, tmp_path / "oracle-classify")
    assert excluded == {"a/drop.log", "a/drop.pyc"}, (
        "the exclusion classifier did not separate a `.gitignore` match "
        "(`a/drop.log`) and a `default_global_exclude` match (`a/drop.pyc`) "
        f"from a plain file: {sorted(excluded)}"
    )

    # the bail-out, exercised rather than described: name the root itself in
    # its own `.gitignore` and hatchling keeps none of the patterns.
    (root / ".gitignore").write_text(f"*.log\n/{root.resolve().relative_to('/').parts[0]}/\n")
    assert _vcs_exclusions_are_discarded(root, tmp_path / "oracle-bail-probe"), (
        "a root whose own absolute path matches its `.gitignore` was not "
        "detected, so the checkout-path guard cannot fire for anybody"
    )
    after = _hatchling_excluded(root, queries, tmp_path / "oracle-bail")
    assert after == {"a/drop.pyc"}, (
        "with the VCS exclusions discarded only `default_global_exclude` "
        f"should survive, and this answered {sorted(after)}"
    )
