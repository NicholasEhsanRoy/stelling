# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What the sdist ships is an ALLOWLIST, and this holds it shut.

The sdist is the one artefact where a mistake is immutable: it cannot be
unpublished from PyPI. Hatchling's *default* sdist takes everything not
gitignored — tracked or not — which is fail-OPEN, and it shipped an internal
file that was protected only by being uncommitted. `.git/info/exclude` does not
protect against it either; hatchling reads ONE `.gitignore` — the one found by
walking up from the build root to the `.git` boundary — and no other exclusion
source. Not the user's `core.excludesFile`, not `.git/info/exclude`, and not a
`.gitignore` in any subdirectory.

`git status` reads all four. **Every source in that difference is a file the
guard below cannot see and the tarball can**, which is what
``test_the_untracked_scan_agrees_with_the_tarball`` now measures rather than
narrates.

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


# Root entries that exist and are deliberately NOT distributed. Each needs a
# reason, because "why is this not in the sdist" is the question a future reader
# will ask. A path in neither this set nor the allowlist fails the test.
WITHHELD = {
    ".git": "the repository itself",
    ".claude": (
        "local agent/tool configuration — untracked AND not gitignored, so the "
        "allowlist is the only thing keeping it out. It is currently EMPTY, so "
        "nothing of substance would have shipped; the point is that it was "
        "undecided, and the decision is recorded before content appears"
    ),
    ".gitignore": "listed in the allowlist; kept here only if it moves",
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
    """A new file at the repo root is shipped or withheld — never neither."""
    allow = _allowlist()
    undecided = sorted(
        p.name
        for p in REPO.iterdir()
        if p.name not in allow and p.name not in WITHHELD
    )
    assert not undecided, (
        "these root paths are neither in pyproject's sdist allowlist nor in "
        "WITHHELD:\n  "
        + "\n  ".join(undecided)
        + "\n\nAdd each to the allowlist (it ships) or to WITHHELD with a reason "
        "(it does not). An sdist on PyPI cannot be unpublished."
    )


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
# rots, so it is not trusted: `test_the_untracked_scan_agrees_with_the_tarball`
# holds every line of it against a real build, and a drift in any of the three
# constants below shows up there as a set difference and not as a silent pass.
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


def _hatchling_excluded(
    root: pathlib.Path, paths: set[str], oracle: pathlib.Path
) -> set[str]:
    """Which of `paths` hatchling's exclusion spec matches — asked of git, in a
    repository that has nothing in it but the patterns hatchling would use.

    `git check-ignore` run in the REAL tree is the wrong instrument for this:
    it answers with the winning pattern across all four of git's sources, and
    three of them are precisely what hatchling cannot see. So the patterns are
    composed here — `default_global_exclude()` first, then the ROOT
    `.gitignore` verbatim — written into a throwaway repository, and matched
    there. That repository has no `.git/info/exclude` entries, no nested
    `.gitignore`, and `core.excludesFile` is pointed at `/dev/null`, so the
    only thing that can decide is the thing hatchling reads.

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
    lines = [*_HATCH_DEFAULT_GLOBAL_EXCLUDE]
    root_gitignore = root / ".gitignore"
    if root_gitignore.is_file():
        lines.extend(root_gitignore.read_text(encoding="utf-8").splitlines())
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


def _untracked_that_would_ship(
    root: pathlib.Path, allow: set[str], oracle: pathlib.Path
) -> list[str]:
    """The property, computed by hatchling's rule: untracked AND admitted by
    the allowlist AND not excluded by the one `.gitignore` hatchling reads.

    Two non-vacuity assertions live here, because the caller cannot make them:
    the index must be non-empty, and **the walk must have reached every tracked
    file**. Zero items examined is the failure mode this whole pass is about,
    and the second assertion is what makes it impossible — blind the walk and
    every tracked file goes missing at once.
    """
    tracked = _tracked_files(root)
    assert tracked, (
        "`git ls-files` reported no tracked files, so there is no tree here to "
        "check and this must not read as 'no untracked file would ship'"
    )
    walked = _walked_files(root)
    unseen = sorted(tracked - walked)
    assert not unseen, (
        f"the filesystem walk reached {len(walked)} files and missed "
        f"{len(unseen)} that git says are tracked, so it is not looking at the "
        "tree git is looking at and its silence about untracked files means "
        "nothing. First few:\n  " + "\n  ".join(unseen[:10])
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
        "and hatchling reads ONE `.gitignore` — the root one. `.git/info/exclude`, "
        "your global `core.excludesFile` and any nested `.gitignore` hide a file "
        "from `git status` and not from the build.\n"
        "Commit the file, add it to the ROOT `.gitignore`, or take its directory "
        "out of the sdist allowlist."
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
    # `stelling-0.1.0/docs/x` -> `docs/x`; PKG-INFO is generated and has no
    # counterpart in the tree, so it drops out of `is_file()`.
    shipped = {
        rel for rel in (n.split("/", 1)[1] for n in members if "/" in n)
        if (staged / rel).is_file()
    }
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
        f"transcribed at {_HATCHLING_READ_AT}, the root `.gitignore` as the only "
        "VCS source, the allowlist matched by first path component) is what has "
        "to move."
    )
