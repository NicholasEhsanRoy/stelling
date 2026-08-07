# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What the sdist ships is an ALLOWLIST, and this holds it shut.

The sdist is the one artefact where a mistake is immutable: it cannot be
unpublished from PyPI. Hatchling's *default* sdist takes everything not
gitignored — tracked or not — which is fail-OPEN, and it shipped an internal
file that was protected only by being uncommitted. `.git/info/exclude` does not
protect against it either; hatchling reads `.gitignore` files and nothing else.

Two tests, in the order they matter:

1. ``test_an_arbitrary_new_file_does_not_ship`` — the real property, by
   INTERVENTION. Drop a file the allowlist has never heard of, build, and
   confirm it is absent. *Absence of a NAMED file is the weaker check and is
   exactly what would have passed before the leak*: the checklist was not
   named anywhere, it simply was not excluded.

2. ``test_every_root_entry_is_a_decision`` — a new path at the repo root must be
   either allowlisted or listed here as deliberately withheld. It cannot be
   neither. This one needs no build backend, so it runs everywhere and fails
   closed when someone adds a file and does not think about distribution.

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
        "directory. Harmless while nothing inside it would ship, and stated "
        "because 'harmless today' is a reason to write it down"
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


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_untracked_file_anywhere_would_ship() -> None:
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
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip("not a git checkout (an unpacked sdist, say)")
    undecided = []
    for ln in proc.stdout.splitlines():
        if not ln.startswith("?? "):
            continue
        path = ln[3:].strip().strip('"')
        if path.split("/", 1)[0] in WITHHELD or path in WITHHELD:
            continue
        undecided.append(path)
    assert not undecided, (
        "these paths are untracked, not gitignored, and would be DISTRIBUTED in "
        "the sdist:\n  " + "\n  ".join(undecided)
        + "\n\nThe root allowlist does not reach inside an allowlisted directory. "
        "Commit the file, gitignore it, or add it to WITHHELD with a reason."
    )


# Names pytest's own tmp copy never needs and hatchling never reads. `.git` is
# in here because a worktree's `.git` is a FILE and copying it produces a tree
# that git commands would follow back out of the copy; the rest are caches and
# build output that make the copy large for nothing. `.gitignore` is NOT here:
# hatchling reads it, and a copy without it would build under different rules
# from the repository, which is the one thing the copy must not do.
_NOT_COPIED = (
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "dist",
    "build",
)


def _tree_to_build(tmp_path: pathlib.Path) -> pathlib.Path:
    """A private copy of the repository, for interventions that must not be
    performed on the checkout the suite is running in.

    Measured, at 33fd1f8: 0.01s and 5.3 MiB with the exclusions above.
    """
    staged = tmp_path / "tree"
    shutil.copytree(REPO, staged, ignore=shutil.ignore_patterns(*_NOT_COPIED), symlinks=True)
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
