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
    "scratchpad": (
        "per-session working directory: the pre-registration a repair pass "
        "writes before it measures anything. It is a record of what was "
        "PREDICTED, so it is kept in the tree it registers AGAINST — a "
        "registration a reader cannot reach from the repository it constrains "
        "is not a registration — and it is not distributed, because an sdist "
        "is the library and not its audit trail"
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
}


def test_every_allowlist_entry_exists() -> None:
    """An allowlist that names a deleted path rots silently and stops
    protecting the thing it was written for."""
    missing = sorted(e for e in _allowlist() if not (REPO / e).exists())
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


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs `uv` to build an sdist")
def test_an_arbitrary_new_file_does_not_ship(tmp_path: pathlib.Path) -> None:
    """THE property, established by intervention rather than by naming a file.

    Break it: replace the allowlist with hatchling's default (delete the
    ``[tool.hatch.build.targets.sdist]`` table) and this fails.
    """
    probe = REPO / "zz_sdist_allowlist_probe.txt"
    assert not probe.exists(), "probe path is already taken"
    probe.write_text("this file must never reach an artefact\n")
    try:
        proc = subprocess.run(
            ["uv", "build", "--offline", "--sdist", "--out-dir", str(tmp_path), str(REPO)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, f"sdist build failed:\n{proc.stderr}"
        built = sorted(tmp_path.glob("*.tar.gz"))
        assert len(built) == 1, f"expected one sdist, got {built}"
        with tarfile.open(built[0]) as tf:
            names = tf.getnames()
        # the control must be able to see something, or its zero is vacuous
        assert any(n.endswith("/pyproject.toml") for n in names), (
            "the built sdist does not even contain pyproject.toml — this test is "
            "not looking at a real artefact"
        )
        leaked = [n for n in names if n.endswith(probe.name)]
        assert not leaked, (
            f"an arbitrary untracked file reached the sdist: {leaked}. The sdist "
            "allowlist is not holding; hatchling's default ships everything that "
            "is not gitignored."
        )
    finally:
        probe.unlink(missing_ok=True)
