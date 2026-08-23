# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every text file in the checkout, for checks that scan the tree rather
than a list of pages.

**A PAGE LIST IS ONLY AS WIDE AS ITS LIST.** That is the defect this module
exists to let a check avoid: `test_the_DISCLOSURES_name_exactly_these_routes`
read three named pages and went red the day a routing moved a falsified
sentence into a fourth, which had been carrying it unexamined all along. A
check that scans the tree and holds a *partition* against a named set —
every file that states the claim must be listed, every listed file must
state it — cannot go quiet that way in either direction.

One walker and not one per caller, because two copies of "which files are
text" are two things to keep in step.

**AND THE SKIP SET IS NOT A SECOND LIST EITHER**, which it was until 0.2.0
D14. :data:`SKIP` was a hand-typed tuple of ten directory names matched
exactly, while `tests/test_sdist_contents.py`'s :data:`~test_sdist_contents.
WITHHELD` — the repository's one record of *"this exists in a checkout and
is deliberately not shipped"* — already named five more: `.claude`,
`.hypothesis`, `.pdm-build`, `.venv-prop` and `.vscode`. `SUFFIXES` covers
`.md` and `.py`, so a file under any of those five was WALKED, and the
callers below hold strict partitions (`set(found) == set(_SITES)`) over what
this yields. Measured on this tree at `a431646`, with the shipped
`SOUNDNESS.md` copied to `.claude/notes.md` — a plausible thing for a local
agent configuration to hold::

    tests/test_tripwire_gate_coverage.py    2 failed
    tests/test_tripwire_eager.py            1 failed

and again with `src/stelling/_tripwire/eager.py` and `SOUNDNESS.md` copied
into a `.venv-prop/lib/python3.12/site-packages/` layout — which is what
installing this project into a venv inside the tree produces::

    tests/test_tripwire_gate_coverage.py    3 failed
    tests/test_tripwire_eager.py            1 failed

Neither is a defect in the repository. Both are *"a check whose input
includes the developer's environment reports a different truth to different
people"*, which is the class four of five consecutive reds on `main` came
from in the week this was fixed.

**EXTENDING THE TUPLE WOULD HAVE BEEN THE SAME DEFECT ONE COMMIT LATER**:
two hand-maintained lists of directory names that must agree is the thing to
remove, not to lengthen. So `SKIP` is DERIVED from `WITHHELD`, and the one
respect in which the two must differ is stated at :data:`SKIP` and gated by
`tests/test_sdist_reference_hygiene.py::
test_the_walkers_skip_set_is_WITHHELD_and_not_a_second_list`.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
from collections.abc import Iterator

REPO = pathlib.Path(__file__).resolve().parents[1]

# The same idiom `tests/test_sdist_reference_hygiene.py` uses to reach the
# record, and for the same reason: `tests/` is not a package, so a module here
# is importable by bare name once its own directory is on the path. Done here
# rather than left to the importer so this module is self-sufficient — three
# callers import it and none of them should have to know where the record
# lives.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_sdist_contents import WITHHELD  # noqa: E402

#: What counts as text. Extensions rather than a content sniff, so the set
#: is readable and a new kind of file is a deliberate addition.
SUFFIXES = (".md", ".py", ".toml", ".cff", ".yml", ".yaml", ".txt")

#: Names a checkout GROWS, which are not the project's own prose. Derived
#: from :data:`test_sdist_contents.WITHHELD` and never typed beside it: that
#: dict is the record of what exists here and is deliberately not shipped,
#: every entry carries a reason a reader can check, and
#: `test_every_root_entry_is_a_decision` already requires a new root entry to
#: land in it. A name that has been through that decision has been through
#: this one.
#:
#: TWO RESPECTS IN WHICH THIS IS NOT SIMPLY `WITHHELD`, both deliberate and
#: both gated by `test_the_walkers_skip_set_is_WITHHELD_and_not_a_second_list`:
#:
#: 1. **`WITHHELD` is keyed by ROOT entry and this matches at ANY DEPTH.**
#:    That is not a widening this module invents — `__pycache__` and
#:    `.pytest_cache` occur nested far more often than at the root, and the
#:    walker has always matched every component. What the derivation adds is
#:    that a *nested* `build/`, `dist/` or `.venv/` is skipped by the same
#:    record that decides the root one, rather than by a coincidence of two
#:    tuples agreeing.
#: 2. **`*.egg-info` cannot be a `WITHHELD` key at all**, because it is a
#:    PATTERN and not a name: an editable install writes `stelling.egg-info`,
#:    a renamed project writes something else, and `WITHHELD` records exact
#:    root entries. It stays a suffix rule in :func:`is_skipped` below. The
#:    literal string `".egg-info"` used to sit in the tuple as well and never
#:    matched anything the suffix rule did not — `".egg-info".endswith(
#:    ".egg-info")` is true — so nothing is lost by its going.
SKIP = tuple(sorted(WITHHELD))

#: The pattern half of the rule above. A tuple so the gate can read it rather
#: than re-typing the string it is checking.
SKIP_SUFFIXES = (".egg-info",)


def is_skipped(relative: str | pathlib.PurePath) -> bool:
    """Does this repo-relative path lie inside something the checkout grew?

    Exposed rather than inlined into :func:`text_files` so the gate can drive
    it on paths that are not on this disk — a check of the walker that had to
    create a `.venv-prop/` to run would be an environment-dependent check of
    an environment-dependence fix.
    """
    parts = pathlib.PurePosixPath(relative).parts
    return any(
        part in SKIP or part.endswith(SKIP_SUFFIXES) for part in parts
    )


def tracked_paths(root: pathlib.Path = REPO) -> list[str] | None:
    """Every path in `root`'s git index, or `None` where git cannot say.

    **A SECOND QUESTION ABOUT "WHICH FILES", AND IT LIVES HERE FOR THE SAME
    REASON THE FIRST ONE DOES.** :func:`text_files` asks what is on this
    disk; this asks what the REPOSITORY has, and callers need one or the
    other depending on what their verdict is about. Two copies of *"ask git
    what is tracked"* would be two things to keep in step, which is what the
    module docstring is about.

    The distinction it exists to serve, stated once: **a check whose subject
    is the working directory may read the working directory; a check whose
    subject is the repository may not.** `tests/test_sdist_contents.py::
    test_every_root_entry_is_a_decision` is the first kind — what is HERE is
    exactly its subject. A pin on *"which roots does this project keep
    Python under"* is the second kind, and reading the directory for it made
    one stray untracked file a red.

    `None` is a THIRD answer and never an empty list, for the reason
    `test_sdist_contents.py::_git` gives: *"a command that could not run
    reports the same 'found nothing' as a command that ran and found
    nothing."* Callers decide what not-knowing means for them; it must never
    read as "this repository tracks nothing".

    A worktree's `.git` is a FILE, hence `exists()` — the same predicate
    `test_sdist_contents.py::_index_is_readable` and
    `test_sdist_reference_hygiene.py::_git_can_read_this_tree` use. A `.git`
    that IS there and a git that still fails is a defect rather than an
    environment, so it asserts.
    """
    if shutil.which("git") is None or not (root / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0 and proc.stdout.strip(), (
        f"`.git` is present under {root} and `git ls-files` failed "
        f"(rc={proc.returncode}, stderr={proc.stderr.strip()!r}), so a check "
        "that reads the repository could not run — and a check that could "
        "not run must not report the same thing as one that ran and was "
        "satisfied"
    )
    return [path for path in proc.stdout.split("\0") if path]


def text_files() -> Iterator[tuple[str, pathlib.Path]]:
    """`(relative posix path, path)` for every text file in the checkout."""
    for path in sorted(REPO.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(REPO).as_posix()
        if is_skipped(rel):
            continue
        yield rel, path


def read_text_files() -> Iterator[tuple[str, str]]:
    """`(relative posix path, text)`, skipping anything undecodable."""
    for rel, path in text_files():
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover
            continue
