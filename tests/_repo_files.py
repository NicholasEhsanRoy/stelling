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
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

REPO = pathlib.Path(__file__).resolve().parents[1]

#: What counts as text. Extensions rather than a content sniff, so the set
#: is readable and a new kind of file is a deliberate addition.
SUFFIXES = (".md", ".py", ".toml", ".cff", ".yml", ".yaml", ".txt")

#: Directories a checkout grows that are not the project's own prose.
#: `scratchpad/` is tracked evidence and is excluded from the sdist; a
#: measurement recorded there is not a claim the project ships.
SKIP = (
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "scratchpad",
    "build", "dist", ".mypy_cache", ".ruff_cache", ".egg-info",
)


def text_files() -> Iterator[tuple[str, pathlib.Path]]:
    """`(relative posix path, path)` for every text file in the checkout."""
    for path in sorted(REPO.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        parts = path.relative_to(REPO).parts
        if any(part in SKIP or part.endswith(".egg-info") for part in parts):
            continue
        yield path.relative_to(REPO).as_posix(), path


def read_text_files() -> Iterator[tuple[str, str]]:
    """`(relative posix path, text)`, skipping anything undecodable."""
    for rel, path in text_files():
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover
            continue
