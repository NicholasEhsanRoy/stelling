# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""ARCHITECTURE.md Rule 2, enforced: core prose is library-neutral.

Coupling lives in prose before it lives in code, and no ordinary test
guards prose — so this one does, structurally (L18: a duty-enforced
prose rule would not survive). Any line of ``src/stelling`` naming a
specific library must be a marked provenance citation: the SAME line
must carry ``census`` (or cite a ``design/`` record). Outward references
("library X's operator", "the X attachment's shape") have no marker and
fail here, named file:line.

The pre-commit hook ``library-identifier-hygiene`` in
.pre-commit-config.yaml enforces the same rule as a grep; the two are
deliberately logically identical — a divergence between them is a bug.
``tests/`` and ``corpus/`` are out of scope by design: census-binding
tests and campaign exhibits name their contacts as provenance by nature.
"""

from __future__ import annotations

import re
from pathlib import Path

# Extensible by design: add a library's name here when it becomes a
# census contact (or a socket host) — the lint then forces every mention
# in core to be a marked provenance citation, never an outward reference.
# Keep this list identical to the grep in .pre-commit-config.yaml's
# library-identifier-hygiene hook.
BANNED_IDENTIFIERS = (
    "mime",
    "maddening",
    "lineax",
    "equinox",
    "diffrax",
    "blackjax",
    "optimistix",
    "optax",
)

_BANNED = re.compile(
    r"\b(" + "|".join(BANNED_IDENTIFIERS) + r")\b", re.IGNORECASE
)
_PROVENANCE = re.compile(r"census|design/", re.IGNORECASE)

_SRC = Path(__file__).resolve().parent.parent / "src" / "stelling"


def test_core_lines_naming_libraries_carry_provenance_markers():
    assert _SRC.is_dir(), f"source tree not found at {_SRC}"
    violations = []
    for path in sorted(_SRC.rglob("*.py")):  # recursive, like the hook's -r
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _BANNED.search(line) and not _PROVENANCE.search(line):
                violations.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not violations, (
        "library-naming line(s) in src/stelling without a provenance "
        "marker (ARCHITECTURE.md Rule 2: the same line must carry "
        "'census' or cite design/):\n  " + "\n  ".join(violations)
    )


# Every `path::test_name` citation in core prose. The trailing `*` is part of
# the citation, not noise: `interval.py` cites `::test_reassociation_*`, a
# FAMILY of tests, and a family citation is dangling when NO test matches the
# prefix. Both forms are resolved rather than one being skipped, because
# skipping is how a citation stops being checked without anyone deciding to.
_TEST_REF = re.compile(r"(tests/[A-Za-z0-9_./]+\.py)::(test_[A-Za-z0-9_]+)(\*?)")
_TESTS = Path(__file__).resolve().parent


def _defined_test_names(text):
    """Every function a test file DEFINES WHERE PYTEST WOULD COLLECT IT, parsed
    rather than grepped — or None when the file does not parse at all.

    **WHAT THIS SEMANTIC IS, EXACTLY, because it is weaker than the sentence a
    reader takes from a citation.** The claim a `path::name` citation makes is
    "this guard is pinned by a test that RUNS". What is checked is "a
    `FunctionDef` of that name exists at a place pytest collects from". Those
    are not the same, and the gap is enumerable:

        inside `if False:`                    reported defined, never runs
        in a never-taken `except`/`else`      reported defined, never runs
        defined and then `del`'d              reported defined, never runs
        rebound to a non-test                 reported defined, never runs
        under `if TYPE_CHECKING:`             reported defined, never runs

    Five shapes, each stated rather than left to be found. They share a
    property that makes them a smaller risk than the two below: every one of
    them is a deliberate act on the cited test itself, whereas the two this
    pass CLOSED are the ordinary ways a test stops existing.

    The two closed here were a `tree.body` scan's blind spots turned into a
    checker's blind spots by `ast.walk`:

    * NESTED IN ANOTHER FUNCTION. A helper's inner `def test_x` is not
      collected by pytest and never was, and `ast.walk` reported it defined;
    * INSIDE A NON-`Test*` CLASS. `class T: def test_gone(self)` was an
      ACCEPTED row of this test's own anti-vacuity block, asserted `True`, and
      `pyproject.toml` sets no `python_classes`, so pytest's default
      `Test*` prefix means that class is not collected either. The row is
      corrected: a method counts only when its class would be collected.

    So: module level, or a method of a `class Test…`. `ast.walk`'s reach is
    kept for finding the definition and thrown away for judging it."""
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    out = set()

    def add(body, in_test_class):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
                # a `def` inside a `def` is not collected, so its body is not
                # walked for collectable names either
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    add(node.body, True)
            elif hasattr(node, "body"):
                # `if`/`try`/`with`/`for` at module scope: pytest collects
                # whatever the module ends up defining, and this checker
                # cannot know which branch runs — see the docstring
                add(node.body, in_test_class)
                add(getattr(node, "orelse", []), in_test_class)
                add(getattr(node, "finalbody", []), in_test_class)
                for handler in getattr(node, "handlers", []):
                    add(handler.body, in_test_class)

    add(tree.body, False)
    return out


def test_every_test_cited_in_core_prose_still_exists():
    """A COMMENT THAT NAMES A TEST IS A CLAIM ABOUT WHAT HOLDS IT SHUT, and a
    renamed test turns it into a claim about nothing.

    `src/stelling` cites tests by `path::name` in nineteen places. Those
    citations are how a reader checks that a guard is pinned rather than
    asserted, and nothing kept them true: commit `218f969` renamed
    `test_a_ONE_SHOT_records_cannot_silence_the_bar` to
    `test_a_ONE_SHOT_records_behaves_EXACTLY_LIKE_THE_TUPLE_it_yields` and left
    `solvers.py`'s citation of the old name behind. It was the only dangling
    one, and it was found by reading rather than by anything running.

    Resolved by SOURCE rather than by collection: a `FunctionDef` of that name
    in the cited file. A collection-based check would go quiet in the zero-dep
    column, where the jax-only files never collect — and the citations are in
    prose that ships either way.

    **PARSED, NOT GREPPED, AND THE DIFFERENCE WAS MEASURED.** `f"def {name}("
    in body` is a raw substring test over file TEXT, so anything that spells
    those characters satisfies it — including the two most ordinary ways a test
    stops existing. Driven at `faefc48`:

        the cited family renamed away entirely     1 failed  (not vacuous)
        the family gone, one `# def test_…(`       1 PASSED
        the exact citation's `def` commented out   1 PASSED
        the `def` gone, a string-literal mention   1 PASSED

    Commenting a test out is how a test most often stops existing, and it
    defeated this check completely. `ast.parse` is equally independent of
    collection — it reads the file and imports nothing — so the reason this
    docstring gives for avoiding collection costs nothing here.
    """
    dangling = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rel, name, star in _TEST_REF.findall(line):
                target = _TESTS.parent / rel
                if not target.is_file():
                    dangling.append(f"{path.name}:{lineno}: no such file {rel}")
                    continue
                defined = _defined_test_names(
                    target.read_text(encoding="utf-8"))
                if defined is None:
                    dangling.append(
                        f"{path.name}:{lineno}: {rel} does not parse, so it "
                        f"defines no test at all")
                    continue
                found = (any(d.startswith(name) for d in defined) if star
                         else name in defined)
                if not found:
                    dangling.append(
                        f"{path.name}:{lineno}: {rel}::{name}{star}")
    assert not dangling, (
        "core prose cites test(s) that do not exist:\n  "
        + "\n  ".join(dangling)
        + "\nA citation is the reader's route from a claim to the thing that "
        "holds it; a renamed test silently turns it into a dead end. Repoint "
        "it, or drop the claim it was supporting."
    )
    # ... and it is not vacuous: the citations really are there to check
    cited = {
        ref for path in _SRC.rglob("*.py")
        for ref in _TEST_REF.findall(path.read_text(encoding="utf-8"))
    }
    assert len(cited) >= 5, (
        f"only {len(cited)} test citation(s) found in src/stelling; the "
        f"pattern has stopped matching how they are written"
    )
    # ... and the RESOLVER is not satisfied by text that only LOOKS like a
    # definition. These are the three rows that were green at `faefc48`,
    # written out rather than described, plus the honest shape it must accept
    # and the two forms `ast.walk` reaches that a `tree.body` scan would not.
    for label, source, name, present in (
        ("commented out", "# def test_gone(x):\n#     pass\n", "test_gone",
         False),
        ("string literal", '_X = "def test_gone(x)"\n', "test_gone", False),
        ("in a docstring", '"""cites def test_gone( here"""\n', "test_gone",
         False),
        ("really defined", "def test_gone(x):\n    pass\n", "test_gone", True),
        # CORRECTED. This row asserted True and was one of the branch's own
        # accepted shapes. `pyproject.toml` sets no `python_classes`, so
        # pytest's default `Test*` prefix means `class T` is not collected and
        # the cited test does not run.
        ("inside a non-Test class",
         "class T:\n    def test_gone(self):\n        pass\n", "test_gone",
         False),
        ("inside a Test class",
         "class TestX:\n    def test_gone(self):\n        pass\n", "test_gone",
         True),
        # ... and a `def` inside a `def` is not collected either
        ("nested in a function",
         "def helper():\n    def test_gone(x):\n        pass\n", "test_gone",
         False),
        ("decorated", "@mark\ndef test_gone(x):\n    pass\n", "test_gone",
         True),
        # ... and the shapes this resolver STILL answers "defined" for, listed
        # here so the gap is a measured row rather than a sentence. Each is a
        # deliberate act on the cited test itself; see the docstring.
        ("under `if False:`", "if False:\n    def test_gone(x):\n        pass\n",
         "test_gone", True),
        ("under TYPE_CHECKING",
         "if TYPE_CHECKING:\n    def test_gone(x):\n        pass\n",
         "test_gone", True),
        ("defined then deleted",
         "def test_gone(x):\n    pass\ndel test_gone\n", "test_gone", True),
    ):
        defined = _defined_test_names(source)
        assert defined is not None and (name in defined) is present, (
            f"the citation resolver disagrees with the {label!r} source about "
            f"whether {name!r} is defined. Every 'False' row here was MEASURED "
            f"green against the substring test this replaced — commenting a "
            f"test out is the most common way a test stops existing, and it "
            f"satisfied `f\"def {{name}}(\" in body`"
        )
    assert _defined_test_names("def broken(:\n") is None, (
        "a file that does not parse is being reported as defining tests"
    )
