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


# --- line citations, in the SHIPPED tree ------------------------------------
#
# The scope is DERIVED from the sdist allowlist rather than typed, and that is
# the correction this check embodies. A stale-figure sweep of this tree was
# described as covering "the tracked, SHIPPED tree" and covered six of the 22
# allowlisted roots — SOUNDNESS.md, docs/, README.md, ARCHITECTURE.md,
# CONTRIBUTING.md, .github/. `/design`, `/corpus`, `/tests`, `/src` and
# `/tools` all ship and were not swept, and TEN wrong own-source citations were
# sitting in them, one of them repeated three times.
#
# WHAT THIS CAN AND CANNOT DO, said plainly, because the gap is most of the
# defect. It resolves a `file.ext:N` citation and asserts N is a line the file
# HAS. That is the whole class it closes — line 1031 of `contracts.py` cited
# in a file of 1022 lines. It cannot know whether line N says what the sentence around
# it says, and eight of those ten were exactly that: off by 20, by 27, by 18,
# pointing at a blank line or at the wrong helper. Nothing cheap catches those,
# which is why the house rule is to cite a SYMBOL and why the fixes took that
# form. This is the mechanical floor under the rule, not the rule.
_LINE_CITATION = re.compile(
    r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:py|md|yml|yaml|toml|cff)):(\d+)"
)
_REPO = Path(__file__).resolve().parent.parent


def _shipped_roots():
    """The `[tool.hatch.build.targets.sdist]` allowlist, read as TEXT.

    Not with `tomllib`, which is 3.11+ while the declared floor is 3.10 — see
    `tests/test_zero_dep_import_discipline.py`, where that is the whole
    subject.
    """
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    block = re.search(
        r"\[tool\.hatch\.build\.targets\.sdist\]\s*\ninclude\s*=\s*\[(.*?)^\]",
        text, re.S | re.M,
    )
    assert block, "the sdist allowlist is not where this expects it"
    return [m.group(1).lstrip("/") for m in re.finditer(r'"([^"]+)"', block.group(1))]


def _shipped_text_files():
    out = []
    for root in _shipped_roots():
        base = _REPO / root
        if base.is_file():
            out.append(base)
        elif base.is_dir():
            out.extend(
                p for p in sorted(base.rglob("*"))
                if p.is_file()
                and p.suffix in (".py", ".md", ".yml", ".yaml", ".toml", ".cff")
                and "__pycache__" not in p.parts
            )
    return out


def _resolve_citation(rel: str):
    """The file a citation names, or None when it names nothing in this repo.

    Third-party targets (`subprocess.py`, jax's `lax.py`, hatchling's
    `builders/config.py`) are the common case and must resolve to None rather
    than to a same-named file here — so a BARE basename is accepted only when
    exactly one file in the tree carries it. `builders/config.py` resolves to
    nothing because no such path exists; `solvers.py` resolves to
    `src/stelling/solvers.py` because exactly one file is called that.
    """
    direct = _REPO / rel
    if direct.is_file():
        return direct
    if "/" in rel:
        return None
    matches = [p for p in _shipped_text_files() if p.name == rel]
    return matches[0] if len(matches) == 1 else None


def test_no_shipped_page_cites_a_line_its_own_tree_does_not_have():
    """A citation past the end of a file is a claim nothing can be.

    Break it: change any `foo.py:12` in a shipped page to `foo.py:999999`.
    Driven the other way at 650e678, before the citations were repaired:
    `corpus/supply/affine_holdout/SCOUT_CASES.md` cited line 1031 of
    `contracts.py`, and `src/stelling/contracts.py` is 1022 lines.
    """
    lengths: dict[Path, int] = {}
    past_eof = []
    checked = 0
    for path in _shipped_text_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for rel, cited in _LINE_CITATION.findall(line):
                target = _resolve_citation(rel)
                if target is None:
                    continue
                if target not in lengths:
                    lengths[target] = len(
                        target.read_text(encoding="utf-8").splitlines()
                    )
                checked += 1
                if int(cited) > lengths[target]:
                    past_eof.append(
                        f"{path.relative_to(_REPO)}:{lineno} cites {rel}:{cited}, "
                        f"and {target.relative_to(_REPO)} has "
                        f"{lengths[target]} lines"
                    )
    assert not past_eof, (
        "shipped page(s) cite a line that does not exist:\n  "
        + "\n  ".join(past_eof)
        + "\nCite the SYMBOL, not the line — a line number in a page nothing "
        "regenerates is a claim nothing checks, and a wrong one sends a reader "
        "to a line that reads plausibly."
    )
    # ...and it looked at something. A regex that stopped matching and a tree
    # with no bad citations return the same empty list.
    assert checked > 30, (
        f"only {checked} in-repo line citation(s) resolved; the pattern has "
        "stopped matching how they are written"
    )


def test_the_citation_sweep_covers_the_whole_allowlist_and_the_resolver_works():
    """The two ways the check above goes quiet without failing."""
    roots = _shipped_roots()
    assert len(roots) >= 20, roots
    swept = {p.relative_to(_REPO).parts[0] for p in _shipped_text_files()}
    # the five roots the sweep this replaces did not reach
    assert {"design", "corpus", "tests", "src", "tools"} <= swept, sorted(swept)
    assert len(_shipped_text_files()) > 200

    # the resolver: a bare basename with exactly one bearer resolves, a
    # third-party path does not, and a bare name matching nothing does not
    assert _resolve_citation("solvers.py") == _REPO / "src" / "stelling" / "solvers.py"
    assert _resolve_citation("subprocess.py") is None
    assert _resolve_citation("builders/config.py") is None
    assert _resolve_citation("does_not_exist_anywhere.py") is None
    assert _resolve_citation("pyproject.toml") == _REPO / "pyproject.toml"

    # ...and the pattern reads the shapes these pages actually use. The
    # samples are BUILT BY CONCATENATION so this file cannot cite anything
    # itself — the same device `tests/test_import_hygiene.py` uses for the
    # private-jax token, and here it is load-bearing: this file is INSIDE the
    # swept tree and is deliberately not exempted from its own check. A
    # checker that exempts itself is a checker whose own claims go unread.
    colon = ":"
    assert _LINE_CITATION.findall(f"(contracts.py{colon}1031), not by raw") == [
        ("contracts.py", "1031")
    ]
    assert _LINE_CITATION.findall(f"`preconditions.py{colon}213-240`") == [
        ("preconditions.py", "213")
    ]
    assert _LINE_CITATION.findall(f"`src/stelling/obligation.py{colon}886`") == [
        ("src/stelling/obligation.py", "886")
    ]
    assert _LINE_CITATION.findall("no citation here") == []
