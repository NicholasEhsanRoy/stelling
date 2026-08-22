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


# Every `path::test_name` citation in shipped prose. The trailing `*` is part
# of the citation, not noise: `interval.py` cites `::test_reassociation_*`, a
# FAMILY of tests, and a family citation is dangling when NO test matches the
# prefix. Both forms are resolved rather than one being skipped, because
# skipping is how a citation stops being checked without anyone deciding to.
#: The citation, allowing the IDENTIFIER to be wrapped. `SOUNDNESS.md`
#: wraps nine of its eighteen citations mid-name, always after an underscore
#: because that is where a long identifier breaks; a line-by-line scan sees
#: `…::test_a_lying_` and calls it dangling. The continuation is tried only
#: when the unwrapped name does NOT resolve, so this can never turn a live
#: citation into a different one by swallowing the next word.
_TEST_REF_WRAPPED = re.compile(
    r"(tests/[A-Za-z0-9_./]+\.py)::(test_[A-Za-z0-9_]+)"
    r"((?<=_)[ \t]*\n[ \t#*>]*[A-Za-z0-9_]+)?(\*?)"
)

#: A citation the SAME PARAGRAPH supersedes: `::<name>` with no path, naming
#: the test the cited one became. `SOUNDNESS.md` records a closed finding by
#: leaving the original sentence standing and annotating it — *"that test is
#: now `::test_the_two_pairing_gates_…`"* — and rewriting the original would
#: destroy the record this file exists to keep. A supersession only licenses
#: the stale citation when the replacement it names RESOLVES, so it cannot be
#: used to license nothing. THE RESIDUE, stated: a paragraph carrying a
#: resolving `::name` licenses every stale citation in it, not only the one
#: the annotation is about. One paragraph in the tree carries a supersession
#: today, so the residue is one paragraph wide; a second would be worth
#: keying the licence to the citation it annotates.
_SUPERSEDED_BY = re.compile(r"(?<!\w)::(test_[A-Za-z0-9_]+)")

#: Where a `path::name` citation is checked. Shipped prose and the scripts
#: CI runs — everything a reader can follow a citation from.
#:
#: **`tests/` IS DELIBERATELY OUT**, and it is not an oversight: three test
#: modules write citation-shaped SOURCE STRINGS as plants
#: (`tests/test_x.py::test_y` in `tests/test_state_guard.py`,
#: `tests/test_some_other_module.py::test_x` in `tests/test_skip_inventory.py`)
#: and those files are supposed to name tests that do not exist. So is
#: `scratchpad/`, which is tracked evidence rather than shipped prose.
#:
#: IT WAS `src/stelling` ALONE UNTIL 2026-08-22, which is 32 of the 70
#: citations in the shipped set. The one it could not see:
#: `.github/scripts/tripwire_canary.py` cited
#: `tests/test_tripwire_record.py::test_the_nightly_workflow_still_runs_the_canary`,
#: renamed away, in the paragraph explaining why `--no-sweep` is safe.
_CITATION_ROOTS = ("src/", "docs/", "design/", ".github/")
_TESTS = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS.parent


def _citation_sources():
    """`(relative path, text)` for every file a citation is checked in."""
    for path in sorted(_REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".md", ".yml", ".txt"):
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel.startswith(_CITATION_ROOTS) or (
            "/" not in rel and rel.endswith(".md")
        ):
            try:
                yield rel, path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):  # pragma: no cover
                continue


def _enclosing_paragraph(text: str, index: int) -> str:
    start = text.rfind("\n\n", 0, index)
    start = 0 if start < 0 else start + 2
    end = text.find("\n\n", index)
    return text[start:len(text) if end < 0 else end]


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

    THE SCAN IS THE SHIPPED SET, NOT `src/stelling` ALONE, from 2026-08-22.
    It read `src/stelling` only — 32 of the 70 citations a reader can follow
    — so a citation in `SOUNDNESS.md`, in `docs/`, in `design/` or in a
    script CI runs was unchecked. Driven: `.github/scripts/tripwire_canary.py`
    cited `tests/test_tripwire_record.py::test_the_nightly_workflow_still_
    runs_the_canary`, renamed away, in the paragraph that explains why
    `--no-sweep` is safe on that script. It was the ONLY dangling one in the
    widened set, and the widening is what found it.

    Two shapes had to be handled for the wider scan to mean anything, and
    both are enumerated beside their patterns: a citation WRAPPED mid-name
    (nine of `SOUNDNESS.md`'s eighteen are), and a citation the same
    paragraph SUPERSEDES by naming the test it became — which is how the
    ledger records a closed finding without rewriting the sentence it closed.
    `tests/` and `scratchpad/` stay out, and `tests/` for a reason rather
    than by omission: three test modules write citation-shaped source strings
    as PLANTS, and are supposed to name tests that do not exist.

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
    def resolves(defined, name, star):
        return (any(d.startswith(name) for d in defined) if star
                else name in defined)

    dangling, seen = [], 0
    for source, text in _citation_sources():
        for m in _TEST_REF_WRAPPED.finditer(text):
            rel, name, wrap, star = m.groups()
            lineno = text.count("\n", 0, m.start()) + 1
            seen += 1
            target = _TESTS.parent / rel
            if not target.is_file():
                dangling.append(f"{source}:{lineno}: no such file {rel}")
                continue
            defined = _defined_test_names(target.read_text(encoding="utf-8"))
            if defined is None:
                dangling.append(
                    f"{source}:{lineno}: {rel} does not parse, so it "
                    f"defines no test at all")
                continue
            if resolves(defined, name, star):
                continue
            # ... then the wrapped spelling, and only then
            if wrap and resolves(defined, name + wrap.split("\n")[1].strip(" \t#*>"), star):
                continue
            # ... and last, a supersession the same paragraph declares
            para = _enclosing_paragraph(text, m.start())
            if any(
                resolves(defined, replacement, "")
                for replacement in _SUPERSEDED_BY.findall(para)
            ):
                continue
            dangling.append(f"{source}:{lineno}: {rel}::{name}{star}")
    assert not dangling, (
        "core prose cites test(s) that do not exist:\n  "
        + "\n  ".join(dangling)
        + "\nA citation is the reader's route from a claim to the thing that "
        "holds it; a renamed test silently turns it into a dead end. Repoint "
        "it, or drop the claim it was supporting."
    )
    # ... and it is not vacuous: the citations really are there to check.
    # Measured at 70 across the shipped set on 2026-08-22, of which 32 are in
    # `src/stelling` — the whole of what this used to read.
    assert seen >= 40, (
        f"only {seen} test citation(s) found across {list(_CITATION_ROOTS)} "
        f"and the root pages; the pattern has stopped matching how they are "
        f"written, or the scan has stopped reaching them"
    )
    assert len({
        source for source, text in _citation_sources()
        if _TEST_REF_WRAPPED.search(text)
    }) >= 10, "the citations have collapsed into one or two files"
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
# described as covering "the tracked, SHIPPED tree" and covered six of the 24
# allowlisted roots — SOUNDNESS.md, docs/, README.md, ARCHITECTURE.md,
# CONTRIBUTING.md, .github/. `/design`, `/corpus`, `/tests`, `/src` and
# `/tools` all ship and were not swept, and TEN wrong own-source citations were
# sitting in them, one of them repeated three times.
#
# THE DENOMINATOR SAID 22 AND WAS NEVER 22 — the third stale figure in a
# paragraph whose subject is stale figures. Counted with this file's own
# `_shipped_roots()` at 43973af, 650e678, 53f9f84 and a61c01f alike: 23. The
# commit that introduced the sentence, and a commit body announcing a re-sweep
# of "ALL 22 ROOTS", carry the same wrong number; those are immutable. This
# comment and its twin in `tests/test_zero_dep_import_discipline.py` are the
# SHIPPED copies, so they are the ones corrected.
#
# AND THE FIGURE IS READ NOW RATHER THAN TRUSTED. It survived because the only
# thing standing over it was `assert len(roots) >= 20` — which 22, 23 and 40
# all satisfy. A guard sitting exactly where the defect was, unable to see it.
# `test_the_shipped_root_count_in_prose_matches_the_allowlist` below parses the
# number out of this comment and out of its twin, and compares both with the
# allowlist it is a count of.
#
# WHAT THIS CAN AND CANNOT DO, said plainly, because the gap is most of the
# defect. It resolves a `file.ext:N` citation and asserts N is a line the file
# HAS. That is the whole class it closes — line 1031 of `contracts.py` cited
# in a file of 1022 lines. It cannot know whether line N says what the sentence around
# it says, and eight of those ten were exactly that: off by 20, by 27, by 18,
# pointing at a blank line or at the wrong helper. Nothing cheap catches those,
# which is why the house rule is to cite a SYMBOL and why the fixes took that
# form. This is the mechanical floor under the rule, not the rule.
# KNOWN-OPEN AND MEASURED. Six citation shapes go unchecked. FOUR of them have
# no live instance in the tree — those probes are synthetic, and `:99999` is a
# line no file here has — so those four are gaps in reach, not defects on the
# page. The RANGE shape has 58 live instances, six of which resolve to an
# in-tree target, and every one of those six ends inside its target: measured,
# so the shape is unchecked and currently carries nothing wrong.
#
# THE SIXTH HAD A LIVE INSTANCE WHILE THIS NOTE SAID NONE DID, which is the
# sentence being corrected here. At 103f3b6 `.github/workflows/ci.yml` said its
# two `grep -qE` gates were at "(lines 361 and 632)" and they were at 371 and
# 642. The citation was CORRECT at 70ed1a5; ten lines added to that file on this
# branch pushed the gates down and left the sentence behind. So the argument for
# citing symbols rather than lines arrived as a live example inside the very
# note that says the sweep cannot check citations. It is fixed by naming the two
# steps, in the commit that corrected this paragraph.
#
# These are written down because "the sweep is derived from the allowlist" reads
# like completeness and is not:
#
#   tools/property_venv.sh:99999   the REGEX declines it. The extension list is
#                                  `py|md|yml|yaml|toml|cff`, and `.sh` is not
#                                  in it; `_resolve_citation` resolves the path
#                                  perfectly well when handed it directly, so
#                                  the pattern is the whole of the limit.
#   contracts.py#L99999            the REGEX declines it. It matches `path:N`
#                                  and nothing else; the GitHub `#L` spelling
#                                  is a different grammar.
#   stelling/contracts.py:99999    the regex MATCHES this one — the RESOLVER
#                                  returns None. A path with a slash is looked
#                                  up as repo-relative, and `stelling/…` is not
#                                  (the file is at `src/stelling/…`), so the
#                                  citation is silently skipped rather than
#                                  resolved and checked. This is the shape
#                                  worth knowing about: it is the one that
#                                  looks like a real path, and the SAME wrong
#                                  line written as a bare basename would be
#                                  caught, while the more specific spelling is
#                                  not. (Written without its colon here on
#                                  purpose — spelled out, this line would be a
#                                  live wrong citation and the sweep below
#                                  would flag this very comment. It was, once,
#                                  while this note was being written.)
#   "lines 361 and 632"            the REGEX declines it, and this is the shape
#                                  that had the live instance. A SELF-
#                                  REFERENTIAL citation — "line N" of the file
#                                  the sentence is in — carries no path token
#                                  at all, and the pattern needs `path` then a
#                                  colon then `N`. The path spelling would not
#                                  have caught it either: this sweep asks only
#                                  whether the file HAS line N, and both
#                                  numbers sit well inside a file of over a
#                                  thousand lines, so that citation resolves
#                                  and PASSES while pointing at a `run: |` and
#                                  a `tee`, which is what they held. Which is
#                                  the "cannot know whether line N says what
#                                  the sentence says" limit above, met in the
#                                  wild. Citing the STEP BY NAME is the fix and
#                                  is the house rule already; nothing cheap
#                                  checks it.
#   a RANGE's second number        the regex takes the START and drops the rest:
#                                  handed `contracts.py`, a colon, and
#                                  `1020-99999`, `findall` returns
#                                  `('contracts.py', '1020')`. So the END of
#                                  every range citation is unchecked, and the
#                                  test below asserts exactly that behaviour
#                                  (`preconditions.py` + colon + `213-240` ->
#                                  `('preconditions.py', '213')`) without
#                                  saying what it costs. Measured across the
#                                  266 swept files: 58 range citations, 6 with
#                                  a target that resolves in-tree, and all 6
#                                  ends are inside their target — so nothing is
#                                  wrong today and nothing would notice if it
#                                  were. Widening the regex to check the end
#                                  too is a small change, but ranges also cite
#                                  THIRD-PARTY sources here (hatchling's
#                                  `builders/config.py`, the MIME socket), and
#                                  which of those should resolve is the same
#                                  decision as the entry above; left with it.
#   an AMBIGUOUS bare basename     the regex takes it and the resolver answers
#                                  with the ROOT file. `_resolve_citation`
#                                  tries `_REPO / rel` FIRST, so a bare name
#                                  that happens to name a file at the repo root
#                                  resolves there however many other files
#                                  carry it — `README.md` resolves to the
#                                  267-line root one, and this tree has four
#                                  files of that name (the root's, `docs/`'s at
#                                  41 lines, `tests/property/`'s at 628, and
#                                  `corpus/supply/affine_holdout/`'s at 40). A
#                                  citation of the root README's line 250
#                                  written inside `docs/README.md` would be
#                                  checked against the wrong file and pass.
#                                  NOT LIVE: measured, `README.md` is the only
#                                  multi-bearer basename in the swept tree and
#                                  zero citations of it exist. The docstring on
#                                  `_resolve_citation` used to describe the
#                                  other behaviour — "accepted only when
#                                  exactly one file in the tree carries it" —
#                                  and has been corrected to describe the code.
#                                  Changing the CODE instead is the same
#                                  ambiguity decision as the two entries above.
#
# The third was first reported as a regex miss. It is not; the regex takes it
# and the resolver drops it. Same outcome, different mechanism, and the
# mechanism is what a fix would have to address — `_resolve_citation` would
# need a suffix-match fallback for slashed paths, which is a decision about
# how much ambiguity to accept and is left to the principal. The last two
# entries are that same decision wearing different clothes, which is why all
# three are recorded rather than half-fixed.
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
    than to a same-named file here. `builders/config.py` resolves to nothing
    because no such path exists; `solvers.py` resolves to
    `src/stelling/solvers.py` because exactly one file is called that.

    TWO RULES, IN THIS ORDER, and the first is the one the docstring used to
    omit. A repo-relative path is tried FIRST, so a bare basename that names a
    file at the repo root resolves there however many other files carry the
    name. Only if that misses is the "exactly one bearer" rule applied — which
    is what this used to claim was the whole of it, and it is not:
    `README.md` resolves to the root README although four swept files are
    called that. Not live (no `README.md` line citation exists in the tree, and
    it is the only multi-bearer basename), and recorded in the reach note above
    rather than changed, because which file an ambiguous basename should mean
    is the same decision left to the principal there.
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


def test_the_shipped_root_count_in_prose_matches_the_allowlist():
    """The count two shipped comments state about the allowlist, DERIVED.

    Both comments state a fraction — "six of the 23 allowlisted roots", "6
    roots out of 23" — and the denominator is a fact about `pyproject.toml`
    that nothing read. It said 22. The allowlist has held 23 root entries at
    every commit anyone has looked at: 43973af, 650e678, 53f9f84 and a61c01f,
    counted with the two parsers those files already use.

    THIS DOCSTRING SAID THOSE FRACTIONS "explain their own scope" AND NEITHER
    DOES. Both describe the OLD hand-typed sweep — the one that covered six
    roots and was replaced — as a cautionary example. The current scope of
    either sweep is derived from the allowlist and is not a fraction anywhere.
    Corrected here rather than left, because a checker that misdescribes its
    own subject is the thing this file exists to catch.

    WHAT LET IT THROUGH IS THE INTERESTING PART. Both files carried a guard
    over exactly this number — `assert len(roots) >= 20` — added in the same
    pass that wrote the wrong figure. It is satisfied by 22, by 23, and by 40,
    so it could not distinguish the count from the claim about the count. A
    bound is not a check of a figure; the figure has to be read.

    So it is read. `>= 20` stays nowhere; the number in the prose is parsed and
    compared against the parsed allowlist, which means adding a root to
    `pyproject.toml` now moves both comments or goes red, and neither comment
    can drift from the other.

    WHAT THIS DOES NOT BUY, measured rather than left to be assumed, because
    "the number in the prose is parsed" reads like more than it is. Each
    pattern is `re.findall` over the WHOLE FILE TEXT with an exactly-once
    requirement, so what it pins is "somewhere in this file, exactly once,
    these words appear with this number after them" — not "the sentence a
    reader sees says this". Three consequences, each driven at cc5ce89 against
    the two files' real text:

    * the pattern here is welded to the CURRENT LINE WRAP (`(\\d+)\\s*\\n#\\s*`
      demands the break, and a `#` at column 0 after it). Reflow the real
      comment onto one line and it stops matching — `findall` returns `[]` and
      this fails loudly, which is the safe half. Reflow it to say 22 AND add a
      decoy that does match, and `findall` returns exactly `['23']`: green
      tree, "22" on the page. Both decoy shapes work — a column-0 comment
      anywhere in the file, and a triple-quoted assertion message in a function
      that is never called.
    * the zero-dep pattern has no wrap dependency, so a plain decoy makes it
      two matches and this fails. Rewording the real sentence instead ("six
      roots out of 22", spelled out, which the literal `6` in the pattern no
      longer takes) plus one decoy gives exactly one match: green tree, "22" on
      the page again.
    * the NUMERATORS (`six`, `6`) are literals in the patterns, compared with
      nothing. That direction is safe by accident and worth saying: change a
      numerator and the pattern stops matching, so it goes red rather than
      quiet.

    Closing this means locating each sentence rather than searching for it —
    anchoring on the enclosing comment block, or moving the figure out of prose
    into a name the comment is generated from. Not done here; the shape of the
    gap is written down so the guard is not read as more than a text search.
    """
    roots = _shipped_roots()
    found = []
    for path, pattern in (
        (Path(__file__), r"covered six of the (\d+)\s*\n#\s*allowlisted roots"),
        (
            _REPO / "tests" / "test_zero_dep_import_discipline.py",
            r"ends up covering 6 roots out of (\d+)",
        ),
    ):
        text = path.read_text(encoding="utf-8")
        matches = re.findall(pattern, text)
        assert len(matches) == 1, (
            f"{path.name}: the shipped root-count sentence no longer matches "
            f"{pattern!r} exactly once (found {len(matches)}). The figure it "
            "states is the thing under test, so this must fail loudly rather "
            "than quietly check nothing."
        )
        found.append((path.name, int(matches[0])))

    wrong = [(name, n) for name, n in found if n != len(roots)]
    assert not wrong, (
        f"the sdist allowlist has {len(roots)} root entries and these shipped "
        "comments say otherwise. The denominator is a count OF THAT LIST, so "
        "it moves whenever the list does:\n  "
        + "\n  ".join(f"{name} says {n}" for name, n in wrong)
        + f"\n  pyproject.toml has {len(roots)}: {sorted(roots)}"
    )


def test_the_citation_sweep_covers_the_whole_allowlist_and_the_resolver_works():
    """The two ways the check above goes quiet without failing."""
    roots = _shipped_roots()
    swept = {p.relative_to(_REPO).parts[0] for p in _shipped_text_files()}
    # the five roots the sweep this replaces did not reach
    assert {"design", "corpus", "tests", "src", "tools"} <= swept, sorted(swept)
    assert len(_shipped_text_files()) > 200
    # …and the sweep stays INSIDE the allowlist. This was `len(roots) >= 20`,
    # a bound that could not tell 22 from 23 and sat directly over a figure
    # that was wrong by one — see
    # `test_the_shipped_root_count_in_prose_matches_the_allowlist`, which now
    # holds the count itself. What is worth asserting here is the relation:
    # every root the sweep reaches is a root the sdist ships.
    assert swept <= set(roots), sorted(swept - set(roots))

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
