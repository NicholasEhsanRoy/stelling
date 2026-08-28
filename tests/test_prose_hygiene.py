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
``tests/`` and ``corpus/`` are out of scope FOR THAT RULE, and the scope
belongs to the rule rather than to this file: census-binding tests and
campaign exhibits name their contacts as provenance by nature, so Rule 2
has nothing to say about their prose. Every other check here makes its own
scope decision, and they do not all come out the same way — the block above
:func:`_prose_only` states one per check with its argument. (This sentence
read "out of scope by design" with no check named, which is how one
check's scope was read as the file's for as long as `tests/` was outside
the citation scan.)
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

# The runner's own conftest, for the one thing it knows and nothing else
# here does: which directories under `tests/` pytest REFUSES to recurse
# into. `_resolvable_test_names` decides whether a citation in shipped prose
# points at a live test, and a file no session can ever collect must not be
# allowed to answer that question. See its docstring for the measured
# false GREEN.
from conftest import _in_a_pruned_directory, collectable_test_files
# and the walker module's `git ls-files`, the one *"what does this
# repository have"* — see `_shipped_text_files`.
from _repo_files import tracked_paths

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
#: wraps EIGHT of its eighteen citations mid-name, always after an underscore
#: because that is where a long identifier breaks; a line-by-line scan sees
#: `…::test_a_lying_` and calls it dangling. The continuation is tried only
#: when the unwrapped name does NOT resolve, so this can never turn a live
#: citation into a different one by swallowing the next word.
#:
#: EIGHT AND NOT NINE, RE-DERIVED. This read *"nine of its eighteen"*, which
#: is the count of everything the plain spelling does not resolve. Staged by
#: how each one actually resolves, `SOUNDNESS.md`'s eighteen are **9 plain /
#: 8 mid-name wrap / 1 supersession / 0 dangling**: the ninth non-plain one
#: is `SOUNDNESS.md`'s citation, in `tests/test_verified_bar.py`, of
#: `test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`,
#: which is not wrapped THERE and resolves through the annotation beside it.
#: (Written here as a bare name rather than as the `path::name` citation
#: `SOUNDNESS.md` writes, because that test no longer exists under that name
#: and this file is now INSIDE its own scan: the bare form is the one
#: :data:`_NAMES_DECLARED_ABSENT` can carry a reason for, and the path form
#: has no such table.) A numeral that counts two mechanisms as one is how
#: a residue gets bounded at the wrong width — which is what happened to
#: :data:`_SUPERSEDED_BY`'s below.
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
#: used to license nothing.
#:
#: **THE LICENCE IS KEYED TO THE CITATION IT ANNOTATES, AND ITS OWN TRIGGER
#: IS WHY.** This comment used to say the residue was one PARAGRAPH wide —
#: any resolving `::name` in a paragraph licensed every stale citation in it
#: — on the ground that *"one paragraph in the tree carries a supersession
#: today … a second would be worth keying the licence to the citation it
#: annotates."* There are two. Both are in `SOUNDNESS.md`, and staging every
#: citation in the shipped set finds exactly them: the `So the correct
#: statement is not "scoping cost a backstop"` paragraph, whose annotation is
#: in use, and the `FOUR arguments, not one` paragraph, whose bare `::name`
#: is not an annotation at all but an ABBREVIATED second citation of a test
#: in the file already named — a licence nothing needs, covering a paragraph
#: 1382 characters wide at `24a77cb`. Driven there: a deleted citation of
#: `tests/test_propagation_identity.py` planted at the head of that paragraph
#: gave **5 passed**; the identical plant one paragraph later gave **1
#: failed**.
#:
#: So an annotation licenses the citation it FOLLOWS, with no other
#: `path::name` citation in between — which is the shape both real ones have
#: (*"…`…test_verified_bar.py::test_x`. **[CLOSED …; that test is now
#: `::test_y`.]**"*) and is what "the citation it annotates" means when the
#: only thing a checker can read is position. (The path in that illustration
#: is elided at its head on purpose: `test_x` is a placeholder, and spelled
#: in full this line would be a live dangling citation in a file the scan
#: now reads. Same device, same reason, as the `stelling/contracts.py`
#: entry in the reach note above `_LINE_CITATION`.)
#:
#: **THE RESIDUE THAT REMAINS, AT THE SUPERSESSION THAT IS ACTUALLY
#: CONSULTED.** This sentence read *"307 characters of that same paragraph,
#: down from 1382"* until 2026-08-22, and both figures reproduce — of the
#: `FOUR arguments, not one` paragraph, the one whose licence is NEVER
#: consulted, and 307 is not the residue there either but the gap from its
#: citation to its bare `::name`. Measured on this tree, at the live
#: supersession — `SOUNDNESS.md`'s `tests/test_verified_bar.py` citation of
#: `test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`, the
#: one annotated `::test_the_two_pairing_gates_bind_the_ESCALATION_AND_
#: the_propagation`:
#:
#:     paragraph                                    5945 chars
#:     forward window — the span still licensed     4507 chars
#:     citation to the annotation that licenses it   153 chars
#:
#: **THAT SUPERSESSION WAS NAMED BY LINE NUMBER, `SOUNDNESS.md:1080`, AND
#: THE NUMBER WAS NOT RIGHT AT ANY REVISION ON THIS BRANCH.** Driven: the
#: citation is at **1157** at `5ad906f` and at **1340** at `becad2b`, so it
#: was already wrong before the 0.2.0 D5 accuracy pass began and that pass's
#: own edits above it then pushed it 183 lines further out — and this commit
#: moves it again, which is the point. All three figures beside it re-derive
#: exactly at both revisions; only the handle was wrong. Nothing could catch
#: it either: `test_no_shipped_page_cites_a_line_its_own_tree_does_not_have`
#: checks only that a cited line is within EOF, and 1080 is. That test's own
#: failure message is the rule followed here — *"Cite the SYMBOL, not the
#: line."*
#:
#: So the licence went from 5945 characters to 4507 there, not from 1382 to
#: 307; the paragraph-wide version licensed the 1438 characters BEFORE the
#: citation as well, and those are what the keying took away. (The
#: `FOUR arguments` paragraph is 1382 with a 535-character window and its
#: `::name` 307 in — a licence nothing needs, over a span nothing reads.)
#: The residue is real and stated: an annotation and the sentence it corrects
#: may still be thousands of characters apart, and the licence still requires
#: only that the replacement RESOLVE IN THE CITED FILE, not that it be
#: related to the name it replaces.
_SUPERSEDED_BY = re.compile(r"(?<!\w)::(test_[A-Za-z0-9_]+)")

#: Where a citation is checked. Shipped prose and the scripts CI runs —
#: everything a reader can follow a citation from — and, since 2026-08-28,
#: `tests/`.
#:
#: **THE EXCLUSION THAT USED TO STAND HERE WAS ARGUED ABOUT CITATIONS AND
#: SPELLED AS A DIRECTORY.** This comment read, in full:
#:
#:     **`tests/` IS DELIBERATELY OUT**, and it is not an oversight: three
#:     test modules write citation-shaped SOURCE STRINGS as plants
#:     (a `path`-then-`::`-then-`name` string in `tests/test_state_guard.py`,
#:     another in `tests/test_skip_inventory.py`) and those files are
#:     supposed to name tests that do not exist.
#:
#: Every word of that is TRUE, and it is still the whole reason the rule
#: below has to tell a plant from a claim — which is why it is quoted here
#: rather than deleted. (The two examples it gave were spelled out as live
#: citations of files this tree does not have; they are described here
#: instead, for the reason the reach note above `_LINE_CITATION` gives about
#: its own third entry — spelled out, they would be dangling citations in
#: the scan's own scope, and the scan now reaches this file.)
#:
#: **WHAT IT DOES NOT LICENSE IS THE RULE IT PRODUCED.** A tuple of path
#: prefixes excludes a DIRECTORY. The argument was about a KIND OF STRING.
#: So the exclusion also took out every SENTENCE in `tests/` that makes a
#: claim about this tree — **181 of the 539 files `git ls-files` names, and
#: 181 of the sdist's 379 members**, both re-derived at `9b5b496` on
#: 2026-08-28: `git ls-files | wc -l` gives 539 and its `tests/` prefix 181;
#: the same list under :func:`_shipped_roots`'s allowlist gives 378, and the
#: 379th member is the `PKG-INFO` the backend writes, which
#: `tests/test_sdist_contents.py` records as the one member in neither the
#: allowlist nor WITHHELD. (Not `_shipped_text_files`, which is 363 here
#: because it keeps only text suffixes — a different question.)
#:
#: The rule to apply generally: put the reason and the rule side by side and
#: see whether they have the same SHAPE. These did not.
#:
#: That gap is not hypothetical: it is how a false present-tense claim
#: shipped in 0.2.0. `tests/test_changelog_names_the_version.py`'s docstring
#: illustrated its own subject in the present tense, the version bump on
#: another branch moved the state it described, and nothing in the tree
#: could see it, because this tuple scoped `tests/` out of the only checker
#: that reads a citation.
#:
#: **SO THE EXCLUSION IS CITATION-SHAPED NOW**, and it lives in
#: :func:`_prose_only` rather than here: in a `.py` file a citation counts
#: when it is in a COMMENT or a DOCSTRING and does not count when it is in
#: a VALUE. The plants are values — test data handed to a function,
#: elements of a fixture tuple, `monkeypatch.setitem` keys — and Python
#: says so structurally. `scratchpad/` stays out, and by an argument about
#: what it IS rather than where it sits: it is tracked EVIDENCE, a record of
#: what a session did, not prose this project ships or maintains.
#:
#: IT WAS `src/stelling` ALONE UNTIL 2026-08-22, which is 32 of the 70
#: citations in the shipped set as it then stood. The one it could not see:
#: `.github/scripts/tripwire_canary.py` cited
#: …test_tripwire_record.py::test_the_nightly_workflow_still_runs_the_canary,
#: renamed away, in the paragraph explaining why `--no-sweep` is safe. (That
#: name is written here without its leading `tests/` for the same reason the
#: two plant examples above are described rather than spelled: the sentence's
#: subject is that the name is GONE, and the scan now reads this file.)
_CITATION_ROOTS = ("src/", "docs/", "design/", ".github/", "tests/")

#: The suffixes a citation is looked for in. Named rather than spelled twice:
#: `test_the_citation_sweep_covers_the_whole_allowlist_and_the_resolver_works`
#: derives the `tests/` half of the scan from this and from `git ls-files`,
#: and two copies of a literal tuple is how that check would go quiet.
_CITATION_SUFFIXES = (".py", ".md", ".yml", ".txt")

_TESTS = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS.parent

# --------------------------------------------------------------------------
# WHICH CHECK IN THIS MODULE SEES `tests/`, AND THE ARGUMENT FOR EACH — one
# decision per check, made by asking what the check is ABOUT rather than
# which directory it happens to have been pointed at. That is the same cut
# the exclusion above failed to make.
#
# `test_core_lines_naming_libraries_carry_provenance_markers`   NOT WIDENED.
#   Its subject is ARCHITECTURE.md Rule 2, which governs the LIBRARY-
#   NEUTRALITY OF THE CORE. A census-binding test names its contact as
#   provenance by nature — naming the library is what the test is for — so
#   the rule has nothing to say about that prose. This is a scope the RULE
#   sets, not a directory the check declines to walk, and the module
#   docstring has said so since before this pass.
#
# `test_every_test_cited_in_core_prose_still_exists`            WIDENED.
# `test_every_bare_test_name_in_shipped_prose_resolves`         WIDENED.
#   Both read `_citation_sources`, both ask "does this citation name a live
#   test", and that question is exactly as meaningful about a sentence in
#   `tests/` as about one in `docs/`. The plants that motivated the old
#   exclusion are values, and `_prose_only` drops values.
#
# `test_no_shipped_page_cites_a_line_its_own_tree_does_not_have`
#   ALREADY THERE, AND DELIBERATELY UNMASKED. It sweeps
#   `_shipped_text_files`, derived from the sdist allowlist, which has
#   carried `/tests` since that scope was fixed —
#   `test_the_citation_sweep_covers_the_whole_allowlist_and_the_resolver_works`
#   asserts it. It is NOT given the prose mask, and the asymmetry is
#   argued rather than inherited: a `file.py:N` token in a value literal is
#   still a claim that this tree has that line, checkable by the same means
#   as one in a comment, and no plant of that shape exists — this module's
#   own line-citation probes are built around a `colon` variable precisely
#   so they are not live citations. Masking there would be a pure
#   narrowing with nothing bought.
#
# `test_the_shipped_root_count_in_prose_matches_the_allowlist`  NO SCOPE.
#   It reads two files it names by path and compares a parsed figure with a
#   parsed allowlist. There is no sweep to widen.
#
# `test_the_declared_absences_are_still_absent`                 NO SCOPE.
#   It reads `_NAMES_DECLARED_ABSENT` against `_resolvable_test_names`,
#   which has always read all of `tests/`. Widening the citation scan grew
#   the table it guards; it did not change what it looks at.
# --------------------------------------------------------------------------


def _prose_only(text: str) -> str:
    """A `.py` source with everything that is not PROSE blanked out.

    Same length, same line numbers, same paragraph structure: a masked
    character becomes a space and a newline stays a newline, so every offset
    a caller computes is still an offset into the real file.

    **WHAT COUNTS AS PROSE, AND WHY IT IS DECIDED STRUCTURALLY.** Two things
    in a `.py` file hold text that is not a value, and Python distinguishes
    both without anybody having to name a file:

    * a COMMENT, which never reaches the AST at all and is read from the
      token stream;
    * a BARE STRING STATEMENT — an `ast.Expr` whose value is a string
      constant — which covers every docstring, module, class and function
      alike, a docstring being only ever the first such statement in a body.

    Everything else is masked, and that is an ALLOW-LIST of two shapes
    rather than a deny-list of the ways a string can be data. The same rule,
    from the same reading of the same grammar, is what
    `tests/test_state_guard.py`'s `state_guard_references` uses to keep a
    guard from firing on its own explanation; it is written here in the
    opposite direction because that scan wants the values and this one wants
    the remarks.

    **WHAT IT COST TO ADOPT, MEASURED BEFORE IT WAS ADOPTED AND NOT AFTER.**
    Applying this uniformly NARROWS the scan in `src/`, `docs/` and
    `.github/` as well as widening it into `tests/`, so it is a coverage
    LOSS wherever a real citation lives in a value literal. Every citation
    the scan checked at `9b5b496` (2026-08-28), the commit before `tests/`
    was added, staged by what it structurally IS — both forms together,
    `path::name` and bare:

        in a non-`.py` file (all prose, rule does not apply)   238
        in a `.py` COMMENT                                      29
        in a `.py` DOCSTRING                                    44
        in a `.py` VALUE literal                                 0
        in a `.py` and in none of those                          0
                                                               ---
                                                               311

    **The value bucket is EMPTY, so the uniform rule loses nothing**, and
    that is a measurement rather than the expectation it happened to
    confirm — the defect being repaired here is an exclusion adopted on an
    argument nobody measured against the rule it produced.

    **AND IT IS LOAD-BEARING, DRIVEN RATHER THAN ARGUED.** Turning this
    function off — `_citation_sources` yielding the raw text for a `.py`
    file — and running the widened scan on this tree (2026-08-28) gives
    **27** dangling `path::name` citations, in three files and no others:
    14 in `tests/test_state_guard.py`, 2 in `tests/test_skip_inventory.py`
    and 11 in this module's own probes. Those are exactly the plants the
    old exclusion was written to protect, and they are exactly what a
    value is. With the mask on, the same scan finds none of them.

    **WHAT THIS CANNOT SEE**, which is the half a mask is always tempted not
    to state:

    * **a citation inside a value literal that really IS a claim.** A nodeid
      sitting in a `Control(...)` tuple is a claim about this tree whose
      staleness matters, and nothing structural distinguishes it from a
      plant. Zero of the 311 above were of this shape;
      `tests/property/positive_controls.py` is where the shape does live,
      and what checks those nodeids is the harness that RUNS them, not this
      scan. (Written without a sample literal: spelled out here it would
      itself be a dangling citation, in a file this scan now reads.)
    * **a citation in an f-string.** An `ast.JoinedStr` is a value and is
      masked whole, so a citation assembled in a failure message is invisible
      — including the ones this very module writes into its own assertions.
    * **a citation assembled from pieces**, in prose or out of it:
      `"tests/" + name`, a `%`-format, a `.join`, or the deliberate device
      this file already uses to keep its own probes from being live
      citations (`test_the_citation_sweep_covers_the_whole_allowlist_and_the_
      resolver_works` builds its samples around a `colon` variable). A scan
      that reads text cannot follow any of them.
    * **a `.py` file that does not PARSE.** The mask degrades to the whole
      text, so every string in it is read as prose. That is the loud
      direction on purpose: a mask that silently returned "no prose here"
      would report absence of evidence as evidence of absence, which is the
      failure this module exists to refuse.
    """
    try:
        keep = [
            (tok.start, tok.end)
            for tok in tokenize.generate_tokens(io.StringIO(text).readline)
            if tok.type == tokenize.COMMENT
        ]
        tree = ast.parse(text)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return text
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            v = node.value
            keep.append(
                ((v.lineno, v.col_offset), (v.end_lineno, v.end_col_offset))
            )
    starts, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    out = ["\n" if ch == "\n" else " " for ch in text]
    for (sl, sc), (el, ec) in keep:
        lo, hi = starts[sl - 1] + sc, starts[el - 1] + ec
        out[lo:hi] = text[lo:hi]
    return "".join(out)


def _citation_sources():
    """`(relative path, text)` for every file a citation is checked in.

    **A `.py` FILE ARRIVES MASKED**, its values blanked and its comments and
    docstrings intact — :func:`_prose_only`, which is where the exclusion
    that used to be spelled as `_CITATION_ROOTS` minus `tests/` now lives.
    A `.md`, `.yml` or `.txt` file has no such grammar and arrives whole:
    all of it is prose, and nothing about those changed.

    **TRACKED FILES ONLY — 0.2.0 D14, the same repair as
    :func:`_shipped_text_files`.** The walk is repo-wide and then narrowed by
    prefix, so two things a checkout GROWS were reaching it: a `docs/_build/`
    under the `docs/` prefix, and — because the last clause takes every
    root-level `.md` — `stelling_0_1_0_release_checklist.md`, the untracked
    internal document `WITHHELD` exists to record. A local working file's
    citations were being held to the standard for this project's SHIPPED
    prose, which is what the gates over this function are about.

    False RED only, unlike `_shipped_text_files`: every consumer here reports
    what it finds, so a wider source set can only say MORE. It is fixed
    anyway rather than recorded, because it is one line of the same
    instrument in the same file, and a known instance of the class left
    beside a repaired one is a trap for the next reader.

    Degrades to the walk where git cannot answer, for the reason given at
    :func:`_shipped_text_files`.
    """
    tracked = tracked_paths(_REPO_ROOT)
    index = None if tracked is None else set(tracked)
    for path in sorted(_REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in _CITATION_SUFFIXES:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if index is not None and rel not in index:
            continue
        if rel.startswith(_CITATION_ROOTS) or (
            "/" not in rel and rel.endswith(".md")
        ):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):  # pragma: no cover
                continue
            yield rel, (_prose_only(text) if rel.endswith(".py") else text)


#: A blank line, WHITESPACE INCLUDED. This was a literal ``"\n\n"`` and had
#: to stop being one when `.py` sources started arriving here MASKED — see
#: :func:`_prose_only`. Masking replaces a non-prose character with a space
#: and keeps the newline, so a line of code between two comment blocks
#: becomes a line of spaces rather than an empty line, and a literal
#: ``"\n\n"`` would have run the two blocks together into one paragraph and
#: widened every supersession licence in the file to match.
#:
#: **IT IS A STRICT NARROWING, WHICH IS THE DIRECTION THAT COSTS NOTHING.**
#: `[ \t]*` may be empty, so every boundary the literal found is still a
#: boundary and there are more of them: the paragraph this returns is always
#: a SUBRANGE of the one the literal returned, and a shorter paragraph is a
#: shorter licence. Measured over the roots this scanned BEFORE `tests/`
#: was added, at `9b5b496` on 2026-08-28: of the 311 citations there, **8
#: get a different paragraph and 0 get a different verdict** — the eight are
#: in `_optional.py`, `falsify.py` (four), `interval.py` and `propagate.py`
#: (two), and none of them was relying on a supersession licence, so
#: narrowing the window changed nothing they resolved by. THIS COMMENT SAID
#: THE TWO SPELLINGS AGREED ON ALL 311, WHICH IS FALSE AND WAS WRITTEN
#: BEFORE IT WAS RUN; the eight are what running it said.
_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def _enclosing_paragraph(text: str, index: int) -> tuple[int, str]:
    """The paragraph around ``index``, and where in ``text`` it starts.

    The offset is returned because the supersession licence is keyed to a
    POSITION inside the paragraph and not to the paragraph — see
    :data:`_SUPERSEDED_BY`.

    Bounded by :data:`_BLANK_LINE` rather than by a literal ``"\n\n"``,
    because a masked `.py` source separates two comment blocks with a line
    of SPACES and not with an empty one.
    """
    start = 0
    for m in _BLANK_LINE.finditer(text, 0, index):
        start = m.end()
    end = _BLANK_LINE.search(text, index)
    return start, text[start:end.start() if end else len(text)]


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
    cited …test_tripwire_record.py::test_the_nightly_workflow_still_runs_the_
    canary, renamed away, in the paragraph that explains why `--no-sweep` is
    safe on that script. It was the ONLY dangling one in the widened set, and
    the widening is what found it. (Quoted without its leading `tests/`: the
    sentence's subject is that the name is GONE, and this file is now inside
    the scan — see the device note above `_CITATION_ROOTS`.)

    Two shapes had to be handled for the wider scan to mean anything, and
    both are enumerated beside their patterns: a citation WRAPPED mid-name,
    and a citation an ANNOTATION BESIDE IT supersedes by naming the test it
    became — which is how the ledger records a closed finding without
    rewriting the sentence it closed. Staged by mechanism, `SOUNDNESS.md`'s
    eighteen citations are **9 plain / 8 mid-name wrap / 1 supersession / 0
    dangling**; this said *"nine of eighteen"* wrap, which counted the
    supersession as a wrap.

    **`tests/` IS IN, FROM 2026-08-28.** This paragraph read *"`tests/` and
    `scratchpad/` stay out, and `tests/` for a reason rather than by
    omission: three test modules write citation-shaped source strings as
    PLANTS, and are supposed to name tests that do not exist."* The reason
    was sound and the rule was not the same shape as it — see the record
    above :data:`_CITATION_ROOTS`. The plants are VALUES, values are
    dropped by :func:`_prose_only`, and the sentences around them are
    read. `scratchpad/` still stays out, as tracked evidence rather than
    maintained prose.

    WHAT THE WIDENING FOUND, run blind against `9b5b496` on 2026-08-28:
    **eight** dangling `path::name` citations, every one of them in
    `tests/`, against **zero** in the roots the scan already had. **Six of
    the eight were in THIS FILE** — its illustrations of the citation form
    and its record of the renamed canary citation, written by a checker
    that had never had to hold itself to its own rule. Of the other two,
    one followed a test renamed at `321209d`
    (`tests/test_aval_lie_both_faces.py`) and one was a template nodeid in
    `tests/property/README.md` spelled as a real one.

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
            # ... and last, a supersession ANNOTATING THIS CITATION: one the
            # same paragraph declares AFTER it, with no other `path::name`
            # citation in between. A paragraph-wide licence was the residue
            # `_SUPERSEDED_BY`'s comment bounded at one paragraph and the
            # tree now has two, one of whose bare `::name`s annotates
            # nothing.
            para_start, para = _enclosing_paragraph(text, m.start())
            after = m.end() - para_start
            following = _TEST_REF_WRAPPED.search(para, after)
            window = para[after:following.start() if following else len(para)]
            if any(
                resolves(defined, replacement, "")
                for replacement in _SUPERSEDED_BY.findall(window)
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
    # Measured on this tree on 2026-08-28, with `tests/` inside the scan:
    # 112 `path::name` citations across 33 files, of which 97 across 21
    # files are outside `tests/`.
    #
    # BOTH FLOORS BELOW ARE ANTI-VACUITY AND NOTHING MORE, which is worth
    # saying because a floor beside a widening reads like a check of the
    # widening. `tests/` contributes 15 of the 112 here — inside ordinary
    # churn — so no floor can tell "the scan lost `tests/`" from "somebody
    # deleted a few citations". That question is decided EXACTLY instead,
    # by the derived reach assertion in
    # `test_the_citation_sweep_covers_the_whole_allowlist_and_the_resolver_works`.
    assert seen >= 90, (
        f"only {seen} test citation(s) found across {list(_CITATION_ROOTS)} "
        f"and the root pages; the pattern has stopped matching how they are "
        f"written, or the scan has stopped reaching them"
    )
    assert len({
        source for source, text in _citation_sources()
        if _TEST_REF_WRAPPED.search(text)
    }) >= 25, "the citations have collapsed into one or two files"
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


#: The suffixes this module treats as shipped prose. Narrower than
#: `tests/_repo_files.SUFFIXES` (no `.txt`), because what is swept here is
#: the SHIPPED set and the two answer different questions.
_TEXT_SUFFIXES = (".py", ".md", ".yml", ".yaml", ".toml", ".cff")


def _shipped_text_files():
    """Every text file under an allowlisted sdist root.

    **WHAT THE REPOSITORY SHIPS, NOT WHAT THE DIRECTORY HOLDS — 0.2.0 D14.**
    This was a bare `rglob` under each allowlisted root, subtracting
    `__pycache__` and nothing else. `/docs` is allowlisted whole, so a
    `docs/_build/` — which `.gitignore` names, which hatchling therefore does
    NOT pack, and which any Sphinx run creates — entered the shipped set. So
    did a `tests/.hypothesis/`, a vendored checkout, or any generated tree
    under one of the eight allowlisted directories.

    **AND THAT WIDENING GOES BOTH WAYS, WHICH IS WHY IT IS FIXED RATHER THAN
    RECORDED.** It is the only one of the six sites this batch touched that
    is false-RED and false-GREEN at once:

    * loud — `test_no_shipped_page_cites_a_line_its_own_tree_does_not_have`
      sweeps this list, so a generated page carrying a `file.py:123` token
      reds a gate about THIS project's prose. **THIS BULLET NAMED A TEST
      THIS TREE HAS NEVER HAD**, at any revision — `git log -S` over
      `tests/` finds no commit that ever defined it — so the one sentence
      saying which gate consumes this list sent the reader nowhere. It
      arrived at `ec69818` and shipped in `v0.2.0` (`git merge-base
      --is-ancestor ec69818 v0.2.0` succeeds), and could not be seen:
      `tests/` was outside the citation scan, which is the exclusion this
      commit replaced;
    * quiet — `_resolve_citation`'s second rule accepts a bare basename
      *"only when exactly one file in the tree carries it"*, and a generated
      copy of a page is a SECOND bearer. `docs/norms.md` and
      `docs/_build/norms.md` make `norms.md` ambiguous, the citation resolves
      to `None`, and a real citation stops being checked with the suite
      green. That is a gate that has stopped looking.

    `git ls-files` decides, for the same reason it decides in
    `tests/test_sdist_reference_hygiene.py::
    test_every_withheld_root_that_holds_content_is_a_content_root`: *"does
    this repository keep this page"* is a question about the repository, and
    it gives the same answer in a fresh clone and in a checkout somebody has
    built documentation in.

    **AND IT DEGRADES RATHER THAN SKIPPING**, for the reason
    `tests/test_zero_dep_import_discipline.py::
    _roots_the_repository_keeps_python_under` gives: where git cannot answer
    — an unpacked sdist, a `git archive` export — the tree contains the
    shipped files and nothing generated, so the walk IS the repository
    there. Measured on this tree at `a431646`: the two enumerations return
    the same 355 files, and the walk's untracked residue is EMPTY, so the
    repair is a no-op on a clean checkout and only ever subtracts what the
    developer's tooling put there.
    """
    tracked = tracked_paths(_REPO)
    out = []
    for root in _shipped_roots():
        base = _REPO / root
        if base.is_file():
            out.append(base)
        elif base.is_dir():
            out.extend(
                p for p in sorted(base.rglob("*"))
                if p.is_file()
                and p.suffix in _TEXT_SUFFIXES
                and "__pycache__" not in p.parts
            )
    if tracked is None:
        return out
    index = set(tracked)
    return [p for p in out if p.relative_to(_REPO).as_posix() in index]


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

    # ------------------------------------------------------------------
    # THE SCAN REACHES `tests/`, DERIVED — the exact instrument for the
    # widening, and the one the anti-vacuity floors in the two citation
    # gates explicitly are not. It is not a count: it is the whole set,
    # compared with what `git ls-files` says the repository keeps under
    # `tests/` at the suffixes a citation is looked for in. A scan that
    # quietly stopped walking `tests/`, or a suffix dropped from
    # :data:`_CITATION_SUFFIXES`, fails here by name rather than by a
    # number that could be argued about.
    tracked = tracked_paths(_REPO_ROOT)
    if tracked is not None:  # pragma: no branch - git answers in this tree
        expected = {
            p for p in tracked
            if p.startswith("tests/") and p.endswith(_CITATION_SUFFIXES)
        }
        got = {
            rel for rel, _ in _citation_sources() if rel.startswith("tests/")
        }
        assert got == expected, (
            f"the citation scan and `git ls-files` disagree about what "
            f"`tests/` holds. Missing from the scan: "
            f"{sorted(expected - got)[:5]}; in the scan and not tracked: "
            f"{sorted(got - expected)[:5]}. `tests/` is 181 of this "
            f"repository's 539 tracked files (`9b5b496`, 2026-08-28), and "
            f"an exclusion that took it out is the defect `_prose_only` "
            f"exists to have replaced."
        )

    # ------------------------------------------------------------------
    # THE PROSE MASK, DRIVEN IN BOTH DIRECTIONS.
    #
    # Every sample below spells a citation of a test this tree does not
    # have, AS A VALUE in this file. Writing them here is safe only while
    # :func:`_prose_only` drops values — which makes this block the mask's
    # own absence half, and a live one rather than a written one: if the
    # mask ever stopped masking, these literals become dangling citations
    # in a file the scan reads, and
    # `test_every_test_cited_in_core_prose_still_exists` reds on them by
    # name. A mask whose failure mode is a green suite is the failure mode
    # this module exists to refuse.
    for label, source, survives in (
        ("a comment", "# see tests/x.py::test_gone\n", True),
        ("a module docstring", '"""cites tests/x.py::test_gone."""\n', True),
        ("a function docstring",
         'def f():\n    """cites tests/x.py::test_gone."""\n', True),
        # ... and every OTHER bare string statement, not only the first in
        # a body. The mask is deliberately wider than "docstring" here:
        # the rule is what the string IS, and a bare string statement is
        # text wherever it sits.
        ("a bare string statement that is not a docstring",
         'def f():\n    x = 1\n    "cites tests/x.py::test_gone"\n', True),
        ("an assignment value", 'X = "tests/x.py::test_gone"\n', False),
        ("a tuple element", 'X = ("tests/x.py::test_gone",)\n', False),
        ("a call argument", 'f("tests/x.py::test_gone")\n', False),
        ("a dict value", 'X = {"k": "tests/x.py::test_gone"}\n', False),
        ("a monkeypatch.setitem key",
         'm.setitem(d, "tests/x.py::test_gone", 1)\n', False),
        # ... and an f-string, which is a value and is masked WHOLE. The
        # allow-list shape is what buys this: nothing had to name
        # `JoinedStr` for it to be dropped.
        ("an f-string", 'X = f"tests/x.py::test_gone {y}"\n', False),
    ):
        masked = _prose_only(source)
        assert bool(_TEST_REF_WRAPPED.search(masked)) is survives, (
            f"the prose mask disagrees about {label}: a citation there is "
            f"{'dropped' if survives else 'kept'} and should be "
            f"{'kept' if survives else 'dropped'}. Prose is a COMMENT or a "
            f"bare string statement; everything else is a value, and the "
            f"plants this widening had to survive are values."
        )
        assert (
            len(masked) == len(source)
            and masked.count("\n") == source.count("\n")
        ), (
            f"the mask changed the shape of {label}, so every line number "
            f"and paragraph boundary computed from it is wrong"
        )
    # ... and a file that does not PARSE degrades to the whole text, loudly
    # rather than quietly: a mask that answered "no prose here" would report
    # absence of evidence as evidence of absence.
    broken = 'X = "tests/x.py::test_gone"\ndef broken(:\n'
    assert _prose_only(broken) == broken, (
        "an unparseable `.py` file is being masked rather than read whole; "
        "the degrade must be toward reading MORE, never toward reading "
        "nothing"
    )


# ---------------------------------------------------------------------------
# BARE test names — the citation form no checker in this tree could see.
# ---------------------------------------------------------------------------

#: A backticked BARE test name: `test_…`, with no `path.py::` in front of it.
#: The optional continuation is the same mid-name wrap `_TEST_REF_WRAPPED`
#: handles, and for the same reason — `SOUNDNESS.md` breaks long identifiers
#: after an underscore. One of the three defects this gate was built on was
#: wrapped that way, and it is the one no earlier sweep reached.
_BARE_TEST_NAME = re.compile(
    r"`(test_[A-Za-z0-9_]+)"
    r"((?<=_)[ \t]*\n[ \t#*>]*[A-Za-z0-9_]+)?(\*?)`"
)

#: THE TWO CITATION FORMS ARE DISJOINT BY CONSTRUCTION, and this is where
#: that is written down rather than guarded against.
#:
#: A first draft of this gate carried a skip: look back 90 characters, and if
#: a `tests/….py::` sits there, leave the match to
#: `test_every_test_cited_in_core_prose_still_exists`. **Driven on this
#: tree, it never fired — 0 of 165 matches** — and it cannot, because
#: :data:`_BARE_TEST_NAME` requires a backtick IMMEDIATELY before `test_`,
#: while a `path::name` citation's backtick sits before the PATH: the
#: characters after it are `tests/`, which is `test` with no underscore. A
#: branch a checker never takes is a branch nobody has read, so the skip is
#: gone and the disjointness is asserted instead, in
#: `test_every_bare_test_name_in_shipped_prose_resolves`, against this
#: number, which is the whole content of the claim.
_PATH_FORM_OVERLAPS = 0

#: NAMES THE PROSE DELIBERATELY MENTIONS AS ABSENT, each with its reason.
#:
#: **This table is the whole cost of this gate and it is stated rather than
#: hidden.** Driven against `5ad906f`, the tree this gate was written on,
#: with the resolver below and before any exemption existed: **162 backticked
#: bare test-name mentions across 30 files** (97 of them in `SOUNDNESS.md`),
#: against **74** `path::name` citations in the same set — so the form
#: nothing could read was more than twice as common as the form something
#: could. **Fifteen** of the 162 resolved to nothing. **Three were defects**
#: and are repaired; the twelve that were not are here, together with the
#: three defect names, which the repairs now quote as the ORIGINALS in the
#: notes recording each rename. So the table is fifteen, and every one of the
#: fifteen sits in a sentence whose SUBJECT is that the name is gone. Three
#: shapes:
#:
#:   * a rename or removal the sentence itself records (eleven),
#:   * another project's test name, cited as another project's (two),
#:   * a name used as an EXAMPLE of a bad name or of a fabricated citation
#:     (two).
#:
#: **The ledger is not rewritten to satisfy a checker.** `SOUNDNESS.md` records
#: a closed finding by leaving the sentence standing and annotating it — the
#: same reason :data:`_SUPERSEDED_BY` exists — so "make the prose resolve"
#: is not available for any of the eleven, and would destroy the record for
#: its own convenience if it were.
#:
#: **The table cannot rot into a blanket**: `test_the_declared_absences_are_
#: still_absent` requires every name here to resolve NOWHERE, so an entry that
#: has become a live test fails rather than silently exempting it.
_NAMES_DECLARED_ABSENT: dict[str, str] = {
    "test_the_nightly_workflow_still_runs_the_canary":
        "renamed to `test_the_canary_and_the_workflow_agree_about_the_two_"
        "legs`; `.github/scripts/tripwire_canary.py` cites the new name by "
        "`path::name` two lines above and names the old one as the rename",
    "test_a_TWO_FACED_records_cannot_show_the_bar_one_thing_and_the_loop_"
    "ANOTHER":
        "the one id REMOVED in the `3e107cf..faefc48` diff SOUNDNESS.md is "
        "reporting; its successor is the parametrised refusal test and the "
        "sentence says so",
    "test_f4wheel2_sweep_the_reproducer_scan_errs_toward_crying_wolf":
        "the de-vacuified test whose ten retired params SOUNDNESS.md is "
        "counting in a `--collect-only` id diff",
    "test_an_assume_after_the_assert_is_pinned_on_BOTH_legs":
        "renamed to `…_withholds_on_BOTH_legs`; quoted in SOUNDNESS.md as "
        "the ORIGINAL name in the note that records the rename (0.2.0 D5)",
    "test_the_two_legs_do_not_yet_agree_on_assume_ordering":
        "renamed to `test_the_two_legs_now_agree_on_assume_ordering`, a "
        "SEMANTIC rename; quoted in SOUNDNESS.md as the original name in the "
        "note that records it (0.2.0 D5)",
    "test_reverse_mode_ad_DOES_preserve_the_clamp":
        "renamed to `test_reverse_mode_ad_preserves_the_clamp_for_the_ds_dus_"
        "pair`; SOUNDNESS.md writes the rename as `old` -> `new` and the new "
        "name resolves",
    "test_gather_dynamic_index_declines_not_crashes":
        "one of the four renamed gather tests SOUNDNESS.md lists as the four "
        "ids removed in an id diff",
    "test_gather_out_of_range_index_declines_not_crashes":
        "one of the four renamed gather tests SOUNDNESS.md lists as the four "
        "ids removed in an id diff",
    "test_fvm_gather_dynamic_index_declines_traced":
        "one of the four renamed gather tests SOUNDNESS.md lists as the four "
        "ids removed in an id diff",
    "test_gather_out_of_range_static_index_declines_traced":
        "one of the four renamed gather tests SOUNDNESS.md lists as the four "
        "ids removed in an id diff",
    "test_the_declaration_reader_is_a_FUNCTION_and_not_a_single_READ":
        "deleted at `f729d70` and replaced by the two door-installs tests; "
        "SOUNDNESS.md's measurement of it is stamped to `30d4b04` and the "
        "paragraph names both successors by `path::name` (0.2.0 D5)",
    "test_integer_overflow":
        "jax's own, cited as `api_test.py:8351` in the sentence that names "
        "it; it is not a test of this repository and never was",
    "test_chees_adaptation":
        "blackjax's own, quoted inside a table of blackjax issue titles in "
        "`design/tracker-probe-classification.md`",
    "test_float_div_is_completely_unchanged":
        "`docs/norms.md` cites it as an EXAMPLE of a name asserting more "
        "than its test checked — the instance is the point, not the test",
    "test_this_page_s_numbered_sections_each_name_a_live_pinning_test":
        "`docs/proposed-decline-messages.md` quotes it as a citation of a "
        "test THAT DOES NOT EXIST, which is the sentence's whole subject",
    # --- and thirteen that arrived with `tests/`, 2026-08-28 -------------
    # Every one was ALREADY in the tree and already unresolvable; what
    # changed is that something can now read them. Eleven are rename or
    # supersession records in a docstring whose subject IS the old name --
    # the same shape as the eleven above, written by the same convention --
    # and two are placeholders in prose about the SHAPE of a nodeid rather
    # than about any test.
    "test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation":
        "superseded by `test_the_two_pairing_gates_bind_the_ESCALATION_AND_"
        "the_propagation`; quoted in this file's own `_TEST_REF_WRAPPED` and "
        "`_SUPERSEDED_BY` comments as the name `SOUNDNESS.md` cites and its "
        "annotation repoints -- the live supersession they are measuring",
    "test_a_ONE_SHOT_records_cannot_silence_the_bar":
        "renamed at `218f969` to `test_a_ONE_SHOT_records_behaves_EXACTLY_"
        "LIKE_THE_TUPLE_it_yields`; quoted as the ORIGINAL in this file's "
        "own note recording that rename, which is the incident that made "
        "`test_every_test_cited_in_core_prose_still_exists` exist",
    "test_a_name_this_tree_has_never_had":
        "a SYNTHETIC plant in `_resolvable_test_names`'s driven block -- the "
        "name written into `SOUNDNESS.md` to drive the pruning repair red; "
        "never a test, and the paragraph says so in the line it appears in",
    "test_x":
        "a PLACEHOLDER nodeid, not a test: `tests/_state_guard.py` uses it "
        "to say how wide a licence keyed to a nodeid is, and this file's "
        "`_SUPERSEDED_BY` comment uses it to draw the shape of an annotated "
        "citation. Held absent so that writing a real `test_x` reds here",
    "test_one":
        "a PLACEHOLDER for the plant in the nested-session measurements of "
        "`tests/_state_guard.py` and `tests/test_state_guard.py` -- the "
        "name given to the generated test that runs `pytest.main` inside "
        "the outer session. It is written into a SOURCE STRING, so no file "
        "defines it and `_defined_test_names` correctly does not see one",
    "test_withholding_is_forward_scoped":
        "superseded by `test_withholding_is_QUERY_scoped`, which pins the "
        "OPPOSITE ruling; `tests/test_assume_constrain.py` names the old one "
        "in the comment recording that the defect was the old scoping",
    "test_the_module_docstring_states_a_scope_and_the_counter_examples_hold":
        "replaced by the four endpoint-discipline tests of "
        "`tests/test_interval.py`; that file's `WHAT THIS REPLACES` block "
        "quotes the old name and the docstring sentence that was wrong",
    "test_the_shipped_page_still_discloses_what_this_files_fixture_did":
        "renamed to `test_the_disclosures_phrases_are_still_PRESENT_and_"
        "nothing_more` because the old NAME asserted more than the test "
        "held; the docstring quotes the retired name as part of the "
        "retraction, which is the whole point of that paragraph",
    "test_a_region_is_lexically_bounded":
        "renamed to `test_a_region_ends_where_the_block_ends_in_straight_"
        "line_code`; `tests/test_tripwire_eager.py` quotes the old name in "
        "the sentence recording that a region is NOT lexically bounded",
    "test_every_deferred_route_is_caught_by_the_transfer_instead":
        "renamed to `test_every_deferred_route_is_declined_by_the_mechanism_"
        "it_declares`; the docstring quotes the old name as a claim that "
        "went stale with the assertion it described",
    "test_the_attribution_fail_safe_refuses_a_misaligned_verdict":
        "replaced by `test_a_conjunct_s_verdict_cannot_land_on_another_"
        "conjunct_s_reason`, whose docstring opens by naming it and says "
        "its subject no longer exists",
    "test_the_discriminant_refuses_a_non_bool_operand":
        "replaced by `test_a_non_bool_and_is_refused_as_a_conjunction_and_"
        "says_so`, whose docstring opens by naming it and records that the "
        "test it replaces could not fail",
    "test_this_files_position_in_the_collection_is_the_measured_one":
        "removed at `ac3038d`, which took the typed collection rank off "
        "`CHANGELOG.md`; `tests/test_probe_oracle.py` and "
        "`tests/test_probe_stability_level.py` both quote it in the "
        "paragraph recording that the mechanism their naming argument "
        "rested on is gone. Its successors are named beside it in both",
    "test_the_LIMIT_of_this_bar_when_the_decided_slice_is_genuinely_the_same":
        "replaced by `test_the_pairing_gate_refuses_the_mispairing_the_bar_"
        "only_narrows`, whose docstring quotes the old name and records "
        "that it asserted the PERMISSIVE outcome",
}


def _resolvable_test_names() -> set[str]:
    """Every name a bare `test_…` mention may legitimately denote.

    TWO POPULATIONS, because a bare name is used for two things. A bare
    `test_the_two_legs_now_agree_on_assume_ordering` names a FUNCTION; a bare
    `test_assume_ledger` names a MODULE (`tests/test_assume_ledger.py`).
    Driven at `5ad906f`: a function-only resolver reported 27 unresolved
    mentions and **eleven of the 27 were module names**. Resolving both is
    what keeps the exemption table down to the names that are genuinely
    gone: with modules out, `_NAMES_DECLARED_ABSENT` would have had to carry
    those eleven too — 26 entries instead of fifteen, eleven of them naming
    live test modules.

    **THAT NUMERAL READ "nine", AND THIS FILE'S OWN ARITHMETIC SAID ELEVEN.**
    It was written rather than measured. Eleven is what closes the books:
    27 − 11 = **16**, the mentions unresolved under BOTH populations, and
    16 − 1 for the single one the supersession licence covers = **15**, which
    is `_NAMES_DECLARED_ABSENT`'s size AND the dangling count with that table
    emptied — and both of those were measured. Nine would have made them 18
    and 17. Re-driven at `5ad906f` twice, once through the helpers below and
    once through a re-implementation importing nothing from this file; both
    give the same eleven: `test_any_pytree`, `test_assume_disclosure_claims`,
    `test_assume_ledger`, `test_assume_scope_identity`,
    `test_ir_message_totality`, `test_pow_row_gauge_jax`,
    `test_scatter_gauge_jax`, `test_sdist_contents`,
    `test_square_row_gauge_jax`, `test_suite_disclosure`, `test_vacuity`.

    Functions come from :func:`_defined_test_names`, so the five shapes that
    resolver deliberately over-reports and the two it closed apply here
    unchanged — one resolver, one semantic, one place to change it.

    **AND THE SWEEP IS PRUNED THE WAY PYTEST PRUNES, WHICH IT WAS NOT.** A
    bare `rglob("*.py")` reads `tests/build/`, `tests/dist/`, `tests/venv/`
    and every dot-directory under `tests/` — none of which any session
    collects. A stale copy of a module in one of them therefore RESOLVED a
    name this tree no longer has, and this gate is the wrong way round for
    that to be harmless: resolving is PASSING. Driven at `a431646`, blind,
    before the pruning went in::

        SOUNDNESS.md   `test_a_name_this_tree_has_never_had`   1 failed
        + tests/build/test_stale_copy.py defining that name    1 PASSED

    and `pytest --collect-only` over the same tree collects **0** tests from
    `tests/build/`, so what silenced the gate was a file no invocation can
    reach. The same pruning that `conftest.collectable_test_files` applies,
    from the same predicate, so "a test of this suite" means one thing here
    and there.

    It reddens `test_the_declared_absences_are_still_absent` in the other
    direction too — that one asserts a declared-absent name does NOT
    resolve, so a stale copy of a RENAMED test would have failed it for a
    file nobody can collect. One repair, both directions.
    """
    names: set[str] = set()
    for path in sorted(_TESTS.rglob("*.py")):
        if _in_a_pruned_directory(path):
            continue
        names.add(path.stem)
        defined = _defined_test_names(path.read_text(encoding="utf-8"))
        if defined:
            names.update(defined)
    return names


def _bare_mentions(text: str):
    """Every backticked bare test name in ``text``, as `(spellings, star, m)`.

    Two spellings are yielded for a mention wrapped mid-name — the plain one
    and the joined one — because the resolver must try both, exactly as
    `test_every_test_cited_in_core_prose_still_exists` does.

    The `path::name` form does not have to be excluded here; it cannot match.
    See :data:`_PATH_FORM_OVERLAPS`.
    """
    for m in _BARE_TEST_NAME.finditer(text):
        name, wrap, star = m.groups()
        spellings = [name]
        if wrap:
            spellings.append(name + wrap.split("\n")[1].strip(" \t#*>"))
        yield spellings, star, m


def test_every_bare_test_name_in_shipped_prose_resolves():
    """A BARE test name is a citation too, and nothing could read one.

    `test_every_test_cited_in_core_prose_still_exists` matches
    `tests/…\\.py::test_…`. Most of this project's prose does not write that
    form. Driven at `5ad906f` over the same shipped set: **74 `path::name`
    citations against 162 backticked bare names**, 97 of the bare ones in
    `SOUNDNESS.md` alone. So the majority form was the unchecked one, and the
    checked form's own docstring says what an unchecked citation is worth —
    *"a renamed test turns it into a claim about nothing."*

    **THREE DEFECTS WERE LIVE WHEN THIS WAS WRITTEN, AND ALL THREE HAD BEEN
    GREEN FOR WEEKS.** Two came from the 0.2.0 D5 accuracy sweep and one this
    gate found on its first run:

        SOUNDNESS.md   test_an_assume_after_the_assert_is_pinned_on_BOTH_legs
                       renamed to `…_withholds_on_BOTH_legs`
        SOUNDNESS.md   test_the_two_legs_do_not_yet_agree_on_assume_ordering
                       renamed to `…_now_agree_on_…` -- the OPPOSITE of what
                       the sentence citing it describes
        SOUNDNESS.md   test_the_declaration_reader_is_a_FUNCTION_and_not_a_
                       single_READ -- DELETED at `f729d70`, cited in the
                       present tense by a sentence written 8.5 hours earlier

    The second is the sharpest: the sentence said the tree pinned a
    *disagreement* between two legs and that a query-scoping change was
    *forthcoming*; the tree pins the *agreement*, and the change landed. A
    reader following that citation would have found a test asserting the
    reverse of the claim it was offered as support for.

    **AND HERE IS THE PIN FAILING, WHICH IS THE ONLY THING THAT MAKES IT A
    PIN.** This file was dropped into a clean `5ad906f` checkout — the tree
    this branch started from — with the three defect
    names removed from :data:`_NAMES_DECLARED_ABSENT` and the other twelve
    left in place. It reds, naming those three and **nothing else**:

        SOUNDNESS.md:2923: `test_an_assume_after_the_assert_is_pinned_on_BOTH_legs`
        SOUNDNESS.md:2925: `test_the_two_legs_do_not_yet_agree_on_assume_ordering`
        SOUNDNESS.md:9980: `test_the_declaration_reader_is_a_FUNCTION_and_not_a_single_READ`

    `1 failed, 6 passed`. So the twelve exemptions are the whole of the
    false-positive surface on the tree this was written against, and the
    three are the whole of what it catches there.

    **WHY THIS IS NOT A SECOND `path::name` CHECKER.** It resolves a NAME,
    against every test function and every test module under `tests/`, with no
    path to key on — which is the only thing that can read the form the prose
    actually uses. The cost is that a name legitimately mentioned as absent
    has to be declared: :data:`_NAMES_DECLARED_ABSENT`, fifteen of them,
    each with the reason, each held to still being absent by
    `test_the_declared_absences_are_still_absent`. That table is the
    honest price and it is bounded — eleven of the fifteen are rename records
    in a ledger that is never rewritten, and those do not churn.

    **THE RESIDUE, STATED RATHER THAN LEFT TO BE FOUND.** Six things this
    does not do. This read *"Four things … each driven or reasoned rather
    than assumed"* until 2026-08-22; the one that was reasoned is the third
    below, the reasoning was WRONG, and all six are driven now.

    * **The exemption is by NAME, not by SITE.** A new sentence that cited
      `test_integer_overflow` as a live pin of something in THIS repository
      would be exempt, because the table cannot tell that mention from the
      one it was written for. Keying it to `(file, name)` would be tighter
      and would also make every entry a position-bearing claim with no line
      number in it; it is not done because all fifteen are names no test in
      this tree currently has, so the widened licence covers nothing real —
      and `test_the_declared_absences_are_still_absent` is what keeps that a
      measurement rather than an assumption.
    * **It resolves EXISTENCE, not identity.** A name that resolves may be a
      different test that happens to share the name, in a different file
      from the one the sentence means. The `path::name` form does not have
      that gap, which is a reason to prefer writing citations that way.
    * **The supersession licence carries `_SUPERSEDED_BY`'s residue, in a
      WIDER window.** Both licences are keyed identically — from the mention
      to the next SAME-KIND mention, else the end of the paragraph — so the
      two spans are directly comparable. Measured on this tree, in
      characters:

          all shipped   bare  n=165  median 260  mean 941  max 24642
                        path  n= 81  median 151  mean 924  max 14317
          SOUNDNESS.md  bare  n=100  median 206  mean 533  max  6325
                        path  n= 26  median 160  mean 377  max  4507

      The bare form's window is the wider one at both scopes and on median,
      mean and max. Wider is not open — it still asks only that the
      replacement RESOLVE, not that it be related to the name it replaces —
      but it is wider, and a residue is worth nothing stated in the wrong
      direction.

      **THIS SAID "NARROWER", AND THE ARGUMENT FOR IT IS THE INSTRUCTIVE
      PART.** The reason given was that bare mentions are commoner — 165 to
      81 across the shipped set, which is true — so the next one must arrive
      sooner and close the window earlier. **AGGREGATE COMMONNESS DOES NOT
      IMPLY PROXIMITY INSIDE THE ENCLOSING PARAGRAPH**, and the measurement
      breaks the argument at every step of it:

          share whose window runs to the END of
              the paragraph, not to a next mention   bare 80.0%  path 80.2%
          mentions per paragraph that holds any      bare  1.25  path  1.25
          gap to the next same-kind mention where
              there is one (median chars)            bare   120  path    22
          mention to end of paragraph (median)       bare   352  path   326

      Four licences in five, of EITHER form, are capped by the end of the
      paragraph and not by a next mention, so the corpus-wide ratio never
      gets to matter — and where it might, the within-paragraph density is
      1.25 for both. Where a next mention DOES cap the window it is the
      `path::name` gap that is the short one: repeated citations sit
      adjacent in a list, 22 characters apart, where a repeated bare mention
      is 120 away. The 165-to-81 ratio is a fact about the whole corpus and
      the window is a fact about one paragraph, and nothing carries the
      first into the second.

      This was the only item in this list that was argued rather than
      driven, and it is the same failure :data:`_SUPERSEDED_BY`'s comment
      above records against itself — a residue bounded at the wrong width
      because a number stood in for a measurement.
    * **The star/prefix branch has never fired on this tree.**
      :data:`_BARE_TEST_NAME`'s trailing `(\\*?)` switches resolution from
      equality to PREFIX, for a family citation. Measured here: **0 of 165**
      bare mentions carry the star, against **2 of 81** `path::name`
      citations that do — so the form that actually has the families is the
      one this gate does not read. The branch is live, not dead: planting a
      real family into `docs/norms.md` gives `7 passed`, and a fabricated
      one gives `1 failed` naming it. But an unexercised branch is a branch
      nobody has read, which is exactly what :data:`_PATH_FORM_OVERLAPS`
      above exists to forbid — and unlike that one this count is neither
      asserted nor pinned. Recorded, not closed: it belongs beside
      `_PATH_FORM_OVERLAPS` and is not there.
    * **The resolver's namespace is wider than pytest's collection.**
      :func:`_resolvable_test_names` scans EVERY `tests/**/*.py`, while
      pytest collects `test_*.py` and `*_test.py` by default and
      `pyproject.toml` sets no `python_files`. So a `test_…` defined in a
      helper module would resolve a citation pytest never runs. **Latent,
      and measured at 0**: 15 of the 170 `.py` files under `tests/` are not
      `test_*.py`, and not one of them defines a `test_`-prefixed name or
      has a stem beginning `test`. Driven by planting a `tests/_helperzz.py`
      holding one `test_…` def — the citation resolves (`7 passed`) while a
      `--collect-only` over `tests/` reports nothing from that file.
    * **`tests/` IS IN, from 2026-08-28, and this bullet used to be the
      reason it was not.** It read *"`tests/` and `scratchpad/` stay out,
      for the same reasons the `path::name` scan gives: three test modules
      write citation-shaped strings as plants, and `scratchpad/` is tracked
      evidence."* The plants are values and :func:`_prose_only` drops
      them; the record is above :data:`_CITATION_ROOTS`. `scratchpad/`
      stays out on the second half of that sentence, which was never about
      a directory being awkward — it is evidence, not maintained prose.
      **This gate is where the widening cost the most and bought the
      most**: run blind against `9b5b496` on 2026-08-28 it named **24**
      dangling bare mentions in `tests/`, against **zero** outside it,
      staged as:

          17 mentions of 14 names whose sentence's SUBJECT is that the
             name is gone -- now in :data:`_NAMES_DECLARED_ABSENT`, which
             is why that table is 29 entries and not fifteen
           5 citations of names NO REVISION of this tree has ever
             defined (`git log -S` finds no commit that wrote the `def`),
             each repointed at the test its sentence means
           2 live names written SHORT -- a citation abbreviated to its
             first few words, which names nothing

      Two of the 17 are in `tests/test_state_guard.py`, a file another
      branch is in; they are covered without editing it, because the
      declaration is keyed to a NAME and not to a site — the residue
      stated three bullets up, here doing the useful half of its work.
    """
    resolvable = _resolvable_test_names()
    dangling, seen, files = [], 0, set()
    for source, text in _citation_sources():
        for spellings, star, m in _bare_mentions(text):
            seen += 1
            files.add(source)
            if any(
                (any(d.startswith(s) for d in resolvable) if star
                 else s in resolvable)
                for s in spellings
            ):
                continue
            if any(s in _NAMES_DECLARED_ABSENT for s in spellings):
                continue
            # ... and last, a supersession ANNOTATING THIS MENTION: a bare
            # `::name` declared after it, with no other bare mention in
            # between. Same rule, same reason and the same residue as
            # `_SUPERSEDED_BY`'s -- a licence keyed to position, because
            # position is the only thing a checker can read.
            para_start, para = _enclosing_paragraph(text, m.start())
            after = m.end() - para_start
            following = _BARE_TEST_NAME.search(para, after)
            window = para[after:following.start() if following else len(para)]
            if any(
                replacement in resolvable
                for replacement in _SUPERSEDED_BY.findall(window)
            ):
                continue
            lineno = text.count("\n", 0, m.start()) + 1
            dangling.append(f"{source}:{lineno}: `{spellings[-1]}`")
    assert not dangling, (
        "shipped prose names test(s) that exist nowhere under `tests/`:\n  "
        + "\n  ".join(dangling)
        + "\nA bare name is the citation form this project's prose mostly "
        "uses, and a renamed test turns it into a claim about nothing just "
        "as surely as it does a `path::name` one. Repoint it, write the "
        "successor beside it as `::new_name`, or -- if the sentence's "
        "SUBJECT is that the name is gone -- declare it in "
        "`_NAMES_DECLARED_ABSENT` with the reason."
    )
    # ... and it is not vacuous. Measured on this tree on 2026-08-28: 788
    # backticked bare mentions across 110 files, of which 214 across 32
    # files are outside `tests/` — this is the form the widening moved
    # most, by more than three to one.
    #
    # UNLIKE THE `path::name` FLOORS, THESE TWO DO BRACKET THE WIDENING:
    # losing `tests/` from the scan takes 788 to 214 and 110 files to 32,
    # so both would red on it. They are still not the instrument for it —
    # that is the derived reach assertion named beside the other pair —
    # and they are set between the two measurements rather than at some
    # fraction of one, so what they mean can be stated.
    assert seen >= 500, (
        f"only {seen} bare test-name mention(s) found across "
        f"{list(_CITATION_ROOTS)} and the root pages; the pattern has stopped "
        f"matching how they are written, or the scan has stopped reaching "
        f"them"
    )
    assert len(files) >= 60, (
        f"bare test names were found in {len(files)} file(s); they have "
        f"collapsed into one or two, or the scan has narrowed"
    )
    # ... and the resolver really resolves: a name that IS a live test, and a
    # name that is a live test MODULE, both come back resolvable, so "not in
    # the set" means something.
    assert "test_every_bare_test_name_in_shipped_prose_resolves" in resolvable
    assert "test_prose_hygiene" in resolvable
    assert "test_no_such_test_anywhere_in_this_tree_at_all" not in resolvable
    # ... and NOTHING A SESSION CANNOT COLLECT ANSWERED ANY OF IT. This is
    # the false GREEN in `_resolvable_test_names`'s docstring, held shut from
    # the gate that suffered it. Empty on a clean checkout and not vacuous
    # where it matters — a stale `tests/build/`, a vendored tree or a
    # dot-directory under `tests/` is the condition, and it is the developer's
    # environment rather than the repository, which is why the resolver may
    # not read one.
    reachable = {
        name
        for path in collectable_test_files()
        for name in _defined_test_names(path.read_text(encoding="utf-8"))
    }
    unreachable = {
        name
        for path in _TESTS.rglob("test_*.py")
        if _in_a_pruned_directory(path)
        for name in _defined_test_names(path.read_text(encoding="utf-8"))
    }
    leaked = sorted((unreachable - reachable) & resolvable)
    assert not leaked, (
        f"{leaked} resolve(s) only through a file under `tests/` that pytest "
        f"will not collect, so a citation of it names a test nobody can run "
        f"and this gate would pass over it. The resolver must prune what "
        f"pytest prunes — see `_resolvable_test_names`."
    )
    # ... and the two citation forms really are disjoint, so neither gate is
    # reading the other's population -- :data:`_PATH_FORM_OVERLAPS`, measured
    # rather than argued.
    overlapping = []
    for source, text in _citation_sources():
        spans = [c.span() for c in _TEST_REF_WRAPPED.finditer(text)]
        for _, _, m in _bare_mentions(text):
            if any(lo <= m.start() < hi for lo, hi in spans):
                overlapping.append(f"{source}:{m.group(1)}")
    assert len(overlapping) == _PATH_FORM_OVERLAPS, (
        f"{len(overlapping)} bare test-name match(es) fall inside a "
        f"`path::name` citation and {_PATH_FORM_OVERLAPS} was measured: "
        f"{overlapping[:5]}. The two forms are disjoint by construction -- a "
        f"backticked bare name cannot start a `tests/…py::` citation -- and "
        f"one gate is now reading the other's population."
    )


def test_the_declared_absences_are_still_absent():
    """Every exemption must still be an absence, or it is a blanket.

    An allowlist of names is only sound while each name is genuinely gone. A
    name that came back — someone re-creates a test under a retired name, or
    an entry is copied in by hand — would sit here silently exempting a live
    citation from ever being checked again, which is the failure mode of
    every allowlist that is written once and read never.

    So the direction is inverted: this asserts the exemptions do NOT resolve.
    The table can only shrink by deleting an entry, never by drifting into
    one that covers something real.
    """
    resolvable = _resolvable_test_names()
    resurrected = sorted(n for n in _NAMES_DECLARED_ABSENT if n in resolvable)
    assert not resurrected, (
        f"name(s) declared absent in `_NAMES_DECLARED_ABSENT` that now "
        f"resolve under `tests/`: {resurrected}. An exemption for a name "
        f"that exists is a live citation nothing checks. Delete the entry "
        f"and let the gate read the name."
    )
    for name, reason in _NAMES_DECLARED_ABSENT.items():
        assert len(reason) >= 40, (
            f"the exemption for {name!r} gives no real reason. Every entry "
            f"here is a claim that a sentence MEANS to name something gone, "
            f"and a claim needs its argument beside it."
        )
    # ... and it is not vacuous: the table really has entries to check, and
    # the resolver it checks them against really is populated.
    assert len(_NAMES_DECLARED_ABSENT) >= 10
    assert len(_resolvable_test_names()) >= 1000
