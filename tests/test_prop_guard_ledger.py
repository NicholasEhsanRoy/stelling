# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The vendoring ledger over ``src/stelling/_tripwire/prop_guard.py``.

That file is a copy of a scored artefact, and the thing that keeps the
artefact's evidence attached to the copy is a note in its header saying
exactly what was changed on the way in.  Documents elsewhere in this tree
restate those figures.  Nothing held them to each other, and a batch that
found a defect in the file **declined to fix it** because a further edit
would have falsified every restatement at once — which is the right instinct
and the wrong resting place, since the price was leaving a sentence a user
reads ungrammatical in order to protect a numeral.

**THIS MODULE STATES NO FIGURE**, and that is deliberate: every number this
file is about is parsed out of the note, so writing one here would be one
more copy to keep in step.  Where a figure is wanted, read the note.

WHAT IS DERIVED, AND FROM WHAT
==============================
The figure is derived from the **enumerated list in the note**, and not from
the ``VENDORING EDIT`` markers in the body, though both are read.

The markers looked like the honest source — they sit at the changed lines,
so they are the closest thing to the artefact ``diff`` the note tells a
reviewer to run.  They are not, and the reason is the whole of the class
this campaign keeps meeting.  **A marker is a comment beside a line of
code, and an editor who forgets to declare an edit forgets its marker at
least as readily as its sentence.**  Deriving the count from the markers
would make the gate's answer a function of exactly the artefact most likely
to be missing, so it would go green on the omission it exists to catch, and
would have gone green on an edit landed with neither marker nor prose.

The list is the ledger.  It is what a reviewer reads and what the other
documents restate, and it carries something no marker can: **the KIND of
each edit.**  A ledger that counts edits without distinguishing them has
stopped being evidence about the artefact — an import reroute, a change to
what the predicate answers after a fault, and a change to the English in an
error message are three different claims about how much of the scoring
survives the copy, and a tally that adds them up says none of them.

So the list is the source, and the markers are **pinned to it in both
directions**: an edit in the list with no marker in the body reddens, and a
marker in the body naming an edit the list does not carry reddens.  Item 1
is the note itself, which cannot mark itself, and that exemption is written
into the pin rather than assumed by it.

WHAT THIS CANNOT SEE, STATED BECAUSE IT IS REAL
===============================================
An edit made with **no marker and no list entry** is invisible here, and no
in-tree check can see it: the only witness is the artefact the note names,
which lives outside this repository.  That is why the note tells a reviewer
to ``diff`` against it, and why the sentence saying so is itself held here.

The tree scan is anchored: a figure is in scope when its own paragraph also
names the file or the vendored predicate.  A restatement written far away
from any such mention is out of the scan's reach, and the scan's floor
below is what stops it from quietly reaching nothing at all.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _repo_files import read_text_files  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
GUARD_REL = "src/stelling/_tripwire/prop_guard.py"
GUARD = REPO / GUARD_REL

#: The kinds an edit may be tagged with in the note, and the words a document
#: may use to qualify a count OF that kind.  A closed vocabulary, because a
#: qualifier this table does not know is REPORTED rather than read as though
#: it were not there: a count of one KIND, silently compared against the
#: total, is a check that passes for the wrong reason.
KIND_ALIASES = {
    "provenance": "provenance",
    "import-route": "import-route",
    "behaviour": "behaviour",
    "behavioural": "behaviour",
    "message-text": "message-text",
}

KINDS = ("provenance", "import-route", "behaviour", "message-text")

#: Sentences elsewhere that point at a single edit by ordinal and claim a
#: kind for it.  An ordinal is a pointer into a list, so it goes stale
#: silently if the list is ever reordered; each is therefore held to the
#: number of the item that actually carries the kind it claims.
#: Whitespace is `\s+` throughout, for the reason `_SP` below is: these
#: sentences are wrapped prose, and a pattern with a literal space in it
#: stops matching the day a word is added above it.
_KIND_CLAIMS = (
    (
        "tests/_state_guard.py",
        re.compile(r"`prop_guard\.py`\s+edit\s+(?P<n>\d+)"),
        "behaviour",
    ),
    (
        "CHANGELOG.md",
        re.compile(r"the\s+\*\*(?P<n>[a-z]+)\*\*\s+edit\s+to\s+the\s+"
                   r"vendored\s+predicate,\s+and\s+the\s+only\s+one\s+"
                   r"that\s+can\s+change\s+an\s+answer"),
        "behaviour",
    ),
    (
        "CHANGELOG.md",
        re.compile(r"the\s+\*\*(?P<n>[a-z]+)\*\*\s+edit\s+to\s+the\s+"
                   r"vendored\s+predicate,\s+and\s+the\s+only\s+one\s+"
                   r"that\s+changes\s+no\s+answer"),
        "message-text",
    ),
)

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _number(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    if token[:-2].isdigit() and token[-2:] in ("st", "nd", "rd", "th"):
        return int(token[:-2])
    return _WORDS.get(token, _ORDINALS.get(token))


# ------------------------------------------------------------------ the note

def _note_and_body() -> tuple[str, str]:
    """The header comment block, and everything after it.

    Split at the module docstring rather than at a line number, which moves
    whenever anything above it is rewritten -- including this note.
    """
    text = GUARD.read_text(encoding="utf-8")
    m = re.search(r'^"""', text, re.M)
    assert m, f"{GUARD_REL} has no module docstring to split the note at"
    return text[:m.start()], text[m.start():]


_ITEM = re.compile(
    r"^#\s+(?P<n>\d+)\. \((?P<kind>[a-z-]+)\)\s+(?P<text>\S.*)$", re.M)
_TALLY = re.compile(
    r"^#\s+TALLY: (?P<total>\d+) = (?P<terms>\d+ [a-z-]+"
    r"(?: \+ \d+ [a-z-]+)*)\s*$", re.M)


def _items() -> tuple[tuple[int, str, str], ...]:
    note, _ = _note_and_body()
    return tuple((int(m.group("n")), m.group("kind"), m.group("text"))
                 for m in _ITEM.finditer(note))


def _tally() -> tuple[int, dict[str, int]]:
    note, _ = _note_and_body()
    m = _TALLY.search(note)
    assert m, (
        f"{GUARD_REL}'s note no longer carries a TALLY line in the form "
        f"`#   TALLY: N = n kind + n kind ...`. That line is the only place "
        f"the by-kind tally is written, and every other document's figures "
        f"are compared against it."
    )
    terms: dict[str, int] = {}
    for term in m.group("terms").split(" + "):
        count, kind = term.split(" ", 1)
        terms[kind] = terms.get(kind, 0) + int(count)
    return int(m.group("total")), terms


def _derived() -> tuple[int, dict[str, int]]:
    """The figure and the by-kind tally, from the enumerated list itself."""
    items = _items()
    counts: dict[str, int] = {}
    for _, kind, _text in items:
        counts[kind] = counts.get(kind, 0) + 1
    return len(items), counts


# ------------------------------------------------------------- the tree scan

_NUM = r"one|two|three|four|five|six|seven|eight|nine|ten|\d+"
_ORD = (r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
        r"|\d+(?:st|nd|rd|th)")

#: The gap between a figure and the word `edit`: ordinary spacing, OR ONE
#: LINE BREAK.  These sentences are wrapped prose, and a pattern that only
#: knows about spaces stops matching the day a word is added above it -- a
#: check that goes QUIET rather than red, which is this module's own subject.
#: At most one newline, so a match cannot cross a blank line into the next
#: paragraph, which is what the anchoring below is keyed on.
_SP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"

#: `<n> [<qualifier>] edit(s)` -- a COUNT, of the whole ledger or of one kind.
_CARDINAL = re.compile(
    rf"(?<![-\w])(?:\*\*)?(?P<n>{_NUM})(?:\*\*)?(?![-\w]){_SP}"
    rf"(?:(?P<qual>[\w-]+){_SP})?edits?(?![-\w])", re.I)
#: `<ordinal> edit` -- a POINTER at one of them.
_ORDINAL = re.compile(
    rf"(?<![-\w])(?:\*\*)?(?P<n>{_ORD})(?:\*\*)?{_SP}edit(?![-\w])", re.I)
#: `edit(s) <lo>-<hi>` -- a pointer at a run of them.
_RANGE = re.compile(
    rf"(?<![-\w])edits?{_SP}(?P<lo>\d+)-(?P<hi>\d+)(?![-\w])", re.I)
#: `edit <n>`, in any case, which is how prose points at one of them.
_REF = re.compile(
    rf"(?<![-\w])(?:VENDORING{_SP})?edits?{_SP}(?P<n>\d+)(?![-\w])", re.I)

#: THE MARKER, and the ONE spelling of it: the comment the note asks for at
#: every changed line.  Written once because both readers below use it -- the
#: pin between the list and the body, and the tree scan's taxonomy -- and two
#: copies of a spelling are two things to keep in step.  Case-sensitive and
#: anchored on the comment hash, so prose that merely NAMES an edit in
#: passing is read as a pointer at one and not as a claim to BE the marker
#: at a changed line.
_MARKER = re.compile(r"# VENDORING EDIT (?P<n>\d+)")

#: A figure is about this ledger when its own paragraph names the file or the
#: predicate.  Paragraph and not a character window: a figure and the thing it
#: counts belong to one another only if they are written together.
_ANCHOR = re.compile(r"prop_guard|vendored predicate|VENDORING EDIT")


class _Figure:
    __slots__ = ("rel", "line", "kind", "n", "hi", "qual", "text")

    def __init__(self, rel, line, kind, n, hi, qual, text):
        self.rel, self.line, self.kind = rel, line, kind
        self.n, self.hi, self.qual, self.text = n, hi, qual, text

    def __repr__(self):  # pragma: no cover - failure text only
        return f"{self.rel}:{self.line} [{self.kind}] {self.text!r}"


def _paragraph_spans(text: str):
    spans, start = [], 0
    for block in re.split(r"\n[ \t#]*\n", text):
        i = text.index(block, start)
        spans.append((i, i + len(block), block))
        start = i + len(block)
    return spans


def _figures():
    """Every edit figure in the tree whose paragraph is about this ledger."""
    found = []
    for rel, text in read_text_files():
        spans = _paragraph_spans(text)
        whole_file = rel == GUARD_REL

        def anchored(pos, spans=spans, whole_file=whole_file):
            # THE NOTE'S OWN FILE IS IN SCOPE THROUGHOUT. Anchoring by
            # paragraph inside it would drop the paragraphs that say
            # "Edits 2-4 exist because ..." -- prose about the ledger, in
            # the ledger, that happens not to repeat the file's own name.
            if whole_file:
                return True
            for a, b, block in spans:
                if a <= pos < b:
                    return bool(_ANCHOR.search(block))
            return False  # pragma: no cover - spans cover the whole text

        taken = []
        for rx, kind in ((_RANGE, "range"), (_CARDINAL, "cardinal"),
                         (_ORDINAL, "reference"), (_REF, "reference")):
            for m in rx.finditer(text):
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue
                if not anchored(m.start()):
                    continue
                taken.append((m.start(), m.end()))
                groups = m.groupdict()
                this = kind
                if kind == "reference" and _MARKER.fullmatch(
                        text[max(0, m.start() - 2):m.end()]):
                    this = "marker"
                found.append(_Figure(
                    rel,
                    text.count("\n", 0, m.start()) + 1,
                    this,
                    _number(groups.get("lo") or groups["n"]),
                    _number(groups["hi"]) if groups.get("hi") else None,
                    (groups.get("qual") or "").lower() or None,
                    " ".join(m.group(0).split()),
                ))
    return found


# --------------------------------------------------------------- the fences

def test_the_note_ENUMERATES_its_edits_and_the_TALLY_counts_that_list():
    """The tally beside the list is derived from the list.

    Numbered from one, contiguous, every item tagged with a kind from the
    closed vocabulary above, and the TALLY line equal to what the list holds
    -- total and by kind.  An item added without a tag, a tag nobody else
    knows, a gap in the numbering and a tally left behind by any of them are
    each a failure here.
    """
    items = _items()
    assert items, (
        f"{GUARD_REL}'s note no longer enumerates its edits in the form "
        f"`#   N. (kind) ...`. That list is the ledger: it is what every "
        f"figure in this tree is compared against."
    )
    numbers = [n for n, _, _ in items]
    assert numbers == list(range(1, len(items) + 1)), (
        f"{GUARD_REL}'s note enumerates {numbers}, which is not a run from "
        f"1. An ordinal reference elsewhere in the tree points INTO this "
        f"list and cannot survive a gap or a repeat."
    )
    unknown = sorted({k for _, k, _ in items} - set(KINDS))
    assert not unknown, (
        f"{GUARD_REL}'s note tags an edit {unknown}, which is not one of "
        f"{list(KINDS)}. Add the kind to `KINDS` deliberately -- what an "
        f"edit IS is the claim this ledger makes, so a new kind is a new "
        f"claim about how much of the artefact's scoring survives the copy."
    )
    total, stated = _tally()
    derived_total, derived = _derived()
    assert total == derived_total, (
        f"{GUARD_REL}'s TALLY line says {total} and its own list enumerates "
        f"{derived_total}: {numbers}"
    )
    assert stated == derived, (
        f"{GUARD_REL}'s TALLY line says {stated} by kind and its own list "
        f"holds {derived}"
    )


def test_every_edit_but_the_note_ITSELF_is_marked_at_the_site_it_changed():
    """The list and the body are pinned to each other, in both directions.

    Item 1 is the note, which cannot carry a marker at a site because it IS
    the site; every other item must be findable in the body by its number.
    The reverse leg is the one that catches an edit made and marked but never
    declared, and it is why this is a set equality and not a subset.
    """
    _, body = _note_and_body()
    marked = {int(m.group("n")) for m in _MARKER.finditer(body)}
    items = _items()
    note_items = [n for n, kind, _ in items if kind == "provenance"]
    assert note_items == [1], (
        f"the provenance item is {note_items} and this pin is written for a "
        f"single one at position 1 -- the note itself, which cannot mark "
        f"itself at a changed line"
    )
    expected = set(range(2, len(items) + 1))
    assert marked == expected, (
        f"{GUARD_REL}'s note enumerates {sorted(expected)} as edits that "
        f"changed a line, and the body carries markers for {sorted(marked)}. "
        f"Missing from the body: {sorted(expected - marked)}; in the body "
        f"and not in the note: {sorted(marked - expected)}. Every edit but "
        f"the note itself is marked `# VENDORING EDIT <n>` at the line it "
        f"changed, so that the list and the file can be read against each "
        f"other."
    )


def test_every_figure_in_the_tree_is_the_ONE_the_note_enumerates():
    """No document states a figure for this ledger that the note does not.

    A partition in both directions: every file that states one is a declared
    site, and every figure a declared site states agrees with the list -- a
    total against the total, a by-kind count against that kind's tally, an
    ordinal or a range against the numbers the list actually has.

    THE QUALIFIER IS READ, NOT SKIPPED.  ``CHANGELOG.md`` qualified its
    count as *import-route* against a note whose import-route items are
    fewer than its total -- the provenance edit had been swept in with them
    -- so a check that read the digit and not the qualifier would have
    compared a count of ONE KIND against the total, and passed.
    """
    total, tally = _derived()
    figures = _figures()
    problems = []
    for fig in figures:
        if fig.kind == "cardinal":
            if fig.qual is None:
                want, what = total, "the total"
            elif fig.qual in KIND_ALIASES:
                kind = KIND_ALIASES[fig.qual]
                want, what = tally.get(kind, 0), f"the {kind} tally"
            else:
                problems.append(
                    f"{fig!r} qualifies a count with {fig.qual!r}, which is "
                    f"not a kind this ledger knows ({sorted(KIND_ALIASES)}). "
                    f"An unread qualifier is how a count of ONE KIND gets "
                    f"compared against the total and passes.")
                continue
            if fig.n != want:
                problems.append(
                    f"{fig!r} states {fig.n} where {what} is {want}")
        elif fig.kind in ("reference", "marker"):
            if not 1 <= fig.n <= total:
                problems.append(
                    f"{fig!r} points at an edit the note does not have; it "
                    f"enumerates 1..{total}")
        elif fig.kind == "range":
            if not (1 <= fig.n < fig.hi <= total):
                problems.append(
                    f"{fig!r} spans edits the note does not have; it "
                    f"enumerates 1..{total}")
    assert not problems, "\n".join(problems)


def test_the_scan_reaches_every_file_that_restates_this_ledger():
    """The declared sites are a partition of what the scan finds.

    A page list is only as wide as its list, so this is not one: the scan
    walks the tree.  What is declared is which files are ALLOWED to carry a
    figure, and the equality catches both a restatement appearing somewhere
    new and a declared one being deleted.
    """
    declared = {GUARD_REL, "CHANGELOG.md", "tests/_state_guard.py"}
    seen = {fig.rel for fig in _figures()}
    assert seen == declared, (
        f"files carrying a figure for this ledger: {sorted(seen)}; declared: "
        f"{sorted(declared)}. New here and undeclared: "
        f"{sorted(seen - declared)} -- add it to `declared` once its figures "
        f"agree. Declared and silent: {sorted(declared - seen)} -- a "
        f"restatement was deleted, or the wording moved out of this scan's "
        f"reach."
    )


@pytest.mark.parametrize(
    ("rel", "pattern", "kind"),
    [(rel, pattern, kind) for rel, pattern, kind in _KIND_CLAIMS],
    ids=[f"{rel}:{kind}" for rel, _, kind in _KIND_CLAIMS],
)
def test_an_ordinal_that_claims_a_KIND_names_the_edit_that_has_it(
        rel, pattern, kind):
    """An ordinal reference is a pointer, and a pointer can go stale quietly.

    Each of these sentences says *the Nth edit* and then says what that edit
    IS.  The number is held to the item the list tags with that kind, and the
    kind is held to being unique -- because "the only one that ..." is the
    other half of every one of these sentences, and it stops being true the
    moment a second edit of the same kind is declared.
    """
    items = _items()
    of_kind = [n for n, k, _ in items if k == kind]
    assert len(of_kind) == 1, (
        f"{len(of_kind)} edits are tagged {kind!r} ({of_kind}), and "
        f"{rel} says one of them is the only one of its kind"
    )
    text = (REPO / rel).read_text(encoding="utf-8")
    m = pattern.search(text)
    assert m, (
        f"{rel} no longer carries the sentence naming the {kind} edit by "
        f"ordinal (pattern {pattern.pattern!r}). It is the sentence that "
        f"tells a reader WHICH edit the claim is about; a restatement that "
        f"drops it stops being checkable."
    )
    stated = _number(m.group("n"))
    assert stated == of_kind[0], (
        f"{rel} calls edit {m.group('n')!r} the {kind} one; the note's list "
        f"tags edit {of_kind[0]} with that kind"
    )


def test_this_module_states_no_figure_of_its_own():
    """The check that keeps this file out of its own subject matter.

    Every number here is parsed from the note.  A figure written into this
    docstring or into a failure message would be one more copy of exactly the
    thing this module exists because copies of.
    """
    rel = pathlib.Path(__file__).resolve().relative_to(REPO).as_posix()
    mine = [fig for fig in _figures() if fig.rel == rel]
    assert not mine, (
        f"{rel} states a figure for the ledger it measures: {mine}. Parse it "
        f"out of the note instead."
    )


def test_the_scan_is_not_reaching_nothing():
    """A floor, because a partition between two empty sets is an equality.

    Every leg above is an agreement, and agreements are free when nothing is
    found.  This is what says the regexes still match the prose they were
    written for: the note's own count, its markers, and a restatement in a
    file that is neither the note nor the module.
    """
    figures = _figures()
    kinds = {fig.kind for fig in figures}
    assert {"cardinal", "marker", "reference"} <= kinds, (
        f"the scan found only {sorted(kinds)}. It has stopped matching one "
        f"of the three shapes a figure for this ledger is written in, and "
        f"every agreement above is passing on what it no longer reads."
    )
    files = {fig.rel for fig in figures}
    assert GUARD_REL in files and len(files) >= 3, (
        f"the scan reached {sorted(files)}; the note itself and at least two "
        f"documents restate this ledger"
    )
