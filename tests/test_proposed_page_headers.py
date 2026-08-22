# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A `proposed-*.md` header is a claim about the tree, so the tree checks it.

``docs/README.md`` tells a reader, in the index and in the only sentence
that says what these pages are, to trust exactly these headers:

    `proposed-*.md` are design proposals in the state their headers
    declare; they are records, not user guides.

**Five of the seven headers were false when this file was written.**
``proposed-decline-messages.md`` said ``PROPOSED, NOT APPLIED`` with all
five of its numbered sections applied; ``proposed-div-straddle-decline.md``
said the same with the decline shipped at ``32c6c56`` and pinned by a test;
``proposed-int-literal-convert.md`` and ``proposed-solver-selection.md``
said ``PROPOSED, NOT BUILT`` about behaviour built at ``cbb1d60``; and
``proposed-tier-clause.md`` said ``PROPOSED`` about a clause shipped at
``9f9b8b7``. The sixth, ``proposed-declaration-dtype-check.md``, had
already met the identical defect, corrected it, and NAMED it -- *"a claim
divergence on a DOCUMENT, which is a new surface for that class"*. One
instance was corrected; five siblings carried it uncorrected, and nothing
in the suite read a header.

**So this is the gate the sixth page's correction earned and did not
get.** It does not try to decide whether a proposal is built -- no test
can -- but every part of a status header that IS mechanically checkable
is checked here:

* the title line ends in a status drawn from a **closed set**, so a page
  cannot invent an unreadable one or drop the status altogether;
* every commit a status names **resolves and is an ancestor of HEAD**, so
  "shipped in ``<sha>``" cannot name a commit this tree does not have;
* every ``tests/*.py`` a page names, anywhere on it, **exists**, so a page
  cannot claim a pin that was deleted;
* a page whose status says BUILT or APPLIED **names at least one existing
  ``tests/…py`` in its STATUS BLOCK**, which is the whole difference
  between a corrected header and an asserted one;
* ``docs/README.md``'s sentence about these pages names the token set
  actually in use, so the index cannot describe a vocabulary the pages
  stopped speaking.

WHAT THIS DOES NOT CATCH, stated because a gate whose scope is guessed at
is worse than none:

* a page that says ``PROPOSED, NOT BUILT`` about behaviour that IS built
  passes here. That direction needs somebody to notice the code, and it is
  exactly the direction that failed five times.
* **the pin is checked for PRESENCE and EXISTENCE, never for relevance.**
  "Names a pinning test" means "the status block names a ``tests/…py``
  that is in this tree" and nothing more. Measured: replacing
  ``proposed-tier-clause.md``'s ``tests/test_tier_clause.py`` with the
  unrelated ``tests/test_doc_examples.py`` leaves this file green, and
  reciprocity would not catch it either -- ``test_doc_examples.py`` names
  that page back, in a ``BLIND_SPOT`` entry. Whether the named test
  actually holds the page's claim down is a reading, and no assertion here
  makes it. The search WAS over the whole page, so a ``tests/…py``
  mentioned four hundred lines below the header satisfied it; that half is
  fixed, and the remaining half is stated rather than implied.

What the gate buys is the other half -- once somebody corrects a header,
the correction is load-bearing and cannot rot silently, and a BUILT claim
must carry a resolvable commit and a live, named test rather than a word.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# The closed set. A status is the text after the last em-dash of the title
# line, with markdown emphasis and any parenthetical stripped.
#
# Deliberately small: every token here is a distinct thing to know about a
# proposal, and a page that needs a sixth should add it in this list, in
# the open, rather than writing prose the index does not describe.
STATUSES = {
    "BUILT",           # the proposal's mechanism is in the tree
    "APPLIED",         # the proposal's TEXT shipped (the message pages)
    "PARTLY BUILT",    # some rows of the change table landed and some did not
    "PROPOSED",        # written, not built
    "PROPOSED, NOT BUILT",
    "PROPOSED, NOT APPLIED",
    "MEASURED",        # a study that recommends something, not a proposal
}

# The statuses that assert something shipped, and therefore must carry a
# pinning test. `PARTLY BUILT` is here on purpose: the part that DID land
# is the part a reader will use.
SHIPPED = {"BUILT", "APPLIED", "PARTLY BUILT"}

_TITLE = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.M)
_SHA = re.compile(r"`([0-9a-f]{7,40})`")
_TESTFILE = re.compile(r"`(tests/[A-Za-z0-9_./-]+\.py)")
_STATUS_LINE = re.compile(r"^\*\*Status:\s*(?P<status>[^.*]+)", re.M)
_HEADING = re.compile(r"^#{1,6} ", re.M)


def _status_block(text: str) -> str | None:
    """The status block: the **Status:** line to the next heading, or
    ``None`` when there is no status line -- or no heading after it.

    Wider than the sha gate's single paragraph, because every shipped page
    in this tree puts the status in one paragraph and the pin that supports
    it in the next; narrower than the page, because a ``tests/…py``
    mentioned in an argument four hundred lines down is not a header making
    a claim.

    **THE SECOND ``None`` IS THE POINT, AND IT WAS A HOLE.** This ran to the
    next heading *or to the end of the file if there was none*, so on a page
    with no heading below its status paragraph the "block" was the whole
    page and the narrowing silently reverted -- the exact defect ``6e9aa22``
    was written to close, reachable again with no diagnostic. Measured on
    ``6387a34`` against ``docs/proposed-tier-clause.md``: remove the pin from
    the status paragraph, demote every heading below it, append 400 filler
    lines and put ``tests/test_tier_clause.py`` at the very bottom -- **18
    passed**. The docstring here said the block was *"narrower than the
    page"*; that was true of the pages in the tree and was not a property of
    this function.

    So the narrowing is now a property of the MECHANISM: a page that does
    not bound its own status block does not get a status block, and
    :func:`test_every_status_block_is_bounded_by_a_heading` is the gate that
    says so in those words rather than leaving the shipped-pin gate to
    report it as a missing pin. No block size is written down here;
    :func:`test_every_status_block_is_bounded_by_a_heading` measures them.
    """
    m = _STATUS_LINE.search(text)
    if not m:
        return None
    h = _HEADING.search(text, m.end())
    if not h:
        return None
    return text[m.start(): h.start()]


def _pages() -> list[pathlib.Path]:
    """Every ``proposed-*.md`` under ``docs/``, at any depth.

    ``rglob``, not ``glob``: a non-recursive population is a ceiling nobody
    can see, and one subdirectory is all it takes to put a page with an
    unchecked header outside every gate in this file. There are no
    subdirectories under ``docs/`` today, so this changes nothing that is
    in the tree -- which is exactly when it is cheap to fix.
    """
    return sorted(DOCS.rglob("proposed-*.md"))


PAGES = _pages()


def test_there_are_proposed_pages_to_check():
    """The anti-vacuity control. Every parametrised test below is empty and
    green if the glob stops matching, which is how a gate over a file set
    goes quietly inert."""
    assert len(PAGES) >= 7, [p.name for p in PAGES]


def _title_of(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = _TITLE.search(text)
    assert m, f"{path.name} has no `# ` title line"
    return m.group("title")


def _status_tail(title: str) -> str:
    """Everything after the LAST em-dash of the title, undecorated.

    Titles carry em-dashes inside them (``int64→float64`` from an integer
    literal — **BUILT**), so the split is on the last one, and the
    parenthetical sha that may follow is not part of the token.
    """
    assert "—" in title, (
        f"title {title!r} carries no status: a proposed-*.md title ends "
        f"'— <STATUS>', with STATUS one of {sorted(STATUSES)}"
    )
    tail = title.rsplit("—", 1)[1]
    tail = tail.split("(")[0]                    # drop '(`32c6c56`)'
    return tail.replace("*", "").replace("`", "").strip()


def _status_of(title: str) -> str | None:
    """The recognised status token the title opens with, or ``None``.

    LONGEST match, so ``PROPOSED, NOT BUILT`` is not read as ``PROPOSED``
    with a stray clause. A trailing qualification is allowed and ignored —
    ``MEASURED, and the answer is mostly NO`` is a status plus the answer,
    and demanding the bare token would make a page less informative to
    satisfy a parser.
    """
    tail = _status_tail(title)
    for token in sorted(STATUSES, key=len, reverse=True):
        if tail == token or tail.startswith(token + ",") or \
                tail.startswith(token + " "):
            return token
    return None


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_the_title_declares_a_status_from_the_closed_set(path):
    """``docs/README.md`` sends a reader to these headers. A header a
    reader cannot classify sends them nowhere."""
    tail = _status_tail(_title_of(path))
    assert _status_of(_title_of(path)) is not None, (
        f"{path.name}'s title declares status {tail!r}, which opens with "
        f"none of {sorted(STATUSES)}. Add the token to STATUSES "
        f"deliberately, or use an existing one -- docs/README.md describes "
        f"this vocabulary to a reader and cannot describe one the pages do "
        f"not speak."
    )


def test_every_status_block_is_bounded_by_a_heading():
    """A status block that runs to the end of the file is the whole page.

    :func:`_status_block` used to fall back to end-of-file, so a page could
    lose its narrowing by having no heading below the status paragraph, and
    nothing said so -- see that function for the measurement (18 passed,
    with the pin 400 lines below the header it was satisfying).

    The block sizes are MEASURED and reported here rather than written into
    a docstring: the last hand-typed figure beside this mechanism said
    "each of the five shipped pages", and six of them are shipped. It was
    wrong the moment it was typed, in the commit whose subject was deleting
    hand-typed numerals.
    """
    unbounded = []
    sizes = {}
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        m = _STATUS_LINE.search(text)
        if not m:
            continue                      # no status paragraph to bound
        block = _status_block(text)
        if block is None:
            unbounded.append(str(path.relative_to(REPO)))
            continue
        sizes[path] = block.count("\n")
    assert sizes, (
        "no proposed-*.md has a bounded status block, so every gate in this "
        "file that reads one measured nothing"
    )
    assert not unbounded, (
        f"these pages carry a **Status:** paragraph with NO heading below "
        f"it: {unbounded}. The status block would be the whole page, so a "
        f"`tests/...py` named anywhere on it -- four hundred lines under the "
        f"header, in an argument -- would satisfy the header at the top. "
        f"Put the status paragraph above a heading."
    )
    whole = {str(p.relative_to(REPO)): n for p, n in sizes.items()
             if n >= p.read_text(encoding="utf-8").count("\n")}
    assert not whole, (
        f"these status blocks are the whole page: {whole}. The narrowing is "
        f"what this gate is."
    )


def test_a_page_that_says_it_shipped_names_a_test_that_pins_it():
    """The difference between a corrected header and an asserted one.

    A page that flips to BUILT with no pin is a header claiming what the
    old header claimed, in the other direction — and the old headers were
    wrong from their shipping commits (``9f9b8b7`` 2026-07-30, ``32c6c56``
    and ``cbb1d60`` 2026-08-12) until this gate, with the suite green
    throughout.

    Aggregated rather than parametrised, so the pages that do not claim to
    have shipped produce no skip: a per-page skip would put five entries
    into ``tests/test_skip_inventory.py`` that say nothing but "this page
    is still a proposal".

    **SCOPE, and it is narrower than the sentence above.** This searches
    the STATUS BLOCK -- the ``**Status:**`` line to the next heading --
    and asks only whether it names a ``tests/…py``. It used to search the
    whole page, so a test named anywhere on it satisfied a header at the
    top. It still cannot tell whether the named test pins THIS page: the
    module docstring's "WHAT THIS DOES NOT CATCH" records the measurement,
    including why reciprocity would not have caught it either.
    """
    shipped = {
        p: _status_of(_title_of(p)) for p in PAGES
        if _status_of(_title_of(p)) in SHIPPED
    }
    assert shipped, (
        "no proposed-*.md claims to have shipped, so this gate measured "
        "nothing. Five of them did when it was written."
    )
    # Keyed by PATH, not by `p.name`: `_pages()` is an rglob now, and
    # `DOCS / p.name` does not resolve for a page in a subdirectory --
    # a nested BUILT page would have raised FileNotFoundError out of this
    # gate rather than being checked by it.
    unpinned = {
        str(p.relative_to(REPO)): status for p, status in shipped.items()
        if not _TESTFILE.findall(
            _status_block(p.read_text(encoding="utf-8")) or ""
        )
    }
    assert not unpinned, (
        f"these pages declare a shipped status and name no `tests/...py` "
        f"in their STATUS BLOCK: {unpinned}. A status that shipped is a "
        f"claim about running code; name what holds it down where the "
        f"claim is made. (`test_every_test_file_a_proposed_page_names_exists`"
        f" is what checks the file is really there, and "
        f"`test_every_status_block_is_bounded_by_a_heading` is what reports "
        f"a page whose status block could not be narrowed at all.)"
    )


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_every_test_file_a_proposed_page_names_exists(path):
    """A pin that was deleted is worse than no pin: the page still reads
    as gated."""
    named = sorted(set(_TESTFILE.findall(path.read_text(encoding="utf-8"))))
    missing = [n for n in named if not (REPO / n).exists()]
    assert not missing, (
        f"{path.name} names test files that are not in this tree: {missing}"
    )


def _git(*args: str):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def test_every_commit_a_status_paragraph_names_is_an_ancestor():
    """"Shipped in ``<sha>``" is checkable, so it is checked.

    Scope: the **Status:** paragraph only. A sha quoted further down a
    page is part of the argument -- a measurement's as-of, a commit that
    broke something -- and may legitimately name history this branch does
    not contain. The status paragraph is the sentence a reader acts on.
    """
    named: list[tuple[str, str]] = []
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        m = _STATUS_LINE.search(text)
        if not m:
            continue
        para = text[m.start():].split("\n\n", 1)[0]
        named += [(path.name, sha) for sha in sorted(set(_SHA.findall(para)))]
    assert named, (
        "no proposed-*.md status paragraph names a commit, so this gate "
        "measured nothing"
    )
    if shutil.which("git") is None:  # pragma: no cover - env-dependent
        pytest.skip("needs git")
    probe = _git("rev-parse", "--verify", "HEAD")
    if probe is None or probe.returncode != 0:  # pragma: no cover
        pytest.skip("needs git")
    for page, sha in named:
        kind = _git("cat-file", "-t", sha)
        assert kind is not None and kind.returncode == 0, (
            f"{page}'s status names `{sha}`, which this tree's git cannot "
            f"resolve: {'' if kind is None else kind.stderr.strip()}"
        )
        assert kind.stdout.strip() == "commit", (
            f"{page}'s status names `{sha}`, which resolves to a "
            f"{kind.stdout.strip()}, not a commit"
        )
        anc = _git("merge-base", "--is-ancestor", sha, "HEAD")
        assert anc is not None and anc.returncode == 0, (
            f"{page} says its behaviour shipped in `{sha}`, which is NOT an "
            f"ancestor of HEAD. Either the page is describing another "
            f"branch's tree, or the commit was rewritten."
        )


# ------------------------------------------------------- the index sentence

# From the sentence that opens the description to the next heading. Not to
# the next blank line: the description is two paragraphs, and a token that
# fell into the second one would read as unmentioned.
_INDEX_SENTENCE = re.compile(
    r"`proposed-\*\.md` are design proposals.*?(?=\n#|\Z)", re.S
)


def test_the_index_describes_the_vocabulary_the_pages_actually_speak():
    """``docs/README.md`` is the only place in the shipped tree that says
    what these pages are, and it says: trust the headers.

    It named ``BUILT`` and "proposed and unbuilt" while five pages carried
    neither -- so the sentence a reader is given to classify a header did
    not cover the headers in front of them. This derives the check from
    the pages rather than from a second hand-written list: every status
    token in use must appear in the index sentence.
    """
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    m = _INDEX_SENTENCE.search(readme)
    assert m, (
        "docs/README.md no longer carries the sentence describing "
        "`proposed-*.md`; it is the only route a reader has to these pages"
    )
    # whitespace-normalised: a token the paragraph wraps across a line
    # break is mentioned, and a check that says otherwise is measuring the
    # line width
    sentence = re.sub(r"\s+", " ", m.group(0))
    # `None` for a title whose status is not in the closed set. Dropped
    # rather than compared: `test_the_title_declares_a_status_from_the_
    # closed_set` is the gate that reports an invented token, and it names
    # the page and the token. Letting the None through put a
    # `TypeError: 'in <string>' requires string as left operand` beside it
    # -- a second red line that says nothing the first did not, from the
    # test that is supposed to be about the INDEX SENTENCE. Measured by
    # retitling `proposed-unit-mechanism.md` `— SPECULATED`.
    in_use = {s for s in (_status_of(_title_of(p)) for p in PAGES) if s}
    unmentioned = sorted(t for t in in_use if t not in sentence)
    assert not unmentioned, (
        f"docs/README.md's `proposed-*.md` sentence tells a reader to read "
        f"the headers, and does not mention the statuses those headers "
        f"actually carry: {unmentioned}.\nThe sentence reads:\n{sentence}"
    )
