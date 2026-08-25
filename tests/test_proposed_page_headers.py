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
* **the commit leg needs the commits, and a shallow checkout does not have
  them.** Where ``git`` cannot reach a sha a status names,
  ``test_every_commit_a_status_paragraph_names_is_an_ancestor`` SKIPS, with
  a ``UserWarning`` naming git's own words and every page and sha that went
  unverified. It used to hard-``assert`` there instead, which made this file
  refuse the publish inside ``.github/workflows/release.yml``'s tagged-tree
  job -- a depth-1 checkout, so every sha older than the tag is out of
  reach. Skipping is the right answer for an environment; what it costs is
  real and is stated at the skip: in that run, nothing checks that a status
  names a commit that exists, that it is a commit rather than a tag, or
  that it is an ancestor of this tree. **In a FULL checkout an unreachable
  sha is still a hard failure** -- a shallow/foreign-repository triage runs
  before the skip, because "this page names a commit that never existed" is
  the sharpest thing this gate catches and it looks identical to a narrow
  clone until somebody asks git which one it is.

What the gate buys is the other half -- once somebody corrects a header,
the correction is load-bearing and cannot rot silently, and a BUILT claim
must carry a resolvable commit and a live, named test rather than a word.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import warnings

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


#: THE REASON THE OUT-OF-REACH SKIP CARRIES, AND IT IS A CONSTANT for exactly
#: the reason `tests/test_soundness_routing.py`'s `_GIT_LESS_SKIP` and
#: `tests/test_soundness_log_reach.py`'s `_TAG_OUT_OF_REACH_SKIP` are constants:
#: `tests/test_skip_inventory.py` excuses a skip only by an EXACT string typed
#: on its own disclosure surface, so a reason carrying git's stderr -- or the
#: sha it failed on -- could never be disclosed there, and its file would exit
#: 1 on the completeness half no matter what. Git's words and the shas go in
#: the WARNING beside the skip, which is the channel that can carry them.
_UNREACHABLE_SKIP = (
    "git cannot reach a commit a `proposed-*.md` status paragraph names, so "
    "that header's shipped-in claim cannot be decided here"
)


def _anchor() -> str | None:
    """This project's root commit, READ from the module that already owns it.

    `tests/test_soundness_routing.py` measured and documented `_ANCHOR` --
    including that an object filter drops blobs and trees but never commits,
    so a partial clone still holds it -- and a second copy of a sha here
    would be a second thing to get wrong. `None` when that module cannot be
    imported, which the one caller reads as "cannot tell", not as "absent".
    """
    try:
        from test_soundness_routing import _ANCHOR
    except Exception:  # noqa: BLE001 - a triage helper may not raise
        return None
    return _ANCHOR


def _the_environment_explains_it() -> str | None:
    """Why this tree could legitimately be missing history, or ``None`` when
    nothing about the environment explains it.

    **THE DIFFERENCE BETWEEN A NARROW CHECKOUT AND A BROKEN PAGE, AND
    WITHOUT IT THE FIX FOR THE RELEASE BLOCKER WOULD HAVE BEEN A HOLE.** If
    every unreachable sha became a skip, then a status paragraph naming a
    commit THAT NEVER EXISTED -- a typo, a sha copied from a fork, a commit
    rewritten out of this history -- would have become a warning and a green
    session in an ordinary full checkout. That is the exact claim this gate
    is here to hold, so it must still be RED where it can be decided, and
    the environment has to be asked before the page is excused.

    The triage is `tests/test_soundness_routing.py`'s, one object over, and
    it asks in this order:

    * `git rev-parse --is-shallow-repository` -- TRUE is the release job's
      own shape and is answered first, so this leg never depends on an
      import;
    * `git cat-file -e <root commit>^{commit}` -- a non-shallow repository
      holding any commit of this history holds the root too, so its absence
      says this is a vendored copy somebody ran `git init` in, a fork with
      rewritten history, or a `GIT_DIR` pointing somewhere else.

    An import failure answers "the environment explains it", which is the
    SAFE direction and the one the constant's own docstring argues for: a
    wrong or unreadable anchor turns a FAIL into a SKIP rather than
    inventing a red inside the one workflow whose mistakes cannot be
    unpublished.
    """
    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow is None:  # pragma: no cover - git vanished between two calls
        return "git stopped answering between two calls"
    if shallow.returncode != 0:  # pragma: no cover - env-dependent
        return (
            f"git cannot say whether this checkout is shallow: "
            f"{shallow.stderr.strip()[:120]}"
        )
    if shallow.stdout.strip() == "true":
        return "a shallow clone, whose history does not reach that commit"
    root = _anchor()
    if root is None:  # pragma: no cover - env-dependent
        return (
            "this tree cannot say whether it is this project's checkout at "
            "all, because tests/test_soundness_routing.py, which owns the "
            "root commit, did not import"
        )
    probe = _git("cat-file", "-e", f"{root}^{{commit}}")
    if probe is None:  # pragma: no cover - git vanished between two calls
        return "git stopped answering between two calls"
    if probe.returncode != 0:
        return (
            f"the git repository rooted here is not this project's: it does "
            f"not have {root[:12]}, this project's root commit, which a "
            f"non-shallow repository holding any commit of this history "
            f"holds too -- so this is a vendored copy somebody ran `git init` "
            f"in, a fork with rewritten history, or a GIT_DIR pointing at "
            f"something else"
        )
    return None


def _status_shas() -> list[tuple[str, str]]:
    """``(page name, sha)`` for every commit a **Status:** paragraph names.

    Hoisted out of the check below so that ``tests/test_skip_inventory.py``'s
    predicate can ask git about THE SAME SHAS this test asks about, in the
    same order, rather than retyping a list that would drift the first time a
    page cited a new commit. The predicate may compute the CONDITION; the
    REASON is a literal typed in both files, which is the thing a rule over
    there may not compute.
    """
    named: list[tuple[str, str]] = []
    for path in PAGES:
        text = path.read_text(encoding="utf-8")
        m = _STATUS_LINE.search(text)
        if not m:
            continue
        para = text[m.start():].split("\n\n", 1)[0]
        named += [(path.name, sha) for sha in sorted(set(_SHA.findall(para)))]
    return named


def _what_goes_unchecked(pairs: list[tuple[str, str]], said: str) -> str:
    """The warning that stands beside the skip: git's own words, and the
    claim that is dropped.

    A skip REASON has to be one exact literal, because that is the only
    channel ``tests/test_skip_inventory.py`` can excuse; this is where the
    detail lives instead. It names every page and sha that went unverified,
    because *a status paragraph naming a sha nobody verified* is precisely
    the claim this gate exists to hold, and a reader who cannot see WHICH
    ones cannot act on it.
    """
    listed = ", ".join(f"{page}:`{sha}`" for page, sha in pairs)
    return (
        f"{len(pairs)} commit(s) named in a `proposed-*.md` **Status:** "
        f"paragraph are not in this tree's object store, so THE HEADERS "
        f"THAT NAME THEM GO UNVERIFIED HERE: {listed}. Three checks are "
        f"dropped for each one, and nothing else in this suite makes them: "
        f"that the sha RESOLVES at all, so a status cannot name a commit "
        f"that never existed or one that only ever lived in a fork; that it "
        f"resolves to a COMMIT and not a tag, a tree or a blob; and that it "
        f"is an ANCESTOR OF HEAD, so \"shipped in `<sha>`\" cannot describe "
        f"another branch's tree or a commit that was rewritten away. A page "
        f"could carry any of those three defects and this run would report "
        f"nothing -- and `docs/README.md` tells a reader to trust these "
        f"headers. {said} THE COMMONEST CAUSE IS A SHALLOW CHECKOUT: "
        f"`actions/checkout@v4` with no `fetch-depth` fetches the triggering "
        f"ref at depth 1, so a tag build holds exactly one commit and every "
        f"sha older than the tag is out of reach -- which is the shape "
        f"`.github/workflows/release.yml`'s job `the suite, on the tagged "
        f"tree` runs in, and that job is refusal point #1 between a tag and "
        f"PyPI. `git fetch --unshallow`, or `fetch-depth: 0` on that "
        f"checkout, restores the check."
    )


def test_every_commit_a_status_paragraph_names_is_an_ancestor():
    """"Shipped in ``<sha>``" is checkable, so it is checked WHERE IT CAN BE.

    Scope: the **Status:** paragraph only. A sha quoted further down a
    page is part of the argument -- a measurement's as-of, a commit that
    broke something -- and may legitimately name history this branch does
    not contain. The status paragraph is the sentence a reader acts on.

    **UNTIL 2026-08-25 THIS WAS A RELEASE BLOCKER, AND A PRE-EXISTING ONE.**
    It guarded with exactly two conditions -- is `git` on `PATH`, and does
    `git rev-parse --verify HEAD` succeed -- and **in a shallow clone BOTH
    PASS**: git is present and `HEAD` resolves to the one commit that was
    fetched. Control then reached a hard `assert` on `git cat-file -t <sha>`,
    which fails for every sha outside a depth-1 history. Driven here, in a
    sandbox built the way the release job builds its tree (`git init`; one
    `git fetch --depth=1` of a single ref into a tag ref; `git checkout` of
    that tag): `1 failed`, on
    ``proposed-declaration-dtype-check.md's status names `89413c2`, which
    this tree's git cannot resolve: fatal: Not a valid object name 89413c2``.
    That is `.github/workflows/release.yml`'s job `the suite, on the tagged
    tree`, which checks out with `persist-credentials: false` alone -- no
    `fetch-depth`, no `fetch-tags` -- and is refusal point #1 between a tag
    and PyPI. A tree nobody can fix after the tag is cut refused the publish
    over an environment, in the name of an integrity claim it could not have
    decided either way.

    So the unreachable case is a DISCLOSED SKIP now, with a `UserWarning`
    beside it carrying git's exit code, git's own words, and the pages and
    shas that went unverified -- the shape
    `tests/test_soundness_routing.py` and
    `tests/test_soundness_log_reach.py` already use one object over. It does
    not silently pass: see :func:`_what_goes_unchecked` for the words, which
    say that a page could name a sha that never existed, one that is a tag
    rather than a commit, or one from another branch, and this run would
    report none of it.

    **AND THE CONTRADICTED SKIP IS GONE.** With `.git` removed and git still
    on `PATH` -- an unpacked sdist -- the `HEAD` probe failed and this test
    skipped as `"needs git"`, blaming a tool that was right there.
    `tests/test_skip_inventory.py` calls that a CONTRADICTED skip and exits
    1 on it, which is why the sdist lane read `1 failed` for this file and
    no other reason. Each of the four states this file can meet now carries
    the reason that is true in it: git off `PATH`; `.git` absent; a `git
    init` with no commit at all; and a checkout whose object store does not
    reach a named sha. The predicate over there asks git the same questions
    in the same order, so the reasons partition the states.

    **THE DECIDABLE ONES ARE STILL DECIDED.** A shallow tree that happens to
    hold one of these shas gets it checked, and a page naming a sha that IS
    here and does not hold up is red here exactly as anywhere; only the
    undecidable remainder becomes a skip, and the assertion runs FIRST so a
    real defect can never be reported as an environment.

    **AND AN UNREACHABLE SHA IS NOT AUTOMATICALLY AN ENVIRONMENT.** The
    obvious fix -- skip whenever `git cat-file` fails -- would have turned
    the sharpest defect this gate catches into a warning: a status naming a
    commit THAT NEVER EXISTED, a typo or a sha copied out of a fork, is
    unreachable in a perfectly ordinary full checkout too. So
    :func:`_the_environment_explains_it` is asked first, and where nothing
    about the tree explains the miss this FAILS and says which page. Driven
    both ways; see that function.
    """
    named = _status_shas()
    assert named, (
        "no proposed-*.md status paragraph names a commit, so this gate "
        "measured nothing"
    )
    if shutil.which("git") is None:  # pragma: no cover - env-dependent
        # Byte-for-byte the reason `tests/test_skip_inventory.py` declares
        # legitimate for git being off `PATH`, and for nothing else.
        pytest.skip("needs git")
    if not (REPO / ".git").exists():  # pragma: no cover - env-dependent
        # A worktree's `.git` is a FILE, hence `exists()` and not `is_dir()`.
        # THIS STATE USED TO SKIP AS "needs git" WITH GIT RIGHT THERE ON
        # `PATH`: the HEAD probe below failed and the reason blamed the
        # wrong thing. Same literal as the sibling files' sdist skip.
        pytest.skip("not a git checkout (an unpacked sdist, say)")

    head = _git("rev-parse", "--verify", "HEAD")
    if head is None or head.returncode != 0:  # pragma: no cover - env-dependent
        # `.git` exists and there is no commit in it: a `git init` nobody
        # has committed in, an sdist somebody ran `git init` inside. Nothing
        # here is decidable -- there is no HEAD for anything to be an
        # ancestor OF -- so every named sha is unverified, and it says so.
        diagnosis = (
            "git stopped answering between two calls."
            if head is None else
            f"`git rev-parse --verify HEAD` exited {head.returncode} and "
            f"said: "
            f"{head.stderr.strip() or '(nothing; git printed no diagnostic)'}"
            f" -- so this tree has no HEAD commit for anything to be an "
            f"ancestor OF, which is what a `git init` nobody has committed "
            f"in looks like."
        )
        warnings.warn(_what_goes_unchecked(named, diagnosis), stacklevel=2)
        pytest.skip(_UNREACHABLE_SKIP)

    unreachable: list[tuple[str, str]] = []
    words: list[str] = []
    wrong: list[str] = []
    for page, sha in named:
        kind = _git("cat-file", "-t", sha)
        if kind is None or kind.returncode != 0:
            unreachable.append((page, sha))
            words.append(
                "git stopped answering between two calls."
                if kind is None else
                f"`git cat-file -t {sha}` exited {kind.returncode} and said: "
                f"{kind.stderr.strip() or '(nothing)'}."
            )
            continue
        if kind.stdout.strip() != "commit":
            wrong.append(
                f"{page}'s status names `{sha}`, which resolves to a "
                f"{kind.stdout.strip()}, not a commit"
            )
            continue
        anc = _git("merge-base", "--is-ancestor", sha, "HEAD")
        if anc is None or anc.returncode != 0:
            wrong.append(
                f"{page} says its behaviour shipped in `{sha}`, which is NOT "
                f"an ancestor of HEAD. Either the page is describing another "
                f"branch's tree, or the commit was rewritten."
            )
    assert not wrong, (
        "these `proposed-*.md` status paragraphs name a commit this tree HAS "
        "and that does not hold up:\n  " + "\n  ".join(wrong)
    )
    if unreachable:  # pragma: no cover - env-dependent
        # THE ENVIRONMENT IS ASKED BEFORE THE PAGE IS EXCUSED. An
        # unreachable sha is a skip only where something about this tree
        # explains it; in a full, non-shallow checkout of this project it is
        # a page naming a commit that is not there, and that is the whole
        # claim this gate holds.
        why = _the_environment_explains_it()
        if why is None:
            root = _anchor()
            pytest.fail(
                f"these `proposed-*.md` status paragraphs name commits that "
                f"are NOT IN THIS HISTORY, in a tree that IS a complete "
                f"checkout of this project -- BOTH conditions asked just now "
                f"rather than assumed: `git rev-parse "
                f"--is-shallow-repository` says false, and this repository "
                f"holds {'' if root is None else root[:12] + ', '}this "
                f"project's root commit. So this is neither the shallow CI "
                f"clone nor the unpacked sdist nor the foreign repository "
                f"this check skips for: what is wrong is the PAGE naming a "
                f"commit that is not here -- a typo, a sha from a fork, or a "
                f"commit rewritten out of this history. "
                + ", ".join(f"{page}:`{sha}`" for page, sha in unreachable)
                + ". " + " ".join(words)
            )
        warnings.warn(
            _what_goes_unchecked(unreachable, " ".join(words) + f" ({why}.)"),
            stacklevel=2,
        )
        pytest.skip(_UNREACHABLE_SKIP)


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
