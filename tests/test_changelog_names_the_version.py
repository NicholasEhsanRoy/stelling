# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""`CHANGELOG.md`'s newest heading and `stelling.__version__`, tied.

Nothing tied them. They agreed when this file was written — `0.2.0.dev0` in
`src/stelling/__init__.py`, `## 0.2.0 — unreleased` at the top of the
changelog — so nothing shipped wrong, and that was the whole reason to write
it then rather than after it did.

**BOTH OF THOSE READINGS HAVE SINCE MOVED, AND THEY WERE WRITTEN IN THE
PRESENT TENSE.** The release bump made `__version__` `0.2.0` and put a date
on the heading — the OTHER of the two states this file checks — so two
sentences that were true on the branch that added them were false on the
merge that carried them, and no branch could have seen it: this file exists
on one side of that merge only, so the merge was textually clean. Nothing in
the tree could catch it either, because `tests/test_prose_hygiene.py` scopes
itself to shipped prose and `tests/` is outside it — while `tests/` is 181 of
the sdist's 379 members. They are dated records now rather than live claims,
which is the only form of illustration that cannot rot: it is the CHECK below
that carries the argument, and the check exercises both states.

It is the same unguarded coupling the tag gate in
`.github/workflows/release.yml` exists to close, one file over: **a value
maintained by hand against a value read from source**, with no check between
them. `pyproject.toml` already reads the version from
`src/stelling/__init__.py` (`[tool.hatch.version]`), so the wheel, the sdist
and the tag cannot disagree with the module. The changelog can, and a release
whose changelog heading names the previous version is a release note that
describes the wrong release.

**THE VERSIONS NEED NOT BE SPELLED THE SAME AND THE COUPLING IS STILL
EXACT.** At that commit `__version__` was `0.2.0.dev0` and the heading said
`0.2.0`, because a heading names the RELEASE and `__version__` names the
BUILD. PEP 440 makes that
relation total rather than a matter of taste: `0.2.0.dev0` is a development
build *of* `0.2.0`, its release segment is `(0, 2, 0)`, and `is_prerelease`
answers whether the build has shipped. So the assertion is on the release
segment, in both directions:

* the newest heading's version is the release `__version__` belongs to;
* and the two agree about whether it has HAPPENED — a `.devN` / `aN` / `bN` /
  `rcN` build's heading must say *unreleased*, and a final version's heading
  must carry a date. Without that second half the check passes forever on a
  changelog frozen at *"0.2.0 — unreleased"* through the 0.2.0 release
  itself, which is the same defect one word to the left.

**THIS DOES NOT GATE THE DATE, AND THAT IS DELIBERATE.** At release the
heading's date should equal the tag's date. The tag does not exist while this
suite runs — it is created by the release workflow, after this suite has to
have passed — so the only date a check here could compare against is
`datetime.date.today()`, i.e. **the developer's clock**. That is the
environment-dependence class that produced four of five recent CI reds on
`main`, arriving in a new place, and it would red every checkout whose author
wrote the heading yesterday. There is no version of it that is right.

**THE DATE CHECK IS IN `release.yml`, WHERE THE TAG EXISTS.** It is the step
*"the tag and the changelog heading must agree"*, in the `build` job, before
anything is uploaded — the fifteenth refusal point between a tag and PyPI, and
that file's header numbers and argues it. This check does not have the tag;
that one does.

**AND UNTIL THE COMMIT THAT ADDED IT, THIS PARAGRAPH WAS A RECOMMENDATION
STANDING AGAINST A FILE THAT DID NOT CARRY IT.** What stood here, verbatim:
*"THE DATE CHECK BELONGS IN `release.yml`, WHERE THE TAG EXISTS, and this file
does not own that workflow. The recommendation, written down so it is routable
rather than remembered: in the release job, after the tag is resolved, assert
that the newest `## <version> — <date>` heading in `CHANGELOG.md` carries the
tag's own committer date — `git log -1 --format=%cd --date=short
"$GITHUB_REF_NAME"` — and that its version equals the tag minus the leading
`v`. That check has the tag; this one cannot."* Every word of that was a
present-tense claim about another file, `.github/workflows/release.yml` did
not have the check, and **nothing in this repository read the sentence** — so
it could not go red, only stale. Measured: this module arrived at `0e79ede`
(committed 2026-08-25T00:32:48+02:00) and `git rev-list --count
0e79ede..9b5b496` is 47, so the recommendation stood over 47 commits
describing a gate that was never written. Prose asserting a check that does
not exist is the same defect as the missing check.

**WHAT HOLDS IT NOW, so it cannot go stale the way it just did.**
:func:`test_the_routed_date_check_is_in_this_workflow` below reads
`.github/workflows/release.yml` and refuses a tree in which the routed step is
gone, renamed, moved into `publish`, moved into a job the upload does not wait
for, or no longer reading the DOCUMENT, the TAG and the DATE this paragraph
names — and refuses one in which it has stopped carrying an `exit 1` at all,
because a step that reads all three and never refuses is a routing target in
name only. What it can and cannot see is in its own docstring, because a
text reader over a workflow file that does not declare its blindness is the
defect `tests/_lanes.py` spends a section on. The step's BEHAVIOUR is a
separate question and a separate instrument:
`tests/test_release_gates.py::test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry`
extracts that step's own body and drives it against planted trees carrying
real annotated tags, red and green.

**AND THE GATE READS A DIFFERENT DATE FROM THE ONE THIS PARAGRAPH ASKED FOR**,
recorded here and not only there. The recommendation named the tag's COMMITTER
date; the gate reads the TAG OBJECT's tagger date —
`git for-each-ref --format='%(taggerdate:short)' refs/tags/"$GITHUB_REF_NAME"`.
Measured 2026-08-28 at `9b5b496`, the two readings AGREE on both of this
repository's real tags — `v0.1.0` 2026-08-12 and `v0.2.0` 2026-08-25, both
ways — so the data cannot discriminate between them and the choice was made on
the argument, which is written beside the step: a heading's date is a claim
about when the RELEASE happened, and a release cut later than its last commit
has a committer date that is not that date, so the committer reading refuses a
correct heading and the only way past it is to write a false one. The
lightweight-tag case, which carries no tagger date at all, is a named refusal
there rather than a fallback.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import pytest

import stelling

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# THE WORKFLOW EXTRACTOR, BORROWED AND NOT RE-WRITTEN. `_step_lines` /
# `_step_body` / `_release_text` are the readers `tests/test_release_gates.py`
# already applies to `release.yml`, with the carriage-return normalisation
# that module had to learn the hard way and an anti-vacuity assertion on the
# extracted block. A second implementation of one file-reading rule is the
# defect `tests/test_workflows_make_no_silent_pick.py` refuses one directory
# over, and it would be a worse one here: two readers could disagree about
# what "the step" is, and this file's whole claim is that the step the OTHER
# module drives is the step this module routes to.
from test_release_gates import _code_lines, _release_text, _step_body  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"

#: The step `release.yml` carries the routed check in. Named ONCE, here, and
#: used by the check below: the name is what `_step_body` addresses the step
#: by, so a rename must move one literal and not two.
ROUTED_STEP = "the tag and the changelog heading must agree"

#: A release heading. `## <version> — <rest>`, em dash, which is the shape
#: every heading in the file uses. `rest` is either `unreleased` or a date,
#: and which one it is carries half the claim.
#:
#: **THE SEPARATORS WERE `\s` AND THAT IS A THIRD WHITESPACE ALPHABET.** They
#: read `^##\s+…\s+—\s+…\s*$`, and Python's `\s` is Unicode-aware: twenty-nine
#: characters where CommonMark's heading grammar knows two. The bash twin's
#: `[[:space:]]` is a fourth (six). Driven by an auditor over every character
#: `str.isspace()` finds: **26 of 28 were read by this twin as a heading the
#: renderer does not put there**, and 23 more made a heading parse here that
#: parses nowhere else — a NBSP or a U+2028 between the version and the em
#: dash. The correct class was three lines away the whole time, in
#: :data:`_ATX_LINE`. It is `[ \t]` in both readers now, and the whole
#: alphabet is swept rather than argued: see `SEPARATOR_FORMS` in
#: `tools/changelog_renderer_corpus.py` and
#: `tests/test_release_gates.py::test_the_whitespace_alphabet_is_swept_in_every_position`.
_HEADING = re.compile(
    r"^##[ \t]+(?P<version>[0-9][0-9A-Za-z._+!-]*)[ \t]+—[ \t]+(?P<rest>.+?)"
    r"[ \t]*$",
    re.M,
)

#: PEP 440's pre-release and development spellings, which are what separates
#: "this build is of 0.2.0" from "0.2.0 has happened".
_UNSHIPPED = re.compile(r"(?:\.dev[0-9]+|[ab][0-9]+|rc[0-9]+)$")

#: The word a heading uses for a release that has not happened.
_UNRELEASED = "unreleased"

#: **`\d` HERE WAS THE LAST OF THE BORROWED ALPHABETS IN THIS MODULE**, and it
#: was still `\d` in the commit that replaced `\d` with `[0-9]` twenty lines
#: up for exactly this reason. Python's `\d` is Unicode-aware, so
#: `## 0.2.0 — <ARABIC-INDIC 2>026-08-25` satisfied "a released version's
#: heading carries a date" here while the bash gate — which compares the
#: heading's date to the tag's by STRING EQUALITY — would refuse it, so no
#: release could ever have shipped one. A repair applied to two of the four
#: places one rule lives is a repair nobody can check; driven at
#: :func:`test_both_halves_of_the_coupling_are_driven`.
_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def release_segment(version: str) -> str:
    """`0.2.0.dev0` -> `0.2.0`; `0.2.0` -> `0.2.0`.

    The release a build belongs to, by string surgery on PEP 440's own
    suffixes rather than by importing `packaging`: this file must run in the
    zero-dep lane, whose whole point is that the core imports nothing, and a
    version comparison that needs a dependency is a version comparison the
    zero-dep job cannot make.
    """
    return _UNSHIPPED.sub("", version)


def is_unshipped(version: str) -> bool:
    """Whether this version names a build that has not been released."""
    return bool(_UNSHIPPED.search(version))


#: A release heading's POSITION, independent of whether it parses — applied to
#: ONE LINE at a time, never to the document.
#:
#: **`^##\s` WAS BOTH TOO NARROW AND TOO WIDE, AND AN AUDITOR DROVE BOTH.** It
#: read `^##\s.*$` over the whole text with `re.M`, which:
#:
#: * **missed a heading indented by one to three spaces**, which CommonMark
#:   renders as an ordinary `<h2>`. A newer heading written that way was
#:   stepped past to an older one — the exact defect this function was added
#:   to close, one spelling out;
#: * **took a `##` line inside an HTML comment or a fenced block**, neither of
#:   which a renderer treats as a heading. `CHANGELOG.md` opens with an HTML
#:   comment, so that route is one edit away at all times;
#: * and `\s` **matches a newline**, so `##` alone on a line fused with the
#:   line below it and parsed as though they were one heading.
#:
#: Four leading spaces is an indented code block and is deliberately NOT a
#: heading, which is why the bound is three.
#:
#: **AND `\s` HERE WAS THE SAME BORROWED ALPHABET, IN THE ONE POSITION WHERE
#: IT IS A FALSE PASS RATHER THAN A FALSE REFUSAL.** CommonMark: the `#` run
#: must be followed by *spaces or tabs, or end of line*. `\s` adds `\v`, `\f`,
#: `\r`, NBSP and twenty-four more, so `##\x0c0.2.1 — 2026-08-28` standing
#: over `## 9.9.9 — 2000-01-01` was READ as the newest heading here and by the
#: bash gate (`[[:space:]]`, rc=0, the tag's own version echoed) while the
#: renderer's newest `<h2>` is `9.9.9 — 2000-01-01`. Three bash false passes
#: (`\v`, `\f`, `\r`) and twenty-six wrong readings here. With `[ \t]` the
#: line is not a heading line, falls to the whitelist below, and is refused BY
#: NAME — which is the design working, once its alphabet is the renderer's.
_HEADING_LINE = re.compile(r"^ {0,3}##(?:[ \t]|$)")

#: A FENCED-CODE DELIMITER, spelled the way CommonMark spells one: up to three
#: leading spaces, then a run of THREE OR MORE backticks or tildes, then the
#: info string. The run and the character are captured because both decide
#: whether a later line CLOSES this block — see :func:`newest_heading_line`.
#:
#: **THIS USED TO BE `^ {0,3}(?:```|~~~)` AND THE SCAN TOGGLED ON IT.** Under
#: that reading any delimiter closed any block, so ```` ``` ```` followed by
#: `~~~` left the scan believing it was OUTSIDE a fence the renderer had left
#: OPEN, and the next `## ` line — code, to a renderer — was taken as the
#: newest heading. Driven; see :func:`newest_heading_line`.
_FENCE_LINE = re.compile(r"^ {0,3}(?P<run>`{3,}|~{3,})(?P<info>.*)$")

#: CommonMark's HTML block **type 2**, which is what an HTML comment is: the
#: line must BEGIN with `<!--`, after at most three spaces. The block ends on
#: the first line CONTAINING `-->`, that line included.
#:
#: **THIS USED TO BE THE SUBSTRING TEST `"<!--" in line`**, which opened a
#: comment on any line MENTIONING the sequence — inside a code span, inside a
#: sentence — and ran it to end of file. Driven on the 2954-document corpus at
#: `tests/_changelog_renderer_corpus.py`: **20** documents were refused with
#: *"CHANGELOG.md holds no `## ` heading at all"* over a document whose newest
#: `<h2>` the renderer reads without difficulty, and which this reader now
#: reads correctly. Every one of the 20 carries the alphabet's `` `<!--` ``
#: token — a code span, which is inline and opens no block at all.
_COMMENT_OPEN = re.compile(r"^ {0,3}<!--")

# --- the shapes the scan may STEP OVER, and why that is a whitelist ---------
#
# THE SCAN READS ONLY THE PREFIX OF THE DOCUMENT UP TO ITS FIRST `## ` LINE.
# Everything below that line is somebody else's problem: the newest heading is
# a POSITION, and no line after it can move it. So the only lines this reader
# has to be right about are the ones standing above the heading — a comment, a
# title, some prose — and the question for each of them is not "what is it"
# but "can this grammar be SURE it is not, and does not hide, a heading".
#
# **THE ANSWER IS A WHITELIST AND THE DEFAULT IS REFUSAL.** A line in that
# prefix that is not one of the shapes below stops the scan with
# :exc:`UndecidedChangelogShape`. That is the opposite shape from a list of
# hazards: a CommonMark construct nobody here thought of — a block quote, a
# list item, an HTML block that is not a comment, a leading tab — lands in the
# refusal branch rather than being stepped over, and a release gate is allowed
# to be conservative. Refusing an exotic-but-valid `CHANGELOG.md` costs a
# maintainer one edit. Passing one costs a wrong release note standing beside
# an artefact that cannot be unpublished, only yanked.
#
# AND THERE IS A STRUCTURAL REASON THE WHITELIST HOLDS, which an audit
# supplied and which is worth having beside the drives rather than instead of
# them. Every CommonMark construct that can open a block or move a heading
# begins with one of `#`, `>`, `-`, `+`, `*`, `_`, `=`, `~`, a backtick, `<`,
# `[`, four spaces, or a digit followed by `.` or `)`. The admitted class is
# `[A-Za-z0-9`~]` with the ordered-list marker carved out, so it intersects
# that set only where the fence branch has already declined. **And because
# UTF-8 is self-synchronising, no byte of a multi-byte character can BE one of
# those markers** — so the bash reader's byte-level matching under `LC_ALL=C`
# cannot be fooled by a non-ASCII document either. Driven rather than left as
# an argument: 20 000 invalid-UTF-8 documents, 3 988 of them read, 0 unsound.
# That measurement was taken elsewhere, 2026-08-29, and is a dated record.
#
# Every member below is DRIVEN INERT against a real CommonMark renderer rather
# than argued from a reading of the specification — see
# `tests/_changelog_renderer_corpus.py` and
# `tests/test_release_gates.py::test_BOTH_readers_of_the_newest_heading_agree`.
# Adding a member means adding a document to that corpus and regenerating it
# with the renderer.
#
# AND THE CORPUS IS ASKED WHETHER IT CAN SEE THESE RULES CHANGE, which is a
# different question from whether it agrees with them today. Measured
# 2026-08-28 on this branch, one mutation at a time, running
# `tests/test_release_gates.py -k "corpus or readers or whitespace"`:
#
#   mutation                                             result
#   `_INDENTED_LINE`  ^ {4}   -> ^ {3}                    3 failed
#   `_FENCE_LINE`     ^ {0,3} -> ^ {0,4}                  1 failed
#   `_HEADING_LINE`   ^ {0,3} -> ^ {0,4}                  2 failed
#   closer run `>= fence[1]` -> `>= 3`                    2 failed
#   closer character need not match the opener            2 failed
#   `_ORDERED_LIST_LINE` never matches                    2 failed
#   `_PARAGRAPH_LINE` widened to `<>=*+-`                 10 failed
#   bash closer indent `-le 3` -> `-le 4`                 1 failed
#
# And the six a SECOND audit found, each re-driven after its repair, over
# `tests/test_changelog_names_the_version.py tests/test_release_gates.py`:
#
#   mutation                                             result
#   `_lines` back to `split("\n")` + trailing-CR strip    3 failed
#   the NUL guard removed, both readers                   2 failed
#   `export LC_ALL=C` deleted from the changelog step     1 failed
#   four `SEPARATOR_FORMS` deleted and regenerated        1 failed
#   four indent-boundary `NAMED` rows deleted, regen'd    1 failed
#   `\d`/`\w` restored to `_HEADING`/`_ORDERED_LIST_LINE` 2 failed
#
# THE LAST THREE ARE MUTATIONS OF THE INSTRUMENT AND NOT OF THE READERS, which
# is the shape this branch keeps having to learn: every authored surface of
# the corpus -- the alphabet, the named documents, the swept positions -- can
# be shrunk by an edit plus the documented regeneration command, and the
# smaller corpus agrees with a live renderer perfectly. Each of the three now
# has a floor asserted from outside the generated file.
#
# AND A THIRD AUDIT FOUND THE FLOORS THEMSELVES WERE BUILT AGAINST THE ATTACK
# AND NOT AGAINST THE CLASS. Each was written the round after somebody shrank
# the surface it guards, and the third one bounded the NAMES of the whitespace
# positions while the SHAPES those names stand for stayed unbounded -- so one
# line, every name intact, 252 rows intact, re-pointed a position out of the
# thing it was there to test and un-pinned the previous round's finding. It
# bounds documents now, as the other two do. Re-driven, each repair reverted
# on its own:
#
#   mutation                                             result
#   the `fence-closer` shape re-pointed, name kept        1 failed
#   the NUL guard removed from PYTHON alone               1 failed
#   the `export LC_ALL=C` CODE line deleted, comments
#     left saying `export LC_ALL=C` three times           1 failed
#   the `v`-only tag refusal removed                      1 failed
#   `_ISO_DATE` back to `\d{4}-\d{2}-\d{2}`                1 failed
#
# THE SECOND AND THIRD ROWS ARE THE INTERESTING ONES. Deleting the NUL guard
# from BASH reddened two rows and deleting it from PYTHON reddened nothing,
# because both NUL documents were refused by the whitelist anyway -- the half
# carrying the DRIFT obligation was unpinned by rows that could not tell guard
# from whitelist. And a present-construct needle read over the whole extracted
# body is satisfied by the body's own PROSE about the construct: MOST of that
# body is comment, and more than one of those comments says `export LC_ALL=C`
# while discussing it. The live figures are derived and printed by
# `tests/test_release_gates.py::test_the_drives_are_reading_the_real_step_bodies`
# rather than restated here; a numeral typed beside a body that grows every
# time somebody explains it is a numeral that is false one commit later, which
# is what happened to the two copies of it that stood here.
#
# The first two rows and the last are the ones that MATTER, because they are
# the ones an auditor found SURVIVING at 39 passed apiece before the
# indent-boundary documents were added to `NAMED`: no document anywhere placed
# a non-heading whitelist member at indent one to three, or a fence closer at
# indent four, so the `>= 4` admission — the most load-bearing of the four
# rules here — had an undriven boundary.

#: Whitespace only. Closes a paragraph and an HTML block of type 6 or 7;
#: cannot make or hide a heading.
_BLANK_LINE = re.compile(r"^[ \t]*$")

#: Four or more leading SPACES. Either an indented code block or the lazy
#: continuation of a paragraph, and neither can be a `##` heading — a heading
#: line is bounded at three spaces of indent, which is the whole reason
#: `_HEADING_LINE`'s bound is three.
_INDENTED_LINE = re.compile(r"^ {4}")

#: An ATX heading of some level OTHER than two — level two has already
#: returned by the time this is reached. A heading closes whatever stands
#: above it and opens nothing.
_ATX_LINE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]|$)")

#: A paragraph line: nothing that begins a CommonMark block begins with an
#: ASCII letter, an ASCII digit, or **a leading backtick or tilde that did not
#: open a fence**, EXCEPT the ordered list below.
#:
#: **THAT LAST CLAUSE SAID "a run shorter than three" AND THE RUN CAN BE
#: THREE.** What stood here was *"a run of three or more of either has already
#: been taken as a fence delimiter above; a shorter run begins no block at
#: all"*, and the code three lines up in :func:`newest_heading_line` says
#: otherwise in as many words: a BACKTICK fence whose info string holds a
#: backtick is not a fence, so ```` ``` a`b ```` falls through to this rule
#: with a leading run of exactly three and is admitted as a paragraph — which
#: is correct, and is what both renderers do with it, and is a corpus row.
#: The BEHAVIOUR was right and the sentence was narrower than the class it
#: was justifying, which is the same defect as a sentence wider than one.
#:
#: The reason a leading backtick or tilde is inert once the fence branch has
#: declined it: a backtick opens a code SPAN, which is inline and cannot
#: swallow the line beneath it, and tildes are ordinary text to CommonMark.
#:
#: **THE TILDE WAS MISSING AND THE TWO ARE ONE RULE**, which an auditor noted:
#: the class admitted a short backtick run and refused a short tilde run, on
#: an argument that covers both. The direction was safe — a refusal — but a
#: rule spelled once and applied to one of its two subjects is a rule nobody
#: can check. Added as a whitelist member is added: with a document in the
#: corpus, driven inert against the renderer, not with an argument.
_PARAGRAPH_LINE = re.compile(r"^ {0,3}[A-Za-z0-9`~]")

#: ...and the one carve-out. `1. item` begins with a digit and is an ordered
#: LIST, whose content is indented four columns — where a `## ` heading is
#: still a heading to a renderer and is invisible to a reader bounded at three.
#: Driven: `1. text` / blank / `    ## 9.9.9 — 2000-01-01` renders an `<h2>`
#: this scan would step over.
_ORDERED_LIST_LINE = re.compile(r"^ {0,3}[0-9]{1,9}[.)](?:[ \t]|$)")


#: A LINE ENDING, spelled the way CommonMark 0.31.2 spells one in *Characters
#: and lines*: *"a line feed, a carriage return not followed by a line feed, or
#: a carriage return and a following line feed."* It is character-for-character
#: `markdown-it`'s own `NEWLINES_RE`, which is how the renderer decides the
#: same question.
_LINE_ENDING = re.compile(r"\r\n?|\n")

#: The character neither reader can carry. CommonMark replaces it with U+FFFD;
#: bash's `read` DROPS it, silently, because a bash variable cannot hold one.
_NUL = "\x00"


def _lines(text: str) -> list[str]:
    """`text` split into lines, by CommonMark's rule and `release.yml`'s alike.

    **THIS CALLED `str.splitlines()`, THEN IT SPLIT ON `\\n` ALONE, AND THE
    SECOND WAS A SOUNDNESS REGRESSION THIS MODULE INTRODUCED.** The reasoning
    written here for the change was:

        *"`str.splitlines()` — what this used to call — breaks on eight
        boundaries CommonMark does not know and `read` does not either: `\\v`,
        `\\f`, `\\x1c`, `\\x1d`, `\\x1e`, `\\x85`, `\\u2028`, `\\u2029`."*

    **That list is correct and it is INCOMPLETE.** `splitlines()` also breaks
    on a bare `\\r`, and that break CommonMark **does** know — *"a carriage
    return not followed by a line feed"* is a line ending. Splitting on `\\n`
    and stripping a trailing `\\r` corrected eight boundaries and broke one,
    and the corpus could not see it, because the reader still pointed at the
    same line NUMBER while holding different text:

        'text\\r## 9.9.9 — 2000-01-01\\n## 0.2.1 — 2026-08-28\\n'
            renderer   line 2, '## 9.9.9 — 2000-01-01'
            that reader line 2, '## 0.2.1 — 2026-08-28'   <- certified sound

    End to end, one invisible CR inserted into this repository's own
    `CHANGELOG.md` after a paragraph line took the bash gate to rc=0 printing
    *"newest heading: 0.2.1 — 2026-08-28"* over a document whose newest
    rendered heading is the 9.9.9.

    **AND WHAT SAVED THE RELEASE PATH WAS AN ACCIDENT THIS MODULE ALREADY
    REFUSES TO RELY ON.** The caller reads `CHANGELOG.md` with
    `read_text(encoding="utf-8")`, whose universal-newline translation
    converts the CR before this function ever sees it — so the coupling check
    reddened on the planted tree and `build: needs: test` would have stopped
    the release. `_one_line_break` in `tests/test_release_gates.py` says of
    exactly that shape: *"these checks are correct because of how the file was
    OPENED and not because of anything they do."* The corpus drives STRINGS
    and has no such protection.

    So the split is :data:`_LINE_ENDING`, which is `markdown-it`'s own rule and
    CommonMark's sentence. The eight spurious boundaries stay corrected: `\\v`,
    `\\f`, `\\x1c`, `\\x1d`, `\\x1e`, `\\x85`, `\\u2028` and `\\u2029` are
    ordinary characters here, to the renderer, and to `read -r` in
    `release.yml`.

    DRIVEN: the corpus rows *"a bare carriage return is a line ending"* and
    *"a form feed is not a line break"*, which are red on the two readings
    this docstring records and green on this one.

    **AND THE MODEL IS EXHAUSTIVELY THE RENDERER'S, WHICH IS AN OUTSIDE
    MEASUREMENT AND IS DATED FOR THAT REASON.** An independent audit compared
    this function, the `release.yml` loop that builds `lines`, and
    `renderer_lines` element-by-element over **all 1093 strings on
    `{a, \\r, \\n}` of length at most 6**: 0 mismatches, all three. It also
    confirmed `NEWLINES_RE` is `\\r\\n?|\\n` at markdown-it-py 3.0.0 and 4.2.0
    alike, and that markdown-it splits the NORMALISED source on `\\n` — so the
    indices `renderer_lines` produces are the indices `token.map` uses.
    Measured 2026-08-29, elsewhere, on the readers this commit ships.
    """
    out = _LINE_ENDING.split(text)
    if out and out[-1] == "":
        out.pop()
    return out


def newest_heading_line(text: str) -> tuple[int, str] | None:
    """`(line number, line)` for the line a RENDERER would make the first `<h2>`.

    `None` means the document has no such line. :exc:`UndecidedChangelogShape`
    means the prefix above it holds a shape this grammar cannot decide, and
    that is a REFUSAL rather than a guess — see the whitelist above.

    **THIS FUNCTION WAS CALLED `heading_lines`, IT YIELDED EVERY HEADING, AND
    ITS DOCSTRING LICENSED THE TWO DEFECTS THIS ONE CLOSES.** What it said,
    verbatim:

        *"**Its reach, stated:** this is a line grammar, not a Markdown
        parser. It does not know setext headings (`Release\\n=======`), a
        fence opened with a longer run than it is closed with, or an HTML
        comment opened inside a fence. Those are all shapes `CHANGELOG.md`
        does not use, and a heading it misses is stepped past — which is the
        direction :func:`newest_heading` refuses, so a miss here becomes a
        refusal there rather than a silent older reading."*

    **THE LAST CLAUSE IS FALSE IN THE PERMISSIVE DIRECTION, AND IT IS THE
    SENTENCE THAT LICENSED LEAVING BOTH SHAPES UNHANDLED.** A miss does not
    become a refusal. The scan steps past the heading it missed and reaches a
    LATER `## ` line, which parses, agrees with the tag, and returns rc=0 —
    with the document's real newest heading naming a different release, or
    naming nothing at all because it is inside a code block. Driven against
    `markdown-it-py`'s CommonMark preset, at the old reader:

        `` ``` `` / `~~~` / `## 0.2.1 — 2026-08-28`
            old reader -> ('0.2.1', '2026-08-28');  renderer -> no <h2> at all
        `text` / `---` / `## 0.2.1 — 2026-08-28`
            old reader -> ('0.2.1', '2026-08-28');  renderer -> the <h2> is 'text'

    **WHAT THIS READER DOES INSTEAD, and it is refusal rather than
    CommonMark.** The fence rule IS implemented, because it is exact and small
    — a fence closes only on the same character, a run at least as long, and
    no info string.

    **AND "EXACT AND SMALL" WAS TRUE OF ITS STRUCTURE AND NOT OF ITS
    CHARACTER CLASSES, WHICH IS WHERE BOTH OF THIS BRANCH'S OWN SOUNDNESS
    BREAKS TURNED OUT TO LIVE.** An auditor could not construct a single
    structural admission that hides a heading — four lines deep, two
    adversarial alphabets, 30 000 Hypothesis examples — and then found two
    defects in the code this design chose to IMPLEMENT rather than refuse,
    both the same mistake: a whitespace alphabet borrowed from the host
    language standing in for one CommonMark defines narrowly. Refusal
    protects a reader where it refuses; where it implements, it owns the
    details. See :data:`_HEADING_LINE`, :data:`_HEADING`, the closing-fence
    test below, and the sweep those three are now held to.

    Setext is NOT implemented and is not meant to be: a
    `---` or `===` line above the first heading is a shape whose first
    character is not on the whitelist, so it stops the scan. So does `>`, so
    does `- `, so does `<div>`, so does a leading tab. The line grammar's
    answer to "I cannot decide this" is now a named refusal instead of a step.

    **THE MEASUREMENT, dated, because the paragraph above is an argument and
    this is the evidence.** Taken 2026-08-28 at this branch's tree, with
    `markdown-it-py` 4.2.0's CommonMark preset as the oracle, over every
    document of at most three lines on a TWENTY-token alphabet — 8420 of them
    — carrying the shapes `tests/_changelog_renderer_corpus.py`'s own
    fourteen-token alphabet cannot reach: an indented heading, a tab-indented
    one, a backtick run of four, a block quote, a bullet list, a `<div>`, a
    fourth-level ATX heading and a link reference definition. The bash gate
    driven here is the real extracted step body of
    `.github/workflows/release.yml`, not a copy of it, and "unsound" means the
    reader points at a LINE the renderer puts no `<h2>` on.

        readers                     unsound   twin drift   refusals the
                                                            renderer reads
        this branch                       0            0            1536
        `a90862b` (Python twin only)    206            —            1028

    **Zero and 1536 is the whole trade in two numbers.** Nothing is read that
    a renderer does not read; a great deal that a renderer reads is refused.
    The 8420 are not in the suite — what is in the suite is the 2954-document
    corpus, at
    `tests/test_release_gates.py::test_the_two_readers_agree_over_the_whole_corpus`,
    where the same two numbers are 0 and 205 and the `a90862b` readers are 42
    and 132 — plus the whitespace alphabet, swept separately at
    `::test_the_whitespace_alphabet_is_swept_in_every_position`, because a
    STRUCTURAL sweep cannot see a character-class defect and this branch
    shipped two of those before an auditor swept for them.

    **WHAT IT STILL CANNOT SEE.**

    * **The ATX closing sequence.** `## 0.2.0 — 2026-08-25 ##` renders with
      the trailing `##` stripped; :data:`_HEADING` keeps it, so the PARSED
      `rest` differs from the rendered content. It is not "the reading" that
      differs, which is what this bullet used to say: the LINE and its SOURCE
      TEXT are identical, and those are the two things the corpus compares —
      which is exactly why no growth of the corpus can surface this and why
      the bullet is here. The VERDICT does not differ either: `2026-08-25 ##`
      is not an ISO date and is not the word *unreleased*, so every caller
      refuses it, in all four variants driven at
      :func:`test_both_halves_of_the_coupling_are_driven`.
    * **Anything below the heading it returns.** The scan stops there.
    * **Whether `CHANGELOG.md` is the document the release ships.** That is
      `pyproject.toml`'s allowlist and `tests/test_sdist_contents.py`.
    """
    # THE NUL, REFUSED BEFORE ANYTHING IS READ, and it is the bash twin this
    # is for. CommonMark replaces U+0000 with U+FFFD; `read -r` DROPS it,
    # because a bash variable cannot hold one — so the gate parsed a line the
    # document does not contain. Driven: `'#\x00# 0.2.1 — 2026-08-28\n'`
    # renders NO <h2> at all and took the bash step to rc=0 with
    # `version=0.2.1`. Python CAN hold it and would refuse this document by
    # the whitelist anyway; it is refused HERE, by name, in both readers,
    # because a line grammar that cannot represent its input must not read it
    # — and because the two readers refusing for different reasons is how they
    # stop being twins.
    if _NUL in text:
        raise UndecidedChangelogShape(
            len(_lines(text[: text.index(_NUL)] + "x")),
            _lines(text[: text.index(_NUL)] + "x")[-1],
            why=(
                "it holds a NUL byte. CommonMark replaces one with U+FFFD and "
                "the bash half of this gate cannot hold one at all — `read` "
                "drops it silently, so that reader would parse a line this "
                "document does not contain. A line grammar that cannot "
                "represent its input must not read it."
            ),
        )
    fence: tuple[str, int] | None = None
    in_comment = False
    for number, line in enumerate(_lines(text), 1):
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        delimiter = _FENCE_LINE.match(line)
        if fence is not None:
            if delimiter is not None:
                run, info = delimiter.group("run"), delimiter.group("info")
                # `info.strip(" \t")` AND NOT `info.strip()`. CommonMark: a
                # closing fence "may be followed only by spaces or tabs".
                # `str.strip()` is Unicode-aware, so ```` ```\u00a0 ```` closed
                # a block here that the renderer AND the bash twin — which
                # spells this `${finfo//[[:blank:]]/}` — both leave open.
                # Driven by an auditor on a 23-token alphabet, 12 719
                # documents: 8 unsound readings and 4 disagreements between
                # the two readers, every one of them a Unicode-space closer,
                # and both of those quantities are asserted zero by the sweep
                # in `tests/test_release_gates.py`.
                if (run[0] == fence[0] and len(run) >= fence[1]
                        and not info.strip(" \t")):
                    fence = None
            continue
        if delimiter is not None:
            run, info = delimiter.group("run"), delimiter.group("info")
            # A BACKTICK fence's info string may not itself hold a backtick;
            # `` ``` a`b `` is a paragraph, not a fence, and falls through to
            # the whitelist below — where a leading backtick is a code span.
            if run[0] != "`" or "`" not in info:
                fence = (run[0], len(run))
                continue
        if _COMMENT_OPEN.match(line):
            if "-->" not in line:
                in_comment = True
            continue
        if _HEADING_LINE.match(line):
            return number, line
        if (_BLANK_LINE.match(line)
                or _INDENTED_LINE.match(line)
                or _ATX_LINE.match(line)
                or (_PARAGRAPH_LINE.match(line)
                    and not _ORDERED_LIST_LINE.match(line))):
            continue
        raise UndecidedChangelogShape(number, line)
    return None


def headings(text: str) -> list[tuple[str, str]]:
    """`(version, rest)` for every release heading that PARSES, newest first.

    File order, not sorted: the changelog is written newest-first and the
    *newest heading* is a position in the document, not the largest version.
    Sorting here would quietly accept a file whose newest entry had been
    inserted in the wrong place.

    **This is not the function to ask for the newest heading** — see
    :func:`newest_heading`, and the paragraph below for why the difference is
    a defect rather than a nicety.
    """
    return [(m.group("version"), m.group("rest")) for m in _HEADING.finditer(text)]


def newest_heading(text: str) -> tuple[str, str] | None:
    """The newest heading, FOUND by position and PARSED second, or `None`.

    **`headings(text)[0]` WAS THE NEWEST HEADING AND IT WAS NOT**, and the
    difference is a false GREEN rather than a confusing red.
    :data:`_HEADING` is applied with `finditer`, which **skips** what it
    cannot match — so a newest heading that does not parse is stepped over and
    `[0]` becomes an OLDER one. Today that usually reds for the wrong reason,
    naming the wrong version. On the day `stelling.__version__` happens to
    equal that older heading's version — which is exactly the state a release
    bump passes through — **it goes green with a malformed heading standing
    above it**, and the one coupling this module exists to hold is not held.

    Found by the branch that built the same gate in
    `.github/workflows/release.yml`, which refuses that shape deliberately:
    its step locates the first `## ` line and *then* parses it, so a malformed
    newest heading is a refusal and never a step past. The two readers now
    agree about what "newest" means, which they did not.

    **THREE OUTCOMES, NOT TWO, AND THE THIRD ARRIVED WITH THE RENDERER.**
    `None` means **no `## ` line at all**; a line that is present and does not
    parse raises :exc:`MalformedNewestHeading`; and a PREFIX above the heading
    that this line grammar cannot decide raises
    :exc:`UndecidedChangelogShape`, which :func:`newest_heading_line` explains
    at length. A can't-read resolving to the same answer as a nothing-to-read
    is how two cases stop being distinguishable, and each of these three is a
    different thing for a maintainer to do.
    """
    first = newest_heading_line(text)
    if first is None:
        return None
    _, line = first
    # `strip(" \t")` AND NOT `strip()`, for :data:`_HEADING`'s reason: the
    # bare call is Unicode-aware and would silently normalise a separator
    # CommonMark does not know before the pattern ever sees it.
    parsed = _HEADING.match(line.strip(" \t"))
    if parsed is None:
        raise MalformedNewestHeading(line)
    return parsed.group("version"), parsed.group("rest")


class MalformedNewestHeading(ValueError):
    """The first `## ` line is not a `## <version> — <rest>` heading.

    Its own exception type rather than a `None`, so that a caller cannot
    handle it by accident with the same branch that handles an empty file.
    """


class UndecidedChangelogShape(ValueError):
    """A line ABOVE the newest heading that this line grammar cannot decide.

    Its own exception type, and a THIRD outcome rather than a second spelling
    of one of the other two, because the three are three different things a
    maintainer has to do: `None` is *write a heading*,
    :exc:`MalformedNewestHeading` is *fix the heading you wrote*, and this is
    *the prefix above your heading uses a construct this gate will not guess
    at — simplify it, or move the heading above it*.

    **IT EXISTS BECAUSE THE ALTERNATIVE WAS A GUESS THAT WENT THE PERMISSIVE
    WAY.** Before it, a shape the scan could not decide was simply stepped
    over, and the scan then found a LATER heading that parsed and agreed with
    the tag. The document's real newest heading — a setext `<h2>`, or nothing
    at all because a fence the scan thought closed was still open — never
    reached the comparison. See :func:`newest_heading_line`.
    """

    def __init__(self, number: int, line: str, why: str | None = None) -> None:
        super().__init__(number, line, why)
        self.number = number
        self.line = line
        self.why = why

    def __str__(self) -> str:
        if self.why is not None:
            return (
                f"the changelog cannot be read at all, from line "
                f"{self.number}: {self.why}"
            )
        return (
            f"line {self.number} of the changelog is a shape this gate "
            f"cannot decide, and it stands ABOVE the newest heading: "
            f"{self.line!r}. This reader is a line grammar, not a CommonMark "
            f"parser, and the shapes it steps over are a whitelist — blank, "
            f"indented four spaces, another ATX level, or a paragraph line "
            f"beginning with an ASCII letter or digit, or a leading backtick "
            f"or tilde that did not open a fence. (That clause has been wrong "
            f"in both directions: it said `a letter, a digit or a code span`, "
            f"which is WIDER than `[A-Za-z0-9`~]` — `École` is refused — and "
            f"then `a short backtick or tilde run`, which is NARROWER, since "
            f"a three-backtick run whose info string holds a backtick is no "
            f"fence and lands here. Both readings were safe and neither was "
            f"true.) Everything "
            f"else stops it here rather than being stepped past to a later "
            f"heading that happens to agree with the tag. Setext underlines "
            f"(`---`, `===`), block quotes, list markers and HTML blocks that "
            f"are not comments all land here. Rewrite the prefix above the "
            f"newest heading, or move the heading above it."
        )


def test_the_newest_changelog_heading_names_this_version():
    """The coupling. Two halves, and the second is what keeps it live.

    Half one: the heading's version is the release `stelling.__version__`
    belongs to. Half two: the heading and the version agree about whether
    that release has happened.
    """
    text = CHANGELOG.read_text(encoding="utf-8")
    try:
        found = newest_heading(text)
    except MalformedNewestHeading as bad:
        raise AssertionError(
            f"{CHANGELOG.name}'s newest `## ` line does not parse as a "
            f"release heading: {str(bad)!r}. It is NOT stepped over to reach "
            f"an older one that does — that step is how this check went green "
            f"over a malformed heading whenever the older one happened to "
            f"name the current version."
        ) from None
    assert found is not None, (
        f"{CHANGELOG.name} has no `## <version> — <something>` heading, so "
        f"there is nothing to tie `stelling.__version__` to. A changelog "
        f"whose headings stopped parsing is a changelog this check reads as "
        f"satisfied, which is why the shape is asserted before the content."
    )
    version = stelling.__version__
    newest, rest = found

    assert newest == release_segment(version), (
        f"`CHANGELOG.md`'s newest heading names {newest!r} and "
        f"`stelling.__version__` is {version!r}, which is a build of "
        f"{release_segment(version)!r}. The changelog is the release note for "
        f"the version in `src/stelling/__init__.py` — `pyproject.toml` reads "
        f"the distribution's version from that same line — so a heading "
        f"naming anything else describes a different release."
    )

    if is_unshipped(version):
        assert rest.strip().lower() == _UNRELEASED, (
            f"`stelling.__version__` is {version!r}, a build of "
            f"{newest!r} that has not shipped, and the heading says "
            f"{rest!r}. It must say {_UNRELEASED!r}: a dated heading over an "
            f"unshipped version is a release note claiming a release date "
            f"the release has not got."
        )
    else:
        assert _ISO_DATE.match(rest.strip()), (
            f"`stelling.__version__` is {version!r}, a released version, and "
            f"the heading says {rest!r}. A released version's heading carries "
            f"the date it was released. (The date's VALUE is not checked "
            f"here, on purpose — see this module's docstring: the only date "
            f"available to this suite is the developer's clock.)"
        )


def test_this_gate_does_not_read_the_clock():
    """The trap, closed by assertion rather than by intention.

    A date check here would have to call `date.today()`, because the tag that
    carries the real date is created after this suite runs. That is the
    environment-dependence class four of five recent reds came from, and the
    way it arrives is somebody adding it later in good faith. So the module is
    checked for the imports that would make it possible.

    Not a substitute for reading the file — it is a tripwire on the one
    mechanism, and it says so. The date check belongs in `release.yml`; the
    recommendation is in this module's docstring, and this agent does not own
    that workflow.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    clocks = {"datetime", "time", "calendar", "today", "now", "utcnow", "monotonic"}
    reached = sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in clocks
        }
        | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in clocks
        }
        | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if alias.name.split(".")[0] in clocks
        }
        | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            and node.module.split(".")[0] in clocks
        }
    )
    assert not reached, (
        f"this module reaches for a clock: {reached}. The tag that carries a "
        f"release's real date does not exist when this suite runs, so a date "
        f"compared here is compared against whoever is running the tests. "
        f"Put the date check in `.github/workflows/release.yml`, after the "
        f"tag is resolved."
    )
    # Read off the SYNTAX and not the bytes, because the prose above has to be
    # able to say `date.today()` in order to explain why it is refused — the
    # first draft of this test scanned the file's lines and failed on its own
    # docstring, which is the same defect one layer down.


def test_the_routed_date_check_is_in_this_workflow():
    """The paragraph above ROUTES a check to another file. This reads that file.

    THE DEFECT IT CLOSES IS ITS OWN HISTORY. That paragraph was a
    recommendation for 47 commits — a present-tense claim about
    `.github/workflows/release.yml`, naming a check that file did not carry,
    with nothing in the tree reading it. It could not go red; it could only go
    stale, and it did. So the claim is now an assertion: the routed step is
    HERE, in a job the upload waits for and is not, reading the three things
    the routing names, and still carrying a refusal.

    WHAT "ACTUALLY THERE" MEANS TO A TEXT READER, and it is four things and not
    one:

    * the STEP resolves, exactly once, by the name this module routes to.
      `_step_lines` refuses zero matches and two matches alike, so a rename is
      a red rather than a scan that finds nothing and is satisfied.
    * its BODY is a real script — `_step_body` refuses a block that does not
      open with the step's own `set -euo pipefail`, so a `run:` read as empty
      cannot satisfy the needles below by containing nothing.
    * that body names `CHANGELOG.md`, `GITHUB_REF_NAME` and
      `%(taggerdate:short)` — the document, the tag and the date, which are
      the three nouns the routing paragraph is made of — and still carries an
      `exit 1`. A step that stopped reading any one of the three is not the
      routed check under its own name, and one that reads all three and
      refuses nothing is a routing target in name only.
    * the step is in a job that `publish` WAITS FOR and is not `publish`
      itself. A refusal that runs beside the upload is not a refusal, and one
      in a job nothing waits for is not in the release path at all.

    **WHAT THIS CHECK CANNOT SEE.** `tests/_lanes.py`'s docstring is the
    reason this paragraph exists: a line-anchored text reader over
    `.github/workflows/nightly-jax-canary.yml` went past NINE legal spellings
    of the thing it was looking for — a quoted key, a flow mapping, an alias,
    `matrix.exclude`, a job-level `if:`, a `defaults:` block, `set -a` with a
    sourced file, `continue-on-error: true`, and `runs-on:` written before
    `name:` — and reported "no setting" for each. This reader is the same kind
    of instrument over a different file, and:

    * IT IS BLIND TO EVERY RESPELLING OF THE STEP — `- {name: …, run: …}`,
      `"name": …`, an anchor and alias, `run: >` instead of `run: |` — but it
      fails LOUDLY on all of them rather than quietly: each one takes
      `_step_lines` or `_step_body` to zero matches, which is an assertion
      error naming the step. That is the difference between this and the
      nightly reader, and it is a property of `_step_lines`, not of this test.
      Measured with PyYAML on the `release.yml` this commit ships,
      2026-08-28: it carries 0 anchors, 0 aliases and 0 merge keys, its jobs
      are `build`, `publish` and `test`, and the routed step parses as
      `jobs.build.steps[1]` with the checkout still `steps[0]`. That is a
      dated reading of one file and not a guarantee about the next edit of
      it — and PyYAML is not a dependency of this project, so the reading
      could be taken but not kept: `tests/_lanes.py` explains why neither
      module parses.
    * IT CANNOT SEE WHETHER THE STEP RUNS. A step-level or job-level `if:`
      switches it off and nothing in this repository reads `if:` at all —
      said plainly because the sibling attribute IS covered:
      `tests/test_release_gates.py::test_the_attributes_that_decide_whether_a_refusal_refuses`
      holds `continue-on-error` absent for the whole file, and
      `::test_the_publish_job_still_needs_the_suite` holds `needs`. `if:` is
      the hole, and it is a hole in `release.yml`'s coverage generally rather
      than one this check introduces.
    * IT CANNOT SEE WHETHER THE CHECK WORKS. Four needles in a script are not
      a verdict about what the script does: a body that names all four and
      never reaches its `exit 1` satisfies every assertion here. What decides
      that is
      `tests/test_release_gates.py::test_the_changelog_gate_refuses_a_heading_the_tag_does_not_carry`,
      which runs this same extracted body against planted trees,
      `::test_the_changelog_gate_reads_the_NEWEST_heading_by_POSITION`, and
      the two drives that hold the step's line grammar to a RENDERER rather
      than to this module's twin of it —
      `::test_BOTH_readers_of_the_newest_heading_agree` and
      `::test_the_two_readers_agree_over_the_whole_corpus`.
    * IT CANNOT SEE WHETHER THE WORKFLOW RUNS AT ALL — the `on:` trigger, the
      repository's own branch protection, whether the `pypi` environment
      exists. That is the runner's furniture and no test here reaches it.
    """
    body = _step_body(ROUTED_STEP)

    # CODE LINES, NOT THE WHOLE BODY, AND THE REASON IS A DEFECT THIS CHECK
    # ACTUALLY HAD. These four needles were matched against the extracted
    # `run:` block including its comments, and `release.yml` is a heavily
    # commented file whose step comments quote the constructs beside them. At
    # `a90862b` `%(taggerdate:short)` appeared on exactly one line of that
    # block and it was CODE, so the needle was live; at `fd03c02` a new
    # comment inside the same block spelled it too, and the needle stopped
    # being able to fail. Measured on both blobs — 0/1 and 1/1 comment/code
    # hits — and driven at `fd03c02`: rewriting the CODE line's
    # `%(taggerdate:short)` to `%(creatordate:short)` left this module green.
    # `tests/test_release_gates.py` carries the same needle and was repaired
    # the same way in the same commit; the two readers of this one literal are
    # written twice and nothing holds them to each other, which is declared
    # here rather than left for a release to discover.
    code = "\n".join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    missing = [
        needle for needle in ("CHANGELOG.md", "GITHUB_REF_NAME",
                              "%(taggerdate:short)", "exit 1")
        if needle not in code
    ]
    assert not missing, (
        f"the step {ROUTED_STEP!r} in `.github/workflows/release.yml` no "
        f"longer reads {missing}. This module routes the changelog DATE check "
        f"there — it cannot make it here, because the tag that carries the "
        f"real date does not exist while this suite runs — and a routing "
        f"paragraph whose target has stopped carrying the check is the exact "
        f"defect this test was written for: it stood as a recommendation for "
        f"47 commits and nothing could see it. Either restore the check or "
        f"rewrite the paragraph above to say where it went."
    )

    # THE JOB IT STANDS IN, because a refusal that runs beside the upload is
    # not a refusal. Read off the code lines: job headers are the only thing
    # in this file at indent 2 that ends in a colon.
    lines = _code_lines(_release_text())
    at = [i for i, line in enumerate(lines)
          if line.strip() == f"- name: {ROUTED_STEP}"]
    assert len(at) == 1, (
        f"expected exactly one step named {ROUTED_STEP!r}; found {len(at)}. "
        f"(`_step_body` above addresses the step by that name, so this cannot "
        f"normally be reached — it is here so that a divergence between the "
        f"two readers is a red rather than a silent disagreement about which "
        f"step is being spoken of.)"
    )
    header = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
    job = next((m.group(1) for m in
                (header.match(line) for line in reversed(lines[:at[0]])) if m),
               None)
    assert job is not None, (
        "could not tell which job the routed step is in, so this check cannot "
        "say whether it runs before the upload. The job-header shape this "
        "reads — a name at indent 2 — has changed."
    )
    assert job != "publish", (
        f"the routed changelog check is in the `{job}` job, which is the job "
        f"that uploads to PyPI. A refusal that runs beside the upload it is "
        f"meant to prevent is not a refusal."
    )
    publish = [i for i, line in enumerate(lines) if line.strip() == "publish:"]
    assert len(publish) == 1, (
        f"expected exactly one `publish:` job in `release.yml`, found "
        f"{len(publish)}; this check cannot say what waits for what."
    )
    # BOUNDED TO THAT JOB, and PARSED rather than substring-tested. Scanning
    # to the end of the file would read some LATER job's `needs:` if the job
    # order ever changed, and `job in needs` over the whole line is true for
    # any job whose name is a substring of anything else on it — a check that
    # a claim is well-formed rather than that it is true, which is the defect
    # class this whole branch is about.
    tail = lines[publish[0] + 1:]
    for stop, line in enumerate(tail):
        if header.match(line):
            after = tail[:stop]
            break
    else:
        after = tail
    needs = next((line for line in after
                  if line.strip().startswith("needs:")), None)
    waits_for = re.findall(r"[A-Za-z0-9_-]+",
                           "" if needs is None else needs.split(":", 1)[1])
    assert job in waits_for, (
        f"`publish` waits for {waits_for} and the routed changelog check "
        f"stands in the `{job}` job (`needs:` line read as {needs!r}). A gate "
        f"in a job the upload does not depend on is a gate the upload does "
        f"not pass through: it goes red beside a green publish."
    )

def test_both_halves_of_the_coupling_are_driven():
    """The rule, observed to fire, on planted changelogs.

    Green on this tree, so the only evidence it works is a plant. Four of
    them: the version wrong, the *unreleased*/dated agreement wrong in each
    direction, and a control that the real pairing is accepted — because a
    check that refused everything would pass all three reds and be useless.
    """
    def verdict(version: str, heading: str) -> str | None:
        """The complaint this rule makes about a `(version, heading)` pair."""
        try:
            found = newest_heading(heading)
        except MalformedNewestHeading:
            # A `## ` line that is not a heading is its OWN complaint, and the
            # drive below plants one standing above a well-formed older
            # heading — the shape `headings()[0]` used to step past.
            return "malformed-newest"
        if found is None:
            return "unparseable"
        newest, rest = found
        if newest != release_segment(version):
            return "version"
        if is_unshipped(version):
            return None if rest.strip().lower() == _UNRELEASED else "should-be-unreleased"
        return None if _ISO_DATE.match(rest.strip()) else "should-be-dated"

    # THE CONTROL, and it is the real pairing rather than an invented one.
    assert verdict(
        stelling.__version__, CHANGELOG.read_text(encoding="utf-8")
    ) is None, "the rule refuses this tree's own version/heading pair"

    assert verdict("0.3.0.dev0", "## 0.2.0 — unreleased\n") == "version", (
        "a heading naming the PREVIOUS release while `__version__` has moved "
        "on is not caught — that is the shape a forgotten changelog bump has"
    )
    assert verdict("0.2.0.dev0", "## 0.2.0 — 2026-08-12\n") == "should-be-unreleased", (
        "a DATED heading over an unshipped `.devN` version is not caught"
    )
    assert verdict("0.2.0", "## 0.2.0 — unreleased\n") == "should-be-dated", (
        "an `unreleased` heading over a RELEASED version is not caught, so "
        "this check would pass forever on a changelog frozen at the moment "
        "before the release it documents"
    )
    # THE ATX CLOSING SEQUENCE, which is the one CommonMark shape both
    # readers read DIFFERENTLY from a renderer and neither refuses at the
    # scan. `## 0.2.0 — 2026-08-25 ##` renders as an `<h2>` whose content is
    # `0.2.0 — 2026-08-25`; :data:`_HEADING` keeps the trailing `##` in
    # `rest`. The claim made about that beside
    # :func:`newest_heading_line` — that the VERDICT is a refusal even though
    # the READING differs — is driven here rather than asserted there, in both
    # of the shapes a caller can be in. It is outside
    # `tests/_changelog_renderer_corpus.py` for the same reason: the corpus
    # holds readings to the renderer's, and this one is not one.
    # THE DATE'S DIGITS ARE ASCII, and `_ISO_DATE` said `\d` until an auditor
    # read it. A heading dated with Arabic-Indic digits satisfied a
    # Unicode-aware `\d{4}-\d{2}-\d{2}` and would be refused by the bash gate,
    # which compares the heading's date to the tag's by string equality — so
    # this was a divergence between the two readers rather than a route to a
    # release, and it is driven here rather than described.
    assert verdict("0.2.0", "## 0.2.0 — \u0662026-08-25\n") == "should-be-dated"
    assert verdict("0.2.0.dev0", "## 0.2.0 — unreleased\n") is None

    assert verdict("0.2.0", "## 0.2.0 — 2026-08-25 ##\n") == "should-be-dated"
    assert verdict("0.2.0.dev0", "## 0.2.0 — unreleased ##\n") == "should-be-unreleased"

    # ... and the release-segment surgery is the thing both halves rest on.
    for build, release in (
        ("0.2.0.dev0", "0.2.0"), ("1.0.0rc1", "1.0.0"), ("0.9.0b2", "0.9.0"),
        ("0.9.0a1", "0.9.0"), ("0.2.0", "0.2.0"), ("1.2.3.dev17", "1.2.3"),
    ):
        assert release_segment(build) == release, build
        assert is_unshipped(build) == (build != release), build


def test_a_malformed_newest_heading_is_REFUSED_and_never_stepped_past():
    """The false GREEN `headings(text)[0]` had, driven on the shape that has it.

    **THIS IS THE ONE THE OLD READER GOT WRONG, AND IT IS A GREEN RATHER THAN
    A RED.** `_HEADING.finditer` skips what it cannot match, so a newest
    heading that does not parse was stepped over and `[0]` became an OLDER
    one. Usually that reds for the wrong reason — it names the wrong version.
    But when the older heading happens to name the current release segment,
    which is exactly the state a version bump passes through, every assertion
    downstream is satisfied by a heading that is not the newest and the file
    is accepted with a malformed heading standing above it.

    The plant below is that state, minimally: a `## ` line with no em dash at
    all, above a well-formed `## 0.2.0 — 2026-08-25`, read against `0.2.0`.
    Both readings are taken here rather than described, so the difference is a
    measurement:

    * the OLD reading — `headings(plant)[0]` — is `("0.2.0", "2026-08-25")`,
      i.e. the second heading, and every check downstream passes;
    * the NEW reading refuses, because it locates the first `## ` line and
      *then* parses it.

    That is the same order `.github/workflows/release.yml`'s changelog step
    uses, deliberately, and the two readers now agree about what "newest"
    means.
    """
    plant = "## 0.2.0 no em dash here\n\n## 0.2.0 — 2026-08-25\n"

    # THE OLD READING, taken rather than remembered. `headings` still exists
    # and still skips; keeping it is what lets this test be a measurement of
    # the difference instead of a story about it.
    stepped_past = headings(plant)
    assert stepped_past and stepped_past[0] == ("0.2.0", "2026-08-25"), (
        "the premise of this test is that `headings()[0]` steps past a "
        "malformed newest heading and returns an older one; it did not, so "
        "either `_HEADING` or the plant has changed and this test is no "
        "longer measuring what it says"
    )

    # ... and every downstream assertion is satisfied by that older heading,
    # which is what makes the old behaviour a GREEN and not a RED.
    older_version, older_rest = stepped_past[0]
    assert older_version == release_segment("0.2.0")
    assert _ISO_DATE.match(older_rest.strip())

    # THE NEW READING refuses it, by its own exception type.
    with pytest.raises(MalformedNewestHeading) as caught:
        newest_heading(plant)
    assert "no em dash here" in str(caught.value), caught.value

    # An empty file and a headingless file are the OTHER case, and they stay
    # distinguishable: `None`, not an exception.
    assert newest_heading("") is None
    assert newest_heading("nothing here\n\nnot a heading either\n") is None
