# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""How many entries of `SOUNDNESS.md`'s `## Log` reach a RELEASE — counted.

**THE DIGIT HAS HELD NINE VALUES.** *"(no releases yet)"*, then S11
alone, then three, then four and five at the same moment on two branches,
then six, then seven, then nine, then eleven, and now twelve. The first
six were
corrections made by a person re-reading the log; the last three were not
— `Versions:` fields moved and the derived count followed
them, which is the paragraph after
next. Not one of the five noticed that the 2026-08-15 B6 entry for audit
finding **S12&prime;** had carried the headline `PRESENT IN THE RELEASED
0.1.0` since `96ab47a` — the commit that added the entry, and left the
digit at three. The omission was live through every correction, and
S12&prime; is the BROADER of the pair it belongs to: S12 is an equation
the interval transfer refuses outright, S12&prime; is an equation neither
leg refused, covering `reduce_sum` as well as `dot_general`, and S12's own
entry defers to it.

A number that has been wrong five times and re-read by hand five times is
not a number to correct a sixth time. It is one to derive, and that is
what this file does.

**WHY IT COULD NOT BE DERIVED BEFORE.** The reach declaration was free
prose. Nine spellings were in use across the log — *"PRESENT IN THE
RELEASED 0.1.0"*, *"PRESENT IN THE RELEASED `v0.1.0`"*, *"REACHES THE
RELEASED 0.1.0"*, *"present in the released tag"*, *"`v0.1.0` included and
reproduced at the tag"*, and four bullets that said it only inside a
measurement paragraph — and NINE of the 54 top-level bullets declared
nothing at all, leaving their reach to be inferred from a batch label in
the headline. Nothing countable, so nothing counted.

Every top-level `## Log` bullet now carries exactly ONE `Versions:` field,
drawn from a closed set of THREE phrases. Three and not two, because the
two the changelog uses cannot express a defect that was fixed before
`v0.1.0` was tagged, which is what 33 of these bullets are about.

**AND A FIELD THIS FILE ACCEPTS CAN STILL BE FALSE.** Every check here is
over what the bullets DECLARE. Six bullets dated after the 2026-08-12 tag
declared *0.1.0 pre-release builds only* — which says their event was over
before it — from the day the fields were introduced until 2026-08-24, and
this file was green throughout, because a closed set holds the vocabulary
and not the truth. Four of the six were 0.2.0 development work and two
reach `v0.1.0`; that correction is what moved the entries count from seven
to nine.

**AND THE SECOND HALF OF THAT IS THE HALF NO CHECK CAN REACH.** Those six
are visible to a comparison between a bullet's DATE and its field: a
post-tag date cannot have an event that was over pre-tag — **a comparison
this docstring DESCRIBES and this file implements NOWHERE**, said here
because describing one reads as having one. Driven on 2026-08-25:
re-scoping a post-tag bullet back to *0.1.0 pre-release builds only*, the
exact shape of the six, leaves the zero-dep lane at **2321 passed, 184
skipped** BOTH WAYS, byte-identical summary lines.
THREE more moved
on 2026-08-25 — S12&Prime;, M17 and the 2026-08-18 query-identity entry —
and they are the mirror: post-tag
dates carrying *0.2.0 development builds only* for defects the released
`v0.1.0` already had. Nothing about those bullets is internally
inconsistent, so the date comparison is structurally silent over them;
they were found by driving each entry against an extracted `v0.1.0`. That
correction moved the entries count from nine to twelve — in TWO passes,
nine to eleven and then eleven to twelve, the second found by a blinded
audit of the first, one bullet further down the same page. It is why the
paragraph above says a closed set holds the vocabulary and not the truth
rather than saying the fields have now been checked, and it is why this
file now holds the CHANGELOG disclosures instead: what a document check
can reach is whether a reader is TOLD, never whether the telling is true.

**THE THREE UNITS, WHICH ARE 12, 12 AND 10.** They are different questions
and this file keeps them apart, because the count was wrong twice by
answering one of them with another's answer — and the first two units
being EQUAL today makes that easier to do, not harder:

* **entries** — top-level `## Log` bullets whose `Versions:` field names
  `v0.1.0`. Twelve.
* **findings** — audit 0.2.0 findings reaching `v0.1.0`. Twelve: the same
  NUMBER as the entries and a different SET. S14 has a routed detail
  section (`SF-0.2.0-59`) and its own reach declaration but no `## Log`
  bullet; the 2026-08-18 query-identity entry has a bullet and is a B13
  instrument fix rather than an audit finding. The two differences cancel
  in the digit and nowhere else.
* **one-liners** — `CHANGELOG.md` entries carrying the `v0.1.0` version
  field IN `### Soundness fixes`. Ten: two fewer than the findings, over
  four terms and not three. S15 and S16 share `SF-0.2.0-14` (−1); M12 and
  M17 have no soundness-fix one-liner at all (−2) — their bullets are in
  *Float32 / float16 / bfloat16 IEEE mode* and *Verification pipeline*,
  and both carry the version field there, where this count does not look
  and a changelog-only reader does; and `SF-0.2.0-11` carries the field
  without being an audit finding at all (+1).

Each numeral in the paragraph is compared against the thing it counts,
and the findings numeral against the list written beside it in that same
paragraph — so a finding added to the list without the digit, or a digit
moved without the list, is a failure either way round.

**AND THE TWO DISCLOSURES THE ONE-LINER COUNT DELIBERATELY DOES NOT LOOK
AT ARE HELD BY THEIR OWN CHECK NOW.** M12's and M17's `v0.1.0`
disclosures were added by hand on 2026-08-25 at a site no derivation
reached: not their phrase (a fourth spelling would have passed), not their
presence (deleting both would have passed), not agreement with the
`## Log` bullets they mirror. A page whose whole argument is that a
hand-maintained figure beside a derived one rots does not get to add two
hand-maintained disclosures and call the gap closed, so
`test_the_v010_disclosures_outside_the_routed_sections_are_a_partition`
makes them a two-way partition against the paragraph that names them.

**AND ONE CHECK BELOW IS ABOUT THE SHAPE OF A CITATION AND NOT ABOUT THE
TELLING — WHICH IS LESS THAN IT WAS SOLD AS.**
`test_a_claim_about_the_tags_TREE_is_decided_against_the_tag` reads the
nine post-tag bullets carrying *0.2.0 development builds only* and the two
`CHANGELOG.md` paragraphs that make the same move for a whole section, and
asserts that each is FORMATTED a certain way: it carries at least one
`v0.1.0:<path>` token or one backticked sha that is not an ancestor of the
tag — anywhere in it, for any reason — and every path token it does carry
agrees with `git cat-file -e` under a polarity read off four literal
phrases. **A citation-FORMAT rule, not an evidence rule**, and the
difference is measured rather than argued: run over the three document
states in which a false `Versions:` field actually shipped, it is GREEN on
all NINE false fields this project has ever found, and the four paragraphs
it does fire on across those states are all TRUE. This paragraph used to
say that the decidable sub-case *"is the justification that cost passes 3,
4 and 5"*. It was ONE of the two justifications pass 5's entry offered, in
a sentence naming no path — which is exactly the half `git cat-file -e`
cannot reach — and it was not the justification of passes 3 or 4 at all.
The nine-of-nine table is in that check's own docstring.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import warnings

import pytest

from _release_record import (  # noqa: E402
    VERSION_FIELDS,
    release_prose,
    release_records,
)
from _soundness_routing_manifest import SECTIONS  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
SOUNDNESS = REPO / "SOUNDNESS.md"
CHANGELOG = REPO / "CHANGELOG.md"

#: The closed set. A `## Log` bullet answers the reach question with
#: exactly one of these and with nothing else, and `REACHES_V010` is the
#: one the count is over.
#: DERIVED from `_release_record.VERSION_FIELDS` and not retyped: the Log
#: writes the same sentence in italics, so the two closed sets are ONE set
#: and cannot drift into disagreeing about the wording.
REACH_FIELDS = tuple(f"*{phrase}*" for phrase in VERSION_FIELDS)
PRE_RELEASE_ONLY, DEV_ONLY, REACHES_V010 = REACH_FIELDS

#: The `CHANGELOG.md` one-liner field for the same fact, unitalicised.
LINER_V010 = VERSION_FIELDS[2]

_NUMBER_WORD = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
}

#: THE CANONICAL SPELLING — `is reached by **N** ENTRIES`. The log has to go
#: on stating the figure in a form a reader can find, and this is the form.
_ENTRIES_CANONICAL_RE = re.compile(
    r"is\s+reached\s+by\s+\*\*(?P<n>[A-Za-z]+|\d+)\*\*\s+ENTRIES", re.S
)

#: EVERY SPELLING, WHICH IS THE ONE THAT MATTERS, AND THE CANONICAL PATTERN
#: ABOVE WAS THE WHOLE CHECK UNTIL 2026-08-25. The record stated this figure
#: TWICE — `is reached by **twelve** ENTRIES of this log` in the
#: reached-release paragraph, and *"Twelve ENTRIES of this log reach
#: `v0.1.0`"* three paragraphs down in the three-units paragraph — and only
#: the first spelling was read. Driven at `b0df85e`: the second set to
#: `Fourteen` left this file at **`11 passed`** and six doc-relevant files
#: at **`140 passed`**, while the OTHER TWO numerals of that paragraph (the
#: findings count and the one-liner count) each give `1 failed, 10 passed`
#: when falsified the same way. A figure that appears twice and agrees with
#: itself but not with the file is the exact shape
#: `test_the_logs_stated_population_is_the_one_the_two_rules_measure`'s
#: docstring legislates against, and it was left standing here.
#:
#: So the unit phrase is what is anchored on — `N ENTRIES of this log` —
#: rather than one sentence's framing of it, and every occurrence is read.
#: The numeral may be bare or bolded; `_numeral` returns `None` for a word
#: that is not a number, which fails the comparison, so an unrecognised
#: spelling reds rather than passing.
_ENTRIES_RE = re.compile(
    r"\*?\*?(?P<n>[A-Za-z]+|\d+)\*?\*?\s+ENTRIES\s+of\s+this\s+log", re.S
)

#: A QUOTED historical value, which this page must be able to write: the
#: digit has held nine values and the paragraph that lists them quotes some
#: of them. `*"…"*` is how this record quotes a retired sentence, and it is
#: the same convention
#: `test_the_release_record_does_not_state_the_old_count_unquoted` already
#: relies on with its `(?<![\"*])` lookbehind. Blanked rather than deleted —
#: every character except a newline becomes a space — so offsets and line
#: numbers into the blanked copy are still the file's own.
#:
#: **AND BLANKING IS A HIDING PLACE UNLESS IT IS PAIRED, WHICH IT WAS NOT.**
#: A quotation is a SPAN and not a speech act: the sentence around it can
#: present the blanked text as current, and nothing here reads it. Driven at
#: `e94f4ea` — `*"Fourteen ENTRIES of this log"* is the count as of
#: 2026-08-25.` appended to both lines that carry the unit phrase, so the
#: file's physical line count does not move — **141 passed** over the six
#: doc-relevant files and **12 passed** over this one. The pairing is
#: `test_no_quotation_carries_the_ENTRIES_figure_the_comparison_cannot_read`
#: below, and what it costs is stated there.
_QUOTED_RE = re.compile(r'\*"[^"]*"\*', re.S)


def _live(text: str) -> str:
    """`text` with every `*"…"*` quotation blanked, offsets preserved."""
    return _QUOTED_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


_FINDINGS_RE = re.compile(
    r"\*\*(?P<n>[A-Za-z]+|\d+)\*\*\s+audit\s+findings\s+reach\s+it:"
    r"\s*(?P<list>[^.]+?)\s*—",
    re.S,
)
_LINERS_RE = re.compile(
    r"\*\*(?P<n>[A-Za-z]+|\d+)\*\*\s*\n?\s*one-liners?\s+in\s+`CHANGELOG\.md`"
    r"\s+carr(?:y|ies)\s+the\s+`v0\.1\.0`\s+version\s+field",
    re.S,
)

#: A finding that reaches `v0.1.0` and has NO `### Soundness fixes`
#: one-liner, together with the `CHANGELOG.md` section its bullet lives in
#: — as the one-liner paragraph itself writes the pair. Read out of the
#: page rather than listed here: a list here would be a fourth
#: hand-maintained copy of the fact this file exists to stop copying.
#: Run over a WHITESPACE-NORMALISED paragraph — the page wraps at 72
#: columns and both of today's two sentences wrap mid-clause, so a pattern
#: that reads the file's own line breaks matches one of them and not the
#: other, which is worse than matching neither.
_DISCLOSED_ELSEWHERE = re.compile(
    r"(?P<who>[A-Za-z0-9&;]+)'s bullet is in \*(?P<sec>[^*]+)\*"
)


def _numeral(word: str) -> int | None:
    return int(word) if word.isdigit() else _NUMBER_WORD.get(word.lower())


#: The preamble — everything between `## Log` and its first entry — is prose
#: about what the fields mean, so it is not an entry and is not counted. It is
#: not UNBOUNDED either: an unbounded preamble is a place an entry can be
#: grown where no per-entry check looks. Sixty non-blank lines, the same
#: ceiling and the same argument as `tests/test_soundness_routing.py`'s
#: `_PREAMBLE_MAX_LINES` on the changelog side.
#:
#: **THE NUMBER IS PINNED BY THE DOCUMENT AND NOT BY THIS COMMENT**, in
#: `test_an_ENTRY_CANNOT_BE_HIDDEN_IN_THE_LOGS_PREAMBLE` — every other leg
#: down there derives its plant FROM this constant, so until 2026-08-22 the
#: constant checked itself and `60 -> 6000` was green. The argument for the
#: value is that the HEADROOM above today's preamble admits an explanation
#: and refuses a typical entry, and that argument is measured at both ends
#: down there: **the test below re-derives today's preamble length, the
#: median `## Log` entry and the headroom between them, and asserts the
#: relations.**
#:
#: **NO FIGURE IS TYPED HERE ANY MORE, BECAUSE THE TYPED ONES WENT STALE
#: TWICE IN THREE DAYS AND THIS COMMENT SAID SO WHILE BEING ONE OF THEM.**
#: It carried the preamble length and the headroom as digits, two lines
#: under its own warning that both "moved under it twice in three days".
#: Measured on `SOUNDNESS.md` at each commit that touched this file, the
#: preamble has been 21 non-blank lines (`68b219d`, `1f55eef`), 34
#: (`1242da4`), 42 (`c198a8d`) and 48 (`161ead8`) — and the comment read 21
#: at `1242da4` and 42 at `161ead8`, wrong at both, the second time written
#: by the pass that added six preamble lines and left the numeral alone. In
#: the file whose subject is derived-versus-typed figures, an illustration
#: standing beside a derivation is the defect and not a convenience, so it
#: is gone: the assertions below print all three values in their failure
#: messages, which is where a reader who needs a live one should get it.
_PREAMBLE_MAX_LINES = 60

#: A list item at column 0. `- ` is an entry; `* ` and `+ ` are the other two
#: markdown spellings of the same thing and are what an entry hidden in the
#: preamble would be written as, since a `- ` there would simply become the
#: first entry and be counted.
_PREAMBLE_BULLET = re.compile(r"^[-*+] ", re.M)


def log_bullets(text: str | None = None) -> list[tuple[int, str]]:
    """`(line, text)` for every top-level bullet of `SOUNDNESS.md`'s `## Log`.

    Top-level means column 0 — the same rule
    `tests/test_soundness_routing.py::split_blocks` uses on the changelog,
    for the same reason: a nested bullet is part of its parent's argument
    and is not an entry.

    **`text` IS THE DOCUMENT TO READ, DEFAULTING TO `SOUNDNESS.md`, AND IT
    EXISTS SO THE PREAMBLE RULE BELOW IS DRIVEN THROUGH THIS FUNCTION.**
    `test_an_ENTRY_CANNOT_BE_HIDDEN_IN_THE_LOGS_PREAMBLE` re-implemented the
    three legs inline on a synthetic section — its docstring said *"the three
    legs of `log_bullets`'s preamble rule, driven"* and a COPY of them was
    what got driven. Measured: deleting all three legs from here left
    `tests/test_soundness_log_reach.py` at **`9 passed`**, the pin green over
    the rule's absence. A test that edits the file it reads can leave the tree
    changed, so a parameter is how the real rule meets a synthetic section.

    **THE PREAMBLE IS HELD SHUT HERE AND NOT IN A TEST BESIDE IT**, because
    every count in this file is derived from what this function returns and a
    section that can hide an entry above the first bullet makes all of them
    wrong at once. Until 2026-08-22 everything before the first column-0
    `- ` was unread and unbounded: driven, a fully-formed reaching entry
    inserted there as a `*` bullet gave **`8 passed`** with the derived count
    still reading seven while eight entries declared reach, and the same
    thing shaped as a paragraph did too. The commit that wrote this file
    closed this exact shape on the CHANGELOG side, with
    `_PREAMBLE_MAX_LINES = 60` and the sentence *"an unbounded preamble is a
    place an entry can be re-grown where no per-entry check looks"*, and did
    not apply it here.

    Three legs, because a ceiling alone bounds the size of the hole rather
    than closing it: no reach FIELD may stand in the preamble (that is the
    marker an entry is identified by, and prose about the fields quotes them
    in backticks rather than writing one), no column-0 list marker of any
    spelling may either, and the whole preamble is bounded.
    """
    if text is None:
        text = SOUNDNESS.read_text(encoding="utf-8")
    lines = text.split("\n")
    lo = next(i for i, line in enumerate(lines, 1) if line.rstrip() == "## Log")
    hi = next(
        i for i, line in enumerate(lines, 1) if i > lo and line.startswith("## ")
    )
    starts = [i for i in range(lo + 1, hi) if lines[i - 1].startswith("- ")]
    assert starts, "SOUNDNESS.md's `## Log` holds no top-level bullet"
    preamble = lines[lo:starts[0] - 1]
    # NOT `text`. This rebound the PARAMETER, so from here down the name no
    # longer meant the document the caller passed — harmless while nothing
    # below reads it, and a trap the first time something does.
    preamble_text = "\n".join(preamble)
    declared = [field for field in REACH_FIELDS if field in preamble_text]
    assert not declared, (
        f"`## Log`'s preamble (lines {lo + 1}-{starts[0] - 1}) carries a "
        f"reach field, {declared}. Everything above the first column-0 `- ` "
        f"is outside every per-entry check here and outside the derived "
        f"count, so an entry written there declares a release reached and is "
        f"counted nowhere."
    )
    stray = [line for line in preamble if _PREAMBLE_BULLET.match(line)]
    assert not stray, (
        f"`## Log`'s preamble holds a column-0 list item: {stray[:3]}. An "
        f"entry belongs below the first `- `, where it is counted; a `* ` or "
        f"`+ ` above it is an entry the count cannot see."
    )
    body = [line for line in preamble if line.strip()]
    assert len(body) <= _PREAMBLE_MAX_LINES, (
        f"`## Log`'s preamble is {len(body)} non-blank lines and the ceiling "
        f"is {_PREAMBLE_MAX_LINES}. It is the one place in this section prose "
        f"can grow without any per-entry check here seeing it."
    )
    out = []
    for n, s in enumerate(starts):
        e = (starts[n + 1] - 1) if n + 1 < len(starts) else hi - 1
        out.append((s, "\n".join(lines[s - 1:e])))
    return out


def test_an_ENTRY_CANNOT_BE_HIDDEN_IN_THE_LOGS_PREAMBLE():
    """The three legs of :func:`log_bullets`'s preamble rule, driven —
    THROUGH IT.

    `log_bullets()` starts at the first column-0 `- ` after `## Log`, so
    everything above it was unread and unbounded until 2026-08-22. Driven on
    the real file: a fully-formed reaching entry inserted there — as a `*`
    bullet, and again as a paragraph — ran **`8 passed`**, with the derived
    reached-release count still reading seven while eight entries declared
    reach. The count is the whole point of this file.

    **THIS PIN DID NOT DRIVE THE RULE IT PINS.** It re-implemented the three
    legs inline — the same `starts` scan, the same body slice, the same
    field, bullet and size readings — and asserted on the COPY. Measured:
    deleting the entire preamble guard from `log_bullets()` left this file at
    **`9 passed`**. A pin over a re-implementation goes green over the
    absence of the thing it pins, which is the defect class this file's own
    subject is about.

    So each leg is now a synthetic section handed to `log_bullets()` itself,
    which is what the `text` parameter is for — a test that edits the file it
    reads can leave the tree changed. **ONE LEG PER SECTION**, because the
    real function reports the FIRST leg that fires: the field leg and the
    bullet leg both catch the realistic shape (a `*` bullet carrying a reach
    field), and asserting both are live needs them isolated. The fourth
    section is the control that keeps the three reds non-vacuous — a
    preamble breaking none of the rules is ACCEPTED and its entry comes back.
    """
    def section(preamble: str) -> str:
        return "\n".join(["## Log", ""] + preamble.split("\n") + [
            "- **an entry.** body.",
            "",
            f"  {REACHES_V010}",
            "",
            "## After",
        ])

    def refusal(preamble: str) -> str | None:
        """`log_bullets`'s complaint about this preamble, `None` if it
        took it."""
        try:
            bullets = log_bullets(section(preamble))
        except AssertionError as exc:
            return str(exc)
        assert len(bullets) == 1, (
            f"the synthetic section did not parse: {bullets}"
        )
        return None

    # THE CONTROL FIRST: a preamble that breaks none of the three is taken,
    # so every red below is the leg named and not the scaffolding.
    control = refusal("prose about what the `*Versions:*` fields mean.")
    assert control is None, (
        "the synthetic section this test is built on is refused before any "
        "leg is planted, so the three reds below would prove nothing"
    )

    # LEG 1, the reach FIELD, isolated: a paragraph, so no list marker.
    field_only = refusal(
        f"**a planted entry, as a paragraph.** body. {REACHES_V010}"
    )
    assert field_only and "reach field" in field_only, (
        "a paragraph carrying a reach field in the preamble is not refused by "
        f"the field leg: {field_only!r}"
    )

    # LEG 2, the column-0 list MARKER, isolated: no reach field on it.
    bullet_only = refusal("* **a planted entry.** body, declaring nothing.")
    assert bullet_only and "column-0 list item" in bullet_only, (
        "a `* ` entry above the first `- ` is not refused by the bullet leg: "
        f"{bullet_only!r}"
    )

    # LEG 3, the CEILING, isolated: neither a field nor a marker in it.
    ceiling = refusal("\n".join(
        f"line {n} of a preamble nobody bounded"
        for n in range(_PREAMBLE_MAX_LINES + 1)
    ))
    assert ceiling and "ceiling" in ceiling, (
        f"an unbounded preamble is not refused by the ceiling leg: {ceiling!r}"
    )
    # ... and the ceiling counts NON-BLANK lines, so it cannot be walked past
    # with blank ones: the same body, blank-separated, is longer in lines and
    # still refused.
    padded = refusal("\n\n".join(
        f"line {n} of a preamble nobody bounded"
        for n in range(_PREAMBLE_MAX_LINES + 1)
    ))
    assert padded and "ceiling" in padded, (
        f"blank lines walk the preamble past its ceiling: {padded!r}"
    )
    # ... and one line under it is accepted, so the ceiling is a ceiling and
    # not a refusal of every preamble.
    assert refusal("\n".join(
        f"line {n} of a preamble somebody bounded"
        for n in range(_PREAMBLE_MAX_LINES)
    )) is None, "the ceiling refuses a preamble that is inside it"

    # ... and the realistic shape — a `*` entry that also declares reach —
    # is refused, which is what both isolated legs are there to make sure of.
    assert refusal(
        "* **a planted entry.** body.\n"
        f"  {REACHES_V010}"
    ), "a `*` bullet carrying a reach field in the preamble is accepted"

    # AND THE CEILING ITSELF IS PINNED, WHICH IT WAS NOT. Every leg above
    # derives its plant FROM `_PREAMBLE_MAX_LINES`, so the constant validated
    # itself: driven, `60 -> 6000` left this file and
    # `tests/test_soundness_routing.py` at **38 passed** together, with a
    # ceiling six thousand lines above a twenty-one-line preamble. A number
    # that only ever appears on both sides of the comparison is not a bound.
    #
    # So it is pinned against the DOCUMENT, in the arithmetic its own comment
    # states: the ceiling "admits an explanation and refuses an entry" means
    # the HEADROOM above today's preamble is smaller than an entry, and both
    # of those are measured here rather than written down.
    lines = SOUNDNESS.read_text(encoding="utf-8").split("\n")
    lo = next(i for i, line in enumerate(lines, 1) if line.rstrip() == "## Log")
    first = next(
        i for i in range(lo + 1, len(lines) + 1)
        if lines[i - 1].startswith("- ")
    )
    today = len([line for line in lines[lo:first - 1] if line.strip()])
    sizes = sorted(
        len([line for line in body.split("\n") if line.strip()])
        for _, body in log_bullets()
    )
    median = sizes[len(sizes) // 2]
    assert today < _PREAMBLE_MAX_LINES, (
        f"`## Log`'s preamble is {today} non-blank lines and the ceiling is "
        f"{_PREAMBLE_MAX_LINES}: there is no headroom, so the ceiling refuses "
        f"the document as it stands"
    )
    headroom = _PREAMBLE_MAX_LINES - today
    assert headroom < median, (
        f"the preamble ceiling leaves {headroom} lines of headroom above "
        f"today's {today}-line preamble, and the median `## Log` entry is "
        f"{median} non-blank lines — so a typical entry FITS in the headroom "
        f"and the ceiling no longer refuses one. That is the whole argument "
        f"for the number; raise the ceiling and this is what stops being true"
    )
    # ...and said rather than left to be found: the SHORTEST entry in the log
    # is six non-blank lines and DOES fit in the headroom. The ceiling bounds
    # unbounded growth of the preamble; the reach-field and column-0-marker
    # legs above are what catch a small entry hidden in it, and neither
    # depends on this number. The assertion below is that measurement, so
    # the day it stops being true the comment gets rewritten rather than
    # quietly overstating what the ceiling does.
    assert sizes[0] < headroom, (
        "the shortest `## Log` entry no longer fits in the preamble's "
        "headroom, so the comment above overstates what the other two legs "
        "are needed for — which is now the safe direction, and worth "
        "rewriting the comment for rather than leaving it wrong"
    )

    # ... and the real preamble passes all three, which is what makes the
    # rule a rule rather than a thing that would have to be relaxed at once.
    assert log_bullets(), "the real `## Log` no longer parses"


def test_every_log_bullet_answers_the_reach_question_exactly_once():
    """The closed set, enforced. This is what makes the count countable.

    A bullet with no field leaves its reach to be inferred from a batch
    label — which is how nine of these bullets stood, and inference is
    not a thing a test can do. A bullet with two fields says two
    different things about the same release.
    """
    bad = []
    for line, text in log_bullets():
        hits = [f for f in REACH_FIELDS if f in text]
        if len(hits) != 1:
            bad.append((line, text.split("\n")[0][:70], len(hits)))
    assert not bad, (
        f"{len(bad)} `## Log` bullet(s) do not carry exactly one of the "
        f"three permitted `Versions:` fields: {bad[:6]}. The fields are "
        f"{REACH_FIELDS}; the count of releases reached is derived from "
        f"them, so a bullet that does not answer makes the count a guess."
    )


def test_the_reached_release_count_is_the_number_of_entries_that_declare_it():
    """The ENTRIES numeral — EVERY spelling of it, in EVERY file of the
    record.

    This digit read **six** against seven declaring bullets, having
    already read *"(no releases yet)"*, S11 alone, three, and four-and-
    five on two branches at once.

    **AND THE RECORD STATES IT TWICE AND THIS CHECK READ IT ONCE**, which
    is what 2026-08-25 fixed. `_ENTRIES_RE` matched
    `is reached by **N** ENTRIES` and nothing else; the three-units
    paragraph three paragraphs below the reached-release one restates the
    same unit as *"Twelve ENTRIES of this log reach `v0.1.0`"*, and
    `re.search` never reached it. **Driven before the fix, on a clean
    checkout of `b0df85e`**: that second numeral set to `Fourteen` left
    this file at **`11 passed`** and `test_soundness_routing`,
    `test_soundness_log_reach`, `test_sdist_reference_hygiene`,
    `test_release_doc_claims`, `test_prose_hygiene` and
    `test_doc_examples` together at **`140 passed`** — while falsifying
    either of the other two numerals in that same paragraph gives
    `1 failed, 10 passed`. One numeral of three, unheld, in the paragraph
    whose entire subject is that these three counts get mistaken for one
    another.

    So the anchor is the UNIT PHRASE — `N ENTRIES of this log` — and not
    one sentence's framing of it, every occurrence is compared, and the
    comparison is per FILE through `release_records()` for the reason
    `tests/_release_record.py` gives: over a concatenation `re.search`
    returns the first match, so a correct copy in one file masks a stale
    copy in the other. A `*"…"*` quotation is blanked first, because the
    paragraph on this digit's history has to be able to quote the values
    it has held — **and that blanking is a hiding place on its own**, which
    is why
    `test_no_quotation_carries_the_ENTRIES_figure_the_comparison_cannot_read`
    stands beside it. Driven at `e94f4ea`, a `Fourteen` inside quotation
    marks beside both correct spellings ran `141 passed`.

    The canonical spelling is required to survive as well: a record that
    stops saying `is reached by **N** ENTRIES` has not stopped having the
    number, it has stopped letting a reader find it.
    """
    entries = [
        line for line, text in log_bullets() if REACHES_V010 in text
    ]

    canonical, wrong = [], []
    for name, text in release_records():
        live = _live(text)
        canonical += [
            (name, m.group("n")) for m in _ENTRIES_CANONICAL_RE.finditer(live)
        ]
        for m in _ENTRIES_RE.finditer(live):
            if _numeral(m.group("n")) != len(entries):
                wrong.append((
                    name,
                    live.count("\n", 0, m.start()) + 1,
                    m.group("n"),
                    " ".join(live[m.start():m.end() + 40].split()),
                ))

    assert canonical, (
        "the release record no longer states how many ENTRIES of the log "
        "reach a release, in the form `is reached by **N** ENTRIES`. "
        f"{len(entries)} do. A log that stops stating it has not stopped "
        "having the number; it has stopped letting a reader check it."
    )
    assert not wrong, (
        f"the release record states the ENTRIES figure {len(wrong)} time(s) "
        f"as something other than the {len(entries)} bullets that carry "
        f"{REACHES_V010!r} (lines {entries}): {wrong}. Every spelling of "
        f"`N ENTRIES of this log` is read, in every file of the record — "
        f"the second spelling of this figure went unread until 2026-08-25 "
        f"and `Fourteen` was green in it. The omitted ENTRY was S12&prime; "
        f"for five corrections running."
    )


def test_no_quotation_carries_the_ENTRIES_figure_the_comparison_cannot_read():
    """The pairing :func:`_live` needed: a live claim may not wear quotation
    marks.

    :func:`_live` blanks every `*"…"*` before the comparison above, so this
    record can go on quoting a retired sentence about a digit that has held
    nine values. What it also does is make a figure INSIDE a quotation
    invisible to the only check that reads figures — and a quotation is a
    SPAN, not a speech act, so the sentence around it can present the
    blanked text as current.

    **DRIVEN AT `e94f4ea`**, on the real `SOUNDNESS.md`, appended to both
    lines carrying the unit phrase so the file's physical line count does
    not move — the Log-population check above reads it:
    `*"Fourteen ENTRIES of this log"* is the count as
    of 2026-08-25.` beside both correct spellings ran **141 passed** over
    the six doc-relevant files and **12 passed** over this one, with the
    file restored byte-identically (sha256) after and `141 passed` again on
    the restored file. A wrong figure, presented as today's, nothing red —
    and the leg immediately above, which this file spent a commit
    teaching to read the SECOND spelling, never saw it.

    **AND ONE COMMIT ANSWERED THE SAME QUESTION BOTH WAYS.** `e94f4ea`
    blanks quotations for the figure check and READS them in the citation
    check: `_TAG_PATH_RE` and `_TAG_PATH_ABSENT_RE` run over raw text, so a
    `v0.1.0:<path>` in quotation dress is decided like any other and reds
    if its polarity disagrees. Two sibling checks over the same two files,
    opposite conventions, and nothing anywhere saying which was right.

    So the blanking STAYS — the record must be able to quote its own
    history, and no reader of a span can tell a quotation from a claim —
    and it is PAIRED here: the unit phrase may not appear inside a
    quotation at all, because inside one nothing reads it.

    **WHAT THAT COSTS, said rather than left to be found.** A retired
    sentence containing the full unit phrase can no longer be quoted
    verbatim in the record. Nothing does today, measured: neither file
    carries a single `*"…"*` quotation containing the phrase, and the
    digit's history is written as the VALUES it has held rather than as
    quoted sentences. `test_the_release_record_does_not_state_the_old_count_unquoted`
    still licenses quoting the canonical FRAMING (`is reached by **six**
    ENTRIES`), which is the shape that check reads.
    """
    hidden = []
    for name, text in release_records():
        for q in _QUOTED_RE.finditer(text):
            flat = " ".join(q.group(0).split())
            for m in _ENTRIES_RE.finditer(flat):
                hidden.append(
                    (name, text.count("\n", 0, q.start()) + 1, m.group("n"), flat)
                )
    assert not hidden, (
        f"{len(hidden)} quotation(s) in the release record carry the "
        f"`N ENTRIES of this log` figure, where `_live()` blanks them and "
        f"the comparison above cannot read them:\n  "
        + "\n  ".join(
            f"{name}:{line}: {n!r} in {flat[:90]}" for name, line, n, flat in hidden
        )
        + "\n\nA quotation is a span and not a speech act: the sentence around "
        "it can assert the blanked text as current, and driven at `e94f4ea` "
        "one did — `Fourteen`, beside both correct spellings, 141 passed over "
        "the six doc-relevant files. Write the figure unquoted, where the "
        "comparison reads it, or write the retired sentence without the unit "
        "phrase."
    )


def test_the_finding_count_is_the_length_of_the_list_written_beside_it():
    """The FINDINGS numeral, against the list in its own sentence.

    Findings are not entries — S15 and S16 are two findings in two
    entries and one changelog one-liner, and S14 is a finding with a
    detail section and no entry — so this count cannot be taken from the
    log's structure. What it CAN be taken from is the enumeration written
    next to it, and that is what makes moving one without the other a
    failure.
    """
    soundness = SOUNDNESS.read_text(encoding="utf-8")
    m = _FINDINGS_RE.search(soundness)
    assert m, (
        "SOUNDNESS.md no longer states how many audit FINDINGS reach a "
        "release, in the form `**N** audit findings reach it: <list> —`. "
        "The entries count and the findings count differ, and dropping "
        "the second is how one gets read as the other."
    )
    named = [x.strip() for x in re.split(r",\s*", m.group("list")) if x.strip()]
    stated = _numeral(m.group("n"))
    assert stated == len(named), (
        f"SOUNDNESS.md says {m.group('n')!r} audit findings reach `v0.1.0` "
        f"and names {len(named)}: {named}"
    )
    assert "S12&prime;" in named or "S12'" in named, (
        f"the findings list does not name S12&prime;: {named}. That is the "
        f"exact omission this check exists for — the broader of the two "
        f"B6 shape findings, left out of the count from `96ab47a` until "
        f"2026-08-21."
    )


def test_the_one_liner_count_is_the_number_of_one_liners_that_carry_the_field():
    """The ONE-LINER numeral, against `CHANGELOG.md`.

    The third unit, and the one easiest to confuse with the first — they
    were both seven on the day this file was written, and they are 9 and
    11 today. They are not the same
    question: entries are bullets of this log, one-liners are entries of
    the changelog, and S15 and S16 are two of the former and one of the
    latter. It is also the NARROWEST unit: it reads
    `### Soundness fixes` only, so a `v0.1.0` disclosure carried by a
    bullet of another changelog section — M12's and M17's both are — is
    deliberately outside it, and the findings numeral is where those two
    are counted.
    """
    changelog = CHANGELOG.read_text(encoding="utf-8")
    section = next(s for s in SECTIONS if s.key == "soundness")
    body = re.search(
        rf"^{re.escape(section.heading)}\n(.*?)(?=^### )",
        changelog, re.S | re.M,
    )
    assert body, f"CHANGELOG.md has no `{section.heading}` section"
    bullet = re.compile(rf"^- \*\*{re.escape(section.id_prefix)}\d{{2}}\*\*")
    paras, cur = [], []
    for line in body.group(1).split("\n"):
        if line.strip():
            cur.append(line)
        elif cur:
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    carrying = [
        para for para in paras
        if bullet.match(para[0])
        and LINER_V010 in " ".join(x.strip() for x in para)
    ]
    soundness = SOUNDNESS.read_text(encoding="utf-8")
    m = _LINERS_RE.search(soundness)
    assert m, (
        "SOUNDNESS.md no longer states how many `CHANGELOG.md` one-liners "
        "carry the `v0.1.0` version field. It is the third of three counts "
        "that have been mistaken for one another."
    )
    stated = _numeral(m.group("n"))
    assert stated == len(carrying), (
        f"SOUNDNESS.md says {m.group('n')!r} one-liners carry "
        f"{LINER_V010!r} and {len(carrying)} do"
    )


def _one_liner_paragraph(soundness: str) -> str:
    """The blank-line-delimited paragraph that states the one-liner count.

    Scoped to the paragraph and not to the file because
    :data:`_DISCLOSED_ELSEWHERE` is a sentence SHAPE, and the shape is
    cheap enough that a second entry could grow one elsewhere and be read
    as an accounting of the changelog disclosures. The accounting is one
    paragraph; so is the search.
    """
    m = _LINERS_RE.search(soundness)
    assert m, (
        "SOUNDNESS.md no longer states how many `CHANGELOG.md` one-liners "
        "carry the `v0.1.0` version field, so the paragraph that accounts "
        "for the disclosures OUTSIDE `### Soundness fixes` cannot be found "
        "either."
    )
    start = soundness.rfind("\n\n", 0, m.start())
    end = soundness.find("\n\n", m.end())
    return soundness[0 if start < 0 else start:len(soundness) if end < 0 else end]


def _changelog_sections(changelog: str) -> list[tuple[str, str]]:
    """`(### heading, body)` for every `### ` section of `CHANGELOG.md`.

    A `## ` release heading CLOSES the open section rather than being
    swallowed by it: `### Known limitations (0.2.0)` is the last `### ` of
    the 0.2.0 release and the next one belongs to 0.1.0, so a naive split
    on `### ` alone would file every line of the 0.1.0 preamble under a
    0.2.0 heading. Nothing in this tree turns on that today; a section
    reader that is wrong about which release a line is in would be.
    """
    out, heading, body = [], None, []
    for line in changelog.split("\n"):
        if line.startswith("### "):
            if heading is not None:
                out.append((heading, "\n".join(body)))
            heading, body = line.rstrip(), []
        elif line.startswith("## "):
            if heading is not None:
                out.append((heading, "\n".join(body)))
            heading, body = None, []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        out.append((heading, "\n".join(body)))
    return out


def test_the_v010_disclosures_outside_the_routed_sections_are_a_partition():
    """THE `v0.1.0` DISCLOSURES A CHANGELOG-ONLY READER DEPENDS ON, HELD.

    Two audit findings that reach the released `v0.1.0` — M12 and M17 —
    have no `### Soundness fixes` one-liner, so the routing check's rule
    that every one-liner carries a version field from the closed set does
    not reach them. Their bullets live in ordinary feature sections, and
    until 2026-08-25 those sections carried no `Versions:` field at all: a
    reader of `CHANGELOG.md` alone got NO disclosure for either, about an
    artefact that is on PyPI today.

    **THE DISCLOSURES ADDED THAT DAY WERE HELD BY NOTHING**, which is the
    whole reason this test exists. Not their phrase — a fourth spelling of
    the version field would have passed. Not their presence — deleting
    both would have passed, because `### Soundness fixes` carries ten of
    its own and any "somewhere in `CHANGELOG.md`" check reads those.
    Not agreement with the `## Log` bullets they mirror. A page whose
    entire argument is that a hand-maintained figure beside a derived one
    rots does not get to close a gap with two hand-maintained sentences at
    a site no derivation reaches.

    So this is a PARTITION, in both directions, over the sections OUTSIDE
    the routed ones:

    * what `CHANGELOG.md` carries — every `### ` section, other than the
      routed ones, whose body contains :data:`LINER_V010`, which is
      derived from `_release_record.VERSION_FIELDS` and not typed here;
    * what `SOUNDNESS.md` accounts for — every section named by the
      one-liner paragraph's own "X's bullet is in *section*" sentences,
      read off a whitespace-normalised copy of that paragraph.

    Equality, so a disclosure that disappears is caught on one side and a
    disclosure nothing accounts for is caught on the other. The routed
    sections are excluded because their one-liners are already held by
    `tests/test_soundness_routing.py`; a `v0.1.0` disclosure appearing in
    one of them is not this test's business and is not a hole.

    **DRIVEN RED FOUR WAYS**, on the real `CHANGELOG.md`, restored
    byte-identically (sha256 verified) after each and `11 passed` on the
    restored file. Every one is `1 failed, 10 passed`:

    * the `Versions:` sentence DELETED from `### Verification pipeline` —
      names that section as accounted for and absent;
    * the same field REWORDED to a fourth spelling
      (`Versions: v0.1.0 and later.`) — the same red, which is what makes
      the phrase's derivation from `VERSION_FIELDS` load-bearing rather
      than decorative;
    * BOTH disclosures deleted — names both sections, where a
      "somewhere in `CHANGELOG.md`" check would have stayed green;
    * an unaccounted disclosure ADDED to `### SMT emission extensions` —
      names that section as present and unaccounted, which is the
      direction a one-sided check has no way to see.

    **WHAT THIS DOES NOT DO**, said here rather than left to be assumed:
    it does not check that a disclosure is TRUE. `## Log`'s own closing
    paragraphs carry that limit in full — a field of the form *0.2.0
    development builds only* on a defect the tag actually carries is
    consistent with every date and every document in this tree, and only
    running an extracted `v0.1.0` refutes it. This test holds the
    TELLING, which is the half a document check can reach.
    """
    soundness = SOUNDNESS.read_text(encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")

    para = " ".join(_one_liner_paragraph(soundness).split())
    named = {
        m.group("sec").strip(): m.group("who")
        for m in _DISCLOSED_ELSEWHERE.finditer(para)
    }
    assert named, (
        "the `## Log` paragraph stating the one-liner count names no "
        "`CHANGELOG.md` section as carrying a `v0.1.0` disclosure for a "
        "finding with no `### Soundness fixes` one-liner. Two findings are "
        "in that position (M12 and M17) and their disclosures are the only "
        "thing a changelog-only reader has. If that has genuinely stopped "
        "being true — both findings gained a soundness one-liner — retire "
        "this check deliberately and say so; do not let an empty "
        "accounting read as a clean one."
    )

    routed = {section.heading for section in SECTIONS}
    found = {
        heading
        for heading, body in _changelog_sections(changelog)
        if heading not in routed and LINER_V010 in " ".join(body.split())
    }
    want = {f"### {name}" for name in named}
    assert found == want, (
        f"the `v0.1.0` disclosures outside the routed changelog sections "
        f"do not match the accounting in `SOUNDNESS.md`.\n"
        f"  accounted for and ABSENT from CHANGELOG.md: "
        f"{sorted(want - found)}\n"
        f"  present in CHANGELOG.md and UNACCOUNTED for: "
        f"{sorted(found - want)}\n"
        f"The phrase is {LINER_V010!r}, derived from "
        f"`_release_record.VERSION_FIELDS`; the accounting is the "
        f"one-liner paragraph's own \"X's bullet is in *section*\" "
        f"sentences ({named}). A disclosure a changelog-only reader "
        f"depends on is not held by having been typed once."
    )


#: The `### What has been driven in this document` bullet states the `## Log`'s
#: SIZE, as the population a sweep did not read line by line, and it states
#: the two rules it measured that size by. Both spellings are read: the
#: bullet's own headline figure and the parenthetical's three-term
#: arithmetic, which must agree with each other as well as with the file.
_LOG_LINES_RE = re.compile(r"It is \*\*([\d,]+) lines\*\* as")
_LOG_POPULATION_RE = re.compile(
    r"the Log is \*\*([\d,]+)\*\*\s*\n?\s*lines, the routed sections "
    r"\*\*([\d,]+)\*\*, and the difference \*\*([\d,]+)\*\*"
)


def test_the_logs_stated_population_is_the_one_the_two_rules_measure():
    """The Log's stated SIZE, derived — it has now rotted on two
    consecutive days.

    The sentence exists to bound a sweep-reach claim: *"the `## Log` below
    was not read line by line"*, over a population the sentence names. It
    read *"11,000-odd lines"*, which was no revision's figure, and then
    stood at 14,926 while the `## Log` measured 15,439 (`6ecb5cd`). The
    2026-08-25 pass re-derived it to 15,628 by the two rules the
    parenthetical states — physical lines from the `## Log` heading to the
    last line of the file, and `_detail_sections` over the 72 routed
    bodies — and that was correct at `161ead8` and false again the next
    day, because **the figure is only ever right in the commit that
    repairs it** and nothing said when it stopped being. That is the
    third hand-maintained figure this pass met that a later edit had
    falsified. The other two — the preamble length in
    the comment on `_PREAMBLE_MAX_LINES` above and the one-liner split in
    `tests/test_soundness_routing.py` — were repaired by DELETING the
    digit, because nothing at those two sites needed a live value. This
    one is different and gets a PIN instead: the sentence needs a
    population to bound its reach claim, so the digit stays and is held to
    the rules the page itself names for it.

    Both spellings are checked, and against each other: the headline
    figure and the parenthetical's `log − routed = difference`. A figure
    that appears twice and agrees with itself but not with the file is the
    shape `tests/test_soundness_routing.py`'s manifest columns are about.

    Driven RED by adding one to the headline figure: `1 failed, 10
    passed`, naming both the stated and the measured value on both rules,
    with `SOUNDNESS.md` restored byte-identically after.

    `_detail_sections` is IMPORTED from `tests/test_soundness_routing.py`
    rather than re-implemented, for the reason `log_bullets`'s own
    docstring gives about a copy of a rule going green over the rule's
    absence — and because the parenthetical names that function by name,
    so the check has to be over the thing the page cites.
    """
    from test_soundness_routing import _detail_sections

    soundness = SOUNDNESS.read_text(encoding="utf-8")
    lines = soundness.splitlines()
    lo = next(i for i, line in enumerate(lines, 1) if line.rstrip() == "## Log")
    log_lines = len(lines) - lo + 1
    routed = sum(
        len(body.splitlines()) for body in _detail_sections(soundness).values()
    )

    head = _LOG_LINES_RE.search(soundness)
    assert head, (
        "SOUNDNESS.md no longer states the `## Log`'s size in the form `It "
        "is **N lines** as`. The size is the POPULATION a sweep-reach claim "
        "is about, and a reach claim over an unstated population is not one."
    )
    body = _LOG_POPULATION_RE.search(soundness)
    assert body, (
        "SOUNDNESS.md no longer states the `Log / routed / difference` "
        "arithmetic that names the two rules the size is measured by. The "
        "rules are what make the historical figures reproducible."
    )

    def _n(text: str) -> int:
        return int(text.replace(",", ""))

    stated_head = _n(head.group(1))
    stated_log, stated_routed, stated_diff = (_n(g) for g in body.groups())
    assert (stated_head, stated_log, stated_routed, stated_diff) == (
        log_lines, log_lines, routed, log_lines - routed
    ), (
        f"the `## Log` population figures in SOUNDNESS.md do not match the "
        f"two rules the page states for them. Stated: headline "
        f"{stated_head:,}, Log {stated_log:,}, routed {stated_routed:,}, "
        f"difference {stated_diff:,}. Measured: Log {log_lines:,} (physical "
        f"lines from the `## Log` heading to the last line of the file), "
        f"routed {routed:,} (`_detail_sections` over the routed bodies), "
        f"difference {log_lines - routed:,}. Write the measured values, or "
        f"the sweep-reach claim is about a population this file no longer "
        f"has."
    )


def test_the_release_record_does_not_state_the_old_count_unquoted():
    """*"is reached by six entries"* may be quoted; it may not be asserted.

    A count with a history this long gets quoted, and it must be able to
    be: this page's own paragraph on the digit lists every value it has
    held. What it may not do is stand as a live claim, so the check is on
    the shape and not on the words.

    **THE LICENCE IS NARROWER SINCE 2026-08-25, AND THE NARROWING IS THE
    OTHER HALF OF THIS ONE.** What may be quoted is the canonical FRAMING —
    `is reached by **six** ENTRIES` — which is the shape this check reads
    and refuses to see asserted. What may NOT be quoted is the full unit
    phrase `N ENTRIES of this log`, because
    `test_the_reached_release_count_is_the_number_of_entries_that_declare_it`
    blanks quotations before comparing and therefore cannot read one there;
    `test_no_quotation_carries_the_ENTRIES_figure_the_comparison_cannot_read`
    holds that half. Between the two, the figure has exactly one place it
    can stand: unquoted, where both are looking.
    """
    prose = release_prose()
    live = re.compile(
        r"(?<![\"*])is\s+reached\s+by\s+\*\*(?P<n>[A-Za-z]+|\d+)\*\*\s+ENTRIES",
        re.S,
    )
    entries = [line for line, text in log_bullets() if REACHES_V010 in text]
    wrong = [
        m.group("n") for m in live.finditer(prose)
        if _numeral(m.group("n")) != len(entries)
    ]
    assert not wrong, (
        f"the release record asserts that {wrong} entries of the log reach "
        f"`v0.1.0` and {len(entries)} do"
    )


@pytest.mark.parametrize("field", REACH_FIELDS)
def test_each_permitted_field_is_actually_used(field):
    """No permitted-but-unused option.

    A closed set with a member nothing ever uses is an option that has
    never been exercised, and this campaign has closed that shape often
    enough to check for it. All three are in use. **The partition is NOT
    typed here**, for the reason the comment on `_PREAMBLE_MAX_LINES`
    gives: measured over the same bullets at each commit that touched this
    file, it has been 39/8/7 (`68b219d`, `1f55eef`, `1242da4`), 33/12/9
    (`c198a8d`, `6ecb5cd`) and 33/10/11 (`161ead8`) — three moves in two
    days before this branch made a fourth, and every one of them a
    correction of a FALSE FIELD rather than a new entry. The live values are in the message below and in `SOUNDNESS.md`'s
    own reached-release paragraph, which is derived from these bullets and
    checked above. (Named
    rather than called "the release": `0.2.0` is a release too now, and no
    bullet reaches it.)
    """
    bullets = log_bullets()
    used = [line for line, text in bullets if field in text]
    assert used, (
        f"no `## Log` bullet carries {field!r}. A phrase in the closed set "
        f"that nothing uses is an untested branch of the rule, not a "
        f"choice the rule offers. The partition over "
        f"{len(bullets)} bullets is "
        + "/".join(
            str(sum(1 for _, text in bullets if f in text))
            for f in REACH_FIELDS
        )
        + "."
    )


# --- THE CITATION RULE ------------------------------------------------------
#
# A `Versions: 0.2.0 development builds only.` field on a post-tag bullet is a
# claim about a released artefact, and three passes have now moved one of these
# fields because the claim was FALSE. Nothing in this tree can decide the
# claim; one SUB-CLASS of the justifications offered for it is decidable in
# milliseconds, and that sub-class is the one that failed.

#: The released tag every `Versions:` field in this file is about. Named once.
_TAG = "v0.1.0"

#: THE REASON THE THIRD SKIP CARRIES, AND IT IS A CONSTANT for exactly the
#: reason `tests/test_soundness_routing.py`'s `_GIT_LESS_SKIP` is one:
#: `tests/test_skip_inventory.py` excuses a skip only by an EXACT string typed
#: on its own disclosure surface, so a reason carrying git's stderr can never
#: be disclosed there and its file exits 1 on the completeness half instead.
#: Derived from :data:`_TAG` rather than retyped, so a rename cannot leave the
#: two silently about different tags — it makes the reason stop matching the
#: rule over there, and an undisclosed skip is the loud direction.
_TAG_OUT_OF_REACH_SKIP = (
    f"git cannot resolve `{_TAG}`, so no claim here about the tag's tree "
    f"can be decided"
)

#: A citation of a PATH IN THE TAG'S TREE — `v0.1.0:src/stelling/foo.py`.
#: Matched WITHOUT its backticks, so the form `git show v0.1.0:<path>` inside
#: one code span is read too; the log uses both. A trailing `.` or `,` is not
#: part of a path here because the character class is anchored by `+` and the
#: page always closes the span, but `_is_path_char` keeps `/` and `.` so a
#: dotted filename survives.
_TAG_PATH_RE = re.compile(r"v0\.1\.0:(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*)")

#: THE ABSENCE VOCABULARY — FOUR LITERAL PHRASES, AND THAT IS ALL IT IS.
#: A tag-path token is read as a claim that the path IS in the tag's tree
#: unless it is immediately followed by one of these four, which are read as
#: claiming it is not. Default-present is the safe default: citing a path
#: that is not in the tag's tree, without saying it is not, is a citation
#: that names nothing.
#:
#: **THIS COMMENT USED TO SAY "AND THERE IS EXACTLY ONE SHAPE OF IT", WHICH
#: IS FALSE IN BOTH DIRECTIONS AND WAS MEASURED FALSE ON 2026-08-25.** A
#: FALSE absence claim about a path that IS in the tag's tree passes in ten
#: spellings this list does not hold — `was not yet written`, `only arrived
#: in 0.2.0`, `post-dates the tag`, `is nowhere in the tag's tree`, `is not
#: in the tag's tree`, `is not there`, `never shipped in the tag`, `arrived
#: later than the tag`, the negation written BEFORE the token, and
#: `, which does not exist`, which is a listed phrase with a comma in front
#: of it. And a TRUE absence claim in any of those spellings REDS, because
#: the token then reads as claiming presence: one of them is this project's
#: own sentence, the one closing *"…neither of which is in the tag's
#: tree"* in `SOUNDNESS.md`'s 2026-08-18 query-identity bullet — named by
#: its content, because a line number into a page that grows every pass is
#: the kind of figure this file exists to stop writing down. The list is
#: deliberately NOT
#: extended — English negation is not a finite set of phrases, an eleventh
#: spelling is one sentence away from any list, and every addition widens
#: the over-fire. What this leg is, said plainly: a check that a document
#: which chose one of these four chose the one that agrees with the tag.
#:
#: Run over a WHITESPACE-NORMALISED copy, because the page wraps at 72
#: columns and any of these phrases can wrap — driven both ways, and that
#: half IS correct.
_TAG_PATH_ABSENT_RE = re.compile(
    r"`v0\.1\.0:(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*)` "
    r"(?:does not exist|did not exist|is absent|does not appear)"
)

#: A COMMIT citation. Seven or more hex digits in a code span. A token that
#: does not resolve to a commit in this repository is simply not a commit
#: citation — it is not an error here, because an ordinary English word can
#: be spelled out of `[0-9a-f]` and this rule has no business red-flagging
#: one. What it costs: a MISTYPED sha does not count as a citation, so the
#: bullet fails the coverage leg below rather than passing on a sha that is
#: not there.
_SHA_RE = re.compile(r"`(?P<sha>[0-9a-f]{7,40})`")

#: The date on a `## Log` bullet's headline.
_BULLET_DATE_RE = re.compile(r"^- \*\*(?P<date>\d{4}-\d{2}-\d{2})")

#: THE SAME INFERENCE, OUTSIDE THE `## Log` AND OUTSIDE ITS FIELDS.
#: `CHANGELOG.md`'s Mode 2 and Mode 3 sections each carry a `*Versions.*`
#: paragraph making this exact move — *"Mode N is 0.2.0 development work
#: throughout, so `v0.1.0` predates all of it"* — for a whole SECTION rather
#: than for one entry. They are in this rule's population because the class
#: it is about is *a justification that is a claim about the tag's tree*, and
#: these two are that, in the most literal form of it: `predates all of it`
#: is the sentence `git cat-file -e` decides. A pass that closed the class
#: inside the `## Log` and left these two would have closed it by where the
#: sentences live rather than by what they claim.
#:
#: **AND THE SELECTOR IS TWO WORDS, WHICH CLOSES THE CLASS BY WHICH WORDS
#: THE SENTENCE HAPPENS TO USE — the same failure the paragraph above names
#: when it argues against closing it by where the sentences live.** Measured
#: on 2026-08-25: `CHANGELOG.md`'s `### Verification pipeline` section
#: makes exactly this move in its own words — *"The count
#: check is in the released tag: `v0.1.0:src/stelling/obligation.py`'s
#: `slice_unknown_obligations` carries it verbatim"* — and is not in this
#: population, because it does not spell it `predates`. It is TRUE, and it
#: is unread. Stated rather than closed: a wider selector is a wider word
#: list, which is the shape :data:`_TAG_PATH_ABSENT_RE`'s comment has just
#: been measured against.
_PREDATES_RE = re.compile(r"`v0\.1\.0` predates")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )


def _the_environment_explains_an_unresolvable_citation() -> str | None:
    """Why THIS TREE could legitimately not hold a cited commit, or ``None``.

    **THE GUARD THAT STAYED HARD-RED AFTER THE TAG GUARD WAS FIXED, AND IT
    IS THE SAME DEFECT ONE LEG OVER.** The tag guard above disclosed the
    case where `v0.1.0` itself is out of reach. It does not cover the case
    where the tag RESOLVES and the CITED COMMITS do not, which is exactly a
    shallow fetch that also fetched the tag refs: `git clone --depth 1
    --branch v0.1.0`, or `actions/checkout` with `fetch-tags: true` and no
    `fetch-depth`. In that tree `postdates_the_tag` answers `None` for every
    cited sha, every commit-cited paragraph reads as citing nothing, and the
    coverage assertion below fired saying *"N claim(s) ... cite nothing this
    check can decide"* -- WHICH IS FALSE THERE. Those paragraphs cite a
    commit; this checkout cannot resolve it. Measured on 2026-08-25 in a
    sandbox built that way over this tree: `1 failed, 12 passed`, naming
    seven `## Log` bullets, one of which quotes `4d793cf` in its own
    headline. `.github/workflows/release.yml` cannot produce that shape --
    both its `actions/checkout@v4` steps take `fetch-depth: 0`, which
    fetches the whole history and every tag -- but a `--depth` anywhere else
    still can, and a live assertion message that is false in an ordinary
    `git clone --depth 1` is the thing this file exists to stop shipping.

    **AND IT MAY NOT SIMPLY BECOME A WARNING**, for the reason
    `tests/test_proposed_page_headers.py` gives at the same fork: in a FULL
    checkout of this project, a cited sha that does not resolve is a
    paragraph citing a commit that never existed -- a typo, a sha from a
    fork, a commit rewritten away -- and that is a defect this rule should
    keep reporting. So the environment is asked first, and only where
    something about THIS TREE explains the miss is the citation reported as
    undecidable rather than absent. The triage is
    `tests/test_soundness_routing.py`'s, in the same order and for the same
    reasons, with `_ANCHOR` READ from the module that owns it rather than a
    second copy of a sha:

    * `git rev-parse --is-shallow-repository` -- answered first, so this leg
      never depends on an import;
    * `git cat-file -e <root commit>^{commit}` -- a non-shallow repository
      holding any commit of this history holds the root too, so its absence
      says this is a vendored copy somebody ran `git init` in, a fork with
      rewritten history, or a `GIT_DIR` pointing somewhere else.

    An import failure or a git that stopped answering returns an
    explanation, which is the SAFE direction: it turns a hard red into a
    disclosed warning rather than inventing a red in a tree nobody can fix.
    """
    shallow = _git("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:  # pragma: no cover - env-dependent
        return (
            f"git cannot say whether this checkout is shallow: "
            f"{shallow.stderr.strip()[:120]}"
        )
    if shallow.stdout.strip() == "true":  # pragma: no cover - env-dependent
        return (
            "this checkout is SHALLOW -- `git rev-parse "
            "--is-shallow-repository` says true -- so its history is "
            "truncated and a commit older than the horizon is simply not "
            "here, however correctly the paragraph cites it"
        )
    try:
        from test_soundness_routing import _ANCHOR
    except Exception:  # noqa: BLE001 - a triage helper may not raise
        return (  # pragma: no cover - env-dependent
            "this tree cannot say whether it is this project's checkout at "
            "all, because tests/test_soundness_routing.py, which owns the "
            "root commit, did not import"
        )
    probe = _git("cat-file", "-e", f"{_ANCHOR}^{{commit}}")
    if probe.returncode != 0:  # pragma: no cover - env-dependent
        return (
            f"the git repository rooted here is not this project's: it does "
            f"not have {_ANCHOR[:12]}, this project's root commit, which a "
            f"non-shallow repository holding any commit of this history "
            f"holds too"
        )
    return None


def _tag_tree_population() -> list[tuple[str, int, str]]:
    """`(where, line, text)` for everything that owes this rule evidence.

    Two shapes, one class:

    * every `## Log` bullet DATED AFTER the tag and carrying
      :data:`DEV_ONLY` — the field that says the release does not carry the
      defect. The tag's own date is read out of git rather than typed, and
      the tag DAY is not "after the tag": an entry written on 2026-08-12
      can be either side of a commit made that afternoon, and demanding
      evidence of it would be demanding it of a bullet whose field may be
      about the pre-tag half of its own day;
    * every `CHANGELOG.md` paragraph carrying :data:`_PREDATES_RE`.
    """
    tagged = _git("log", "-1", "--format=%cs", f"{_TAG}^{{commit}}")
    assert tagged.returncode == 0, (
        f"`git log -1 --format=%cs {_TAG}` failed, so the date that decides "
        f"which bullets are post-tag cannot be read: {tagged.stderr.strip()}"
    )
    tag_day = tagged.stdout.strip()

    out: list[tuple[str, int, str]] = []
    for line, text in log_bullets():
        if DEV_ONLY not in text:
            continue
        m = _BULLET_DATE_RE.match(text)
        assert m, (
            f"`## Log` bullet at line {line} carries {DEV_ONLY!r} and opens "
            f"with no date, so whether it is post-tag cannot be decided: "
            f"{text.splitlines()[0][:70]!r}"
        )
        if m.group("date") > tag_day:
            out.append((f"SOUNDNESS.md:{line}", line, text))

    changelog = CHANGELOG.read_text(encoding="utf-8")
    para, start = [], 0
    for n, raw in enumerate(changelog.split("\n"), 1):
        if raw.strip():
            if not para:
                start = n
            para.append(raw)
        elif para:
            body = "\n".join(para)
            if _PREDATES_RE.search(body):
                out.append((f"CHANGELOG.md:{start}", start, body))
            para = []
    if para and _PREDATES_RE.search("\n".join(para)):
        out.append((f"CHANGELOG.md:{start}", start, "\n".join(para)))
    return out


def test_a_claim_about_the_tags_TREE_is_decided_against_the_tag():
    """ELEVEN PARAGRAPHS ARE FORMATTED AS CITATIONS, AND THE FORMAT IS
    WHAT THIS DECIDES.

    **WHAT IT ESTABLISHES, MEASURED RATHER THAN CONCEDED.** Every member
    of :func:`_tag_tree_population` contains **any** `v0.1.0:<path>` token
    **or any** backticked sha that is not an ancestor of the tag —
    anywhere in the paragraph, for any reason, **including a sha the
    paragraph itself names as the commit that INTRODUCED the defect** —
    and every path token it does contain agrees with `git cat-file -e`
    under a polarity read off **four literal phrases**. That is the whole
    of what comes out of this check. **A citation-FORMAT rule, not an
    evidence rule.**

    **AND IT IS GREEN ON ALL NINE FALSE `Versions:` FIELDS THIS PROJECT
    HAS EVER FOUND.** Re-derived on 2026-08-25 by running this rule, as it
    stands, over the three document states in which a false field actually
    shipped — a linked worktree at each commit with today's `tests/`
    copied into it, so the rule is today's and the documents are theirs:

    * **`c198a8d^`** — SIX false fields, all *0.1.0 pre-release builds
      only* on post-tag bullets. **GREEN on all six**, and outside the
      population BY CONSTRUCTION: :func:`_tag_tree_population` keys on the
      OTHER field. Measured there: 6 post-tag bullets carry that phrase, 0
      of them are members.
    * **`161ead8^`** — TWO false fields, S12&Prime; and M17, both *0.2.0
      development builds only*. **GREEN on both**, and both were IN the
      population and COVERED: each mentions `v0.1.0` **zero** times, and
      what certifies them is worth writing out. S12&Prime; is covered by
      `4d793cf`, which its own body names two lines under the field as the
      commit that added a false docstring SENTENCE, in the same breath as
      saying the crash itself is *"pre-existing"* — that is, by a sha it
      cites as introducing the defect while it says the released tag has
      it. M17 is covered by six shas that occur only in a TEST-COUNT
      reconciliation table (*"+34 for `4d793cf`"*, *"+48 for `96ab47a`"*),
      none of them offered as evidence about the tag at all.
    * **`b0df85e^`** — ONE false field, the 2026-08-18 query-identity
      entry. **GREEN**, in the population and covered: it cites
      `v0.1.0:src/stelling/obligation.py`, correctly, a few paragraphs
      away from the sentence that was the false one — *"`v0.1.0` predates
      the whole file"*, about `src/stelling/reproduce.py`, which names NO
      path and is therefore the half `git cat-file -e` cannot reach.

    **AND WHERE IT DOES FIRE ON THE HISTORICAL RECORD IT FIRES ON FOUR
    TRUE PARAGRAPHS.** Across those three states the uncovered set is the
    2026-08-14 rational-`pow` entry, the 2026-08-15 B7 M10 VERIFIED-bar
    entry, and `CHANGELOG.md`'s Mode 2 and Mode 3 `*Versions.*`
    paragraphs — four, all four true, all four since checked by hand and
    at an extracted tag. **A maintainer following this rule at any of
    those three states would have added citations to four correct
    paragraphs and shipped the false one.**

    **AND IT ADDS NO CONSTRAINT TO SEVEN OF ITS OWN ELEVEN MEMBERS.**
    Seven of today's eleven cite no path at all and are covered by a sha,
    and all seven carried those shas at `161ead8`, two commits before this
    rule existed. The rule found them already compliant and asks them for
    nothing they did not have.

    **SO THE SENTENCE THAT USED TO STAND HERE IS GONE.** It read *"What it
    removes from the board is one specific way of being wrong, and that
    way is the one that actually happened … three consecutive passes
    shipped one that a millisecond would have caught."* The SUFFICIENCY
    disclaimer stood in six places around it, honestly and first — this
    docstring's opening, the `uncovered` assertion message, the module
    docstring, `SOUNDNESS.md`'s two paragraphs on this rule and
    `CHANGELOG.md`'s Mode 2 sentence — and not one of them disclaimed
    that, which is a NECESSITY claim on the other axis standing
    unqualified beside them, and it is the sentence a reader takes away.
    It is false nine times out of nine. What is true is the paragraph at
    the top of this docstring, and it is smaller.

    **THE RULE, unchanged in what it does.** Every member of
    :func:`_tag_tree_population` must contain at least one of:

    * a **`v0.1.0:<path>`** token, decided with `git cat-file -e` against
      the claim's POLARITY. The token claims the path IS in the tag's tree
      unless it is immediately followed by one of the four phrases in
      :data:`_TAG_PATH_ABSENT_RE`; **or**
    * a **commit**, decided with `git merge-base --is-ancestor <sha>
      v0.1.0` to be NOT an ancestor of the tag.

    Both legs run over the WHOLE population and not only over the members
    that need them, so a paragraph cannot buy the coverage leg with a sha
    and make a false tag-tree claim beside it.

    **WHAT THE COMMIT LEG CANNOT TELL.** It cannot tell WHICH cited sha is
    *"the commit that introduced the defect"* — prose says that, and prose
    is what this file has given up reading — so it asks only for one cited
    commit that postdates the tag. That is why the S12&Prime; measurement
    above is not a surprise: a bullet whose subject is a PRE-EXISTING
    crash cites the post-tag commit that added a false sentence about it,
    and the leg is satisfied by the citation without ever meeting the
    claim.

    **THE POLARITY LEG IS A FOUR-PHRASE VOCABULARY AND NOT A NEGATION
    DETECTOR. IT MISSES IN BOTH DIRECTIONS.** Driven on 2026-08-25 on the
    real `SOUNDNESS.md`, one spelling at a time, the file restored
    byte-identically (sha256) after each and `1 passed` on the restored
    file. A FALSE absence claim about `v0.1.0:src/stelling/obligation.py`,
    which IS in the tag's tree, planted in a population bullet, PASSES in
    TEN spellings: *was not yet written*, *only arrived in 0.2.0*,
    *post-dates the tag*, *is nowhere in the tag's tree*, *is not in the
    tag's tree*, *is not there*, *never shipped in the tag*, *arrived
    later than the tag*, the negation written BEFORE the token (*"There is
    no `v0.1.0:<path>`"*), and **`, which does not exist`** — a listed
    phrase with a comma in front of it. The control, the same sentence
    with `does not exist` immediately after the token, is `1 failed`.

    It OVER-FIRES on the same axis, on honest text. Three of those
    spellings written about `v0.1.0:src/stelling/_tripwire/eager.py`,
    which genuinely is NOT in the tag's tree, each give `1 failed` reading
    *"claims PRESENT, and it is NOT in the tag's tree"*. **One of the
    three is this project's own sentence** — the one in `SOUNDNESS.md`'s
    2026-08-18 query-identity bullet that cites both tripwire modules and
    closes *"…neither of which is in the tag's tree"*, named by its
    CONTENT and not by a line number, because a line number into a page
    that grows every pass is the kind of figure this file exists to stop
    writing down. It is TRUE, it is refused by this leg the moment it is
    planted in a member, and it stands today only because its bullet
    carries the THIRD `Versions:` field, which the population does not
    look at.

    **THE LIST IS NOT EXTENDED, AND THAT IS THE FINDING RATHER THAN A
    SHORTFALL.** English negation is not a finite set of phrases: an
    eleventh spelling is one sentence away from any list, and every
    addition widens the over-fire by making more honest sentences read as
    a polarity nobody wrote. What this leg is, said plainly: a check that
    a document which chose one of four phrases chose the one that agrees
    with the tag.

    **THE LINE-BREAK CASE IS HANDLED CORRECTLY**, and is worth separating
    from the vocabulary. Both scans run over `" ".join(text.split())`, so
    a listed phrase that wraps at the page's 72 columns is still read:
    driven, `` `v0.1.0:src/stelling/obligation.py` does not `` / `` exist.
    `` split across two lines reds exactly as the unwrapped form does, and
    the same wrapped sentence about a genuinely absent path passes.

    **THE POPULATION IS WIDER THAN THE `## Log` AND IT IS NOT COMPLETE.
    Five gaps, measured on 2026-08-25 and stated rather than closed:**

    * **the OTHER false-claiming field is outside by construction.**
      *0.1.0 pre-release builds only* produced SIX of the nine, and the
      DATE-versus-FIELD comparison this module's own docstring describes
      is implemented NOWHERE. Driven at `e94f4ea`, with the plant and
      without it in one environment: re-scoping a post-tag bullet back to
      that phrase leaves the zero-dep lane at **2321 passed, 184 skipped**
      BOTH WAYS, byte-identical summary lines.
    * **the `REACHES_V010` bullets are unread by the polarity leg.**
      Twelve carry that field; five of them make path citations,
      FOURTEEN tokens between them on this tree, and none is
      polarity-checked. Driven: a false
      absence claim planted in one of them is **green**. Extending the
      leg to them is the cheapest real gain available and it is not taken
      here, because the cost was DRIVEN rather than argued — one line,
      `DEV_ONLY not in text` widened to admit the third field, gives
      `1 failed` naming `SOUNDNESS.md:12614` twice, once for
      `v0.1.0:src/stelling/_tripwire/eager.py` and once for
      `perimeter.py`, both *"claims PRESENT, and it is NOT in the tag's
      tree"*, on the honest sentence named above. Adopting it means
      rewriting that sentence into the four-phrase vocabulary and holding
      twelve more bullets' prose to it, for a gain that is a FORMAT gain
      (14 more of the 21 tokens read) and not an evidence gain.
    * **`CHANGELOG.md`'s `### Verification pipeline` makes this exact move
      and is unreached.**
      *"The count check is in the released tag:
      `v0.1.0:src/stelling/obligation.py`'s `slice_unknown_obligations`
      carries it verbatim"* is a live tag's-tree justification, and
      :data:`_PREDATES_RE` selects paragraphs by TWO WORDS. The widening
      that reached `CHANGELOG.md` at all closed the class by which words
      the sentence happens to use, which is the same failure this file
      names when it argues against closing it by where the sentences live.
    * **six of twenty-one.** On this tree the two files carry 21
      `v0.1.0:<path>` tokens by :data:`_TAG_PATH_RE` over a
      whitespace-normalised copy of each; 6 fall inside the polarity leg
      and 15 do not. It was 6 of 18 at `e94f4ea`: THREE of the fifteen
      were added by `3617e35`'s own prose, which is exactly how a ratio
      like this moves and why it is dated rather than left standing.
      `3617e35` is named here rather than *"this commit"*, which is what
      this bullet said: an indexical in a figure that outlives its own
      commit re-points at whichever commit reads it next. Re-derived on
      2026-08-25 over the amendment `SOUNDNESS.md`'s 2026-08-18 entry now
      carries, which adds prose to that entry and no token to either
      file: **unmoved, 6 of 21**.
    * **the floor is `>= 3`, not eleven.** The population is measured, and
      the assertion under it only refuses a COLLAPSE. Driven: re-dating
      eight of the nine bullets to the tag day leaves 3 members and the
      check is **green**.

    **WHEN GIT IS NOT THERE, AND THE THIRD STATE THAT USED TO BLOCK THE
    RELEASE.** Three conditions, three DISCLOSED skips, each on
    byte-for-byte the predicate `tests/test_skip_inventory.py` declares it
    legitimate for:

    * `git` is not on `PATH` — `needs git`;
    * `.git` is not here (an unpacked sdist, say) — `not a git checkout
      (an unpacked sdist, say)`;
    * git is here, this IS a checkout, and it cannot resolve the tag —
      :data:`_TAG_OUT_OF_REACH_SKIP`, added on 2026-08-25 with its `RULES`
      entry.

    All three were driven on the tree this commit produces, each giving
    `1 skipped` on this node with the reason named above and, for the
    third, a warning. **AND ONE OF THE THREE IS STILL NOT A GREEN SUITE,
    which the pass before last's "driven with git off PATH" line implied
    and did not measure.** RE-DRIVEN on this tree on 2026-08-25, zero-dep
    lane, whole suite, `STELLING_SKIP_INVENTORY_VERDICT` set:

    * git off `PATH` — a symlink farm of every `bin` dir minus `git*`, so
      `uv` and the rest are still there — is `3 failed, 2306 passed, 209
      skipped`, `verdict=made`. The three are
      `tests/test_reuse_pins.py` once and
      `tests/test_sdist_reference_hygiene.py` twice: they guard on `.git`
      and not on `shutil.which("git")`, so git's absence reaches them as an
      uncaught `FileNotFoundError`. Neither file is this one and neither is
      fixed here. IT WAS FOUR, and the fourth was the skip inventory
      reading their skips; that one is gone — measured identical, `3
      failed, 2306 passed, 209 skipped`, on this tree WITHOUT the changes
      in this commit, so it is not this commit that fixed it.
    * `.git` removed, git still on `PATH` — an unpacked sdist — is
      `2328 passed, 190 skipped`, exit 0, `verdict=made`. The `1 failed`
      recorded here for that lane was
      `tests/test_proposed_page_headers.py` skipping as `needs git` with
      git right there; that reason now names the condition that holds.
    * an unpacked copy with `.git` removed, `git init` run and everything
      committed — so `git ls-files` answers and
      `tests/test_doc_examples.py` collects — is `2332 passed, 186
      skipped`, exit 0, `verdict=made`. (`git init` with NOTHING committed
      is a different tree: `git ls-files` returns nothing there and that
      file still errors at collection.)

    They are listed because "it skips" is not "the lane is green", and the
    difference is the whole subject of the file it is written in.

    **THE THIRD WAS A HARD RED AND IT FIRED INSIDE REFUSAL POINT #1 FOR
    PyPI.** `.github/workflows/release.yml`'s job *"the suite, on the
    tagged tree"* USED TO check out with `actions/checkout@v4` and
    `persist-credentials: false` alone — no `fetch-depth`, no
    `fetch-tags` — which meant depth 1 on the triggering tag ref. On any
    release after `v0.1.0` that tree had `.git`, had git, and did not
    have `v0.1.0`: the old assertion fired, the suite went red, and the
    publish was refused with a message reading as an integrity failure of
    the record. The precedent ran against it three ways, all of them in
    this tree: `tests/test_soundness_routing.py` SKIPS eight tests on the
    same condition (git present, `.git` present, the object unreachable)
    with a declared reason and a `UserWarning` naming git's words;
    `tests/test_skip_inventory.py` already carried a rule whose `when`
    names *"a shallow CI clone that does not reach it"*, so the pattern
    was written and only an entry was missing; and
    `tests/test_sdist_contents.py` says of its own environment-absence
    skip that making it a hard failure *"would be flaky in the environment
    where it matters least"*. `fetch-depth: 0` in that workflow was named
    here as a recommendation; it has since LANDED, and the paragraph below
    says what that changes and what it does not.

    **AND THAT WORKFLOW NO LONGER PRODUCES THE CONDITION — DERIVED FROM
    THE FILE, NOT FROM A SENTENCE ABOUT IT.** Parsed with PyYAML,
    `release.yml` has TWO `actions/checkout@v4` steps —
    `jobs.test.steps[0]` and `jobs.build.steps[0]`; `publish` checks
    nothing out, and `grep -c "uses: actions/"` reading EIGHT there counts
    two artefact steps and four comment mentions alongside them. Both
    checkouts carry `persist-credentials: false`, `ref: ${{ github.sha }}`
    and `fetch-depth: 0` — the `ref:` input arrived later than this sentence,
    which read "exactly `persist-credentials: false` and `fetch-depth: 0`",
    and it is what stops the action re-fetching `+<github.sha>:refs/tags/<tag>`
    and overwriting a tag ref — and no `fetch-tags`, which `fetch-depth: 0` makes
    inert because the all-history refspec already contains
    `+refs/tags/*:refs/tags/*`. Built the way that config builds a tree —
    `git init`, ONE fetch of `+refs/heads/*:refs/remotes/origin/*` and
    `+refs/tags/*:refs/tags/*` with no depth limit, `git checkout --force`
    of the tag — the sandbox comes back NOT SHALLOW and `v0.1.0`
    RESOLVES. (1079 commits at `e6b35dc`, 1081 at the commit that wrote
    this sentence. The count is a property of the CHECKOUT the sandbox is
    built from, not of the workflow, so it moves on every commit and no
    live value for it is pinned here; the two answers that are about the
    workflow are the two above.) So this check decides there rather than
    skipping.

    **AND THE SANDBOX THAT SENTENCE DESCRIBES IS THE WRONG SHAPE, WHICH IS
    SAID HERE BECAUSE THE CONCLUSION SURVIVES AND THE METHOD DOES NOT.**
    *"Built the way that config builds a tree -- `git init`, ONE fetch ..."*
    is not how `actions/checkout@v4` builds one for a `release: published`
    event: it fetches all refs and THEN re-fetches
    `+<github.sha>:refs/tags/<tag>`, force-writing the release commit over
    that one tag ref. A correct measurement of the wrong process is this
    project's L28, and it is what let `release.yml`'s changelog gate refuse a
    correctly annotated `v0.2.1` on 2026-08-28.
    :func:`tests.test_release_gates._checkout_the_way_actions_checkout_does`
    is the fixture that runs the action's own refspecs.

    **WHAT IS ASSERTED HERE IS UNAFFECTED, AND THAT IS A MEASUREMENT AND NOT A
    HOPE.** The second fetch rewrites the TRIGGERING tag's ref and no other.
    Driven 2026-08-28 against a mirror of this repository with the real
    two-fetch sequence for `v0.2.1`: `refs/tags/v0.1.0`,
    `refs/tags/v0.1.0-scaffolding` and `refs/tags/v0.2.0` all still read
    `tag`, only `refs/tags/v0.2.1` reads `commit`,
    `git rev-parse --is-shallow-repository` is `false`, and
    `v0.1.0^{commit}` resolves to `e67688e`. :data:`_TAG` here is `v0.1.0`,
    which is not the tag any release run triggers on while a later version is
    being cut -- so this check reads an intact ref. IT WOULD NOT IF `_TAG`
    EVER BECAME THE TRIGGERING TAG, and every reader of it in this module
    peels (`{_TAG}^{{commit}}`, `{_TAG}:<path>`, `merge-base --is-ancestor`),
    which returns the same answer through a tag object or through the commit
    itself -- driven side by side on a clobbered and an intact ref.

    Refusal point #1 is therefore no longer blocked by this check or by
    `tests/test_proposed_page_headers.py`, whose ancestry check decides
    too: `deadbee` planted in a **Status:** paragraph is `1 failed` in
    that sandbox, named as a page defect.

    **AND THE FOURTH STATE, WHICH `fetch-depth: 0` DID NOT REACH.** The
    three skips above all key on *the tag* being out of reach. A tree can
    resolve the tag and still not hold the COMMITS the paragraphs cite --
    `git clone --depth 1 --branch v0.1.0`, or `actions/checkout` with
    `fetch-tags: true` and no `fetch-depth`, which the workflow's own header
    measures and rejects. There the coverage assertion below fired with a
    message that was FALSE of the paragraphs it named: seven `## Log`
    bullets reported as citing *"nothing this check can decide"* while one
    of them quotes `4d793cf` in its own headline. Driven in that sandbox on
    this tree: `1 failed, 12 passed`. It is now separated at the source --
    :func:`postdates_the_tag` already answered `True`/`False`/`None` and the
    caller was reading all three as two -- so a paragraph whose only
    citations are commits THIS TREE CANNOT RESOLVE is dropped from the
    coverage assertion and disclosed in a `UserWarning` naming the
    paragraph, the shas and what goes unchecked, and the rest of the rule
    still runs. It is NOT a
    skip: the path leg and every other paragraph are still decided, and a
    check that can decide most of its population should not throw that away.
    And it is not unconditional either:
    :func:`_the_environment_explains_an_unresolvable_citation` is asked
    first, so in a complete checkout of this project an unresolvable
    citation is still a hard red -- a sha that never existed is a defect,
    not an environment. See that function.

    **THE SKIP STAYS, AND SO DOES THE REASON FOR IT.** `fetch-depth: 0`
    makes the condition unreachable FROM THIS WORKFLOW; it does not make
    it impossible. A shallow clone cut anywhere else, an unpacked sdist
    with no `.git`, and a `git init`'d copy with no tags all still produce
    it, and the three reasons above partition exactly those. What follows
    from the change is a reading rule: THIS SKIP FIRING INSIDE THAT
    WORKFLOW IS NOW A SIGNAL THAT SOMETHING CHANGED — the `fetch-depth: 0`
    came off one of the two checkouts — and not a fact about the
    environment CI happens to give.

    **AND THE PROMISE THAT SKIP REPLACED WAS EMPTY.** The old red said it
    carried *"git's own stderr"*, and the probe was
    `rev-parse --verify --quiet`, which SUPPRESSES stderr — so the message
    read ``git said: ''`` in every environment it could ever fire in. The
    `--quiet` is gone from this one probe (it stays on
    :func:`postdates_the_tag`, where a token that is not a commit is not
    an error), git's exit code and words go into the WARNING beside the
    skip, and an empty stderr is reported as an empty stderr rather than
    as a quotation.

    **DRIVEN RED FIVE WAYS**, on the real files, restored byte-identically
    (sha256) after each and the whole file back green:

    * BOTH shas removed from the 2026-08-15 B6 regression bullet, leaving
      it with no citation of either kind — names that bullet, uncovered;
    * `v0.1.0:src/stelling/_tripwire/eager.py` in `CHANGELOG.md`'s Mode 2
      paragraph reworded off the ABSENCE spelling, so a path that is NOT
      in the tag's tree reads as a claim that it is — names the file, the
      line, the path, and which way it disagrees;
    * a path that IS in the tag's tree
      (`v0.1.0:src/stelling/obligation.py`) written with `does not exist`
      after it;
    * the `predates` sentence's whole evidence clause deleted from Mode 3
      — names that paragraph, uncovered, which is the direction a rule
      scoped to the `## Log` would never have looked in;
    * both shas in that same B6 bullet replaced by `e67688e`, which is the
      tag's own commit and therefore an ancestor of it — names the bullet
      uncovered, which is the commit leg's non-vacuity.
    """
    if shutil.which("git") is None:
        # Byte-for-byte the predicate `tests/test_skip_inventory.py` declares
        # legitimate for this reason string.
        pytest.skip("needs git")
    if not (REPO / ".git").exists():
        # A worktree's `.git` is a FILE, hence `exists()` and not `is_dir()`.
        pytest.skip("not a git checkout (an unpacked sdist, say)")

    # NO `--quiet` HERE. It suppresses stderr, which is what made the old hard
    # red's promise of "git's own stderr" read `git said: ''` in every
    # environment it could ever have fired in. It stays on
    # `postdates_the_tag`, where a token that is not a commit is not an error.
    tag = _git("rev-parse", "--verify", f"{_TAG}^{{commit}}")
    if tag.returncode != 0:
        said = tag.stderr.strip() or "(nothing; git printed no diagnostic)"
        warnings.warn(
            f"`{_TAG}` is not reachable from this checkout, so BOTH legs of "
            f"the tag-tree citation rule are unverified here: no "
            f"`{_TAG}:<path>` token's polarity is decided against `git "
            f"cat-file -e`, and no cited commit is decided to postdate the "
            f"tag with `git merge-base --is-ancestor`. What survives "
            f"unchecked is a path cited as being in the tag's tree when it "
            f"is not, the same citation carrying a polarity phrase that "
            f"disagrees with the tag, and a member of "
            f"`_tag_tree_population()` citing nothing at all. `git rev-parse "
            f"--verify {_TAG}^{{commit}}` exited {tag.returncode} and said: "
            f"{said}. THREE ENVIRONMENTS PRODUCE THIS: a SHALLOW CLONE, "
            f"where `actions/checkout@v4` at its default `fetch-depth: 1` "
            f"or a plain `git clone --depth 1` fetches the tip alone; an "
            f"UNPACKED SDIST or an export, which has no `.git` to hold a "
            f"tag; and a `git init`'d COPY of a tree, which has the files "
            f"and none of the refs. `.github/workflows/release.yml` IS NOT "
            f"ONE OF THEM ANY MORE: both of its `actions/checkout@v4` "
            f"steps take `persist-credentials: false` and `fetch-depth: 0`, "
            f"which fetches every branch and every tag, so `{_TAG}` "
            f"resolves in that job. This warning arriving from inside that "
            f"workflow is a signal that the checkout changed, not that git "
            f"is narrow."
        )
        pytest.skip(_TAG_OUT_OF_REACH_SKIP)

    population = _tag_tree_population()
    assert len(population) >= 3, (
        f"this rule found {len(population)} subject(s) and there were ELEVEN "
        f"on 2026-08-25 — nine `## Log` bullets carrying {DEV_ONLY!r} "
        f"with a post-tag date, and `CHANGELOG.md`'s two `predates` "
        f"paragraphs. A population that has collapsed is a green that "
        f"measured nothing. If the class has genuinely emptied, retire this "
        f"check deliberately."
    )

    exists: dict[str, bool] = {}

    def in_the_tag(path: str) -> bool:
        if path not in exists:
            probe = _git("cat-file", "-e", f"{_TAG}:{path}")
            exists[path] = probe.returncode == 0
        return exists[path]

    ancestor: dict[str, bool | None] = {}

    def postdates_the_tag(sha: str) -> bool | None:
        """`True` / `False` / `None` when `sha` is no commit in this repo."""
        if sha not in ancestor:
            if _git(
                "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"
            ).returncode != 0:
                ancestor[sha] = None
            else:
                walk = _git("merge-base", "--is-ancestor", sha, _TAG)
                ancestor[sha] = walk.returncode != 0
        return ancestor[sha]

    uncovered, wrong_polarity, cite_unresolvable = [], [], []
    for where, _line, text in population:
        flat = " ".join(text.split())
        absent = [
            (m.start(), m.group("path"))
            for m in _TAG_PATH_ABSENT_RE.finditer(flat)
        ]
        # The k-th tag-path citation of the flattened copy is the k-th of the
        # raw one: whitespace normalisation cannot reorder or drop a match,
        # because no citation spans whitespace. So polarity is read off the
        # flattened copy and the citation is reported from it too.
        claimed_absent = {start for start, _ in absent}
        paths = []
        for m in _TAG_PATH_RE.finditer(flat):
            # `m.start()` is the `v`; the absence pattern starts one backtick
            # earlier.
            paths.append((m.group("path"), (m.start() - 1) in claimed_absent))
        for path, says_absent in paths:
            if in_the_tag(path) == says_absent:
                wrong_polarity.append((
                    where,
                    path,
                    "claims ABSENT, and it IS in the tag's tree"
                    if says_absent else
                    "claims PRESENT, and it is NOT in the tag's tree",
                ))
        cited = dict.fromkeys(m.group("sha") for m in _SHA_RE.finditer(text))
        # THREE ANSWERS, NOT TWO. `postdates_the_tag` returns True (a commit
        # this repository has, and it is not an ancestor of the tag), False
        # (a commit this repository has, and it IS an ancestor -- which
        # covers nothing, and is the commit leg's non-vacuity) and None (no
        # commit HERE, whatever it is elsewhere). Reading all three as
        # falsy collapsed the last two, so a paragraph that cites a commit
        # this tree cannot resolve was reported as citing nothing at all.
        verdicts = {sha: postdates_the_tag(sha) for sha in cited}
        shas = [sha for sha, after in verdicts.items() if after]
        if not paths and not shas:
            headline = text.splitlines()[0][:64]
            uncovered.append((where, headline))
            missing = [sha for sha, after in verdicts.items() if after is None]
            if missing:
                cite_unresolvable.append((where, headline, missing))

    # THE ENVIRONMENT IS ASKED ONCE, AFTER THE LOOP, AND ONLY WHERE IT COULD
    # MATTER. A paragraph whose only citations are commits THIS TREE CANNOT
    # RESOLVE is an uncovered claim in a complete checkout -- a sha that
    # never existed -- and an unverifiable one in a tree that explains the
    # miss. The assertion below keeps the first; the warning discloses the
    # second and the rest of the rule still runs on everything else.
    why = (
        _the_environment_explains_an_unresolvable_citation()
        if cite_unresolvable else None
    )
    if why is not None:  # pragma: no cover - env-dependent
        listed = "; ".join(
            f"{where} ({headline}) cites "
            + ", ".join(f"`{sha}`" for sha in missing)
            for where, headline, missing in cite_unresolvable
        )
        warnings.warn(
            f"{len(cite_unresolvable)} claim(s) that the released `{_TAG}` "
            f"does not carry a defect cite ONLY commits this checkout "
            f"cannot resolve, so the commit leg of the tag-tree citation "
            f"rule is UNVERIFIED for them: {listed}. What goes unchecked is "
            f"whether each cited sha is a commit at all and whether it "
            f"POSTDATES `{_TAG}` -- a paragraph could cite a commit that is "
            f"an ancestor of the tag, which covers nothing, and this run "
            f"would not say so. This is reported rather than failed because "
            f"{why}; in a complete checkout of this project an unresolvable "
            f"citation FAILS below as an uncovered claim, which is where a "
            f"sha that never existed is caught. "
            f"`.github/workflows/release.yml` does not build this shape: "
            f"both its `actions/checkout@v4` steps take `fetch-depth: 0`, "
            f"so this warning arriving from that workflow means the "
            f"checkout changed.",
            stacklevel=2,
        )
        excused = {where for where, _headline, _missing in cite_unresolvable}
        uncovered = [u for u in uncovered if u[0] not in excused]

    assert not wrong_polarity, (
        f"{len(wrong_polarity)} tag-tree citation(s) disagree with the "
        f"tag:\n  "
        + "\n  ".join(
            f"{w}: `{_TAG}:{p}` {why}" for w, p, why in wrong_polarity
        )
        + f"\nA `{_TAG}:<path>` token is read as claiming the path IS in "
        f"the tag's tree unless it is immediately followed by one of FOUR "
        f"LITERAL PHRASES — `does not exist`, `did not exist`, `is absent`, "
        f"`does not appear` — and `git cat-file -e` decides it. That is a "
        f"citation FORMAT and not a negation detector: ten other spellings "
        f"of absence were measured to pass a FALSE claim here, and three "
        f"true ones to red like this. If this fired on an honest sentence, "
        f"write it in one of the four phrases; the alternative is a longer "
        f"list, which is what this leg has already been measured not to be."
    )
    assert not uncovered, (
        f"{len(uncovered)} claim(s) that the released `{_TAG}` does not carry "
        f"a defect cite nothing this check can decide:\n  "
        + "\n  ".join(f"{w}: {h}" for w, h in uncovered)
        + f"\nEach must contain a `{_TAG}:<path>` token (decided with "
        f"`git cat-file -e` against the claim's polarity) or a backticked "
        f"commit that is NOT an ancestor of `{_TAG}` (decided with "
        f"`git merge-base --is-ancestor`) — anywhere in it, for any reason, "
        f"INCLUDING a sha the paragraph cites as the commit that introduced "
        f"the defect. THIS IS A CITATION-FORMAT RULE AND NOT AN EVIDENCE "
        f"RULE: satisfying it establishes that the paragraph is formatted "
        f"this way and nothing about whether the released `{_TAG}` carries "
        f"the defect, which only running an extracted `{_TAG}` decides. "
        f"Measured over the three document states in which a false "
        f"`Versions:` field shipped, this rule is green on all NINE of them. "
        f"IF A LISTED PARAGRAPH DOES CITE A BACKTICKED SHA, then this tree "
        f"cannot resolve it AND nothing about this tree explains that -- "
        f"`git rev-parse --is-shallow-repository` said false and this "
        f"repository holds this project's root commit -- so the citation "
        f"names a commit that is not in this history: a typo, a sha from a "
        f"fork, or one rewritten away. In a shallow clone the same "
        f"paragraph is a WARNING instead, not this."
    )
