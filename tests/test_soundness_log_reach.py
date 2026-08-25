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
post-tag date cannot have an event that was over pre-tag. THREE more moved
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

**AND ONE SUB-CASE OF "IS THE FIELD TRUE" IS DECIDABLE, WHICH IS WHERE
THIS FILE STOPS BEING ONLY ABOUT THE TELLING.** A field cannot be
checked; a JUSTIFICATION offered for one sometimes can, when the
justification is a claim about the TAG'S TREE — *"`v0.1.0` predates the
whole file"* is settled by `git cat-file -e` in a millisecond, and it is
the justification that cost passes 3, 4 and 5.
`test_a_claim_about_the_tags_TREE_is_decided_against_the_tag` decides it,
over the nine post-tag bullets carrying *0.2.0 development builds only*
and the two `CHANGELOG.md` paragraphs that make the same move for a whole
section. **It checks a citation's shape and polarity and never the
defect** — necessary, not sufficient — and it says so before it says
anything else.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

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
    it has held.

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

#: A citation of a PATH IN THE TAG'S TREE — `v0.1.0:src/stelling/foo.py`.
#: Matched WITHOUT its backticks, so the form `git show v0.1.0:<path>` inside
#: one code span is read too; the log uses both. A trailing `.` or `,` is not
#: part of a path here because the character class is anchored by `+` and the
#: page always closes the span, but `_is_path_char` keeps `/` and `.` so a
#: dotted filename survives.
_TAG_PATH_RE = re.compile(r"v0\.1\.0:(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*)")

#: THE ABSENCE SPELLING, AND THERE IS EXACTLY ONE SHAPE OF IT. A citation of
#: a tag path is a claim that the path IS in the tag's tree unless it is
#: immediately followed by one of these four phrases, which claim it is not.
#: Default-present is the safe default: citing a path that is not in the
#: tag's tree, without saying it is not, is a citation that names nothing —
#: which is the failure this rule exists for and not a spelling to forgive.
#: Run over a WHITESPACE-NORMALISED copy, because the page wraps at 72
#: columns and any of these phrases can wrap.
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
_PREDATES_RE = re.compile(r"`v0\.1\.0` predates")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )


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
    """EVERY post-tag *0.2.0 development builds only* claim cites something
    `git` can decide, and every tag-tree path it cites is decided.

    **WHAT THIS DOES NOT ESTABLISH, FIRST, BECAUSE IT IS THE HALF A READER
    WILL OTHERWISE ASSUME.** This check does not run `v0.1.0`. It does not
    decide whether the released artefact carries the defect, and it CANNOT:
    settling one of these costs an extraction of the tag, an interpreter,
    and in one case two interpreters carrying two jaxes. **It reads a
    CITATION'S SHAPE AND POLARITY, not the defect.** A bullet can satisfy
    every leg here and still carry a false field — necessary, not
    sufficient. What it removes from the board is one specific way of being
    wrong, and that way is the one that actually happened: a justification
    of the form *"`v0.1.0` predates the whole file"* is a claim about the
    TAG'S TREE, `git cat-file -e` refutes it in a millisecond, and three
    consecutive passes shipped one that a millisecond would have caught.

    **WHY IT IS HERE AT ALL, GIVEN THAT THE LAST PASS DECLINED IT.** That
    pass argued *"with the false claim removed there is no live subject,
    and this campaign withdraws permitted-but-unused rules"*. Both halves
    are wrong. The campaign's norm is about a DEAD OPTION A CLOSED SET
    OFFERS — a `Versions:` phrase no entry can ever use — which is a
    different thing from a guard that currently finds nothing; every
    regression test in this tree is the latter, and withdrawing them all
    is not a policy anyone holds. And the premise is false one step out:
    the class is not *"the one sentence that was wrong"*, it is *"a
    justification that is a claim about the tag's tree"*, and that class
    has NINE live subjects in the `## Log` plus TWO in `CHANGELOG.md`.
    They were checked by hand and found true. They existed.

    **THE RULE.** Every member of :func:`_tag_tree_population` must cite at
    least one thing this check can decide:

    * a **`v0.1.0:<path>`**, decided with `git cat-file -e` against the
      claim's POLARITY. The citation claims the path IS in the tag's tree
      unless it is immediately followed by one of the four phrases in
      :data:`_TAG_PATH_ABSENT_RE`, which claim it is not; **or**
    * a **commit**, decided with `git merge-base --is-ancestor <sha>
      v0.1.0` to be NOT an ancestor of the tag.

    Both legs run over the WHOLE population and not only over the members
    that need them: the polarity leg checks every tag-path citation any
    member makes, including the ones whose coverage came from a sha, so a
    bullet cannot buy the coverage leg with a sha and then make a false
    tag-tree claim beside it.

    **WHAT THE COMMIT LEG CANNOT TELL, SAID BECAUSE IT BOUNDS THE LEG.**
    It cannot tell WHICH cited sha is *"the commit that introduced the
    defect"* — prose says that, and prose is what this file has given up
    reading. So it asks for at least one cited commit that postdates the
    tag, which is what a genuine 0.2.0-development defect must have and
    what a bullet whose whole documentary basis predates the tag cannot
    produce. A bullet that cites a pre-tag commit for context alongside is
    not failed for it, because citing one is not a claim about anything.

    **POPULATION, MEASURED HERE AND NOT TYPED**: nine `## Log` bullets on
    2026-08-25 (all of 2026-08-14 and 2026-08-15; the earliest is two days
    after the tag) and two `CHANGELOG.md` paragraphs, the Mode 2 and Mode 3
    `*Versions.*` sentences. Seven of the nine already cited a sha. The two
    that did not are the two whose evidence was driven at the tag for this
    round:

    * the 2026-08-14 rational-`pow` entry, whose *"The row does not exist
      in `v0.1.0`"* was a tag-tree claim naming no path at all. Driven at
      an extracted `v0.1.0` under jax 0.11.0, x64, z3 + cvc5 wheels:
      `x ** 0.1 <= 1e30` over `x` declared `((), "float64", (1.0, 1e300))`
      returns UNKNOWN quoting *"escalation declined — primitive 'pow' is
      outside the supported emission set"*, and `"pow"` appears zero times
      in `v0.1.0:src/stelling/obligation.py`. It now cites that path and
      the commit that built the row;
    * the 2026-08-15 B7 M10 VERIFIED-bar entry, whose one `v0.1.0` mention
      carried no path and no sha. Driven at the tag: `relational_assumes`
      appears **zero** times anywhere under `v0.1.0:src/stelling/`,
      `slice_obligation` there takes `(closed, index, env, *,
      top_primitives=None)` and has no parameter `_bar_scope` could omit,
      and the word `assumes` does not occur in
      `v0.1.0:src/stelling/obligation.py` at all — so the recorded script
      has no axiom lines for a re-emission to be short of.

    **WHEN GIT IS NOT THERE**, this skips, with the two reason strings
    `tests/test_skip_inventory.py` already declares and on byte-for-byte
    the predicates it declares them legitimate for, so the skip is
    disclosed by construction rather than by a second copy of a condition.
    Driven with `git` off `PATH`: `1 skipped`, reason `needs git`, and
    `tests/test_skip_inventory.py` run beside it in that same session
    reports everything it saw as disclosed.
    A checkout whose git cannot resolve the tag is a HARD RED carrying
    git's own stderr and not a skip: that is the shape
    `tests/test_sdist_contents.py`'s own gate was corrected for — a
    control reporting green while it did not run — and no rule over there
    declares it legitimate.

    **DRIVEN RED FIVE WAYS**, on the real files, restored byte-identically
    (sha256) after each and the whole file back at `12 passed`:

    * BOTH shas removed from the 2026-08-15 B6 regression bullet, leaving
      it with no citation of either kind — names that bullet, uncovered;
    * `v0.1.0:src/stelling/_tripwire/eager.py` in `CHANGELOG.md`'s Mode 2
      paragraph reworded off the ABSENCE spelling, so a path that is NOT
      in the tag's tree reads as a claim that it is — names the file, the
      line, the path, and which way it disagrees;
    * a path that IS in the tag's tree
      (`v0.1.0:src/stelling/obligation.py`) written with `does not exist`
      after it — the exact shape that cost passes 3, 4 and 5;
    * the `predates` sentence's whole evidence clause deleted from Mode 3
      — names that paragraph, uncovered, which is the direction a rule
      scoped to the `## Log` would never have looked in;
    * both shas in that same B6 bullet replaced by `e67688e`, which is the
      tag's own commit and therefore an ancestor of it — names the bullet
      uncovered, which is the leg's non-vacuity: the commit leg accepts a
      commit for POSTDATING the tag and not for being seven hex digits.
    """
    if shutil.which("git") is None:
        # Byte-for-byte the predicate `tests/test_skip_inventory.py` declares
        # legitimate for this reason string.
        pytest.skip("needs git")
    if not (REPO / ".git").exists():
        # A worktree's `.git` is a FILE, hence `exists()` and not `is_dir()`.
        pytest.skip("not a git checkout (an unpacked sdist, say)")

    tag = _git("rev-parse", "--verify", "--quiet", f"{_TAG}^{{commit}}")
    assert tag.returncode == 0, (
        f"this is a git checkout and git cannot resolve `{_TAG}`, so every "
        f"claim below about the tag's tree is unverified here. Not a skip: a "
        f"control that reports green while it did not run is the defect this "
        f"page is about. git said: {tag.stderr.strip()!r}"
    )

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

    uncovered, wrong_polarity = [], []
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
        shas = [sha for sha in cited if postdates_the_tag(sha)]
        if not paths and not shas:
            uncovered.append((where, text.splitlines()[0][:64]))

    assert not wrong_polarity, (
        f"{len(wrong_polarity)} tag-tree citation(s) disagree with the "
        f"tag:\n  "
        + "\n  ".join(
            f"{w}: `{_TAG}:{p}` {why}" for w, p, why in wrong_polarity
        )
        + f"\nA `{_TAG}:<path>` citation claims the path IS in the tag's tree "
        f"unless it is immediately followed by `does not exist`, `did not "
        f"exist`, `is absent` or `does not appear`. `git cat-file -e` decides "
        f"it. This is exactly what refutes the justification that cost passes "
        f"3, 4 and 5."
    )
    assert not uncovered, (
        f"{len(uncovered)} claim(s) that the released `{_TAG}` does not carry "
        f"a defect cite nothing this check can decide:\n  "
        + "\n  ".join(f"{w}: {h}" for w, h in uncovered)
        + f"\nEach must cite a `{_TAG}:<path>` (decided with `git cat-file "
        f"-e` against the claim's polarity) or a commit that is NOT an "
        f"ancestor of `{_TAG}` (decided with `git merge-base "
        f"--is-ancestor`). NECESSARY, NOT SUFFICIENT: neither decides "
        f"whether the tag carries the defect — only running an extracted "
        f"`{_TAG}` does that, and it is the work the entry is supposed to "
        f"have done. What this refuses is a justification about the tag's "
        f"TREE that nothing in the tree was ever asked about."
    )
