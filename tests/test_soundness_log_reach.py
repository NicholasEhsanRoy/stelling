# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""How many entries of `SOUNDNESS.md`'s `## Log` reach a RELEASE — counted.

**THE DIGIT HAS HELD SIX VALUES.** *"(no releases yet)"*, then S11 alone,
then three, then four and five at the same moment on two branches, then
six, and now seven. Each correction was made by a person re-reading the
log. Not one of the five noticed that the 2026-08-15 B6 entry for audit
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
`v0.1.0` was tagged, which is what 39 of these bullets are about.

**THE THREE UNITS, WHICH ARE 7, 8 AND 7.** They are different questions
and this file keeps them apart, because the count was wrong twice by
answering one of them with another's answer:

* **entries** — top-level `## Log` bullets whose `Versions:` field names
  `v0.1.0`. Seven.
* **findings** — audit 0.2.0 findings reaching `v0.1.0`. Eight: one more
  than the entries, because S14 has a routed detail section
  (`SF-0.2.0-59`) and its own reach declaration but no `## Log` bullet.
* **one-liners** — `CHANGELOG.md` entries carrying the `v0.1.0` version
  field. Seven: one fewer than the findings, because S15 and S16 share
  `SF-0.2.0-14`.

Each numeral in the paragraph is compared against the thing it counts,
and the findings numeral against the list written beside it in that same
paragraph — so a finding added to the list without the digit, or a digit
moved without the list, is a failure either way round.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from _release_record import VERSION_FIELDS, release_prose  # noqa: E402
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

_ENTRIES_RE = re.compile(
    r"is\s+reached\s+by\s+\*\*(?P<n>[A-Za-z]+|\d+)\*\*\s+ENTRIES", re.S
)
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
#: constant checked itself and `60 -> 6000` was green. Today's preamble is 21
#: non-blank lines, not the 20 this comment claimed, and the median `## Log`
#: entry is 135 — so the 39 lines of headroom admit an explanation and refuse
#: an entry, which is the argument, measured, at both ends.
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
    """The ENTRIES numeral, against the entries.

    This digit read **six** against seven declaring bullets, having
    already read *"(no releases yet)"*, S11 alone, three, and four-and-
    five on two branches at once.
    """
    entries = [
        line for line, text in log_bullets() if REACHES_V010 in text
    ]
    soundness = SOUNDNESS.read_text(encoding="utf-8")
    m = _ENTRIES_RE.search(soundness)
    assert m, (
        "SOUNDNESS.md no longer states how many ENTRIES of the log reach a "
        "release, in the form `is reached by **N** ENTRIES`. "
        f"{len(entries)} do. A log that stops stating it has not stopped "
        "having the number; it has stopped letting a reader check it."
    )
    stated = _numeral(m.group("n"))
    assert stated == len(entries), (
        f"SOUNDNESS.md says {m.group('n')!r} entries of the log reach "
        f"`v0.1.0` and {len(entries)} bullets carry {REACHES_V010!r} "
        f"(lines {entries}). The omitted one was S12&prime; for five "
        f"corrections running."
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

    The third unit, and the one that would be easiest to confuse with the
    first because today they are both seven. They are not the same
    question: entries are bullets of this log, one-liners are entries of
    the changelog, and S15 and S16 are two of the former and one of the
    latter.
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
    enough to check for it. All three are in use: 39 bullets pre-release
    only, 8 in 0.2.0 development only, 7 reaching `v0.1.0`. (Named rather
    than called "the release": `0.2.0` is a release too now, and no bullet
    reaches it.)
    """
    used = [line for line, text in log_bullets() if field in text]
    assert used, (
        f"no `## Log` bullet carries {field!r}. A phrase in the closed set "
        f"that nothing uses is an untested branch of the rule, not a "
        f"choice the rule offers."
    )
