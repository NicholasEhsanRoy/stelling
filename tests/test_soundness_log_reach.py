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
two the changelog uses cannot express a defect that was fixed before the
only release was tagged, which is what 39 of these bullets are about.

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


def log_bullets() -> list[tuple[int, str]]:
    """`(line, text)` for every top-level bullet of `SOUNDNESS.md`'s `## Log`.

    Top-level means column 0 — the same rule
    `tests/test_soundness_routing.py::split_blocks` uses on the changelog,
    for the same reason: a nested bullet is part of its parent's argument
    and is not an entry.
    """
    lines = SOUNDNESS.read_text(encoding="utf-8").split("\n")
    lo = next(i for i, line in enumerate(lines, 1) if line.rstrip() == "## Log")
    hi = next(
        i for i, line in enumerate(lines, 1) if i > lo and line.startswith("## ")
    )
    starts = [i for i in range(lo + 1, hi) if lines[i - 1].startswith("- ")]
    assert starts, "SOUNDNESS.md's `## Log` holds no top-level bullet"
    out = []
    for n, s in enumerate(starts):
        e = (starts[n + 1] - 1) if n + 1 < len(starts) else hi - 1
        out.append((s, "\n".join(lines[s - 1:e])))
    return out


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
    only, 8 in 0.2.0 development only, 7 reaching the release.
    """
    used = [line for line, text in log_bullets() if field in text]
    assert used, (
        f"no `## Log` bullet carries {field!r}. A phrase in the closed set "
        f"that nothing uses is an untested branch of the rule, not a "
        f"choice the rule offers."
    )
