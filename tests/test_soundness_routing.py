# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The routing is CHECKED, block by block, not asserted in prose.

`CHANGELOG.md`'s `### Soundness fixes` section carried 2990 of that file's
3778 lines at `8f0adf2`, and its `### The eager construction-site detector
(Mode 2), DEFAULT-OFF` section carried 242 of 1158 at `de80ad8`.
`DOCUMENTATION_ARCHITECTURE.md` §8.3 says the Soundness section is
one-liners linking to the ledger and *"never restates the predicate — one
source of truth"*. Routing the detail out was chosen over summarising it
**because routing is lossless and mechanically verifiable and summarising
is neither**: every block that leaves has to arrive, and that is a
property a test can hold. So this file holds it, for every section in
`_soundness_routing_manifest.SECTIONS`.

**THE DEFECT CLASS THIS FILE IS SHAPED AGAINST.** A check spelled *"the
changelog must not make claim X"* passes when the paragraph making claim X
is deleted — including when the correctly-scoped replacement is deleted
with it. Absence reads as compliance, and this campaign has closed that
shape more than five times. So nothing here is a "must not contain"
check. Every check below is a PARTITION, in both directions:

* every ID in `_soundness_routing_manifest.ROUTED` has exactly one
  one-liner in `CHANGELOG.md` and exactly one detail section in
  `SOUNDNESS.md`;
* every one-liner in `CHANGELOG.md` and every `#### ` routed heading in
  `SOUNDNESS.md` is in the manifest;
* every non-blank line of the changelog section after its preamble
  belongs to exactly one one-liner or context block, and each of those
  names exactly one routed ID;
* the detail section's text hashes to the manifest's `dest_sha256`.

Delete an entry from either file and its counterpart is orphaned, so the
partition breaks and this goes red naming the ID. Delete a whole section
and its IDs are orphaned at once. There is no state in which "gone" is
quiet.

**WHAT IS CHECKED HERE AND WHAT IS NOT.**

* Checked from this tree alone: that each detail section is present, is
  unique, hashes to the pinned value, is linked by exactly one one-liner
  under the anchor GitHub will mint for its heading, and that each
  one-liner is a one-liner — short, and carrying a version field drawn
  from a closed set of phrases.
* Checked only where git and the section's `source_commit` are present,
  by the three git-gated tests:
  `test_the_source_span_is_derived_and_not_trusted` recomputes
  `Section.source_span` from the source file;
  `test_the_source_hashes_reproduce_from_git` re-splits the pre-routing
  `CHANGELOG.md` with the same splitter, compares every `src_sha256`,
  and MEASURES each edited block's `src_lines_not_carried` against the
  source; `test_the_splitter_partitions_the_source` shows every non-blank
  source line was in some block. Those three SKIP in an sdist.
* **NOT checked outside git, and this is the sharpest limit here.** The
  destination checks are a partition of the manifest against the two
  shipped files, so they catch any mutation that touches only one of the
  three. They do NOT catch a COORDINATED one: drop a block's row from
  this manifest, its one-liner from `CHANGELOG.md` and its detail section
  from `SOUNDNESS.md`, and every destination check still agrees, because
  the three now describe the same smaller world. What refuses that is the
  source: the section still splits into the old number of blocks at the
  old spans. So for the coordinated-deletion class the git leg is not
  merely *a* guard, it is the SOLE guard — an sdist keeps the other
  checks and loses this one entirely. The skip messages say the leg is
  gone; this says what goes with it.
* NOT checked: whether a one-liner's sentence is a *good* summary of its
  detail. It is not a summary — it is the block's own headline, moved,
  or a sentence written to stand alone in its place — but nothing here
  proves that, and a rewritten one-liner would pass.
* NOT checked: anything about `SOUNDNESS.md`'s `## Log`, which is a
  different section with a different job and was not routed into.
  `tests/test_soundness_log_reach.py` holds that section.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

import pytest

from _release_record import CHANGELOG_VERSION_FIELDS  # noqa: E402
from _soundness_routing_manifest import (  # noqa: E402
    DROPPED,
    ROUTED,
    SECTIONS,
    Section,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"
SOUNDNESS = REPO / "SOUNDNESS.md"

#: Every routed ID, as a pattern. Built from the sections rather than
#: written out, so a new section's IDs are seen by every check here on the
#: day the section is added and not on the day someone remembers.
_ID = re.compile(
    "|".join(re.escape(s.id_prefix) + r"\d{2}" for s in SECTIONS)
)
#: The two phrases a 0.2.0 one-liner may carry, from the vocabulary
#: `tests/_release_record.py` defines once for this file and for
#: `tests/test_soundness_log_reach.py`. See that module for why the
#: vocabulary has three and this subset has two.
_VERSION_FIELD = CHANGELOG_VERSION_FIELDS
#: A one-liner is one paragraph. Wrapped at the file's column width the
#: longest is four physical lines; five is the ceiling, and it is here so
#: the section cannot re-grow a paragraph at a time.
_ONE_LINER_MAX_LINES = 5

#: The section's preamble — everything before its first one-liner bullet —
#: says what the section is and points at the ledger. It is prose on
#: purpose, so it is not held to `_ONE_LINER_MAX_LINES`; but "not a
#: one-liner" is not "unbounded", because an unbounded preamble is a place
#: an entry can be re-grown where no per-entry check looks. Sixty lines,
#: measured against both ends: today's preambles are 29 and 16 non-blank
#: lines, and the average routed soundness block is 45 (2989 source lines
#: over 66 blocks), so this admits an explanation and refuses an entry.
_PREAMBLE_MAX_LINES = 60

#: `**N blocks were edited in transit**` — the count beside the data, in
#: the paragraph whose subject is that nothing was lost. The digit is
#: DERIVED from the manifest by
#: `test_the_record_says_how_many_blocks_were_edited_in_transit`; it read
#: "Two" against a manifest declaring three until the B8c fixup, in the
#: one sentence in the record that is about the edits.
_EDIT_COUNT_RE = re.compile(
    r"\*\*(?P<n>[A-Za-z]+|\d+)\s+blocks?\s+(?:was|were)\s+edited\s+in\s+transit\*\*",
    re.S,
)
_NUMBER_WORD = {
    "no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# --------------------------------------------------------------- reading


def slug(heading: str) -> str:
    """GitHub's heading-anchor rule, as this repository already relies on it.

    Lowercase; drop every character that is not alphanumeric, hyphen,
    underscore or space; spaces to hyphens. `test_the_anchor_rule_is_the_one_this_repo_already_links_by`
    pins it against three anchors that predate this file, so the rule is
    not this file's guess about GitHub — it is the repository's own
    working evidence.
    """
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower(), flags=re.UNICODE)
    return s.replace(" ", "-")


def split_blocks(text: str, span: tuple[int, int]) -> list[tuple[int, int, str]]:
    """The partition the routing used, re-implemented once and shared.

    A block starts at a top-level bullet (`- ` at column 0) or at a
    non-indented non-bullet line that follows a blank one, and runs to the
    line before the next start with trailing blanks trimmed. Every
    non-blank line of the span lands in exactly one block —
    `test_the_splitter_partitions_the_source` is that claim, driven.
    """
    lines = text.split("\n")
    start, end = span
    starts = []
    for i in range(start, end + 1):
        line, prev = lines[i - 1], lines[i - 2]
        if line.startswith("- ") or (
            line and not line[0].isspace() and prev.strip() == ""
        ):
            starts.append(i)
    out = []
    for n, s in enumerate(starts):
        e = (starts[n + 1] - 1) if n + 1 < len(starts) else end
        while e > s and lines[e - 1].strip() == "":
            e -= 1
        out.append((s, e, "\n".join(lines[s - 1:e])))
    return out


def derive_source_span(text: str, heading: str) -> tuple[int, int]:
    """`Section.source_span`, COMPUTED from the source rather than trusted.

    The span was a hand-written literal, and a hand-written literal that
    nothing checks is where a deletion hides. Shrink it by one block,
    drop that block's row from the manifest, its one-liner from
    `CHANGELOG.md` and its detail section from `SOUNDNESS.md`, and every
    other check in this file agrees with every other: the source splits
    into as many blocks as the manifest has, at the spans the manifest
    gives, and both shipped documents name the same smaller ID set. A
    whole soundness entry leaves the project green. Deriving the span
    from the heading structure is what refuses that, because the heading
    the section ends at is not the manifest's to choose.

    The span is the section BODY: the line after `heading` through the
    line before the next heading at the same level, 1-based inclusive.
    """
    lines = text.split("\n")
    level = len(heading) - len(heading.lstrip("#"))
    same = [
        i for i, line in enumerate(lines, 1)
        if line.startswith("#" * level + " ")
        and not line.startswith("#" * (level + 1))
    ]
    here = [i for i in same if lines[i - 1].rstrip() == heading]
    assert len(here) == 1, (
        f"{heading!r} appears {len(here)} times in the source file; the "
        f"span of a section that is not exactly one section is not a "
        f"thing this can derive"
    )
    after = [i for i in same if i > here[0]]
    assert after, (
        f"{heading!r} is the last heading at its level in the source "
        f"file, so its end is the end of the file and the derivation "
        f"below would be silently open-ended"
    )
    return (here[0] + 1, after[0] - 1)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lines_not_carried(src_text: str, dest_text: str) -> list[str]:
    """The block's non-blank source lines that the destination does not carry.

    Verbatim and line-for-line, leading whitespace included: a block that
    arrived re-wrapped or re-indented did not arrive, it was retyped, and
    this is the number `Block.src_lines_not_carried` is measured against.
    """
    dest = set(dest_text.split("\n"))
    return [ln for ln in src_text.split("\n") if ln.strip() and ln not in dest]


def _detail_sections(soundness: str) -> dict[str, str]:
    """`{id: body}` for every routed `#### …` heading in `SOUNDNESS.md`.

    The body runs to the next heading of any level, so a section that
    swallowed its successor is a hash mismatch and not a silent pass.
    """
    lines = soundness.split("\n")
    heads = [
        (i, m.group(0))
        for i, line in enumerate(lines)
        if line.startswith("#### ") and (m := _ID.fullmatch(line[5:].strip()))
    ]
    out: dict[str, str] = {}
    for n, (i, ident) in enumerate(heads):
        stop = len(lines)
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("#"):
                stop = j
                break
        body = "\n".join(lines[i + 1:stop]).strip("\n")
        assert ident not in out, f"{ident} has more than one detail section"
        out[ident] = body
    return out


def _changelog_section(changelog: str, heading: str) -> str:
    """A `### …` section body, or a loud failure.

    Loud rather than empty on purpose: an empty string would make every
    "is not present" check below pass, which is exactly the shape this
    file exists to refuse.
    """
    m = re.search(
        rf"^{re.escape(heading)}\n(.*?)(?=^### )", changelog, re.S | re.M
    )
    assert m, (
        f"CHANGELOG.md has no `{heading}` section followed by another "
        f"`###` heading. This is a section the routing is about; if it "
        f"was renamed or removed, every check in this file about it is "
        f"about nothing and says so here rather than passing vacuously."
    )
    return m.group(1)


def _paragraphs(section: str) -> list[list[str]]:
    """The section's blank-line-separated runs of non-blank lines.

    The PARAGRAPH is the unit of every length and shape check below, and
    the indented continuation line is not, because indentation is a
    typographic choice and the thing being bounded is prose.
    """
    out: list[list[str]] = []
    cur: list[str] = []
    for line in section.split("\n"):
        if line.strip():
            cur.append(line)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _one_liner_blocks(section: str, prefix: str) -> dict[str, list[str]]:
    """`{id: lines}` for each `- **<prefix>NN**` bullet, WITH EVERY LINE OF
    ITS PARAGRAPH.

    Counting only the INDENTED continuation lines is how a ceiling of
    five lines admitted thirty: the bullet was dropped on the first
    column-0 line, so column-0 prose was never counted against anything.
    Thirty unindented lines glued to a one-liner passed; the same thirty
    indented failed. The paragraph is the unit now, so prose is the
    one-liner's length whatever its indentation, and prose given a
    paragraph of its own is caught by
    `test_the_section_holds_only_one_liners_and_context_blocks`.
    """
    pat = re.compile(rf"^- \*\*({re.escape(prefix)}\d{{2}})\*\*")
    out: dict[str, list[str]] = {}
    for para in _paragraphs(section):
        m = pat.match(para[0])
        if not m:
            continue
        ident = m.group(1)
        assert ident not in out, f"{ident} has more than one one-liner"
        out[ident] = para
    return out


@pytest.fixture(scope="module")
def files() -> tuple[str, str]:
    return (
        CHANGELOG.read_text(encoding="utf-8"),
        SOUNDNESS.read_text(encoding="utf-8"),
    )


def _source_changelog(section: Section) -> str:
    """`section.source_commit:CHANGELOG.md`, or a skip that says what is lost.

    The skip is not "this check is unavailable"; it is "the coordinated
    deletion described in this module's docstring has no guard left in
    this tree", and the message says so where a reader will meet it.
    """
    ref = f"{section.source_commit}:CHANGELOG.md"
    try:
        r = subprocess.run(
            ["git", "show", ref],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        pytest.skip(
            f"git unavailable, so `{section.key}`'s source is unverified "
            f"here and a COORDINATED deletion across the manifest and both "
            f"shipped files has no remaining guard: {e}"
        )
    if r.returncode != 0:
        pytest.skip(
            f"cannot read {ref} (shallow clone or sdist), so `{section.key}`'s "
            f"source span, src_sha256 and src_lines_not_carried are "
            f"unverified here and a COORDINATED deletion across the manifest "
            f"and both shipped files has no remaining guard: {r.stderr[:200]}"
        )
    return r.stdout


_SECTION = pytest.mark.parametrize(
    "section", SECTIONS, ids=[s.key for s in SECTIONS]
)


# ---------------------------------------------------------------- checks


def test_the_anchor_rule_is_the_one_this_repo_already_links_by():
    """`slug` is not a guess about GitHub; it is checked against links that
    predate this file and are known to resolve.

    Without this, a wrong slug rule would be wrong in the links AND wrong
    in the checker that reads them — self-consistently green, and every
    link on the page broken.
    """
    assert slug("any_array(shape, dtype, bounds)") == "any_arrayshape-dtype-bounds"
    assert (
        slug("Don't hand-roll a traversal when a canonical accessor exists")
        == "dont-hand-roll-a-traversal-when-a-canonical-accessor-exists"
    )
    assert (
        slug("What this checks — and what it doesn't yet")
        == "what-this-checks--and-what-it-doesnt-yet"
    )
    for block in ROUTED:
        assert slug(block.id) == block.anchor, block.id


def test_every_routed_block_arrives_in_soundness_md(files):
    """Every block that left `CHANGELOG.md` is in `SOUNDNESS.md`, hashed.

    This is the whole reason routing was chosen over summarising, and it
    is the one claim that cannot be made by writing a sentence.
    """
    _, soundness = files
    sections = _detail_sections(soundness)
    missing = [b.id for b in ROUTED if b.id not in sections]
    assert not missing, (
        f"{len(missing)} routed block(s) have no detail section in "
        f"SOUNDNESS.md: {missing[:8]}"
    )
    wrong = [
        (b.id, _sha(sections[b.id])[:12], b.dest_sha256[:12])
        for b in ROUTED
        if _sha(sections[b.id]) != b.dest_sha256
    ]
    assert not wrong, (
        "detail section(s) no longer hash to the manifest's dest_sha256 — "
        "the routed text has been edited without the manifest being "
        f"updated: {wrong[:5]}"
    )


def test_every_routed_detail_section_sits_under_its_own_destination_heading(files):
    """A section's blocks arrive under the `##` heading the manifest names.

    Without this the destination heading is decoration: the Mode 2 detail
    could be filed under the soundness-fix heading, or under none at all,
    and every hash would still match. The heading is what tells a reader
    which ledger they are in.
    """
    _, soundness = files
    lines = soundness.split("\n")
    owner: dict[str, str] = {}
    current = ""
    for line in lines:
        if line.startswith("## ") and not line.startswith("###"):
            current = line.rstrip()
        elif line.startswith("#### ") and _ID.fullmatch(line[5:].strip()):
            owner[line[5:].strip()] = current
    for section in SECTIONS:
        for block in section.blocks:
            assert owner.get(block.id) == section.dest_heading, (
                f"{block.id} is filed under {owner.get(block.id)!r} and the "
                f"manifest routes it to {section.dest_heading!r}"
            )


def test_soundness_md_holds_no_detail_section_nothing_routed(files):
    """The other direction: a detail section nobody links to is orphaned.

    Without this the partition is one-way, and one-way is how a file grows
    a section that no changelog entry points at.
    """
    _, soundness = files
    known = {b.id for b in ROUTED}
    extra = sorted(set(_detail_sections(soundness)) - known)
    assert not extra, f"SOUNDNESS.md details IDs that are not routed: {extra}"


@_SECTION
def test_every_routed_block_has_exactly_one_changelog_one_liner(files, section):
    """Both directions, over the changelog side.

    An entry deleted from `CHANGELOG.md` fails here rather than passing as
    "the claim is gone". An ID added to `CHANGELOG.md` with no manifest
    row fails too.
    """
    changelog, _ = files
    body = _changelog_section(changelog, section.heading)
    entries = {b.id for b in section.blocks if b.kind == "entry"}
    context = {b.id for b in section.blocks if b.kind == "context"}

    liners = _one_liner_blocks(body, section.id_prefix)
    assert set(liners) == entries, (
        f"one-liners present but not routed: {sorted(set(liners) - entries)}; "
        f"routed but with no one-liner: {sorted(entries - set(liners))}"
    )
    for ident in context:
        assert body.count(ident) == 1, (
            f"{ident} is a batch heading or provenance note and must be "
            f"named exactly once in the changelog section; found "
            f"{body.count(ident)}"
        )
    mentioned = set(_ID.findall(body))
    assert mentioned == entries | context, (
        f"changelog names IDs that are not routed to `{section.key}`: "
        f"{sorted(mentioned - (entries | context))}"
    )


@_SECTION
def test_every_changelog_line_links_to_its_own_anchor(files, section):
    """The link is to the ID's OWN section, not merely to `SOUNDNESS.md`.

    A link to the file is what the section had before the routing and is
    what §8.3 already asked for; a link to the entry is what makes the
    one-liner usable.
    """
    changelog, _ = files
    body = _changelog_section(changelog, section.heading)
    for block in section.blocks:
        target = f"(SOUNDNESS.md#{block.anchor})"
        assert body.count(target) == 1, (
            f"{block.id}: expected exactly one link to {target} in the "
            f"changelog section, found {body.count(target)}"
        )


@_SECTION
def test_the_changelog_entries_are_one_liners_with_a_version_field(files, section):
    """*Strict one-liners: ID, one-sentence statement, affected versions,
    link.* Each half of that is checked.

    The version field is drawn from a closed set of phrases rather than
    pattern-matched, so an entry cannot satisfy it with a hedge. The
    length ceiling is over the whole PARAGRAPH — see `_one_liner_blocks`
    for why the indented-lines-only version of this check admitted thirty
    lines of column-0 prose.
    """
    changelog, _ = files
    body = _changelog_section(changelog, section.heading)
    liners = _one_liner_blocks(body, section.id_prefix)
    long = {k: len(v) for k, v in liners.items() if len(v) > _ONE_LINER_MAX_LINES}
    assert not long, (
        f"one-liner(s) longer than {_ONE_LINER_MAX_LINES} lines — the "
        f"section is re-growing prose: {long}"
    )
    for ident, lines in liners.items():
        text = " ".join(x.strip() for x in lines)
        assert sum(text.count(v) for v in _VERSION_FIELD) == 1, (
            f"{ident} does not carry exactly one of the "
            f"{len(_VERSION_FIELD)} permitted version fields: {text[:160]!r}"
        )


@_SECTION
def test_the_section_holds_only_one_liners_and_context_blocks(files, section):
    """Every paragraph after the preamble names exactly one routed ID.

    This is the other half of the ceiling. Bounding the length of the
    one-liners bounds nothing if prose may stand between them: thirty
    lines in a paragraph of their own belong to no one-liner, so no
    one-liner grew, and the section grew by thirty lines. Requiring every
    paragraph after the first named ID to name exactly one leaves prose
    nowhere in the section to be, and requiring the CEILING of every such
    paragraph leaves it nowhere inside a block to be either.
    """
    changelog, _ = files
    body = _changelog_section(changelog, section.heading)
    paras = _paragraphs(body)
    bullet = re.compile(rf"^- \*\*{re.escape(section.id_prefix)}\d{{2}}\*\*")
    first = next((n for n, p in enumerate(paras) if bullet.match(p[0])), None)
    assert first is not None, (
        f"`{section.heading}` holds no one-liner bullet at all, so this "
        f"check and every other one about it is about nothing"
    )
    preamble = sum(len(p) for p in paras[:first])
    assert preamble <= _PREAMBLE_MAX_LINES, (
        f"`{section.heading}`'s preamble is {preamble} non-blank lines and "
        f"the ceiling is {_PREAMBLE_MAX_LINES}. Everything before the first "
        f"one-liner is exempt from the per-entry ceiling, so it is the one "
        f"place in the section prose can grow without any other check here "
        f"seeing it."
    )
    stray = []
    for para in paras[first:]:
        ids = sorted(set(_ID.findall("\n".join(para))))
        if len(ids) != 1:
            stray.append((para[0].strip()[:72], ids, len(para)))
    assert not stray, (
        f"paragraph(s) in `{section.heading}` after its preamble that name "
        f"no routed ID, or more than one — the section is re-growing prose "
        f"between its one-liners: {stray[:5]}"
    )
    long = [
        (para[0].strip()[:72], len(para))
        for para in paras[first:]
        if len(para) > _ONE_LINER_MAX_LINES
    ]
    assert not long, (
        f"paragraph(s) longer than {_ONE_LINER_MAX_LINES} lines — the "
        f"section is re-growing prose: {long[:5]}"
    )


@pytest.mark.parametrize("field", _VERSION_FIELD)
def test_each_permitted_version_field_is_actually_used(files, field):
    """No permitted-but-unused phrase.

    A closed set is worth having because it refuses a hedge; a member of
    it that no entry ever carries is a branch of the rule nothing has
    exercised, and this campaign has withdrawn several. Both are in use
    across the two routed sections: 57 one-liners carry the 0.2.0-only
    phrase (51 soundness, 6 Mode 2) and 7 carry the `v0.1.0` one.
    """
    changelog, _ = files
    used = 0
    for section in SECTIONS:
        body = _changelog_section(changelog, section.heading)
        for lines in _one_liner_blocks(body, section.id_prefix).values():
            used += field in " ".join(x.strip() for x in lines)
    assert used, (
        f"no one-liner in any routed section carries {field!r}. A phrase "
        f"the closed set permits and nothing uses is an option the rule "
        f"has never been tested on."
    )


def test_nothing_was_dropped_and_every_edit_is_declared():
    """A routing that quietly loses material is a summarisation wearing
    routing's name, so the two ways it could happen are both closed.

    `DROPPED` must justify anything deliberately not routed, and an
    "edited" block must declare BOTH how many of its non-blank source
    lines are not present verbatim in the destination AND which lines
    those are. This function checks the declaration is well formed and
    self-consistent; `test_the_source_hashes_reproduce_from_git` is where
    it is MEASURED against the source, and until this fixup nothing
    measured it at all — a 367-line block replaced by a three-line
    summary, with `src_lines_not_carried=3` and an honest `edit_note`,
    passed the whole file.
    """
    for sha, reason in DROPPED:
        assert len(reason) >= 40, f"dropped block {sha[:12]} has no real reason"
    edited = [b for b in ROUTED if b.src_sha256 != b.dest_sha256]
    assert edited, (
        f"no block of the {len(ROUTED)} routed differs from the text it "
        f"left as, which would mean the corrections these routings made "
        f"in transit are gone. They are declared, not optional."
    )
    for block in edited:
        assert block.edit_note, (
            f"{block.id} differs from the text that left CHANGELOG.md and "
            f"declares no reason. An undeclared edit is how a routing loses "
            f"material while every hash still matches."
        )
        assert len(block.edit_note) >= 40, f"{block.id}'s reason says nothing"
        assert block.src_lines_not_carried == len(block.not_carried), (
            f"{block.id} declares {block.src_lines_not_carried} source "
            f"line(s) not carried and quotes {len(block.not_carried)}. The "
            f"count and the lines are one measurement written twice and "
            f"they have to agree before either is worth reading."
        )
    for block in ROUTED:
        if block.src_sha256 == block.dest_sha256:
            assert block.src_lines_not_carried == 0
            assert not block.not_carried
            assert not block.edit_note


@_SECTION
def test_the_record_says_how_many_blocks_were_edited_in_transit(files, section):
    """The count beside the data is DERIVED from the data.

    `CHANGELOG.md` said *"Two blocks were edited in transit"* while this
    manifest declared three and named all three — a hand-maintained count
    standing in the one paragraph whose subject is *"nothing was lost and
    every edit is declared"*, and the block it left out was the route
    census the same batch had just corrected. A number written beside the
    thing it counts is the cheapest kind of drift there is, so it is not
    written any more: the sentence is found by pattern and its numeral is
    compared with `len(edited)`, and every edited block has to be named in
    that same paragraph.
    """
    changelog, _ = files
    body = _changelog_section(changelog, section.heading)
    edited = [b for b in section.blocks if b.src_sha256 != b.dest_sha256]
    m = _EDIT_COUNT_RE.search(body)
    assert m, (
        f"`{section.heading}` no longer states how many of its blocks were "
        f"edited in transit, in the form `**N blocks were edited in "
        f"transit**`. {len(edited)} were, and a routing that stops saying "
        f"so has stopped being checkable by a reader."
    )
    word = m.group("n")
    stated = (
        int(word) if word.isdigit() else _NUMBER_WORD.get(word.lower())
    )
    assert stated is not None, (
        f"`{section.heading}` says {word!r} blocks were edited in transit "
        f"and that is not a number this can compare with the manifest's "
        f"{len(edited)}"
    )
    assert stated == len(edited), (
        f"`{section.heading}` says {stated} block(s) were edited in transit "
        f"and the manifest declares {len(edited)}: "
        f"{[b.id for b in edited]}"
    )
    para = next(
        para for para in _paragraphs(body)
        if _EDIT_COUNT_RE.search("\n".join(para))
    )
    text = "\n".join(para)
    unnamed = [b.id for b in edited if b.id not in text]
    assert not unnamed, (
        f"`{section.heading}` states the count of blocks edited in transit "
        f"and does not name {unnamed}. A count without the names is a "
        f"number a reader cannot check."
    )


@_SECTION
def test_the_source_span_is_derived_and_not_trusted(section):
    """`Section.source_span` is recomputed from the source file.

    A hand-written span is the one input to this whole file that nothing
    else constrains, and shrinking it is a complete, self-consistent way
    to delete an entry: the span no longer covers the last block, the
    manifest no longer lists it, neither shipped file mentions it, the
    block count matches, every hash matches, and the partition is exact
    over what remains. Driven before this check existed: a whole
    soundness entry left both shipped documents and the suite reported
    `10 passed`.
    """
    text = _source_changelog(section)
    derived = derive_source_span(text, section.heading)
    assert derived == section.source_span, (
        f"`{section.key}`: the manifest's source_span is "
        f"{section.source_span} and `{section.heading}` occupies {derived} "
        f"in {section.source_commit}:CHANGELOG.md. A span that does not "
        f"match the source is a span that can be shrunk to make a deleted "
        f"block disappear from every other check here."
    )


@_SECTION
def test_the_source_hashes_reproduce_from_git(files, section):
    """`src_sha256` is a measurement, re-taken here from the pre-routing file
    — and so is `src_lines_not_carried`.

    This is the leg that closes "every block that leaves must arrive": the
    checks above pin the destination, and this one pins the destination to
    what actually left. It needs git and the section's `source_commit`, so
    it SKIPS in an sdist rather than passing there.

    **The second half of this is new in the B8c fixup and it is the
    reason the file exists.** `src_lines_not_carried` was declared per
    edited block and NEVER MEASURED — the only assertion on it was
    `<= 3`. So a 367-line block could be replaced by a three-line
    summary, `dest_sha256` updated as any honest editor would, `3`
    declared, a reason written, and the suite ran green. The manifest's
    own docstring said *"the test holds each to the number recorded"*,
    which was false about this file. It is true now: the lines are
    counted here, against the source text, and the count and the quoted
    lines must both be exactly right.
    """
    _, soundness = files
    details = _detail_sections(soundness)
    text = _source_changelog(section)
    blocks = split_blocks(text, section.source_span)
    assert len(blocks) == len(section.blocks), (
        f"`{section.key}`: the source section splits into {len(blocks)} "
        f"blocks and the manifest has {len(section.blocks)}"
    )
    for (s, e, src), block in zip(blocks, section.blocks):
        assert (s, e) == block.src_span, f"{block.id} moved: {(s, e)}"
        assert _sha(src) == block.src_sha256, (
            f"{block.id}: the source block at {s}-{e} of "
            f"{section.source_commit}:CHANGELOG.md does not hash to the "
            f"manifest's src_sha256"
        )
        assert block.id in details, f"{block.id} has no detail section"
        missing = _lines_not_carried(src, details[block.id])
        assert missing == list(block.not_carried), (
            f"{block.id}: the destination does not carry {len(missing)} of "
            f"this block's non-blank source lines and the manifest quotes "
            f"{len(block.not_carried)}. Not carried and not declared: "
            f"{[m for m in missing if m not in block.not_carried][:5]}; "
            f"declared and in fact carried: "
            f"{[d for d in block.not_carried if d not in missing][:5]}"
        )
        assert len(missing) == block.src_lines_not_carried, (
            f"{block.id} declares src_lines_not_carried="
            f"{block.src_lines_not_carried} and {len(missing)} of its "
            f"non-blank source lines are not present verbatim in "
            f"SOUNDNESS.md"
        )


@_SECTION
def test_the_splitter_partitions_the_source(section):
    """Every non-blank line of the source section is in exactly one block.

    Without this, "every block arrived" is compatible with a line that was
    never in a block at all — the loss the manifest could not see.
    """
    text = _source_changelog(section)
    lines = text.split("\n")
    blocks = split_blocks(text, section.source_span)
    covered: set[int] = set()
    for s, e, _ in blocks:
        span = set(range(s, e + 1))
        assert not (span & covered), "blocks overlap"
        covered |= span
    lo, hi = section.source_span
    uncovered = [
        i for i in range(lo, hi + 1)
        if i not in covered and lines[i - 1].strip()
    ]
    assert not uncovered, (
        f"{len(uncovered)} non-blank source line(s) belong to no block, so "
        f"they were routed by nothing: {uncovered[:10]}"
    )


# ------------------------------------------------------------ does it bite


@_SECTION
def test_these_checks_bite(files, section):
    """The mutations this file exists to catch, driven against copies.

    A partition check is only worth its docstring if the ways a routing
    can go wrong actually turn it red, and most of them are DELETIONS —
    the shape that a "must not contain claim X" check would wave through.
    """
    changelog, soundness = files
    victim = next(b for b in section.blocks if b.kind == "entry")
    body = _changelog_section(changelog, section.heading)

    # 1. the entry is deleted from CHANGELOG.md
    liners = _one_liner_blocks(body, section.id_prefix)
    cut = changelog.replace("\n".join(liners[victim.id]) + "\n", "")
    assert cut != changelog
    with pytest.raises(AssertionError, match="no one-liner"):
        test_every_routed_block_has_exactly_one_changelog_one_liner(
            (cut, soundness), section)

    # 2. the whole section is deleted from CHANGELOG.md — the case a
    #    "must not contain" check passes on
    gutted = re.sub(
        rf"^{re.escape(section.heading)}\n.*?(?=^### )",
        section.heading + "\n\n", changelog, flags=re.S | re.M,
    )
    assert victim.id not in gutted
    with pytest.raises(AssertionError):
        test_every_routed_block_has_exactly_one_changelog_one_liner(
            (gutted, soundness), section)

    # 3. the detail is deleted from SOUNDNESS.md
    sections = _detail_sections(soundness)
    dropped = soundness.replace(
        f"#### {victim.id}\n\n{sections[victim.id]}\n", "")
    assert dropped != soundness
    with pytest.raises(AssertionError, match="no detail section"):
        test_every_routed_block_arrives_in_soundness_md((changelog, dropped))

    # 4. the detail is TRUNCATED rather than removed — the shape a
    #    presence check cannot see
    detail = sections[victim.id]
    trimmed = soundness.replace(detail, "\n".join(detail.split("\n")[:3]))
    assert trimmed != soundness
    with pytest.raises(AssertionError, match="dest_sha256"):
        test_every_routed_block_arrives_in_soundness_md((changelog, trimmed))

    # 5. a one-liner re-grows into a paragraph, INDENTED
    grown = changelog.replace(
        "\n".join(liners[victim.id]),
        "\n".join(liners[victim.id]) + "\n  and then a further"
        "\n  five\n  lines\n  of\n  prose",
    )
    with pytest.raises(AssertionError, match="re-growing prose"):
        test_the_changelog_entries_are_one_liners_with_a_version_field(
            (grown, soundness), section)

    # 6. and UNINDENTED, which the indented-lines-only ceiling never saw:
    #    thirty column-0 lines glued to a one-liner passed the whole file
    glued = changelog.replace(
        "\n".join(liners[victim.id]),
        "\n".join(liners[victim.id])
        + "".join(f"\nre-grown prose line {n}" for n in range(30)),
    )
    with pytest.raises(AssertionError, match="re-growing prose"):
        test_the_changelog_entries_are_one_liners_with_a_version_field(
            (glued, soundness), section)

    # 7. the same thirty lines in a paragraph of their own, which belongs
    #    to no one-liner at all and so grows no one-liner
    beside = changelog.replace(
        "\n".join(liners[victim.id]),
        "\n".join(liners[victim.id])
        + "\n\n" + "\n".join(f"re-grown prose line {n}" for n in range(30)),
    )
    with pytest.raises(AssertionError, match="re-growing prose"):
        test_the_section_holds_only_one_liners_and_context_blocks(
            (beside, soundness), section)

    # 8. an edited block declares fewer lines not carried than it dropped
    victim_edit = next(
        (b for b in section.blocks if b.src_sha256 != b.dest_sha256), None)
    if victim_edit is not None:
        under = section._replace(blocks=tuple(
            b._replace(src_lines_not_carried=0, not_carried=())
            if b.id == victim_edit.id else b
            for b in section.blocks
        ))
        with pytest.raises(AssertionError, match="not carried|does not carry"):
            test_the_source_hashes_reproduce_from_git(files, under)
