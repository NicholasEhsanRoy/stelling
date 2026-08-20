# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The routing is CHECKED, block by block, not asserted in prose.

`CHANGELOG.md`'s `### Soundness fixes` section carried 2990 of that file's
3778 lines at `8f0adf2`. `DOCUMENTATION_ARCHITECTURE.md` §8.3 says that
section is one-liners linking to the ledger and *"never restates the
predicate — one source of truth"*. Routing the detail out was chosen over
summarising it **because routing is lossless and mechanically verifiable
and summarising is neither**: every block that leaves has to arrive, and
that is a property a test can hold. So this file holds it.

**THE DEFECT CLASS THIS FILE IS SHAPED AGAINST.** A check spelled *"the
changelog must not make claim X"* passes when the paragraph making claim X
is deleted — including when the correctly-scoped replacement is deleted
with it. Absence reads as compliance, and this campaign has closed that
shape more than five times. So nothing here is a "must not contain"
check. Every check below is a PARTITION, in both directions:

* every ID in `_soundness_routing_manifest.ROUTED` has exactly one
  one-liner in `CHANGELOG.md` and exactly one detail section in
  `SOUNDNESS.md`;
* every `SF-` one-liner in `CHANGELOG.md` and every `#### SF-` heading in
  `SOUNDNESS.md` is in the manifest;
* the detail section's text hashes to the manifest's `dest_sha256`.

Delete an entry from either file and its counterpart is orphaned, so the
partition breaks and this goes red naming the ID. Delete the whole
section and 66 IDs are orphaned at once. There is no state in which
"gone" is quiet.

**WHAT IS CHECKED HERE AND WHAT IS NOT.**

* Checked from this tree alone: that each detail section is present, is
  unique, hashes to the pinned value, is linked by exactly one one-liner
  under the anchor GitHub will mint for its heading, and that each
  one-liner is a one-liner — short, and carrying a version field drawn
  from a closed set of two phrases.
* Checked only where git and `SOURCE_COMMIT` are present:
  `test_the_source_hashes_reproduce_from_git` re-splits the pre-routing
  `CHANGELOG.md` with the same splitter and compares all 66
  `src_sha256`. That is the leg that makes `src_sha256` a measurement
  rather than a promise, and it SKIPS in an sdist. It is named here
  because a reader is entitled to know which leg they have.
* NOT checked: whether a one-liner's sentence is a *good* summary of its
  detail. It is not a summary — it is the block's own headline, moved —
  but nothing here proves that, and a rewritten one-liner would pass.
* NOT checked: anything about `SOUNDNESS.md`'s `## Log`, which is a
  different section with a different job and was not routed into.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

import pytest

from _soundness_routing_manifest import (  # noqa: E402
    DROPPED,
    ROUTED,
    SOURCE_COMMIT,
    SOURCE_SPAN,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"
SOUNDNESS = REPO / "SOUNDNESS.md"

_ID = re.compile(r"SF-0\.2\.0-\d{2}")
_VERSION_FIELD = (
    "Versions: 0.2.0 development builds only.",
    "Versions: `v0.1.0` and 0.2.0 development builds.",
)
#: A one-liner is one bullet. Wrapped at the file's column width the
#: longest is four physical lines; five is the ceiling, and it is here so
#: the section cannot re-grow a paragraph at a time.
_ONE_LINER_MAX_LINES = 5


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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detail_sections(soundness: str) -> dict[str, str]:
    """`{id: body}` for every `#### SF-…` heading in `SOUNDNESS.md`.

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


def _changelog_section(changelog: str) -> str:
    """The `### Soundness fixes` section body, or a loud failure.

    Loud rather than empty on purpose: an empty string would make every
    "is not present" check below pass, which is exactly the shape this
    file exists to refuse.
    """
    m = re.search(
        r"^### Soundness fixes\n(.*?)(?=^### )", changelog, re.S | re.M
    )
    assert m, (
        "CHANGELOG.md has no `### Soundness fixes` section followed by "
        "another `###` heading. This is the section the routing is about; "
        "if it was renamed or removed, every check in this file is about "
        "nothing and says so here rather than passing vacuously."
    )
    return m.group(1)


def _one_liner_blocks(section: str) -> dict[str, list[str]]:
    """`{id: lines}` for each `- **SF-…**` bullet in the changelog section."""
    out: dict[str, list[str]] = {}
    cur: str | None = None
    for line in section.split("\n"):
        m = re.match(r"^- \*\*(SF-0\.2\.0-\d{2})\*\*", line)
        if m:
            cur = m.group(1)
            assert cur not in out, f"{cur} has more than one one-liner"
            out[cur] = [line]
        elif cur is not None and line.startswith("  "):
            out[cur].append(line)
        elif not line.strip():
            cur = None
        else:
            cur = None
    return out


@pytest.fixture(scope="module")
def files() -> tuple[str, str]:
    return (
        CHANGELOG.read_text(encoding="utf-8"),
        SOUNDNESS.read_text(encoding="utf-8"),
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


def test_soundness_md_holds_no_detail_section_nothing_routed(files):
    """The other direction: a detail section nobody links to is orphaned.

    Without this the partition is one-way, and one-way is how a file grows
    a section that no changelog entry points at.
    """
    _, soundness = files
    known = {b.id for b in ROUTED}
    extra = sorted(set(_detail_sections(soundness)) - known)
    assert not extra, f"SOUNDNESS.md details IDs that are not routed: {extra}"


def test_every_routed_block_has_exactly_one_changelog_one_liner(files):
    """Both directions, over the changelog side.

    An entry deleted from `CHANGELOG.md` fails here rather than passing as
    "the claim is gone". An ID added to `CHANGELOG.md` with no manifest
    row fails too.
    """
    changelog, _ = files
    section = _changelog_section(changelog)
    entries = {b.id for b in ROUTED if b.kind == "entry"}
    context = {b.id for b in ROUTED if b.kind == "context"}

    liners = _one_liner_blocks(section)
    assert set(liners) == entries, (
        f"one-liners present but not routed: {sorted(set(liners) - entries)}; "
        f"routed but with no one-liner: {sorted(entries - set(liners))}"
    )
    for ident in context:
        assert section.count(ident) == 1, (
            f"{ident} is a batch heading or provenance note and must be "
            f"named exactly once in the changelog section; found "
            f"{section.count(ident)}"
        )
    mentioned = set(_ID.findall(section))
    assert mentioned == entries | context, (
        f"changelog names IDs that are not routed: "
        f"{sorted(mentioned - (entries | context))}"
    )


def test_every_changelog_line_links_to_its_own_anchor(files):
    """The link is to the ID's OWN section, not merely to `SOUNDNESS.md`.

    A link to the file is what the section had before the routing and is
    what §8.3 already asked for; a link to the entry is what makes the
    one-liner usable.
    """
    changelog, _ = files
    section = _changelog_section(changelog)
    for block in ROUTED:
        target = f"(SOUNDNESS.md#{block.anchor})"
        assert section.count(target) == 1, (
            f"{block.id}: expected exactly one link to {target} in the "
            f"changelog section, found {section.count(target)}"
        )


def test_the_changelog_entries_are_one_liners_with_a_version_field(files):
    """*Strict one-liners: ID, one-sentence statement, affected versions,
    link.* Each half of that is checked.

    The version field is drawn from a closed set of two phrases rather
    than pattern-matched, so an entry cannot satisfy it with a hedge.
    """
    changelog, _ = files
    liners = _one_liner_blocks(_changelog_section(changelog))
    long = {k: len(v) for k, v in liners.items() if len(v) > _ONE_LINER_MAX_LINES}
    assert not long, (
        f"one-liner(s) longer than {_ONE_LINER_MAX_LINES} lines — the "
        f"section is re-growing prose: {long}"
    )
    for ident, lines in liners.items():
        text = " ".join(x.strip() for x in lines)
        assert sum(text.count(v) for v in _VERSION_FIELD) == 1, (
            f"{ident} does not carry exactly one of the two permitted "
            f"version fields: {text[:160]!r}"
        )


def test_nothing_was_dropped_and_every_edit_is_declared():
    """A routing that quietly loses material is a summarisation wearing
    routing's name, so the two ways it could happen are both closed.

    `DROPPED` must justify anything deliberately not routed, and an
    "edited" block must declare how many source lines it did not carry —
    the number the destination is held to, so an edit cannot be a deletion
    with a note attached.
    """
    for sha, reason in DROPPED:
        assert len(reason) >= 40, f"dropped block {sha[:12]} has no real reason"
    edited = [b for b in ROUTED if b.src_sha256 != b.dest_sha256]
    assert edited, (
        "no block differs from its source, which would mean the two "
        "corrections this routing made are gone — an unreproducible hash "
        "literal and a solver workaround's obsolete justification"
    )
    for block in edited:
        assert block.edit_note, (
            f"{block.id} differs from the text that left CHANGELOG.md and "
            f"declares no reason. An undeclared edit is how a routing loses "
            f"material while every hash still matches."
        )
        assert len(block.edit_note) >= 40, f"{block.id}'s reason says nothing"
        assert block.src_lines_not_carried <= 3, (
            f"{block.id} did not carry {block.src_lines_not_carried} source "
            f"lines: that is a rewrite, not an edit in transit, and a rewrite "
            f"is a summarisation with a note attached"
        )
    for block in ROUTED:
        if block.src_sha256 == block.dest_sha256:
            assert block.src_lines_not_carried == 0
            assert not block.edit_note


def test_the_source_hashes_reproduce_from_git(files):
    """`src_sha256` is a measurement, re-taken here from the pre-routing file.

    This is the leg that closes "every block that leaves must arrive": the
    checks above pin the destination, and this one pins the destination to
    what actually left. It needs git and `SOURCE_COMMIT`, so it SKIPS in
    an sdist rather than passing there — an sdist keeps the destination
    checks and loses this one, and saying so is the point of the skip
    message.
    """
    try:
        r = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:CHANGELOG.md"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        pytest.skip(f"git unavailable, so src_sha256 is unverified here: {e}")
    if r.returncode != 0:
        pytest.skip(
            f"cannot read {SOURCE_COMMIT}:CHANGELOG.md (shallow clone or "
            f"sdist), so src_sha256 is unverified here: {r.stderr[:200]}"
        )
    blocks = split_blocks(r.stdout, SOURCE_SPAN)
    assert len(blocks) == len(ROUTED), (
        f"the source section splits into {len(blocks)} blocks and the "
        f"manifest has {len(ROUTED)}"
    )
    for (s, e, text), block in zip(blocks, ROUTED):
        assert (s, e) == block.src_span, f"{block.id} moved: {(s, e)}"
        assert _sha(text) == block.src_sha256, (
            f"{block.id}: the source block at {s}-{e} of "
            f"{SOURCE_COMMIT}:CHANGELOG.md does not hash to the manifest's "
            f"src_sha256"
        )


def test_the_splitter_partitions_the_source(files):
    """Every non-blank line of the source section is in exactly one block.

    Without this, "every block arrived" is compatible with a line that was
    never in a block at all — the loss the manifest could not see.
    """
    try:
        r = subprocess.run(
            ["git", "show", f"{SOURCE_COMMIT}:CHANGELOG.md"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as e:  # pragma: no cover
        pytest.skip(f"git unavailable: {e}")
    if r.returncode != 0:
        pytest.skip("source commit unavailable (shallow clone or sdist)")
    lines = r.stdout.split("\n")
    blocks = split_blocks(r.stdout, SOURCE_SPAN)
    covered: set[int] = set()
    for s, e, _ in blocks:
        span = set(range(s, e + 1))
        assert not (span & covered), "blocks overlap"
        covered |= span
    uncovered = [
        i for i in range(SOURCE_SPAN[0], SOURCE_SPAN[1] + 1)
        if i not in covered and lines[i - 1].strip()
    ]
    assert not uncovered, (
        f"{len(uncovered)} non-blank source line(s) belong to no block, so "
        f"they were routed by nothing: {uncovered[:10]}"
    )


# ------------------------------------------------------------ does it bite


def test_these_checks_bite(files):
    """The mutations this file exists to catch, driven against copies.

    A partition check is only worth its docstring if the four ways a
    routing can go wrong actually turn it red, and three of the four are
    DELETIONS — the shape that a "must not contain claim X" check would
    wave through.
    """
    changelog, soundness = files
    victim = ROUTED[2]           # an ordinary entry block
    assert victim.kind == "entry"

    # 1. the entry is deleted from CHANGELOG.md
    liners = _one_liner_blocks(_changelog_section(changelog))
    cut = changelog.replace("\n".join(liners[victim.id]) + "\n", "")
    assert cut != changelog
    with pytest.raises(AssertionError, match="no one-liner"):
        test_every_routed_block_has_exactly_one_changelog_one_liner((cut, soundness))

    # 2. the whole section is deleted from CHANGELOG.md — the case a
    #    "must not contain" check passes on
    gutted = re.sub(
        r"^### Soundness fixes\n.*?(?=^### )", "### Soundness fixes\n\n",
        changelog, flags=re.S | re.M,
    )
    assert "SF-0.2.0-03" not in gutted
    with pytest.raises(AssertionError):
        test_every_routed_block_has_exactly_one_changelog_one_liner((gutted, soundness))

    # 3. the detail is deleted from SOUNDNESS.md
    sections = _detail_sections(soundness)
    dropped = soundness.replace(
        f"#### {victim.id}\n\n{sections[victim.id]}\n", "")
    assert dropped != soundness
    with pytest.raises(AssertionError, match="no detail section"):
        test_every_routed_block_arrives_in_soundness_md((changelog, dropped))

    # 4. the detail is TRUNCATED rather than removed — the shape a
    #    presence check cannot see
    body = sections[victim.id]
    trimmed = soundness.replace(body, "\n".join(body.split("\n")[:3]))
    assert trimmed != soundness
    with pytest.raises(AssertionError, match="dest_sha256"):
        test_every_routed_block_arrives_in_soundness_md((changelog, trimmed))

    # 5. a one-liner re-grows into a paragraph
    grown = changelog.replace(
        "\n".join(liners[victim.id]),
        "\n".join(liners[victim.id]) + "\n  and then a further"
        "\n  five\n  lines\n  of\n  prose",
    )
    with pytest.raises(AssertionError, match="re-growing prose"):
        test_the_changelog_entries_are_one_liners_with_a_version_field(
            (grown, soundness))
