# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The release record is TWO files now, and a check that reads one is blind.

Batch B8c routed `CHANGELOG.md`'s `### Soundness fixes` detail into
`SOUNDNESS.md` — 2990 lines of predicate, measurement and derivation, block
for block, under `DOCUMENTATION_ARCHITECTURE.md` §8.3, which makes
`SOUNDNESS.md` the ledger and leaves the changelog one-liners that link to
it. `tests/test_soundness_routing.py` checks the move itself.

**FIVE test files went red on that move, in SEVEN functions, and every one
of them was RIGHT to.** They pin a sentence the project committed to — a
claim scoped, a route named, a figure re-derived — by reading
`CHANGELOG.md`. The sentence did not stop being committed to; it moved. So
the fix is not to delete the assertion, and it is not to weaken it: it is
to read the record where the record now is. The seven are named, so that
every sentence here is re-driveable by checking the named test out at
`8f0adf2` and running it against this tree:

* `test_assume_disclosure_claims.py::test_the_changelog_scopes_the_disclosure_claim_to_the_dispatch_path`
* `test_aval_lie_both_faces.py::test_the_element_count_census_covers_propagate_TOO`
* `test_canonicalization_routes.py::test_the_DISCLOSURES_name_exactly_these_routes`
* `test_ir_screen.py::test_the_batch_ships_an_attribution_table_that_adds_up`
* `test_ir_screen.py::test_an_attribution_row_may_not_quote_a_test_that_does_not_exist`
* `test_ir_screen.py::test_the_R1c_disclosure_EXHIBITS_its_pre_emption`
* `test_pow_row_gauge_jax.py::test_the_DOCSTRING_and_CHANGELOG_battery_SIZE_is_the_one_that_RAN`

**AND THE SIXTH FILE THE BATCH EDITED IS A DIFFERENT STORY, SEPARATED HERE
BECAUSE IT WAS TOLD AS PART OF THIS ONE.** `tests/test_tripwire_gate_coverage.py`
at `8f0adf2` runs **`11 passed`** against this tree. It reads no
documentation at all — `grep -cE 'read_text|CHANGELOG|SOUNDNESS|README|\\.md'`
over it returns **0** — so the routing could not have moved anything out
from under it. Its change in that batch (`len(unwatched) == 8` beside the
pre-existing `len(closed) == 6`) is an unrelated correction that rode
along, and calling it a sixth casualty of the routing filed a real fix
inside a story it has nothing to do with. The routing broke five files; the
denominator fix is its own entry.

**THE TRAP THIS CAMPAIGN KEEPS MEETING IS REAL, AND SMALLER THAN THE FIRST
ACCOUNT OF IT.** Three of the eight functions the batch touched pair *"the
retracted claim is absent"* with *"the scoped replacement is present"* over
the release record: `test_the_changelog_scopes_the_disclosure_claim_to_the_dispatch_path`'s
two legs over one `text`, `test_the_element_count_census_covers_propagate_TOO`'s
*"no caller anywhere"* against its named sites, and
`test_the_DISCLOSURES_name_exactly_these_routes`'s guarded-quotation scan
against its route list. The other five assert presence only. And **six of
the seven reds are on a presence leg**, which is the trap as advertised:
the absence leg passed because the paragraph was gone, so the retracted
wording was gone with it, and a checker built from absence alone would have
gone green on a routing that deleted these claims outright.

**The seventh red is the counterexample, and it points the other way.**
`test_the_DISCLOSURES_name_exactly_these_routes` failed on its **ABSENCE**
leg — *"SOUNDNESS.md still claims that only an `object.__setattr__` reaches
the residue"* — and its presence leg was never reached. The causal
direction is inverted too: nothing was deleted out of a page that check
read. The falsified sentence was quoted in `CHANGELOG.md`, which was not on
that check's page list, and the routing moved the quotation INTO
`SOUNDNESS.md`, which was. A page-list guard is only as wide as its list,
and what the routing did was widen what the list had to cover. That is a
third failure mode beside the two above, and it is the one a "read the
record where the record is" module cannot fix by itself.

`release_prose()` is deliberately a CONCATENATION and not a choice between
the files. A claim may live in either — a one-liner in the changelog, its
predicate in the ledger — and which one is an editorial decision that
should not red a test about the claim. What must never happen is the claim
being in NEITHER, and that is exactly what a concatenation still catches:
delete it from both and every caller goes red naming it.

**FOR A FIGURE, USE `release_records()`, AND THAT IS NOT A STYLE
PREFERENCE.** A concatenation widens a PRESENCE check and NARROWS a
`re.search` that reads a number out of the record, because `search` returns
the FIRST match: a correct digit in `CHANGELOG.md` masks a stale one in
`SOUNDNESS.md`, and the check passes while the ledger ships the wrong
number. Driven at this commit — the battery-size sentence written correctly
into `CHANGELOG.md` and as `999` into `SOUNDNESS.md` — the `release_prose()`
form ran **`1 passed`**. So a figure is checked per file, in every file of
the record that carries it, and `release_records()` is the accessor that
makes that possible.
"""

from __future__ import annotations

import itertools
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The files that together are the 0.2.0 release record. Adding one widens
#: every presence check that reads through `release_prose`; it never
#: narrows one. It does not widen a FIGURE check — see `release_records`.
RELEASE_FILES: tuple[str, ...] = ("CHANGELOG.md", "SOUNDNESS.md")

#: THE VERSION-FIELD VOCABULARY, written once and read by both checks that
#: use it: `tests/test_soundness_routing.py` over `CHANGELOG.md`'s
#: one-liners, and `tests/test_soundness_log_reach.py` over `SOUNDNESS.md`'s
#: `## Log` bullets, which carry the same sentence in italics. Two closed
#: sets that agreed by hand would be two things to keep in step; this is
#: one thing.
#:
#: THREE PHRASES AND NOT TWO. The Log has to be able to say that a defect
#: reached the 0.1.0 pre-release builds and no release — 33 of its 54
#: bullets are that, and it was 39 until six bullets dated after the
#: `v0.1.0` tag were re-scoped on 2026-08-24 — and neither of the two the
#: changelog uses can say it.
#: `test_soundness_log_reach.py` derives the reached-release count from
#: these fields, and a count derived from a vocabulary that cannot express
#: a third of its entries is a count over the entries that happened to fit.
#:
#: NINE BULLETS HAVE NOW BEEN RE-SCOPED AND THE 33 ABOVE ONLY MOVED FOR
#: SIX OF THEM, which is the arithmetic and not a rounding: 39 − 6 = 33,
#: and the other three came out of a DIFFERENT phrase. On 2026-08-25
#: S12&Prime; and M17 left `Versions: 0.2.0 development builds only` for
#: the `v0.1.0` phrase, because the released tag carries both defects —
#: taking the second phrase from 12 bullets to 10 and the third from 9 to
#: 11 — and later that day the 2026-08-18 query-identity entry left it for
#: the same reason, taking them to 9 and 12. The first has not moved
#: since. The rounds failed for
#: opposite reasons, too: the six were post-tag dates claiming a pre-tag
#: event, which a date-versus-field comparison catches on its own; these
#: three are post-tag dates carrying a post-tag phrase, internally
#: consistent, and false only against an extracted `v0.1.0`. A closed set
#: holds the vocabulary and not the truth, and nothing in this module
#: knows which of the three a given entry OUGHT to carry — which is why
#: `tests/test_soundness_log_reach.py` now holds the CHANGELOG
#: DISCLOSURES that quote these phrases, and still cannot hold their
#: truth.
#: AND THE THREE ARE NOW GENERATED, WHICH IS THE REPAIR FOR THE LAST
#: SENTENCE ABOVE. *"A closed set holds the vocabulary and not the truth"*
#: was written as a limitation and it was also a diagnosis: three literal
#: sentences, with the reached-release count derived by STRING EQUALITY
#: against the third of them and every consumer reaching the phrases
#: POSITIONALLY (`VERSION_FIELDS[0]`, `VERSION_FIELDS[2]`, a three-tuple
#: unpack). A positional reference into a vocabulary is the same defect as
#: a positional reference into a numbered list, which is how the 0.2.1
#: plan's own limitation table came to be shifted by one.
#:
#: SO THERE IS A REGISTRY, A GENERATOR AND A PARSER, and the enumeration is
#: an OUTPUT rather than the definition. :data:`RELEASED` is the one place a
#: release registers; :data:`OPEN_LINE` is the development line open after
#: the newest of them; :func:`reach` reads a field back to the set of
#: releases it names. Today's three phrases are all GENERATED, byte for
#: byte — asserted in `tests/test_soundness_log_reach.py`, because a
#: generator that quietly reworded 65 existing fields would be a
#: documentation migration wearing a refactor's clothes — so no bullet in
#: `SOUNDNESS.md` moves.
#:
#: THE ACCEPTANCE TEST IS MEMBERSHIP AND DELIBERATELY NOT POSITION. An
#: earlier draft of this paragraph said *"the first three generated phrases
#: are today's three"*, and it was FALSE the moment `0.2.1` was registered:
#: the generator emits the release-free forms together, so the third slot
#: is `0.2.1 development builds only.` A vocabulary whose correctness is
#: stated as an ORDER has the defect this whole change removes, one
#: paragraph above the code that removes it.
#:
#: WHAT IT ADDS, AND THE SECOND ONE IS THE POINT. It adds the phrases for a
#: post-0.2.0 event, which is arithmetic. And it adds a form for an event
#: that is **not over**: every one of the three phrases ends the entry's
#: event (*"was over before the tag"*, *"was over before the release"*), so
#: there was no way to record that a defect is present in a published
#: release and UNREPAIRED. That is not a gap in what can be spelled, it is a
#: gap in what can be RECORDED, and the FTZ class is exactly the entry that
#: could not be written: present in `v0.1.0` and `v0.2.0`, both yanked,
#: unrepaired, release train frozen. A ledger whose grammar can only
#: describe closed events quietly requires every entry to be a fix.

#: THE RELEASES THIS RECORD CAN NAME, oldest first. The one place a new
#: release registers; everything below is derived from it.
RELEASED: tuple[str, ...] = ("0.1.0", "0.2.0")

#: The development line open after the newest member of :data:`RELEASED`.
OPEN_LINE: str = "0.2.1"


def _dev_lines() -> tuple[str, ...]:
    """The development line that CLOSED each release after the first, plus
    the one open now — indexed to match :data:`RELEASED`, so
    ``_dev_lines()[i]`` is the line that follows ``RELEASED[i]``."""
    return tuple(RELEASED[1:]) + (OPEN_LINE,)


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _generate_version_fields() -> dict[str, tuple[frozenset, str | None]]:
    """``{phrase: (releases reached, the line that closed the event)}``.

    ``closed`` is ``None`` for the unrepaired form. Subsets rather than
    prefixes: a defect introduced in 0.2.0 development and released reaches
    `v0.2.0` and not `v0.1.0`, and a prefix-only grammar could not say so.
    """
    out: dict[str, tuple[frozenset, str | None]] = {}
    out[f"Versions: {RELEASED[0]} pre-release builds only."] = (frozenset(), RELEASED[0])
    for line in _dev_lines():
        out[f"Versions: {line} development builds only."] = (frozenset(), line)
    for k in range(1, len(RELEASED) + 1):
        for subset in itertools.combinations(range(len(RELEASED)), k):
            tags = [RELEASED[i] for i in subset]
            closed = _dev_lines()[subset[-1]]
            ticked = [f"`v{v}`" for v in tags]
            body = _join(ticked + [f"{closed} development builds"])
            out[f"Versions: {body}."] = (frozenset(tags), closed)
            out[f"Versions: {_join(ticked)}, unrepaired."] = (frozenset(tags), None)
    return out


#: ``{phrase: (reach, closed_in)}`` — the generated vocabulary.
VERSION_FIELD_MAP: dict[str, tuple[frozenset, "str | None"]] = _generate_version_fields()

#: The vocabulary as a tuple, today's three first and byte-identical.
VERSION_FIELDS: tuple[str, ...] = tuple(VERSION_FIELD_MAP)

_TAG_RE = re.compile(r"`v(\d[0-9A-Za-z._+!-]*)`")
_FIELD_RE = re.compile(r"Versions: (?P<body>.+?)\.\s*$")
_NO_RELEASE_RE = re.compile(
    r"[0-9][0-9A-Za-z._+!-]* (?:pre-release|development) builds only"
)


def reach(text: str) -> "frozenset | None":
    """The RELEASED versions a `Versions:` field names, or ``None`` if the
    text is not a well-formed field.

    A PARSE and not a lookup, so that the question *"does this entry reach
    `v0.1.0`?"* is answered from the sentence rather than from its position
    in a list. ``frozenset()`` — reaches no release — and ``None`` — not a
    field at all — are different answers and callers must not conflate them.
    """
    m = _FIELD_RE.search(text.strip())
    if m is None:
        return None
    body = m.group("body")
    tags = frozenset(_TAG_RE.findall(body))
    if not tags:
        return frozenset() if _NO_RELEASE_RE.fullmatch(body) else None
    return tags if tags <= set(RELEASED) else None


def changelog_version_fields(line: str) -> tuple[str, ...]:
    """The subset a `CHANGELOG.md` one-liner for release ``line`` may carry.

    DERIVED from the closing line rather than sliced off the front, which is
    what `VERSION_FIELDS[1:]` used to be. An entry in a release's changelog
    records a fix that landed in that release, so the event was over during
    that release's development line — which excludes the pre-release phrase
    (a permitted phrase no entry can use is an untested branch of a rule,
    and this campaign has withdrawn several), every phrase belonging to a
    later line, and the unrepaired form, whose whole content is that the
    event is not over.
    """
    return tuple(p for p, (_, closed) in VERSION_FIELD_MAP.items() if closed == line)


#: The subset a `CHANGELOG.md` 0.2.0 one-liner may carry.
CHANGELOG_VERSION_FIELDS: tuple[str, ...] = changelog_version_fields("0.2.0")


def field_for(reaches: "tuple[str, ...]" = (), closed: "str | None" = None) -> str:
    """The one generated phrase with this reach and this closing line.

    THE ACCESSOR THAT REPLACES `VERSION_FIELDS[n]`. A consumer that wants
    *"the phrase for an event that reached `v0.1.0` and was over during
    0.2.0 development"* now asks for it by what it MEANS. Indexing asked for
    it by where it happened to sit, which survived exactly as long as the
    vocabulary had three members.

    Asserts a unique hit rather than returning the first, because two
    phrases with one meaning would make every count that reads through here
    a count over whichever the generator emitted first.
    """
    want = (frozenset(reaches), closed)
    hits = [p for p, v in VERSION_FIELD_MAP.items() if v == want]
    assert len(hits) == 1, (
        f"the version-field vocabulary has {len(hits)} phrases for "
        f"reach={sorted(want[0])} closed={closed!r}, and a consumer asking "
        f"for one by meaning needs exactly one: {hits}"
    )
    return hits[0]


def release_records() -> tuple[tuple[str, str], ...]:
    """`((name, text), …)` — the release record, file by file.

    For a check that reads a FIGURE out of the record. A figure published
    anywhere in the record has to be right everywhere it is published, and
    that is a per-file question: over the concatenation one correct copy
    hides every stale copy behind it.
    """
    out = []
    for name in RELEASE_FILES:
        path = REPO / name
        assert path.is_file(), (
            f"{name} is not in this tree. The release record is "
            f"{list(RELEASE_FILES)} and a missing member would make every "
            f"presence check that reads through here quietly weaker."
        )
        out.append((name, path.read_text(encoding="utf-8")))
    return tuple(out)


def release_prose() -> str:
    """`CHANGELOG.md` and `SOUNDNESS.md`, concatenated, as text.

    Joined with a blank line so no claim can be formed by one file's last
    line running into the next file's first.

    For PRESENCE and for ABSENCE. NOT for a figure read with `re.search`,
    which sees the first match only — `release_records()` is that.
    """
    return "\n\n".join(text for _name, text in release_records())
