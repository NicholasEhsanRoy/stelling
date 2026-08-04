# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The release documents' PROSE, re-derived — not just the blocks under it.

``tests/test_doc_examples.py`` mechanised the measurement: the registry
census in ``docs/state-0.1.0.md`` is executed and its fence compared byte
for byte. That left the step AFTER the measurement — measurement to
prose — exactly where it was, and that step is where all three of this
project's recorded unit failures happened
(``docs/proposed-unit-mechanism.md``). The demonstration was cheap: the
clause's headline could be falsified to *"Four of them ... `pow`"* and
its unit line to *"n = 3 of the 40"* with the whole suite green, three
lines above a block that executes and prints ``neither 1 ['iota']``.

**So this file re-derives the sentence, not only the block.** Clause 3's
headline count and named member, its unit line's two populations, and
the census's own ``recorded`` list are all computed here from
``stelling.propagate.TRANSFERS`` and ``stelling.obligation._SUPPORTED``
and compared against what the page says. A number in that clause that
stops matching the registries goes red.

**It imports no jax, and that is the point, not an accident.**
``test_doc_examples`` skips every example when jax is absent
(``pytest.importorskip("jax")``), so the one block whose entire purpose
is to gate a ZERO-DEP-CORE registry count was the one block skipped in
the zero-dep configuration — a configuration this project measures and
reports (``SOUNDNESS.md``: *"674 zero-dep"*). Both registries import in a
bare environment, so this gate runs everywhere the census's claim is
about.

**EXACTLY WHAT IS AND IS NOT CHECKED HERE.**

* the census block's printed output against its fence — again, and
  without jax;
* that the block *responds to the registries*: with ``iota`` added to
  ``TRANSFERS`` and ``square`` removed from ``_SUPPORTED`` its output has
  to move, which a hard-coded truth table printing the right answer
  would not;
* that the block is still an unmarked (compared) example, so the page's
  claim that ``test_doc_examples`` compares it stays true;
* clause 3's headline sentence and unit line, against the same
  computation;
* the census population against clause 1's list of external terminals,
  under the identification clause 2 states — so dropping a member from
  ``recorded`` and editing the fence to match no longer passes;
* every file the release page names is either in this tree or written
  with the ``stelling-sweeps/`` prefix that says it is not;
* the quotations REGISTERED BELOW are verbatim in the file they are
  attributed to, and verbatim in the file that quotes them.

NOT checked: anything over the MADDENING/MIME corpus, which is not here;
and quotations that nobody registered — ``QUOTATIONS`` is a list a human
maintains, it does not discover new quotations, and a claim it does not
carry is unchecked by it.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import textwrap

from stelling import obligation, propagate

REPO = pathlib.Path(__file__).resolve().parents[1]
STATE = REPO / "docs" / "state-0.1.0.md"
SOUNDNESS = REPO / "SOUNDNESS.md"
CONTRIBUTING = REPO / "CONTRIBUTING.md"

# ------------------------------------------------------------------ reading

_FENCE = re.compile(r"^(\s*)```(\w*)\s*$")
_MARKER = re.compile(r"<!--\s*doc-example:\s*(illustrative|run-only)\s*-->")


def _blocks(text: str) -> list[tuple[str, str, int, str | None]]:
    """(lang, dedented body, 1-based fence line, marker or None) per fence.

    Written here rather than imported from ``test_doc_examples`` on
    purpose: a gate that shares its reader with the thing it is gating
    fails in the same direction as it.
    """
    lines = text.split("\n")
    out: list[tuple[str, str, int, str | None]] = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < len(lines) and not _FENCE.match(lines[j]):
            body.append(lines[j])
            j += 1
        marker = None
        for ln in lines[max(0, i - 2):i]:
            found = _MARKER.search(ln)
            if found:
                marker = found.group(1)
        out.append((m.group(2), textwrap.dedent("\n".join(body)), i + 1, marker))
        i = j + 1
    return out


def _flat(text: str) -> str:
    """One long line. The page wraps its sentences; its claims do not."""
    return re.sub(r"\s+", " ", text)


CENSUS_SIGNATURE = 'buckets = {"neither"'


def _census() -> tuple[tuple[str, str, int, str | None], tuple[str, str, int, str | None]]:
    """The registry-census block and the fence it claims as its output."""
    blocks = _blocks(STATE.read_text(encoding="utf-8"))
    hits = [
        (n, b) for n, b in enumerate(blocks)
        if b[0] == "python" and CENSUS_SIGNATURE in b[1]
    ]
    assert len(hits) == 1, (
        f"expected exactly one registry census in {STATE.name} — a ```python "
        f"block containing {CENSUS_SIGNATURE!r} — and found {len(hits)}. "
        "If the census moved or was rewritten, move this gate with it; if it "
        "was deleted, clause 3 is hand-written again and says it is not."
    )
    n, block = hits[0]
    assert n + 1 < len(blocks) and blocks[n + 1][0] == "", (
        f"{STATE.name}:{block[2]} — the census block has no plain ``` fence "
        "after it, so there is no recorded output to compare it against"
    )
    return block, blocks[n + 1]


def _run(source: str) -> tuple[str, dict]:
    """Execute a doc block in-process and return (stdout, its namespace)."""
    ns: dict = {"__name__": "__census__"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(source, "<state-0.1.0.md census>", "exec"), ns)
    return buf.getvalue(), ns


_CENSUS_LINE = re.compile(r"^(?P<bucket>\S+(?: \S+)*?)\s+(?P<n>\d+)\s+\[(?P<members>.*)\]$")


def _buckets(out: str) -> dict[str, list[str]]:
    got: dict[str, list[str]] = {}
    for line in out.rstrip("\n").split("\n"):
        m = _CENSUS_LINE.match(line)
        assert m, f"census output line is not in the shape this gate reads: {line!r}"
        members = re.findall(r"'([^']*)'", m.group("members"))
        assert int(m.group("n")) == len(members), (
            f"the census printed a count that is not its own member list: {line!r}"
        )
        got[m.group("bucket")] = members
    assert got, "the census printed nothing"
    return got


# ------------------------------------------------- the census is a measurement


def test_the_census_block_prints_the_fence_beneath_it():
    """Same comparison ``test_doc_examples`` makes, minus the jax skip.

    This is the one that has to hold in the zero-dep configuration,
    because a zero-dep-core registry count is what it is counting."""
    block, fence = _census()
    out, _ = _run(block[1])
    assert out.rstrip("\n").split("\n") == fence[1].rstrip("\n").split("\n"), (
        f"{STATE.name}:{block[2]} — the census printed something other than "
        f"the fence recorded under it.\n--- fence ---\n{fence[1]}\n"
        f"--- printed ---\n{out}"
    )


def test_the_census_block_reads_the_registries_it_names(monkeypatch):
    """Positive control on the instrument, not on the number.

    A block that printed ``neither 1 ['iota']`` from a hard-coded table
    would satisfy every comparison above. This one has to MOVE when the
    registries move, in the direction the registries moved — once for
    each of the two registries it claims to read."""
    block, _ = _census()
    before = _buckets(_run(block[1])[0])
    assert "iota" in before["neither"], (
        "this control assumes iota starts in the `neither` bucket; the "
        "registries have moved and the control needs rewriting, not deleting"
    )
    assert "square" in before["both"], (
        "this control assumes square starts in the `both` bucket; the "
        "registries have moved and the control needs rewriting, not deleting"
    )

    monkeypatch.setattr(propagate, "TRANSFERS", {**propagate.TRANSFERS, "iota": None})
    after = _buckets(_run(block[1])[0])
    assert "iota" not in after["neither"] and "iota" in after["transfer only"], (
        "the census did not notice `iota` gaining an interval transfer — it "
        "is not reading stelling.propagate.TRANSFERS, so its numbers are a "
        "record of what someone typed and not of what is registered"
    )

    monkeypatch.setattr(
        obligation, "_SUPPORTED",
        frozenset(obligation._SUPPORTED) - {"square"},
    )
    after = _buckets(_run(block[1])[0])
    assert "square" not in after["both"] and "square" in after["transfer only"], (
        "the census did not notice `square` losing its emission row — it is "
        "not reading stelling.obligation._SUPPORTED"
    )


def test_the_census_block_is_still_a_compared_example():
    """Clause 3 says ``test_doc_examples`` compares this block byte for
    byte. Marking it ``illustrative`` or ``run-only`` takes it out of
    that module's compared set and leaves the sentence asserting it."""
    block, _ = _census()
    assert block[3] is None, (
        f"{STATE.name}:{block[2]} — the census block is marked "
        f"'{block[3]}', so tests/test_doc_examples.py no longer compares "
        "it, while clause 3 still says it does"
    )


# ------------------------------------------------------- the prose is the same
#                                                          measurement

NUMBER_WORDS = {
    0: "None", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
}

_HEADLINE = re.compile(
    r"\*\*(?P<count>[A-Za-z]+) of them ha(?:s|ve) neither an interval "
    r"transfer nor an emission row: (?P<members>[^*]+?)\.\*\*"
)
_UNIT_LINE = re.compile(
    r"n = (?P<primitives>\d+) primitives of the (?P<listed>\d+) recorded items"
)
_TEN = re.compile(
    r"External code was measured to terminate on (?P<items>[^.]+)\. "
)
_LISTED_UNIT = re.compile(r"n = (?P<listed>\d+) listed items")

# The two translations clause 1's list needs before it is a list of
# registry keys. Both are the PAGE's, stated in clause 2 and in clause
# 3's unit line; ``test_the_translations_are_the_pages_own`` is what
# keeps them from quietly becoming this file's.
IDENTIFICATIONS = {"int64→float64": "convert_element_type"}
NOT_A_PRIMITIVE = "nan_to_num"


def _clause_1_items() -> list[str]:
    flat = _flat(STATE.read_text(encoding="utf-8"))
    m = _TEN.search(flat)
    assert m is not None, (
        "clause 1's list of external terminals is gone or reworded past this "
        "gate. Two later clauses count over it; re-point this pattern rather "
        "than dropping the check."
    )
    items = [x.strip() for x in m.group("items").split(",")]
    assert all(items), f"clause 1's list has an empty item: {items!r}"
    return items


def _census_population_from_clause_1() -> list[str]:
    out = []
    for item in _clause_1_items():
        if NOT_A_PRIMITIVE in item:
            continue  # no registry key: clause 3's unit line says why
        names = re.findall(r"`([^`]+)`", item)
        assert len(names) == 1, (
            f"clause 1 item {item!r} does not name exactly one term, so this "
            "gate cannot map it to a registry key"
        )
        out.append(IDENTIFICATIONS.get(names[0], names[0]))
    return out


def test_the_translations_are_the_pages_own():
    """This gate rewrites two of clause 1's ten before looking them up.
    Both rewrites have to be things the page itself asserts, or the gate
    is checking the page against the gate's opinions."""
    flat = _flat(STATE.read_text(encoding="utf-8"))
    for spelled, primitive in IDENTIFICATIONS.items():
        assert f"`{spelled}` **is** `{primitive}`" in flat, (
            f"this gate maps `{spelled}` to `{primitive}`, and the page no "
            "longer states that identification"
        )
    assert f"`{NOT_A_PRIMITIVE}`'s `inf` is not a primitive" in flat, (
        f"this gate drops `{NOT_A_PRIMITIVE}`'s `inf` from the census "
        "population, and the page no longer states why"
    )


def test_clause_3_headline_sentence_is_the_registries():
    """THE TRANSCRIPTION BOUNDARY. The sentence a reader actually reads,
    against the registries the block beneath it reads."""
    block, _ = _census()
    neither = _buckets(_run(block[1])[0])["neither"]
    m = _HEADLINE.search(_flat(STATE.read_text(encoding="utf-8")))
    assert m is not None, (
        "clause 3's headline sentence is gone or reworded past this gate. It "
        "is the sentence the census exists to hold; rewrite this pattern with "
        "it rather than removing the check."
    )
    expected = NUMBER_WORDS.get(len(neither))
    assert expected is not None, (
        f"{len(neither)} primitives are now in the `neither` bucket and this "
        "gate has no word for that count"
    )
    assert m.group("count") == expected, (
        f"clause 3 says '{m.group('count')} of them' have neither a transfer "
        f"nor an emission row; the registries say {expected} "
        f"({len(neither)}): {neither}"
    )
    named = re.findall(r"`([^`]+)`", m.group("members"))
    assert named == neither, (
        f"clause 3 names {named} as having neither; the registries say "
        f"{neither}"
    )


def test_clause_3_unit_line_is_the_registries():
    """The other half of the same sentence: both of its populations."""
    block, _ = _census()
    recorded = list(_run(block[1])[1]["recorded"])
    listed = _clause_1_items()
    m = _UNIT_LINE.search(_flat(STATE.read_text(encoding="utf-8")))
    assert m is not None, (
        "clause 3's unit line is gone or reworded past this gate; it carries "
        "the two populations every count in the clause is over"
    )
    assert int(m.group("primitives")) == len(recorded), (
        f"clause 3's unit line says n = {m.group('primitives')} primitives; "
        f"the census block censuses {len(recorded)}: {recorded}"
    )
    assert int(m.group("listed")) == len(listed), (
        f"clause 3's unit line says {m.group('listed')} recorded items; "
        f"clause 1 lists {len(listed)}: {listed}"
    )


def test_clause_2_counts_clause_1s_population():
    """Clause 2's unit line quantifies over the same list."""
    listed = _clause_1_items()
    hits = _LISTED_UNIT.findall(_flat(STATE.read_text(encoding="utf-8")))
    assert hits, "clause 2's 'n = N listed items' unit line is gone"
    for got in hits:
        assert int(got) == len(listed), (
            f"a unit line says n = {got} listed items; clause 1 lists "
            f"{len(listed)}: {listed}"
        )


def test_the_census_population_is_clause_1s_list():
    """The census's own POPULATION, which nothing checked before.

    Dropping a member from ``recorded`` and editing the fence to match
    left every comparison green and clause 1 and clause 3 both stating
    the old population."""
    block, _ = _census()
    recorded = list(_run(block[1])[1]["recorded"])
    expected = _census_population_from_clause_1()
    assert recorded == expected, (
        "the census block's `recorded` list is not clause 1's list of "
        "external terminals under the identifications clause 2 states.\n"
        f"  block     {recorded}\n  clause 1  {expected}"
    )


# --------------------------------------------------- placement and quotation

_FILE_REF = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|md))`")
_LINK_REF = re.compile(r"\]\(([A-Za-z0-9_][A-Za-z0-9_./-]*\.md)\)")
EXTERNAL_PREFIX = "stelling-sweeps/"


def test_every_file_the_release_page_names_is_placed():
    """An instrument named without a location reads as in-tree.

    Five were: three named as *"re-measured in this tree"* while living
    in the campaign repo. The rule this page states — a population that
    is not in the tree gets a sha, one that is gets a gate — cannot be
    applied by a reader who cannot tell which case a name is. So a name
    is either resolvable here or carries the prefix that says it is not.

    *Scope:* bare ``‘name.ext’`` references and markdown links, in
    ``docs/state-0.1.0.md``. A reference written with a line number
    (``gnn.py:312``) is outside it."""
    text = STATE.read_text(encoding="utf-8")
    refs = sorted(set(_FILE_REF.findall(text)) | set(_LINK_REF.findall(text)))
    # both branches of the check must be reachable, or it cannot fail
    assert any(r.startswith(EXTERNAL_PREFIX) for r in refs), (
        "the scanner found no external reference at all; it has stopped "
        "reaching the case it exists for"
    )
    assert any(not r.startswith(EXTERNAL_PREFIX) for r in refs), (
        "the scanner found no in-tree reference at all"
    )
    unplaced = [
        r for r in refs
        if not r.startswith(EXTERNAL_PREFIX)
        and not (REPO / r).exists()
        and not (STATE.parent / r).exists()
    ]
    assert not unplaced, (
        f"{STATE.name} names files that are not in this tree and are not "
        f"marked as living elsewhere: {unplaced}\nEither commit them, or "
        f"write them as {EXTERNAL_PREFIX}<name> with the sha they were read "
        "at."
    )


# (quoting file, file it is attributed to, the quoted text). Whitespace
# is normalised on both sides; nothing else is.
QUOTATIONS = [
    (STATE, CONTRIBUTING,
     "An over-permissive stub's ZERO is conclusive; its NONZERO is not"),
    (STATE, CONTRIBUTING,
     "A stub that grants **more** than a real implementation could deliver "
     "**upper-bounds** the benefit."),
    (STATE, CONTRIBUTING,
     "state what the stub grants, before reporting what it produced"),
    (SOUNDNESS, CONTRIBUTING,
     "A figure in a norm states the UNIT it counts"),
    (STATE, STATE,
     "count over a population that is not in the tree gets a sha; a count "
     "that is computable from the tree gets a gate"),
]


def test_registered_quotations_are_verbatim():
    """An elision inside quotation marks can widen a norm's scope.

    One did: *"a figure states the UNIT it counts"* for a norm that reads
    *"A figure IN A NORM states the UNIT it counts"*, applied to four
    counts that are not in a norm.

    *Scope, and it is a real limit:* this checks the quotations listed in
    ``QUOTATIONS`` and finds no others. A quotation nobody registered is
    a hand-check, exactly like an unattached fence."""
    sources = {p: _flat(p.read_text(encoding="utf-8")) for _, p, _ in QUOTATIONS}
    quoters = {p: _flat(p.read_text(encoding="utf-8")) for p, _, _ in QUOTATIONS}
    for quoter, source, quoted in QUOTATIONS:
        flat = _flat(quoted)
        assert flat in sources[source], (
            f"{quoter.name} quotes {source.name} as saying {quoted!r}, and "
            f"{source.name} does not say that"
        )
        assert flat in quoters[quoter], (
            f"{quoter.name} no longer contains the quotation {quoted!r} that "
            "this registry pins for it — the registry is now describing a "
            "document that has moved on"
        )
