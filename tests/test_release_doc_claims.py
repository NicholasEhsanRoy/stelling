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
* every file **any page under ``docs/``** names is either in this tree,
  or written with the ``stelling-sweeps/`` prefix that says it is not, or
  listed in :data:`NOT_IN_THIS_TREE` as a decision — the scope was one
  page, which is why four campaign scripts sat uncited in
  ``gauge-coverage.md`` with this file green;
* the quotations REGISTERED BELOW are verbatim in the file they are
  attributed to, and verbatim in the file that quotes them.

NOT checked: anything over the MADDENING/MIME corpus, which is not here;
and quotations that nobody registered — ``QUOTATIONS`` is a list a human
maintains, it does not discover new quotations, and a claim it does not
carry is unchecked by it.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import pathlib
import re
import subprocess
import sys
import textwrap

from stelling import obligation, propagate

REPO = pathlib.Path(__file__).resolve().parents[1]
STATE = REPO / "docs" / "state-0.1.0.md"
SOUNDNESS = REPO / "SOUNDNESS.md"
CONTRIBUTING = REPO / "CONTRIBUTING.md"
NORMS = REPO / "docs" / "norms.md"

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


# Directories a bare basename may legitimately be short for. A page writing
# `obligation.py` means `src/stelling/obligation.py`, and demanding the full
# path everywhere would make the docs worse to satisfy a scanner.
_SHORTHAND_ROOTS = (
    "", "src/", "src/stelling/", "src/stelling/_tripwire/",
    "tests/", "docs/", "design/", "corpus/", "tools/",
)

# References that resolve nowhere in this tree and are NOT campaign
# instruments. Each is a decision, made here in the open, and each says which
# of the two legitimate cases it is. A name that is neither is a defect: it
# reads to a reader as a file they can open, and it is not one.
NOT_IN_THIS_TREE = {
    # (1) files the reader is instructed to CREATE. The page is telling them
    # what to name it, so "not in this tree" is the point of the sentence.
    "docs/quickstart.md": {
        "quickstart.py", "statuses.py", "witness.py", "nonvacuity_demo.py",
    },
    "docs/overflow-tripwire.md": {
        # (2) another package's source, cited as the site of a jax behaviour.
        # It is in the installed jax, not here, and rewriting it with the
        # campaign prefix would be a lie about where it lives.
        "jax/_src/random/prng.py",
    },
}


def test_every_file_the_docs_name_is_placed():
    """An instrument named without a location reads as in-tree.

    Five were, on ``docs/state-0.1.0.md``: three named as *"re-measured in
    this tree"* while living in the campaign repo. The rule that page states
    — a population that is not in the tree gets a sha, one that is gets a
    gate — cannot be applied by a reader who cannot tell which case a name
    is. So a name is either resolvable here, or carries the prefix that says
    it is not, or is listed above as a decision.

    **THE SCOPE WAS ONE PAGE, AND THAT IS WHY THIS KEPT HAPPENING.** With
    ``docs/state-0.1.0.md`` gated and nothing else, ``gauge-coverage.md``
    rested a whole table column on four campaign scripts a reader had no way
    to learn were not here, ``proposed-unit-mechanism.md`` cited two more,
    and ``docs/verdict-ledger.md``'s central entry pointed at a third — all
    with this test green, because it was not looking. Widened to every page
    under ``docs/``.

    *Scope:* bare ``name.ext`` references in backticks, and markdown links.
    A reference written with a line number (``gnn.py:312``) is outside it,
    and so is a symbol reference with no extension (``frontier.warm()``) —
    the second is a real hole and is stated here rather than implied."""
    seen_external = False
    seen_intree = False
    unplaced: dict[str, list[str]] = {}
    pages = sorted((REPO / "docs").glob("*.md"))
    assert len(pages) >= 20, f"the docs glob found only {len(pages)} pages"
    for page in pages:
        text = page.read_text(encoding="utf-8")
        refs = sorted(
            set(_FILE_REF.findall(text)) | set(_LINK_REF.findall(text))
        )
        allowed = NOT_IN_THIS_TREE.get(f"docs/{page.name}", set())
        for ref in refs:
            if ref.startswith(EXTERNAL_PREFIX):
                seen_external = True
                continue
            if ref in allowed:
                continue
            if (REPO / ref).exists() or (page.parent / ref).exists() or any(
                (REPO / (root + ref)).exists() for root in _SHORTHAND_ROOTS
            ):
                seen_intree = True
                continue
            unplaced.setdefault(page.name, []).append(ref)
    # both branches must be reachable, or the check cannot fail
    assert seen_external, (
        "the scanner found no external reference at all; it has stopped "
        "reaching the case it exists for"
    )
    assert seen_intree, "the scanner found no in-tree reference at all"
    assert not unplaced, (
        f"docs/ names files that are not in this tree and are not marked as "
        f"living elsewhere: {unplaced}\nEither commit them, write them as "
        f"{EXTERNAL_PREFIX}<name> with the sha they were read at, or add "
        f"them to NOT_IN_THIS_TREE with which case they are."
    )


def test_the_placement_exemptions_are_all_still_needed():
    """An exemption for a reference a page no longer makes is a licence
    nobody is using, and it would silently cover the next one."""
    stale: dict[str, list[str]] = {}
    for page_path, refs in NOT_IN_THIS_TREE.items():
        text = (REPO / page_path).read_text(encoding="utf-8")
        found = set(_FILE_REF.findall(text)) | set(_LINK_REF.findall(text))
        gone = sorted(r for r in refs if r not in found)
        if gone:
            stale[page_path] = gone
    assert not stale, (
        f"NOT_IN_THIS_TREE exempts references these pages no longer make: "
        f"{stale}. Drop them."
    )


# (quoting file, file it is attributed to, the quoted text). Whitespace
# is normalised on both sides; nothing else is.
# The norms moved out of CONTRIBUTING.md into docs/norms.md. All four of these
# point at NORMS rather than CONTRIBUTING, including the two whose TITLES still
# appear in CONTRIBUTING's index: an index entry would satisfy the check while
# the norm's own text drifted underneath it, which is the drift this list exists
# to catch.
QUOTATIONS = [
    (STATE, NORMS,
     "An over-permissive stub's ZERO is conclusive; its NONZERO is not"),
    (STATE, NORMS,
     "A stub that grants **more** than a real implementation could deliver "
     "**upper-bounds** the benefit."),
    (STATE, NORMS,
     "state what the stub grants, before reporting what it produced"),
    (SOUNDNESS, NORMS,
     "A figure in a norm states the UNIT it counts"),
    (STATE, STATE,
     "count over a population that is not in the tree gets a sha; a count "
     "that is computable from the tree gets a gate"),
]


# ------------------------------------------------------- the zero-dep claim

_ZERO_DEP_PROBE = '''\
import importlib.util
import sys


class _NoJax:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("jax", "jaxlib"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _NoJax())

spec = importlib.util.spec_from_file_location("_gate_under_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in sys.argv[2:]:
    getattr(mod, name)()
assert "jax" not in sys.modules, "this gate pulled jax in after all"
print("ok")
'''


def test_this_gate_holds_with_jax_absent():
    """The property that makes this file worth having, pinned instead of
    asserted in its docstring.

    The census counts a ZERO-DEP-CORE registry. A gate for it that only
    runs where jax is installed leaves the count unchecked in exactly the
    configuration it describes, which is the state this file was written
    to end. So every fixture-free check above is re-run in a subprocess
    with ``jax`` and ``jaxlib`` made unimportable — the list is computed,
    not typed, so a check added later cannot quietly fall out of it."""
    names = [
        n for n, obj in sorted(globals().items())
        if n.startswith("test_")
        and n != "test_this_gate_holds_with_jax_absent"
        and callable(obj)
        and not inspect.signature(obj).parameters
    ]
    assert len(names) >= 8, (
        f"the zero-dep probe is only covering {names} — if the checks above "
        "have grown fixtures, give it an explicit list rather than letting it "
        "shrink to nothing"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    r = subprocess.run(
        [sys.executable, "-c", _ZERO_DEP_PROBE, str(pathlib.Path(__file__).resolve()), *names],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (
        "the release-doc gates do not hold with jax absent, which is the "
        f"configuration the registry census is about.\n--- stderr ---\n"
        f"{r.stderr[-3000:]}"
    )


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


# ------------------------------------------------- the norm count, three files

# The spelled forms this project uses. A count outside this range needs an
# entry here, which is the point: adding one is a decision, and a number this
# table cannot spell is a number nobody can check.
_SPELLED = {
    20: "twenty", 21: "twenty-one", 22: "twenty-two", 23: "twenty-three",
    24: "twenty-four", 25: "twenty-five", 26: "twenty-six",
    27: "twenty-seven", 28: "twenty-eight", 29: "twenty-nine", 30: "thirty",
    31: "thirty-one", 32: "thirty-two", 33: "thirty-three",
    34: "thirty-four", 35: "thirty-five",
}
_NORM_COUNT_CLAIM = re.compile(
    r"\b(?P<word>[Tt]wenty(?:-\w+)?|[Tt]hirty(?:-\w+)?)\s+rules\b"
)
DOCS_README = REPO / "docs" / "README.md"


def _norm_headings() -> list[str]:
    """Every ``## `` heading in docs/norms.md. The file has no non-rule ones,
    which the test below asserts rather than assumes."""
    return [
        ln[3:].strip()
        for ln in NORMS.read_text(encoding="utf-8").split("\n")
        if ln.startswith("## ")
    ]


def test_the_norm_count_is_the_same_number_in_all_three_files():
    """A hand-maintained numeral beside data that derives it, in the page
    that legislates against exactly that.

    Measured when this test was written: ``docs/norms.md`` said
    *"Twenty-six rules"* and held 26; ``docs/README.md``'s index row — the
    first number a contributor reads — said *"twenty-three rules"*, three
    behind, and had been since ``d7bd6ae``; ``CONTRIBUTING.md`` carried 26
    links, one per norm, and claimed in as many words that its index exists
    "so this file and that one cannot drift apart", with no test enforcing
    it. Two of the three were right and nothing compared them.

    So: the count is computed from the headings, and all three files are
    held to it — the two spelled numerals, and the length of
    ``CONTRIBUTING.md``'s index.
    """
    headings = _norm_headings()
    n = len(headings)
    assert n >= 20, f"only {n} `## ` headings in {NORMS.name}"
    spelled = _SPELLED.get(n)
    assert spelled is not None, (
        f"{NORMS.name} holds {n} rules, which _SPELLED cannot spell. Add it."
    )

    for path in (NORMS, DOCS_README):
        text = path.read_text(encoding="utf-8")
        found = _NORM_COUNT_CLAIM.findall(text)
        assert found, (
            f"{path.name} no longer states a spelled rule count; it is the "
            f"claim this test holds to {NORMS.name}'s headings"
        )
        wrong = [w for w in found if w.lower() != spelled]
        assert not wrong, (
            f"{path.name} says {wrong} rules where {NORMS.name} has {n} "
            f"({spelled}). Do not retype it in a third place — this test is "
            f"here so the digit is derived."
        )

    # CONTRIBUTING.md's index: one link per norm, and it says so itself
    links = re.findall(
        r"^- \[(.+?)\]\(docs/norms\.md#", CONTRIBUTING.read_text("utf-8"), re.M
    )
    assert len(links) == n, (
        f"CONTRIBUTING.md indexes {len(links)} norms and {NORMS.name} has "
        f"{n}. That file claims its index exists 'so this file and that one "
        f"cannot drift apart'."
    )
    assert links == headings, (
        "CONTRIBUTING.md's index titles are not docs/norms.md's headings, in "
        f"order:\n  only in the index: {sorted(set(links) - set(headings))}\n"
        f"  only in norms.md:  {sorted(set(headings) - set(links))}"
    )


_NORM_LETTER = re.compile(r"\bNorm [A-Z]\b")
# `src/stelling/propagate.py` carries one more ("Norm G"). It was owned by a
# sibling batch when this gate was written, which is why it went in as a
# NAMED debt rather than a silent exemption. That batch has since landed
# (B18, merged at `9cb2c0f`), so the reason for the exemption has expired and
# the entry is now a one-line cleanup owed by whoever next opens that file:
# replace it with "An instrument must declare its SCOPE, and an acceptance
# criterion must check that the scope covers the claim". Left here rather
# than taken, because this branch is a documentation batch and that file was
# out of its scope; `test_the_norm_letter_debt_is_still_real` will fail the
# day it is fixed, which is how the entry gets removed.
_LETTER_DEBT = {"src/stelling/propagate.py"}
# This file names the letters in order to forbid them, so it cannot scan
# itself. Excluded by path rather than by a marker, because a marker in a
# docstring is the shape this gate exists to refuse.
_LETTER_SELF = "tests/test_release_doc_claims.py"


def test_no_norm_is_cited_by_a_letter_it_does_not_have():
    """``docs/norms.md`` carries no letters, and never did in this form.

    Five references — three in the file itself, six more in ``tests/`` and
    ``src/`` — pointed at *Norm C*, *D*, *E*, *G*, *I*, *J*. The scheme was
    positional in the July file this one grew out of; the sections have since
    been reordered and grown from 13 to 26, so every letter resolves to
    nothing and a reader following one lands nowhere.

    A title moves loudly — ``grep`` comes back empty — which is why the
    convention everywhere else in this file is the title.
    """
    assert not _NORM_LETTER.search(NORMS.read_text(encoding="utf-8")), (
        f"{NORMS.name} has no lettered sections, no legend and no numbering, "
        f"so a 'Norm X' reference in it resolves to nothing. Use the norm's "
        f"title, which is this file's own convention everywhere else."
    )
    offenders: dict[str, list[str]] = {}
    for sub in ("docs", "tests", "src"):
        for path in sorted((REPO / sub).rglob("*.py")) + sorted(
            (REPO / sub).rglob("*.md")
        ):
            rel = str(path.relative_to(REPO))
            if rel in _LETTER_DEBT or rel == _LETTER_SELF:
                continue
            hits = _NORM_LETTER.findall(path.read_text(encoding="utf-8"))
            if hits:
                offenders[rel] = sorted(set(hits))
    assert not offenders, (
        f"these cite a norm by a letter no norm has: {offenders}\nReplace "
        f"each with the norm's title from docs/norms.md."
    )


def test_the_norm_letter_debt_is_still_real():
    """An exemption for a file that no longer offends is a licence nobody is
    using, and it would silently cover the next one."""
    settled = [
        rel for rel in sorted(_LETTER_DEBT)
        if not _NORM_LETTER.search((REPO / rel).read_text(encoding="utf-8"))
    ]
    assert not settled, (
        f"_LETTER_DEBT names files that no longer cite a norm by letter: "
        f"{settled}. Drop them from the set."
    )
