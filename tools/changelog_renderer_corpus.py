# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Regenerate `tests/_changelog_renderer_corpus.py` from a CommonMark renderer.

    python tools/changelog_renderer_corpus.py            # show what would change
    python tools/changelog_renderer_corpus.py --write    # rewrite the corpus

**WHY THIS TOOL EXISTS, and it is the whole of the argument.** Two readers in
this repository decide which line of `CHANGELOG.md` is the newest heading — a
bash line grammar in `.github/workflows/release.yml` and its Python twin in
`tests/test_changelog_names_the_version.py`. Held to each other they can be
found to DISAGREE and can never be found to be WRONG, and at `a90862b` they
did not disagree about a single one of the 2954 documents below. What found
them wrong was
a real renderer, and `markdown-it-py` is installed in NONE of this project's
three merge lanes — measured 2026-08-28 on `stelling-jax`, `stelling-nojax`
and `stelling-jax010`, all three `ModuleNotFoundError: No module named
'markdown_it'`. So the oracle cannot run where a merge is gated.

The answer is this tool plus a checked-in corpus: the renderer's verdict on
every document is DERIVED here and recorded there, and
`tests/test_release_gates.py` holds both readers to that recorded column in
every lane. What a lane without the renderer loses is stated in that module
and in the corpus's own docstring — it is the FRESHNESS of the column, not the
driving of it.

**THE INPUTS COME OUT OF THE CORPUS ITSELF.** `ALPHABET`, `MAX_LINES` and the
documents and `must_read` flags of `NAMED` are read from the module this tool
rewrites; only the renderer columns are recomputed. To add a document, add a
row to `NAMED` there with `None` for its renderer verdict and run this with
`--write`. That keeps one copy of the inputs, so the corpus and its generator
cannot disagree about what was rendered.

**WHAT THIS TOOL IS NOT.** It is not a second reader. It calls
`markdown_it.MarkdownIt("commonmark")` and reports the content of the first
`<h2>` token it finds, or `None`. Every judgement about what our own readers
should do with that lives in the tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys

HEADER = '''# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What a real CommonMark renderer says the newest `<h2>` of a document is.

**GENERATED. DO NOT EDIT THE RENDERER COLUMNS BY HAND.** Regenerate with

    python tools/changelog_renderer_corpus.py --write

which needs `markdown-it-py` — a dev-group dependency, and installed in NONE
of this project's three merge lanes.

**WHY A CHECKED-IN ORACLE, AND WHAT IT REPLACED.** Two readers decide which
line of `CHANGELOG.md` is the newest heading: a bash line grammar in
`.github/workflows/release.yml` and its Python twin in
`tests/test_changelog_names_the_version.py`. What used to hold them was
`tests/test_release_gates.py::_HEADING_SHAPES` — ten rows of
`(label, document, expected reading)` where the expected reading was **typed
by the author of the readers**. That is two implementations of one idea plus
one person's reading of CommonMark, and it agreed with itself: none of its ten
rows was in ANY of the four classes on which both readers disagreed with a
renderer, and all four were false PASSES.

**FOUR, AND THE COUNT WENT TWO, THREE, FOUR AS THE ALPHABET GREW, WHICH IS THE
ARGUMENT FOR MAKING IT DERIVABLE.** The item that opened this named two — the
fence asymmetry and setext. A twelve-token alphabet found a third, an HTML
block that is not a comment. An auditor found a fourth, a heading indented four
columns inside a list item, and found it OUTSIDE the sweep: the repair was
already in the whitelist (the ordered-list carve-out is written for exactly
that shape) while the count was not, because no token of the alphabet was a
list. Both of the missing tokens are in `ALPHABET` now, so the number is a
property of the corpus and not of what anybody remembered.

So the `expected` column is gone from that table and the renderer's column is
here instead. Nothing in this file is anybody's opinion about what CommonMark
says; every verdict is what `markdown_it.MarkdownIt("commonmark")` returned,
on the date and version recorded below.

**WHAT A LANE WITHOUT THE RENDERER LOSES — and it is less than it sounds.**
Every lane drives both readers against this column, so the DEFECT-FINDING
power is present everywhere. What only a renderer-carrying environment can do
is notice that the column has gone STALE — that a newer CommonMark, or a newer
`markdown-it-py`, answers differently — and that is
`tests/test_release_gates.py::test_the_recorded_renderer_verdicts_are_still_the_renderers`,
which skips where the library is absent and is registered as a skip in
`tests/test_skip_inventory.py`.

**AND WHAT NOTHING HERE REACHES.** A checked-in oracle can be doctored: an
author who edits a verdict below AND flips the reader to match gets a green
suite in the three merge lanes. That is stated rather than defended — the
defences are that the edit is visible in the diff as a change to a generated
file, that regeneration is one command, and that the renderer-lane test above
turns the doctoring red the moment anyone runs it with the library installed.
There is no construction that removes it short of shipping a CommonMark parser
in this repository, which is the thing this whole item exists not to do.

**THE COLUMNS.**

* :data:`PRODUCT` — one verdict per document, for EVERY document of 1 to
  :data:`MAX_LINES` lines over :data:`ALPHABET`, in `itertools.product` order.
  The documents are not stored; they are re-derived by
  `tools.changelog_renderer_corpus.documents`, which is the one function both
  the generator and the tests call, so a reordering cannot silently re-label a
  column.
* :data:`VERDICTS` — the distinct verdicts, `PRODUCT`'s and
  :data:`SEPARATORS`' entries indexing into it. Each is
  `(1-based line, inline content)`, and index 0 is `(None, None)`, meaning
  **the renderer made no `<h2>` at all** — a different answer from "it made
  one that does not parse as a release heading", and kept distinguishable for
  that reason.

  **THE LINE IS THE ORACLE. THE CONTENT IS FOR THE MESSAGE.** CommonMark
  decides which LINE of a document is a heading and has nothing to say about
  `<version> — <date>`, which is this project's own grammar. Comparing
  CONTENT — which this file did until an auditor swept the whitespace alphabet
  — imports the renderer's own deviations into the comparison: markdown-it's
  ATX rule ends in a JavaScript-derived `.trim()` that strips the whole
  Unicode whitespace set where CommonMark strips spaces and tabs, so 25
  documents disagreed in a direction where OUR readers follow the
  specification and the oracle does not. See `newest_h2` in the generator.
* :data:`NAMED` — `(label, document, must_read, renderer verdict)` for
  documents worth a name. `must_read` is the ONE authored field in this file.
  It cannot license a wrong reading — `True` demands that both readers read
  EXACTLY the renderer's verdict, `False` demands that both REFUSE, and
  neither spelling supplies a reading — so it cannot weaken the SOUNDNESS
  half at all. It CAN weaken the liveness half: flipping `True` to `False`
  after a reader stopped reading a row turns a red green. That is why the
  liveness floors that matter are derived rather than authored, in
  `tests/test_release_gates.py`.
* :data:`SEPARATOR_CHARS` / :data:`SEPARATORS` — the whitespace alphabet
  sweep. One verdict per `(character, form)` pair, characters in code-point
  order and forms in `SEPARATOR_FORM_NAMES` order, both re-derived by
  `tools.changelog_renderer_corpus.separator_documents`. The CHARACTER SET is
  re-derived from `str.isspace()` at test time and the recorded copy is held
  to it in every lane, so this family cannot be shrunk by editing this file.

**WHAT NOTHING IN THIS FILE CAN STOP, and an auditor built it rather than
imagining it.** `ALPHABET` and `MAX_LINES` live here, and cutting either one
and then running the documented `--write` produces a corpus that is perfectly
self-consistent with a live renderer — so even the freshness check stays green.
Driven: `ALPHABET` cut to two tokens takes the product from four figures to
14 rows, and every lane passes. What stands against it is outside this file
and has to be: `tests/test_release_gates.py` asserts that a fixed handful of
DISCRIMINATING documents, typed there, are members of
`documents(ALPHABET, MAX_LINES)`, and that the recorded whitespace alphabet is
the live one. A shrink that drops any of them is a red in every lane.
"""
'''

REPO = pathlib.Path(__file__).resolve().parents[1]
CORPUS = REPO / "tests" / "_changelog_renderer_corpus.py"


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("_corpus_under_edit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def documents(alphabet: tuple[str, ...], max_lines: int) -> list[str]:
    """Every document of 1..`max_lines` lines over `alphabet`, in product order.

    The ORDER is the contract between this tool and the corpus: the corpus
    stores one verdict per document and nothing else, so the documents
    themselves are re-derived rather than stored, and a reordering here is a
    silent re-labelling there. `tests/test_release_gates.py` re-derives them
    with this same function.
    """
    out = []
    for n in range(1, max_lines + 1):
        for combo in itertools.product(alphabet, repeat=n):
            out.append("\n".join(combo) + "\n")
    return out


#: The FIVE positions in this project's heading grammar where one reader or the
#: other spelled "a separator" with a character class BORROWED FROM ITS HOST
#: LANGUAGE instead of the one CommonMark defines.
#:
#: **THIS FAMILY EXISTS BECAUSE AN AUDITOR FOUND TWO SOUNDNESS BREAKS IN IT AND
#: BOTH WERE THE SAME MISTAKE.** CommonMark says an ATX heading's `#` run must
#: be followed by *"spaces or tabs, or end of line"* and a closing fence may be
#: followed *"only by spaces or tabs"*. The bash gate wrote `[[:space:]]`
#: (six characters) and the Python twin wrote `\s` (Unicode-aware, twenty-nine)
#: and `str.strip()` (the same set). Refusal protects a reader where it
#: refuses; where it IMPLEMENTS a rule it owns that rule's alphabet, and
#: "the fence rule is exact and small" was true of its structure and not of its
#: character classes.
#:
#: So the alphabet is swept rather than argued: every character
#: :func:`whitespace_alphabet` finds, in every one of these positions, with the
#: renderer's own verdict recorded beside it.
SEPARATOR_FORMS = (
    # the ATX run's follower — `heading_any` / `_HEADING_LINE`
    ("atx-run", "##{sep}0.2.1 — 2026-08-28\n## 9.9.9 — 2000-01-01\n"),
    # the two separators inside the heading grammar — `heading_re` / `_HEADING`
    ("version-em-dash", "## 0.2.1{sep}— 2026-08-28\n"),
    ("em-dash-date", "## 0.2.1 —{sep}2026-08-28\n"),
    # what a closing fence may be followed by — `${finfo//...}` / `info.strip()`
    ("fence-closer", "```\n```{sep}\n## 0.2.1 — 2026-08-28\n"),
    # and the tail, which both readers strip before parsing
    ("heading-tail", "## 0.2.1 — 2026-08-28{sep}\n"),
)


def whitespace_alphabet() -> tuple[str, ...]:
    """Every character Python calls whitespace, except the line separator.

    DERIVED, over the whole code space, and not a list anybody typed: a list
    would be a fourth alphabet standing beside the three this family exists to
    compare. `\n` is excluded because it cannot occur INSIDE a line — it is
    what ends one, in `read -r` and in `_lines` alike.

    `tests/test_release_gates.py::test_the_recorded_whitespace_alphabet_is_the_live_one`
    re-derives this and holds the recorded copy to it, IN EVERY LANE, so the
    family cannot be shrunk by editing the generated file.
    """
    return tuple(c for c in map(chr, range(0x110000))
                 if c.isspace() and c != "\n")


def separator_documents(chars: tuple[str, ...]) -> list[str]:
    """One document per `(character, form)`, in that order. Derived, not stored."""
    return [shape.format(sep=char) for char in chars for _, shape in SEPARATOR_FORMS]


def newest_h2(renderer, text: str) -> tuple[int | None, str | None]:
    """`(1-based line, inline content)` of the FIRST `<h2>`, or `(None, None)`.

    **THE LINE IS THE ORACLE AND THE CONTENT IS FOR THE MESSAGE, AND THAT
    SPLIT WAS BOUGHT.** This returned the CONTENT alone, and the tests
    compared it — parsed by this project's `<version> — <date>` grammar —
    against what our readers read. Driven over the whitespace alphabet, that
    put 25 documents in the false column that are not defects in either
    reader: `markdown-it` is a port of a JavaScript library and its ATX rule
    ends in `.trim()`, which in ECMAScript and in Python alike strips the
    whole Unicode whitespace set. CommonMark strips *spaces or tabs*. So
    `## 0.2.1 — 2026-08-28<NBSP>` renders with the NBSP GONE, our readers keep
    it, and **our readers are the ones following the specification**.

    Comparing content was importing the oracle's own deviation into the
    comparison. The question this corpus exists to ask has always been
    *"which LINE of this document is the newest heading"* — CommonMark decides
    that and has nothing to say about `<version> — <date>`, which is this
    project's grammar and is held between the two readers by the drives. So
    the recorded oracle is `token.map[0] + 1`, which is exact, carries no
    trimming, and is what `newest_heading_line` answers with.

    `(None, None)` is "this document has no `<h2>` at all", still a different
    answer from "it has one that does not parse as a release heading".
    """
    tokens = renderer.parse(text)
    for at, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2":
            return token.map[0] + 1, tokens[at + 1].content
    return None, None


def render(corpus):
    """`(product, named, separator, chars, renderer version)` for `corpus`.

    The separator alphabet is re-derived HERE rather than read out of the
    corpus, so regenerating cannot preserve a shrunken copy of it.
    """
    import markdown_it  # noqa: PLC0415 - the whole point is that it is optional

    renderer = markdown_it.MarkdownIt("commonmark")
    product = [newest_h2(renderer, text)
               for text in documents(corpus.ALPHABET, corpus.MAX_LINES)]
    named = [newest_h2(renderer, text) for _, text, _, _ in corpus.NAMED]
    chars = whitespace_alphabet()
    separators = [newest_h2(renderer, text) for text in separator_documents(chars)]
    return product, named, separators, chars, markdown_it.__version__


def _lit(value) -> str:
    return "None" if value is None else repr(value)


def _verdict(value) -> str:
    line, content = value
    return f"({_lit(line)}, {_lit(content)})"


def emit(corpus, product, named, separators, chars, version, today) -> str:
    """The whole corpus module, as text."""
    verdicts: list[tuple[int | None, str | None]] = [(None, None)]
    for verdict in [*product, *separators]:
        if verdict not in verdicts:
            verdicts.append(verdict)
    index = {verdict: at for at, verdict in enumerate(verdicts)}

    lines = [HEADER.rstrip("\n"), ""]
    lines.append('RENDERER = "markdown-it-py"')
    lines.append(f"RENDERER_VERSION = {version!r}")
    lines.append('RENDERER_PRESET = "commonmark"')
    lines.append(f"GENERATED = {today!r}")
    lines.append("")
    lines.append("ALPHABET = (")
    for token in corpus.ALPHABET:
        lines.append(f"    {token!r},")
    lines.append(")")
    lines.append("")
    lines.append(f"MAX_LINES = {corpus.MAX_LINES}")
    lines.append("")
    lines.append("VERDICTS = (")
    for verdict in verdicts:
        lines.append(f"    {_verdict(verdict)},")
    lines.append(")")
    lines.append("")
    lines.append("PRODUCT = (")
    row: list[str] = []
    for verdict in product:
        row.append(str(index[verdict]))
        if len(row) == 24:
            lines.append("    " + " ".join(f"{n}," for n in row))
            row = []
    if row:
        lines.append("    " + " ".join(f"{n}," for n in row))
    lines.append(")")
    lines.append("")
    lines.append("NAMED = (")
    for (label, text, must_read, _), verdict in zip(corpus.NAMED, named):
        lines.append("    (")
        lines.append(f"        {label!r},")
        lines.append(f"        {text!r},")
        lines.append(f"        {must_read!r},")
        lines.append(f"        {_verdict(verdict)},")
        lines.append("    ),")
    lines.append(")")
    lines.append("")
    lines.append("SEPARATOR_CHARS = (")
    row = []
    for char in chars:
        row.append(f"0x{ord(char):04x},")
        if len(row) == 8:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append(")")
    lines.append("")
    lines.append("SEPARATOR_FORM_NAMES = (")
    for name, _ in SEPARATOR_FORMS:
        lines.append(f"    {name!r},")
    lines.append(")")
    lines.append("")
    lines.append("SEPARATORS = (")
    row = []
    for verdict in separators:
        row.append(f"{index[verdict]},")
        if len(row) == 20:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append(")")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="rewrite tests/_changelog_renderer_corpus.py")
    parser.add_argument("--date", default=None,
                        help="the date to record (default: today, UTC)")
    args = parser.parse_args(argv)

    import datetime  # noqa: PLC0415 - a recorded generation date, not a gate

    today = args.date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    corpus = _load(CORPUS)
    product, named, separators, chars, version = render(corpus)
    text = emit(corpus, product, named, separators, chars, version, today)
    counts = (f"{len(product)} product rows, {len(named)} named rows, "
              f"{len(separators)} separator rows over {len(chars)} characters, "
              f"markdown-it-py {version}")
    if args.write:
        CORPUS.write_text(text, encoding="utf-8")
        print(f"wrote {CORPUS} ({counts})")
        return 0
    same = CORPUS.read_text(encoding="utf-8") == text
    print(f"{CORPUS}: {'up to date' if same else 'WOULD CHANGE'} ({counts})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
