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
did not disagree about a single one of the 1884 documents below. What found
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
rows was in ANY of the three classes on which both readers disagreed with a
renderer, and all three were false PASSES. Three, not two: the item that
opened this named the fence asymmetry and setext, and the sweep that replaced
its ten-token alphabet with a twelve-token one found a third — an HTML block
that is not a comment, `<div>` say, which swallows the lines below it and
which no ten-token alphabet without an HTML tag could reach.

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

**THE THREE COLUMNS.**

* :data:`PRODUCT` — one verdict per document, for EVERY document of 1 to
  :data:`MAX_LINES` lines over :data:`ALPHABET`, in `itertools.product` order.
  The documents are not stored; they are re-derived by
  `tools.changelog_renderer_corpus.documents`, which is the one function both
  the generator and the tests call, so a reordering cannot silently re-label a
  column.
* :data:`VERDICTS` — the distinct verdicts, `PRODUCT`'s entries indexing into
  it. Index 0 is `None`, meaning **the renderer made no `<h2>` at all**, which
  is a different answer from "it made one that does not parse as a release
  heading" and is kept distinguishable for that reason.
* :data:`NAMED` — `(label, document, must_read, renderer verdict)` for
  documents worth a name. `must_read` is the ONE authored field in this file
  and it cannot license a wrong reading: `True` demands that both readers read
  EXACTLY the renderer's verdict, `False` demands that both REFUSE. Neither
  spelling supplies a reading, so flipping one can only move a green to a red.
  It exists because a gate that refused every document would satisfy a
  soundness relation and be useless.
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


def newest_h2(renderer, text: str) -> str | None:
    """The inline content of the FIRST `<h2>` the renderer makes, or `None`.

    `None` is "this document has no `<h2>` at all" and is a different answer
    from "it has one that does not parse as a release heading" — the second is
    a string this returns and the tests refuse.
    """
    tokens = renderer.parse(text)
    for at, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h2":
            return tokens[at + 1].content
    return None


def render(corpus):
    """`(product verdicts, named verdicts, renderer version)` for `corpus`."""
    import markdown_it  # noqa: PLC0415 - the whole point is that it is optional

    renderer = markdown_it.MarkdownIt("commonmark")
    product = [newest_h2(renderer, text)
               for text in documents(corpus.ALPHABET, corpus.MAX_LINES)]
    named = [newest_h2(renderer, text) for _, text, _, _ in corpus.NAMED]
    return product, named, markdown_it.__version__


def _lit(value) -> str:
    return "None" if value is None else repr(value)


def emit(corpus, product, named, version, today) -> str:
    """The whole corpus module, as text."""
    verdicts: list[str | None] = [None]
    for verdict in product:
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
        lines.append(f"    {_lit(verdict)},")
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
        lines.append(f"        {_lit(verdict)},")
        lines.append("    ),")
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
    product, named, version = render(corpus)
    text = emit(corpus, product, named, version, today)
    if args.write:
        CORPUS.write_text(text, encoding="utf-8")
        print(f"wrote {CORPUS} ({len(product)} product rows, "
              f"{len(named)} named rows, markdown-it-py {version})")
        return 0
    same = CORPUS.read_text(encoding="utf-8") == text
    print(f"{CORPUS}: {'up to date' if same else 'WOULD CHANGE'} "
          f"({len(product)} product rows, {len(named)} named rows, "
          f"markdown-it-py {version})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
