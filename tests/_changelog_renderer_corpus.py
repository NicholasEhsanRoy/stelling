# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
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

RENDERER = "markdown-it-py"
RENDERER_VERSION = '4.2.0'
RENDERER_PRESET = "commonmark"
GENERATED = '2026-08-29'

ALPHABET = (
    '## 0.2.1 — 2026-08-28',
    '    ## 9.9.9 — 2000-01-01',
    '1. item',
    'text',
    '---',
    '===',
    '```',
    '``` t',
    '````',
    '~~~',
    '<!--',
    '-->',
    '`<!--`',
    '<div>',
)

MAX_LINES = 3

VERDICTS = (
    (None, None, None),
    (1, '## 0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (2, '## 0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (2, '    ## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    (1, 'text', 'text'),
    (1, '===', '==='),
    (1, '-->', '-->'),
    (1, '`<!--`', '`<!--`'),
    (3, '## 0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (3, '    ## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    (2, 'text', 'text'),
    (2, '===', '==='),
    (2, '-->', '-->'),
    (2, '`<!--`', '`<!--`'),
    (1, 'text', 'text\n    ## 9.9.9 — 2000-01-01'),
    (1, 'text', 'text\ntext'),
    (1, 'text', 'text\n-->'),
    (1, 'text', 'text\n`<!--`'),
    (1, '===', '===\n    ## 9.9.9 — 2000-01-01'),
    (1, '===', '===\ntext'),
    (1, '===', '===\n-->'),
    (1, '===', '===\n`<!--`'),
    (1, '-->', '-->\n    ## 9.9.9 — 2000-01-01'),
    (1, '-->', '-->\ntext'),
    (1, '-->', '-->\n-->'),
    (1, '-->', '-->\n`<!--`'),
    (1, '`<!--`', '`<!--`\n    ## 9.9.9 — 2000-01-01'),
    (1, '`<!--`', '`<!--`\ntext'),
    (1, '`<!--`', '`<!--`\n-->'),
    (1, '`<!--`', '`<!--`\n`<!--`'),
    (2, '## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    (1, '##\t0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\t— 2026-08-28', '0.2.1\t— 2026-08-28'),
    (1, '## 0.2.1 —\t2026-08-28', '0.2.1 —\t2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\t', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x0b— 2026-08-28', '0.2.1\x0b— 2026-08-28'),
    (1, '## 0.2.1 —\x0b2026-08-28', '0.2.1 —\x0b2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x0b', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x0c— 2026-08-28', '0.2.1\x0c— 2026-08-28'),
    (1, '## 0.2.1 —\x0c2026-08-28', '0.2.1 —\x0c2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x0c', '0.2.1 — 2026-08-28'),
    (1, '##', ''),
    (1, '## 0.2.1', '0.2.1'),
    (1, '## 0.2.1 —', '0.2.1 —'),
    (4, '## 0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x1c— 2026-08-28', '0.2.1\x1c— 2026-08-28'),
    (1, '## 0.2.1 —\x1c2026-08-28', '0.2.1 —\x1c2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x1c', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x1d— 2026-08-28', '0.2.1\x1d— 2026-08-28'),
    (1, '## 0.2.1 —\x1d2026-08-28', '0.2.1 —\x1d2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x1d', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x1e— 2026-08-28', '0.2.1\x1e— 2026-08-28'),
    (1, '## 0.2.1 —\x1e2026-08-28', '0.2.1 —\x1e2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x1e', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x1f— 2026-08-28', '0.2.1\x1f— 2026-08-28'),
    (1, '## 0.2.1 —\x1f2026-08-28', '0.2.1 —\x1f2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x1f', '0.2.1 — 2026-08-28'),
    (1, ' ## 0.2.1 — 2026-08-28', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28 ', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\x85— 2026-08-28', '0.2.1\x85— 2026-08-28'),
    (1, '## 0.2.1 —\x852026-08-28', '0.2.1 —\x852026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\x85', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\xa0— 2026-08-28', '0.2.1\xa0— 2026-08-28'),
    (1, '## 0.2.1 —\xa02026-08-28', '0.2.1 —\xa02026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\xa0', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u1680— 2026-08-28', '0.2.1\u1680— 2026-08-28'),
    (1, '## 0.2.1 —\u16802026-08-28', '0.2.1 —\u16802026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u1680', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2000— 2026-08-28', '0.2.1\u2000— 2026-08-28'),
    (1, '## 0.2.1 —\u20002026-08-28', '0.2.1 —\u20002026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2000', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2001— 2026-08-28', '0.2.1\u2001— 2026-08-28'),
    (1, '## 0.2.1 —\u20012026-08-28', '0.2.1 —\u20012026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2001', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2002— 2026-08-28', '0.2.1\u2002— 2026-08-28'),
    (1, '## 0.2.1 —\u20022026-08-28', '0.2.1 —\u20022026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2002', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2003— 2026-08-28', '0.2.1\u2003— 2026-08-28'),
    (1, '## 0.2.1 —\u20032026-08-28', '0.2.1 —\u20032026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2003', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2004— 2026-08-28', '0.2.1\u2004— 2026-08-28'),
    (1, '## 0.2.1 —\u20042026-08-28', '0.2.1 —\u20042026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2004', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2005— 2026-08-28', '0.2.1\u2005— 2026-08-28'),
    (1, '## 0.2.1 —\u20052026-08-28', '0.2.1 —\u20052026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2005', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2006— 2026-08-28', '0.2.1\u2006— 2026-08-28'),
    (1, '## 0.2.1 —\u20062026-08-28', '0.2.1 —\u20062026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2006', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2007— 2026-08-28', '0.2.1\u2007— 2026-08-28'),
    (1, '## 0.2.1 —\u20072026-08-28', '0.2.1 —\u20072026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2007', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2008— 2026-08-28', '0.2.1\u2008— 2026-08-28'),
    (1, '## 0.2.1 —\u20082026-08-28', '0.2.1 —\u20082026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2008', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2009— 2026-08-28', '0.2.1\u2009— 2026-08-28'),
    (1, '## 0.2.1 —\u20092026-08-28', '0.2.1 —\u20092026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2009', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u200a— 2026-08-28', '0.2.1\u200a— 2026-08-28'),
    (1, '## 0.2.1 —\u200a2026-08-28', '0.2.1 —\u200a2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u200a', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2028— 2026-08-28', '0.2.1\u2028— 2026-08-28'),
    (1, '## 0.2.1 —\u20282026-08-28', '0.2.1 —\u20282026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2028', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u2029— 2026-08-28', '0.2.1\u2029— 2026-08-28'),
    (1, '## 0.2.1 —\u20292026-08-28', '0.2.1 —\u20292026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u2029', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u202f— 2026-08-28', '0.2.1\u202f— 2026-08-28'),
    (1, '## 0.2.1 —\u202f2026-08-28', '0.2.1 —\u202f2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u202f', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u205f— 2026-08-28', '0.2.1\u205f— 2026-08-28'),
    (1, '## 0.2.1 —\u205f2026-08-28', '0.2.1 —\u205f2026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u205f', '0.2.1 — 2026-08-28'),
    (1, '## 0.2.1\u3000— 2026-08-28', '0.2.1\u3000— 2026-08-28'),
    (1, '## 0.2.1 —\u30002026-08-28', '0.2.1 —\u30002026-08-28'),
    (1, '## 0.2.1 — 2026-08-28\u3000', '0.2.1 — 2026-08-28'),
)

PRODUCT = (
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 10, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0,
    11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0,
    0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0,
    14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 8, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0,
    12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 8, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 21, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 8, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 25, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 8, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 9,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 8, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 29, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,
)

NAMED = (
    (
        'plain',
        '## 0.2.0 — 2026-08-25\n',
        True,
        (1, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'older below the newest',
        '## 0.2.0 — 2026-08-25\n\nbody\n\n## 0.1.0 — 2026-08-12\n',
        True,
        (1, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'indented one space',
        '  ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        True,
        (1, '  ## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'indented three spaces',
        '   ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        True,
        (1, '   ## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'indented four spaces is a code block',
        '    ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'inside an HTML comment',
        '<!--\n## 9.9.9 — 2000-01-01\n-->\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (5, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'inside a fenced block',
        '```\n## 9.9.9 — 2000-01-01\n```\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (5, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'bare ## above a real heading',
        '##\n0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        False,
        (1, '##', ''),
    ),
    (
        'no em dash',
        '## 0.2.0 - 2026-08-25\n',
        False,
        (1, '## 0.2.0 - 2026-08-25', '0.2.0 - 2026-08-25'),
    ),
    (
        'no heading at all',
        'nothing here\n',
        False,
        (None, None, None),
    ),
    (
        "the comment and the title this repository's own changelog opens with",
        '<!--\nSPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy\nSPDX-License-Identifier: Apache-2.0\n-->\n\n# Changelog\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (8, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'prose above the newest heading',
        'This file records what changed.\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a one-line HTML comment',
        '<!-- a note -->\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'CRLF',
        '## 0.2.0 — 2026-08-25\r\n',
        True,
        (1, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a fence closed by a LONGER run of the same character',
        '```\n````\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (4, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a fence the closer does not match by CHARACTER',
        '```\n~~~\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a fence whose closer carries an INFO STRING',
        '```\n``` t\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a fence the closer does not match by RUN LENGTH',
        '````\n```\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'an unclosed fence swallows the rest of the file',
        '```\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'an unclosed comment swallows the rest of the file',
        '<!--\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a setext <h2> above the newest ATX heading',
        'text\n---\n## 0.2.0 — 2026-08-25\n',
        False,
        (1, 'text', 'text'),
    ),
    (
        'a setext <h1> above the newest ATX heading',
        'text\n===\n## 0.2.0 — 2026-08-25\n',
        False,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'the comment closer alone, under a setext underline',
        '-->\n---\n## 0.2.0 — 2026-08-25\n',
        False,
        (1, '-->', '-->'),
    ),
    (
        'a code span holding the comment opener',
        '`<!--`\n## 0.2.0 — 2026-08-25\n',
        True,
        (2, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a sentence mentioning the comment opener',
        'an HTML comment opens with <!-- and this one never closes\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'an HTML block that is not a comment',
        '<div>\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a heading inside a block quote',
        '> ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (1, '> ## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'a heading inside a list item',
        '1. text\n\n    ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (3, '    ## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'a bullet list above the newest heading',
        '- item\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a tab-indented line above the newest heading',
        '\t## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a form feed is not a line break',
        'note\x0c## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'an ATX heading of another level above the newest',
        '#### notes\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a bare carriage return is a line ending',
        'text\r## 9.9.9 — 2000-01-01\n## 0.2.0 — 2026-08-25\n',
        True,
        (2, '## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'a bare carriage return beside a CRLF one',
        'text\r\ntext\r## 9.9.9 — 2000-01-01\r\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'a NUL byte cannot be represented and is refused',
        '#\x00# 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a NUL byte above a heading the renderer does read',
        '#\x00# 0.2.0 — 2026-08-25\n## 9.9.9 — 2000-01-01\n',
        False,
        (2, '## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'an ordered-list marker with a non-ASCII digit is not a list',
        '1٢. item\n    ## 9.9.9 — 2000-01-01\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a version with a non-ASCII digit does not parse',
        '## ٢0.0 — 2026-08-25\n',
        False,
        (1, '## ٢0.0 — 2026-08-25', '٢0.0 — 2026-08-25'),
    ),
    (
        'a setext underline at indent THREE',
        'text\n   ---\n## 0.2.0 — 2026-08-25\n',
        False,
        (1, 'text', 'text'),
    ),
    (
        'a block quote at indent THREE',
        '   > ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (1, '   > ## 9.9.9 — 2000-01-01', '9.9.9 — 2000-01-01'),
    ),
    (
        'an HTML block at indent THREE',
        '   <div>\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a fence closer at indent FOUR is not a closer',
        '```\n    ```\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'a short TILDE run is a paragraph, as a short backtick run is',
        '~x\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a backtick fence whose INFO STRING holds a backtick is a paragraph',
        '``` a`b\n\n## 0.2.0 — 2026-08-25\n',
        True,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
    (
        'a TILDE fence whose info string holds a tilde IS a fence',
        '~~~ a~b\n## 0.2.0 — 2026-08-25\n',
        False,
        (None, None, None),
    ),
    (
        'seven hashes is not a heading',
        '####### 9.9.9\n\n## 0.2.0 — 2026-08-25\n',
        False,
        (3, '## 0.2.0 — 2026-08-25', '0.2.0 — 2026-08-25'),
    ),
)

SEPARATOR_CHARS = (
    0x0009, 0x000b, 0x000c, 0x000d, 0x001c, 0x001d, 0x001e, 0x001f,
    0x0020, 0x0085, 0x00a0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
    0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
    0x2029, 0x202f, 0x205f, 0x3000,
)

SEPARATOR_FORM_NAMES = (
    'indent',
    'atx-run',
    'atx-other-run',
    'version-em-dash',
    'em-dash-date',
    'heading-tail',
    'fence-closer',
    'blank-line',
    'ordered-list-marker',
)

SEPARATORS = (
    30, 31, 2, 32, 33, 34, 8, 8, 3, 30, 30, 2, 35, 36, 37, 0, 0, 8, 30, 30,
    2, 38, 39, 40, 0, 0, 8, 2, 41, 8, 42, 43, 1, 8, 8, 44, 30, 30, 2, 45,
    46, 47, 0, 0, 8, 30, 30, 2, 48, 49, 50, 0, 0, 8, 30, 30, 2, 51, 52, 53,
    0, 0, 8, 30, 30, 2, 54, 55, 56, 0, 0, 8, 57, 1, 2, 1, 1, 58, 8, 8,
    3, 30, 30, 2, 59, 60, 61, 0, 0, 8, 30, 30, 2, 62, 63, 64, 0, 0, 8, 30,
    30, 2, 65, 66, 67, 0, 0, 8, 30, 30, 2, 68, 69, 70, 0, 0, 8, 30, 30, 2,
    71, 72, 73, 0, 0, 8, 30, 30, 2, 74, 75, 76, 0, 0, 8, 30, 30, 2, 77, 78,
    79, 0, 0, 8, 30, 30, 2, 80, 81, 82, 0, 0, 8, 30, 30, 2, 83, 84, 85, 0,
    0, 8, 30, 30, 2, 86, 87, 88, 0, 0, 8, 30, 30, 2, 89, 90, 91, 0, 0, 8,
    30, 30, 2, 92, 93, 94, 0, 0, 8, 30, 30, 2, 95, 96, 97, 0, 0, 8, 30, 30,
    2, 98, 99, 100, 0, 0, 8, 30, 30, 2, 101, 102, 103, 0, 0, 8, 30, 30, 2, 104,
    105, 106, 0, 0, 8, 30, 30, 2, 107, 108, 109, 0, 0, 8, 30, 30, 2, 110, 111, 112,
    0, 0, 8, 30, 30, 2, 113, 114, 115, 0, 0, 8,
)
