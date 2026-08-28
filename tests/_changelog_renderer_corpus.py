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

RENDERER = "markdown-it-py"
RENDERER_VERSION = '4.2.0'
RENDERER_PRESET = "commonmark"
GENERATED = '2026-08-28'

ALPHABET = (
    '## 0.2.1 — 2026-08-28',
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
    None,
    '0.2.1 — 2026-08-28',
    'text',
    '===',
    '-->',
    '`<!--`',
    'text\ntext',
    'text\n-->',
    'text\n`<!--`',
    '===\ntext',
    '===\n-->',
    '===\n`<!--`',
    '-->\ntext',
    '-->\n-->',
    '-->\n`<!--`',
    '`<!--`\ntext',
    '`<!--`\n-->',
    '`<!--`\n`<!--`',
)

PRODUCT = (
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

NAMED = (
    (
        'plain',
        '## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'older below the newest',
        '## 0.2.0 — 2026-08-25\n\nbody\n\n## 0.1.0 — 2026-08-12\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'indented one space',
        '  ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'indented three spaces',
        '   ## 0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'indented four spaces is a code block',
        '    ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'inside an HTML comment',
        '<!--\n## 9.9.9 — 2000-01-01\n-->\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'inside a fenced block',
        '```\n## 9.9.9 — 2000-01-01\n```\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'bare ## above a real heading',
        '##\n0.2.0 — 2026-08-25\n\n## 0.1.0 — 2026-08-12\n',
        False,
        '',
    ),
    (
        'no em dash',
        '## 0.2.0 - 2026-08-25\n',
        False,
        '0.2.0 - 2026-08-25',
    ),
    (
        'no heading at all',
        'nothing here\n',
        False,
        None,
    ),
    (
        "the comment and the title this repository's own changelog opens with",
        '<!--\nSPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy\nSPDX-License-Identifier: Apache-2.0\n-->\n\n# Changelog\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'prose above the newest heading',
        'This file records what changed.\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a one-line HTML comment',
        '<!-- a note -->\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'CRLF',
        '## 0.2.0 — 2026-08-25\r\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a fence closed by a LONGER run of the same character',
        '```\n````\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a fence the closer does not match by CHARACTER',
        '```\n~~~\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'a fence whose closer carries an INFO STRING',
        '```\n``` t\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'a fence the closer does not match by RUN LENGTH',
        '````\n```\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'an unclosed fence swallows the rest of the file',
        '```\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'an unclosed comment swallows the rest of the file',
        '<!--\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'a setext <h2> above the newest ATX heading',
        'text\n---\n## 0.2.0 — 2026-08-25\n',
        False,
        'text',
    ),
    (
        'a setext <h1> above the newest ATX heading',
        'text\n===\n## 0.2.0 — 2026-08-25\n',
        False,
        '0.2.0 — 2026-08-25',
    ),
    (
        'the comment closer alone, under a setext underline',
        '-->\n---\n## 0.2.0 — 2026-08-25\n',
        False,
        '-->',
    ),
    (
        'a code span holding the comment opener',
        '`<!--`\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a sentence mentioning the comment opener',
        'an HTML comment opens with <!-- and this one never closes\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'an HTML block that is not a comment',
        '<div>\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'a heading inside a block quote',
        '> ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        '9.9.9 — 2000-01-01',
    ),
    (
        'a heading inside a list item',
        '1. text\n\n    ## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        '9.9.9 — 2000-01-01',
    ),
    (
        'a bullet list above the newest heading',
        '- item\n\n## 0.2.0 — 2026-08-25\n',
        False,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a tab-indented line above the newest heading',
        '\t## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        False,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a form feed is not a line break',
        'note\x0c## 9.9.9 — 2000-01-01\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'an ATX heading of another level above the newest',
        '#### notes\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a backtick fence whose INFO STRING holds a backtick is a paragraph',
        '``` a`b\n\n## 0.2.0 — 2026-08-25\n',
        True,
        '0.2.0 — 2026-08-25',
    ),
    (
        'a TILDE fence whose info string holds a tilde IS a fence',
        '~~~ a~b\n## 0.2.0 — 2026-08-25\n',
        False,
        None,
    ),
    (
        'seven hashes is not a heading',
        '####### 9.9.9\n\n## 0.2.0 — 2026-08-25\n',
        False,
        '0.2.0 — 2026-08-25',
    ),
)
