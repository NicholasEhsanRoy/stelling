# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The fences over what CI actually runs.

``tests/_lanes.py`` reads ``ci.yml``; this holds every claim in the tree that
depends on a lane to the lane that delivers it. Three claims, one shape — a
coverage statement checked against an enumeration instead of against the
machine that would have to produce it:

* ``TESTED_JAX_SERIES`` — *"an entry with no lane is a claim, not a test"* is
  the rule written above that constant. This is the first thing that enforces
  it.
* the supported install configurations — the ``[jax]``-without-``[solvers]``
  one had no whole-suite lane, and 72 tests were failing in it.
* the randomised-order lane, which is not a coverage claim but an
  *unenumerated* backstop, and which has to exist before anything can say the
  suite is order-independent.

None of these needs the workflow to be RUN. They need it to be READ.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

import _lanes
from stelling._optional import TESTED_JAX_SERIES


def test_the_declared_lane_table_is_what_ci_yml_says():
    """The measured/declared pin. Same idiom as the skip inventory: a lane
    added, removed or re-provisioned is a line in a diff, never a silent
    change in what CI measures."""
    measured = {
        lane.job: (
            lane.jax,
            lane.solvers,
            lane.whole_suite,
            lane.random_order,
            lane.verdict_channel,
        )
        for lane in _lanes.lanes()
    }
    assert measured == _lanes.EXPECTED_LANES, (
        "the CI lanes moved.\n"
        f"  declared {_lanes.EXPECTED_LANES}\n"
        f"  measured {measured}\n"
        "Update EXPECTED_LANES *and* check what else in this file depends on "
        "the lane that changed. This table is a claim about which environment "
        "CI actually provisions; it is not a list to be re-typed until the "
        "suite goes green."
    )


def test_a_matrix_job_is_EXPANDED_and_a_field_its_entries_disagree_on_is_None():
    """THE DEFECT THIS INSTRUMENT HAD, WHICH IS THE ONE IT EXISTS TO CLOSE.

    ``_classify`` used to write ``solvers = True`` for any job whose install
    line was a matrix expansion, with the comment *"every matrix entry names an
    extras set containing it"* — a permissive constant standing in for a
    reading, inside the module whose docstring said in bold that nothing is
    inferred from a matrix. The measured/declared pin could not see it: both
    sides were the same constant. DRIVEN, with ``extras: jax`` in both
    ``acceptance-reproducer`` entries and nothing else changed::

        before: 8 passed
                Lane(job='acceptance-reproducer', jax='matrix', solvers=True, …)

    A can't-tell that resolves to the permissive answer is not a can't-tell;
    it is a guess with an alibi. ``design/lessons-ledger.md`` L23.

    The resolution rule is driven here on synthetic bodies rather than on
    ci.yml, because ci.yml exercises exactly one case of it — entries that
    agree, and agree the solvers are there — and every other case is one that
    has to fail closed.
    """
    body = _reproducer_body()
    assert _lanes._matrix_include(body), (
        "the matrix include block is no longer readable, so every field a "
        "matrix job carries has gone back to being a constant"
    )
    extras = _lanes._matrix_values(body, "EXTRAS")
    assert extras and len(extras) >= 2, (
        f"the `${{EXTRAS}}` chain — install line to step env to matrix key — "
        f"no longer resolves to per-entry values, got {extras!r}"
    )
    # each case of the rule, each one a whole synthetic job
    assert _synthetic(["solvers,jax", "solvers"]).solvers is True
    assert _synthetic(["jax", ""]).solvers is False
    assert _synthetic(["solvers,jax", "jax"]).solvers is None, (
        "entries that disagree about the solver extra must read as the named "
        "can't-tell; one Lane cannot describe two provisionings"
    )
    assert _lanes._classify(_UNREADABLE_MATRIX).solvers is None, (
        "a matrix whose expansion this module cannot follow must read None, "
        "not the permissive True"
    )
    # A TRAILING COMMENT IS NOT PART OF THE VALUE. This is the whole finding
    # one level down: `extras: jax   # solvers come in with jaxfluids` in both
    # entries — nothing else changed — used to read `solvers = True`, because
    # the comment was unquoted along with the value and `\bsolvers\b` found
    # itself in the prose. `11 passed`, and the lane credited with a solver
    # extra it does not install: a can't-tell resolved permissively by a
    # comment, in the module whose own docstring says this file "is more
    # comment than code, deliberately".
    assert _synthetic(
        ["jax   # solvers come in with jaxfluids"] * 2, quoted=False
    ).solvers is False
    assert _synthetic(
        ['"jax"   # quoted value, comment outside it'] * 2, quoted=False
    ).solvers is False
    assert _synthetic(["solvers   # and not jax"] * 2, quoted=False).solvers is True
    # the quote-aware half: inside quotes a `#` is three characters of the
    # value, not the start of a comment, so this one really does name an
    # extras set with `solvers` in it — absurd as an install, correct as a
    # reading, and the control that keeps the strip from eating quoted text.
    assert _synthetic(["jax # solvers"]).solvers is True
    # RECOGNISING A MATRIX AND FOLLOWING IT ARE SEPARATE. A `${{ matrix.x }}`
    # interpolated straight into the install line names no shell variable, so
    # the chain stops at its first link — and the reading has to be `"matrix"`
    # plus a can't-tell, never an ordinary literal install with no solvers.
    direct = _lanes._classify(_DIRECT_INTERPOLATION)
    assert (direct.jax, direct.solvers) == ("matrix", None), direct
    # and the substring trap the literal path already guards against
    assert _synthetic(["not-solvers-really", "not-solvers-really"]).solvers is True
    assert _synthetic(["nosolversatall", "nosolversatall"]).solvers is False


def test_every_link_of_the_matrix_CHAIN_fails_closed_when_it_is_broken():
    """The four links :func:`_lanes._matrix_values` enumerates, each broken.

    The docstring above it names four — the install line has to name a shell
    variable, an ``env:`` has to bind it to a matrix key, the include block has
    to be readable, and every entry has to carry that key — and the fence next
    door drove the first two. The code failed closed at all four; NOTHING
    pinned links 3 and 4, so the day one of them starts returning a partial
    reading instead of ``None`` the permissive answer is back and no test in
    this tree moves.

    Failing closed means ``solvers is None``: the named can't-tell, which
    ``test_no_lane_a_claim_rests_on_is_a_cant_tell`` then refuses to let any
    coverage claim rest on. Reading ``False`` would be just as wrong as
    reading ``True`` — it is a lane about which this module knows nothing.

    THE NESTED SHAPES ARE THE SAME LINK MEASURED THREE WAYS, and the third of
    them is why ``_matrix_include`` refuses nesting by COLUMN now rather than
    by a repeated key. A repeat is not the only way flattening changes an
    answer: an entry with no ``extras`` of its own whose nested mapping
    supplies one reads ``solvers = True`` with nothing repeated anywhere.
    """
    # 1 — the install line names no shell variable
    assert _lanes._classify(_DIRECT_INTERPOLATION).solvers is None
    # 2 — no `env:` binds that variable to a matrix key
    assert _lanes._classify(_UNREADABLE_MATRIX).solvers is None
    # 3 — there is no readable include block
    assert _lanes._matrix_include(_NO_INCLUDE_BLOCK) == []
    assert _lanes._classify(_NO_INCLUDE_BLOCK).solvers is None
    # 4 — an entry does not carry the key
    assert _lanes._matrix_include(_KEY_MISSING_FROM_AN_ENTRY), (
        "this case must break at link 4, not earlier: the entries have to PARSE"
    )
    assert _lanes._matrix_values(_KEY_MISSING_FROM_AN_ENTRY, "EXTRAS") is None
    assert _lanes._classify(_KEY_MISSING_FROM_AN_ENTRY).solvers is None
    # and the two nested shapes, which are link 3 in the shape it really has
    assert _lanes._matrix_include(_BARE_LIST_ITEM) == []
    assert _lanes._classify(_BARE_LIST_ITEM).solvers is None
    assert _lanes._matrix_include(_NESTED_KEY_SHADOWS_THE_ENTRY) == [], (
        "a nested mapping repeating a key the entry already has used to "
        "OVERRIDE it: entry-level `extras: \"jax\"` with a nested `extras: "
        "solvers` read solvers=True, out of a structure this parser does not "
        "model"
    )
    assert _lanes._classify(_NESTED_KEY_SHADOWS_THE_ENTRY).solvers is None
    # ... and the shape the "only a REPEATED key can change an answer"
    # sentence said could not exist. Nothing is repeated here: the entry has
    # no `extras` at all and the nesting supplies it. Read flat, that is
    # `solvers = True` out of a structure this parser does not model — the
    # permissive answer, from the one field this module actually reads.
    assert _lanes._matrix_include(_NESTED_KEY_SUPPLIES_A_NEW_ONE) == [], (
        "an entry with no `extras` of its own read one out of a NESTED "
        "mapping: every key the nesting contributed was new, so the "
        "repeated-key check never fired and the lane was credited with a "
        "solver extra by a structure this parser does not model"
    )
    assert _lanes._classify(_NESTED_KEY_SUPPLIES_A_NEW_ONE).solvers is None
    # ... and a nested SEQUENCE, which "ALL NESTING IS REFUSED, BY COLUMN" was
    # not true of: the column rule read the KEY's column and not the `- `'s,
    # so a deeper `- ` started a PHANTOM entry instead of ending the read.
    # Measured at `844ba48`: three entries for the first body and two for the
    # second, against the one and two their files really have.
    assert _lanes._matrix_include(_NESTED_SEQUENCE_OF_MAPPINGS) == [], (
        "a nested LIST of mappings flattens into extra entries: a `- ` at a "
        "deeper column started a phantom entry rather than ending the read, "
        "so this returned an entry list with a member ci.yml does not have"
    )
    assert _lanes._classify(_NESTED_SEQUENCE_OF_MAPPINGS).solvers is None
    assert _lanes._matrix_include(_NESTED_SEQUENCE_OF_A_DIFFERENT_KEY) == []
    assert _lanes._classify(_NESTED_SEQUENCE_OF_A_DIFFERENT_KEY).solvers is None
    # and the repeated-key check is still the one refusing a duplicate key at
    # the entry's OWN column, which no column rule can reach
    assert _lanes._matrix_include(_KEY_REPEATED_AT_THE_ENTRYS_OWN_COLUMN) == []
    assert _lanes._classify(_KEY_REPEATED_AT_THE_ENTRYS_OWN_COLUMN).solvers is None


def test_a_comment_is_stripped_the_same_way_everywhere():
    """:func:`_lanes._strip_comment`, which is why there is one of it.

    Three readings in ``_lanes.py`` were each defeated by a trailing comment,
    in three different places, all in the permissive direction. The unit is
    here so the rule itself — ``#`` opens a comment when it begins a word and
    is not inside quotes — is pinned once rather than three times, and so that
    the two things it must NOT do are pinned at all.
    """
    strip = _lanes._strip_comment
    assert strip("extras: jax   # solvers arrive transitively") == "extras: jax"
    assert strip("# whole-line") == ""
    assert strip("    # indented whole-line") == ""
    assert strip("plain, no comment") == "plain, no comment"
    # NOT a comment: `#` inside a word, and `#` inside a quoted string. The
    # first is how a colour or a fragment is spelled; the second is ordinary
    # shell, and `run:` bodies in this workflow are shell.
    assert strip("url: https://example.test/x#frag") == "url: https://example.test/x#frag"
    assert strip("""echo "a # b" """.rstrip()) == 'echo "a # b"'
    assert strip("""echo 'a # b' # tail""") == "echo 'a # b'"
    # a TAB opens a comment exactly as a space does — legal in YAML after a
    # scalar and legal in the shell — and dropping `\t` from the rule left
    # every reader in `_lanes.py` green with the permissive reading back
    assert strip("extras: jax\t# solvers come in with jaxfluids") == "extras: jax"
    # and a `#` at column 0 wins over any later quote, which is what makes a
    # comment line carrying an apostrophe safe
    assert strip("# pytest's own exit code") == ""
    # `_code_lines` KEEPS THE EMPTIED LINE, which is the property its own
    # docstring states — "the list still indexes like the file" — and which
    # nothing in this module indexes today. Dropping the emptied lines instead
    # leaves every reader in `_lanes.py` green, so the claim is either pinned
    # here or it is decoration.
    assert _lanes._code_lines("a: 1\n# whole-line\nb: 2  # tail\n") == [
        "a: 1", "", "b: 2",
    ], "a comment line vanished instead of becoming an empty one"


def test_the_lane_reader_reads_ONE_FILE_however_its_lines_end():
    """:func:`_lanes._code_lines` splits on every break, and that is checked.

    THE SHAPE THIS IS A FENCE AGAINST is the one
    `tests/test_tripwire_record.py` met three times: a check written
    `^`-anchored under ``re.M``, where Python's ``^`` matches after a newline
    and NOT after a carriage return, whose correctness then rests on
    ``read_text()`` having translated the file's line breaks on the way in.
    That is a property of how the file was OPENED, not of the check.

    This module is not exposed to it, and this test is why that is a
    measurement rather than a hope: every pattern here is matched against one
    line of ``_code_lines``, and ``str.splitlines()`` breaks on CR, CRLF,
    U+0085, U+2028 and U+2029 as well as on LF. Re-render `ci.yml` with each
    of those in place of every newline and THE SAME blocks and THE SAME lanes
    come back — which is the claim, and it is why the assertion below compares
    each rendering against the clean read rather than against a number. (This
    sentence said *"the same eleven blocks and eight lanes"*, then *"13 and
    10"*; re-measured 2026-08-28 at the commit that added the sdist lane it is
    14 and 11, and it will be something else the next time a lane is added.
    The equality is the property; the size is not.)
    ``split("\\n")`` would pass every other test in this file and fail this
    one.
    """
    text = _lanes.CI.read_text(encoding="utf-8")
    assert "\r" not in text, "this test re-renders the breaks and needs LF in"

    def read(rendered: str):
        blocks = _lanes._blocks(_lanes._code_lines(rendered))
        return blocks.keys(), tuple(sorted(
            (job, dataclasses.astuple(lane))
            for job, lane in ((job, _lanes._classify(body))
                              for job, body in blocks.items())
            if lane is not None
        ))

    jobs, lanes = read(text)
    assert len(jobs) > 1 and len(lanes) > 1, (
        f"this test watches nothing unless the clean file reads as several "
        f"jobs and several lanes: {sorted(jobs)}, {len(lanes)} lanes"
    )
    for label, brk in (("CRLF", "\r\n"), ("CR", "\r"), ("U+0085", "\u0085"),
                       ("U+2028", "\u2028"), ("U+2029", "\u2029")):
        assert read(text.replace("\n", brk)) == (jobs, lanes), (
            f"`ci.yml` re-rendered with {label} in place of every newline "
            f"does not read as the same jobs and the same lanes, so what "
            f"this repository believes CI runs depends on how the file's "
            f"lines happen to end"
        )


def test_a_COMMENT_cannot_change_what_a_LANE_INSTALLS():
    """The third site of the same strip: :func:`_lanes._code_lines`.

    It dropped whole-line comments and nothing else, so every pattern
    ``_classify`` runs — the series pin, the matrix recogniser, the install
    line — was reading the workflow's prose as well as its code. This file is
    mostly prose, on purpose, and prose is where a jax version gets QUOTED
    while being discussed.

    Driven on a job that floats: a comment mentioning a pinned requirement
    makes it read as a lane that pins the series, which is exactly the claim
    ``test_every_tested_series_has_a_lane`` exists to hold.
    """
    job = """\
  test-jax:
    steps:
      - run: uv pip install -e ".[solvers,jax]" pytest{comment}
      - run: python -m pytest -q
"""

    def read(comment: str):
        text = job.format(comment=comment)
        body = _lanes._blocks(_lanes._code_lines(text))["test-jax"]
        return _lanes._classify(body)

    assert read("").jax == "floating"
    assert read('   # NOT "jax>=0.11,<0.12" — the ceiling is what pins').jax == "floating", (
        "a version RANGE written in a comment read as this job's pin"
    )
    assert read("   # installs from .[${EXTRAS}] in the reproducer lane").jax == "floating", (
        "a comment mentioning a matrix expansion read as this job BEING one"
    )
    # the positive half: the same text in the code still reads
    assert read(' "jax>=0.11,<0.12"').jax == "0.11"


def test_a_COMMENT_cannot_narrow_a_whole_suite_lane():
    """The conservative direction of the same missing strip, pinned anyway.

    ``python -m pytest -q  # the whole tree`` read ``the``, ``whole`` and
    ``tree`` as path arguments, so a whole-suite step read as a narrowed one.
    That under-credits rather than over-credits — it would have gone red, not
    silently green — but a fence whose reading a comment can move is a fence
    that a reflow of this workflow's prose can turn off, and this file is
    mostly prose on purpose.
    """
    commented = [
        line.replace("python -m pytest -q", "python -m pytest -q  # the whole tree")
        for line in _MATRIX_JOB.format(entries='          - extras: "solvers"\n').splitlines()
    ]
    assert _lanes._classify(commented).whole_suite is True
    # the negative half: a real path argument still narrows it
    narrowed = [
        line.replace("python -m pytest -q", "python -m pytest -q tests/test_lanes.py")
        for line in _MATRIX_JOB.format(entries='          - extras: "solvers"\n').splitlines()
    ]
    assert _lanes._classify(narrowed).whole_suite is False


def _reproducer_body() -> list[str]:
    blocks = _lanes._blocks(
        _lanes._code_lines(_lanes.CI.read_text(encoding="utf-8"))
    )
    return blocks["acceptance-reproducer"]


_MATRIX_JOB = """\
    strategy:
      matrix:
        include:
{entries}
    steps:
      - name: install
        env:
          EXTRAS: ${{{{ matrix.extras }}}}
        run: |
          uv pip install -e ".[${{EXTRAS}}]" pytest
      - run: python -m pytest -q
"""

#: A matrix job whose install line names a variable no ``env:`` binds to a
#: matrix key — the chain broken at its second link.
_UNREADABLE_MATRIX = _MATRIX_JOB.format(
    entries='          - extras: "solvers"\n'
).replace("EXTRAS: ${{ matrix.extras }}", "EXTRAS: solvers").splitlines()

#: A matrix job that interpolates the matrix value straight into the install
#: line, naming no shell variable at all — broken at the FIRST link.
_DIRECT_INTERPOLATION = (
    _MATRIX_JOB.format(entries='          - extras: "solvers"\n')
    .replace('".[${EXTRAS}]"', '".[${{ matrix.extras }}]"')
    .splitlines()
)

#: A matrix job with no ``include:`` block at all — the THIRD link.
_NO_INCLUDE_BLOCK = [
    line
    for line in _MATRIX_JOB.format(entries='          - extras: "solvers"\n').splitlines()
    if "include:" not in line and "- extras:" not in line
]

#: Two entries, and the second does not carry ``extras`` — the FOURTH link.
#: The entries themselves parse, which is what makes this link 4 and not 3.
_KEY_MISSING_FROM_AN_ENTRY = _MATRIX_JOB.format(
    entries='          - extras: "solvers"\n          - series: "0.10"\n'
).splitlines()

#: An entry whose value is a nested LIST. No ``key: value`` on the item line,
#: so the block is refused — the half of the nesting story that was already
#: true.
_BARE_LIST_ITEM = _MATRIX_JOB.format(
    entries='          - extras:\n              - solvers\n'
).splitlines()

#: A job with both halves of the verdict channel — the ``env:`` binding that
#: makes ``tests/conftest.py`` write the file, and the assertion that fails the
#: step on anything but ``verdict=made``. Trimmed from ``test-no-jax``.
_VERDICT_JOB = """\
    steps:
      - name: pytest, with the skip-inventory verdict asserted off a file channel
        env:
{binding}
        run: |
          set -euo pipefail
          .venv/bin/python -m pytest -q -ra
          verdict="${{RUNNER_TEMP}}/skip-inventory-verdict.txt"
{assertion}
"""

_BINDING = "          STELLING_SKIP_INVENTORY_VERDICT: ${{ runner.temp }}/v.txt"
_ASSERTION = """\
          if ! head -1 "${verdict}" | grep -qx 'verdict=made'; then
            exit 1
          fi\
"""

_VERDICT_BOTH = _VERDICT_JOB.format(
    binding=_BINDING, assertion=_ASSERTION
).splitlines()
#: The ``env:`` binding kept, the assertion deleted — an ordinary refactor,
#: and no comment involved anywhere.
_VERDICT_BINDING_ONLY = _VERDICT_JOB.format(
    binding=_BINDING, assertion="          cat \"${verdict}\""
).splitlines()
#: The assertion kept, the binding gone: a grep against a file nothing writes.
_VERDICT_ASSERTION_ONLY = _VERDICT_JOB.format(
    binding="          UNRELATED: 1", assertion=_ASSERTION
).splitlines()
#: The step gutted, the variable's NAME surviving in the comment left behind.
_VERDICT_IN_A_COMMENT = _VERDICT_JOB.format(
    binding="          UNRELATED: 1",
    assertion="          cat log   # the STELLING_SKIP_INVENTORY_VERDICT channel was here",
).splitlines()

#: The binding kept, and the token `verdict=made` PRESENT but only as
#: something the step WRITES. Pins the `grep` half of `_VERDICT_ASSERTED`,
#: which nothing did: dropping `\bgrep\b.*` from that pattern left every job
#: below reading exactly as before.
_VERDICT_ECHOED_NOT_GREPPED = _VERDICT_JOB.format(
    binding=_BINDING,
    assertion="          echo 'verdict=made' >> \"${verdict}\"",
).splitlines()

#: The `env:` key present and bound to NOTHING. Pins the `\s*\S` half of
#: `_VERDICT_BOUND`, and it is the half that matters: `conftest.py`'s
#: `_write_the_verdict_somewhere_last_writer_wins_cannot_reach` returns early
#: on `if not destination`, so a null-valued key means the file is never
#: written at all — the "binds and nothing reads it" case this reading exists
#: to refuse, wearing the binding's own spelling.
_VERDICT_BOUND_TO_NOTHING = _VERDICT_JOB.format(
    binding="          STELLING_SKIP_INVENTORY_VERDICT:",
    assertion=_ASSERTION,
).splitlines()

#: The name MENTIONED mid-line, in a shell string, with the assertion intact
#: and no `env:` binding anywhere. Pins the `^\s*` anchor — the part of the
#: pattern that makes it a mapping KEY rather than an occurrence — which
#: `_VERDICT_IN_A_COMMENT` does not reach, because the comment strip catches
#: that one a step earlier.
_VERDICT_MENTIONED_MID_LINE = _VERDICT_JOB.format(
    binding="          UNRELATED: 1",
    assertion=(
        '          echo "set STELLING_SKIP_INVENTORY_VERDICT: yes" >> log\n'
        + _ASSERTION
    ),
).splitlines()

#: An entry with a nested MAPPING that repeats a key the entry already has.
#: This is the half that was NOT true: the scan is flat, so the nested
#: ``extras: solvers`` used to overwrite the entry's own ``extras: "jax"`` and
#: the job read as installing a solver extra.
_NESTED_KEY_SHADOWS_THE_ENTRY = _MATRIX_JOB.format(
    entries='          - extras: "jax"\n            with:\n              extras: solvers\n'
).splitlines()

#: An entry with NO ``extras`` OF ITS OWN whose nested mapping supplies one.
#: The shape the "only shape in which flattening can change an answer"
#: sentence said could not exist: every key the nesting contributes is NEW,
#: and one of them is the field this module reads. Both entries are written
#: this way so that :func:`_lanes._agreed` sees agreement rather than a
#: disagreement that would mask the reading.
_NESTED_KEY_SUPPLIES_A_NEW_ONE = _MATRIX_JOB.format(
    entries=(
        '          - series: "0.10"\n            with:\n              extras: solvers\n'
        '          - series: floating\n            with:\n              extras: solvers\n'
    )
).splitlines()

#: A nested SEQUENCE of mappings — the shape *"ALL NESTING IS REFUSED, BY
#: COLUMN"* was not true of. A ``- `` line matched :data:`_lanes._MATRIX_ITEM`
#: at any column, so a deeper one started a PHANTOM entry and reset the
#: entry's key column to its own rather than ending the read. Measured at
#: `844ba48`, on this exact body: three entries, where ``ci.yml``'s shape has
#: two — an entry list with a member the file does not have, which is what
#: :func:`_lanes._matrix_include`'s own *"never an entry list that happens to
#: be short"* forbids in the other direction.
_NESTED_SEQUENCE_OF_MAPPINGS = _MATRIX_JOB.format(
    entries=(
        '          - extras: jax\n            variants:\n'
        '              - extras: solvers\n                pin: ""\n'
        '          - extras: solvers\n            pin: "0.10"\n'
    )
).splitlines()

#: The same shape with the nesting carrying a key the real entries do not, so
#: that the phantom is the only entry that could supply one. Two entries at
#: `844ba48`, one of them invented.
_NESTED_SEQUENCE_OF_A_DIFFERENT_KEY = _MATRIX_JOB.format(
    entries=(
        '          - extras: solvers\n            variants:\n'
        '              - name: a\n'
    )
).splitlines()

#: The same key twice at the entry's OWN column — a duplicate key in one
#: mapping. Unreachable by the column rule above, which is why the
#: repeated-key check is still in front of the value.
_KEY_REPEATED_AT_THE_ENTRYS_OWN_COLUMN = _MATRIX_JOB.format(
    entries='          - extras: "jax"\n            extras: solvers\n'
).splitlines()


def _synthetic(extras_per_entry, quoted: bool = True):
    """One matrix job with these expansions, classified by the real parser.

    ``quoted=False`` writes the value bare, which is how ``ci.yml`` writes
    ``extras`` and is the only spelling in which a trailing ``#`` is a COMMENT
    rather than three more characters of the string.
    """
    fmt = '          - extras: "{}"\n' if quoted else "          - extras: {}\n"
    entries = "".join(fmt.format(e) for e in extras_per_entry)
    return _lanes._classify(_MATRIX_JOB.format(entries=entries).splitlines())


def test_no_lane_a_claim_rests_on_is_a_cant_tell():
    """The consumer side of the rule above: ``None`` must be HANDLED.

    A named can't-tell is only worth more than a guess if something refuses to
    proceed on it. These are the two lists in this module that say "this job
    delivers configuration X", and neither may rest on a job whose
    provisioning ci.yml does not state with one voice.

    THERE ARE TWO SPELLINGS OF THE CAN'T-TELL AND ONLY ONE WAS REFUSED.
    ``Lane.solvers`` says ``None``; ``Lane.jax`` says ``"matrix"``, because it
    is a ``str`` field and a matrix job's series is not read per-entry (see
    ``_lanes.py``'s docstring for why). This used to check the first only, so
    a series-bearing lane whose series ci.yml does not state would have got
    past here and died inside ``_lanes._newest`` on ``int("matrix")`` — a
    can't-tell reported as a crash in a helper.
    """
    by_job = {lane.job: lane for lane in _lanes.lanes()}
    for job in (*_lanes.SERIES_BEARING, *SUPPORTED.values()):
        # existence is the fence next door's; this one is about the READING
        if job not in by_job:
            continue
        assert by_job[job].solvers is not None, (
            f"{job} is credited with a definite configuration, but ci.yml "
            f"does not say with one voice whether it installs a solver extra "
            f"— a matrix whose entries disagree, or one this module cannot "
            f"read. Resolve it in the workflow or stop resting a claim on it."
        )
        assert by_job[job].jax != "matrix", (
            f"{job} is credited with a definite configuration, but its jax is "
            f"a matrix expansion, and `matrix` is Lane.jax's can't-tell — one "
            f"Lane cannot hold two series. Pin the series in the job or stop "
            f"resting a claim on it."
        )
    # and the helper refuses it too, rather than carrying it into `int()`
    matrix_lane = _lanes.Lane(
        job=_lanes.SERIES_BEARING[0], jax="matrix", solvers=True,
        whole_suite=True, random_order=False, verdict_channel=True,
    )
    with pytest.raises(ValueError, match="can't-tell"):
        _drive_lane_series((matrix_lane,))


def _drive_lane_series(fake_lanes):
    """``_lanes.lane_series()`` over a substituted reading of ci.yml."""
    original = _lanes.lanes
    try:
        _lanes.lanes = lambda: tuple(fake_lanes)
        return _lanes.lane_series()
    finally:
        _lanes.lanes = original


def test_every_whole_suite_lane_asserts_the_verdict_channel():
    """The skip inventory's completeness verdict, off a channel the exit code
    cannot be taken from — and the one lane that does without it is NAMED.

    Every whole-suite lane but one sets ``STELLING_SKIP_INVENTORY_VERDICT``
    and fails the step on anything but ``verdict=made``. That the exception
    does not was true and undisclosed; (this said *"six of the seven"* — see
    ``_lanes.Lane.verdict_channel`` for why the fraction is gone rather than
    bumped)
    :data:`_lanes.VERDICT_CHANNEL_EXEMPT` carries the argument, and this holds
    the exemption list to being a list rather than a habit.
    """
    missing = sorted(
        lane.job
        for lane in _lanes.lanes()
        if lane.whole_suite and not lane.verdict_channel
    )
    assert missing == sorted(_lanes.VERDICT_CHANNEL_EXEMPT), (
        f"whole-suite lanes without the verdict channel: {missing}, declared "
        f"{sorted(_lanes.VERDICT_CHANNEL_EXEMPT)}. A whole-suite lane that "
        f"asserts only pytest's exit code cannot tell a green session from "
        f"one whose completeness claim was never made — see "
        f"tests/conftest.py's `_write_the_verdict_somewhere_last_writer_wins_"
        f"cannot_reach`. Either add the channel to the job or add it here "
        f"with the reason."
    )
    for job, why in _lanes.VERDICT_CHANNEL_EXEMPT.items():
        assert why.strip(), f"{job} is exempted with no reason given"
    assert any(
        lane.verdict_channel for lane in _lanes.lanes()
    ), "no lane asserts the verdict channel at all, so this fence measures nothing"


def test_the_verdict_channel_reading_needs_the_BINDING_AND_THE_ASSERTION():
    r"""What ``Lane.verdict_channel`` claims, held to what it measures.

    Its docstring says the job *asserts* the verdict and the fence above says
    the lanes *"fail the step on anything but ``verdict=made``"*. The reading
    was one pattern for the variable's NAME, anywhere in the job body, and two
    ordinary edits walked through it — ``11 passed`` both times:

    * the ``grep -qx 'verdict=made'`` block deleted, the ``env:`` binding
      kept. **This one needs no comment at all**: it is what a refactor looks
      like, and the binding alone only makes ``tests/conftest.py`` WRITE a
      file. Nothing reads it.
    * the whole step gutted, the variable's name surviving in the comment left
      where it had been — the same missing comment-strip as everywhere else in
      ``_lanes.py``.

    Both halves are required now, and both directions are driven here, because
    a conjunction that is really a constant is the shape this whole batch is
    about.

    AND SO IS EACH CONJUNCT ON ITS OWN, WHICH THIS DID NOT DO AT FIRST. The
    four cases above pin the ``and``: drop either half of the reading and one
    of them moves. They do not pin either half's own SHAPE, and two mutations
    walked through all four with every matched string intact —

    * ``_VERDICT_ASSERTED`` reduced from ``\bgrep\b.*\bverdict=made\b`` to
      ``\bverdict=made\b``: a step that merely WRITES the token now asserts
      the verdict.
    * ``_VERDICT_BOUND`` reduced from ``...VERDICT:\s*\S`` to
      ``...VERDICT:``: a key bound to NOTHING now counts as the binding —
      and that is the worst of the four, because a null-valued key makes
      ``conftest._write_the_verdict_somewhere_last_writer_wins_cannot_reach``
      return early on ``if not destination``, so the file is never written.
      "Binds and nothing reads it" is exactly what the conjunction exists to
      refuse, and this is that case wearing the binding's spelling.

    The ``^\s*`` anchor had no control either. ``_VERDICT_IN_A_COMMENT``
    looks like one and is not: the comment strip removes that line before any
    pattern sees it, so the anchoring was never what refused it.
    """
    assert _lanes._classify(_VERDICT_BOTH).verdict_channel is True
    assert _lanes._classify(_VERDICT_BINDING_ONLY).verdict_channel is False, (
        "a job that binds STELLING_SKIP_INVENTORY_VERDICT and never reads the "
        "file it names asserts nothing"
    )
    assert _lanes._classify(_VERDICT_ASSERTION_ONLY).verdict_channel is False, (
        "a job that greps for verdict=made without binding the variable is "
        "reading a file conftest.py was never told to write"
    )
    assert _lanes._classify(_VERDICT_IN_A_COMMENT).verdict_channel is False, (
        "the channel survived in a comment, which is where a deleted step "
        "leaves its name"
    )
    assert _lanes._classify(_VERDICT_ECHOED_NOT_GREPPED).verdict_channel is False, (
        "a step that WRITES `verdict=made` into the file was read as one that "
        "asserts it; the `grep` is the assertion and nothing held the pattern "
        "to containing one"
    )
    assert _lanes._classify(_VERDICT_BOUND_TO_NOTHING).verdict_channel is False, (
        "`STELLING_SKIP_INVENTORY_VERDICT:` with no value was read as a "
        "binding. conftest.py returns early on an empty destination, so that "
        "job writes no file at all and the grep it runs is against nothing"
    )
    assert _lanes._classify(_VERDICT_MENTIONED_MID_LINE).verdict_channel is False, (
        "the variable's name MENTIONED mid-line was read as an `env:` mapping "
        "key; `^\\s*` is what makes the pattern a key rather than an "
        "occurrence, and only this drives it"
    )


def test_every_series_bearing_job_is_a_whole_suite_lane_that_exists():
    by_job = {lane.job: lane for lane in _lanes.lanes()}
    for job in _lanes.SERIES_BEARING:
        assert job in by_job, f"SERIES_BEARING names {job!r}, which ci.yml has not"
        assert by_job[job].whole_suite, (
            f"{job} is credited with delivering a series, but it does not run "
            f"the whole suite, so `test_doc_example` may never execute in it"
        )


def test_every_tested_series_has_a_lane():
    """THE RULE ``_optional.py`` STATES AND NOTHING ENFORCED.

    ``TESTED_JAX_SERIES`` is a claim about what CI runs. Today ``"0.11"`` is
    delivered by the FLOATING lane alone — no job pins it — so the entry is
    true only for as long as 0.11 is the newest jax there is.

    DRIVEN, the jax-0.12 scenario end to end. Bump the tuple to
    ``("0.10", "0.11", "0.12")``, which is what a maintainer does after
    ``test_tested_jax_series_is_silent`` and the tripwire's known-hash row have
    both gone red and been cleared the obvious way, and re-run::

        AssertionError: TESTED_JAX_SERIES claims a series no lane runs: 0.11 —
        no job pins it and the floating lane resolves 0.12. Add the lane (pin
        the series, the way `test-jax-0-10` does) or drop the entry; an entry
        with no lane is a claim, not a test.

    and, from the doc-hash inventory next door, in the same run::

        AssertionError: a documented query content hash is compared on NO
        tested jax lane … {'quickstart.md#0': 'quickstart.md:66'}
        The lanes resolve ('0.10', '0.12'); TESTED_JAX_SERIES claims
        ('0.10', '0.11', '0.12').

    WHAT IT LOOKS LIKE WITHOUT THIS, measured in the same tree: keyed on
    ``TESTED_JAX_SERIES`` the inventory recomputes ``quickstart.md#0`` as
    ``("0.11",)`` — unchanged, matching what ``EXPECTED_HASH_COVERAGE``
    declares — so that half is green while no lane runs 0.11 at all. Only
    ``harness-api.md#0`` moves, because it carries no stamp and is compared
    everywhere, and updating it is the obvious clearing the failure message
    itself asks for. That is B13's hole one level up: the escape's condition
    widening until it covers every lane there is.

    The remedy both messages ask for is a ``test-jax-0-11`` lane beside
    ``test-jax-0-10``, which is what makes the tuple true again.
    """
    lanes = _lanes.lane_series()
    orphaned = sorted(set(TESTED_JAX_SERIES) - set(lanes))
    assert not orphaned, (
        f"TESTED_JAX_SERIES claims a series no lane runs: {', '.join(orphaned)} "
        f"— no job pins it and the floating lane resolves "
        f"{_lanes._newest(TESTED_JAX_SERIES)}. Add the lane (pin the series, "
        f"the way `test-jax-0-10` does) or drop the entry; an entry with no "
        f"lane is a claim, not a test."
    )
    untested = sorted(set(lanes) - set(TESTED_JAX_SERIES))
    assert not untested, (
        f"a lane runs jax {', '.join(untested)}, which TESTED_JAX_SERIES does "
        f"not claim. Either the constant is behind the workflow or a pin is "
        f"pointed at a series nobody has verified."
    )


def test_the_series_derivation_is_not_vacuous():
    """The anti-vacuity control for :func:`_lanes.lane_series`.

    Without it the fence above passes for free the day the parser stops
    recognising a pin — every lane reads as ``"absent"``, the derived set is
    empty, and an empty set has no orphans to name.
    """
    lanes = _lanes.lanes()
    assert lanes, "no lanes were parsed at all, so nothing above measures anything"
    assert any(l.jax == "floating" for l in lanes), "the floating lane went missing"
    assert any(
        l.jax not in ("absent", "floating", "matrix") for l in lanes
    ), "no lane pins a series any more, so `lane_series` is one inference wide"
    assert _lanes.lane_series(), "the derived lane series is empty"
    # the pin regex must want a CEILING: a floor alone is not a pin, and
    # `.[jax]`'s requirement has a floor of 0.10 today
    assert _lanes._SERIES_PIN.search('"jax>=0.10,<0.11"')
    assert not _lanes._SERIES_PIN.search('"jax>=0.10"')
    # and the newest-series rule really orders numerically, not lexically
    assert _lanes._newest(("0.9", "0.10")) == "0.10"


#: The environments the project's install story tells a user they can have,
#: and which of the two optional pieces each provides. Every one of them must
#: have a whole-suite lane, or "stelling works in this configuration" is a
#: sentence nothing measures.
#:
#: THE FOURTH CELL IS DELIBERATELY ABSENT AND NAMED HERE. Neither jax nor a
#: solver is a configuration in which stelling can decide anything at all —
#: there is no trace to transcribe — so there is no verdict to check and no
#: lane. What that configuration DOES have to satisfy is that the package
#: imports and the CLI runs, and that is measured by
#: `tests/test_zero_dep_import_discipline.py` in a subprocess with both
#: blocked, in every lane there is.
SUPPORTED = {
    ("jax", "solvers"): "test-jax",
    ("jax", "no solvers"): "test-jax-no-solvers",
    ("no jax", "solvers"): "test-no-jax",
}


@pytest.mark.parametrize("config,job", sorted(SUPPORTED.items()))
def test_every_supported_configuration_has_a_whole_suite_lane(config, job):
    """The hole item 1 of this batch came from.

    ``pip install -e ".[jax]" --group dev`` — jax, no solvers — is what a
    contributor gets by following CONTRIBUTING.md without asking for the
    solver extra, and no job ran the suite in it. Measured in that exact
    environment on 2026-08-20, before the lane existed: **72 failed, 3796
    passed, 196 skipped**, every failure a test that needs a solver and never
    said so.
    """
    by_job = {lane.job: lane for lane in _lanes.lanes()}
    assert job in by_job, (
        f"the {config} configuration is supposed to be covered by {job!r}, "
        f"which ci.yml does not have"
    )
    lane = by_job[job]
    assert lane.whole_suite, f"{job} does not run the whole suite"
    assert lane.solvers is not None, (
        f"{job} is meant to be the {config[1]} lane and ci.yml does not say "
        f"with one voice which it is; see Lane.solvers"
    )
    assert lane.solvers == (config[1] == "solvers"), (
        f"{job} is meant to be the {config[1]} lane and ci.yml provisions it "
        f"the other way"
    )
    # BOTH FIELDS HAVE A CAN'T-TELL AND `jax`'s IS A STRING. `"matrix"` is not
    # a series; read through `!= "absent"` it silently means "has jax", which
    # is the permissive answer to a question ci.yml has not answered.
    assert lane.jax != "matrix", (
        f"{job} is meant to be the {config[0]} lane and ci.yml expands it from "
        f"a matrix whose entries this module cannot reduce to one series; "
        f"`matrix` is Lane.jax's can't-tell, not a reading"
    )
    assert _lanes.has_jax(lane) == (config[0] == "jax"), (
        f"{job} is meant to be the {config[0]} lane and ci.yml provisions it "
        f"the other way"
    )


def test_no_reader_asks_whether_a_lane_HAS_JAX_by_comparing_the_STRING():
    """`Lane.jax`'s can't-tell fails OPEN, so the spelling is what is closed.

    ``"matrix"`` is a TRUTHY STRING in a field whose no-jax sentinel is also a
    string, so ``lane.jax != "absent"`` answers YES for a job ci.yml has not
    answered for. The declined alternative does not help — ``None !=
    "absent"`` is True too — and a sentinel whose comparisons raise would
    break ``==``, ``in`` and ``repr`` for the four consumers that
    legitimately ask ``lane.jax == "matrix"``. **So the VALUE still fails
    open. That is stated in the field's own comment rather than fixed.**

    What can be closed is that nothing FORCED a consumer to refuse the
    can't-tell. Nine did, by hand, correctly; the tenth would not have to.
    :func:`_lanes.has_jax` raises on ``"matrix"`` and this refuses the bare
    comparison anywhere under ``tests/``, so reaching the permissive answer
    now costs deleting a test rather than forgetting a line.

    AST, not grep: a comparison is a shape, and `# lane.jax != "absent"` in a
    comment is prose about the rule rather than a use of it — which this
    file's own docstrings are full of.

    **THE EXEMPTION IS KEYED TO ONE PATH AND NOT TO A NAME**, and it took
    two goes to get there because the first go replaced one typeable name
    with another. It collected `FunctionDef`s called `has_jax` PER FILE, so
    any file under `tests/` licensed the bare comparison inside a function it
    chose to call that: appending
    `def has_jax(lane): return lane.jax != "absent"` to this very file gave
    **`1 passed`**, while the identical body in a function called anything
    else is **`1 failed`**. That was repaired to `path.name == "_lanes.py"`
    on 2026-08-22 — WHICH IS THE SAME DEFECT WITH A LONGER NAME, because the
    scan is an `rglob` and a basename is as typeable as a function name.
    Driven: `tests/helpers/_lanes.py` carrying that exact body is
    **`1 passed`**, and the same body in `tests/helpers/_other.py` is
    **`1 failed`**. It is `path == tests_dir / "_lanes.py"` now — the ONE
    file, not any file wearing its name. Still named rather than line-ranged
    INSIDE that file, so moving the accessor up `_lanes.py` changes nothing
    here; what may not move is which file it is in.

    **AND THE THIRD GO WAS THE NESTING.** `ast.walk` descends, so the
    exemption was every `has_jax` anywhere in `_lanes.py` and not the
    accessor: driven at `1f55eef`, `def _outer(): def has_jax(lane): return
    lane.jax != "absent"` appended to `tests/_lanes.py` is **`1 passed`**,
    and so is the same body as a method on a class there — while the
    identical body under a different module-level NAME is **`1 failed`**.
    It is `tree.body` now, so the exemption is the module-level definition
    and a nested one is an ordinary offender.

    **AND THE COMPARISON IS READ FROM EITHER SIDE.** `"absent" != lane.jax`
    reaches the same permissive answer and was invisible — the scan looked at
    `node.left` alone, so the reversed spelling passed anywhere in the tree
    (driven: `1 passed`). Both operand orders are now the same shape to this.
    **What it still cannot see, said rather than left to be found:** a local
    binding (`j = lane.jax` and then `j != "absent"`), the comparison built
    through `getattr(lane, "jax")`, and `"absent"` reached through a name
    rather than written as a literal. None is in the tree today; an AST shape
    scan is incomplete by nature and this is which incompleteness it has.
    """
    import ast

    from conftest import _in_a_pruned_directory

    tests_dir = pathlib.Path(__file__).resolve().parent
    offenders = []
    for path in sorted(tests_dir.rglob("*.py")):
        # THE SWEEP IS THE SUITE AND NOT THE DIRECTORY — 0.2.0 D14. This was
        # a bare `rglob`, so a stale copy under `tests/build/` or
        # `tests/.junk/` — directories pytest refuses to recurse into, and
        # which `pytest --collect-only` reaches 0 tests in — was reported as
        # an offender. Driven at `a431646`: the same body in both gives
        # `1 failed` naming both paths. A comparison in a file no invocation
        # can collect cannot answer a lane question for anybody, so the red
        # was about the developer's directory. Direction noted, because it
        # is not the same repair as `tests/test_prose_hygiene.py`'s: this is
        # an `assert not offenders` scan, so the unpruned walk could only
        # ever report MORE and was never a false GREEN.
        if _in_a_pruned_directory(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        # The accessor's own body is the one place the comparison belongs,
        # and that body is THE MODULE-LEVEL `def has_jax` of `_lanes.py` —
        # see the docstring for the two plants that made a name-keyed and
        # then a basename-keyed exemption travel, and for the third, which
        # is that `ast.walk` reaches every nesting level: a `has_jax`
        # defined INSIDE another function, or inside a class body, is not
        # the accessor and is as typeable as the other two were.
        exempt = {
            (n.lineno, n.end_lineno)
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "has_jax"
        } if path == tests_dir / "_lanes.py" else set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if any(lo <= node.lineno <= hi for lo, hi in exempt):
                continue
            # EITHER OPERAND ORDER. `lane.jax != "absent"` and
            # `"absent" != lane.jax` are one shape with the sides swapped and
            # answer the same forbidden question.
            operands = [node.left, *node.comparators]
            reads_jax = any(
                isinstance(n, ast.Attribute) and n.attr == "jax"
                for n in operands
            )
            names_absent = any(
                isinstance(n, ast.Constant) and n.value == "absent"
                for n in operands
            )
            if reads_jax and names_absent:
                offenders.append(
                    f"{path.relative_to(tests_dir.parent)}:{node.lineno}"
                )
    assert offenders == [], (
        f"these ask whether a lane has jax by comparing `Lane.jax` with "
        f"`'absent'`: {offenders}. `matrix` is a truthy string in that field, "
        f"so the comparison answers YES for a job whose provisioning ci.yml "
        f"does not state. Use `_lanes.has_jax(lane)`, which raises on the "
        f"can't-tell instead of guessing past it."
    )
    # ... and the accessor really does refuse it, rather than being a name
    # for the same comparison.
    absent = _lanes.Lane(job="j", jax="absent", solvers=None,
                         whole_suite=False, random_order=False,
                         verdict_channel=False)
    assert _lanes.has_jax(absent) is False
    assert _lanes.has_jax(dataclasses.replace(absent, jax="0.11")) is True
    with pytest.raises(ValueError, match="CAN'T-TELL"):
        _lanes.has_jax(dataclasses.replace(absent, jax="matrix"))


def test_exactly_one_lane_runs_in_randomised_order():
    """The unenumerated backstop, and why there is exactly one.

    Test order in this repository is deterministic file order in every lane,
    so an order-dependent failure is invisible BY CONSTRUCTION — which is how
    the state-pollution incident that `tests/_state_guard.py` documents
    survived two audit rounds. One lane shuffles.

    Not more than one, and not the merge-bearing lanes: a randomised lane is
    flaky by design and names a symptom rather than a culprit, so it belongs
    beside the state guard's inventory rather than instead of it. Its job is
    to find the pollution nobody enumerated; the inventory's job is to say
    who.
    """
    shuffled = [lane.job for lane in _lanes.lanes() if lane.random_order]
    assert shuffled == ["random-order"], (
        f"expected exactly the `random-order` lane to shuffle, got {shuffled}. "
        f"A second shuffled lane doubles the flakiness for no new information; "
        f"none at all and order-dependent pollution is invisible again."
    )
    for lane in _lanes.lanes():
        if lane.job in _lanes.SERIES_BEARING:
            assert not lane.random_order, (
                f"{lane.job} is a merge-bearing lane and must not shuffle"
            )


# ── SELECTING an interpreter, and REPORTING which one arrived ───────────────


def test_a_line_that_SELECTS_an_interpreter_is_still_caught_after_the_split():
    """The fail-closed direction of ``_lanes._interpreter_reading``.

    ``_OTHER_INTERPRETER_TOKEN`` used to be one pattern over two different
    acts. A line that PROVISIONS an interpreter and a line that PRINTS which
    one it got were both recorded as ``unreadable:``, and since no entry of
    ``EXPECTED_PYTHON`` carries such a reading, the honest repair the
    `random-order` lane needed — say in the log what was resolved — reddened
    the very table that exists to catch a second pin.

    Splitting the pattern is a change to a fail-closed reader, and the only
    direction that matters is the one where it must NOT open. So every
    selection spelling the module knows is driven here, each on its own and
    each in combination with a query, and the query spellings are driven as
    the control that shows the split did something.

    THE ARGUMENT FOR THE LINE BEING STRUCTURAL rather than a list of the steps
    this branch adds: the five selection alternatives name TOOLS AND KEYS THAT
    PROVISION and the two query alternatives are EXPRESSIONS WHOSE VALUE IS
    the running interpreter's version. Nothing here is keyed to a step's text.
    A query cannot decide the answer on its own — something has to consume it,
    and every consumer is itself a reading — which is the last group of
    assertions below.

    **AND THAT LAST GROUP BUILDS ITS COMBINATIONS OUT OF THE FIVE TOKENS,
    WHICH IS THE HALF ITS AUTHOR HAD THOUGHT OF AND NOT THE HALF THAT WAS
    OPEN.** ``export UV_PYTHON=…``, ``pyenv local …`` and ``uv venv --python
    …`` are each caught by a test EARLIER in ``_interpreter_reading`` than the
    query, so all three passed under a reading that decided the query branch
    with a ``search`` — and so did every OTHER selection spelling beside a
    query, as ``reporting``. A drive whose population is drawn from the set
    the code already enumerates cannot find the members the set is missing.
    ``test_a_selection_this_module_has_NO_TOKEN_FOR_is_still_caught_beside_a_query``
    is the population this one could not reach, and it is where the boundary
    is now driven; this test is kept because the five are still the spellings
    the SELECTION branch has to keep catching on its own.
    """
    reading = _lanes._interpreter_reading

    # SELECTION: every alternative, each still unreadable on its own.
    for line in (
        "      - uses: actions/setup-python@v5",
        "          python-version: '3.13'",
        "        run: export UV_PYTHON=3.13",
        "        run: pyenv local 3.13",
        "        run: sudo add-apt-repository ppa:deadsnakes/ppa",
    ):
        assert reading(line).startswith("unreadable:"), (
            f"{line!r} selects an interpreter in a spelling this module does "
            f"not parse, and must stay a named can't-tell: the README's "
            f"'exactly one job pins' rests on nothing else noticing"
        )

    # The `uv` branch, unchanged and still in front of both.
    assert reading("      - run: uv venv --python 3.12") == "pin:3.12"
    assert reading("      - run: uv venv") == "runner-default"
    assert reading('      - run: uv venv --python "${PY}"').startswith("unreadable:")

    # REPORTING: the control. Without these the split above is invisible and
    # this test would pass on the unsplit pattern.
    assert reading("      - run: .venv/bin/python --version") == "reporting"
    assert reading("      - run: python3 --version") == "reporting"
    assert reading(
        '        run: python -c "import sys; print(sys.version_info)"'
    ) == "reporting"

    # AND A LINE THAT DOES BOTH IS A SELECTION. This is the direction a
    # permissive split would have broken, and each of these carries a query.
    for line in (
        '        run: export UV_PYTHON="$(python3 --version)"',
        "        run: pyenv local 3.13 && python --version",
        '      - run: uv venv --python "$(python --version)"',
    ):
        assert reading(line).startswith("unreadable:"), (
            f"{line!r} both asks the interpreter its version and provisions "
            f"one; a spelling that could be either has to read as the "
            f"selection, which is the half that fails closed"
        )

    # a line with no interpreter in it at all is still nothing
    assert reading('      - run: uv pip install -e ".[jax]" pytest') is None
    # and a comment is not a line: the strip runs first everywhere
    assert _lanes._code_lines("      # then run python --version by hand\n") == [""]

    # THE TABLE MAY NOT CARRY A CAN'T-TELL, which is the property that makes
    # `unreadable:` a red rather than a row. Stated in `_lanes.py`'s own block
    # comment and held by nothing until here.
    unreadable = sorted(
        job
        for job, readings in _lanes.EXPECTED_PYTHON.items()
        for r in readings
        if r.startswith("unreadable:")
    )
    assert not unreadable, (
        f"EXPECTED_PYTHON declares an `unreadable:` reading for {unreadable}. "
        f"That is a spelling nobody has read; declaring it makes the "
        f"measured/declared pin agree about a can't-tell instead of reddening "
        f"on it."
    )


def test_the_lane_whose_configuration_nothing_pins_is_the_lane_that_reports_it():
    """`random-order` resolves both halves elsewhere, so it prints both.

    Its `uv venv` names no interpreter and its `.[solvers,jax]` carries no
    ceiling, so what it ran is decided by the runner image and by PyPI on the
    day. It printed neither, and a red run named the SEED — an order in an
    environment nobody recorded.

    The reporting is held here rather than only in ``EXPECTED_PYTHON`` because
    the table would still agree if the step were deleted from the job and the
    row deleted from the table in the same commit; this says which job has to
    have it and why.
    """
    reporting = _lanes.python_reporting()
    assert "ci.yml:random-order" in reporting, (
        "the `random-order` lane no longer prints the interpreter it was "
        "given. It is the one lane in this file whose interpreter AND whose "
        "jax are both resolved by somebody else on the day it runs, and a "
        "seed is not a reproducer without the configuration it was drawn "
        "against."
    )
    # and the jax half, which is the same defect in the other variable. Read
    # off the job body rather than off a step name, which a rename would move.
    body = _lanes._blocks(
        _lanes._code_lines(_lanes.CI.read_text(encoding="utf-8"))
    )["random-order"]
    assert any("jax.__version__" in line for line in body), (
        "the `random-order` lane no longer prints the jax it resolved. Every "
        "other jax lane in ci.yml does; this one floats and is the one whose "
        "resolved version is least predictable."
    )


# ── the two lanes whose subject is the shape of the working tree ────────────


def test_every_reason_the_shallow_lane_expects_to_lose_is_disclosed():
    """:data:`_lanes.SHALLOW_CLONE_LOSES` against the file that owns the words.

    The set is REASON STRINGS, and those strings are literals in
    ``tests/test_skip_inventory.py``'s ``RULES``. A reason reworded there and
    not here would leave the shallow lane comparing against a set nothing can
    ever produce — and its own "declared but not lost" half would then fail on
    a runner, a day late and in the wrong file. This is the same check a
    commit ahead of the lane.

    Each of these is also a rule whose CONDITION is computable, which is what
    makes the skip legitimate rather than merely disclosed: a rule with
    ``legitimate=None`` discloses a skip it cannot check the direction of, and
    a shallow clone's losses are exactly the ones git can be asked about.
    """
    import test_skip_inventory as inventory

    assert _lanes.SHALLOW_CLONE_LOSES, (
        "nothing is declared lost to a `--depth 1` clone, so the shallow lane "
        "compares against an empty set and passes for free"
    )
    for reason in sorted(_lanes.SHALLOW_CLONE_LOSES):
        rules = [rule for rule in inventory.RULES if rule.matches(reason)]
        assert len(rules) == 1, (
            f"{reason!r} is declared as a guarantee a shallow clone costs, and "
            f"tests/test_skip_inventory.py's RULES disclose it {len(rules)} "
            f"times. Zero means the wording moved and the shallow lane is now "
            f"waiting for a reason nothing emits."
        )
        assert rules[0].legitimate is not None, (
            f"the rule disclosing {reason!r} cannot check its own condition, "
            f"so a skip carrying it is disclosed but never held to being "
            f"legitimate. A shallow clone's losses are all git questions and "
            f"git can be asked."
        )


#: A `pytest -q -ra` log, trimmed to what :func:`_lanes.run_report` reads. The
#: real ones are two whole-suite runs on a runner; these are the shape.
_LOG = """\
....s.......s...
{skips}
{counts} in 12.34s (0:00:12)
"""


def _log(counts: str, skips: tuple[str, ...] = ()) -> str:
    return _LOG.format(counts=counts, skips="\n".join(skips))


_GIT_GATED = tuple(sorted(_lanes.SHALLOW_CLONE_LOSES))


def test_a_run_report_is_READ_and_a_log_it_cannot_read_is_not_agreement():
    """:func:`_lanes.run_report`, and the one way it must not fail.

    A log with no summary counts in it is a run that did not finish, a
    truncated file, or an invocation this function does not know. Answering
    "nothing" for any of those would make two such logs compare EQUAL, and
    both lanes that use this decide by comparing two logs — so the permissive
    answer is a green tick on an experiment that did not happen.
    """
    report = _lanes.run_report(
        _log("2394 passed, 191 skipped, 10 warnings",
             ("SKIPPED [8] tests/test_soundness_routing.py:897: " + _GIT_GATED[3],))
    )
    assert dict(report.counts) == {"passed": 2394, "skipped": 191, "warnings": 10}
    assert report.skipped == (
        (8, "tests/test_soundness_routing.py:897", _GIT_GATED[3]),
    )
    assert report.reasons == frozenset({_GIT_GATED[3]})

    with pytest.raises(ValueError, match="no pytest summary counts"):
        _lanes.run_report("this is not a pytest log\n")
    with pytest.raises(ValueError, match="no pytest summary counts"):
        _lanes.run_report("2394 passed, 191 skipped\n")  # no ` in <duration>`
    # a sentence that merely CONTAINS a count is not the summary line
    with pytest.raises(ValueError, match="no pytest summary counts"):
        _lanes.run_report("we saw 12 passed in that run, which was wrong\n")


def test_two_runs_of_one_commit_are_compared_by_OUTCOME_and_by_WHICH_SKIPPED():
    """:func:`_lanes.outcome_differences` — the venv lane's assertion.

    Both directions, because the whole point is that the two runs are supposed
    to be identical: a comparison that could only report a difference and
    never report agreement would fail the lane on every push, and one that
    could only report agreement is not a comparison.

    THE SKIP LINES ARE COMPARED AS WELL AS THE COUNTS, and that is not
    belt-and-braces. Two runs can report the same `N skipped` while skipping a
    DIFFERENT N tests, which is exactly the shape a working-directory-sensitive
    guard produces when one check starts skipping and another stops.
    """
    a_skip = "SKIPPED [1] tests/test_lanes.py:1: something"
    b_skip = "SKIPPED [1] tests/test_optional.py:1: something else"

    same = _log("10 passed, 1 skipped", (a_skip,))
    assert _lanes.outcome_differences(
        "clean", _lanes.run_report(same), "stray", _lanes.run_report(same)
    ) == []

    moved = _lanes.outcome_differences(
        "clean", _lanes.run_report(_log("10 passed, 1 skipped", (a_skip,))),
        "stray", _lanes.run_report(_log("9 passed, 2 skipped", (a_skip, b_skip))),
    )
    assert any("different outcomes" in line for line in moved), moved
    assert any(b_skip.split("] ")[1] in line for line in moved), moved

    # SAME COUNTS, DIFFERENT TESTS — the case the counts alone cannot see.
    swapped = _lanes.outcome_differences(
        "clean", _lanes.run_report(_log("10 passed, 1 skipped", (a_skip,))),
        "stray", _lanes.run_report(_log("10 passed, 1 skipped", (b_skip,))),
    )
    assert not any("different outcomes" in line for line in swapped), swapped
    assert len(swapped) == 2, swapped

    # `warnings` is the one word not compared, and it is measured that two
    # runs of one commit really do differ in it: the git-gated tests disclose
    # what they could not check by emitting one. See `_lanes._NOT_AN_OUTCOME`.
    assert _lanes.outcome_differences(
        "clean", _lanes.run_report(_log("10 passed, 1 skipped", (a_skip,))),
        "stray", _lanes.run_report(_log("10 passed, 1 skipped, 3 warnings", (a_skip,))),
    ) == []


def test_the_shallow_lane_refuses_a_loss_it_does_not_declare_AND_one_that_vanishes():
    """:func:`_lanes.shallow_clone_differences` — the shallow lane's assertion.

    Three legs, and the middle one is the anti-vacuity half a lane like this
    stands or falls on. A `--depth 1` clone is GREEN: it exits 0 and writes
    `verdict=made`, because every skip it causes is disclosed. So a lane that
    asserted an exit code would pass whether or not the checkout was shallow
    at all — and the failure mode of "the checkout quietly stopped being
    shallow" is that the declared losses stop happening, which is what the
    second leg refuses.
    """
    lost = tuple(
        f"SKIPPED [1] tests/test_soundness_routing.py:{n}: {reason}"
        for n, reason in enumerate(_GIT_GATED, start=1)
    )
    shallow = _lanes.run_report(_log("2394 passed, 191 skipped", lost))
    full = _lanes.run_report(_log("2407 passed, 178 skipped"))
    assert _lanes.shallow_clone_differences(shallow, full) == []

    # 1 — a loss nothing declares
    extra = _lanes.run_report(
        _log("2393 passed, 192 skipped",
             lost + ("SKIPPED [1] tests/test_lanes.py:1: needs a runner",))
    )
    bad = _lanes.shallow_clone_differences(extra, full)
    assert bad and all("needs a runner" in line for line in bad), bad

    # 2 — a declared loss that did NOT happen, which is what a checkout that
    #     stopped being shallow looks like from here
    bad = _lanes.shallow_clone_differences(full, full)
    assert len(bad) == len(_GIT_GATED), bad
    assert all("nothing was lost" in line for line in bad), bad

    # 3 — deepening the clone CREATED a skip, so the two runs differ in
    #     something the experiment does not control
    bad = _lanes.shallow_clone_differences(
        shallow,
        _lanes.run_report(
            _log("2406 passed, 179 skipped",
                 ("SKIPPED [1] tests/test_lanes.py:1: needs a runner",))
        ),
    )
    assert any("CREATED a skip" in line for line in bad), bad

    # and the counts are deliberately NOT compared here: these two runs are
    # SUPPOSED to disagree about how many tests ran.
    assert _lanes.shallow_clone_differences(
        _lanes.run_report(_log("1 passed, 400 skipped", lost)), full
    ) == []


#: SELECTING SPELLINGS THIS MODULE HAS NO TOKEN FOR, each written beside a
#: query on one line. **The population is chosen for what it is NOT**: not one
#: of them contains `setup-python`, `python-version`, `UV_PYTHON`, `pyenv` or
#: `deadsnakes`, so none is reachable by the ordering that
#: `test_a_line_that_SELECTS_an_interpreter_is_still_caught_after_the_split`
#: relies on. Under the `search` that decided the query branch until
#: :data:`_lanes._REPORTING_LINE` existed, every one of these read
#: ``reporting`` — an accepted reading of a line that pins an interpreter.
#:
#: They are not a new enumeration to be extended when somebody thinks of a
#: twentieth: the reading they drive is a TOTAL grammar, so what they
#: demonstrate is that a line carrying a second command is refused whatever
#: that command is. Adding one costs nothing and proves nothing new; the
#: value is in the variety of shapes — `&&`, `;`, a command-prefix
#: assignment, a redirect, a wrapper that takes the command as an argument.
_SELECTS_WITH_NO_TOKEN_THIS_MODULE_KNOWS = (
    '        run: conda create -y -n ci python=3.13 && python --version',
    '        run: micromamba create -n ci python=3.12 && python --version',
    '        run: mise use python@3.13 && python --version',
    '        run: asdf local python 3.11.9 && python --version',
    '        run: source /opt/py311/bin/activate && python --version',
    '        run: . /opt/py311/bin/activate; python --version',
    '        run: export PATH=/opt/python3.10/bin:$PATH && python --version',
    '        run: PATH=/usr/local/py313/bin:$PATH python --version',
    '        run: echo "/opt/python3.10/bin" >> "$GITHUB_PATH" && python --version',
    '        run: apt-get install -y python3.13 && python3 --version',
    '        run: update-alternatives --set python3 /usr/bin/python3.11 && python3 --version',
    '        run: ln -sf /opt/python3.13/bin/python3 /usr/local/bin/python && python --version',
    '        run: docker run --rm python:3.13 python --version',
    '        run: nix-shell -p python312 --run "python --version"',
    '        run: hatch run python --version',
    '        run: conda activate ci && python -c "import sys; print(sys.version_info)"',
    '        run: PATH=/opt/py/bin:$PATH python -c "import sys; print(sys.version_info)"',
    '        run: python --version && conda activate ci',
)

#: The other half, and it is what stops the fix above from being "refuse
#: everything". Each of these IS the question and nothing else, in the shapes
#: `ci.yml` writes: a `- run:` step, an inline `run:`, a bare line inside a
#: block scalar; the interpreter by path, by name, with a version suffix; the
#: question asked directly and through one `-c` program, in either quote.
_ASKS_AND_DOES_NOTHING_ELSE = (
    "      - run: python --version",
    "      - run: python3 --version",
    "      - run: python3.12 --version",
    "      - run: .venv/bin/python --version",
    "        run: ${RUNNER_TEMP}/venv/bin/python --version",
    "          /usr/bin/python3 --version",
    '        run: python -c "import sys; print(sys.version_info)"',
    "        run: python -c 'import sys; print(sys.version_info)'",
)


def test_a_selection_this_module_has_NO_TOKEN_FOR_is_still_caught_beside_a_query():
    """The half the split left open, and the population that can see it.

    THE DEFECT. Moving ``python --version`` and ``sys.version_info`` out of
    ``_OTHER_INTERPRETER_TOKEN`` and into a query reading was argued safe on
    the ground that ``_interpreter_reading`` tests the selection before the
    query, so a line doing both is decided by the selection. **That is true of
    a line whose selection is one of five spellings and false of every other
    one.** Under the ``search`` this replaces, a line reading

        conda create -y -n ci python=3.13 && python --version

    was ``reporting`` — an ACCEPTED reading, not a can't-tell — so a job
    could start pinning an interpreter with `EXPECTED_PYTHON` unmoved and the
    README's *"exactly one job pins an interpreter"* still green. The drive
    written with the split built its combinations out of the same five tokens
    and could not reach a single one of these.

    THE REPAIR IS A CHANGE OF QUESTION, not a longer list. "Does this line
    contain a selection I recognise?" is asked of an open set. "Is this line
    NOTHING BUT a question put to an interpreter?" is asked of a closed shape,
    and :data:`_lanes._REPORTING_LINE` decides it — so everything else,
    including every spelling nobody has thought of, falls to ``unreadable:``.

    Both directions are here. Refusing everything would satisfy the first
    assertion and destroy the reading, which is what the second is for.
    """
    reading = _lanes._interpreter_reading

    fell_through = []
    for line in _SELECTS_WITH_NO_TOKEN_THIS_MODULE_KNOWS:
        for token in ("setup-python", "python-version", "UV_PYTHON", "pyenv",
                      "deadsnakes"):
            assert token not in line, (
                f"{line!r} carries {token!r}, so it is reachable by the "
                f"selection branch and belongs in the other drive. This "
                f"population exists to be the one that branch cannot see."
            )
        assert _lanes._INTERPRETER_QUERY.search(line), (
            f"{line!r} carries no query at all, so it never reaches the "
            f"branch this test is about and proves nothing here"
        )
        got = reading(line)
        if got is None or not got.startswith("unreadable:"):
            fell_through.append(f"{line.strip()}  ->  {got}")
    assert not fell_through, (
        "these lines SELECT an interpreter and ask it its version on the same "
        "line, and this module read them as `reporting` — an accepted "
        "reading, so a job that started pinning would leave "
        "`python_provisioning()` unmoved:\n  " + "\n  ".join(fell_through)
    )

    reddened = [
        f"{line.strip()}  ->  {reading(line)}"
        for line in _ASKS_AND_DOES_NOTHING_ELSE
        if reading(line) != "reporting"
    ]
    assert not reddened, (
        "these lines are the question and nothing else, and the grammar "
        "refuses them — which is the way this fix turns into 'no lane may "
        "report at all' and puts the original defect back:\n  "
        + "\n  ".join(reddened)
    )


def test_the_reporting_step_ci_yml_ACTUALLY_HAS_still_reads_as_one():
    """The grammar against the real file, not against a specimen.

    ``_ASKS_AND_DOES_NOTHING_ELSE`` above is prose this repository does not
    ship. The line that matters is the one in `ci.yml`, and a grammar tightened
    until the specimens pass while the real step reads ``unreadable:`` would
    redden `EXPECTED_PYTHON` and be reverted by whoever met it — which is how a
    gate gets loosened again.

    Read off the file rather than restated: every line in `.github/workflows/`
    that carries a query at all, and what it reads.
    """
    misread = []
    reported = 0
    for path in sorted(
        p for p in _lanes.WORKFLOWS.iterdir() if p.suffix in (".yml", ".yaml")
    ):
        for line in _lanes._code_lines(path.read_text(encoding="utf-8")):
            if not _lanes._INTERPRETER_QUERY.search(line):
                continue
            got = _lanes._interpreter_reading(line)
            if got == "reporting":
                reported += 1
            else:
                misread.append(f"{path.name}: {line.strip()}  ->  {got}")
    assert reported, (
        "no line in `.github/workflows/` asks an interpreter its version any "
        "more, so this test watches nothing and `python_reporting()` is empty "
        "for a reason nobody wrote down"
    )
    assert not misread, (
        "a line in a shipped workflow asks an interpreter its version and "
        "this module does not read it as a report. Either the step grew a "
        "second command — in which case it also SELECTS and the reading is "
        "right — or `_lanes._REPORTING_LINE` no longer admits the shape this "
        "repository writes:\n  " + "\n  ".join(misread)
    )


def test_a_reporting_line_that_ALSO_SELECTS_moves_the_DECLARED_provisioning(
    tmp_path, monkeypatch
):
    """End to end, which is where the fail-open was actually reachable.

    A unit reading is not the claim. The claim is that
    ``test_the_workflows_provision_the_interpreters_they_are_declared_to`` and
    the README's count go RED when a lane starts choosing an interpreter, and
    under the ``search`` they did not: editing the `random-order` reporting
    line so that it ALSO prepends a directory to `PATH` left
    ``python_provisioning()`` BYTE-IDENTICAL, because the line still carried a
    query and nothing looked at the rest of it.

    So the mutation is run against the real workflow directory, copied and
    edited, with a control: the unedited copy has to reproduce
    ``EXPECTED_PYTHON`` exactly, or the mutation is not the only difference and
    the red below means nothing.
    """
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    query_lines = {}
    for path in sorted(
        p for p in _lanes.WORKFLOWS.iterdir() if p.suffix in (".yml", ".yaml")
    ):
        text = path.read_text(encoding="utf-8")
        (workflows / path.name).write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if _lanes._INTERPRETER_QUERY.search(_lanes._strip_comment(line)):
                query_lines[path.name] = line

    monkeypatch.setattr(_lanes, "WORKFLOWS", workflows)
    assert _lanes.python_provisioning() == _lanes.EXPECTED_PYTHON, (
        "the CONTROL failed: a byte-for-byte copy of `.github/workflows/` does "
        "not reproduce the declared provisioning, so nothing below is a "
        "measurement of the mutation"
    )
    assert query_lines, "no workflow line asks an interpreter its version"

    for name, line in query_lines.items():
        path = workflows / name
        clean = path.read_text(encoding="utf-8")
        # The mutation the audit drove: the SAME question, on a line that now
        # also puts an interpreter of somebody's choosing in front of `PATH`.
        # It carries no token `_OTHER_INTERPRETER_SELECTION` knows.
        pinned = 'export PATH=/opt/python3.10/bin:$PATH && ' + line.lstrip()
        path.write_text(
            clean.replace(line, line[: len(line) - len(line.lstrip())] + pinned),
            encoding="utf-8",
        )
        try:
            measured = _lanes.python_provisioning()
        finally:
            path.write_text(clean, encoding="utf-8")
        assert measured != _lanes.EXPECTED_PYTHON, (
            f"{name}'s reporting line was edited to select an interpreter as "
            f"well as report one, and `python_provisioning()` did not move. "
            f"The measured/declared pin cannot see a lane that started "
            f"pinning, and the README's count of pinning jobs rests on it."
        )
        assert any(
            r.startswith("unreadable:")
            for readings in measured.values()
            for r in readings
        ), (
            f"{name}'s reading moved, but not to a named can't-tell. A line "
            f"this module cannot parse has to say so by name, because that is "
            f"what makes the declared table red rather than merely different."
        )
