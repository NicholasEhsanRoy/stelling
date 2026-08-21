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

    Six of the seven whole-suite lanes set
    ``STELLING_SKIP_INVENTORY_VERDICT`` and fail the step on anything but
    ``verdict=made``. That the seventh does not was true and undisclosed;
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
    assert (lane.jax != "absent") == (config[0] == "jax"), (
        f"{job} is meant to be the {config[0]} lane and ci.yml provisions it "
        f"the other way"
    )


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
