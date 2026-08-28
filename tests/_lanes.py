# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""What CI actually runs, read off ``ci.yml`` instead of assumed.

THE DEFECT THIS ANSWERS, and it has three faces in one shape. A claim about
coverage is only worth the lane that delivers it, and this repository kept
making such claims against a *list* rather than against the workflow:

* ``_optional.TESTED_JAX_SERIES`` says which series are tested. Its own comment
  states the rule — *"an entry with no lane is a claim, not a test"* — and
  nothing enforced it. Today ``"0.11"`` is delivered by the FLOATING lane and
  by nothing else, so the day jax 0.12 ships and the tuple is bumped, 0.11 has
  no lane and the tuple still says it is tested.
* ``tests/test_doc_examples.py``'s ``EXPECTED_HASH_COVERAGE`` recomputes
  coverage over ``TESTED_JAX_SERIES`` while its failure text says *"compared on
  NO tested jax LANE"*. A series entry is not a lane, nothing forced the two to
  agree, and the difference is invisible until it matters.
* The configuration ``pip install -e ".[jax]" --group dev`` produces — jax, no
  solvers — had **no whole-suite lane at all**, so 72 tests could sit failing
  in it without anything going red. (Measured 2026-08-20 in that exact
  environment: 72 failed, 3796 passed, 196 skipped.)

So the lanes are read, and every claim that depends on one is checked against
what was read. ``tests/test_lanes.py`` holds the fences.

────────────────────────────────────────────────────────────────────────────
HOW IT PARSES, AND WHAT IT REFUSES TO GUESS
────────────────────────────────────────────────────────────────────────────

By text, not by a YAML parser: this suite has no yaml dependency and is not
acquiring one for a fence (``tests/test_release_gates.py`` reads
``release.yml`` the same way, for the same reason). Comments go first — this
workflow is more comment than code, deliberately — and what is left is scanned
for job headers, ``uv pip install`` lines and ``pytest`` invocations.

**AND A TEXT READER HAS TO SAY WHAT IT CANNOT SEE, WHICH IS WHAT THE NIGHTLY
CANARY'S READER LEARNED THE EXPENSIVE WAY.** `tests/test_tripwire_record.py`
read `.github/workflows/nightly-jax-canary.yml` with line-anchored patterns
and went past NINE legal spellings, measured: a quoted key, a flow mapping
and an alias — three ways of writing one ``env:`` mapping, each of which it
reported as *"no setting"* — plus ``matrix.exclude``, ``runs-on:`` written
before ``name:``, a job-level ``if:``, a ``defaults:`` block, ``set -a`` with
a sourced file, and ``continue-on-error: true``. It parses with ``yaml``
where PyYAML is importable and falls back to a TOTAL LINE GRAMMAR where it
is not, which is what lets it say *"there is no setting"* rather than only
*"I did not find one"*.

**THIS MODULE IS STILL A TEXT READER OVER A DIFFERENT FILE, and this
paragraph is not a claim that it is safe from the same thing.** What it has
that the nightly reader did not is :func:`_agreed` and the four-link
``${EXTRAS}`` chain, which produce a NAMED can't-tell rather than a
permissive default — so what is exposed here is the FIND layer, not the
value layer. Re-measured 2026-08-28 at the commit that added the sdist
lane, off PyYAML's own event stream rather than off a regex for `&` and
`*`: `ci.yml` carries **0 anchors, 0 aliases and 0 merge keys**, and
:data:`_JOB` finds EVERY job a real parse finds — 13 of them now, where this
sentence said *"12 of them then"* and, before that, *"all ten"* — plus
`push:` from under `on:`, which :func:`_classify` drops for having no
``python -m pytest`` line. The figure is the rotting half and the AGREEMENT
is the claim: what matters is that the two enumerations differ by that one
known extra and by nothing else, at whatever size the file has reached. That
gap is written down in `SWEEP-CARRY-FORWARD.md` rather than closed here.

**AND THE MEASUREMENT NAMED TWO FILES AND COVERED ONE.** The paragraph
above names ``release.yml`` as read the same way for the same reason, and
the anchor/alias/merge-key figure was taken on `ci.yml` alone — a narrower
measurement than the exposure it was disclosing. Taken now, the same way:
``release.yml`` carries **0 anchors, 0 aliases and 0 merge keys**, and its
three jobs are ``build``, ``publish`` and ``test``. (Both halves re-measured
the same way at the commit that added the sdist lane; `release.yml` has not
moved.) So nothing is wrong in either file today; what was wrong was a
figure standing for a file it had not been taken on.

**COMMENTS ARE STRIPPED IN ONE PLACE, :func:`_strip_comment`, AND THAT IS THE
POINT.** This module used to drop *whole-line* comments only, and a comment is
the cheapest thing there is to add to this file. Three readings were defeated
by a trailing one, each in the permissive direction and none of them noisy:

* ``extras: jax   # solvers come in with jaxfluids`` in both
  ``acceptance-reproducer`` entries — nothing else changed — read as an extras
  set literally containing the characters ``solvers``, so the lane was
  credited with a solver extra it does not install and
  ``tests/test_lanes.py`` stayed green (``11 passed``). The same edit WITHOUT
  the comment is caught.
* a job whose verdict-channel step was deleted kept the reading, because the
  token survived in the comment left where the step had been.
* ``python -m pytest -q  # the whole tree`` read ``the``, ``whole`` and
  ``tree`` as path arguments, so a whole-suite lane read as narrowed. That one
  is conservative — it under-credits — but it is the same missing strip.

The rule is YAML's and the shell's alike, which is why one function serves
both: ``#`` opens a comment when it *begins a word* — at the start of the
line, or after whitespace — and is not inside a quoted string. ``foo#bar`` and
a URL fragment are left alone.

**A JOB WHOSE INSTALL LINE IS A MATRIX EXPANSION IS EXPANDED, ENTRY BY ENTRY,
AND A FIELD ITS ENTRIES DISAGREE ON IS A NAMED CAN'T-TELL.**
``acceptance-reproducer`` installs ``.[${EXTRAS}]``, and one ``Lane`` cannot
describe two provisionings. So :func:`_matrix_values` follows the one chain
the install line actually names — ``${EXTRAS}`` to the step's
``EXTRAS: ${{ matrix.extras }}`` to the ``strategy.matrix.include`` entries —
and :func:`_agreed` resolves it: **the entries agree and the value is theirs,
or they do not and it is ``None``.** Today the two entries name
``solvers,jax`` and ``solvers``, which agree that a solver extra is installed,
so :attr:`Lane.solvers` reads ``True`` because that is what ci.yml says and
not because a matrix job is assumed to have everything.

**THAT RULE IS APPLIED TO EXACTLY ONE FIELD, AND SAYING OTHERWISE WAS THE
OVERCLAIM.** This paragraph used to read *"the resolution rule is the same for
every field"*. It is not: ``solvers`` goes through :func:`_agreed`, and
:attr:`Lane.jax` is the constant ``"matrix"`` written down the moment the
install line is recognised as an expansion — a SECOND SPELLING OF THE SAME
CAN'T-TELL, in a field whose other values are strings, and not a reading of
the entries at all. It is not read per-entry because the series is decided by
a *different* variable on the same install line (``${PIN}``, bound to
``matrix.pin``, empty for the floating entry), and following a second chain to
learn what ``"matrix"`` already says would be new inference for no new claim.
What the constant costs is that every consumer has to treat it as the
can't-tell it is: :func:`lane_series` REFUSES a ``"matrix"`` series-bearing
lane by name rather than crashing on it, and
``test_no_lane_a_claim_rests_on_is_a_cant_tell`` refuses both spellings —
``jax == "matrix"`` and ``solvers is None`` — for any job a claim rests on.

The two entries do disagree about the series — one pins ``0.10`` and one
floats — which is also why the job is excluded from :data:`SERIES_BEARING`.
That exclusion is a policy on top of the reading, not a substitute for it:
ci.yml's own text says the two acceptance jobs *must not* be required checks,
so a coverage claim resting on them rests on a job whose red does not block.

THIS WAS A HARDCODED ``True``, and the sentence here said *"nothing is
inferred from it"* while the code inferred the one thing a permissive constant
can: that the solvers were there. The measured/declared pin could not see it —
both sides were the same constant — so stripping ``solvers`` from **both**
matrix entries left the whole file green. ``design/lessons-ledger.md`` L23 is
about exactly that shape, and it was sitting inside the instrument that
carries L23. A can't-tell that resolves to the permissive answer is not a
reading.

**WHAT THE FLOATING LANE DELIVERS IS AN INFERENCE, and it is the only one
here.** ``test-jax`` installs ``.[solvers,jax]``, whose requirement carries no
upper bound, so which series it runs is decided by PyPI and is not readable
from this file. What IS readable is the constraint the suite already enforces:
``tests/test_transcribe.py::test_tested_jax_series_is_silent`` fails the moment
the resolved jax is outside ``TESTED_JAX_SERIES``, so a green floating lane ran
some entry of that tuple — and since a resolver picks the NEWEST release, that
entry is the newest series in it. Hence ``max(TESTED_JAX_SERIES)``.

The inference can be wrong in exactly one way, and it fails RED. If a series is
added to the tuple before it exists on PyPI, this module credits the floating
lane with the new series while the lane in fact still runs the old one — and
the old one is then credited to nobody, so
``test_every_tested_series_has_a_lane`` goes red and names it. That is the
correct outcome for adding an entry with no lane, which is the rule this exists
to hold.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from stelling._optional import TESTED_JAX_SERIES

REPO = pathlib.Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"

_JOB = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
_INSTALL = re.compile(r"uv pip install\b(.*)$")
_PYTEST = re.compile(r"python -m pytest\b(.*)$")
#: A pinned SERIES: `jax>=0.10,<0.11`. The floor names the series; the ceiling
#: is what makes it a pin rather than a floor, and a spec with no ceiling is
#: not a pin however high its floor.
_SERIES_PIN = re.compile(r'"jax>=(\d+\.\d+)[^"]*,\s*<\s*\d+\.\d+"')
#: An extras set that is a matrix expansion (`.[${EXTRAS}]`) rather than a
#: literal. Deliberately narrow: `property` installs `.[jax]` and a
#: `"${HYP}"` requirement read out of pyproject.toml, and a pattern that
#: matched any `${...}` classified that lane as a matrix and credited it
#: with solvers it does not have.
#:
#: RECOGNISING ONE IS NOT THE SAME AS FOLLOWING IT, and the two are separate
#: groups here for that reason. `.[$` alone decides "this is a matrix job" —
#: exactly the reach it has always had — while the capture is the shell
#: variable name, present for `${NAME}` and `$NAME` and absent for a direct
#: `${{ matrix.x }}` interpolation. No capture means the chain cannot be
#: followed, which is a can't-tell; a narrower recogniser would instead have
#: read such a job as an ordinary literal install with no solvers at all.
_MATRIX = re.compile(r"\.\[\$(?:\{(\w+)\}|(\w+))?")
#: `EXTRAS: ${{ matrix.extras }}` — a step's `env:` binding that shell
#: variable to a matrix key. This is the link the install line names and the
#: only reason this module can say which key to read.
_ENV_FROM_MATRIX = re.compile(r"^\s*(\w+):\s*\$\{\{\s*matrix\.(\w+)\s*\}\}\s*$")
#: The `include:` list header of a `strategy.matrix`.
_INCLUDE = re.compile(r"^(\s*)include:\s*$")
#: One `key: value` line inside a matrix entry, `- ` marking a new entry.
_MATRIX_ITEM = re.compile(r"^\s*(-\s+)?([A-Za-z_][\w-]*):\s*(.*?)\s*$")
#: THE VERDICT CHANNEL TAKES TWO LINES TO ASSERT, AND BOTH ARE REQUIRED. This
#: was one pattern matching the variable's NAME anywhere in a job body, while
#: :attr:`Lane.verdict_channel` said it measured whether the job *asserts* the
#: verdict. Two ordinary edits went straight through it:
#:
#:   the `verdict=made` assertion deleted, `env:` kept   ->  11 passed (missed)
#:   the step gutted, the name surviving in a comment    ->  11 passed (missed)
#:
#: The first needs no comment at all — deleting the check while leaving the
#: binding is what a refactor looks like — and it is the one that matters,
#: because the binding alone only makes ``tests/conftest.py`` WRITE the file.
#: Reading it and failing the step is the other line, and a job with the first
#: and not the second asserts nothing.
#:
#: `env:` binding, as a mapping key with a value — not a mention. BOTH HALVES
#: OF THAT SENTENCE ARE HELD by
#: `test_the_verdict_channel_reading_needs_the_BINDING_AND_THE_ASSERTION`, and
#: neither was: the fence pinned the CONJUNCTION while every string it matched
#: stayed intact under either conjunct being dropped. Driven, as mutations of
#: these two lines with the whole lane suite green afterwards:
#:
#:   `\bgrep\b.*\bverdict=made\b` -> `\bverdict=made\b`  the fence GREEN
#:   `...VERDICT:\s*\S`           -> `...VERDICT:`        the fence GREEN
#:
#: The second is the one that matters. A key bound to NOTHING —
#: `STELLING_SKIP_INVENTORY_VERDICT:` with no value — makes
#: `conftest._write_the_verdict_somewhere_last_writer_wins_cannot_reach`
#: return early on `if not destination`, so the file is never written: the
#: exact "binds and nothing reads it" case the second half exists to refuse.
#: The `^\s*` anchor is what separates a mapping key from a mention, and it had
#: no control either — `_VERDICT_IN_A_COMMENT` is caught by the comment strip
#: one step earlier, not by the anchoring.
_VERDICT_BOUND = re.compile(r"^\s*STELLING_SKIP_INVENTORY_VERDICT:\s*\S")
#: and the assertion the step fails on, which is the file's first line
#: compared against `verdict=made`. `grep` is required, not just the string:
#: a step that ECHOES `verdict=made` into the file asserts nothing about it.
_VERDICT_ASSERTED = re.compile(r"\bgrep\b.*\bverdict=made\b")


@dataclass(frozen=True)
class Lane:
    """One job in ``ci.yml``, in the terms a coverage claim needs."""

    job: str
    #: ``"absent"``, ``"floating"``, ``"matrix"``, or a series like ``"0.10"``.
    #:
    #: **THE CAN'T-TELL IS A TRUTHY STRING IN A FIELD WHOSE "NO" IS ALSO A
    #: STRING, AND THAT FAILS OPEN.** ``"matrix"`` means *ci.yml expands this
    #: job and its entries do not agree on a series*; read through the obvious
    #: ``lane.jax != "absent"`` it silently means *this job has jax*, which is
    #: the permissive answer to a question the workflow has not answered.
    #: There is no cheap spelling of the VALUE that fails safe: ``None`` reads
    #: the same way (``None != "absent"`` is True), and a sentinel whose
    #: comparisons raise would break ``==``, ``in`` and ``repr`` for the four
    #: consumers that legitimately ask ``lane.jax == "matrix"``. So the value
    #: still fails open, and that is stated rather than fixed.
    #:
    #: What is closed is the SPELLING. :func:`has_jax` is the way to ask, it
    #: raises on the can't-tell, and
    #: ``test_no_reader_asks_whether_a_lane_HAS_JAX_by_comparing_the_STRING``
    #: refuses the bare comparison anywhere under ``tests/`` — so a consumer
    #: added tomorrow cannot reach the permissive answer without deleting a
    #: test. Nine consumers refused ``"matrix"`` by name before that check
    #: existed, correctly and by hand, and nothing made the tenth.
    jax: str
    #: Whether the job installs a solver extra — or ``None``, the named
    #: can't-tell, when the workflow does not say with one voice. A matrix job
    #: whose expansions disagree, or whose expansions this module cannot read,
    #: reads ``None`` rather than the permissive ``True``; a consumer that
    #: needs a definite answer has to check for it, which is the whole
    #: difference between a can't-tell and a guess.
    solvers: bool | None
    #: Whether some step runs ``pytest`` over the WHOLE tree (no path argument).
    whole_suite: bool
    random_order: bool
    #: Whether the job asserts the skip inventory's verdict off the FILE
    #: channel (``STELLING_SKIP_INVENTORY_VERDICT``) rather than off pytest's
    #: exit code. BOTH HALVES ARE MEASURED — the ``env:`` binding that makes
    #: the file get written, and a ``verdict=made`` assertion in the same job
    #: that fails the step on anything else — because a job with the binding
    #: alone writes a file nobody reads. EVERY whole-suite lane BUT ONE has
    #: both, and the exception is named in :data:`VERDICT_CHANNEL_EXEMPT`
    #: with its reason. (This said *"six of the seven"*, and two lanes have
    #: been added since. A fraction of a population that grows every time a
    #: lane is added is a numeral beside something that moves; the fact it
    #: was standing for — all but the named one — does not.)
    verdict_channel: bool


def _strip_comment(text: str) -> str:
    """``text`` with a trailing ``#`` comment removed. THE ONE STRIP.

    Every reading in this module goes through here — the line scan below, the
    matrix values, the ``pytest`` argv — because they were written three times
    and were wrong three times, each in the direction that credits a lane with
    something it does not have. See the module docstring for the three drives.

    The rule is the one YAML and the shell share: ``#`` opens a comment when it
    begins a word — at the start, or after whitespace — and never inside a
    quoted string. So ``foo#bar`` survives, ``"a # b"`` survives, and
    ``value  # note`` becomes ``value``. Quoting is tracked by the first quote
    character seen, which is all a single line of either language needs.
    """
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip()
    return text.rstrip()


def _code_lines(text: str) -> list[str]:
    """``text``'s lines with every comment removed, indentation kept.

    A whole-line comment becomes an empty string rather than disappearing —
    every reader below already skips blanks, and keeping the line means the
    list still indexes like the file. NOTHING INDEXES IT TODAY, so that second
    half is a property rather than a use, and
    ``test_a_comment_is_stripped_the_same_way_everywhere`` pins it: dropping
    the emptied lines instead leaves every reader in this module green, which
    makes it exactly the stated-but-unheld shape this file is a fence against.

    **AND "LINE" IS ``str.splitlines()``'s, WHICH IS WHY THIS READER IS NOT
    EXPOSED TO THE THING THE NIGHTLY READER WAS.**
    `tests/test_tripwire_record.py` met the same question three times --
    ``^---``, ``^\\s*needs:`` and the shell reader's line -- and each time the
    check was `^`-anchored under ``re.M``, where Python's ``^`` matches after
    a newline and NOT after a carriage return. Every pattern in this module
    is matched against ONE LINE of this list instead, and ``splitlines()``
    breaks on CR, CRLF, U+0085, U+2028 and U+2029 as well as on LF.
    Re-measured 2026-08-28 on this repository's own ``ci.yml``, re-rendered
    with each of those five in place of every newline: **14 job blocks and 11
    pytest lanes, identical tuples, in all five renderings**. (This read
    *"13 job blocks and 10 pytest lanes"*, and before that *"11 job blocks
    and 8 pytest lanes … in all four renderings"*. The first two figures move
    every time a lane is added, which is what a count of a growing file does;
    the *"four"* was wrong when it was written — there are five break
    characters in the list it is counting, and
    ``test_the_lane_reader_reads_ONE_FILE_however_its_lines_end`` drives all
    five.) The counts are a record of a run. What does not rot is the
    EQUALITY: the same tuples come back however the lines end. So the
    ``read_text()`` above does not
    have to translate anything for this reader to be right, which is exactly
    what could not be said of the nightly reader's scans;
    ``test_the_lane_reader_reads_ONE_FILE_however_its_lines_end`` holds it,
    because ``split("\\n")`` here would be green until the day a workflow
    arrived over a wire.
    """
    return [_strip_comment(line) for line in text.splitlines()]


def _blocks(lines: list[str]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        m = _JOB.match(line)
        if m:
            current = blocks.setdefault(m.group(1), [])
            continue
        if current is not None:
            current.append(line)
    return blocks


def _matrix_include(body: list[str]) -> list[dict[str, str]]:
    """The ``strategy.matrix.include`` entries of one job, as key/value maps.

    Empty when there is no such block or its shape is not the one read here —
    an unreadable matrix is a can't-tell, never an entry list that happens to
    be short.

    WHAT A NESTED STRUCTURE ACTUALLY DOES HERE, because the sentence that used
    to be in this place named the wrong failure. It said a nested structure
    *"reads as unreadable rather than as a partial entry"*. It does not: this
    is a flat scan, so nesting FLATTENS, and there are two shapes of it —

    * a bare list item (``- solvers`` on its own line under a key) carries no
      ``key: value``, so :data:`_MATRIX_ITEM` does not match and the whole
      block is refused. That half was true.
    * a nested MAPPING flattens into the entry, and BOTH DIRECTIONS OF THAT
      CHANGE THE ANSWER. Driven: entry-level ``extras: "jax"`` with a nested
      ``extras: solvers`` two lines below read ``solvers: True`` — the inner
      one won — and an entry carrying NO ``extras`` OF ITS OWN whose nested
      mapping supplies one read ``solvers: True`` as well, from a key the
      entry does not have. Either way the field this module actually reads was
      decided by a structure this parser does not model.

    So ALL NESTING IS REFUSED, BY COLUMN: an entry's own keys are the ones at
    the column of the key on its ``- `` line, and an item line at any other
    column ends the read at ``[]``. Refusing the repeated-key shape alone was
    not enough, and the sentence that claimed it was — that a repeat is *"the
    only shape in which flattening can change an answer rather than merely add
    junk keys beside it"* — was false in the second case above. Column is the
    discriminator rather than a key check because YAML puts every key of one
    mapping at one column, so this refuses the shape itself instead of one
    symptom of it; ``ci.yml``'s own two entries are flat at one column and
    read unchanged.

    **AND THE ``- `` COLUMN, NOT ONLY THE KEY'S, because "ALL NESTING" was
    not true of a nested SEQUENCE.** A nested list of mappings puts a ``- ``
    at a deeper column, and a ``- `` line matched :data:`_MATRIX_ITEM`'s first
    group at ANY column — so it started a phantom entry and reset ``column``
    to its own, rather than ending the read. That violates this function's own
    *"never an entry list that happens to be short"*: the result was an entry
    list with members ci.yml does not have. Driven in four shapes; each reads
    ``[]`` now, ``EXPECTED_LANES`` recomputes equal, and the lane suite stays
    green.

    **NOT A HOLE BEFORE THE FIX**, which is why this is a claim repaired
    rather than a defect closed: :func:`_matrix_values` returns ``None``
    unless EVERY entry (real and phantom) carries the key, and a phantom's
    value is one that stands inside the nesting, so any definite answer was
    one the real entries already agreed on. ``.solvers`` read ``None`` in all
    four driven shapes. The sentence was wrong; the answer was not.

    The repeated-key check stays in front of the value all the same: a key
    repeated at the SAME column is a duplicate key in one mapping, which the
    column rule cannot reach and which is unreadable for its own reasons.

    Values are unquoted, and a trailing comment is stripped BEFORE unquoting
    (:func:`_strip_comment`) — ``extras: jax  # solvers arrive with jaxfluids``
    otherwise reads as an extras set containing ``solvers``.
    """
    entries: list[dict[str, str]] = []
    depth: int | None = None
    column: int | None = None
    dash: int | None = None
    for line in body:
        if depth is None:
            m = _INCLUDE.match(line)
            if m:
                depth = len(m.group(1))
            continue
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= depth:
            break  # dedented out of the include block
        item = _MATRIX_ITEM.match(line)
        # THE SECOND DISJUNCT IS DEFENCE IN DEPTH AND IS REDUNDANT IN EFFECT,
        # said here because the previous version of this comment did not say
        # it. `not entries` is true exactly when no `- ` line has been seen,
        # i.e. when `column is None`; a key-only line then falls to the
        # `elif` below, where `item.start(2) != None` is True and the read
        # ends at `[]` anyway. Driven: replacing the whole condition with
        # `if not item:` leaves the lane suite green and a differential over
        # 18 shapes distinguishes the two on 0 of 18. It stays because a
        # cheap guard in front of an `entries[-1]` is worth its line, not
        # because anything reaches it.
        if not item or (item.group(1) is None and not entries):
            return []  # a shape this cannot read
        if item.group(1) is not None:
            if dash is not None and item.start(1) != dash:
                return []  # a nested SEQUENCE: a `- ` at another column
            dash = item.start(1)
            entries.append({})
            column = item.start(2)  # this entry's own keys live here
        elif item.start(2) != column:
            return []  # nested under one of them, or otherwise not a key of it
        key = item.group(2)
        if key in entries[-1]:
            return []  # the same key twice at the entry's own column
        value = _strip_comment(item.group(3))
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        entries[-1][key] = value
    return entries


def _matrix_values(body: list[str], variable: str | None) -> list[str] | None:
    """What ``${variable}`` expands to in each matrix entry, or ``None``.

    Four links, and any of them failing is a can't-tell rather than a default:
    the install line has to name a shell variable, the step's ``env:`` has to
    bind it to a matrix key, the include block has to be readable, and every
    entry has to carry that key.
    """
    if variable is None:
        return None
    key = None
    for line in body:
        m = _ENV_FROM_MATRIX.match(line)
        if m and m.group(1) == variable:
            key = m.group(2)
            break
    if key is None:
        return None
    entries = _matrix_include(body)
    if not entries or any(key not in e for e in entries):
        return None
    return [e[key] for e in entries]


def _agreed(values, read):
    """``read`` applied to every expansion, if they all agree; else ``None``."""
    if values is None:
        return None
    seen = {read(v) for v in values}
    return seen.pop() if len(seen) == 1 else None


def _has_solvers(extras: str) -> bool:
    return bool(re.search(r"\bsolvers\b", extras))


def _classify(body: list[str]) -> Lane | None:
    installs = [m.group(1) for line in body if (m := _INSTALL.search(line))]
    pytests = [m.group(1) for line in body if (m := _PYTEST.search(line))]
    if not pytests:
        return None  # not a test lane at all (`reuse`, `dco`)

    joined = " ".join(installs)
    matrix = _MATRIX.search(joined)
    if matrix:
        # THE EXPANSION IS READ, NOT ASSUMED. See the module docstring: the
        # entries agree and the value is theirs, or they do not and it is the
        # named can't-tell. `jax` is "matrix" because the entries genuinely
        # disagree about the series and one `Lane` cannot hold two.
        jax = "matrix"
        variable = matrix.group(1) or matrix.group(2)
        solvers = _agreed(_matrix_values(body, variable), _has_solvers)
    else:
        pin = _SERIES_PIN.search(joined)
        if pin:
            jax = pin.group(1)
        elif re.search(r'\.\[[^"]*\bjax\b', joined):
            jax = "floating"
        else:
            jax = "absent"
        solvers = bool(re.search(r'\.\[[^"]*\bsolvers\b', joined))

    # A step with a path argument runs part of the tree; the whole-suite claim
    # needs a `pytest` with no path. Options and their values are dropped
    # first, so `-p no:randomly` is not mistaken for a path.
    whole = False
    for argv in pytests:
        words = []
        # The trailing comment goes first, by the same function the rest of
        # this module uses: `python -m pytest -q  # the whole tree` read
        # `the`, `whole` and `tree` as path arguments and made a whole-suite
        # lane read as narrowed.
        for word in _strip_comment(argv).split():
            # the shell's, not pytest's: a `| tee "$log"` or a `> file` ends
            # the command, and `tee` counted as a path argument once, which
            # read a whole-suite step as a narrowed one.
            if word in ("|", "||", "&&", ";", ">", "2>", ">>") or word.startswith(">"):
                break
            if word == "\\":
                continue
            words.append(word)
        paths = [
            w
            for i, w in enumerate(words)
            if not w.startswith("-") and (i == 0 or not words[i - 1].startswith("-p"))
        ]
        if not paths:
            whole = True
    random_order = any("-p randomly" in a for a in pytests) or any(
        "pytest-randomly" in i for i in installs
    )
    # BOTH LINES, not the name anywhere. See `_VERDICT_BOUND` above for the
    # two edits the one-pattern version missed.
    verdict = any(_VERDICT_BOUND.match(line) for line in body) and any(
        _VERDICT_ASSERTED.search(line) for line in body
    )
    return Lane(job="", jax=jax, solvers=solvers, whole_suite=whole,
                random_order=random_order, verdict_channel=verdict)


def lanes() -> tuple[Lane, ...]:
    """Every job in ``ci.yml`` that runs pytest, as read from the file."""
    found = []
    for job, body in _blocks(_code_lines(CI.read_text(encoding="utf-8"))).items():
        lane = _classify(body)
        if lane is not None:
            found.append(Lane(job, lane.jax, lane.solvers, lane.whole_suite,
                              lane.random_order, lane.verdict_channel))
    return tuple(sorted(found, key=lambda l: l.job))


#: THE MEASURED LANES, DECLARED. Same idiom as
#: ``tests/test_skip_inventory.py``'s pins and ``EXPECTED_HASH_COVERAGE``: the
#: right-hand side is recomputed from the workflow and compared, so a lane
#: added, removed or re-provisioned is a line in a diff rather than a silent
#: change in what CI measures.
#:
#: ``(jax, solvers, whole_suite, random_order, verdict_channel)``. ``solvers``
#: may be ``None`` — the named can't-tell — and no entry is ``None`` today,
#: which is a measurement rather than a coincidence: every job in ci.yml either
#: names its extras literally or has matrix entries that agree about them.
EXPECTED_LANES: dict[str, tuple[str, bool | None, bool, bool, bool]] = {
    "acceptance-any-pytree": ("floating", True, True, False, True),
    "acceptance-reproducer": ("matrix", True, True, False, True),
    "property": ("floating", False, False, False, False),
    "random-order": ("floating", True, True, True, False),
    # THE LANES WHOSE SUBJECT IS THE SHAPE OF THE WORKING TREE, and their
    # readings are byte-identical to `test-no-jax`'s. That is expected rather
    # than a duplicate: what distinguishes them is not the environment they
    # provision but the TREE they provision it in — a `.venv` in the checkout,
    # a `--depth 1` clone, and an unpacked sdist — and this reading has no
    # field for that and does not pretend to.
    #
    # (This sentence said "THE TWO LANES" and named two. `sdist-suite` is the
    # third, and it arrived one commit later. The count is gone rather than
    # bumped, for the reason `Lane.verdict_channel` gives about its own
    # fraction: a numeral over a population that grows every time a lane is
    # added is a defect with a delay fuse, and the fact it stood for — these
    # rows differ from `test-no-jax` in the tree and in nothing this table
    # reads — does not move.)
    #
    # THE OTHER THING THIS TABLE CANNOT SEE is how many runs a job makes.
    # `venv-in-the-working-tree` and `shallow-clone` each run the suite TWICE
    # and compare the two reports; `sdist-suite` runs it once, because a
    # distribution's suite EXITING 1 is visible in an exit code where a stray
    # `.venv` and a shallow clone are not. `whole_suite` means *some* step runs
    # the whole tree, never *exactly one* does.
    "sdist-suite": ("absent", True, True, False, True),
    "shallow-clone": ("absent", True, True, False, True),
    "test-jax": ("floating", True, True, False, True),
    "test-jax-0-10": ("0.10", True, True, False, True),
    "test-jax-no-solvers": ("floating", False, True, False, True),
    "test-no-jax": ("absent", True, True, False, True),
    "venv-in-the-working-tree": ("absent", True, True, False, True),
}

#: The whole-suite lanes that do NOT assert the skip-inventory verdict off the
#: file channel, each with the reason. Every other one does; this is the
#: exception, and it was an undisclosed asymmetry until it was written down.
#: (This read *"six of the seven"*; see :attr:`Lane.verdict_channel` for why
#: the fraction is gone rather than bumped.)
#:
#: ``random-order`` is not a required check and its green gates nothing — that
#: is the workflow's own stated policy for it — so the property the file
#: channel buys, *a verdict pytest's exit code cannot be taken from*, is
#: bought for a signal nobody merges on. What its red has to be instead is
#: ACTIONABLE, and the step is built around that: on failure it re-runs the
#: same commit in file order and annotates which KIND of failure this was.
#: And an order-dependent undisclosed skip — the failure this lane is most
#: likely to find — reddens `test_no_session_skip_is_undisclosed` in the
#: ordinary way, which the classification step then triages.
#:
#: THE MECHANICAL HALF OF THIS ARGUMENT WAS FALSE AND IS DELETED. It said a
#: bare `exit 1` "would sit in front of that classification". It would not:
#: the step's classification runs only on `status != 0`, behind
#: `if [ "${status}" -eq 0 ]; then exit 0; fi`, so a verdict check belongs in
#: the `status -eq 0` branch and could never reach it. What is left is the
#: cost/benefit half, which stands on its own: the channel's guarantee is
#: about a verdict a merge gate can trust, and nothing merges on this lane.
VERDICT_CHANNEL_EXEMPT = {"random-order": "not a required check; see the comment above"}

#: The lanes a documented-hash or series claim may rest on: whole-suite, and
#: not one of the two the workflow's own policy says must NOT be a required
#: check. Declared here rather than inferred from the prose beside those jobs —
#: a fence that reads a comment is a fence a reflow can turn off — and held to
#: the parse by ``test_lanes.py``.
#:
#: WHOLE-SUITE IS NECESSARY AND NOT SUFFICIENT, and the gap has a name.
#: ``test_every_series_bearing_job_is_a_whole_suite_lane_that_exists`` checks
#: that ``test_doc_example`` *may* run in each of these; it cannot check that a
#: particular block's comparison EXECUTES there. ``test_doc_example`` skips any
#: block whose source sets ``solver_timeout_ms`` when neither solver wheel is
#: installed, so on ``test-jax-no-solvers`` those blocks do not run — while
#: this tuple credits that job with a jax series for doc-hash coverage all the
#: same. Measured on this tree: 2 of the 30 collected blocks opt in
#: (``preconditions.md:63``, ``quickstart.md:160``) and NEITHER carries a
#: documented query hash, so no entry of ``EXPECTED_HASH_COVERAGE`` rests on a
#: skipped comparison; ``test-jax`` delivers the same 0.11 with the wheels
#: present in any case. The day a hash-bearing block opts into escalation,
#: this job's contribution to that hash becomes a skip and this tuple would
#: still credit it. Same shape as ``test-no-jax``, whose ``"absent"`` reading
#: ``lane_series`` already discards for exactly this reason.
SERIES_BEARING = ("test-jax", "test-jax-0-10", "test-jax-no-solvers", "test-no-jax")


def _newest(series: tuple[str, ...]) -> str:
    return max(series, key=lambda s: tuple(int(p) for p in s.split(".")))


def has_jax(lane: "Lane") -> bool:
    """Whether this job installs jax at all — and it RAISES on the can't-tell.

    The one question :attr:`Lane.jax` cannot be asked with ``!=``. See the
    field's own comment for why the value fails open and why no cheap
    spelling of it does not; this is the accessor that does, and the check
    named there is what keeps the bare comparison out of the tree.
    """
    if lane.jax == "matrix":
        raise ValueError(
            f"{lane.job} is expanded from a matrix whose entries this module "
            f"cannot reduce to one series, so whether it installs jax is a "
            f"CAN'T-TELL — `matrix` is Lane.jax's spelling of one. Ask "
            f"`lane.jax == 'matrix'` first and decide what a can't-tell means "
            f"for your claim; `lane.jax != 'absent'` answers YES here, which "
            f"is the permissive answer to a question ci.yml has not answered."
        )
    return lane.jax != "absent"


def lane_series() -> tuple[str, ...]:
    """The jax series the merge-bearing lanes actually resolve.

    This is what a coverage claim about "a lane" means. See the module
    docstring for why the floating lane contributes ``max(TESTED_JAX_SERIES)``
    and for the one direction in which that inference can be wrong.

    ``"matrix"`` IS REFUSED BY NAME. It is :attr:`Lane.jax`'s spelling of the
    can't-tell, and this function used to carry it into ``_newest``'s
    ``int(p)`` and die on ``ValueError: invalid literal for int() with base 10:
    'matrix'`` — a can't-tell reported as a crash in an unrelated helper, one
    line away from being reported as itself. ``test_lanes.py``'s
    ``test_no_lane_a_claim_rests_on_is_a_cant_tell`` reaches it first and says
    so in the workflow's terms; this is the backstop for every other caller.
    """
    by_job = {lane.job: lane for lane in lanes()}
    found = set()
    for job in SERIES_BEARING:
        lane = by_job.get(job)
        if lane is None:
            continue
        if lane.jax == "matrix":
            raise ValueError(
                f"{job} is credited with delivering a jax series, but ci.yml "
                f"expands it from a matrix whose entries do not agree on one "
                f"— `matrix` is Lane.jax's can't-tell. Pin the series in the "
                f"job, or drop {job!r} from SERIES_BEARING; a coverage claim "
                f"may not rest on a lane whose series the workflow does not "
                f"state."
            )
        if not has_jax(lane):
            continue
        found.add(_newest(TESTED_JAX_SERIES) if lane.jax == "floating" else lane.jax)
    return tuple(sorted(found, key=lambda s: tuple(int(p) for p in s.split("."))))


# ── which INTERPRETER a job gets, which is a different question from ────────
# which environment it provisions
#
# THE DEFECT THIS ANSWERS. `README.md:4` carried
# `![python: 3.12, the version CI measures](…badge/python-3.12%20tested…)` — a
# badge asserting a tested python — while `requires-python` declares `>=3.10`,
# exactly one job in `.github/workflows/` names an interpreter at all, and
# nothing tests the floor. The badge was therefore a claim about whatever
# `ubuntu-latest` happens to ship: approximately true, held by nothing, and
# quietly false the day the runner image moves. Three of the last four defects
# to redden `main` were environment-dependent in exactly that way.
#
# So the README's replacement paragraph is read back off the workflows, the
# same way :func:`lanes` reads what they install. WHAT IS READ IS THE
# INTERPRETER SELECTION, not the interpreter: which version a runner hands an
# unpinned `uv venv` is not in this file and this module does not pretend to
# know it — that unknowability is the thing the README now says out loud.
#
# AND SINCE A LANE MAY NOW *REPORT* THE VERSION IT GOT, THERE ARE TWO ACTS
# HERE AND NOT ONE. A step that prints `python --version` chooses nothing; it
# writes the runner's answer into the log, which is the only place that answer
# has ever existed. This module used to record such a line as a selection it
# could not parse — see :data:`_OTHER_INTERPRETER_SELECTION` for what that
# cost — and now reads it as ``reporting``. The unknowability is unchanged:
# what is readable HERE is still only the selection.
#
# ALL THREE WORKFLOWS, not `ci.yml` alone. The claim being held is "exactly one
# job pins an interpreter", and a pin added to the nightly canary or to
# `release.yml` would falsify it just as squarely.
#
# FAIL-CLOSED ON A SPELLING IT DOES NOT KNOW, and on a FILE or a LINE it
# cannot place. A reader that silently ignores `actions/setup-python`, or a
# `.yaml` workflow, or a line that falls outside every job header it
# recognises, would leave the README's count green while a second job pinned an
# interpreter — the permissive answer to a question the workflow HAS answered.
# Each of those three is recorded as `unreadable:…`, which no entry of
# :data:`EXPECTED_PYTHON` carries, so it fails the measured/declared pin by
# name rather than by absence.
WORKFLOWS = REPO / ".github" / "workflows"

#: `uv venv`, `uv run`, `uv sync` — the subcommands that can choose an
#: interpreter. `uv pip install` is not one: it installs INTO an environment
#: that already exists, so scanning it would record a `runner-default` reading
#: for a line that provisions nothing.
_UV_INTERPRETER_CMD = re.compile(r"\buv\s+(?:venv|run|sync)\b(.*)$")
#: `--python 3.12`, `--python=3.12`, `-p 3.12` on one of those commands.
_UV_PYTHON_FLAG = re.compile(r"--python[= ]\s*(\S+)|(?:^|\s)-p\s+(\S+)")
#: What such a flag has to look like for its value to be READ as a version.
#: `-p` is also pytest's plugin flag, so `uv run pytest -p no:randomly` would
#: otherwise be recorded as a pin on an interpreter called `no:randomly`. It
#: does not appear in this tree today; when it does, the reading has to be a
#: can't-tell and not a confident wrong answer.
_A_VERSION = re.compile(r"\d+(?:\.\d+)*|python\d(?:\.\d+)*")

#: Every OTHER way this repository could SELECT an interpreter. Recognising
#: one is not reading it: each is recorded as unreadable, so it has to be
#: looked at by a person before the README's count can go green again.
#: `python-version` covers `actions/setup-python`'s key, which is how a second
#: pin would most likely arrive.
#:
#: **THIS PATTERN USED TO CARRY TWO MORE ALTERNATIVES —
#: ``python3?\s+--version`` AND ``sys\.version_info`` — AND THEY ARE NOT THE
#: SAME ACT.** It was named ``_OTHER_INTERPRETER_TOKEN`` and its comment said
#: *"select or assert"*, which is two verbs for one reading: a line that
#: PROVISIONS an interpreter and a line that PRINTS which one it got were both
#: recorded as ``unreadable:``. The cost was not theoretical. `ci.yml`'s
#: `random-order` lane resolves both its interpreter and its jax by whatever
#: the runner and PyPI hand it that day and printed NEITHER, so a red run
#: named the seed and not the configuration it had shuffled — and the obvious
#: fix, a step that runs `python --version`, was read here as an interpreter
#: selection this module cannot parse and reddened
#: :data:`EXPECTED_PYTHON`, which carries no ``unreadable:`` reading for any
#: entry. The instrument made the repair it exists to ask for unavailable.
_OTHER_INTERPRETER_SELECTION = re.compile(
    r"setup-python|python-version|UV_PYTHON|pyenv|deadsnakes"
)

#: The other half of that pattern: the spellings whose VALUE IS the running
#: interpreter's version. `python --version` and `sys.version_info` are
#: QUERIES — they evaluate to the answer — where every alternative above names
#: a TOOL OR A KEY THAT PROVISIONS: `actions/setup-python` and its
#: `python-version:` key install one, `UV_PYTHON` is uv's own request
#: variable, `pyenv` chooses one, `deadsnakes` is the archive a runner adds to
#: get one. That is the structural line between the two sets, and it is a
#: property of what each token IS rather than of any line this repository
#: happens to write today.
#:
#: **WHY A QUERY CANNOT DECIDE THE ANSWER, which is the whole argument for
#: reading it as anything other than a can't-tell.** For a queried version to
#: change which interpreter a job GETS, something has to consume it — and
#: every consumer this module knows about is itself a reading. `uv venv
#: --python "${PY}"` reaches :data:`_UV_PYTHON_FLAG` with a value
#: :data:`_A_VERSION` refuses, so it is ``unreadable:``; `UV_PYTHON=…`,
#: `pyenv`, `setup-python` all carry a token from the set above. So capturing
#: a query into a variable moves the fail-closed decision onto the line that
#: USES it and does not lose it.
#:
#: **"AND A LINE THAT DOES BOTH IS ``unreadable:``, NOT REPORTING" WAS TRUE OF
#: FIVE SPELLINGS AND FALSE OF EVERY OTHER, AND THE DRIVE UNDER IT COULD NOT
#: SEE THAT.** What stood here said the three tests are ordered — the `uv`
#: command, then selection, then query — so that a line doing both is decided
#: by the selection, and named `uv venv --python "$(python --version)"`,
#: `export UV_PYTHON="$(python3 --version)"` and `pyenv local 3.13 && python
#: --version` as the evidence. Every one of those three carries a token from
#: the set ABOVE. **The ordering only ever protected lines whose selection is
#: one of those five**, and any other selection written beside a query fell
#: through to ``reporting`` — an accepted reading, not a can't-tell. The drive
#: was built out of the same five tokens, so it measured the half its author
#: had thought of; the open half was the half that was open. ELEVEN spellings
#: were driven against the two readers by the audit that found this —
#: `unreadable:` under the old pattern, `reporting` under the split — among
#: them `conda create -y -n ci python=3.13 && python --version`, `mise use
#: python@3.13 && python --version`, `source /opt/py311/bin/activate &&
#: python --version` and `export PATH=/opt/python3.10/bin:$PATH && python
#: --version`. The fence carries EIGHTEEN, in
#: `tests/test_lanes.py`'s `_SELECTS_WITH_NO_TOKEN_THIS_MODULE_KNOWS`, and
#: the number is not the point: they are shapes — `&&`, `;`, a command-prefix
#: assignment, a redirect, a wrapper taking the command as an argument — and
#: what refuses them is a grammar rather than their membership of a list.
#: End to end, a `random-order` reporting line edited to ALSO prepend a
#: directory to `PATH` left `python_provisioning()` byte-identical, so the
#: README's *"exactly one job pins an interpreter"* stayed green over a lane
#: that pins.
#:
#: **SO THE QUERY BRANCH IS NOT A ``search`` ANY MORE; IT IS A TOTAL LINE
#: GRAMMAR** (:data:`_REPORTING_LINE`), and the asymmetry is deliberate. "Does
#: this line contain a selection I recognise?" is a question about an OPEN set
#: and can only ever be answered with the members somebody has enumerated.
#: "Is this line NOTHING BUT a question put to an interpreter?" is a question
#: about a CLOSED shape and is decidable, so it is decided — and everything
#: the grammar does not cover is ``unreadable:`` rather than accepted. That is
#: the same move `tests/test_tripwire_record.py` made for the same reason:
#: a total line grammar is what lets a reader say *"there is no setting"*
#: rather than only *"I did not find one"*.
#:
#: **``sys.version_info`` IS THE WEAKER OF THE TWO AND IS READ AS REPORTING
#: ANYWAY; here is what that costs.** It is a READ of the interpreter already
#: running, so it cannot choose one — but a `if sys.version_info < (3, 11)`
#: GATE branches on it, and what such a branch then does is not modelled here.
#: What it cannot do is change which interpreter the job was given, and that
#: is the only question this section asks. A line that both branches on the
#: version and provisions from the branch carries a selection token on the
#: provisioning line, where this module reads it.
#:
#: **THE TRIGGER IS ``python[\d.]*``, NOT ``python3?``.** `python3.12
#: --version` is as much a
#: query as `python3 --version`, and under the old alternative it matched
#: NEITHER pattern and read ``None`` — invisible rather than accepted, so
#: not the fail-open this block is about, but a gap all the same, and one
#: :data:`_REPORTING_LINE` advertises by admitting the spelling its own
#: trigger could not fire on. Widening a trigger that feeds a total grammar
#: can only move lines OUT of ``None`` and into ``reporting`` or
#: ``unreadable:``, which is why it is safe to widen here and would not have
#: been under the ``search``.
_INTERPRETER_QUERY = re.compile(r"python[\d.]*\s+--version|sys\.version_info")

#: THE WHOLE LINE, and that is the point. A line reads ``reporting`` only when
#: it is an interpreter, the question, and nothing else — an optional `- ` and
#: `run:` in front, a trailing `\` behind, and no other command anywhere on
#: it. So `export PATH=/opt/python3.10/bin:$PATH && python --version` does not
#: match and is ``unreadable:``, and so is every other spelling that puts a
#: second command beside the question, whether or not this module has a token
#: for it. See :data:`_INTERPRETER_QUERY` for the finding that made this a
#: grammar rather than a search.
#:
#: WHAT IT ADMITS AND WHY EACH ONE IS THERE:
#:
#: * a PATH before the interpreter (`.venv/bin/python`,
#:   `${RUNNER_TEMP}/venv/bin/python`) — every step in `ci.yml` names its
#:   interpreter by path, and a bare `python` is the exception;
#: * `python3`, `python3.12` — the version suffix is part of the program name,
#:   not an argument;
#: * `-c` with ONE quoted program, because that is how the only reporting step
#:   in this repository asks the question — it prints the interpreter and the
#:   resolved jax in one line — and `--version` alone cannot;
#: * a trailing `\`, because that step continues into `| tee` on the next
#:   line.
#:
#: WHAT IT DOES NOT REACH, and the first of these is not the grammar's:
#:
#: * **THIS IS A LINE READER.** A selection on a DIFFERENT line from the query
#:   is invisible here, and no whole-line grammar can change that — `conda
#:   activate ci` on one line and `python --version` on the next reads
#:   ``(None, "reporting")``. That is the same exposure the selection branch
#:   above has had since it was written, and it is why that branch fails
#:   closed on the spellings it does know rather than pretending to know them
#:   all.
#: * **WHAT A ``-c`` PROGRAM DOES IS NOT MODELLED.** A program that prints
#:   `sys.version_info` and also shells out reads ``reporting``. It cannot
#:   change which interpreter THIS step got — it runs inside it — so what it
#:   could change is a LATER step, through `$GITHUB_PATH` or `$GITHUB_ENV`;
#:   and a line writing to either is invisible to this module already, on any
#:   line, query or no query. `ci.yml`'s `venv-in-the-working-tree` job
#:   contains exactly such a line (`echo … >> "$GITHUB_PATH"`) and it reads
#:   ``None`` today. That is the section's blind spot and not this grammar's,
#:   and it is named here rather than left to be found.
#: * a python program that prints the version on its OWN line inside a
#:   heredoc — `print(sys.version_info)` with the `python - <<'PY'` two lines
#:   up — reads ``unreadable:``, because a line reader cannot tell a heredoc
#:   body from a command. That is the fail-closed direction, and the one-line
#:   `-c` form above is the spelling that reads.
_REPORTING_LINE = re.compile(
    r"""^\s*                                # indentation
        (?:-\s+)?                           # the `- ` of a step, if any
        (?:run:\s*)?                        # `run:` when the command is inline
        (?:[\w./${}-]*/)?python[\d.]*       # the interpreter: a path, or bare
        \s+
        (?:--version                        # asked directly …
           |-c\s+(?:"[^"]*"|'[^']*'))       # … or through ONE quoted program
        \s*\\?\s*$                          # and nothing else on the line
    """,
    re.VERBOSE,
)


def _interpreter_reading(line: str) -> str | None:
    """This line's reading, or ``None`` if it neither selects nor reports one.

    ONE RULE, TWO CALLERS — the per-job scan and the per-file count below it.
    Written twice they could disagree, and the disagreement would read as
    "a line fell outside a job" rather than as a bug here.

    THE ORDER OF THE THREE TESTS MATTERS AND IS NOT WHAT MAKES THIS FAIL
    CLOSED. That sentence used to stand here on its own and it was wrong by
    the size of an open set: ordering the selection test in front of the query
    test protects a line only when its selection is one of the five spellings
    :data:`_OTHER_INTERPRETER_SELECTION` names. What decides the query branch
    now is :data:`_REPORTING_LINE`, a TOTAL grammar over the line: a line that
    mentions a query and is not entirely a query is ``unreadable:``, whatever
    else is on it and whether or not this module has heard of it.
    """
    cmd = _UV_INTERPRETER_CMD.search(line)
    if cmd:
        flag = _UV_PYTHON_FLAG.search(cmd.group(1))
        if flag is None:
            return "runner-default"
        version = flag.group(1) or flag.group(2)
        if _A_VERSION.fullmatch(version):
            return f"pin:{version}"
        return f"unreadable:{line.strip()}"
    if _OTHER_INTERPRETER_SELECTION.search(line):
        return f"unreadable:{line.strip()}"
    if _INTERPRETER_QUERY.search(line):
        # THE WHOLE LINE, NOT A SUBSTRING OF IT. See `_REPORTING_LINE`: a line
        # that mentions a query and carries anything else is a can't-tell,
        # which is what the `search` this replaced could not say.
        if _REPORTING_LINE.match(line):
            return "reporting"
        return f"unreadable:{line.strip()}"
    return None


def python_provisioning() -> dict[str, tuple[str, ...]]:
    """How every job in `.github/workflows/` gets its interpreter.

    Keyed ``"<workflow file>:<job>"``, because two files may hold a job of the
    same name. The values are readings, in file order:

    * ``"pin:<version>"`` — the job names the interpreter;
    * ``"runner-default"`` — it creates an environment without naming one, so
      the version is the runner image's and is not stated anywhere here;
    * ``"reporting"`` — it PRINTS which interpreter it got and chooses none.
      See :data:`_INTERPRETER_QUERY` for why that is a different act from the
      three above and for the direction in which the distinction fails closed;
    * ``"unreadable:<line>"`` — a spelling this module does not read. Never
      dropped: see the block above.

    A job that neither provisions nor reports an interpreter is absent rather
    than empty. One key is not a job: ``"<file>:<outside any job>"`` carries an
    ``unreadable:`` reading when the file holds interpreter lines that landed
    in no job block, which is the one way a pin could be invisible here rather
    than merely misfiled.

    THE FUNCTION'S NAME IS NARROWER THAN WHAT IT NOW RETURNS, and that is said
    here rather than repaired: a ``reporting`` reading is not a provisioning.
    It is kept in the same tuple because the per-file accounting below counts
    every line :func:`_interpreter_reading` answers for, and a reading dropped
    from the map would make that count disagree with itself and report a
    ``<outside any job>`` line that does not exist. :func:`python_pins` filters
    on ``pin:`` and is unaffected either way.
    """
    found: dict[str, tuple[str, ...]] = {}
    # BOTH EXTENSIONS. GitHub reads `.yml` and `.yaml` alike, this repository
    # happens to use one of them, and a reader that globs the habit rather than
    # the rule cannot see the first file that breaks it.
    for path in sorted(p for p in WORKFLOWS.iterdir() if p.suffix in (".yml", ".yaml")):
        lines = _code_lines(path.read_text(encoding="utf-8"))
        attributed = 0
        for job, body in _blocks(lines).items():
            readings: list[str] = []
            for line in body:
                reading = _interpreter_reading(line)
                if reading is not None:
                    readings.append(reading)
            if readings:
                attributed += len(readings)
                found[f"{path.name}:{job}"] = tuple(readings)
        # EVERY SUCH LINE IN THE FILE HAS TO LAND IN SOME JOB. `_blocks` finds a
        # job by a header pattern, and a line before the first header it
        # recognises is DROPPED — the one way a pin can be invisible here rather
        # than merely misfiled. So the file's own count is taken independently
        # and the difference is reported as what it is.
        present = sum(1 for line in lines if _interpreter_reading(line) is not None)
        if present != attributed:
            found[f"{path.name}:<outside any job>"] = (
                f"unreadable:{present - attributed} interpreter line(s) in this "
                f"file were not inside a job block this module can find",
            )
    return found


#: THE MEASURED INTERPRETER PROVISIONING, DECLARED — same idiom as
#: :data:`EXPECTED_LANES`. A job that starts pinning, stops pinning, or moves
#: its pin is a line in this diff, and the README paragraph that rests on it is
#: held to the reading rather than to this table.
EXPECTED_PYTHON: dict[str, tuple[str, ...]] = {
    "ci.yml:acceptance-any-pytree": ("runner-default",),
    "ci.yml:acceptance-reproducer": ("pin:3.12",),
    "ci.yml:property": ("runner-default",),
    # THE ONE JOB THAT REPORTS. It creates an environment without naming an
    # interpreter, like almost every other, and then PRINTS which one it got —
    # because it is the lane whose interpreter and jax are both resolved by
    # somebody else on the day it runs, and a seed is not a reproducer without
    # the configuration it was drawn against.
    "ci.yml:random-order": ("runner-default", "reporting"),
    "ci.yml:sdist-suite": ("runner-default",),
    "ci.yml:shallow-clone": ("runner-default",),
    "ci.yml:test-jax": ("runner-default",),
    "ci.yml:test-jax-0-10": ("runner-default",),
    "ci.yml:test-jax-no-solvers": ("runner-default",),
    "ci.yml:test-no-jax": ("runner-default",),
    # TWO readings, and the second is not a duplicate line: this job builds its
    # environment OUTSIDE the checkout and then plants a second, real venv IN
    # it, which is the whole experiment. Both are unpinned.
    "ci.yml:venv-in-the-working-tree": ("runner-default", "runner-default"),
    "nightly-jax-canary.yml:control": ("runner-default",),
    "nightly-jax-canary.yml:nightly": ("runner-default",),
    "release.yml:test": ("runner-default",),
}


def python_pins() -> dict[str, str]:
    """``{"<workflow>:<job>": "<version>"}`` for every job that names one."""
    return {
        job: reading.split(":", 1)[1]
        for job, readings in python_provisioning().items()
        for reading in readings
        if reading.startswith("pin:")
    }


def python_reporting() -> tuple[str, ...]:
    """``"<workflow>:<job>"`` for every job that PRINTS the interpreter it got.

    The other half of :func:`python_pins`. A lane whose interpreter and jax are
    both decided by somebody else's resolver on the day it runs has one honest
    obligation — say in its own log what it was handed — and this is what the
    README's paragraph about that is held to.
    """
    return tuple(
        sorted(
            job
            for job, readings in python_provisioning().items()
            if "reporting" in readings
        )
    )


# ── what a lane's two runs REPORTED, compared against each other ────────────
#
# THE DEFECT THIS ANSWERS. Two shapes of working tree were checked by hand at
# the 0.2.0 release and by no standing lane: a checkout carrying a `.venv`, and
# a `--depth 1` clone. Both are properties of the TREE rather than of the
# package set, so neither is expressible as another install line, and both are
# the shape this repository has repeatedly got wrong in one direction —
#
#   * `tests/test_sdist_reference_hygiene.py`'s header records a check whose
#     answer moved with whether somebody had run `uv venv` in the checkout: the
#     same commit, one `mkdir` apart, `17 passed` and then `2 failed, 15
#     passed`. A check whose SUBJECT is the repository may not depend on what a
#     developer's tooling left lying in it.
#   * the `fetch-depth: 0` comment standing at checkout after checkout in
#     `ci.yml` — eight of its twelve `actions/checkout` steps at the commit
#     that added these lanes, counted — exists because a depth-1 clone turns a
#     set of guarantees into SKIPS, and pytest exits 0 on a skip.
#
# Neither question is answerable by a single green tick, and both are answerable
# by the same instrument: RUN THE SUITE TWICE OVER ONE COMMIT IN ONE
# ENVIRONMENT, changing only the tree, and compare the two reports. That is the
# idiom `ci.yml`'s own header asks for one level up — *"a comparison between
# runs of ONE commit rather than against a constant written here"* — and it is
# why nothing below carries a pass count.
#
# WHAT THESE FUNCTIONS ARE NOT. They read a `pytest -q -ra` LOG, which is a
# rendering and not a data structure, and they are as good as that rendering:
# a pytest that changed the shape of its summary line would make
# :func:`run_report` raise rather than answer, which is the direction that
# fails closed, and a pytest that changed the shape of its `SKIPPED` lines
# would make every skip invisible to :attr:`RunReport.skipped` while the counts
# still parsed — that one does NOT fail closed, and it is why
# :func:`outcome_differences` compares the COUNTS as well as the lines.

#: One `-ra` short-summary skip line: `SKIPPED [8] tests/test_x.py:897: why`.
_SKIPPED_LINE = re.compile(r"^SKIPPED \[(\d+)\] (\S+): (.*)$")
#: The final counts line of a `-q` run: `2394 passed, 191 skipped, 10 warnings
#: in 120.06s (0:02:00)`. Anchored at both ends of the count list so that a
#: sentence merely CONTAINING `12 passed` cannot be read as the summary.
_RUN_COUNTS = re.compile(r"^(\d+ [a-z]+(?:, \d+ [a-z]+)*) in \d")

#: The one outcome word :func:`outcome_differences` does not compare, and it is
#: named rather than described. `warnings` is not an outcome: it counts what
#: the run PRINTED, and the git-gated tests disclose what they could not check
#: by emitting one. Measured 2026-08-28 at 9b5b496, the whole suite on the
#: zero-dep interpreter, one commit, two clones: the full clone reported
#: `2407 passed, 178 skipped` with no warnings summary and the `--depth 1`
#: clone `2394 passed, 191 skipped, 10 warnings`. Comparing it would make the
#: shallow lane's own subject read as a difference to be refused.
#:
#: WHAT THAT COSTS, said rather than left to be found: a change in warning
#: behaviour is invisible to both lanes below. The warnings themselves are in
#: the log — `-ra` prints the warnings summary — so a reader has them; nothing
#: compares them.
_NOT_AN_OUTCOME = "warnings"


@dataclass(frozen=True)
class RunReport:
    """What one `pytest -q -ra` log says happened, in comparable form."""

    #: `(("passed", 2394), ("skipped", 191), …)`, sorted, `warnings` included
    #: — the exclusion is made where the comparison is, not where the reading
    #: is, so a reader of a report still sees everything the run said.
    counts: tuple[tuple[str, int], ...]
    #: `((8, "tests/test_soundness_routing.py:897", "git cannot read …"), …)`,
    #: sorted.
    skipped: tuple[tuple[int, str, str], ...]

    @property
    def reasons(self) -> frozenset[str]:
        """The distinct skip REASONS, which is the granularity a rule has."""
        return frozenset(reason for _, _, reason in self.skipped)


def run_report(log: str) -> RunReport:
    """Read a `pytest -q -ra` log. RAISES when it cannot find the counts line.

    Absence is not an empty report. A log with no summary counts in it is a run
    that did not finish, a log that was truncated, or an invocation whose output
    this function does not know — and answering `{}` for any of those would make
    two such logs compare EQUAL, which is the permissive answer to a question
    nothing has answered.
    """
    counts: dict[str, int] | None = None
    skipped: list[tuple[int, str, str]] = []
    for line in log.splitlines():
        stripped = line.strip()
        found = _RUN_COUNTS.match(stripped)
        if found:
            # the LAST one: a nested session's summary can precede the outer
            # one, and it is the outer one that describes this run.
            counts = {}
            for part in found.group(1).split(", "):
                number, _, word = part.partition(" ")
                counts[word] = int(number)
        skip = _SKIPPED_LINE.match(stripped)
        if skip:
            skipped.append((int(skip.group(1)), skip.group(2), skip.group(3)))
    if counts is None:
        raise ValueError(
            "this log carries no pytest summary counts line, so what the run "
            "reported is not readable from it. A run that did not reach a "
            "summary is not a run that reported nothing."
        )
    return RunReport(
        counts=tuple(sorted(counts.items())), skipped=tuple(sorted(skipped))
    )


def outcome_differences(
    label_a: str, a: RunReport, label_b: str, b: RunReport
) -> list[str]:
    """Every way two runs of ONE commit disagree. Empty is the pass.

    Used by the lane whose subject is that a check's answer must not move with
    what a developer's tooling left in the tree: the two runs differ in the
    tree and in nothing else, so any difference here is the defect.
    """
    bad = []
    counts_a = {w: n for w, n in a.counts if w != _NOT_AN_OUTCOME}
    counts_b = {w: n for w, n in b.counts if w != _NOT_AN_OUTCOME}
    if counts_a != counts_b:
        bad.append(
            f"the two runs report different outcomes: {label_a} {counts_a}, "
            f"{label_b} {counts_b}"
        )
    only_a = sorted(set(a.skipped) - set(b.skipped))
    only_b = sorted(set(b.skipped) - set(a.skipped))
    for label, lines in ((label_a, only_a), (label_b, only_b)):
        for count, where, reason in lines:
            bad.append(f"skipped only in {label}: [{count}] {where}: {reason}")
    return bad


#: THE GUARANTEES A `--depth 1` CLONE TURNS INTO SKIPS, DECLARED BY THE REASON
#: EACH SKIP CARRIES. Same idiom as :data:`EXPECTED_LANES`: the shallow lane
#: measures the difference between a depth-1 run and a full-depth run of the
#: SAME commit in the SAME environment, and this is the declaration that
#: difference is compared against — so a guarantee newly lost to a shallow
#: clone, or one that stops being lost, is a line in this diff.
#:
#: REASONS AND NOT COUNTS, deliberately. The strings are literals in
#: `tests/test_skip_inventory.py`'s `RULES`, so they move only when somebody
#: edits a disclosure; the NUMBER of tests each reason covers moves whenever
#: anybody adds a test to one of those modules, and a numeral beside something
#: that moves is a defect with a delay fuse.
#: `tests/test_lanes.py::test_every_reason_the_shallow_lane_expects_to_lose_is_disclosed`
#: holds every one of these to that file, so a reason renamed there reddens
#: here rather than silently emptying this set.
#:
#: MEASURED 2026-08-28 at 9b5b496 — a `git clone --depth 1` of this repository
#: and a full clone of the same ref, the whole suite on an interpreter with
#: neither jax nor a solver wheel, `PY_COLORS=0`. The shallow run skipped 13
#: tests the full run did not, in exactly these four reasons and no others:
#: 8 in `tests/test_soundness_routing.py`, 3 in
#: `tests/test_prerelease_scope_from_the_tag.py`, 1 in
#: `tests/test_proposed_page_headers.py` and 1 in
#: `tests/test_soundness_log_reach.py`. Both runs exited 0 and both wrote
#: `verdict=made`, which is exactly why a lane that asserted only the exit code
#: or only the verdict would have measured nothing here.
SHALLOW_CLONE_LOSES = frozenset(
    {
        "git cannot read this tree's own history, so the routing manifest's "
        "source-side columns are unverified here",
        "git cannot read `v0.1.0:SOUNDNESS.md`, so the pre-release scope "
        "cannot be derived from the tag",
        "git cannot resolve `v0.1.0`, so no claim here about the tag's tree "
        "can be decided",
        "git cannot reach a commit a `proposed-*.md` status paragraph names, "
        "so that header's shipped-in claim cannot be decided here",
    }
)


def shallow_clone_differences(shallow: RunReport, full: RunReport) -> list[str]:
    """The depth-1 run against the same commit at full depth. Empty is the pass.

    Three claims, and the middle one is the anti-vacuity half:

    * every guarantee the shallow run lost is one :data:`SHALLOW_CLONE_LOSES`
      declares — *a shallow clone may skip the git-gated set and nothing else*;
    * every reason declared there really was lost — a lane that measured a set
      it had stopped producing would pass by having nothing to compare, and
      that is the failure mode a `--depth 1` that silently was not shallow
      produces;
    * deepening the clone did not itself create a skip. Nothing should get
      WORSE when the history arrives, and a difference in that direction means
      the two runs differ in something other than the depth.

    The counts are deliberately NOT compared here, which is the difference
    between this and :func:`outcome_differences`: the two runs are supposed to
    disagree about how many tests ran.
    """
    lost = shallow.reasons - full.reasons
    gained = full.reasons - shallow.reasons
    bad = []
    for reason in sorted(lost - SHALLOW_CLONE_LOSES):
        bad.append(
            f"a `--depth 1` clone skipped for a reason nothing declares it may "
            f"lose: {reason!r}. Either the history-gating spread to a check "
            f"that did not have it, or this is a skip that has nothing to do "
            f"with the depth. Add it to _lanes.SHALLOW_CLONE_LOSES with the "
            f"measurement, or fix the gate."
        )
    for reason in sorted(SHALLOW_CLONE_LOSES - lost):
        bad.append(
            f"nothing was lost to the shallow clone for the declared reason "
            f"{reason!r}. Either the guarantee stopped being git-gated — good "
            f"news, and this set has to say so — or this run was not shallow, "
            f"in which case the lane measured nothing."
        )
    for reason in sorted(gained):
        bad.append(
            f"deepening the clone CREATED a skip: {reason!r}. The two runs are "
            f"supposed to differ in the depth of the checkout and in nothing "
            f"else, so this is a difference the experiment does not control."
        )
    return bad
