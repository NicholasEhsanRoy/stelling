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
value layer. Measured at this commit: `ci.yml` carries no anchor, no alias
and no merge key, and :data:`_JOB` finds all ten jobs a real parse finds
(and `push:` from under `on:`, which :func:`_classify` drops for having no
``python -m pytest`` line). That gap is written down in
`SWEEP-CARRY-FORWARD.md` rather than closed here.

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
    #: alone writes a file nobody reads. Six of the seven whole-suite lanes
    #: have both; the seventh is named in :data:`VERDICT_CHANNEL_EXEMPT` with
    #: its reason.
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
    "test-jax": ("floating", True, True, False, True),
    "test-jax-0-10": ("0.10", True, True, False, True),
    "test-jax-no-solvers": ("floating", False, True, False, True),
    "test-no-jax": ("absent", True, True, False, True),
}

#: The whole-suite lanes that do NOT assert the skip-inventory verdict off the
#: file channel, each with the reason. Six of the seven do; this is the
#: seventh, and it was an undisclosed asymmetry until it was written down.
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
