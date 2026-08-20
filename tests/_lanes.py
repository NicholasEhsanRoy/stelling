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
``release.yml`` the same way, for the same reason). Comment-only lines go
first — this workflow is more comment than code, deliberately — and what is
left is scanned for job headers, ``uv pip install`` lines and ``pytest``
invocations.

**A JOB WHOSE INSTALL LINE IS A MATRIX EXPANSION IS EXPANDED, ENTRY BY ENTRY,
AND A FIELD ITS ENTRIES DISAGREE ON IS A NAMED CAN'T-TELL.**
``acceptance-reproducer`` installs ``.[${EXTRAS}]``, and one ``Lane`` cannot
describe two provisionings. So :func:`_matrix_values` follows the one chain
the install line actually names — ``${EXTRAS}`` to the step's
``EXTRAS: ${{ matrix.extras }}`` to the ``strategy.matrix.include`` entries —
and the resolution rule is the same for every field: **the entries agree and
the value is theirs, or they do not and it is ``None``.** Today the two
entries name ``solvers,jax`` and ``solvers``, which agree that a solver extra
is installed, so :attr:`Lane.solvers` reads ``True`` because that is what
ci.yml says and not because a matrix job is assumed to have everything.
:attr:`Lane.jax` stays ``"matrix"``: the entries do disagree there — one pins
``0.10`` and one floats — so there is no single series, which is also why the
job is excluded from :data:`SERIES_BEARING`. That exclusion is a policy on top
of the reading, not a substitute for it: ci.yml's own text says the two
acceptance jobs *must not* be required checks, so a coverage claim resting on
them rests on a job whose red does not block.

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
#: The skip inventory's FILE channel, set by a job that asserts the
#: completeness verdict off something pytest's exit code cannot be taken from.
_VERDICT_CHANNEL = re.compile(r"\bSTELLING_SKIP_INVENTORY_VERDICT\b")


@dataclass(frozen=True)
class Lane:
    """One job in ``ci.yml``, in the terms a coverage claim needs."""

    job: str
    #: ``"absent"``, ``"floating"``, ``"matrix"``, or a series like ``"0.10"``.
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
    #: exit code. Six of the seven whole-suite lanes do; the seventh is named
    #: in :data:`VERDICT_CHANNEL_EXEMPT` with its reason.
    verdict_channel: bool


def _code_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


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
    be short. Values are unquoted; a nested structure inside an entry is not
    supported and reads as unreadable rather than as a partial entry.
    """
    entries: list[dict[str, str]] = []
    depth: int | None = None
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
        if not item or (item.group(1) is None and not entries):
            return []  # a shape this cannot read
        if item.group(1) is not None:
            entries.append({})
        value = item.group(3)
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        entries[-1][item.group(2)] = value
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
        for word in argv.split():
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
    verdict = any(_VERDICT_CHANNEL.search(line) for line in body)
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
#: same commit in file order and annotates which KIND of failure this was. A
#: bare `exit 1` in front of that classification would replace the one thing
#: this lane is for with a verdict every other lane already carries. And an
#: order-dependent undisclosed skip — the failure this lane is most likely to
#: find — reddens `test_no_session_skip_is_undisclosed` in the ordinary way,
#: which the classification step then triages.
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


def lane_series() -> tuple[str, ...]:
    """The jax series the merge-bearing lanes actually resolve.

    This is what a coverage claim about "a lane" means. See the module
    docstring for why the floating lane contributes ``max(TESTED_JAX_SERIES)``
    and for the one direction in which that inference can be wrong.
    """
    by_job = {lane.job: lane for lane in lanes()}
    found = set()
    for job in SERIES_BEARING:
        lane = by_job.get(job)
        if lane is None or lane.jax == "absent":
            continue
        found.add(_newest(TESTED_JAX_SERIES) if lane.jax == "floating" else lane.jax)
    return tuple(sorted(found, key=lambda s: tuple(int(p) for p in s.split("."))))
