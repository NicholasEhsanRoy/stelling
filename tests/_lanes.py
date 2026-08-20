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

**A job whose install line is a matrix expansion is classified ``"matrix"``
and nothing is inferred from it.** ``acceptance-reproducer`` installs
``.[${EXTRAS}]`` with ``EXTRAS`` coming from ``strategy.matrix``, and reading
the series out of that would mean reading a second file's worth of YAML
structure to learn something this module then must not use anyway: ci.yml's
own policy says in as many words that the two acceptance jobs *must not* be
required checks, so a coverage claim resting on them rests on a job whose red
does not block. Classified, excluded from :data:`SERIES_BEARING`, and said so.

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
_MATRIX = re.compile(r"\.\[\$")


@dataclass(frozen=True)
class Lane:
    """One job in ``ci.yml``, in the terms a coverage claim needs."""

    job: str
    #: ``"absent"``, ``"floating"``, ``"matrix"``, or a series like ``"0.10"``.
    jax: str
    solvers: bool
    #: Whether some step runs ``pytest`` over the WHOLE tree (no path argument).
    whole_suite: bool
    random_order: bool


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


def _classify(body: list[str]) -> Lane | None:
    installs = [m.group(1) for line in body if (m := _INSTALL.search(line))]
    pytests = [m.group(1) for line in body if (m := _PYTEST.search(line))]
    if not pytests:
        return None  # not a test lane at all (`reuse`, `dco`)

    joined = " ".join(installs)
    if _MATRIX.search(joined):
        jax = "matrix"
        solvers = True  # every matrix entry names an extras set containing it
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
    return Lane(job="", jax=jax, solvers=solvers, whole_suite=whole,
                random_order=random_order)


def lanes() -> tuple[Lane, ...]:
    """Every job in ``ci.yml`` that runs pytest, as read from the file."""
    found = []
    for job, body in _blocks(_code_lines(CI.read_text(encoding="utf-8"))).items():
        lane = _classify(body)
        if lane is not None:
            found.append(Lane(job, lane.jax, lane.solvers, lane.whole_suite,
                              lane.random_order))
    return tuple(sorted(found, key=lambda l: l.job))


#: THE MEASURED LANES, DECLARED. Same idiom as
#: ``tests/test_skip_inventory.py``'s pins and ``EXPECTED_HASH_COVERAGE``: the
#: right-hand side is recomputed from the workflow and compared, so a lane
#: added, removed or re-provisioned is a line in a diff rather than a silent
#: change in what CI measures.
#:
#: ``(jax, solvers, whole_suite, random_order)``.
EXPECTED_LANES: dict[str, tuple[str, bool, bool, bool]] = {
    "acceptance-any-pytree": ("floating", True, True, False),
    "acceptance-reproducer": ("matrix", True, True, False),
    "property": ("floating", False, False, False),
    "random-order": ("floating", True, True, True),
    "test-jax": ("floating", True, True, False),
    "test-jax-0-10": ("0.10", True, True, False),
    "test-jax-no-solvers": ("floating", False, True, False),
    "test-no-jax": ("absent", True, True, False),
}

#: The lanes a documented-hash or series claim may rest on: whole-suite, and
#: not one of the two the workflow's own policy says must NOT be a required
#: check. Declared here rather than inferred from the prose beside those jobs —
#: a fence that reads a comment is a fence a reflow can turn off — and held to
#: the parse by ``test_lanes.py``.
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
