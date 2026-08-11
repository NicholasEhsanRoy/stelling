# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Point the property suite at an arbitrary tree, revision, or mutant.

    python tools/property_check.py --tree /path/to/some/worktree
    python tools/property_check.py --rev  fb34e0d
    python tools/property_check.py --controls
    python tools/property_check.py --control widen -v

**The separation this tool exists for**: the PROPERTIES come from the checkout
this file lives in; the CODE UNDER TEST comes from wherever you point it. They
are joined by ``PYTHONPATH=<tree>/src`` and nothing else — no install, no
editable wheel, no worktree of the tests. That is what makes a property usable
as an independent third check on somebody else's branch: you do not have to
merge, rebase, or trust their copy of the tests.

It is also the whole mechanism behind ``--controls``. A positive control runs
**today's property** against **yesterday's defect** and asserts the run comes
back RED. Without that, a green property is indistinguishable from a property
whose strategy generates nothing — the failure mode this project has shipped
more than any other.

MATERIALISING A TREE. ``--rev`` uses ``git archive``, not ``git worktree``:
this repository is worked on by several agents at once, and ``git archive``
touches nothing under ``.git`` while ``git worktree add`` writes state that
somebody else's ``git worktree list`` then has to reason about. Only ``src/``
is extracted, because that is all ``PYTHONPATH`` needs.

EXIT CODE. 0 if every requested run had the outcome it was supposed to have.
For ``--controls`` that means every control FAILED where it was supposed to
fail; a control that passes is reported as ``CONTROL DID NOT FIRE`` and exits
non-zero, because a control that cannot demonstrate its property is worth
nothing.

WHERE ``expect_message`` IS MATCHED, and why it is not the captured output.
pytest's long traceback prints a frame's ENTIRE function source — docstring
included — from ``def`` down to the failing line. So the strings a property
BUILDS its failure messages from, and any string quoted in the docstring of a
function on the traceback path, are echoed into the run's output whether or
not the property's oracle ever ran. Matching ``expect_message`` against that
output makes a CRASH indistinguishable from a demonstration. Measured, on
``tests/property/test_cvc5_protocol.py`` with its two clause sentences quoted
in ``_judge``'s docstring: a one-place defect that raises ``TypeError`` before
the oracle is evaluated scored ``3/3 controls fired`` against three probe
controls carrying the three shipped guard strings. So the match is made
against the failure pytest itself RECORDS — the ``message`` attribute of each
``<failure>``/``<error>`` in ``--junitxml``. The same run now scores ``0/3``.

WHAT THAT ATTRIBUTE ACTUALLY HOLDS. It was described here as "the crash line
and the exception's own text and nothing else", and that is not what it is.
Measured on pytest 9.1.1 with hypothesis 6.165.2, reading one real failure of
``test_the_parent_never_trusts_an_unspoken_transcript_flat`` out of the XML::

    AssertionError: <clause (2)'s sentence> as 'unknown' [flat]  <- the
      read : '…'                                             exception's text
      full : '…'
    Failing test case: search(                              <- hypothesis note
        item=('…', '…', 0),
    )
    Explanation:                                            <- hypothesis note
        These lines were always and only run by failing test cases:
            …/tests/property/test_cvc5_protocol.py:443
    You can reproduce this test case by temporarily adding
    @reproduce_failure('6.165.2', b'…') as a decorator …    <- print_blob=True

and, where the failure is a bare ``assert``, pytest's assertion-rewrite
explanation as well — which embeds the **reprs of the asserted expression's
operands**, i.e. generated data. So the attribute carries strategy output, and
whether some generator could draw a guard string into it is a per-guard
question rather than something this file settles by construction. Of the twelve
guards registered today, eleven are shouted English phrases or bracketed leg
tags; ``reorder``'s ``transposed`` is the one lower-case word, and it is the
one to look at first if that question is ever asked in earnest. NOTHING here
is exploitable by any guard registered today.

THE BLIND SPOT THIS RULE HAS AND THE OUTPUT MATCH DID NOT. When hypothesis
finds MORE THAN ONE distinct failure it raises an ``ExceptionGroup``, and the
sub-exceptions' texts live in the traceback BODY only. Driven, two assert
sites in one ``@given`` function::

    <failure message="ExceptionGroup: Hypothesis found 2 distinct failures.
                      (2 sub-exceptions)">

Neither sentinel appears in the attribute. So a genuine demonstration that
found two defects at once is scored "wrong failure" here, where the old
output match would have scored it FIRED. Not live for any of the twelve today
— every one of them shrinks to a single interesting origin — but it is not far
off. Counted over the suite's ``@given`` functions: the reordering property
and the conjunct property have THREE raise/assert sites each, the wrap-class
oracle leg and the refutation property TWO each, and ``reorder``'s own guard
``transposed`` is written at two of the reordering property's three. A
second interesting origin is one mutant away, and the failure mode is a
control reported NOT DEMONSTRATED while demonstrating twice over. That is the
SAFE direction — it refuses, it never passes — which is why it is disclosed
here rather than worked around: reading sub-exception texts out of the
traceback body would put the guard back in reach of the property's own echoed
source, which is the defect this whole rule exists to close.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PROPERTY_DIR = REPO / "tests" / "property"
PYPROJECT = REPO / "pyproject.toml"

#: The ``pytest11`` entry point this project declared first, kept as a FLOOR
#: rather than as the answer: :func:`_entry_points_to_block` reads the live
#: declaration and blocks all of it, and falls back to this name alone if it
#: cannot. See :func:`_run` for what the block is for.
TRIPWIRE_ENTRY_POINT = "stelling_overflow"

#: The `[project.entry-points.pytest11]` table, up to the next table header,
#: and the names in it. Used ONLY when no TOML parser can be imported — see
#: `_declared_pytest11`, which says what these do not read.
_PYTEST11 = re.compile(
    r"""^\[project\.entry-points\.(?:pytest11|"pytest11"|'pytest11')\]\s*$"""
    r"(.*?)(?=^\[|\Z)",
    re.M | re.S,
)
_ENTRY_NAME = re.compile(r"""^\s*["']?([\w.-]+)["']?\s*=""", re.M)

sys.path.insert(0, str(PROPERTY_DIR))
import positive_controls as pc  # noqa: E402


def _entry_points_to_block() -> tuple[str, ...]:
    """Every ``pytest11`` entry point name this project declares.

    READ, NOT RESTATED, and the difference is the whole point. A hard-coded
    name closes the entry point that exists today and reopens the fault on the
    commit that adds the second one — measured: a second name under
    ``[project.entry-points.pytest11]`` puts every commit-pinned control back
    to dying before collection, with the whole test suite still green, because
    a membership assertion against the declaration permits the declaration to
    grow.

    THE FLOOR IS A FLOOR AND NOTHING ABOVE IT IS GUARANTEED. This said "FAILS
    CLOSED ON THE FLOOR, never open", which read as a property of the whole
    result and is a property of one name in it. :data:`TRIPWIRE_ENTRY_POINT`
    is in the answer whatever happens — an absent ``pyproject.toml``, an
    unparsable one, a checkout the file did not ship to. Every OTHER declared
    name fails OPEN, and silently: if this function does not see it, nothing
    blocks it and nothing says so. That is why the parse below is a real one.
    """
    names = {TRIPWIRE_ENTRY_POINT}
    try:
        raw = PYPROJECT.read_bytes()
    except OSError:
        return tuple(sorted(names))
    names.update(_declared_pytest11(raw))
    return tuple(sorted(names))


def _declared_pytest11(raw: bytes) -> set:
    """The declared names, by a real TOML parse where one can be had.

    ``tomllib`` is 3.11+ and this project's floor is 3.10, so the import is
    INSIDE the function and falls back to ``tomli`` — the same shape, and for
    the same measured reason, as ``tests/test_sdist_contents.py``: spelled at
    column 0 it is a collection error on the floor, exit 2, zero tests. pytest
    itself requires ``tomli`` below 3.11, so every environment that can run
    this tool at all has one of the two.

    WHY NOT THE REGEX ALONE, which is what this did first. TOML has more than
    one spelling of one table, and the regex reads one of them. Driven, five
    legal spellings of the same declaration, every one of them collapsing the
    answer to the floor name:

        [project.entry-points."pytest11"]        quoted table key
        [project.entry-points]                   + `pytest11 = {…}` inline
          indented header                        legal, and not `^\\[`
        "second.thing" = "…"                     quoted entry point name
        [project.entry-points.pytest11]  # note  trailing comment

    Four of those hit the "no such section" path, which is at least loud. The
    quoted ENTRY POINT NAME was silent: the section matched, the name did not,
    and a name this function does not return is a name nothing blocks. The
    regexes now read the quoted spellings of both, which shrinks that set
    without emptying it — the fallback is still a regex, and still narrower
    than the parser it stands in for. It runs only where neither ``tomllib``
    nor ``tomli`` imports.

    ``tomllib.loads`` wants ``str``; the bytes are decoded as UTF-8 because
    TOML MANDATES UTF-8, which makes it the specification rather than a guess.
    ``pyproject.toml`` carries 33 non-ASCII bytes today, the first an em-dash
    at offset 1257, so a locale-decided decode is not hypothetical: under
    ``LC_ALL=C`` with UTF-8 mode off, ``read_text()`` raised
    ``UnicodeDecodeError`` — a ``ValueError``, NOT an ``OSError``, so the
    caller's fallback did not catch it and the tool died with a traceback
    before running anything.

    THAT LAST REPAIR IS NOT DRIVEN BY ANY TEST, and this is the disclosure
    rather than the closure. Reverting the explicit decode leaves every test
    in this repository green, because the environments they run in have a
    UTF-8 locale and the fault needs one that does not — which no in-process
    fixture can arrange, and which PEP 538 and PEP 540 make hard to reach on a
    modern Linux default anyway. A test that mocked the locale would measure
    the mock. Ten other mutations of this change are each caught by exactly
    one test; this one is caught by none, and is here because it is right and
    costs nothing, not because anything watches it.
    """
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
    except ImportError:
        tomllib = None
    if tomllib is not None:
        try:
            table = (
                tomllib.loads(raw.decode("utf-8"))
                .get("project", {})
                .get("entry-points", {})
                .get("pytest11", {})
            )
        except (UnicodeDecodeError, ValueError, AttributeError):
            return set()
        return set(table) if isinstance(table, dict) else set()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return set()
    table = _PYTEST11.search(text)
    return set(_ENTRY_NAME.findall(table.group(1))) if table else set()


# ── materialising the tree under test ────────────────────────────────────────


def _materialise(rev: str, into: pathlib.Path, repo: pathlib.Path) -> pathlib.Path:
    """Put ``<rev>``'s ``src/`` under ``into`` and return the tree root."""
    into.mkdir(parents=True, exist_ok=True)
    if rev in ("HEAD", "WORKTREE"):
        # The WORKING TREE's src, not the committed one: a control run while
        # you are editing the remedy must test what you are editing.
        shutil.copytree(repo / "src", into / "src")
        return into
    tar = subprocess.run(
        ["git", "-C", str(repo), "archive", rev, "src"],
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=tar, check=True)
    return into


def _apply(mutation, tree: pathlib.Path) -> None:
    path = tree / mutation.path
    text = path.read_text()
    n = text.count(mutation.old)
    if n != 1:
        raise SystemExit(
            f"MUTATION DID NOT APPLY: {mutation.path} contains "
            f"{n} occurrences of the target text, expected exactly 1.\n"
            f"  looking for: {mutation.old!r}\n"
            "The registry has drifted from the source. Fix the registry — a "
            "control that silently stops mutating is a control that always "
            "passes."
        )
    path.write_text(text.replace(mutation.old, mutation.new))


# ── running ──────────────────────────────────────────────────────────────────


def _run(tree, targets, *, python, profile, scale, extra_env=None, runxfail=False,
         verbose=False, extra_args=(), junit=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(tree) / "src")
    env["JAX_PLATFORMS"] = env.get("JAX_PLATFORMS", "cpu")
    env["STELLING_PROPERTY_PROFILE"] = profile
    env["STELLING_PROPERTY_SCALE"] = str(scale)
    # This child's output is READ, not merely echoed: `_verdict` decides
    # ECHOED on `expect_message in out`, and pytest paints traceback source —
    # docstrings included — whenever it believes a human is watching, which it
    # does whenever `FORCE_COLOR` is set in the environment this inherits. An
    # SGR escape landing inside the expected string turns a control that FIRED
    # correctly into a WRONG verdict. Same rule the children of
    # `tests/test_skip_inventory.py` follow, and set before `extra_env` so a
    # caller that wants colour can still ask for it. The DEMONSTRATED path is
    # unaffected either way: it reads the junit XML, which carries no colour.
    env["PY_COLORS"] = "0"
    env.pop("STELLING_PROPERTY_DB", None)
    if extra_env:
        env.update(extra_env)
    # THE CHILD MUST NOT LOAD THIS PROJECT'S OWN pytest PLUGIN, and the reason
    # is the `PYTHONPATH` line at the top of this function rather than anything
    # about what the plugin does.
    #
    # `PYTHONPATH=<tree>/src` is the whole mechanism of this file: the child's
    # `stelling` is the REVISION's, not the checkout's. Its `pytest11` entry
    # point, though, comes from the DISTRIBUTION METADATA in the environment,
    # which describes the checkout — and pytest loads entry points before it
    # collects anything, with no exception handling: an entry point naming a
    # module the tree on `PYTHONPATH` does not have kills the session at
    # `Config.parse`, before a single test is seen.
    #
    # Measured at 260527b, the commit that added `stelling._tripwire`, in the
    # `property` CI job (which installs with `-e`, so the metadata is there):
    # a clean partition of the nine per-push controls. All five whose tree is
    # `HEAD` — `_materialise` COPIES the working tree's `src/`, which has the
    # package — came back FIRED. All four pinned to a `commit` — `git archive`
    # of a revision that predates it — came back `ModuleNotFoundError: No
    # module named 'stelling._tripwire'` and were reported NOT DEMONSTRATED.
    # Nothing was wrong with those four properties or those four trees.
    #
    # This is PERMANENT for any revision older than a module the entry point
    # names, so it is fixed here and not by moving the pin. Blocking is by
    # entry point NAME, which is what `-p no:` matches, and the names are READ
    # from `pyproject.toml` rather than written here: a rename OR AN ADDITION
    # that this file did not follow would silently restore the crash on every
    # commit-pinned control at once. See `_entry_points_to_block`.
    #
    # NOT `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and the reason written here first
    # was FALSE: it said that would take hypothesis's own plugin away from the
    # property suite, "the one plugin these runs cannot do without". Measured,
    # in the installed venv, with autoload disabled and no `-p no:` at all:
    # `test_a_refuted_is_false_at_some_admitted_point` -> `1 passed`, and two
    # commit-pinned controls -> `2/2 controls fired`, exit 0, DEMONSTRATED off
    # the recorded junit message. The suite plainly can do without it.
    #
    # The real reason is scope. `-p no:<name>` changes exactly one thing about
    # the child; disabling autoload changes what EVERY plugin does in it, and
    # this child is an instrument whose readings are compared across trees and
    # across machines. What the property suite needs from hypothesis's plugin
    # was never the question — what it needs is to be the same instrument in
    # CI, in a dev venv, and on a box with pytest-randomly installed.
    #
    # TWO COSTS, DISCLOSED. `src/stelling/overflow.py` registers the same
    # plugin under this same name by hand, so a future `pytest_plugins =
    # ["stelling.overflow"]` under `tests/property/` would be blocked in these
    # children — and pluggy's `register()` returns None for a blocked name
    # rather than raising, so it would be blocked SILENTLY. Nothing opts in
    # today (the plugin is registered always and active never). The day the
    # property suite wants the tripwire armed, this block is what stops it.
    #
    # And these children can no longer notice that this project's own plugin
    # is BROKEN, because they never load it. That is the right scope for this
    # tool — a control's verdict must depend on the tree, not on whether some
    # plugin in the environment imports — and it is covered next door:
    # `tests/test_tripwire_plugin.py` pins the entry point against
    # `pyproject.toml` and drives the plugin itself.
    argv = [
        python, "-m", "pytest", "-ra", "-p", "no:cacheprovider",
        *[arg for name in _entry_points_to_block() for arg in ("-p", f"no:{name}")],
        "--no-header", *extra_args,
    ]
    if runxfail:
        argv.append("--runxfail")
    if junit is not None:
        argv.append(f"--junitxml={junit}")
    argv += list(targets)
    if verbose:
        print("   $ PYTHONPATH=%s STELLING_PROPERTY_PROFILE=%s %s"
              % (env["PYTHONPATH"], profile, " ".join(argv)))
    proc = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True, text=True)
    return proc


def _tail(proc, n=25):
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = out.strip().splitlines()
    return "\n".join("   | " + ln for ln in lines[-n:])


def _crashes(junit) -> list[str]:
    """What the run REPORTED as its failures, as pytest itself records them.

    One string per ``<failure>``/``<error>``: pytest's ``message`` attribute,
    and NOT the traceback body. That distinction is the whole point (see this
    module's docstring): the traceback body carries the property's own source,
    so a guard string is present there even when nothing evaluated the oracle.

    IT IS NOT "THE CRASH LINE AND NOTHING ELSE", which is what this said. The
    attribute is the exception type and its own text, PLUS pytest's assertion
    -rewrite explanation where the failure is a bare ``assert`` (which embeds
    the reprs of the asserted expression's operands, i.e. generated data),
    PLUS hypothesis's own notes — ``Failing test case: …``, the
    ``Explanation:`` file:line list, and the ``@reproduce_failure`` blob that
    ``print_blob=True`` in ``_profiles.py`` asks for. Measured, verbatim, in
    this module's docstring.

    AND ONE SHAPE IT CANNOT SEE: when hypothesis finds more than one distinct
    failure it raises an ``ExceptionGroup``, whose ``message`` is
    ``"Hypothesis found N distinct failures. (N sub-exceptions)"`` — the
    sub-exceptions' texts are in the traceback body alone. A control whose
    property found two defects at once is therefore scored "wrong failure"
    here. Disclosed rather than closed, because the safe direction is to
    refuse; this module's docstring has the driven measurement and the reason.

    An empty list is the honest answer for a run that produced no XML at all —
    a pytest usage error, an interpreter that died before collection — and the
    caller treats it as "did not carry", which is the safe direction. WHICH IS
    NOT THE SAME AS THE RIGHT SENTENCE, and this docstring used to stop here
    as though it were: see :func:`_reported`, which is what separates "ran and
    recorded the wrong failure" from "never ran".
    """
    try:
        root = ET.parse(junit).getroot()
    except (OSError, ET.ParseError):
        return []
    return [
        child.get("message") or ""
        for case in root.iter("testcase")
        for child in case
        if child.tag in ("failure", "error")
    ]


def _reported(junit) -> bool:
    """Did pytest get far enough to report on ANY test?

    THE QUESTION THIS ANSWERS is the one `_crashes` alone cannot: a run that
    recorded nothing and a run that recorded the wrong thing both produce an
    empty crash list, and they are not the same event. The first says nothing
    whatever about the property; the second says the property ran and failed
    somewhere else.

    THREE SPELLINGS, and the third was missing from this function while this
    docstring already claimed the general question. Measured on pytest 9.1.1,
    each with the exit code it produces:

    * a plugin that could not import           rc 1, NO XML AT ALL
    * a nodeid that matches nothing            rc 4, XML, zero ``<testcase>``
    * a TEST MODULE that could not import      rc 2, XML, ONE ``<testcase>``
      with ``classname=""`` and a single ``<error message="collection
      failure">``

    The third is the near neighbour of the event that produced this file's
    remedy, one step to the left: the checkout's property module imports
    something the REVISION's ``src/`` does not have. Keying only on "is there
    a ``<testcase>``" answered yes to it, and it was printed as ``FIRED, but
    the failure did not carry '…'`` — the exact sentence the rest of this
    change exists to stop. A collection failure is pytest reporting that it
    could not reach a test, so it is not a report ON a test.

    THE DISCRIMINATOR IS THE EMPTY ``classname``, which pytest writes for a
    collection failure and never for a test that ran (a test's ``classname``
    is its module path). It is a convention, not a guarantee, so the fail
    direction matters: if pytest ever stopped writing it, a collection failure
    would score ``WRONG`` again — the safe-but-wrongly-worded behaviour this
    replaced, never a green verdict.

    KEYED POSITIVELY, NOT BY SUBTRACTION. The first implementation asked
    ``any(not _is_collection_failure(case) ...)``, subtracting one known-bad
    shape. Two more shapes exist that are not real test reports and not
    collection failures in that sense:

    * ``classname="pytest" name="internal"`` — a ``pytest_internalerror``
      session, rc 3, where pytest itself broke. Not a collection failure
      (classname is not empty), not a test report.
    * ``classname="" ... <skipped>`` — a module gated by ``importorskip``,
      rc 5, where pytest chose not to collect. Same empty classname as the
      ``<error>`` shape, different child tag.

    Subtracting each one is a race against shapes nobody has seen yet. The
    positive key — a testcase "looks like a test" when its ``classname`` is
    present AND is not ``"pytest"`` — handles all known shapes and any future
    shape that shares either marker, which is the right default for an
    instrument whose failure mode is to say something it cannot support.

    MEASURED, and the reason this exists. At 260527b every control pinned to a
    commit older than ``stelling._tripwire`` died at entry-point load, wrote no
    XML, and was printed as ``FIRED, but the failure did not carry '…'`` —
    which is false twice over: it did not fire, and there was no failure to
    carry anything. Four of the nine per-push controls, reported in the shape
    of a property defect, over an environment fault. The remedy for the fault
    is in :func:`_run`; this is the remedy for the sentence.
    """
    try:
        root = ET.parse(junit).getroot()
    except (OSError, ET.ParseError):
        return False
    return any(
        case.get("classname") and case.get("classname") != "pytest"
        for case in root.iter("testcase")
    )


def _is_collection_failure(case) -> bool:
    """pytest's shape for "I could not reach this module", measured at 9.1.1.

    THE DISCRIMINATOR IS THE EMPTY ``classname``, and nothing else is needed.
    pytest writes ``classname=""`` for TWO collection-level events — an
    ``<error>`` when the module could not import, and a ``<skipped>`` when it
    was gated by ``importorskip`` — and never for a test that ran (a test's
    ``classname`` is its module path). Keying on the child's tag would miss
    the second shape; keying on the empty classname catches both.

    Checked at 9.1.1 under ``junit_family=xunit1`` and ``junit_logging=all``:
    the shape does not move.
    """
    return not case.get("classname")


# ── the decision ─────────────────────────────────────────────────────────────
#
# THE ONE JUDGEMENT THIS TOOL MAKES, in one place so that something can execute
# it. It was four lines inside `check_controls`, reachable only by running a
# real control against a real tree, and nothing in the repository asserted
# anything about it: reverting `carried` to match the run's echoed output
# scored a pure crash — a `TypeError` raised before the oracle was evaluated at
# all — as `FIRED`, including for a control that is in the per-push gate, and
# every gate stayed green. It is a function now, and
# `tests/property/test_suite_disclosure.py` drives it end to end over three
# synthetic controls whose outcomes are known.

DEMONSTRATED = "demonstrated"
ECHOED = "echoed, not raised"
WRONG = "wrong failure"
DID_NOT_FIRE = "passed where it must fail"
NEVER_RAN = "the session reported on no test at all"


def _verdict(expect_message, crashes, out, *, fired, reported=True) -> str:
    """Did this run demonstrate the defect, and if not, in which way not?

    ``crashes`` is ``_crashes(junit)`` — what pytest RECORDED. ``reported`` is
    ``_reported(junit)`` — whether it recorded anything at all. ``out`` is
    everything the run echoed. The order of the tests below is the whole
    decision procedure:

    * green -> ``DID_NOT_FIRE``, which is the failure this registry exists for;
    * red, and pytest reported on NO test -> ``NEVER_RAN``. It comes SECOND, on
      purpose, and ahead of every match: a session that died in configuration
      still echoes a traceback, and this file's own rule is that the echoed
      output is not evidence. Scoring it against ``expect_message`` at all is
      how four controls came to be printed as "FIRED, but the failure did not
      carry …" over a missing module;
    * the guard is in a RECORDED failure -> ``DEMONSTRATED``;
    * the guard is only in the echoed output -> ``ECHOED``, its own outcome
      because the remedy differs (nothing is wrong with the control; a
      docstring or a message template on the traceback path put it there);
    * red, and the guard is nowhere -> ``WRONG``.

    Everything except ``DEMONSTRATED`` is NOT DEMONSTRATED and exits non-zero,
    so the added outcome moves no verdict from red to green. What it moves is
    what the report SAYS, which is the whole product of this tool.

    IT CAN MOVE ONE VERDICT THE OTHER WAY, and the argument that it cannot was
    written here first and was wrong. It said ``_crashes`` reads only from
    ``<testcase>``, so a non-empty crash list implies ``reported`` — it does
    not: ``_crashes`` reads the ``<error>`` child of a COLLECTION-FAILURE
    ``<testcase>`` too, and on a real rc-2 run ``_crashes`` returns
    ``['collection failure']`` while ``_reported`` returns ``False``. Since
    ``NEVER_RAN`` is tested first, a control whose ``expect_message`` were a
    substring of ``collection failure`` would move from ``DEMONSTRATED`` to
    ``NEVER_RAN`` — green to RED, the safe direction, and not true of any of
    the twelve guards registered today (checked against the registry). Worth
    keeping in view for the same reason the module docstring keeps
    ``reorder``'s lower-case ``transposed`` in view.

    ``reported`` defaults to ``True`` so that the three-argument call still
    means what it always meant: "pytest ran and this is what it recorded".
    """
    if not fired:
        return DID_NOT_FIRE
    if not reported:
        return NEVER_RAN
    if any(expect_message in c for c in crashes):
        return DEMONSTRATED
    if expect_message in out:
        return ECHOED
    return WRONG


# ── the two modes ────────────────────────────────────────────────────────────


def check_tree(args) -> int:
    with tempfile.TemporaryDirectory(prefix="stelling-prop-") as tmp:
        if args.tree:
            tree = pathlib.Path(args.tree).resolve()
            label = f"tree {tree}"
        else:
            tree = _materialise(args.rev, pathlib.Path(tmp) / "t",
                                pathlib.Path(args.repo).resolve())
            label = f"rev {args.rev}"
        if args.mutant:
            control = pc.by_name(args.mutant)
            if control.mutation is None:
                raise SystemExit(f"control {args.mutant!r} is not a mutant")
            _apply(control.mutation, tree)
            label += f" + mutant {args.mutant}"
        targets = args.select or [str(PROPERTY_DIR)]
        print(f"== property suite against {label}")
        print(f"   properties from {PROPERTY_DIR}")
        print(f"   profile {args.profile} x{args.scale}, python {args.python}")
        proc = _run(tree, targets, python=args.python, profile=args.profile,
                    scale=args.scale, verbose=True,
                    extra_env=_cross_env(args))
        print(_tail(proc, 40 if proc.returncode else 12))
        print(f"== exit {proc.returncode}")
        return proc.returncode


def _cross_env(args):
    return {"STELLING_PROPERTY_OTHER_PYTHON": args.other_python} \
        if args.other_python else None


def check_controls(args) -> int:
    wanted = ([pc.by_name(n) for n in args.control]
              if args.control else list(pc.CONTROLS))
    failures = []
    for control in wanted:
        if control.series == "both" and not args.other_python:
            print(f"-- {control.name}: SKIPPED (needs --other-python, an "
                  f"interpreter with the other jax series AND hypothesis — "
                  f"tools/property_venv.sh builds one)")
            failures.append((control.name, "not demonstrated: no second series"))
            continue
        with tempfile.TemporaryDirectory(prefix="stelling-ctl-") as tmp:
            tree = _materialise(control.at, pathlib.Path(tmp) / "t",
                                pathlib.Path(args.repo).resolve())
            if control.mutation is not None:
                _apply(control.mutation, tree)
            print(f"-- {control.name}  [{control.kind} {control.at}]")
            print(f"   {control.nodeid}")
            # The control's own scale multiplies the caller's: some searches
            # need more room than a per-push budget, and burying that in a
            # global flag would make every control pay for the slowest one.
            scale = float(args.scale) * control.scale
            junit = pathlib.Path(tmp) / "junit.xml"
            proc = _run(tree, [control.nodeid], python=args.python,
                        profile=args.profile, scale=scale,
                        runxfail=True, verbose=args.verbose,
                        extra_env=_cross_env(args), junit=junit)
            out = (proc.stdout or "") + (proc.stderr or "")
            crashes = _crashes(junit)
            verdict = _verdict(control.expect_message, crashes, out,
                               fired=proc.returncode != 0,
                               reported=_reported(junit))
            if verdict == DEMONSTRATED:
                print("   FIRED — the property failed where it is supposed to")
                if args.verbose:
                    print(_tail(proc, 30))
            elif verdict == NEVER_RAN:
                # NOT a statement about the property, and deliberately worded
                # so that it cannot be read as one. pytest exited non-zero
                # having reported on no test, so the tree, the mutation and
                # the oracle are all unexamined.
                #
                # AND NOT A STATEMENT ABOUT THE ENVIRONMENT EITHER, which is
                # what this said first. The environment is the likeliest
                # culprit, not the only one: a nodeid this registry spells
                # wrongly reaches here too, and that is a defect in the
                # CONTROL. Naming a culprit is the one thing the tool cannot
                # do from what it has, so it names the candidates.
                print("   THE PROPERTY NEVER RAN — pytest exited without "
                      "reporting on any test, so this run says nothing about "
                      "the property, the tree or the mutation. Something "
                      "stopped the session before it reached a test: a plugin "
                      "that could not import, a test module that could not "
                      "import, an interpreter that failed to start, or a "
                      "nodeid in the registry that matches nothing.")
                print(_tail(proc, 30))
                failures.append((control.name, NEVER_RAN))
            elif verdict == ECHOED:
                # The string is in the run's OUTPUT but not in any failure the
                # run recorded, which is the shape a docstring or a message
                # template echoed by the traceback produces. Reported as its
                # own outcome rather than folded into "wrong failure", because
                # the remedy is different: nothing is wrong with the control.
                print(f"   FIRED, but the failure ITSELF did not carry "
                      f"{control.expect_message!r} — the string is only in the "
                      f"run's echoed output (a traceback prints the property's "
                      f"own source, docstring included)")
                for c in crashes:
                    print(f"   what pytest recorded: {c.splitlines()[0][:120]}")
                print(_tail(proc, 30))
                failures.append((control.name, ECHOED))
            elif verdict == WRONG:
                print(f"   FIRED, but the failure did not carry "
                      f"{control.expect_message!r}")
                for c in crashes:
                    print(f"   what pytest recorded: {c.splitlines()[0][:120]}")
                print(_tail(proc, 30))
                failures.append((control.name, WRONG))
            else:
                print("   CONTROL DID NOT FIRE — this property cannot be shown "
                      "to detect anything")
                print(_tail(proc, 30))
                failures.append((control.name, DID_NOT_FIRE))
    print()
    print(f"== {len(wanted) - len(failures)}/{len(wanted)} controls fired")
    for name, why in failures:
        print(f"   NOT DEMONSTRATED: {name} — {why}")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    where = p.add_mutually_exclusive_group()
    where.add_argument("--tree", help="a checkout to test (uses <tree>/src)")
    where.add_argument("--rev", help="a revision to materialise and test")
    p.add_argument("--repo", default=str(REPO),
                   help="the git repository --rev is read from")
    p.add_argument("--mutant", help="also apply this registered control's mutation")
    p.add_argument("--controls", action="store_true",
                   help="run every positive control and assert each FAILS")
    p.add_argument("--control", action="append",
                   help="run one positive control by name (repeatable)")
    p.add_argument("--select", action="append",
                   help="pytest target(s); default is the whole property suite")
    p.add_argument("--profile", default="ci", choices=("ci", "dev", "nightly"))
    p.add_argument("--scale", default="1.0")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter with hypothesis and jax installed")
    p.add_argument("--other-python",
                   help="interpreter with the OTHER jax series AND hypothesis, "
                        "for the cross-series property (tools/property_venv.sh "
                        "builds one; the child imports _grammar, so a bare jax "
                        "venv fails there rather than differing)")
    p.add_argument("--list", action="store_true", help="list the controls")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for c in pc.CONTROLS:
            print(f"{c.name:20s} {c.kind:7s} {c.at:10s} x{c.scale:<5g} {c.nodeid}")
        return 0
    if args.controls or args.control:
        return check_controls(args)
    if not (args.tree or args.rev):
        args.rev = "HEAD"
    return check_tree(args)


if __name__ == "__main__":
    raise SystemExit(main())
