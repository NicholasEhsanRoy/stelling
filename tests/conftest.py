# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Records what every test in a session actually did, for the skip inventory.

``tests/test_skip_inventory.py`` pins which tests skip and under which
condition. It cannot do that by reading the tree: a skip is only observable by
RUNNING. There are well over a hundred ``importorskip`` sites in ``tests/`` and
in any given environment almost none of them fire, so "which sites exist" and
"which tests skipped" are different questions and only the second one is the
one that drifts.

pytest already computes the answer; this plugin is the cheap way to keep it.
It must be loaded before collection, which is why it is a conftest and not
part of the test module that consumes it.

Two report streams are needed, because skips arrive by two routes:

* ``pytest_runtest_logreport`` — a test skipped by a marker, by a fixture's
  ``importorskip``, or by a ``pytest.skip()`` in its own body. ``nodeid`` is
  the test.
* ``pytest_collectreport`` — a module whose top-level ``importorskip`` gate
  fired. The module never yields tests at all, and ``nodeid`` is the FILE.
  This is the shape the zero-dep CI job is built on, and a recorder that
  watched only the first stream would see none of it.

**An xfail arrives down the first of those streams and is not a skip**, and
the difference between "not a skip" and "not recorded" is a route this
recorder was blind to for two commits:

    @pytest.mark.xfail(run=False, reason="a planted reason nobody disclosed")
    def test_planted_never_runs():
        assert False        # appended to tests/test_affine.py

    pytest -q -rs   ->  2010 passed, 2 skipped, 1 xfailed, EXIT 0, no banner
                        and the claim MADE     (measured at b277083)

``xfail(run=False)`` raises ``XFailed`` in SETUP, so its report is
``report.skipped`` with ``wasxfail`` set. The recorder returned on ``wasxfail``
and recorded nothing at all — and the pin then went on to certify, positively,
"no undisclosed skip in this session" about a session containing a test that
never ran. ``pytest.xfail("…")`` in a body is the same shape one phase later,
and ``pytest.skip()`` with that same string in that same place is exit 1.
So an xfail is recorded, in :data:`XFAILED` and not in :data:`SKIPPED`, and the
completeness claim is WITHDRAWN on it — withdrawn and not failed, because
``N xfailed`` is in pytest's own summary line exactly as ``N deselected`` is,
and that is the cut this pin makes everywhere else. An ``xpassed`` report
(``wasxfail`` set, outcome PASSED) is a test that ran and passed, and is
recorded as having run.

The same plant at a80d60c is ``2021 passed, 3 skipped, 1 xfailed``, EXIT 0, and
the extra skip is the pin withdrawing and saying why. (A count with no commit
beside it is a fact about one tree wearing the shape of a fact about the
mechanism; ``here`` was the label until the suite grew under it.)

**The whole taxonomy was then re-walked for anything else pytest DISCLOSES
that this recorder drops**, by driving one test of every shape past it and
printing the channels — measured, not read off the docs:

=================================  ==================  =====================
what pytest reports                channel here        can it hide a skip?
=================================  ==================  =====================
passed                             ``RAN``             no
failed                             ``RAN``             no (session is red)
skipped: marker / skipif           ``SKIPPED``         no
skipped: fixture ``importorskip``  ``SKIPPED``         no
skipped: ``pytest.skip()`` in body ``SKIPPED``         no
skipped: in TEARDOWN               ``SKIPPED``         no
skipped: a parametrised case       ``SKIPPED``         no
skipped: module-level gate         ``SKIPPED`` (file)  no
skipped: ``allow_module_level``    ``SKIPPED`` (file)  no
xfailed: ``run=False``             ``XFAILED``         no — WAS the route
xfailed: ``pytest.xfail()``        ``XFAILED``         no — was the sibling
xfailed: marker + a real failure   ``XFAILED``         no
xpassed (and strict xpass)         ``RAN``             no
deselected                         ``DESELECTED``      no
ERROR in setup                     ``STARTED`` only    no — `N errors`, red
ERROR in teardown                  ``RAN``             no
a collection ERROR                 —                   no — exit 2, or red
a file with no tests               not in SEEN_FILES   no — reads as unseen
=================================  ==================  =====================

One row has no outcome channel: a test whose SETUP errored is in ``STARTED``
and nowhere else. It is accounted for (so the order check does not fire on it)
and it cannot hide anything, because pytest reports ``N errors`` and the
session is non-zero — which is a claim about the EXIT CODE, and is therefore
only as good as the exit code is, a dependence :func:`pytest_unconfigure`
takes seriously and this row inherits. That is the whole of what the walk
found.

**And the whole of what that walk COULD find, which is less than it looks.**
The table asks one question — *what channel does this report land in* — so
every route it can see is a report shape. Three routes are not report shapes.
In the first two the skip is recorded in :data:`SKIPPED` exactly as it should
be, and nothing ever reads it: the DECISION does not run. Both were exit 0 and
silent when this table was complete and correct.

* ``pytest.exit(reason, returncode=0)`` from inside the run loop. ``Exit``
  propagates through :func:`pytest_runtestloop`'s ``yield``, so the close is
  skipped, and ``wrap_session`` assigns ``session.exitstatus`` from the
  returncode, which overrides the ``testsfailed`` the close would have set.
  Answered by :func:`pytest_sessionfinish`, and the hole THAT left is answered
  by :func:`pytest_unconfigure`; see both.
* an undisclosed skip in the pin's own file, AFTER the pin has claimed. The pin
  runs last among files, not among tests. Answered by :func:`_close_the_session`
  re-asking rather than returning on :data:`CLAIM_MADE`; see there.

The third is not even that. ``pytest.exit(reason, returncode=0)`` raised from
inside ``pytest_sessionstart`` (or ``pytest_configure``) leaves
``wrap_session``'s ``initstate`` below 2, so ``pytest_sessionfinish`` is never
called AT ALL — and there is no record, because nothing was collected. Nothing
is owed and nothing is hidden; what was wrong was the sentence claiming the
mechanism covered it. Measured, and answered structurally, in
:func:`pytest_unconfigure`.

The lesson is about the shape of the taxonomy and not about these entries:
a table of channels bounds what can be MISREPORTED, and says nothing about
whether anybody looked.

Two more things are recorded, and neither of them is an outcome.

**How much of the suite this session was allowed to look at.** The inventory's
completeness half claims something about THE SUITE's skip set, and a session
narrowed by ``-k``, by an explicit path or by ``--lf`` can only support a claim
about what it collected. Nothing in pytest's report distinguishes the two —
``1927 passed, 2 skipped`` (measured at 384597e, with an undisclosed skip
planted and ``-k "not verdict"``) reads exactly like a whole run — so the scope is
recorded here, where the collection is, and asserted there, where the claim is.

**Whether the pin got to see the whole session.** Scope is not the only way
that claim goes vacuous and it is not the interesting one: the pin reads
OUTCOMES, so it has to run last, and *collecting* the whole suite says nothing
about the ORDER the suite runs in. Four routes, each measured on a whole-tree
session with an undisclosed skip planted, and each exit 0 before this was
written:

* ``pytest --nf``. ``NFPlugin.pytest_collection_modifyitems`` is
  ``@hookimpl(wrapper=True, tryfirst=True)``, so it re-sorts ``items`` AFTER
  every non-wrapper hookimpl, including the ordering one below. The pin ran
  first and saw an empty session.
* a plugin that reorders ``items`` the same way. Nothing in this repo sets
  ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` or ``addopts``, so an installed
  pytest-randomly / pytest-random-order / pytest-reverse does exactly this
  with no flag at all.
* a plugin that drops items from ``items[:]`` without calling
  ``config.hook.pytest_deselected`` (the hookspec says it must be called;
  nothing enforces it). Its summary line was byte-identical to a clean run.
* ``--deselect tests/test_skip_inventory.py``, which removes the pin outright.

A fifth and a sixth, found after the first four were closed and closed by the
same mechanism:

* ``--ignore=tests/test_skip_inventory.py`` (and ``--ignore-glob``, and the
  same thing through ``PYTEST_ADDOPTS``). The pin's own file is never
  collected, which reads as an ordinary narrowing — so the guard below neither
  claimed, nor withdrew, nor printed.
* the same, plus a ``-k`` that selects only tests which skip: nothing reaches
  a call phase, so an empty ``RAN`` was read as "nothing ran".

So the order is CHECKED rather than trusted, and the claim is made from a
place no invocation can remove:

* :func:`pending_items` lets the pin ask, at the moment it runs, whether the
  session still owed anything — so an ordering hook that won a fight becomes a
  deferral instead of a silent pass;
* and if the pin never made its claim — reordered too early, filtered out of
  ``items``, deselected, or ignored — the same claim is made HERE, at the end
  of the session, where the record is complete. That is the one place an
  ``items`` filter cannot reach, because removing a conftest is not something
  a filter can do.

:func:`pytest_collection_modifyitems` puts the pin last, which is where every
other plugin does its ordering and inside the window pytest-xdist builds its
index map from. It is a convenience, not the mechanism: something that
re-sorts after it makes the pin DEFER, and the deferral is answered below. An
earlier version also re-sorted inside :func:`pytest_runtestloop`, which is
after xdist freezes that map; see there for the two measurements that took it
back out.

The session's skips are recorded as pytest reports them; see
``tests/test_skip_inventory.py`` for what is done with them, and for why this
cannot be a static read of the tree.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import pathlib
import sys

import pytest

# nodeid -> skip reason. Test nodeids and (for module-level gates) file paths.
SKIPPED: dict[str, str] = {}

# nodeids that reached their call phase, whatever the verdict. "Did not skip"
# is half of what the inventory asserts, so passing is recorded too.
RAN: set[str] = set()

# nodeid -> xfail reason, for every test pytest counted as ``xfailed``. An
# xfail is NOT a skip and is deliberately not in ``SKIPPED``: no rule, pin or
# MEASURED entry has to excuse it and none of them can. But it is a test that
# handed back no verdict this pin can read — with ``run=False`` it never
# started — so the completeness claim withdraws on it. Kept separate from
# ``SKIPPED`` so that the two never get confused for one another again: the
# early return that used to stand here dropped the report entirely and let the
# claim be MADE over a test that never ran.
XFAILED: dict[str, str] = {}

# nodeids that entered their runtest protocol at all — setup counts, and so
# does an error in it. "The session did not silently drop you" is a weaker and
# different question from ``RAN``, and it is the one the order check asks.
STARTED: set[str] = set()

# Basenames of every test FILE this session collected, deselected, or skipped
# at collection. Basenames rather than nodeids because nodeids are relative to
# rootdir and the suite is run from several working directories.
SEEN_FILES: set[str] = set()

# nodeids pytest collected and then threw away (-k, -m, --deselect, --sw). A
# session that did this looked at the whole tree and discarded part of the
# answer, which is a different thing from never having looked.
DESELECTED: list[str] = []

# The selection filters this invocation actually carried, in the words the
# developer typed. Read off ``config``, not inferred from the fact that
# something called ``pytest_deselected``: ``_pytest/stepwise.py`` calls that
# hook too, so ``--sw`` was indistinguishable from ``-k`` to this recorder and
# the pin failed a clean tree telling the developer to drop a filter they had
# not passed.
USER_FILTERS: list[str] = []

# Every node the collect reports handed back, and every node that reported
# itself as a collector. Between them: the tests this session COLLECTED,
# whatever later became of them. Collect reports are emitted before any plugin
# can filter ``items``, which is what makes this the only record able to see a
# test that was collected and then silently dropped.
_COLLECTED_CHILDREN: list[str] = []
_COLLECTORS: set[str] = set()

# Appended to by the pin when it makes the completeness claim itself, so that
# the session-end guard below knows whether the claim still needs making.
CLAIM_MADE: list[str] = []

# (verdict, message) the session-end guard wants printed. Terminal summary
# only; the verdict itself is carried by ``session.testsfailed``.
_NOTES: list[tuple[str, str]] = []

# Set the first time :func:`_close_the_session` runs, so that the two places
# that call it — the end of the run loop, and :func:`pytest_sessionfinish` for
# the sessions that never reach the end of the run loop — cannot both make the
# claim. Not a lock: the ordinary path sets it and the backstop reads it.
_CLOSED: list[bool] = []

# The session, stashed for :func:`pytest_unconfigure`, which pytest hands only
# a ``config``. Two independent routes to it, because the interesting sessions
# are the ones that skip a hook: ``pytest_sessionstart`` is the documented one
# and a plugin that exits from ITS OWN ``pytest_sessionstart`` can get in front
# of it, so ``config.pluginmanager.get_plugin("session")`` — pytest registers
# the ``Session`` under that name in ``Session.__init__``, which runs before
# ``config._do_configure()`` — is the fallback.
_SESSION: list = []

# One entry per increment this guard made to ``session.testsfailed``. Without
# it the anchor cannot tell "this session is red because a TEST failed" (stay
# quiet, the reader has the news) from "this session is red because WE said so,
# and something has since un-said it" (say it again). Both look identical from
# ``session.testsfailed`` alone, and the second is the whole attack.
_OUR_FAILURES: list[bool] = []

# The verdicts that actually reached the reader's screen, appended by
# :func:`pytest_terminal_summary` at the moment it writes them — NOT by the
# code that decides to. ``_NOTES`` is a request; this is a receipt. A plugin
# that clears ``_NOTES`` from its own ``pytest_sessionfinish`` leaves the
# request looking honoured and the screen blank, which is measured in
# ``tests/test_skip_inventory.py``.
_DELIVERED: list[str] = []

# The one channel here that is not ``session.exitstatus``. Set to a path and
# the verdict is written there on the way out; see
# :func:`_write_the_verdict_somewhere_last_writer_wins_cannot_reach`.
_VERDICT_FILE_ENV = "STELLING_SKIP_INVENTORY_VERDICT"

_INVENTORY_MODULE = "test_skip_inventory.py"

_TESTS = pathlib.Path(__file__).resolve().parent

# The directory patterns pytest itself refuses to recurse into. Replaced in
# :func:`pytest_configure` by the value THIS invocation carries, so a project
# that sets ``norecursedirs`` in its ini gets its own answer rather than this
# one; the literal below is pytest's documented default, and it is what the
# scope check uses if it is ever asked before configure.
#
# Not cosmetic. ``files_the_suite_has`` used a bare ``rglob``, so a file pytest
# would never open counted as part of "the suite" and the two disagreed. Both
# directions were measured on the whole tree AT bd1fa04, whose clean suite is
# ``1993 passed, 2 skipped`` — the numbers below are that suite's and not this
# one's:
#
# * ``tests/build/test_zz_helper.py`` — a unique basename in a directory
#   pytest prunes — put a file in the suite that no session can ever collect,
#   so the completeness claim was WITHDRAWN on a clean whole run and the pin
#   silently stopped asserting anything. ``1992 passed, 3 skipped``, exit 0.
# * ``tests/.junk/test_affine.py`` — a COLLIDING basename in a pruned
#   directory — made a clean whole run FAIL on "two test files share a
#   basename". ``2 failed``, exit 1.
#
# Neither file was collectable; a scratch directory, a stale build tree or a
# vendored checkout under ``tests/`` is enough for either.
_NORECURSEDIRS: tuple[str, ...] = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)


def _file_of(nodeid: str) -> str:
    return pathlib.PurePosixPath(nodeid.split("::", 1)[0]).name


def _reason(longrepr) -> str:
    """pytest hands skips over as ``(path, lineno, "Skipped: <reason>")``."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        text = str(longrepr[2])
        prefix = "Skipped: "
        return text[len(prefix):] if text.startswith(prefix) else text
    return str(longrepr)


# --- what the session collected, what it ran, what it never got round to -----


def collected_items() -> list[str]:
    """The test nodeids this session collected, whatever became of them.

    Taken from ``report.result`` rather than from ``items``: everything that
    narrows a session does it to ``items``, so a record built there cannot see
    its own gap. Collectors (the session, the directory, each module, each
    class) report themselves as well as their children, so they are subtracted
    rather than guessed at.
    """
    return [nodeid for nodeid in _COLLECTED_CHILDREN if nodeid not in _COLLECTORS]


def pending_items(excluding: str = "") -> list[str]:
    """Collected tests that have not started and were not disclosed as gone.

    The pin asks this to find out whether it is really running last; the
    session-end guard asks it to find out whether anything vanished without
    being reported. ``excluding`` drops one module by substring — the pin's own
    module, whose remaining tests are legitimately still owed while the pin is
    the one running.
    """
    accounted = STARTED | set(DESELECTED)
    return [
        nodeid
        for nodeid in collected_items()
        if nodeid not in accounted and not (excluding and excluding in nodeid)
    ]


def deselected_items(excluding: str = "") -> list[str]:
    """Deselections, optionally minus one module's own.

    Deselecting the pin removes the CLAIM, and the session-end guard makes it
    instead; deselecting anything else removes the EVIDENCE, and nothing brings
    back a test that never ran.
    """
    return [nodeid for nodeid in DESELECTED if not (excluding and excluding in nodeid)]


def _pytest_would_prune(directory: pathlib.Path, pattern: str) -> bool:
    """``_pytest.pathlib.fnmatch_ex``, which is what ``norecursedirs`` is
    matched with: a pattern containing no separator is matched against the
    directory's NAME, and one containing a separator against its whole path.

    Reimplemented rather than imported because it is private, and pinned
    against the real thing by
    ``test_the_scope_check_prunes_exactly_what_pytest_prunes``, which builds a
    tree and asks an actual pytest what it collected.
    """
    if os.sep not in pattern and "/" not in pattern:
        return fnmatch.fnmatch(directory.name, pattern)
    if directory.is_absolute() and not os.path.isabs(pattern):
        pattern = f"*{os.sep}{pattern}"
    return fnmatch.fnmatch(str(directory), pattern)


def collectable_test_files() -> list[pathlib.Path]:
    """Every test file under ``tests/`` that pytest would actually open.

    ``rglob`` rather than ``glob`` because a future ``tests/sub/test_x.py``
    must not fall out of the scope check by living one directory down — and
    then ``norecursedirs`` on top of it, because ``rglob`` alone walks into
    ``.tox/``, ``build/``, ``dist/``, ``node_modules/``, ``venv/`` and every
    dot-directory, none of which pytest will ever collect. A file in one of
    those is not part of the suite by the only definition that matters here:
    no invocation can collect it, so no session can ever be complete with
    respect to it. See :data:`_NORECURSEDIRS` for the two measured failures.
    """
    return [
        path
        for path in sorted(_TESTS.rglob("test_*.py"))
        if not _in_a_pruned_directory(path)
    ]


def _in_a_pruned_directory(path: pathlib.Path) -> bool:
    """Is ``path`` under a directory pytest will not recurse into?

    Both sides resolved, because the callers reach ``tests/`` by different
    routes — this module's own ``__file__`` and another module's — and a path
    that is not relative to the root cannot be pruned by it.
    """
    root = _TESTS.resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return False
    directory = root
    for part in relative.parts[:-1]:
        directory = directory / part
        if any(_pytest_would_prune(directory, p) for p in _NORECURSEDIRS):
            return True
    return False


def files_the_suite_has() -> set[str]:
    """Every collectable test file under ``tests/``, by basename.

    Basenames because that is all a nodeid reliably gives back — they are
    relative to rootdir, which moves with the invocation. The key is therefore
    lossy in exactly one way, and that way is checked rather than assumed: two
    files with the same basename in different directories would collapse into
    one, so :func:`colliding_basenames` names them and the pin fails.
    """
    return {path.name for path in collectable_test_files()}


def colliding_basenames() -> list[str]:
    """Basenames belonging to more than one collectable file under ``tests/``."""
    counts: dict[str, int] = {}
    for path in collectable_test_files():
        counts[path.name] = counts.get(path.name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def unseen_files() -> list[str]:
    """Suite files this session never collected. Non-empty means narrowed."""
    return sorted(files_the_suite_has() - SEEN_FILES)


# --- the hooks ---------------------------------------------------------------


def pytest_configure(config) -> None:
    """The selection filters, read off the invocation that carried them."""
    global _NORECURSEDIRS
    try:
        _NORECURSEDIRS = tuple(config.getini("norecursedirs"))
    except (ValueError, KeyError):  # pragma: no cover - pytest always has it
        pass
    if getattr(config.option, "keyword", ""):
        USER_FILTERS.append(f"-k {config.option.keyword!r}")
    if getattr(config.option, "markexpr", ""):
        USER_FILTERS.append(f"-m {config.option.markexpr!r}")
    for pattern in getattr(config.option, "deselect", None) or ():
        USER_FILTERS.append(f"--deselect {pattern}")


def pytest_runtest_logreport(report) -> None:
    """Sort one report into the channel that describes it.

    The ``wasxfail`` arm used to be ``return``, which is the difference
    between "an xfail is not a skip" — true — and "an xfail is not anything",
    which let a session containing a test that never ran certify that it
    contained no undisclosed skip. Measured shapes, on the installed pytest,
    rather than reasoned from the docs:

    ================================  =======  ========  =========
    shape                             when     outcome   wasxfail
    ================================  =======  ========  =========
    ``xfail(run=False)``              setup    skipped   set
    ``pytest.xfail()`` in the body    call     skipped   set
    ``xfail`` marker, test failed     call     skipped   set
    ``xfail`` marker, test passed     call     passed    set
    ================================  =======  ========  =========

    The first three are what pytest counts as ``xfailed`` and are recorded as
    such; the last is ``xpassed``, a test that ran and passed, and is recorded
    as having RUN. Nothing here distinguishes the second row from the third,
    and nothing needs to: both handed back an expected-failure instead of a
    verdict, and the claim withdraws on either.
    """
    STARTED.add(report.nodeid)
    wasxfail = getattr(report, "wasxfail", None)
    if wasxfail is not None:
        if report.skipped:
            XFAILED.setdefault(report.nodeid, str(wasxfail))
        elif report.when == "call":
            RAN.add(report.nodeid)  # xpassed: the body ran, and it passed
        return
    if report.skipped:
        SKIPPED.setdefault(report.nodeid, _reason(report.longrepr))
    elif report.when == "call":
        RAN.add(report.nodeid)


def pytest_collectreport(report) -> None:
    if report.skipped:
        SKIPPED.setdefault(report.nodeid, _reason(report.longrepr))
    if report.nodeid:
        _COLLECTORS.add(report.nodeid)
        # `report.result`, not merely a nodeid: under `--lf`,
        # `LFPluginCollSkipfiles` returns `CollectReport(nodeid, "passed",
        # result=[])` for every file it has decided not to run. A recorder that
        # keyed on the nodeid alone read all 83 files as collected while 82
        # were never imported — so the withdrawal below never fired and
        # `pytest --lf` PASSED a session with an undisclosed skip planted in a
        # file it had not even opened. A file was collected if it yielded
        # something, or if its own import gate skipped it.
        #
        # The corner this leaves, deliberately: a test file containing NO tests
        # reports exactly like the one `--lf` declined to open, so it reads as
        # unseen and the claim is withdrawn until it has a test in it. That is
        # the safe direction — the withdrawal names the file — and the
        # alternative discriminators are all internals of the report object,
        # which would fail the other way round and in silence.
        if report.skipped or report.result:
            SEEN_FILES.add(_file_of(report.nodeid))
    for node in report.result:
        _COLLECTED_CHILDREN.append(node.nodeid)


def pytest_deselected(items) -> None:
    """``-k``, ``-m``, ``--deselect`` — and ``--sw`` — land here, and nowhere
    else.

    Recorded rather than ignored because a deselected test is one the session
    COLLECTED and then declined to run: its file is still part of what this
    session looked at, but its skip — if it has one — went unobserved. WHICH of
    those did it is not knowable from here, which is exactly why
    :data:`USER_FILTERS` is read off the config instead of inferred from this.
    """
    for item in items:
        DESELECTED.append(item.nodeid)
        SEEN_FILES.add(_file_of(item.nodeid))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items) -> None:
    """The inventory reads the session's outcomes, so it must run last.

    Collection order is alphabetical by file, which would put
    ``test_skip_inventory.py`` in the middle and let every module after it skip
    unobserved. ``sort`` is stable, so this moves that one file to the end and
    disturbs nothing else.

    ``trylast`` is deliberate and is NOT what makes the order safe: it only
    orders this against other non-wrapper hookimpls, while a
    ``wrapper=True, tryfirst=True`` one (``NFPlugin``, pytest-randomly) re-sorts
    after all of them regardless. Nothing makes the order safe. What holds is
    that the pin CHECKS it — :func:`pending_items` — and that the session-end
    guard makes the claim when the pin could not. This sort is here so that in
    an ordinary session the verdict carries the pin's own nodeid instead of
    arriving as a terminal summary section.

    The file set is taken here as well. This hook may run before or after the
    builtin ``-k``/``-m`` filtering depending on plugin registration order, so
    ``pytest_deselected`` adds the discarded ones back: between the two, every
    file the session collected is seen either way round.
    """
    for item in items:
        SEEN_FILES.add(_file_of(item.nodeid))
    items.sort(key=lambda item: _INVENTORY_MODULE in item.nodeid)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtestloop(session):
    """Close the session up where the record is finally complete.

    After the loop the session's record is complete, and this is the only point
    at which that is true. An exception on the way out — ``-x``, ``--maxfail``,
    ``--sw`` stopping at its failure, an interrupt — propagates through the
    ``yield`` and skips the close: an aborted session is a narrowed one and has
    nothing to claim.

    **That carve-out is safe for exactly one reason, and it is not the one the
    sentence above gives.** It is not that an aborted session has nothing to
    say; it is that every abort listed there leaves ``session.testsfailed > 0``
    or an ``ExitCode`` that is not ``OK``, so the session is already red and a
    banner would add nothing a reader does not have. Take the exit code away and
    the carve-out becomes a hole — which is precisely what
    ``pytest.exit(reason, returncode=0)`` does, and why
    :func:`pytest_sessionfinish` below exists. The setup-error row of the
    taxonomy at the top of this file leans on the same fact (``N errors``, and
    the session is non-zero), so it inherits the same single point of failure
    and is disclosed there too.

    **This hook used to re-sort ``session.items`` on the way in, and no longer
    does.** The sort was there to beat a ``wrapper=True, tryfirst=True``
    ``pytest_collection_modifyitems`` (``NFPlugin``, pytest-randomly), which
    re-orders after every ordinary collection hookimpl. Two measurements took
    it out:

    * a re-derived 2x2x2 mutation matrix over (this sort, :func:`pending_items`,
      the session-end guard) x the four ordering routes, discounting REDs that
      are only the guard's own self-test failing when the guard is removed:
      the check and the guard together close all four routes with this sort
      absent, and the sort alone closes only the reordering ones. What the
      sort bought was that the pin CLAIMED rather than DEFERRED — a nicer
      nodeid for the verdict, not the verdict.
    * against that, a cost. pytest-xdist freezes an index -> nodeid map at
      ``pytest_collection_finish`` and resolves work indices against
      ``session.items`` at run time. Sorting ``session.items`` HERE is after
      that point, so with any re-sorting plugin also present the two disagree.
      Measured with a stand-in that freezes the map exactly where
      ``xdist/remote.py`` sends it and resolves indices exactly where it
      resolves them, on a ten-item tree, all four cells:

          sort present + an NFPlugin-shaped re-sorter   10/10 mismatched
          sort present + no re-sorter                    0/10
          sort absent  + an NFPlugin-shaped re-sorter    0/10
          sort absent  + no re-sorter                    0/10

      xdist is not installed here, so whether the installed xdist indexes into
      ``session.items`` as its source says, and whether a controller would
      surface the desync, are UNVERIFIED. But the sort was buying a nodeid,
      and this is workers running tests the controller thinks are other tests.

    So the ordering is left to :func:`pytest_collection_modifyitems`, which
    sorts where every other plugin sorts and inside the window xdist's map is
    built from; when something re-sorts after it, the pin DEFERS and the claim
    is made below.
    """
    result = yield
    _close_the_session(session)
    return result


def pytest_sessionfinish(session, exitstatus) -> None:
    """The loop is not the only way out of a session, and one of the others
    keeps the exit code.

    ``pytest.exit(reason, returncode=0)`` raises ``Exit`` from wherever it is
    called. Inside the run loop that means two things at once, and the second is
    the one the taxonomy at the top of this file cannot see:

    * it propagates through :func:`pytest_runtestloop`'s ``yield``, so
      :func:`_close_the_session` never runs — the skips this session saw are in
      :data:`SKIPPED`, correctly recorded, and nothing reads them;
    * ``_pytest.main.wrap_session`` catches it and assigns
      ``session.exitstatus = exc.returncode``, which OVERRIDES the
      ``session.testsfailed`` that ``_main`` would otherwise have turned into
      ``ExitCode.TESTS_FAILED``. So the pytest-cov idiom the guard below uses to
      carry a verdict cannot reach the exit code on this path at all.

    Measured at a80d60c, same worktree, same plant, one ``-p`` apart::

        pytest -q -rs tests/test_affine.py         41 passed, 1 skipped  EXIT 1
                                                   and a banner naming the skip
        pytest -q -rs -p <exit0> tests/test_affine.py
                                                   41 passed, 1 skipped  EXIT 0
                                                   and no banner at all

    Byte-identical summary lines, opposite verdicts. On the whole tree, exiting
    after the first call phase: ``1 passed``, exit 0, no banner, 2023 of the
    2024 collected tests never run. ``pytest.exit`` is public API — the same
    class of thing as "a plugin that drops items", which is already one of the
    closed routes.

    So the close happens here as well as at the end of the loop. ``wrap_session``
    calls ``pytest_sessionfinish`` from a ``finally``, and this is the latest
    point at which a verdict can still be printed by the ORDINARY route — the
    terminal reporter's own ``pytest_sessionfinish`` is what calls
    ``pytest_terminal_summary``, so a verdict formed after this hook has to
    find its own way to the screen. That is why the close is here and not only
    in :func:`pytest_unconfigure`.

    **This is NOT the last hook, and the version of this docstring that said so
    was wrong twice over.** ``wrap_session``'s ``finally`` calls
    ``config._ensure_unconfigure()`` — and therefore ``pytest_unconfigure`` —
    AFTER this hook and BEFORE it reads ``session.exitstatus``; and this hook is
    not called at all on two ways out of a session. Both are measured in
    :func:`pytest_unconfigure`, which is where the anchor now is.

    **The carve-out is kept, and stated as what it is.** This does nothing when
    the session is already going to be non-zero: ``-x`` and ``--maxfail`` arrive
    as ``ExitCode.TESTS_FAILED``, ``--sw`` and an interrupt as
    ``ExitCode.INTERRUPTED``, an internal error as ``ExitCode.INTERNAL_ERROR``,
    and a session that is red says everything a banner would. What is left is
    exactly the abort that claims success, and that one is judged like any
    other complete-or-incomplete record: an ``Exit`` mid-loop leaves collected
    tests neither started nor deselected, which is the ``still_owed`` failure.

    **The route that beats THIS hook**, and it is the same route one level out:
    a plugin whose own ``pytest_sessionfinish`` raises
    ``pytest.exit(returncode=0)``. ``wrap_session`` catches that and re-assigns
    ``session.exitstatus`` from the returncode, undoing the assignment below;
    and the terminal reporter's own ``pytest_sessionfinish`` is a WRAPPER around
    every ordinary one, so an ``Exit`` raised inside the chain never reaches the
    code after its ``yield`` and ``pytest_terminal_summary`` is never called at
    all — whichever order the two ordinary hookimpls ran in. Measured at
    1b1c843, on ``tests/test_affine.py`` with an undisclosed skip planted::

        no plugin                       EXIT 1, 1 banner
        + exit0 from a sessionfinish    EXIT 0, 0 banners
        + the same, tryfirst            EXIT 0, 0 banners

    That route is answered in :func:`pytest_unconfigure`. What is NOT true, and
    was written here as a consolation, is that the defeat always leaves
    ``Exit: <reason>`` behind. It does when the plugin RAISES; a plugin's
    ``pytest_sessionfinish`` need not raise::

        def pytest_sessionfinish(session, exitstatus):
            session.exitstatus = 0

    Driven at 1b1c843, same plant: EXIT 0, the banner still printed, and **no
    ``Exit:`` line anywhere**. Add ``_NOTES.clear()`` and it is EXIT 0 with zero
    banners, whose ``diff`` against the unplugged session is exactly the five
    banner lines and nothing else — precisely the byte-identical green this file
    exists to end. So the consolation is withdrawn: the general sentence below
    covers this, and the specific one did not.

    So, plainly: this hook RESTS ON THE EXIT CODE, exactly as the ``-x``
    carve-out and the setup-error row of the taxonomy do. What breaks all three
    is the same thing — anything that assigns the process exit code after the
    verdict has been formed. :func:`pytest_unconfigure` moves the anchor later
    and adds a channel that is not the exit code at all; it does not make that
    sentence false, and see there for why nothing could.
    """
    if _CLOSED:
        return  # the run loop finished and the claim was made there
    if hasattr(session.config, "workerinput"):
        return  # a distributed worker claims nothing; see _close_the_session
    if exitstatus != pytest.ExitCode.OK or session.testsfailed:
        return  # already red: an aborted session that says so needs no banner
    _close_the_session(session)
    if session.testsfailed:
        # `_main` has already returned, so the idiom the guard uses cannot be
        # read any more; this is the assignment `wrap_session` returns.
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_sessionstart(session) -> None:
    """Stash the session for :func:`pytest_unconfigure`, which is handed a
    ``config`` and nothing else.

    Deliberately does nothing else. The anchor has a second, independent route
    to the same object (``config.pluginmanager.get_plugin("session")``) for the
    sessions that leave BEFORE this hook returns — see there.
    """
    _SESSION.append(session)


def pytest_unconfigure(config) -> None:
    """The anchor. Later than :func:`pytest_sessionfinish`, and not the last.

    ``_pytest.main.wrap_session``, in the ``finally`` this whole mechanism is
    built on::

            if initstate >= 2:
                try:
                    config.hook.pytest_sessionfinish(...)
                except exit.Exception as exc: ...
            config._ensure_unconfigure()      # pytest_unconfigure fires HERE
        return session.exitstatus              # and only THEN is this read

    Three things follow, and all three are measured rather than read off that
    listing. On ``tests/test_affine.py`` with an undisclosed skip planted,
    driven at 1b1c843 (the commit before this hook existed):

    * **This hook runs strictly after ``pytest_sessionfinish``.** A marker
      plugin printing from both prints ``MARKER-SESSIONFINISH-RAN`` and then
      ``MARKER-UNCONFIGURE-RAN``, in that order.
    * **An assignment made here still reaches the shell.** A plugin whose
      ``pytest_unconfigure`` sets ``session.exitstatus = 97`` produces EXIT 97.
    * **This hook is strictly more reliable than ``pytest_sessionfinish``.**
      ``initstate = 2`` is assigned only after ``pytest_sessionstart``
      RETURNS, so a ``pytest.exit(returncode=0)`` raised from inside
      ``pytest_sessionstart`` — or from ``pytest_configure``, where
      ``initstate`` is still 0 — makes ``if initstate >= 2`` False and
      ``pytest_sessionfinish`` is never called at all::

          marker only                EXIT 1  SESSIONFINISH-RAN, UNCONFIGURE-RAN
          marker + exit0 at start    EXIT 0  UNCONFIGURE-RAN only
          marker + exit0 at config   EXIT 0  UNCONFIGURE-RAN only

      That is a FIFTH zero-arriving abort, and it is outside the mechanism the
      other four are inside — it is not a report shape, and it is not the
      decision failing to run over a record; there IS no record. It also
      falsifies what the previous version of :func:`pytest_sessionfinish`
      claimed for itself — "on every way out of a session that got as far as
      ``pytest_sessionstart``" — because a session that exits FROM
      ``pytest_sessionstart`` got that far and was not covered. Off by one.

    **WHAT THIS ANCHOR IS NOT: last.** The version of this reasoning that
    shipped one commit ago ended "there is no hook after the last hook", and
    that was an unregistered sentence written alongside a measurement, which is
    how it survived. The general truth is stronger and survives its own test:

        ``pytest_cmdline_main`` as a ``wrapper=True`` hookimpl returns over
        everything here, and beyond it lie ``_pytest.config._main``,
        ``console_main``, ``atexit`` and ``os._exit``. **Every anchor has a
        later one.** ``session.exitstatus`` is a LAST-WRITER-WINS channel, so
        any mechanism that carries its verdict there is beatable by
        construction — including this one.

    Driven at 1b1c843, same plant: a ``pytest_cmdline_main`` wrapper returning
    98 gives EXIT 98 with the banner still on the screen, and a ``trylast``
    ``pytest_unconfigure`` assigning ``session.exitstatus = 0`` gives EXIT 0
    with the banner still on the screen. Moving the anchor here buys the four
    routes that exist; it does not and cannot buy the last word.

    Two of those four later points were DRIVEN and two were READ: the
    ``pytest_cmdline_main`` wrapper and the ``trylast`` unconfigure are the two
    above; ``atexit`` and ``os._exit`` are read off ``_pytest.config._main``
    and ``_console_main`` and are not measured here. The distinction is kept
    because collapsing it is the exact mistake this docstring is a repair
    for — a sentence that reasons its way past a measurement, in the same
    paragraph as one, reads like the measurement.

    **So the verdict is also carried off ``session.exitstatus`` entirely**, by
    :func:`_write_the_verdict_somewhere_last_writer_wins_cannot_reach`. See
    there for which of the three available channels was picked and why.

    The carve-out from :func:`pytest_runtestloop` is kept and is now exact.
    "Already red" may not mean "red because this guard said so": a plugin's
    ``pytest_sessionfinish`` that does nothing but ``session.exitstatus = 0``
    leaves ``session.testsfailed`` at 1 with the exit code at 0, which is
    indistinguishable from a real failure if the only thing you look at is
    ``testsfailed``. :data:`_OUR_FAILURES` is what tells the two apart, and
    without it this hook would fall silent on exactly the attack it is for.
    """
    session = _SESSION[-1] if _SESSION else config.pluginmanager.get_plugin("session")
    if session is None:  # no session was ever created; nothing to say
        return
    if hasattr(config, "workerinput"):
        return  # a distributed worker claims nothing; see _close_the_session
    verdict, message = _consult_the_pin(session)
    aborted = _red_for_a_reason_that_is_not_ours(session)
    _write_the_verdict_somewhere_last_writer_wins_cannot_reach(
        session, verdict, message, aborted
    )
    if aborted or verdict != "failed":
        return
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    if not _DELIVERED:
        # The ordinary route to the screen is `pytest_terminal_summary`, and it
        # is called from inside `pytest_sessionfinish` — which is over. A
        # verdict formed or un-printed after that point has to write itself.
        #
        # The provenance line is not decoration. "The banner appeared" and "the
        # banner appeared BECAUSE THE ORDINARY ROUTE WAS GONE" are different
        # observations, and a case that cannot tell them apart cannot tell a
        # session with a note-clearing plugin from a session without one.
        title = _TITLES.get(verdict, verdict)
        sys.stderr.write(
            f"\n{'=' * 12} {title} {'=' * 12}\n"
            "(written from pytest_unconfigure: pytest_terminal_summary, which "
            "pytest calls from inside pytest_sessionfinish, did not deliver "
            "this verdict)\n"
            f"{message}\n"
        )
        sys.stderr.flush()
        _DELIVERED.append(verdict)


def _red_for_a_reason_that_is_not_ours(session) -> bool:
    """Is this session already red for something this guard did not do?

    The carve-out the whole mechanism leans on — do not add a second verdict to
    a session pytest has already failed — read so that an attack cannot dress
    itself up as one. ``session.testsfailed`` minus this guard's own increments
    is what a TEST failing looks like; anything in ``exitstatus`` other than
    ``OK`` or the ``TESTS_FAILED`` this guard's own increment produces is an
    abort pytest is already reporting (``INTERRUPTED`` for ``--sw`` and an
    interrupt, ``NO_TESTS_COLLECTED``, ``USAGE_ERROR``, ``INTERNAL_ERROR``).
    """
    if session.testsfailed - len(_OUR_FAILURES) > 0:
        return True
    status = getattr(session, "exitstatus", pytest.ExitCode.OK)
    return status not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED)


def _consult_the_pin(session) -> tuple[str, str]:
    """Ask the decision function, over the record exactly as it stands now.

    One function, so that the run-loop close and the anchor cannot drift into
    asking different questions; it is deliberately free of any policy about
    what to DO with the answer, which is the part the two callers differ on.
    """
    no_call_phase = _no_call_phase_in_this_mode(session.config)
    try:
        import test_skip_inventory as inventory

        return inventory.the_claim_this_session_can_make(
            at_session_end=True, no_call_phase=no_call_phase
        )
    except Exception as exc:  # the pin is what says what a skip means
        return "failed", (
            f"the completeness pin could not be consulted at the end of a "
            f"session it was not part of: {exc!r}"
        )


def _write_the_verdict_somewhere_last_writer_wins_cannot_reach(
    session, verdict, message, aborted
) -> None:
    """The channel that is not the exit code, and why it is this one.

    ``session.exitstatus`` is last-writer-wins and the reader's screen is
    whatever the last hook to touch it left there. Three channels were
    available for a verdict that has to survive both, and this is the argument
    for the one taken:

    (a) **A file written on the way out, asserted separately by whoever ran
        pytest.** Taken. The path comes from the ENVIRONMENT, not from anything
        in the session, so a plugin cannot re-assign it the way it re-assigns
        an exit code — it cannot un-write what it does not know exists. And
        ABSENCE is itself a failure signal: a run in which this conftest was
        removed, ignored or never reached leaves no file, and the assertion
        that reads it fails. Every other channel here can only report; this one
        also reports that it did not run.
    (b) **A non-zero-only signal** — ``os.kill(os.getpid(), SIGTERM)`` or
        ``os._exit(1)``. Strictly stronger against re-assignment: nothing
        downstream gets to run at all. Rejected because "nothing downstream
        gets to run at all" is also its cost — ``os._exit`` skips ``atexit``,
        so coverage data, xdist teardown and pytest's own flushing are lost,
        and a TEST SUITE that kills its own process is a worse citizen than
        the defect it is closing. It is the right tool for a release gate and
        the wrong one for a conftest.
    (c) **A ``--strict``-style configuration assertion** — refuse to start
        unless the session is one that can support the claim. Structurally
        weaker than both: it is evaluated before anything has happened, so it
        cannot see what the session DID, and this whole file exists because
        "which tests skipped" is only observable by running.

    **The limit, stated rather than left to be found.** This is not
    adversary-proof and is not meant to be. A plugin that reads the same
    environment variable can delete the file — driven, and it does. What it
    defends against is the entire class this mechanism actually meets: a plugin
    that re-assigns an exit code, clears a note list, or exits a session early
    while knowing nothing about this pin. Against an adversary who has read
    this file, no in-process channel is defensible, and the honest answer is
    that the assertion has to live outside the process.

    Silent on ``OSError`` on purpose: an unwritable path leaves no file, and
    absence is already the signal.
    """
    destination = os.environ.get(_VERDICT_FILE_ENV)
    if not destination:
        return
    try:
        path = pathlib.Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"verdict={verdict}\n"
            f"exitstatus={int(getattr(session, 'exitstatus', -1))}\n"
            f"aborted={'yes' if aborted else 'no'}\n"
            f"---\n{message}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _no_call_phase_in_this_mode(config) -> bool:
    """``--collect-only`` / ``--setup-only`` / ``--setup-plan``.

    Read off the INVOCATION rather than inferred from an empty :data:`RAN`,
    which is the same channel :data:`USER_FILTERS` is read from and for the
    same reason: a session in which every collected test SKIPPED also has an
    empty ``RAN``, and that one has skips to judge and no tests owed. Inferring
    from the effect conflated them, and
    ``pytest --ignore=tests/test_skip_inventory.py -k <only skipping tests>``
    with an undisclosed skip planted was `1 skipped, 1983 deselected`, exit 0,
    with nothing printed (measured at bd1fa04).

    These three modes run no call phase, so a ``pytest.skip()`` in a test BODY
    cannot fire in them and "no undisclosed skip in this session" is not a
    claim any of them is entitled to make.

    **That is where the reason used to stop, and it was too narrow by one
    phase.** ``--setup-only`` EXECUTES fixture setup — that is the whole point
    of it — so a ``pytest.skip()`` in a FIXTURE fires under it, is reported,
    and is recorded here. With a fixture-level
    ``pytest.skip("a planted reason nobody disclosed")`` in
    ``tests/test_affine.py``, measured at b277083::

        pytest --setup-only -k uses_the_gate   1 skipped, 2012 deselected,
                                               EXIT 0, nothing printed
        pytest -k uses_the_gate                EXIT 1, banner naming the skip

    The plant was SELECTED, the skip FIRED, and the recorder HAD it. So this
    predicate no longer gates the guard's silence; it gates the CLAIM. A
    no-call-phase session still gets its disclosure half checked, and still
    fails loudly on a skip it saw and cannot explain — it just may not certify
    the suite.
    """
    return any(
        getattr(config.option, name, False)
        for name in ("collectonly", "setuponly", "setupplan")
    )


def _close_the_session(session) -> None:
    """Make the completeness claim here if the pin did not get to make it.

    Deliberately narrow: this is not a second copy of the pin, it is the same
    check called from the only place a filtered session cannot remove. It runs
    only when the pin did not run it, and only when the session is one that
    ran or skipped something at all.

    **What this guard is not allowed to do is fall silent**, which is what two
    of its early returns used to do. Both were measured AT bd1fa04 (clean
    suite ``1993 passed, 2 skipped``), on the whole tree, with
    ``pytest.skip("a planted reason nobody disclosed")`` planted in
    ``tests/test_affine.py``:

    * ``pytest --ignore=tests/test_skip_inventory.py`` — the pin's own file is
      not collected, so ``unseen_files()`` named it, "narrowed session" was the
      classification, and the guard returned before it could say anything.
      ``1981 passed, 3 skipped``, exit 0, the planted skip on the screen and
      no verdict anywhere. Same via ``--ignore-glob`` and via
      ``PYTEST_ADDOPTS``. That is the sibling of
      ``--deselect tests/test_skip_inventory.py``, which was closed, and it was
      QUIETER than the byte-identical green the whole repair was written to
      end.
    * ``--ignore=tests/test_skip_inventory.py -k <an expression selecting only
      tests that skip>`` — nothing reached its call phase, ``RAN`` was empty,
      and "nothing ran" was inferred from the effect. ``1 skipped, 1983
      deselected``, exit 0.

    The scope of a session and the DISCLOSURE of its skips are different
    questions, and only the first one is narrowed by narrowing the invocation.
    Whatever a session saw, it is in a position to say whether those skips were
    disclosed — so the pin is consulted for every session that saw anything,
    and ``the_claim_this_session_can_make`` answers the scope question itself,
    in the one place the answer is written down.

    Two silences are kept, deliberately, and they are the only two. A session
    that did not collect the whole tree and has nothing else to report says
    nothing: ``pytest tests/test_affine.py`` is the commonest invocation there
    is, and a banner naming the 82 files it did not run is a cost on every
    developer for news they already have. And a session that ran no call phase
    at all (``--collect-only`` and friends) says nothing when it has nothing to
    say, for the same reason. Both still FAIL loudly if a skip they DID see is
    undisclosed, which is the half that was actually missing — the second of
    them measurably so: ``pytest --setup-only`` executes fixture setup, and a
    ``pytest.skip()`` in a fixture used to vanish into
    :func:`_no_call_phase_in_this_mode`'s early return.

    **What the silences may not do is swallow a shortfall pytest cannot
    report.** They used to, because the decision function answered the SCOPE
    questions in the wrong order: ``unseen`` and ``filtered_out`` — both
    WITHDRAWALS, both silenced here — were tested before ``still_owed``, the
    undisclosed drop that keeps the failure. Measured at b277083 with a plugin
    that removes one item from ``items[:]`` and never calls
    ``pytest_deselected``::

        whole tree                        2008 passed, 3 skipped, EXIT 1, banner
        --ignore=tests/test_square_row.py 1991 passed, 3 skipped, EXIT 0, SILENT
        pytest tests/test_affine.py       40 passed,              EXIT 0, SILENT
        -k "not test_op_add"              2007 p, 3 s, 1 desel,   EXIT 0, SILENT

    Any narrowing at all bought a silent pass on the one shortfall the whole
    mechanism exists for. The order is fixed where it is made, in
    ``the_claim_this_session_can_make``: the undisclosed drop is decided
    BEFORE either withdrawal. The same four commands at a80d60c::

        whole tree                        2020 passed, 3 skipped, EXIT 1, banner
        --ignore=tests/test_square_row.py 2003 passed, 3 skipped, EXIT 1, banner
        pytest tests/test_affine.py       40 passed,              EXIT 1, banner
        -k "not test_op_add"              2019 p, 3 s, 1 desel,   EXIT 1, banner

    The counts in the second block are a80d60c's and are labelled as such
    rather than as "here", which is what they used to say: the suite grows, and
    a figure whose label is the tree it happens to be sitting in is a figure
    that goes wrong without anybody editing it. What the two blocks are FOR is
    the EXIT and the BANNER columns, which is what §53 means by recording
    failures rather than totals — those four cells are the finding, and they
    are the same at any suite size.
    """
    _CLOSED.append(True)
    # What the pin said, if it got to say anything — NOT a reason to stay quiet.
    #
    # This used to be `if CLAIM_MADE: return`, and the gap in it is that the pin
    # makes its claim from inside the session it is claiming about. It runs last
    # among FILES, not among TESTS: its own module still has tests after it (29
    # of the 41 in it, at a80d60c), and every one of them runs with the claim
    # already made and the guard already disarmed. Measured on the whole tree at
    # a80d60c, with `pytest.skip("a planted reason nobody disclosed")` appended
    # to `tests/test_skip_inventory.py` so that it fires after the pin:
    #
    #     pytest -q -rs   ->  2022 passed, 3 skipped, EXIT 0, no banner, and the
    #                         planted skip printed on the screen
    #
    # The skip was recorded in `SKIPPED` exactly as it should have been. Nothing
    # read it. That is not a report shape the taxonomy above could have caught —
    # the report landed in the right channel — it is the DECISION not running,
    # and the same line of code as the `--ignore` route this guard was written
    # for. So the claim is re-asked here, over the record as it finally stands,
    # and the guard stays quiet only when the answer is the one the pin already
    # delivered.
    #
    # The corner that leaves, deliberately: a session in which the pin ALREADY
    # failed and a second undisclosed skip lands afterwards is not told about
    # the second one, because the verdict did not change. It is already red and
    # already says undisclosed skips are present; fixing the first and re-running
    # names the second.
    already = CLAIM_MADE[-1] if CLAIM_MADE else ""
    # There is deliberately no "and nothing ran" return here. The old
    # `if not RAN: return` is the second silent route above; narrowing it to
    # `not RAN and not SKIPPED` would only have moved it, and would have left a
    # branch that suppresses the verdict for a session which selected nothing
    # (`pytest -k <matches nothing>`) — a session pytest already reports as
    # "no tests ran". It withdraws through the ordinary scope answer instead,
    # which is checked at the surface by
    # `test_the_session_end_guard_answers_every_shortfall[a-filter-that-selects-nothing]`.
    if hasattr(session.config, "workerinput"):
        # A pytest-xdist worker runs a share of the session, not the suite, and
        # reports its own share only. xdist is not installed here, so this is
        # the documented worker marker and a stand-in plugin of the same shape
        # is all that has been measured against it. KNOWN AND OPEN: with the
        # pin and an undisclosed skip on different workers, both workers exit
        # 0 and nothing is printed. The controller sees every worker's reports
        # and its own collection, so it is the place that could answer — that
        # is reasoned, not measured, because xdist is not installed here.
        return

    no_call_phase = _no_call_phase_in_this_mode(session.config)
    verdict, message = _consult_the_pin(session)
    if already and verdict == already:
        return  # the pin said this already, and nothing has changed since
    if verdict == "withdrawn" and (no_call_phase or unseen_files()):
        # the developer narrowed collection, or asked for a mode that runs
        # nothing; neither is news. A FAILED verdict is never silenced here.
        return
    _NOTES.append((verdict, message or "no undisclosed skip in this session."))
    if verdict == "failed":
        # The pytest-cov idiom: the exit code follows `session.testsfailed`,
        # which `_pytest.main._main` turns into ExitCode.TESTS_FAILED. There is
        # no test to fail here, and that is the situation being reported.
        session.testsfailed += 1
        # …and record that THIS is who put it there. `pytest_unconfigure` has to
        # be able to tell a session that is red because a test failed from one
        # that is red because this guard said so and a plugin has since un-said
        # it, and `session.testsfailed` alone cannot tell them apart.
        _OUR_FAILURES.append(True)


_TITLES = {
    "made": "skip inventory: pin absent from this session, claim made at the end",
    "withdrawn": "skip inventory: completeness claim WITHDRAWN",
    "failed": "skip inventory: completeness claim FAILED at the end of the session",
}


def pytest_terminal_summary(terminalreporter) -> None:
    """Print the session-end verdict, including when it is fine.

    Printed even when nothing is wrong, on purpose: a session that removed the
    pin from ``items`` used to produce a summary line byte-identical to a clean
    whole run, and "the check that was silently dropped says so out loud" is
    most of the repair.

    :data:`_DELIVERED` is appended HERE, at the moment the bytes go out, and
    not where the note is queued. ``_NOTES`` is a request and this is the
    receipt, and the difference is a measured attack: a plugin's
    ``pytest_sessionfinish`` that clears ``_NOTES`` leaves the request looking
    honoured and the screen blank.
    """
    for verdict, message in _NOTES:
        terminalreporter.write_sep(
            "=", _TITLES.get(verdict, verdict), red=verdict == "failed"
        )
        terminalreporter.write_line(message)
        _DELIVERED.append(verdict)


# ---------------------------------------------------------------------------
# One shared helper for the tripwire's pytester tests, which is here because
# both of them need it and it must not drift between them.
# ---------------------------------------------------------------------------

TRIPWIRE_PLUGIN = "stelling._tripwire.plugin"


def tripwire_plugin_args() -> tuple[str, ...]:
    """``("-p", <plugin>)``, or ``()`` if the entry point already registers it.

    MEASURED, AND IT IS NOT A CONVENIENCE. A nested ``pytester`` session loads
    setuptools entry points of its own, so in an environment where stelling is
    actually INSTALLED the plugin is already registered there under its entry
    point name — and adding ``-p stelling._tripwire.plugin`` on top raises
    ``ValueError: Plugin already registered under a different name:
    stelling_overflow=...``. Seventeen tests failed that way in a throwaway
    venv with a real ``pip install -e``, and passed in the two shared dev
    venvs, which have the source on ``PYTHONPATH`` and no distribution
    metadata. CI installs with ``-e``, so CI is the installed case.

    So the argument is decided by asking the environment rather than by
    assuming either answer.

    AND THE QUESTION HAD TO BE THE RIGHT ONE. This asked *"is the distribution
    installed?"* when what it needs is *"will the nested session AUTOLOAD
    it?"*, and the two answers differ under
    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, a common CI hygiene setting: the
    entry point is still declared, so the probe below said "already
    registered", no ``-p`` was passed, and the nested sessions ran with no
    tripwire in them at all. Measured in an installed environment: **17
    failed**, the same seventeen the entry-point probe was added to fix. The
    environment variable is checked first for that reason.
    """
    import importlib.metadata
    import os

    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        return ("-p", TRIPWIRE_PLUGIN)
    for entry in importlib.metadata.entry_points(group="pytest11"):
        if entry.value.split(":")[0] == TRIPWIRE_PLUGIN:
            return ()
    return ("-p", TRIPWIRE_PLUGIN)


def xdist_plugin_args() -> tuple[str, ...]:
    """``("-p", "xdist")`` where a nested session would not autoload it.

    Same question as :func:`tripwire_plugin_args` about a different plugin.
    xdist reaches a session by ``pytest11`` entry point exactly as the
    tripwire does, so ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` takes ``-n 2`` away
    from the nested sessions too — and a test that asks for two workers and
    gets none reports what a one-process run reports, which is not nothing but
    is not what it claims to measure.
    """
    import os

    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        return ("-p", "xdist")
    return ()


# ---------------------------------------------------------------------------
# The process-global state guard, registered here because an autouse fixture
# has to live in a conftest to reach the whole tree.
# ---------------------------------------------------------------------------
#
# `tests/_state_guard.py` holds the inventory, the readers, the exemption list
# and the decision; this file holds one line of wiring, so that the guard can
# also be loaded as a plugin (`-p _state_guard`) by the nested sessions in
# `tests/test_state_guard.py` that prove it fires. Same module, both routes:
# a copy of the fixture living in a test's temporary tree would be proving
# things about the copy.
#
# LOADED BY PATH RATHER THAN BY NAME. `import _state_guard` works only once
# pytest has put `tests/` on `sys.path`, which it does for test modules and
# for this conftest — but relying on that would make the import order part of
# the contract, and `sys.path` is itself process-global state this file has
# opinions about. The path is right here next to us.

def _load_sibling(name: str):
    import importlib.util

    path = _TESTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


_state_guard_module = _load_sibling("_state_guard")

#: The autouse fixtures themselves, re-exported into this conftest's namespace
#: so pytest registers them for `tests/` and everything below it. See
#: `tests/_state_guard.py` for what they watch and what they do not.
#:
#: TWO, AND THE SECOND IS NOT A DUPLICATE. The function-scoped one brackets a
#: test and the fixtures that test owns; a MODULE-scoped fixture that never
#: restores is set up before its `before` and torn down after its `after`, so
#: it escaped every window the first one has. Measured: the restore deleted
#: from `test_0_2_0_regression.py`'s module-scoped `_x64` left the file at
#: `21 passed` with `jax_enable_x64` True at session finish and nothing said
#: so. The module-scoped guard reports what moved OUTSIDE every test of a
#: module — where the state sits whenever no test is running, which includes
#: before the first one and after the last — so a test's own change is outside
#: its reading by construction. It used to suppress by entry NAME instead,
#: which meant one exempted or xfailing test blinded it to every
#: `STELLING_*`/`JAX_*` key there is; and it used to fold the WHOLE reading of
#: an entry whenever one part of it moved, which put a test's leak back into
#: the module's own report. `tests/_state_guard.py` carries both drives.
state_guard = _state_guard_module.state_guard
module_state_guard = _state_guard_module.module_state_guard


def deterministic_order_args() -> tuple[str, ...]:
    """``("-p", "no:randomly")`` — what a NESTED session must always carry.

    A nested session is not a small copy of the outer one. The sessions this
    suite spawns plant two or three tests whose ORDER is the property being
    measured — ``test_tripwire_plugin.py``'s ``DETACHES_MIDWAY`` arms, detaches
    in test 2, and asserts that test 3's constant is not reported; the
    miniature sessions in ``test_skip_inventory.py`` assert which test the
    completeness pin runs after. Shuffling those does not test anything
    harder; it destroys the fixture.

    AND THE OUTER SESSION'S ``-p no:randomly`` DOES NOT REACH THEM.
    ``pytester.runpytest`` builds a fresh ``Config`` and autoloads entry-point
    plugins, and ``runpytest_subprocess`` starts a fresh interpreter, so an
    installed ``pytest-randomly`` is active in the child whatever the parent
    was told. MEASURED, in a venv with ``pytest-randomly`` 4.1.0 installed and
    ``-p no:randomly`` on the OUTER command, three consecutive runs of
    ``tests/test_tripwire_plugin.py`` alone::

        2 failed, 17 passed
        2 failed, 17 passed
        1 failed, 18 passed

    — and 19 passed, every time, in a venv without it. This is what the
    `random-order` lane in ci.yml would otherwise have reported as an
    order-dependent defect in stelling, and it is neither: it is the
    randomiser reaching a session that is about ordering.

    Always the flag and never a probe, unlike its two siblings above. ``-p
    no:X`` BLOCKS a plugin name and does not require it to exist — driven with
    ``pytest-randomly`` absent, exit 0 — so there is no question to ask the
    environment and therefore no answer to get wrong.
    """
    return ("-p", "no:randomly")


@contextlib.contextmanager
def lowered_perimeter():
    """The dunder perimeter DOWN for this block, and back exactly as found.

    Yields the faces that were armed on the way in — ``()`` when there was
    nothing to lower, in which case this does nothing at all.

    **WHY THIS EXISTS RATHER THAN A REGION.** ``expected_truncation`` is one
    declaration covering BOTH runtime instruments, by design. That is right
    for code whose subject is a narrowing, and it is exactly wrong for a test
    whose subject is *the eager detector firing on* a narrowing: a region
    opened to permit Mode 3's refusal silences Mode 2 in the same breath, and
    the inventory in ``tests/test_tripwire_gate_coverage.py`` then reads every
    ``raises`` row as ``silent`` — a detector reported blind by the very
    declaration added to keep it running. So where an instrument UNDER the
    perimeter is the subject, the perimeter is taken out of the way instead,
    exactly as that file's ``detached()`` takes Mode 2 out of the way when the
    subject is the unpatched program.

    **AND WHY IT IS HERE RATHER THAN WRITTEN OUT FIVE TIMES.** Three files
    call it — the perimeter's own autouse fixture, and two places in the eager
    inventories — and hand-rolled surgery on ``perimeter._installed`` and ``._owners``
    is precisely what this batch was faulted for: the unconditional restore in
    ``tests/test_narrowing_perimeter.py::_isolate`` unhooked the session's own
    hold and left ~4,300 tests running unprotected with nothing red. One
    implementation, which hands the hold back through the shipped ``arm()``
    with its self-check, restores the owner list BY IDENTITY, and **raises**
    if it cannot — a lowering that fails to re-arm must be a red test here and
    not a silent hole in everything after it.

    **AND IT RAISES ON A PARTIAL HAND-BACK, NOT ONLY A TOTAL ONE**, which is
    the failure worth designing for: a lowering that comes half-way back reads
    exactly like one that came all the way back. Four things are checked and
    ``status.armed`` sees only the first — that ``arm()`` agreed, that every
    face is installed again, that ``live_check()`` says every slot IS the live
    binding, and that each slot's SAVED ORIGINAL is the same object that was
    lowered. The last has no other witness: something that binds over a slot
    while the perimeter is down becomes the "original" the next ``arm()``
    captures, the self-check still passes (an interloper that delegates still
    refuses the reference defect), every other reading says ``armed`` — and
    the object a later ``disarm()`` restores is no longer jax's.
    """
    # IMPORTED HERE AND NOT AT MODULE SCOPE. Every one of this tree's 146 test
    # files imports this conftest, the zero-dep lane included, and
    # `perimeter.py` is only numpy-free because it binds the predicate lazily.
    # A module-scope import would still be safe today and would be a standing
    # invitation to stop being; this is also why this is a plain function and
    # not a fixture -- nothing here runs, arms or collects anything unless a
    # caller enters the block.
    from stelling._tripwire import perimeter

    faces = perimeter.armed_faces()
    if not faces:
        yield ()
        return
    # AN OWNERLESS INSTALL IS POSSIBLE and is not this helper's to diagnose:
    # `arm()` registers its owner only after the self-check passes, so a
    # `BaseException` between the two leaves faces installed with `_owners`
    # empty. Re-arming under a fresh token puts the SLOTS back, which is what
    # everything after this block depends on, and leaves the owner list as
    # empty as it was found.
    owners = list(perimeter._owners)

    # WHAT MUST COME BACK, CAPTURED BY IDENTITY BEFORE ANYTHING MOVES, and it
    # is the SAVED ORIGINAL rather than the wrapper: the wrapper is a fresh
    # closure on every `arm()` and its identity is meaningless across a
    # lowering, while the original is jax's own function and is what `disarm()`
    # will one day put back. See `_hand_back` for the partial failure this
    # capture is the only thing that can see.
    originals = {
        face: {
            slot: original
            for slot, (original, _wrapper) in perimeter._installed[face]["slots"].items()
        }
        for face in faces
    }

    def _hand_back():
        """Re-arm and PROVE it, or raise. Never returns having half-worked."""
        for face in list(perimeter._installed):
            perimeter._restore_face(face)
        del perimeter._owners[:]
        status = perimeter.arm(faces, owner=owners[0] if owners else object())
        del perimeter._owners[:]
        perimeter._owners.extend(owners)

        # FOUR SEPARATE FAILURES, and `status.armed` alone sees only the first.
        # A hand-back that half-worked is worse than one that did not: the
        # owner list is back, `armed_faces()` reads right, and the protection
        # every later test depends on is not there.
        wrong = []
        if not status.armed:
            wrong.append(f"arm() refused [{status.code}]: {status.detail}")
        if perimeter.armed_faces() != faces:
            wrong.append(
                f"faces came back as {perimeter.armed_faces()}, not {faces}"
            )
        live = perimeter.live_check()
        if live != "armed":
            wrong.append(f"live_check() is {live!r}, not 'armed'")
        for face, slots in originals.items():
            entry = perimeter._installed.get(face)
            if entry is None:
                continue  # already named by the faces check above
            for slot, original in slots.items():
                back = entry["slots"].get(slot, (None, None))[0]
                if back is not original:
                    # THE SILENT ONE. Something bound over the slot while the
                    # perimeter was down, so `arm()` captured THAT as the
                    # original and succeeded -- self-check included, because
                    # an interloper that delegates still refuses the reference
                    # defect. Every reading above says "armed"; the object
                    # `disarm()` will restore is no longer jax's.
                    wrong.append(
                        f"{face}.{slot} was re-armed over a DIFFERENT object "
                        f"than the one lowered: something rebound it while "
                        f"the perimeter was down"
                    )
        if wrong:
            raise AssertionError(
                "the perimeter was lowered and did not come back intact:\n  "
                + "\n  ".join(wrong)
                + "\nEverything after this point would have run unprotected."
            )

    del perimeter._owners[:]
    for face in list(perimeter._installed):
        perimeter._restore_face(face)
    try:
        yield faces
    finally:
        # A raise in here during an unwind CHAINS onto the block's own
        # exception rather than replacing it -- Python prints both -- and that
        # is the right order: a leaked perimeter outlives whatever the block
        # was failing about.
        _hand_back()
