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

Two more things are recorded, and neither of them is an outcome.

**How much of the suite this session was allowed to look at.** The inventory's
completeness half claims something about THE SUITE's skip set, and a session
narrowed by ``-k``, by an explicit path or by ``--lf`` can only support a claim
about what it collected. Nothing in pytest's report distinguishes the two —
``1927 passed, 2 skipped`` reads exactly like a whole run — so the scope is
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

So the order is made robust and then CHECKED, because robust-and-unchecked is
what the first two routes already defeated:

* :func:`pytest_runtestloop` re-sorts ``session.items`` after collection is
  completely over, which is later than any ``pytest_collection_modifyitems``
  wrapper can reach;
* :func:`pending_items` lets the pin ask, at the moment it runs, whether the
  session still owed anything — so an ordering hook that lost a fight becomes
  a withdrawal instead of a silent pass;
* and if the pin never made its claim — reordered too early, filtered out of
  ``items``, or deselected — the same claim is made HERE, at the end of the
  session, where the record is complete. That is the one place an ``items``
  filter cannot reach, because removing a conftest is not something a filter
  can do.

The session's skips are recorded as pytest reports them; see
``tests/test_skip_inventory.py`` for what is done with them, and for why this
cannot be a static read of the tree.
"""

from __future__ import annotations

import pathlib

import pytest

# nodeid -> skip reason. Test nodeids and (for module-level gates) file paths.
SKIPPED: dict[str, str] = {}

# nodeids that reached their call phase, whatever the verdict. "Did not skip"
# is half of what the inventory asserts, so passing is recorded too.
RAN: set[str] = set()

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

_INVENTORY_MODULE = "test_skip_inventory.py"

_TESTS = pathlib.Path(__file__).resolve().parent


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


def files_the_suite_has() -> set[str]:
    """Every test file under ``tests/``, by basename.

    Basenames because that is all a nodeid reliably gives back — they are
    relative to rootdir, which moves with the invocation. ``rglob`` rather than
    ``glob`` because a future ``tests/sub/test_x.py`` must not fall out of the
    scope check by living one directory down. The key is therefore lossy in
    exactly one way, and that way is checked rather than assumed: two files
    with the same basename in different directories would collapse into one, so
    :func:`colliding_basenames` names them and the pin fails.
    """
    return {path.name for path in _TESTS.rglob("test_*.py")}


def colliding_basenames() -> list[str]:
    """Basenames belonging to more than one file under ``tests/``."""
    counts: dict[str, int] = {}
    for path in _TESTS.rglob("test_*.py"):
        counts[path.name] = counts.get(path.name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def unseen_files() -> list[str]:
    """Suite files this session never collected. Non-empty means narrowed."""
    return sorted(files_the_suite_has() - SEEN_FILES)


# --- the hooks ---------------------------------------------------------------


def pytest_configure(config) -> None:
    """The selection filters, read off the invocation that carried them."""
    if getattr(config.option, "keyword", ""):
        USER_FILTERS.append(f"-k {config.option.keyword!r}")
    if getattr(config.option, "markexpr", ""):
        USER_FILTERS.append(f"-m {config.option.markexpr!r}")
    for pattern in getattr(config.option, "deselect", None) or ():
        USER_FILTERS.append(f"--deselect {pattern}")


def pytest_runtest_logreport(report) -> None:
    STARTED.add(report.nodeid)
    if getattr(report, "wasxfail", None) is not None:
        return  # an xfail is reported as skipped and is not one
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
    after all of them regardless. :func:`pytest_runtestloop` is what holds, and
    the pin checks the result rather than trusting either of them.

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
    """Order the pin last where no ``items`` hook can reach it, then close up.

    Everything that reorders or filters a session does it from
    ``pytest_collection_modifyitems``. By the time this hook is entered
    collection is over and ``session.items`` is the list the loop will walk, so
    the sort here survives a ``wrapper=True, tryfirst=True`` reorder that beats
    the collection hook — measured against ``--nf`` and against a stand-in for
    pytest-randomly, both of which ran the pin first before this existed.

    After the loop the session's record is complete, and this is the only point
    at which that is true. An exception on the way out — ``-x``, ``--maxfail``,
    ``--sw`` stopping at its failure, an interrupt — propagates through the
    ``yield`` and skips the close: an aborted session is a narrowed one and has
    nothing to claim.
    """
    session.items.sort(key=lambda item: _INVENTORY_MODULE in item.nodeid)
    result = yield
    _close_the_session(session)
    return result


def _close_the_session(session) -> None:
    """Make the completeness claim here if the pin did not get to make it.

    Deliberately narrow: this is not a second copy of the pin, it is the same
    check called from the only place a filtered session cannot remove. It runs
    only when the pin did not run it, and only when the session is one the
    claim could be made from at all.
    """
    if CLAIM_MADE:
        return  # the pin ran with a complete record and said its piece
    if not RAN:
        return  # --collect-only, --setup-plan, an empty selection: nothing ran
    if hasattr(session.config, "workerinput"):
        # A pytest-xdist worker runs a share of the session, not the suite, and
        # reports its own share only. xdist is not installed here, so this is
        # the documented worker marker and a stand-in plugin of the same shape
        # is all that has been measured against it.
        return
    if unseen_files():
        return  # narrowed session: the claim is withdrawn, and correctly so

    try:
        import test_skip_inventory as inventory

        verdict, message = inventory.the_claim_this_session_can_make(
            at_session_end=True
        )
    except Exception as exc:  # the pin is what says what a skip means
        verdict = "failed"
        message = (
            f"the completeness pin could not be consulted at the end of a "
            f"session it was not part of: {exc!r}"
        )
    _NOTES.append((verdict, message or "no undisclosed skip in this session."))
    if verdict == "failed":
        # The pytest-cov idiom: the exit code follows `session.testsfailed`,
        # which `_pytest.main._main` turns into ExitCode.TESTS_FAILED. There is
        # no test to fail here, and that is the situation being reported.
        session.testsfailed += 1


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
    """
    for verdict, message in _NOTES:
        terminalreporter.write_sep(
            "=", _TITLES.get(verdict, verdict), red=verdict == "failed"
        )
        terminalreporter.write_line(message)
