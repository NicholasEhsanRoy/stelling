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
"""

from __future__ import annotations

# nodeid -> skip reason. Test nodeids and (for module-level gates) file paths.
SKIPPED: dict[str, str] = {}

# nodeids that reached their call phase, whatever the verdict. "Did not skip"
# is half of what the inventory asserts, so passing is recorded too.
RAN: set[str] = set()

_INVENTORY_MODULE = "test_skip_inventory.py"


def _reason(longrepr) -> str:
    """pytest hands skips over as ``(path, lineno, "Skipped: <reason>")``."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        text = str(longrepr[2])
        prefix = "Skipped: "
        return text[len(prefix):] if text.startswith(prefix) else text
    return str(longrepr)


def pytest_runtest_logreport(report) -> None:
    if getattr(report, "wasxfail", None) is not None:
        return  # an xfail is reported as skipped and is not one
    if report.skipped:
        SKIPPED.setdefault(report.nodeid, _reason(report.longrepr))
    elif report.when == "call":
        RAN.add(report.nodeid)


def pytest_collectreport(report) -> None:
    if report.skipped:
        SKIPPED.setdefault(report.nodeid, _reason(report.longrepr))


def pytest_collection_modifyitems(items) -> None:
    """The inventory reads the session's outcomes, so it must run last.

    Collection order is alphabetical by file, which would put
    ``test_skip_inventory.py`` in the middle and let every module after it
    skip unobserved. ``sort`` is stable, so this moves that one file to the
    end and disturbs nothing else.
    """
    items.sort(key=lambda item: _INVENTORY_MODULE in item.nodeid)
