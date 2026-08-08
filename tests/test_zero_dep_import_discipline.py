# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""No test module may import a heavy dependency at module scope.

The zero-dep CI job installs stelling and nothing else. A bare
``import jax`` at module scope in a test file is a COLLECTION error there, and
collection errors abort the whole run — so one unguarded import takes the
entire zero-dep job down, not just its own module. That is why this is checked
rather than remembered: the failure is disproportionate to the mistake and
invisible in any environment that has jax.

It has happened at least three times: four modules at once in one session, one
importing ``maddening``, and ``test_scatter_row_gates.py`` importing ``jax``.
Each was found by CI rather than locally, because locally jax is always there.

The rule: reach a heavy dependency through ``pytest.importorskip``, and place
that call BEFORE the first import that needs it.

This module is deliberately dependency-free and reads the files as text. It
must never import the things it is checking for. It does import
``tests/conftest.py``, which is one of the files it checks — that costs
nothing, because a conftest carrying a heavy import kills the zero-dep session
before any test of it could run (``ImportError while loading conftest``, exit
4), while in an environment that HAS the dependency the import succeeds and
the text read below still catches it.
"""
from __future__ import annotations

import pathlib
import re

# The one import, and it is the runner's own conftest: dependency-free
# (``__future__``, ``fnmatch``, ``os``, ``pathlib``, ``pytest``) and already
# imported by the collection that imports this file. What it supplies is the
# ``norecursedirs`` pruning, so that "a file of this suite" means the same
# thing here as it does to the skip inventory's scope check.
from conftest import _in_a_pruned_directory

# `hypothesis` is here for the same reason as the other three and one more:
# it is a DEV-GROUP dependency, so the zero-dep job and both shared jax venvs
# lack it, and an unguarded `from hypothesis import given` at module scope in
# `tests/property/` would abort collection for the whole suite rather than skip
# its own module. Measured shape, same as jax's.
HEAVY = ("jax", "maddening", "mime", "hypothesis")

# a module-scope import: column 0, not inside a function, try, or if
_MODULE_SCOPE_IMPORT = re.compile(
    r"^(?:import|from)\s+(" + "|".join(HEAVY) + r")(?:\s|\.|$)"
)
_SKIP = re.compile(r"importorskip\(\s*[\"'](" + "|".join(HEAVY) + r")[\"']")


def _scanned():
    """Every file in ``tests/`` that pytest imports for itself.

    ``conftest.py`` is in scope and is the WORST case, not an edge one. Both
    blast radii measured on a four-module tree, under DEFAULT flags:

    * an unguarded import in a test module — ``ERROR tests/test_broken.py``,
      ``Interrupted: 1 error during collection``, exit 2. The other three
      modules' twelve tests were collected and NONE of them ran.
    * the same import in ``tests/conftest.py`` — ``ImportError while loading
      conftest``, exit 4. Nothing was collected and there was no session.

    So under default flags both radii are total, and an earlier version of
    this docstring was wrong to say the module case leaves "the other modules'
    results still on the screen": that is true only under
    ``--continue-on-collection-errors`` (``12 passed, 1 error``, exit 1), which
    neither CI nor a plain ``pytest`` uses. What actually separates them is
    that the module error is attributable to a file and RECOVERABLE by that
    flag, while the conftest error is neither — it stays exit 4 with that flag
    set, because pytest never gets far enough for a collection error to be
    something it could continue past.

    ``tests/conftest.py`` was out of scope only because there was no such file
    until the skip inventory needed one. What it imports today is
    ``__future__``, ``pathlib`` and ``pytest`` — the runner itself, which is
    present wherever a conftest is being imported at all — and this is what
    keeps a heavy one out.
    """
    here = pathlib.Path(__file__).parent
    # rglob, not glob: a module one directory down is imported by exactly the
    # same collection and takes the run down in exactly the same way, and a
    # scope that stops at the top level would say nothing about it.
    files = sorted(here.rglob("test_*.py")) + sorted(here.rglob("conftest.py"))
    # …and then only the ones pytest will actually open. `__pycache__` was
    # subtracted here from the start; the rest of `norecursedirs` was not, so a
    # `tests/build/` or a `tests/.venv/` with a module-scope `import jax` in it
    # failed this test for a file no collection can ever reach. Same predicate
    # as the skip inventory's scope check, from the same place, so the two
    # cannot disagree about what "a file of this suite" means.
    return [
        path
        for path in files
        if "__pycache__" not in path.parts and not _in_a_pruned_directory(path)
    ]


def _offenders():
    bad = []
    for path in _scanned():
        if path.name == pathlib.Path(__file__).name:
            continue
        guarded: dict[str, int] = {}
        for lineno, line in enumerate(
            path.read_text().splitlines(), start=1
        ):
            m = _SKIP.search(line)
            if m:
                guarded.setdefault(m.group(1), lineno)
            m = _MODULE_SCOPE_IMPORT.match(line)
            if m:
                dep = m.group(1)
                # guarded only if the importorskip for THAT dep came first
                if guarded.get(dep, 1 << 30) > lineno:
                    bad.append((path.name, lineno, dep, line.strip()))
    return bad


def test_no_test_module_imports_a_heavy_dependency_unguarded():
    bad = _offenders()
    assert not bad, (
        "these module-scope imports abort collection in the zero-dep CI job, "
        "taking the whole run down rather than skipping their own module — "
        "use `dep = pytest.importorskip(\"dep\")` above the first import that "
        "needs it:\n"
        + "\n".join(
            f"  {name}:{lineno}  {src}" for name, lineno, _dep, src in bad
        )
    )


def test_the_checker_can_actually_see_an_offender():
    """Anti-vacuity: without this, a checker whose regex matches nothing and a
    codebase with no offenders produce the same empty list."""
    assert _MODULE_SCOPE_IMPORT.match("import jax")
    assert _MODULE_SCOPE_IMPORT.match("import jax.numpy as jnp")
    assert _MODULE_SCOPE_IMPORT.match("from jax import lax")
    assert _MODULE_SCOPE_IMPORT.match("import maddening")
    # and must NOT fire on the legitimate forms
    assert not _MODULE_SCOPE_IMPORT.match("    import jax")  # inside a function
    assert not _MODULE_SCOPE_IMPORT.match("import jaxlib_unrelated")
    assert not _MODULE_SCOPE_IMPORT.match("# import jax")
    assert _SKIP.search('jax = pytest.importorskip("jax")')
    # and the property suite's own two forms, which is where `hypothesis`
    # entered HEAVY
    assert _MODULE_SCOPE_IMPORT.match("from hypothesis import given")
    assert _MODULE_SCOPE_IMPORT.match("import hypothesis")
    assert _SKIP.search('pytest.importorskip("hypothesis", reason="needs hypothesis")')


def test_at_least_one_module_is_actually_scanned():
    """A glob that matched nothing would also report zero offenders.

    The other half — that the glob does not match MORE than pytest opens — is
    pinned where the shared predicate lives, by
    ``test_the_scope_check_prunes_exactly_what_pytest_prunes``, which compares
    it against a real pytest's collection of a probe tree.
    """
    scanned = _scanned()
    assert len(scanned) > 20
    # and the conftest specifically, since it is the file whose failure is
    # total and the one the glob used to miss
    assert pathlib.Path(__file__).parent / "conftest.py" in scanned
