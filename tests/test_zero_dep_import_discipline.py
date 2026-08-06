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
must never import the things it is checking for.
"""
from __future__ import annotations

import pathlib
import re

HEAVY = ("jax", "maddening", "mime")

# a module-scope import: column 0, not inside a function, try, or if
_MODULE_SCOPE_IMPORT = re.compile(
    r"^(?:import|from)\s+(" + "|".join(HEAVY) + r")(?:\s|\.|$)"
)
_SKIP = re.compile(r"importorskip\(\s*[\"'](" + "|".join(HEAVY) + r")[\"']")


def _scanned():
    """Every file in ``tests/`` that pytest imports for itself.

    ``conftest.py`` is in scope and is the WORST case, not an edge one: pytest
    imports it before collection, in every environment, and an ImportError
    there is a usage error — pytest exits 4 having collected nothing at all.
    A test module that fails to import is exit 2 with the other modules'
    results still on the screen. Measured, both.

    It was out of scope only because there was no ``tests/conftest.py`` until
    the skip inventory needed one; the file it added imports nothing but the
    standard library, and this is what keeps that true.
    """
    here = pathlib.Path(__file__).parent
    files = sorted(here.glob("test_*.py"))
    conftest = here / "conftest.py"
    if conftest.exists():
        files.append(conftest)
    return files


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


def test_at_least_one_module_is_actually_scanned():
    """A glob that matched nothing would also report zero offenders."""
    scanned = _scanned()
    assert len(scanned) > 20
    # and the conftest specifically, since it is the file whose failure is
    # total and the one the glob used to miss
    assert pathlib.Path(__file__).parent / "conftest.py" in scanned
