# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""No shipped module may import, at module scope, a name its target lacks.

Two axes, one failure. A *heavy* dependency (jax, hypothesis, …) is absent
from the environment; a *too-new stdlib* module (``tomllib``) is absent from
the interpreter. Either one, spelled at column 0, is a COLLECTION error, and
under default flags a collection error aborts the whole run.

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

THE SECOND AXIS ARRIVED THE SAME WAY AND WAS SHIPPED. ``tomllib`` is 3.11+;
``pyproject.toml`` declares ``requires-python = ">=3.10"`` and the sdist ships
``/tests``. ``tests/test_sdist_contents.py`` spelled ``import tomllib`` at
column 0, and at 650e678 on uv-managed CPython 3.10.20 with the core
installed, ``pytest -q`` came back rc=2 — junitxml ``tests=48 failures=0
errors=1 skipped=47``, ``Interrupted: 1 error during collection`` — while
``--collect-only`` reported 1335 collected and 1 error against 1352 clean on
CPython 3.11.15. A 3.10 user running the shipped suite got zero tests. Nothing
in the repository was watching that axis, which is why
:func:`test_no_module_imports_a_stdlib_module_the_floor_lacks` exists; the rule
there is the same shape as ``importorskip`` and just as cheap — put the import
behind a ``sys.version_info`` branch, which indents it off column 0.

WHAT THAT CHECK DOES NOT BUY, stated here rather than left to be discovered:
it is live only on an interpreter ABOVE the floor (see
:func:`_too_new_for_the_floor`), it is a *static* read of import statements, and
it says nothing about SYNTAX. A construct newer than the floor — the PEP 701
nested f-string that made this same file uncollectable one commit earlier — is
a different defect with the same blast radius, and ``ast.parse(src,
feature_version=(3, 10))`` was measured to PARSE that construct on a 3.12 host,
so no same-interpreter check can see it. Catching that one needs a real floor
interpreter, and no job in ``.github/workflows/`` runs one.

This module is deliberately dependency-free and reads the files as text. It
must never import the things it is checking for. It does import
``tests/conftest.py``, which is one of the files it checks — that costs
nothing, because a conftest carrying a heavy import kills the zero-dep session
before any test of it could run (``ImportError while loading conftest``, exit
4), while in an environment that HAS the dependency the import succeeds and
the text read below still catches it.
"""
from __future__ import annotations

import ast
import operator
import pathlib
import re
import sys

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

REPO = pathlib.Path(__file__).resolve().parents[1]
_PYPROJECT = (REPO / "pyproject.toml").read_text(encoding="utf-8")
# read as TEXT, and that is not a stylistic choice: the one parser for this
# file in the stdlib is `tomllib`, which is the very name this check exists to
# keep out of a module-scope import. A checker that needed the thing it checks
# for could not run on the floor it is checking.
_UNCOMMENTED = "\n".join(
    line for line in _PYPROJECT.splitlines() if not line.lstrip().startswith("#")
)


def _requires_python() -> str:
    """The WHOLE ``requires-python`` value, verbatim."""
    m = re.search(r'^requires-python\s*=\s*"([^"]*)"', _UNCOMMENTED, re.M)
    assert m, (
        'pyproject.toml has no `requires-python = "…"` this can read, so the '
        "floor the table below is written against cannot be confirmed"
    )
    return m.group(1).strip()


def _declared_floor() -> tuple[int, int]:
    """``requires-python`` from `pyproject.toml`, as a ``(major, minor)``.

    THE WHOLE SPECIFIER IS MATCHED, NOT ITS ``>=`` HALF. This read
    ``r'^requires-python\\s*=\\s*"\\s*>=\\s*(\\d+)\\.(\\d+)'`` — anchored at the
    left and unanchored at the right — so every clause after the floor was
    invisible to it, and to everything else in this repository. The argument
    for why that matters is in
    :func:`test_requires_python_declares_a_floor_and_not_a_ceiling`.
    """
    value = _requires_python()
    m = re.fullmatch(r">=\s*(\d+)\.(\d+)", value)
    assert m, (
        f'`requires-python` is "{value}", and this reads a bare ">=X.Y". Any '
        "other clause changes what the built artefact will INSTALL ON, and "
        "nothing else in this repository reads the field — see "
        "test_requires_python_declares_a_floor_and_not_a_ceiling."
    )
    return (int(m.group(1)), int(m.group(2)))


# THE FLOOR'S OWN STANDARD LIBRARY, recorded rather than reasoned about. This
# is `sys.stdlib_module_names` read off uv-managed CPython 3.10.20 on
# 2026-08-09 — all 303 names, private ones included, reproducible in one line:
#
#     python3.10 -c "import sys; print(sorted(sys.stdlib_module_names))"
#
# A SET, NOT A TABLE OF ADDITIONS, and that is the whole design. A table listing
# "modules added after 3.10" is short the moment a newer interpreter ships one
# nobody added to it, and it is short SILENTLY. The check below instead asks a
# question the running interpreter can answer completely: is this name stdlib
# HERE and absent THERE. Differenced when this was written: 3.11.15 minus
# 3.10.20 is exactly {'_tokenize', '_typing', 'tomllib'} and 3.12.3 minus
# 3.11.15 is {'_pydatetime', '_pylong', '_sha2'} — but the check does not
# depend on that difference being current, which is the point of recording the
# set instead. The private names are kept for the same reason `__future__` has
# to be here: filtering to "public" names would make `from __future__ import
# annotations`, which every file in this tree opens with, look like an import
# the floor lacks.
#
# THE MIRROR DIRECTION IS OUT OF SCOPE AND SAID SO RATHER THAN LEFT IMPLIED: a
# module the floor has and a later interpreter dropped (3.10.20 minus 3.12.3 is
# {'_bootsubprocess', '_sha256', '_sha512', 'asynchat', 'asyncore', 'binhex',
# 'distutils', 'imp', 'smtpd'}) breaks the NEWER interpreter, where this suite
# runs every day and the failure is immediate. This set cannot see that one,
# and does not need to.
_FLOOR_STDLIB = frozenset(
    """
    __future__ _abc _aix_support _ast _asyncio _bisect _blake2
    _bootsubprocess _bz2 _codecs _codecs_cn _codecs_hk _codecs_iso2022
    _codecs_jp _codecs_kr _codecs_tw _collections _collections_abc
    _compat_pickle _compression _contextvars _crypt _csv _ctypes _curses
    _curses_panel _datetime _dbm _decimal _elementtree _frozen_importlib
    _frozen_importlib_external _functools _gdbm _hashlib _heapq _imp _io
    _json _locale _lsprof _lzma _markupbase _md5 _msi _multibytecodec
    _multiprocessing _opcode _operator _osx_support _overlapped _pickle
    _posixshmem _posixsubprocess _py_abc _pydecimal _pyio _queue _random
    _scproxy _sha1 _sha256 _sha3 _sha512 _signal _sitebuiltins _socket
    _sqlite3 _sre _ssl _stat _statistics _string _strptime _struct
    _symtable _thread _threading_local _tkinter _tracemalloc _uuid
    _warnings _weakref _weakrefset _winapi _zoneinfo abc aifc
    antigravity argparse array ast asynchat asyncio asyncore atexit
    audioop base64 bdb binascii binhex bisect builtins bz2 cProfile
    calendar cgi cgitb chunk cmath cmd code codecs codeop collections
    colorsys compileall concurrent configparser contextlib contextvars
    copy copyreg crypt csv ctypes curses dataclasses datetime dbm
    decimal difflib dis distutils doctest email encodings ensurepip enum
    errno faulthandler fcntl filecmp fileinput fnmatch fractions ftplib
    functools gc genericpath getopt getpass gettext glob graphlib grp
    gzip hashlib heapq hmac html http idlelib imaplib imghdr imp
    importlib inspect io ipaddress itertools json keyword lib2to3
    linecache locale logging lzma mailbox mailcap marshal math mimetypes
    mmap modulefinder msilib msvcrt multiprocessing netrc nis nntplib nt
    ntpath nturl2path numbers opcode operator optparse os ossaudiodev
    pathlib pdb pickle pickletools pipes pkgutil platform plistlib
    poplib posix posixpath pprint profile pstats pty pwd py_compile
    pyclbr pydoc pydoc_data pyexpat queue quopri random re readline
    reprlib resource rlcompleter runpy sched secrets select selectors
    shelve shlex shutil signal site smtpd smtplib sndhdr socket
    socketserver spwd sqlite3 sre_compile sre_constants sre_parse ssl
    stat statistics string stringprep struct subprocess sunau symtable
    sys sysconfig syslog tabnanny tarfile telnetlib tempfile termios
    textwrap this threading time timeit tkinter token tokenize trace
    traceback tracemalloc tty turtle turtledemo types typing unicodedata
    unittest urllib uu uuid venv warnings wave weakref webbrowser winreg
    winsound wsgiref xdrlib xml xmlrpc zipapp zipfile zipimport zlib
    zoneinfo
    """.split()
)
# the interpreter `_FLOOR_STDLIB` was read off, so the pair can be checked
# against what `pyproject.toml` currently declares
_FLOOR_MEASURED_ON = (3, 10)

# any module-scope import, of anything: column 0, not inside a function, try or
# if. Which of the names it yields are a problem is decided against the two
# stdlib sets, not by a list written here.
_ANY_MODULE_SCOPE_IMPORT = re.compile(r"^(import|from)\s+([A-Za-z_][A-Za-z_0-9]*)")

# Every allowlisted sdist root that holds a `.py`. DERIVED from the allowlist
# rather than typed, because a hand-typed list of areas to sweep is how a
# "sweep of the shipped tree" ends up covering 6 roots out of 23.
#
# THAT DENOMINATOR READ 22 AND THE ALLOWLIST HAS NEVER HELD 22 — measured with
# `_sdist_roots()` at 43973af, 650e678, 53f9f84 and a61c01f: 23 at every one.
# It survived because the guard over it asserted `len(roots) >= 20`, which is
# true of 22 and of 23. The figure is now checked against the list it counts,
# in `tests/test_prose_hygiene.py::
# test_the_shipped_root_count_in_prose_matches_the_allowlist`, which reads this
# comment too.
_SHIPPED_PY_ROOTS_PIN = ("corpus", "docs", "src", "tests", "tools")


def _sdist_roots() -> list[str]:
    """The `[tool.hatch.build.targets.sdist]` allowlist's root entries."""
    block = re.search(
        r"\[tool\.hatch\.build\.targets\.sdist\]\s*\ninclude\s*=\s*\[(.*?)^\]",
        _UNCOMMENTED,
        re.S | re.M,
    )
    assert block, "the sdist allowlist is not where this expects it"
    return [m.group(1).lstrip("/") for m in re.finditer(r'"([^"]+)"', block.group(1))]


def _shipped_python_files() -> list[pathlib.Path]:
    """Every `.py` under a root the sdist ships.

    Wider than :func:`_scanned` on purpose, because the failure is wider. What
    changes across the areas is only the blast radius, never the cause:

    * ``tests/`` — a collection error, total, exit 2 and no tests at all;
    * ``src/`` — ``import stelling`` raises for a wheel user on the floor;
    * ``tools/``, ``docs/``, ``corpus/`` — the script dies when run, and each
      of these is referenced by something that ships (``tests/property/README``
      points at ``tools/property_check.py``, which is why ``/tools`` is in the
      allowlist at all).

    ``scratchpad/`` is excluded because it is not in the allowlist and does not
    ship; that is read off the allowlist here, not assumed.
    """
    out: list[pathlib.Path] = []
    for root in _sdist_roots():
        base = REPO / root
        if base.is_dir():
            out.extend(sorted(base.rglob("*.py")))
        elif base.is_file() and base.suffix == ".py":
            out.append(base)
    return [p for p in out if "__pycache__" not in p.parts]


def _module_scope_imports(text: str):
    """``(lineno, top-level name, line)`` for every column-0 import."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _ANY_MODULE_SCOPE_IMPORT.match(line)
        if not m:
            continue
        if m.group(1) == "import":
            # `import os, tomllib` is one statement and two names
            rest = line[len("import") :].split("#", 1)[0]
            for piece in rest.split(","):
                name = piece.strip().split(" as ")[0].strip().split(".")[0]
                if name.isidentifier():
                    yield lineno, name, line.strip()
        else:
            yield lineno, m.group(2), line.strip()


def _too_new_for_the_floor(name: str) -> bool:
    """Stdlib on the interpreter running this, absent from the floor's.

    BOTH HALVES ARE LOAD-BEARING, and the first is what makes this check LIVE
    ONLY ABOVE THE FLOOR. "Stdlib here" is how a too-new stdlib name is told
    apart from an ordinary third-party one (which is the other axis's
    business), and it is read off the interpreter running the suite. Run this
    suite ON 3.10 and ``tomllib`` is not stdlib there either, so this returns
    False and the scan finds nothing — vacuous, by construction.

    That is not a hole, because on the floor the interpreter performs the check
    itself and far more loudly: the import raises and collection stops. What
    the scan buys is seeing it from 3.11 and 3.12, where the defect is
    invisible — which is every environment this project actually develops and
    releases in, and is exactly how ``import tomllib`` shipped.
    """
    return name in sys.stdlib_module_names and name not in _FLOOR_STDLIB


def _floor_offenders():
    bad = []
    for path in _shipped_python_files():
        text = path.read_text(encoding="utf-8")
        for lineno, name, line in _module_scope_imports(text):
            if _too_new_for_the_floor(name):
                bad.append((path.relative_to(REPO).as_posix(), lineno, name, line))
    return bad


def test_no_module_imports_a_stdlib_module_the_floor_lacks():
    """The class `import tomllib` at column 0 belongs to, held shut.

    This file is NOT exempted from its own scan, unlike the heavy-import one.
    Every mention of a too-new name here is indented or inside a string, so the
    column-0 rule leaves it alone with no exception needed — and an exception
    is what would have let the checker ship the defect it checks for.
    """
    floor = _declared_floor()
    assert floor == _FLOOR_MEASURED_ON, (
        f"`requires-python` now declares {floor[0]}.{floor[1]} while "
        f"`_FLOOR_STDLIB` was read off {_FLOOR_MEASURED_ON[0]}."
        f"{_FLOOR_MEASURED_ON[1]}. Re-read `sys.stdlib_module_names` on the new "
        "floor interpreter; until then this check is measuring the wrong one."
    )
    bad = _floor_offenders()
    assert not bad, (
        f"these module-scope imports do not resolve on CPython "
        f"{floor[0]}.{floor[1]}, the floor `pyproject.toml` declares, and the "
        "sdist ships every one of these files. In a test module that is a "
        "COLLECTION error and the whole session goes with it — exit 2, zero "
        "tests. Put the import behind a `sys.version_info` branch (which "
        "indents it off column 0) with a fallback, and declare the fallback:\n"
        + "\n".join(
            f"  {name}:{lineno}  {src}   (`{dep}` is not in the "
            f"{floor[0]}.{floor[1]} stdlib)"
            for name, lineno, dep, src in bad
        )
    )


def test_the_floor_checker_can_actually_see_an_offender():
    """Anti-vacuity, same argument as the heavy scan's.

    Three ways this check could be green while blind — a regex that matches
    nothing, a floor set that contains everything, a scan over no files — and
    the first two are driven here on the exact line that shipped. The third is
    :func:`test_the_shipped_sweep_covers_every_allowlisted_root_with_python_in_it`.
    """
    got = list(_module_scope_imports("import tomllib\n"))
    assert got == [(1, "tomllib", "import tomllib")], got
    assert [n for _, n, _ in _module_scope_imports("from tomllib import loads")] == [
        "tomllib"
    ]
    assert [n for _, n, _ in _module_scope_imports("import os, tomllib")] == [
        "os",
        "tomllib",
    ]
    assert [n for _, n, _ in _module_scope_imports("import tomllib  # noqa")] == [
        "tomllib"
    ]
    # …and must NOT fire on the forms that are fine
    assert not list(_module_scope_imports("    import tomllib"))  # version branch
    assert not list(_module_scope_imports("# import tomllib"))
    assert [n for _, n, _ in _module_scope_imports("import tomli as tomllib")] == [
        "tomli"
    ]

    # the verdict half: the name that shipped is too new, an ordinary one is
    # not, and a third-party name is not this axis's business at all
    above_the_floor = sys.version_info[:2] > _FLOOR_MEASURED_ON
    assert _too_new_for_the_floor("tomllib") is above_the_floor, (
        "on an interpreter above the floor `tomllib` must read as too new; ON "
        "the floor it must not, because it is not stdlib there either and the "
        "interpreter's own ImportError is what does the checking"
    )
    assert not _too_new_for_the_floor("pathlib")
    assert not _too_new_for_the_floor("pytest")
    # the floor set is a real reading, not an empty set that would pass
    # everything nor the running interpreter's own set that would pass nothing
    assert len(_FLOOR_STDLIB) == 303
    assert {"tarfile", "zipfile", "subprocess", "__future__"} <= _FLOOR_STDLIB
    assert "tomllib" not in _FLOOR_STDLIB
    if sys.version_info[:2] == _FLOOR_MEASURED_ON:
        # the one interpreter that can audit the recording itself, and it does:
        # run this suite on the floor and `_FLOOR_STDLIB` is compared name for
        # name against the thing it claims to be a copy of
        assert _FLOOR_STDLIB == set(sys.stdlib_module_names)
    else:
        # …and anywhere else it must NOT be the running interpreter's set,
        # which is what a copy taken from the wrong python would look like
        assert _FLOOR_STDLIB != set(sys.stdlib_module_names)


# --- the version branch's CONDITION, not merely its indentation -------------
#
# THE GUARD ABOVE CANNOT SEE THE NEXT INSTANCE OF ITS OWN CLASS, and this is
# the repair. `test_no_module_imports_a_stdlib_module_the_floor_lacks` decides
# by COLUMN: an import at column 0 is a problem, an indented one is not. The
# remedy it tells you to apply — "put the import behind a `sys.version_info`
# branch (which indents it off column 0)" — is therefore accepted by the check
# on the strength of the INDENT it produces, and the branch's THRESHOLD is
# read by nothing.
#
# So the fix for the original defect can be undone by one character, and was:
# `if sys.version_info >= (3, 11):` -> `>= (3, 10)` in
# `tests/test_sdist_contents.py`. Driven, both directions, at 7cb6ab6:
#
#   CPython 3.11.15 (above the floor)  1332 passed, 94 skipped — FULLY GREEN,
#                                      the guard sees nothing
#   CPython 3.10.20 (the floor)        ModuleNotFoundError: No module named
#                                      'tomllib', "Interrupted: 1 error during
#                                      collection", 47 skipped, 1 error
#
# which is the original defect restored exactly: the shipped suite cannot be
# COLLECTED on the floor it declares, and the interpreter every developer and
# every CI job runs is green about it.
#
# WHAT THIS CHECKS, and it is a different question from the one above. Not
# "is this import indented", nor even "does the branch's THRESHOLD exclude the
# floor", but: does a floor interpreter REACH this import. Those are not the
# same question, and the difference was a live hole in the first version of
# this guard, which is worth spelling out because it is the same shape as the
# defect it was written for.
#
# THAT FIRST VERSION COULD NOT SEE THE FALLBACK ITS OWN REMEDY MESSAGE ASKS
# FOR. It matched `sys.version_info >= (a, b)` and walked `node.body`. The main
# scan's message says "put the import behind a `sys.version_info` branch …
# **with a fallback, and declare the fallback**" — and the fallback goes in the
# `else`, which is the one side a 3.10 interpreter is CERTAIN to run and the
# one side the guard never looked at. So the invited remedy, written wrong,
# passed. Driven at 103f3b6 — `else: import tomli as tomllib` ->
# `else: import tomllib` in `tests/test_sdist_contents.py`, line-neutral, one
# mutation alone, `JAX_ENABLE_X64=1`, figures from `--junitxml`:
#
#   CPython 3.11.15 (above the floor)  tests=1430 failures=0 errors=0
#                                      skipped=94 — FULLY GREEN
#   CPython 3.10.20 (the floor)        rc=2, tests=48 failures=0 errors=1
#                                      skipped=47, ModuleNotFoundError: No
#                                      module named 'tomllib', "Interrupted: 1
#                                      error during collection"
#
# — the same junit signature as the BEFORE row above, i.e. the original defect
# restored, through the door the remedy message opens. Eleven other spellings
# of the same condition were probed at the same commit and nine of them were
# missed as well.
#
# So the condition is now DECIDED rather than pattern-matched: `_floor_value`
# evaluates it with `sys.version_info` bound to the floor, `_FloorReach` walks
# only the side that interpreter takes, and `else` is a side.
#
# THE REACH LIMIT, measured and left open, because "the condition is evaluated"
# reads like completeness and is not. Each of these is a spelling a floor
# interpreter DOES reach and this guard does NOT see; each is a row in
# `_MISSED_THOUGH_REACHED`, so the list cannot rot silently:
#
#   PY311 = sys.version_info >= (3, 11)   the condition bound to a NAME. The
#   if PY311: … else: import tomllib     branch's test carries no
#                                        `sys.version_info`, so it is not a
#                                        version branch to `_version_branches`
#                                        at all. Following it would mean
#                                        constant-propagating module scope.
#   import_module("tom" + "llib")        a COMPUTED module name. Only a literal
#                                        first argument is read; anything else
#                                        is not knowable statically.
#   (3, 10, 5) <= sys.version_info       true only STRICTLY INSIDE the floor
#           < (3, 10, 9)                 series. The floor is a range of
#                                        interpreters and only its two ends are
#                                        evaluated (`_FLOOR_MICROS`), so a
#                                        window between them is invisible.
#
# And two limits that are about the whole module rather than this guard: it is
# a STATIC read, so an import reached through `eval`/`exec` is out of reach by
# construction; and a dynamic import OUTSIDE any version branch — a column-0
# `__import__("tomllib")` — is seen by neither this guard (no version branch)
# nor the main scan (its regex is `^(import|from)`). That one is a gap in the
# scan above, not here, and is not closed.
#
# ON THE FLOOR ITSELF THIS IS INERT, for the same reason the scan above is and
# with the same argument: `_too_new_for_the_floor` asks the RUNNING
# interpreter, and on 3.10 `tomllib` is not stdlib there either, so the name
# does not read as too-new and the interpreter's own ImportError is what does
# the checking — as the 3.10.20 row above shows it doing. The direction that
# needed covering is the other one, where the suite is green and silent.


_UNDECIDED = object()

# `sys.version_info` on the floor is not ONE tuple, it is the whole `X.Y.*`
# series, and a condition can differ across it: `>= (3, 10, 5)` is False on
# 3.10.0 and True on 3.10.20, and the second of those reaches the import. Both
# ends are evaluated and the findings unioned. A condition true only STRICTLY
# BETWEEN them is not seen; it is in the reach note below.
_FLOOR_MICROS = (0, 9999)
_VERSION_FIELDS = ("major", "minor", "micro", "releaselevel", "serial")
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _is_version_info(node) -> bool:
    """`sys.version_info`, spelled as an attribute of `sys`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "version_info"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _floor_value(node, version):
    """`node`'s value with `sys.version_info` bound to `version`, or `_UNDECIDED`.

    THIS REPLACED A MATCHER FOR ONE SPELLING. Its predecessor, `_version_gate`,
    recognised `sys.version_info >= (a, b)` and its `[:2]` variant and nothing
    else — so `sys.version_info.minor >= 10`, `(3, 10) <= sys.version_info` and
    `not sys.version_info < (3, 10)` are the same test written three other
    ways, all three of them invisible to it, and each one restores the defect.
    Deciding the condition by EVALUATING it is the same size of code and does
    not have to be extended once per spelling.

    Deliberately tiny: constants, tuples of them, `sys.version_info` and the
    ways a piece of it is spelled (a subscript, `[:2]` or `[0]`; the named
    fields), the six order comparisons, `and`/`or`, and `not`. Anything else is
    `_UNDECIDED`, and an `_UNDECIDED` anywhere in a condition makes the whole
    condition undecided — which the reach walk treats as "either side may run",
    the conservative direction.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        parts = [_floor_value(e, version) for e in node.elts]
        return _UNDECIDED if any(p is _UNDECIDED for p in parts) else tuple(parts)
    if _is_version_info(node):
        return version
    if isinstance(node, ast.Attribute) and _is_version_info(node.value):
        if node.attr not in _VERSION_FIELDS:
            return _UNDECIDED
        return version[_VERSION_FIELDS.index(node.attr)]
    if isinstance(node, ast.Subscript) and _is_version_info(node.value):
        return _version_slice(version, node.slice)
    if isinstance(node, ast.Compare):
        left = _floor_value(node.left, version)
        verdict = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _floor_value(comparator, version)
            compare = _COMPARISONS.get(type(op))
            if compare is None or left is _UNDECIDED or right is _UNDECIDED:
                return _UNDECIDED
            try:
                verdict = verdict and compare(left, right)
            except TypeError:
                return _UNDECIDED
            left = right
        return verdict
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = _floor_value(node.operand, version)
        return _UNDECIDED if inner is _UNDECIDED else (not inner)
    if isinstance(node, ast.BoolOp):
        values = [_floor_value(v, version) for v in node.values]
        if any(v is _UNDECIDED for v in values):
            return _UNDECIDED
        return all(values) if isinstance(node.op, ast.And) else any(values)
    return _UNDECIDED


def _version_slice(version, node):
    """`sys.version_info[...]` on `version`, or `_UNDECIDED`."""
    if isinstance(node, ast.Slice):
        bounds = []
        for part in (node.lower, node.upper, node.step):
            if part is None:
                bounds.append(None)
                continue
            value = _floor_value(part, version)
            if not isinstance(value, int) or isinstance(value, bool):
                return _UNDECIDED
            bounds.append(value)
        return version[slice(*bounds)]
    index = _floor_value(node, version)
    if not isinstance(index, int) or isinstance(index, bool):
        return _UNDECIDED
    try:
        return version[index]
    except IndexError:
        return _UNDECIDED


def _decided_on(test, version):
    """True/False if `test` decides on `version`, None if this cannot tell."""
    value = _floor_value(test, version)
    return None if value is _UNDECIDED else bool(value)


def _dynamic_import_name(node) -> str | None:
    """The module `__import__("x")` / `importlib.import_module("x")` names.

    Neither binds an `ast.Import` node, and neither matches the column-0 regex
    the main scan uses either. Only a literal first argument is read; a
    computed one is not knowable here and is in the reach note.
    """
    func = node.func
    dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
        isinstance(func, ast.Attribute) and func.attr == "import_module"
    )
    if not dynamic or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


class _FloorReach(ast.NodeVisitor):
    """Every import REACHED when this source runs on one floor interpreter.

    THE `else` IS THE HALF THAT WAS MISSING. The predecessor walked `node.body`
    and never `node.orelse`, so the fallback the main scan's own remedy message
    tells you to write — "put the import behind a `sys.version_info` branch …
    with a fallback, and declare the fallback" — was the one place the guard
    could not look, and on the floor it is the only side that RUNS.

    A branch this can decide contributes only the side that interpreter takes;
    one it cannot decide contributes both, which is the conservative direction:
    a false positive here is a loud failure with a line number on it, a false
    negative is the defect this exists to catch.
    """

    def __init__(self, version):
        self.version = version
        self.found: list[tuple[str, int]] = []

    def visit_If(self, node):
        self._branch(node.test, node.body, node.orelse)

    def visit_IfExp(self, node):
        # `x = __import__("tomllib") if sys.version_info >= (3, 10) else None`
        # is a version branch that is not an `if` STATEMENT at all
        self._branch(node.test, [node.body], [node.orelse])

    def _branch(self, test, body, orelse):
        taken = _decided_on(test, self.version)
        if taken is not False:
            for child in body:
                self.visit(child)
        if taken is not True:
            for child in orelse:
                self.visit(child)

    def visit_Import(self, node):
        self.found.extend((a.name.split(".")[0], node.lineno) for a in node.names)

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module:
            self.found.append((node.module.split(".")[0], node.lineno))

    def visit_Call(self, node):
        name = _dynamic_import_name(node)
        if name is not None:
            self.found.append((name.split(".")[0], node.lineno))
        self.generic_visit(node)


def _version_branches(tree):
    """Every `if` / conditional expression whose test reads `sys.version_info`."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.If, ast.IfExp))
        and any(_is_version_info(child) for child in ast.walk(node.test))
    ]


def _weak_version_branches(text: str, floor: tuple[int, int]):
    """`(lineno, name, condition)` for each too-new import a floor interpreter
    REACHES through a `sys.version_info` branch."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    bad: dict[tuple[int, str], str] = {}
    for node in _version_branches(tree):
        for micro in _FLOOR_MICROS:
            reach = _FloorReach((floor[0], floor[1], micro, "final", 0))
            reach.visit(node)
            for name, lineno in reach.found:
                if _too_new_for_the_floor(name):
                    bad[(lineno, name)] = ast.unparse(node.test)
    return [(lineno, name, cond) for (lineno, name), cond in sorted(bad.items())]


def test_requires_python_declares_a_floor_and_not_a_ceiling():
    """`requires-python` is the one field here that decides what INSTALLS.

    THE FLOOR CHECK ABOVE READ HALF OF IT. The regex was anchored on the left
    and open on the right, so `">=3.10"` and `">=3.10,<3.11"` were the same
    string to it — the floor came back (3, 10) either way, every table in this
    module agreed with it, and no other test, hook or lint in this repository
    reads the field at all.

    DRIVEN, `">=3.10"` -> `">=3.10,<3.11"`, mutation applied alone (CPython
    3.11.15, pytest 9.1.1, JAX_ENABLE_X64=1, figures from --junitxml):

        the full suite      tests=1428 failures=0 errors=0 skipped=94 — GREEN
        `uv build --wheel`  succeeds, and the artefact carries
                            `Requires-Python: <3.11,>=3.10` in its METADATA
        pip, CPython 3.12.3 ERROR: Package 'stelling' requires a different
                            Python: 3.12.3 not in '<3.11,>=3.10'

    So the whole suite passes, the wheel builds, and the wheel cannot be
    installed on the interpreter this project is developed on. That is the
    shape worth naming: the defect is not in anything that runs, it is in what
    the run PRODUCES, and a green suite is exactly what it looks like.

    NOT A GENERAL PEP 440 PARSER, and said so rather than implied. This
    demands the one form the project actually declares and rejects everything
    else, including forms that would be harmless (`>= 3.10`, with a space).
    A checker that understood the grammar would have to decide which
    combinations are safe, which is a bigger claim than this project needs;
    demanding the exact shape is the smaller one, and it fails loudly on
    anything it was not told about rather than passing it.
    """
    value = _requires_python()
    assert re.fullmatch(r">=\s*\d+\.\d+", value), (
        f'`requires-python` is now "{value}". This project declares a FLOOR '
        "and no ceiling. An upper bound here is not a preference — it is "
        "baked into the built wheel's `Requires-Python`, where pip enforces "
        'it: `">=3.10,<3.11"` builds cleanly, keeps the whole suite green, '
        "and produces an artefact that refuses to install on CPython 3.12. "
        "Nothing else in this repository reads this field."
    )
    # …and the floor reader agrees with the string it was read from, so the
    # two cannot drift into disagreeing about what the floor is
    floor = _declared_floor()
    assert value.replace(" ", "") == f">={floor[0]}.{floor[1]}", (value, floor)


def test_no_version_branch_puts_a_too_new_import_where_the_floor_reaches_it():
    """The side of the branch the floor RUNS, which nothing read.

    A `sys.version_info` branch is the remedy this module's main check tells
    you to apply, and that check accepts it on the strength of the indentation
    it produces. What does the work is which side of it a floor interpreter
    reaches — and that is not only the threshold: the `else` is reached on the
    floor by definition, and the fallback the remedy message asks for lives
    there. Both undo the original defect while every other test in this file
    stays green. The reach limit is in the note above; the spellings are in
    `_SPELLINGS`.
    """
    floor = _declared_floor()
    bad = []
    for path in _shipped_python_files():
        for lineno, name, condition in _weak_version_branches(
            path.read_text(encoding="utf-8"), floor
        ):
            bad.append(
                f"  {path.relative_to(REPO).as_posix()}:{lineno}  imports "
                f"`{name}` on a side of `{condition}` that CPython "
                f"{floor[0]}.{floor[1]} reaches"
            )
    assert not bad, (
        "a `sys.version_info` branch puts a too-new import on the side the "
        "declared floor actually runs, so the import fails there — a "
        "COLLECTION error in a test module, which is exit 2 and zero tests "
        "for anyone on the floor. The indentation is not the guard, and "
        "neither is the `if` side alone; an `else` runs ON the floor by "
        "definition:\n" + "\n".join(bad)
    )



# THE SPELLINGS, and this table IS the reach claim. Each row is source text and
# whether THIS guard must flag it. Which rows actually break the floor was
# measured by EXECUTING each snippet on uv-managed CPython 3.10.20 (with
# `tomli` installed, so the shipped form's fallback really resolves) rather
# than read off the page: 14 of these 17 raise `ModuleNotFoundError: No module
# named 'tomllib'` there. Thirteen of the 14 are the rows marked `True`. The
# fourteenth is the bare column-0 `import tomllib`, marked `False` because it
# is `test_no_module_imports_a_stdlib_module_the_floor_lacks`'s business and
# not this one's — that is the only row where "breaks the floor" and "this
# guard must flag it" come apart, and it is here to keep the division visible.
_SPELLINGS = (
    # (label, source, must this guard flag it)
    ("the shipped form", "if sys.version_info >= (3, 11):\n    import tomllib\n"
     "else:\n    import tomli as tomllib\n", False),
    ("lowered threshold",
     "if sys.version_info >= (3, 10):\n    import tomllib\n", True),
    ("lowered threshold, sliced",
     "if sys.version_info[:2] >= (3, 10):\n    import tomllib\n", True),
    ("the else-fallback the remedy message invites",
     "if sys.version_info >= (3, 11):\n    import tomllib\n"
     "else:\n    import tomllib\n", True),
    ("the else-fallback, `from` form",
     "if sys.version_info >= (3, 11):\n    from tomllib import loads\n"
     "else:\n    from tomllib import loads\n", True),
    ("inverted, with the import in the else",
     "if sys.version_info < (3, 10):\n    raise SystemExit\n"
     "else:\n    import tomllib\n", True),
    ("the `.minor` field",
     "if sys.version_info.minor >= 10:\n    import tomllib\n", True),
    ("operands reversed",
     "if (3, 10) <= sys.version_info:\n    import tomllib\n", True),
    ("negated",
     "if not sys.version_info < (3, 10):\n    import tomllib\n", True),
    ("a `(major, minor)` pair",
     "if (sys.version_info.major, sys.version_info.minor) >= (3, 10):\n"
     "    import tomllib\n", True),
    ("an elif landing on the floor",
     "if sys.version_info >= (3, 12):\n    import tomllib\n"
     "elif sys.version_info >= (3, 10):\n    import tomllib\n", True),
    ("a micro threshold inside the floor series",
     "if sys.version_info >= (3, 10, 5):\n    import tomllib\n", True),
    ("a conditional expression and `__import__`",
     'tomllib = __import__("tomllib") if sys.version_info >= (3, 10) else None\n',
     True),
    ("`importlib.import_module` in the else",
     "if sys.version_info >= (3, 11):\n    import tomllib\n"
     'else:\n    tomllib = importlib.import_module("tomllib")\n', True),
    ("a threshold that really does exclude the floor",
     "if sys.version_info >= (3, 11):\n    import tomllib\n", False),
    ("no version branch at all — the other scan's business",
     "import tomllib\n", False),
    ("a version branch with nothing too-new in it",
     "if sys.version_info >= (3, 10):\n    import pathlib\n", False),
)

# …and the ones it still does not see, which is the reach note made executable.
# A row here going red is good news that needs the note above rewritten.
# The first two were executed on 3.10.20 and both raise there. The third does
# NOT raise on 3.10.20 — its window closed at 3.10.9 — and is reached on
# 3.10.5 through 3.10.8, which `requires-python = ">=3.10"` permits; that is
# precisely why two endpoints are not a proof.
_MISSED_THOUGH_REACHED = (
    ("the condition bound to a name first",
     "PY311 = sys.version_info >= (3, 11)\nif PY311:\n    import tomllib\n"
     "else:\n    import tomllib\n"),
    ("a computed module name",
     'if sys.version_info >= (3, 11):\n    import tomllib\nelse:\n'
     '    tomllib = importlib.import_module("tom" + "llib")\n'),
    ("true only strictly inside the floor series",
     "if (3, 10, 5) <= sys.version_info < (3, 10, 9):\n    import tomllib\n"),
    # A CONSTRUCT, not a condition. `_version_branches` enumerates `ast.If` and
    # `ast.IfExp` only, so this is not a version branch at all — and the import
    # is indented, so the column-0 scan does not see it either. `_floor_value`
    # decides `sys.version_info[:2]` correctly; it is never handed this subject.
    # MEASURED, driven into this file: 3.11.15 full suite green, 3.10.20
    # `tests=48 errors=1 skipped=47`, `Interrupted: 1 error during collection`
    # — the signature the original defect had. `while` is the same shape.
    # "does not have to be extended once per spelling" is true of CONDITIONS
    # and not of constructs.
    ("a match statement on sys.version_info",
     "match sys.version_info[:2]:\n    case (3, 10):\n        import tomllib\n"
     "    case _:\n        import tomllib\n"),
)


def test_the_version_branch_check_sees_every_spelling_it_claims_to():
    """Anti-vacuity, driven on the whole table rather than on one edit.

    Its predecessor drove ONE spelling — `>= (3, 11)` lowered to `>= (3, 10)` —
    and the guard under it recognised exactly that one. Eleven other ways to
    write the same condition, and the `else` the remedy message asks for, all
    went through it green while breaking the floor. So the table above is the
    unit of the claim, and the four ways this could be green while blind — a
    condition evaluator that decides nothing, an import walk that finds no
    import, a reach walk that never looks at the side the floor takes, a
    comparison against the wrong version — are all driven by it.
    """
    floor = (3, 10)
    # inert ON the floor, for the reason `_too_new_for_the_floor` gives: there
    # `tomllib` is not stdlib either, and the interpreter's own ImportError is
    # what does the checking. So every positive row is asserted against that.
    above_the_floor = sys.version_info[:2] > _FLOOR_MEASURED_ON

    for label, source, must_flag in _SPELLINGS:
        seen = bool(_weak_version_branches(source, floor))
        assert seen is (must_flag and above_the_floor), (
            f"{label}: this guard must "
            f"{'flag' if must_flag else 'leave alone'} this, and it "
            f"{'flagged' if seen else 'did not flag'} it. Source:\n{source}"
        )

    for label, source in _MISSED_THOUGH_REACHED:
        assert not _weak_version_branches(source, floor), (
            f"{label}: the guard now SEES this, which the reach note above "
            "says it does not. That note is the claim; move this row up into "
            f"`_SPELLINGS` and rewrite the note. Source:\n{source}"
        )

    # the pieces, so a whole-table failure can be localised
    assert _decided_on(
        ast.parse("sys.version_info >= (3, 11)", mode="eval").body, (3, 10, 20)
    ) is False
    assert _decided_on(
        ast.parse("sys.version_info.minor >= 10", mode="eval").body, (3, 10, 20)
    ) is True
    assert _decided_on(ast.parse("x >= (3, 11)", mode="eval").body, (3, 10, 20)) is None
    assert _floor_value(
        ast.parse("sys.version_info[:2]", mode="eval").body, (3, 10, 20)
    ) == (3, 10)
    reach = _FloorReach((3, 10, 20))
    reach.visit(ast.parse("if sys.version_info >= (3, 11):\n    import tomllib\n"))
    assert reach.found == [], reach.found


def test_the_shipped_sweep_covers_every_allowlisted_root_with_python_in_it():
    """The scope of the sweep is DERIVED from the allowlist, not typed.

    A hand-maintained list of directories to sweep is the failure mode this
    guards: it silently stops covering a root the moment the allowlist gains
    one. So the roots come out of `pyproject.toml`, and the pin below is only a
    cross-check that the derivation returned what a reader expects.
    """
    roots = _sdist_roots()
    # This was `len(roots) >= 20` — a bound satisfied by 22, 23 and 40 alike,
    # standing directly over a comment in this file that said 22 when the
    # allowlist has always held 23. The COUNT is checked against both shipped
    # comments in `tests/test_prose_hygiene.py::
    # test_the_shipped_root_count_in_prose_matches_the_allowlist`; what belongs
    # here is that the parse returned the allowlist and not a fragment of it.
    assert "src" in roots and "pyproject.toml" in roots, roots
    assert len(roots) == len(set(roots)), f"a root is listed twice: {sorted(roots)}"
    with_python = tuple(
        sorted(
            r
            for r in roots
            if (REPO / r).is_dir() and any((REPO / r).rglob("*.py"))
        )
    )
    assert with_python == _SHIPPED_PY_ROOTS_PIN, (
        "the set of shipped roots containing Python has moved; the sweep "
        f"follows it automatically, but the pin has not: {with_python}"
    )
    files = _shipped_python_files()
    assert len(files) > 100, len(files)
    # the file whose module-scope `import tomllib` shipped, and `src/`, which
    # a wheel user imports
    rel = {p.relative_to(REPO).as_posix() for p in files}
    assert "tests/test_sdist_contents.py" in rel
    assert "src/stelling/solvers.py" in rel
    assert "tools/property_check.py" in rel
    # and nothing from the one area that does NOT ship
    assert not any(r.startswith("scratchpad/") for r in rel)


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
