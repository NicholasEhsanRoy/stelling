# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Import hygiene: jax only in _jax_compat; private jax modules nowhere;
solvers never at module level — the zero-dep surface must import clean.

Two kinds of check live here and they measure different things.

*Token scans* (:func:`test_jax_imported_only_in_compat_module` and friends)
enforce the **churn boundary**: exactly one file names jax, so a jax release
that moves a symbol has a blast radius of one file. That is a real property
and worth pinning — but it says nothing about what a user without jax sees.
It was mistaken for that guarantee once: ``harness.py`` contained no ``import
jax`` token and still died at import in a bare environment, with the token
scan green, because it imported the module that imports jax.

*Behavioural checks* (everything below :data:`_NO_JAX_PRELUDE`) measure the
thing the token scan cannot: **import the module with jax genuinely absent
and read the failure.** A user who runs ``pip install stelling`` and copies
the first line out of the README gets that failure, and it has to name the
extra that fixes it rather than a private module two levels down.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import stelling

REPO = Path(__file__).resolve().parents[1]
# the tree the suite actually imported, which is what the subprocesses must
# exercise too — an editable install pointing at another checkout would
# otherwise let this file measure a package nobody is editing
SRC = Path(stelling.__file__).resolve().parents[1]
JAX_IMPORT = re.compile(r"^\s*(import jax\b|from jax\b)")
# built by concatenation so this test file cannot match itself
PRIVATE_JAX = "jax." + "_src"
SOLVER_MODULES = ("z3", "cvc5")

# What `require("jax")` promises the user, and the only substring of it that
# other files (docs/harness-api.md) quote. Kept here so a reworded message
# and a stale doc cannot drift apart quietly.
JAX_EXTRA_HINT = 'pip install "stelling[jax]"'


def test_jax_imported_only_in_compat_module():
    """The churn boundary: one file names jax, so a jax release breaks one file.

    NOT a claim about bare environments — a module can require jax at import
    without ever spelling the token. That property is measured by
    :func:`test_importing_harness_without_jax_names_the_extra`.
    """
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        if path.name == "_jax_compat.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if JAX_IMPORT.match(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "jax may only be imported in stelling/_jax_compat.py:\n" + "\n".join(offenders)


def test_private_jax_modules_banned_everywhere():
    offenders = []
    for directory in ("src", "tests"):
        for path in (REPO / directory).rglob("*.py"):
            if path.name == Path(__file__).name:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if PRIVATE_JAX in line:
                    offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, f"{PRIVATE_JAX} is banned outright:\n" + "\n".join(offenders)


def _module_level_statements(tree: ast.Module):
    """Module-level statements, descending through module-level compound
    statements (if/try/with/for/while) but never into function or class
    bodies — an import inside those is the allowed lazy pattern."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for attr in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, attr, ()) or ())
            for handler in getattr(node, "handlers", ()) or ():
                stack.extend(handler.body)


def test_solver_imports_are_lazy_everywhere_in_src():
    """The solver mirror of the jax rule: no module-level ``import z3`` /
    ``import cvc5`` (or ``from z3/cvc5 import …``) anywhere under src/.
    Solvers are reached exclusively through ``stelling._optional`` inside
    functions; the zero-dep core must import with no solver installed."""
    offenders = []
    for path in (REPO / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _module_level_statements(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in SOLVER_MODULES:
                    offenders.append(
                        f"{path.relative_to(REPO)}:{node.lineno}: "
                        f"module-level solver import of {name!r}"
                    )
    assert not offenders, (
        "solver imports must be lazy (inside functions, via stelling._optional):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# The behavioural half: what an environment WITHOUT jax actually sees.
# ---------------------------------------------------------------------------

# A meta_path finder that makes jax — and only jax — look uninstalled, raising
# the exact exception a bare interpreter raises (a ModuleNotFoundError whose
# ``name`` is the missing top-level package, which is what
# ``stelling._optional.require`` keys on). Returning None for every other name
# means "no opinion, keep looking", so the normal finders still run: this hides
# one distribution, it does not sandbox the interpreter.
#
# The hook is load-bearing for every test below it, so it is itself measured in
# both directions first — see test_the_no_jax_hook_blocks_jax_and_nothing_else.
_NO_JAX_PRELUDE = """
import importlib.util as _u, json, sys

_JAX_WAS_INSTALLED = _u.find_spec("jax") is not None


class _NoJax:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top == "jax":
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None


sys.meta_path.insert(0, _NoJax())
"""


def _run_without_jax(code: str, *, path_head: list[str] | None = None):
    """Run ``code`` in a fresh interpreter where importing jax fails."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([*(path_head or []), str(SRC)])
    proc = subprocess.run(
        [sys.executable, "-c", _NO_JAX_PRELUDE + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_probe_subprocess_imports_the_tree_under_test():
    """Anti-vacuity: an editable install pointing at another checkout would
    make every probe below measure a package this suite is not editing."""
    got = _run_without_jax(
        """
        import stelling
        print(json.dumps({"file": stelling.__file__}))
        """
    )
    assert Path(got["file"]).resolve().parents[1] == SRC, got


def test_the_no_jax_hook_blocks_jax_and_nothing_else(tmp_path):
    """The hook, measured in both directions before anything relies on it.

    Blocks: ``jax`` and its submodules, with a real ModuleNotFoundError whose
    ``.name`` is ``jax`` — the shape ``require`` matches on, so a probe cannot
    pass by raising something merely similar.

    Does not block: the standard library, other installed distributions, the
    package under test, or names that merely *start with* ``jax`` (``jaxlib``
    is a separate distribution, and a hook that ate it would be measuring a
    different environment than the one users have). The two locally created
    modules make that direction checkable in any environment, installed jax or
    not.
    """
    (tmp_path / "jaxen.py").write_text("VALUE = 'not jax'\n")
    (tmp_path / "jax_shaped").mkdir()
    (tmp_path / "jax_shaped" / "__init__.py").write_text("VALUE = 'also not jax'\n")

    got = _run_without_jax(
        """
        out = {"jax_was_installed": _JAX_WAS_INSTALLED}

        for name in ("jax", "jax.numpy", "jax.extend.core"):
            try:
                __import__(name)
            except ModuleNotFoundError as e:
                out[name] = {"blocked": True, "name": e.name, "msg": str(e)}
            else:
                out[name] = {"blocked": False}

        allowed = {}
        for name in ("json", "pathlib", "pytest", "stelling", "stelling.ir",
                     "jaxen", "jax_shaped"):
            try:
                __import__(name)
            except BaseException as e:
                allowed[name] = f"{type(e).__name__}: {e}"
            else:
                allowed[name] = "ok"
        out["allowed"] = allowed
        print(json.dumps(out))
        """,
        path_head=[str(tmp_path)],
    )

    for name in ("jax", "jax.numpy", "jax.extend.core"):
        assert got[name]["blocked"], f"{name} was not blocked: {got[name]}"
        # `.name` must be the missing TOP-LEVEL package, exactly as a bare
        # interpreter reports it; require() refuses to translate anything else
        assert got[name]["name"] == "jax", got[name]
        assert got[name]["msg"] == "No module named 'jax'", got[name]

    assert got["allowed"] == {
        "json": "ok",
        "pathlib": "ok",
        "pytest": "ok",
        "stelling": "ok",
        "stelling.ir": "ok",
        "jaxen": "ok",
        "jax_shaped": "ok",
    }, got["allowed"]

    # and, where jax IS installed, the hook is what did the blocking rather
    # than the environment happening to lack it
    if got["jax_was_installed"]:
        probe = subprocess.run(
            [sys.executable, "-c", "import jax; print('jax imported')"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(SRC), "JAX_PLATFORMS": "cpu"},
        )
        assert "jax imported" in probe.stdout, probe.stderr


def test_importing_harness_without_jax_names_the_extra():
    """The property the token scan could not see.

    ``stelling.harness`` is the one module a harness author cannot avoid —
    ``any_array``, ``assert_``, ``assume``, ``nonvacuity`` and ``trace`` all
    live there, and every documented example opens with ``from
    stelling.harness import ...``. With jax absent it must fail with the
    sentence that names the extra, not with a raw ``No module named 'jax'``
    from ``stelling/_jax_compat.py``.
    """
    got = _run_without_jax(
        """
        import os.path, traceback

        out = {}
        for label, stmt in (
            ("module", "import stelling.harness"),
            ("from", "from stelling.harness import any_array"),
            ("names", "from stelling.harness import (any_array, any_pytree, "
                      "assert_, assume, nonvacuity, trace)"),
        ):
            for mod in [m for m in sys.modules if m.startswith("stelling")]:
                del sys.modules[mod]
            try:
                exec(stmt, {})
            except BaseException as e:
                out[label] = {
                    "raised": True,
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "is_import_error": isinstance(e, ImportError),
                    "msg": str(e),
                    "frames": [
                        os.path.basename(f.filename)
                        for f in traceback.extract_tb(e.__traceback__)
                    ],
                }
            else:
                out[label] = {"raised": False}
        print(json.dumps(out))
        """
    )

    for label in ("module", "from", "names"):
        detail = got[label]
        assert detail["raised"], f"{label}: stelling.harness imported without jax"
        # the sentence that names the fix
        assert JAX_EXTRA_HINT in detail["msg"], detail
        assert "jax is required for tracing harnesses to jaxprs" in detail["msg"], detail
        # ... and specifically NOT the raw one it used to be
        assert detail["msg"] != "No module named 'jax'", detail
        # still an ImportError, so `except ImportError` in user code and in
        # docs/harness-api.md keeps catching it
        assert detail["is_import_error"], detail
        assert detail["type"] == "stelling._optional.OptionalDependencyError", detail
        # the guard sits at the PUBLIC door, so the traceback stops at the
        # module the user typed: `_jax_compat.py` — the private jax bridge —
        # is never reached and never shown. (A guard only in `_jax_compat`
        # would print the same sentence but hand back a traceback through a
        # file the user has no business reading.)
        assert "harness.py" in detail["frames"], detail
        assert "_jax_compat.py" not in detail["frames"], detail

    # `import stelling.harness` alone fails, so the requirement is at IMPORT
    # time, which is what the module docstring now claims. If a future change
    # makes the façade lazy, this test is the one that says so out loud — and
    # the docstring must move with it.
    doc = (SRC / "stelling" / "harness.py").read_text()
    assert "at import time" in doc, "harness.py's docstring must state when jax is needed"


def test_no_public_module_fails_without_naming_the_jax_extra():
    """The generalisation: for EVERY public submodule, importing it with jax
    absent either works or produces the sentence naming the extra.

    ``harness`` is the module that failed this in practice, but nothing stops
    the next module from growing a module-scope ``stelling._jax_compat``
    import — the token scan would stay green for that too, because the token
    is in the module being imported, not the importer.
    """
    got = _run_without_jax(
        """
        import importlib, pkgutil
        import stelling

        names = sorted(
            m.name for m in pkgutil.iter_modules(stelling.__path__)
            if not m.name.startswith("_")
        )
        out = {"scanned": names, "failures": {}}
        for name in names:
            try:
                importlib.import_module("stelling." + name)
            except BaseException as e:
                out["failures"][name] = {
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "msg": str(e),
                }
        print(json.dumps(out))
        """
    )

    # a broken enumeration would report zero failures just as loudly
    assert len(got["scanned"]) >= 17, got["scanned"]
    assert "harness" in got["scanned"] and "ir" in got["scanned"], got["scanned"]

    nameless = {
        name: detail
        for name, detail in got["failures"].items()
        if JAX_EXTRA_HINT not in detail["msg"]
    }
    assert not nameless, (
        "these public modules die in a jax-less environment without naming the "
        f"extra that fixes it (expected {JAX_EXTRA_HINT!r} in the message):\n"
        + "\n".join(f"  stelling.{n}: {d['type']}: {d['msg']}" for n, d in nameless.items())
    )


def test_the_lazy_call_sites_also_name_the_jax_extra():
    """The other door into jax, which importing nothing can protect.

    :mod:`stelling.preconditions` imports jax-free on purpose and reaches
    ``stelling._jax_compat`` inside its functions — so it passes the
    import-time check above and still, before this fix, answered
    ``check(harness, ...)`` with a bare ``No module named 'jax'``. The guard
    therefore sits in ``_jax_compat`` as well as at the public façade, and
    both doors are measured.
    """
    got = _run_without_jax(
        """
        import stelling.preconditions as P

        out = {}
        for label, call in (
            ("check", lambda: P.check(lambda: (), vacuity_mode="inputs-only")),
            ("field_positive", lambda: P.field_positive((2,), "f", (0.0, 1.0))),
            ("private_module", lambda: __import__("stelling._jax_compat")),
        ):
            try:
                call()
            except BaseException as e:
                out[label] = {
                    "raised": True,
                    "type": type(e).__module__ + "." + type(e).__qualname__,
                    "msg": str(e),
                }
            else:
                out[label] = {"raised": False}
        print(json.dumps(out))
        """
    )

    for label in ("check", "field_positive", "private_module"):
        detail = got[label]
        assert detail["raised"], f"{label} succeeded with jax blocked: {detail}"
        assert JAX_EXTRA_HINT in detail["msg"], f"{label}: {detail}"
        assert detail["type"] == "stelling._optional.OptionalDependencyError", (
            label,
            detail,
        )


def test_the_documented_hint_is_the_hint_the_code_emits():
    """docs/harness-api.md quotes the install line at the reader; a reworded
    message with a stale doc is the drift this closes."""
    doc = (REPO / "docs" / "harness-api.md").read_text()
    assert JAX_EXTRA_HINT in doc, (
        f"docs/harness-api.md must quote {JAX_EXTRA_HINT!r}, the line "
        "stelling._optional.require actually prints"
    )
