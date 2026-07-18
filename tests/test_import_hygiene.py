# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Import hygiene: jax only in _jax_compat; private jax modules nowhere;
solvers never at module level — the zero-dep surface must import clean."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JAX_IMPORT = re.compile(r"^\s*(import jax\b|from jax\b)")
# built by concatenation so this test file cannot match itself
PRIVATE_JAX = "jax." + "_src"
SOLVER_MODULES = ("z3", "cvc5")


def test_jax_imported_only_in_compat_module():
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
