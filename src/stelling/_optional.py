# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Optional-dependency plumbing.

The core package must import cleanly in a bare environment: jax drags in
jaxlib, and the solver wheels are tens of megabytes each. Everything outside
the standard library is therefore declared as an extra in ``pyproject.toml``
and reached exclusively through :func:`require`, never by a module-level
import.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import shutil
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class _Optional:
    module: str  # import name
    dist: str  # distribution name on PyPI
    extra: str  # name under [project.optional-dependencies]
    purpose: str


_OPTIONAL: dict[str, _Optional] = {
    "jax": _Optional("jax", "jax", "jax", "tracing harnesses to jaxprs"),
    "z3": _Optional("z3", "z3-solver", "z3", "the Z3 SMT backend"),
    "cvc5": _Optional("cvc5", "cvc5", "cvc5", "the cvc5 SMT backend"),
}


# The jax series stelling is actually tested against. This is its own fact,
# deliberately NOT derived from the [jax] extra floor in pyproject.toml: the
# floor is loose bootstrap metadata, while this constant is the honest runtime
# claim. Update it only when CI actually verifies a new series.
#
# 0.11 verified 2026-07-17: full suite + census re-run; the series bump
# changed scan's params (ft_in/ft_out flattree metadata, layout params gone)
# and merged Jaxpr/ClosedJaxpr — see design/maintenance-treadmill.md.
#
# EACH ENTRY HAS ITS OWN CI LANE, and until 2026-08-07 one of them did not.
# Every jax job installed `.[solvers,jax]`, whose floor is `jax>=0.5`, so
# every job resolved the NEWEST jax: whatever this tuple said, CI only ever
# exercised one series, and 0.10 rested on a developer's re-verification run.
# `test-jax-0-10` in .github/workflows/ci.yml now pins `jax>=0.10,<0.11` and
# asserts the resolved series, so a second entry here is a second lane there.
# The rule that follows: an entry with no lane is a claim, not a test — add
# the lane with the entry, or do not add the entry.
#
# Running the suite on a real 0.10.2 found the mirror image of the 0.11 bump,
# which is the failure mode to expect from any code written on one series:
# `propagate._is_add_combiner` tested `isinstance(v, ir.ClosedJaxpr)`, which
# is a fact about the jax that produced the param and not about the combiner
# (0.11 merged the two classes; 0.10 has not), so every `.at[].add` row
# declined on 0.10 — VERIFIED becoming UNKNOWN.
#
# TEN TESTS FAILED; NINE OF THEM WERE THIS. Measured by applying the
# `_is_add_combiner` hunk alone: nine go green, and `test_doc_example`
# [quickstart.md:37] does not — it is the `jit.inline` param, a different
# cause with its own fix. An earlier commit message and an earlier version of
# this comment said "ten tests, every one scatter-shaped"; that was not
# measured, and it is wrong.
#
# NOR WAS IT SILENT — and the truth is worse than silence. The coverage line
# disclosed the loss (`1 ⊤ (scatter-add ×1)`), but the note read
# `'scatter-add' has no sound rule for params {...}`, which blames the
# caller's program. There WAS a sound rule; the oracle misread the container.
# The direction is safe — VERIFIED → UNKNOWN, never the reverse — so this was
# capability loss, not unsoundness. What it cost was a true reason.
TESTED_JAX_SERIES = ("0.10", "0.11")


def jax_series_tested(version_str: str) -> bool:
    """Whether a jax version string belongs to a series stelling was tested against."""
    return ".".join(version_str.split(".")[:2]) in TESTED_JAX_SERIES


class OptionalDependencyError(ImportError):
    """An optional dependency is needed but not installed."""


def _spec(name: str) -> _Optional:
    try:
        return _OPTIONAL[name]
    except KeyError:
        raise ValueError(
            f"unknown optional dependency {name!r}; known: {sorted(_OPTIONAL)}"
        ) from None


def available(name: str) -> bool:
    """Return whether optional dependency ``name`` is installed, without importing it."""
    return importlib.util.find_spec(_spec(name).module) is not None


def require(name: str) -> ModuleType:
    """Import and return optional dependency ``name``.

    Raises :class:`OptionalDependencyError` with the exact ``pip install``
    hint if the dependency is not installed. An installed-but-broken
    dependency raises its own error unchanged.
    """
    spec = _spec(name)
    try:
        return importlib.import_module(spec.module)
    except ModuleNotFoundError as e:
        if e.name != spec.module:
            raise  # some module *inside* the dependency is missing, not the dependency
        raise OptionalDependencyError(
            f"{spec.module} is required for {spec.purpose} but is not installed; "
            f'run: pip install "stelling[{spec.extra}]"'
        ) from e


def version(name: str) -> str | None:
    """Installed version of optional dependency ``name``, or None; does not import it."""
    spec = _spec(name)
    if not available(name):
        return None
    try:
        return importlib.metadata.version(spec.dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def cvc5_binary() -> str | None:
    """Path to an external cvc5 binary, or None if none is configured or found.

    Checks the ``STELLING_CVC5`` environment variable first, then ``cvc5`` on
    PATH. An external binary is how you get more than the PyPI wheel ships —
    e.g. a from-source ``--gpl --auto-download`` build with CLN, GLPK, and
    CoCoA — since stelling talks SMT-LIB 2 to it over a subprocess and never
    links or distributes it.
    """
    return os.environ.get("STELLING_CVC5") or shutil.which("cvc5")
