# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Harness-driven verification for JAX scientific computing.

``stelling`` traces verification harnesses to jaxprs and discharges them with
SMT solvers, abstract interpretation, and a region-aware fuzzer. See
``design/founding.md`` in the repository for the roadmap.

Everything heavier than the standard library — ``jax`` and the SMT backends —
is an optional extra, imported on first use:

* ``pip install "stelling[jax]"`` — tracing harnesses
* ``pip install "stelling[z3]"`` — Z3 backend
* ``pip install "stelling[cvc5]"`` — cvc5 backend
* ``pip install "stelling[all]"`` — everything

``python -m stelling`` reports which of these are importable and runs a
one-formula smoke test against each installed solver.

The jax-free IR lives in :mod:`stelling.ir`; jax itself is touched only
inside :mod:`stelling._jax_compat`, the designated churn boundary.

Two names are exported here rather than from a submodule, because both are
things a user writes in their own source and one of them is a type they may
have to name in an ``except`` clause:

* :func:`intentional_wrap` -- declare that an integer constant is MEANT to
  wrap into a narrower dtype, and get the wrapped value. Pure Python, needs
  no jax, and behaves identically whether or not anything is armed.
* :class:`EagerTruncationError` -- what the opt-in eager construction-site
  detector raises. It inherits from ``BaseException`` so that an ordinary
  ``except Exception:`` cannot swallow a soundness alarm;
  ``stelling/_tripwire/eager.py`` carries that argument in full, including
  what the choice does NOT claim and what it costs.

**Importing them imports no jax.** ``stelling._tripwire.eager`` is pure
Python and reaches the adapter lazily, inside :func:`~stelling._tripwire.arm_eager`.
"""

from __future__ import annotations

from stelling._optional import OptionalDependencyError, available, require
from stelling._tripwire.eager import EagerTruncationError, intentional_wrap
from stelling._tripwire.perimeter import NarrowingError

# PEP 440 RELEASE version. It read "0.2.0.dev0" from 2026-08-11 (`a4e4056`)
# to 2026-08-24, and the argument for that is kept rather than deleted,
# because it is the argument for why this string is readable at all and
# because the window it describes is what "0.2.0 development builds only"
# now names.
#
# WHAT IT MEASURED, and re-derived rather than retyped. Every verdict stamps
# this as provenance (`Stamp.stelling_version`), and SOUNDNESS.md's
# per-finding "which versions are affected" rows key on the distinction
# between the released `v0.1.0` and a 0.2.0 development build. The figure
# written here was "29 references at the time of writing";
# `git show a4e4056^:SOUNDNESS.md | grep -o 'v0\.1\.0' | wc -l` returns 29,
# and the same count on this tree (2026-08-24) is 61. The countable unit
# today is sharper than a grep: `## Log` carries 54 top-level bullets, each
# with exactly one `Versions:` field from a closed set of three — 39 scoped
# to 0.1.0 pre-release builds, 8 to 0.2.0 development builds only, 7 to
# `v0.1.0` and 0.2.0 development builds — and `tests/test_soundness_log_reach.py`
# derives the reached-release count from those fields.
#
# WHY IT WAS NOT "0.1.0" AND NOT "0.2.0" THEN. A development build that
# stamped "0.1.0" pointed a reader at the wrong rows in BOTH directions: it
# claimed defects that tree had fixed, and it disclaimed the ones scoped to
# "0.2.0 development builds only". Stamping "0.2.0" before release was the
# mirror error, a development build indistinguishable from the release.
# `.dev0` claimed neither and sorted before "0.2.0".
#
# WHAT THE TREE READS NOW, and what changed in meaning with it. "0.2.0", set
# on 2026-08-24 for the 0.2.0 release. The `.dev0` window is CLOSED, which is
# what makes the mirror error impossible now: a build from this tree stamps
# the release, and "0.2.0 development builds only" is a claim about the
# builds that stamped `0.2.0.dev0` — no release, this one included. Each such
# row was re-read against this release rather than carried over; the ones
# that survive record defects fixed before the bump, and their `Versions:`
# field is exactly as narrow as it was.
__version__ = "0.2.0"

__all__ = [
    "EagerTruncationError",
    "NarrowingError",
    "OptionalDependencyError",
    "__version__",
    "available",
    "intentional_wrap",
    "require",
]
