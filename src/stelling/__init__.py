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

# PEP 440 development version, and the reason it is not "0.1.0" or "0.2.0".
# Every verdict stamps this as provenance (`Stamp.stelling_version`), and
# SOUNDNESS.md's per-finding "which versions are affected" rows key on the
# distinction between the released `v0.1.0` and a 0.2.0 development build —
# 29 references at the time of writing. A development build that stamped
# "0.1.0" pointed a reader at the wrong rows in BOTH directions: it claimed
# defects this tree fixed, and it disclaimed the ones scoped to "0.2.0
# development builds only". Stamping "0.2.0" would be the mirror error, a
# development build indistinguishable from the release. `.dev0` claims
# neither, sorts before "0.2.0", and makes those rows answerable from the
# stamp. It becomes "0.2.0" at release.
__version__ = "0.2.0.dev0"

__all__ = [
    "EagerTruncationError",
    "OptionalDependencyError",
    "__version__",
    "available",
    "intentional_wrap",
    "require",
]
