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
"""

from __future__ import annotations

from stelling._optional import OptionalDependencyError, available, require

__version__ = "0.1.0"

__all__ = [
    "OptionalDependencyError",
    "__version__",
    "available",
    "require",
]
