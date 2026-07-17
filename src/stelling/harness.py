# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The harness API: declare inputs and obligations inside a traced function.

``any_array(shape, dtype, (lo, hi))`` declares an arbitrary bounded input;
``assume(pred)`` records an assumption (inert in the MVP propagation);
``assert_(pred)`` states an obligation. All three bind real jax primitives
(``stelling_any`` / ``stelling_assume`` / ``stelling_assert``) so the
declarations land in the traced jaxpr — the query's content hash covers
them. ``trace(harness)`` returns the transcribed :class:`stelling.ir`
query.

This module is a jax-free façade: the primitives live in
``stelling._jax_compat`` (the only module allowed to import jax), so
importing :mod:`stelling.harness` requires the ``[jax]`` extra at call
time but keeps the import-hygiene boundary intact.
"""

from __future__ import annotations

from stelling._jax_compat import any_array, assert_, assume, nonvacuity, trace

__all__ = ["any_array", "assert_", "assume", "nonvacuity", "trace"]
