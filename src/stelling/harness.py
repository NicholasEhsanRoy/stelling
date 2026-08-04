# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The harness API: declare inputs and obligations inside a traced function.

``any_array(shape, dtype, (lo, hi))`` declares an arbitrary bounded input;
``any_pytree(tree, bounds)`` is tracing-time sugar declaring one bounded
input per array leaf of a prototype pytree (identical trace, hence
identical content hash, to the hand declaration); ``assume(pred)`` records
an assumption (inert in the MVP propagation); ``assert_(pred)`` states an
obligation. The declarations bind real jax primitives (``stelling_any`` /
``stelling_assume`` / ``stelling_assert``) so they land in the traced
jaxpr — the query's content hash covers them. ``trace(harness)`` returns
the transcribed :class:`stelling.ir` query.

This module names no jax symbol of its own — the primitives live in
``stelling._jax_compat``, the only module allowed to import jax — which
keeps the import-hygiene boundary intact. It is **not** jax-free at run
time, and the earlier claim here that it needed the extra "at call time"
was wrong: every name it re-exports is a bound jax primitive, so
:mod:`stelling.harness` requires the ``[jax]`` extra **at import time**.
That is the honest contract, and it is the one the documented usage
wants: ``from stelling.harness import any_array`` is the first line of
every harness in the docs, so import and first use are the same moment
for every real caller, and a lazy façade would only delay the failure
past the line that caused it.

What the extra is needed *for* is therefore said here, at the public
door: without the guard below a jax-less environment gets a bare
``No module named 'jax'`` raised inside ``stelling/_jax_compat.py``, a
private module the user never typed, naming neither ``stelling`` nor the
extra that fixes it.
"""

from __future__ import annotations

from stelling._optional import require

# Ask for jax by name before touching the private module that imports it, so
# the frame that raises is the module the user actually imported. `require`
# raises OptionalDependencyError — 'jax is required for tracing harnesses to
# jaxprs but is not installed; run: pip install "stelling[jax]"' — and
# re-raises anything else (a broken jaxlib, say) unchanged. A no-op when jax
# is installed: the import below then finds it in sys.modules. _jax_compat
# carries the same guard for the lazy callers that reach it directly.
require("jax")

from stelling._jax_compat import (  # noqa: E402  (must follow the guard above)
    any_array,
    any_pytree,
    assert_,
    assume,
    nonvacuity,
    trace,
)

__all__ = ["any_array", "any_pytree", "assert_", "assume", "nonvacuity", "trace"]
