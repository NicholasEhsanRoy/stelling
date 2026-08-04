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
time: ``_jax_compat`` imports jax at module scope and the six names below
are re-exported from it, so importing this module imports jax. It
requires the ``[jax]`` extra **at import time**, not at first call; the
earlier claim here was the other way round, and measured, it is this one.

Eager rather than lazy, and deliberately. ``from stelling.harness import
any_array`` is the first line of every harness in the docs, and that form
raises *at that line* even behind a module ``__getattr__``: a lazy façade
would delay only a bare ``import stelling.harness``, which no documented
example writes, so it would buy the documented caller nothing while
costing the re-export below its static visibility — a name produced by
``__getattr__`` is not there to be read until it is asked for.

What the extra is needed *for* is said at this door rather than left to
the private bridge: ``require("jax")`` below raises
:class:`stelling.OptionalDependencyError` naming the extra from
``harness.py``, the file the user typed.
"""

from __future__ import annotations

from stelling._optional import require

# Ask for jax by name before touching the private module that imports it.
# This line does not supply the message: `_jax_compat` carries the same guard
# for the callers that reach it directly, so with this one deleted the type,
# the ImportError-ness and the sentence are byte-identical. What it buys is
# the FRAME — the exception is raised from the module the user typed, and the
# traceback never descends into the private bridge. Measured, that is the one
# assertion that fails without it (`"_jax_compat.py" not in frames`, in
# tests/test_import_hygiene.py). `require` re-raises anything that is not a
# missing jax (a broken jaxlib, say) unchanged, and is a no-op when jax is
# installed: the import below then finds it in sys.modules.
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
