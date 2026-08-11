# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The only file in stelling permitted to name a private jax module.

**Skeleton.** It locates the registry and reports what it found; it does not
read it, install anything, wrap anything, or restore anything yet.

WHY THE EXEMPTION EXISTS, in one paragraph, because a reader who finds this
file needs the reason before the code. The tripwire attaches to
``const_fold_rules``, keyed on ``convert_element_type_p``. The primitive is
public — ``jax.extend.core.primitives`` — and is the same object; the registry
keyed on it is not exported by any public or ``jax.extend`` module. Measured on
both tested series: seven candidate modules, including
``jax.interpreters.partial_eval``, which imports cleanly and does not carry it.
So there is no route to this hook that does not name a private module, and the
choice was an exemption or no feature. ``design/private-jax-boundary.md``
carries the table and the alternatives that were rejected.

WHAT THE EXEMPTION IS, EXACTLY. It is rule 2 only — ``jax._src`` — and it is
pinned to this one path in both controls (the ``jax-import-hygiene`` hook
filters on the anchored path; ``tests/test_import_hygiene.py`` compares the
repo-relative path). **This file is still subject to rule 1**: it may not spell
``import jax`` or ``from jax``, and it does not. The private module is reached
through :func:`importlib.import_module` with a plain literal name — which is
also what the fail-closed contract needs, since this adapter must *probe* and
report rather than import and die. When this grows to need public jax for
real work (tracing, for the self-check), that goes through
``stelling._jax_compat`` like everything else in the package: the exemption
bought one private module, not a second jax boundary.

NOTHING JAX-SHAPED LEAVES THIS MODULE. The function here returns a string, and
so must its successors. That is this rule's requirement — the exemption is void
the moment the private object is reachable from a module that does not have it
— and it is also the tripwire's own, since a finding has to survive an xdist
process boundary. One discipline, two payoffs.
"""

from __future__ import annotations

import importlib

from stelling import _optional

# The private module and the attribute on it. Spelled as data rather than as an
# import statement on purpose: this module must import cleanly with jax absent,
# report ``no-module``, and never raise at import.
PRIVATE_MODULE = "jax._src.interpreters.partial_eval"
REGISTRY_ATTR = "const_fold_rules"


def locate() -> str:
    """Report whether the const-fold registry is where the tripwire expects it.

    A code from the tripwire's fail-closed vocabulary, never an exception and
    never a jax object:

    ``no-module``
        jax is not installed. Static checking is unaffected by this.
    ``no-registry``
        jax is installed and the private module or its registry is not where
        this expects it — a series moved it, which is what a tool keyed on a
        private surface is supposed to survive by refusing rather than by
        guessing.
    ``located``
        the registry is there.

    The rest of the vocabulary (``no-entry``, ``not-invoked``, ``cries-wolf``,
    ``below-floor``, ``unexpected:<ExcType>``) belongs with the code that
    installs and probes a wrapper, which is not here yet. Codes are stable and
    greppable.
    """
    if not _optional.available("jax"):
        return "no-module"
    try:
        module = importlib.import_module(PRIVATE_MODULE)
    except ImportError:
        return "no-registry"
    return "located" if getattr(module, REGISTRY_ATTR, None) is not None else "no-registry"
