# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Objects bound HERE whose class or function lives somewhere else.

An object is not bound where its class is defined. The target resolver
drew its candidate modules only from the object's own, its type's and its
``func``'s ``__module__``, so a module-level callable instance or a
``functools.partial`` assembled in one module out of parts from another —
the shape the fixture guidance in ``docs/reproducing-a-witness.md``
recommends — was reported uncallable and the reproducer said nothing
about a program it could have run.

Deliberately a separate file: putting these beside their class is exactly
what hid the defect from the first round's tests.
"""

from __future__ import annotations

import functools

from reproduce_subjects import _CallableInstance, product_against_bound

# the class is in reproduce_subjects; the NAME is here
BOUND_ELSEWHERE = _CallableInstance()
PARTIAL_ELSEWHERE = functools.partial(product_against_bound)


def lazily_reaches_stelling(a, b):
    """Imports stelling INSIDE the call, so no scan of this module's top
    level sees it and no ``sys.modules`` check made before the call does
    either."""
    import stelling  # noqa: F401

    return product_against_bound(a, b)


def lazily_reaching_precondition(a, b):
    """A caller PRECONDITION that reaches stelling lazily, while running.

    It has to live in a module that does not import stelling at module
    scope, or importing the precondition already loads the tool and the
    phase this fixture exists to distinguish never happens.
    """
    import stelling  # noqa: F401

    return True


def lazily_reaches_stelling_then_raises(a, b):
    """Loads the tool during the call and then raises in BOTH modes.

    The path that showed the disclosure was still placement-dependent
    after being moved into `_sidecar`: it keyed on a phase recorded at
    four fixed points, and this one returns before the last of them. The
    disclosure keys on ``sys.modules`` now, and the recorded phase only
    supplies the attribution.
    """
    import os

    import stelling  # noqa: F401  — lazy, during the call

    if os.environ.get("STELLING_REPRO_RAISE"):
        raise ValueError("raises in every execution mode")
    return product_against_bound(a, b)


def only_mentions_it_in_prose(a, b):
    """A module whose text contains the line but never runs it::

        import stelling

    The source scan accused this; nothing here loads the tool.
    """
    return product_against_bound(a, b)
