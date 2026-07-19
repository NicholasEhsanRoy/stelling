# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Assumed-precondition obligation templates — input-side, pre-solve.

The class (`design/precondition-class.md`): properties a numerical
method's validity rests on, living in the input/coefficient data before
the expensive computation, that the code assumes — often literally
asserts to a library — and never verifies, and that fail silently (a
wrong answer under a healthy-looking convergence flag). Pointwise or
scalar properties of declared inputs: no solve, no mask, no relational
machinery — the interval core decides these today.

**The class boundary is exactly the solve.** These templates state
preconditions on the *inputs* to a solve (coefficient positivity over an
envelope, a config scalar being nonzero). Properties of the *solve's
behaviour* — conditioning, residual-implies-error — are out of class; an
obligation that needs the solve's behaviour, or that crosses an
active-set mask, has left this module's territory.

Two templates are implemented. Four more class members are named in
`design/precondition-class.md` for the template design and deliberately
not implemented until an instance demands them: operator symmetry,
non-zero/positive diagonal, special-function domain, and (input-side
only) operating-envelope bounds.

Each template declares its inputs via :func:`stelling.harness.any_array`
— so the envelope lands in the traced query and the content hash covers
it — applies the *code's own* transform, and states the obligation. The
transform argument exists so the harness traces the target's real
construction (faithfulness); passing a hand-simplified transform is an
L11 hand step and should be disclosed and priced wherever the verdict
travels.

This module imports jax-free (the harness import happens inside the
templates, at call time): importing it costs nothing in a bare
environment, but *calling* a template requires the ``[jax]`` extra —
tracing happens through the harness primitives.
"""

from __future__ import annotations

__all__ = ["field_positive", "scalar_nonzero"]


def field_positive(shape, dtype, envelope, transform=None, *, bound=0.0):
    """Pointwise positivity/coercivity of a (transformed) input field.

    Declares ``field`` over ``envelope`` and states
    ``transform(field) > bound`` at every point. The canonical instance
    is variable-coefficient ellipticity: every CG or Cholesky on
    ``-∇·(a∇·)`` assumes the coefficient ``a`` is positive, and the
    assumption usually lives in the inputs unchecked.

    Returns ``(field, obligation)`` — return the obligation from the
    harness so it cannot be dropped as dead code; the field is returned
    so callers can state further obligations over the same declaration.
    """
    from stelling.harness import any_array, assert_

    field = any_array(shape, dtype, envelope)
    value = transform(field) if transform is not None else field
    return field, assert_(value > bound)


def scalar_nonzero(dtype, envelope):
    """A config scalar the method's well-posedness needs to be nonzero.

    Declares the scalar over ``envelope`` (its admissible configuration
    range — pose the *range*, not only the default: the question is
    whether the configuration space admits the singular value, not
    whether the default happens to avoid it) and states ``scalar ≠ 0``.
    The canonical instance is a mass/shift term that removes a
    differential operator's nullspace under periodic boundary
    conditions.

    Returns ``(scalar, obligation)``.
    """
    from stelling.harness import any_array, assert_

    scalar = any_array((), dtype, envelope)
    return scalar, assert_(scalar != 0.0)
