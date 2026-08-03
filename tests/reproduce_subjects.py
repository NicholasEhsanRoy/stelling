# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The programs the reproducer acceptance executes — and NOT a test module.

It is separate from ``test_reproduce_acceptance.py`` for a reason the
acceptance itself measured: a reproducer imports the module its target
lives in, so **the target's module must not import stelling**. The
harness module does; the program module must not, or the emitted file
drags the tool back in through the side door and the one independent leg
stops being independent.

That is not an artefact of this test. It is the shape a real user's tree
already has — the program is one module, the stelling harness is another
— and this file exists to keep the acceptance honest about it. Nothing
heavy is imported at module scope, so importing this costs nothing.
"""

from __future__ import annotations


def weno5_central_vs_neighbours(u0, u1, u2, u3, u4):
    """JAX-Fluids' own Jiang-Shu indicators, called unbound and verbatim.

    ``WENO5Base.smoothness`` never touches ``self`` (its coefficients are
    literals in the body) and ``WENO5Base`` is abstract, so ``self=None``
    calls the shipped code with nothing standing in for it — the same
    call ``tests/test_square_acceptance_jaxfluids.py`` makes. Returns the
    two sides of ``beta_1 <= max(beta_0, beta_2)``.
    """
    import jax.numpy as jnp
    from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import (
        WENO5Base,
    )

    b0, b1, b2 = WENO5Base.smoothness(None, u0, u1, u2, u3, u4)
    return b1, jnp.maximum(b0, b2)


def heat_node_max_principle(T):
    """MADDENING's real ``HeatNode.update`` at the flagship's refuting
    configuration (docs/verdict-ledger.md: α = 1.0, n_cells = 4, dt = 0.1,
    T ∈ [0, 100]^4 float32). Returns the two sides of ``T_new <= 100.0``.

    The node is constructed HERE, inside a module-level function, and
    that is the whole shape of the fixture answer: a target needing a
    constructor argument is callable from an emitted file exactly when
    someone wrote the construction down where a file can import it.
    """
    from maddening.nodes.heat import HeatNode

    node = HeatNode(
        "h", timestep=0.1, n_cells=4, length=1.0, thermal_diffusivity=1.0
    )
    return node.update({"temperature": T}, {}, 0.1)["temperature"], 100.0


def underflowing_square(x):
    """``x * x`` against ``0``, in float32 — the DIVERGED case.

    In ℝ the product is positive at every ``x > 0``, so ``x*x <= 0`` is
    genuinely false there. Declared over ``[0, 2^-100]`` in **float32**,
    every product is at most ``2^-200``, which is far below float32's
    smallest subnormal ``2^-149`` and flushes to exactly ``0.0``. The
    violation is real and the program's own dtype cannot hold it.

    A multiplication, deliberately, and not an algebraic identity like
    ``(1 + x) - 1``: XLA's algebraic simplifier rewrites that one to
    ``x`` and the two execution modes then disagree (see
    :func:`absorbed_increment`, which is kept for exactly that). Nothing
    rewrites a product, so this case is stable in both modes, which is
    what DIVERGED requires.
    """
    return x * x, 0.0


def absorbed_increment(x):
    """``(1 + x) - 1`` against ``0`` — the EXECUTION-MODE case.

    In ℝ this IS ``x``, so ``(1+x)-1 <= 0`` is false at every ``x > 0``,
    and in binary64 it is exactly ``0.0`` for every ``x`` below half an
    ulp of 1: the increment is absorbed. **But XLA's algebraic
    simplifier rewrites the expression to ``x``** — measured on jax
    0.11.0, the compiled HLO for this function is literally
    ``tuple(copy(x), 0)`` — so under ``jax.jit`` the violation is present
    and eagerly it is not. Kept because it is the reason the reproducer
    runs both modes and reports which one showed what.
    """
    return (1.0 + x) - 1.0, 0.0


def weight_pair_sum(w0, w1):
    """Two nonlinear weights out of one normalization, against ``1.5``."""
    return w0 + w1, 1.5


def weights_are_normalized(w0, w1):
    """The caller precondition for :func:`weight_pair_sum`.

    A predicate, deliberately — not a measured span. See
    :mod:`stelling.reproduce`'s docstring for the two episodes that
    settled the difference.
    """
    return bool(abs(float(w0) + float(w1) - 1.0) <= 1e-12)


def product_against_bound(a, b):
    """A nonlinear pair no interval alone decides: ``a*b`` vs ``6``."""
    return a * b, 6.0


class _Holder:
    """A target whose method is bound to an instance built at run time —
    the uncallable-target shape, kept here so the acceptance can exhibit
    it without inventing one at the point of use."""

    def sides(self, a, b):
        return product_against_bound(a, b)
