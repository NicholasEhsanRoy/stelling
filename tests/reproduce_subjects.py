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

import functools


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


def mixed_dtype_underflow(x32, unused64):
    """float32 underflow beside an untouched float64 declaration.

    Two things at once, both of them audit findings: the DIVERGED detail
    must name every declared dtype (it hard-wired ``envelope[0]``), and a
    declaration the obligation never reaches produces a witness element no
    solver and no replay ever assigned — which the sidecar must mark as
    invented rather than publish beside solved values.
    """
    return x32 * x32, 0.0


def scatter_write(T):
    """``(T*2).at[0].set(0)`` against ``100`` — the standard JAX write.

    ``.at[i].set(v)`` is the functional-update idiom every JAX program
    uses, and it does not exist on ``numpy.ndarray``. A reproducer that
    built numpy inputs raised ``AttributeError`` here and reported "no
    execution result" for a violation it could have confirmed — while the
    identical numpy input worked fine under ``jax.jit``. Kept as an
    acceptance, because the most idiomatic write pattern in the library
    must not be the one that silently yields nothing.
    """
    return (T * 2.0).at[0].set(0.0), 100.0


def only_under_jit(a, b):
    """Raises on a concrete argument and runs on a tracer.

    The shape of the numpy-input defect, kept as a target in its own
    right: the reproducer must try BOTH modes and report the one that
    ran, rather than stopping at the first that raised. It tests the
    argument's type in plain Python — no jax internals — because that is
    exactly how the original failure worked (``.at[].set()`` exists on a
    tracer and not on ``numpy.ndarray``).
    """
    if "Tracer" not in type(a).__name__:
        raise TypeError(
            "only_under_jit: this target runs only under jax.jit; eagerly "
            "it raises, which is the shape the two-mode rule exists for"
        )
    return a * b, 6.0


def donating_step(x):
    """A donated input buffer — the standard JAX step-function idiom.

    ``jax.jit(..., donate_argnums=0)`` DELETES its argument. The two
    execution modes shared one set of buffers, so the eager call destroyed
    the input and the jit call then raised "Array has been deleted"; the
    file reported the eager answer as the whole answer and printed
    "Nothing here is wrong" at a witness where the compiled form violates.
    Each mode now builds its own inputs.
    """
    import jax

    donated = jax.jit(lambda v: v * 1.0, donate_argnums=0)(x)
    return (1.0 + donated) - 1.0, 0.0


def widening_square(x):
    """``float64(x)**2`` against 0 — a target whose ANSWER depends on
    ``jax_enable_x64``.

    Under x64 off, the float64 request is truncated to float32 and the
    product underflows to exactly 0.0, so the assertion holds: DIVERGED.
    With x64 forced on it is computed in float64, stays positive, and the
    assertion is false: CONFIRMED. The emitted file must restore the
    setting its query was traced under, and this is what measures that it
    does — mutating the file to force ``True`` survived every test before
    it existed.
    """
    import jax.numpy as jnp

    wide = jnp.asarray(x, jnp.float64)
    return wide * wide, 0.0


# Two targets whose EXECUTION behaviour is switched by an environment
# variable the tracing process does not set. A target that simply raises
# cannot be traced at all, so a harness could never be built from it — and
# a test that used an UNCONSTRUCTIBLE target instead measured the "nothing
# was attempted" shape rather than the "every attempt failed" one it was
# named for.


def raises_in_every_mode(a, b):
    """Traces cleanly; raises in BOTH execution modes.

    Silence is correct here, and this is the only shape it is correct for.
    """
    import os

    if os.environ.get("STELLING_REPRO_RAISE"):
        raise ValueError("this target raises in every execution mode")
    return a * b, 6.0


def always_producible(a, b):
    """A caller precondition that HOLDS at every point of the envelope.

    Not a stand-in for "no precondition": it is declared, it runs, and it
    returns true — which the sidecar must be able to report as a MEASURED
    reachability rather than the ``null`` that means "not declared".
    """
    return True


def eager_only_absorbs(x):
    """Traces cleanly; HOLDS eagerly and cannot run under ``jax.jit``.

    The numpy round-trip is ordinary interop that works on a concrete
    array and raises ``TracerArrayConversionError`` on a tracer. Eagerly
    the increment is absorbed and the assertion holds; under jit there is
    no measurement at all. DIVERGED must not be claimed from the one mode
    that ran — an unmeasured mode is not an agreeing one.
    """
    import os

    import jax.numpy as jnp
    import numpy as np

    if os.environ.get("STELLING_REPRO_NUMPY"):
        x = jnp.asarray(np.asarray(x))
    return (1.0 + x) - 1.0, 0.0


def overflowing_bound(x, bound):
    """A witness FORCED above the declared dtype's finite range.

    ``x`` is declared float32 over a box that reaches past float32's
    finite maximum; ``bound`` is a float64 point at 3.5e38, above that
    maximum, so any witness must exceed it and therefore converts to inf.
    The comparison is widened to float64 because a float32 literal at
    3.5e38 would itself be inf and the assertion would be vacuous.

    ``_nearest`` computed ``Fraction(float(inf))``, which raises
    OverflowError, so the file reported NO EXECUTION RESULT for a
    violation the program does exhibit. Overflow to inf IS the correctly
    rounded conversion; there is no finite neighbour to search for.
    """
    import jax.numpy as jnp

    return jnp.asarray(x, jnp.float64) * 1.0, bound


class _Holder:
    """A target whose method is bound to an instance built at run time,
    with NO module-level name holding that instance — the genuinely
    uncallable shape, kept here so the acceptance can exhibit it without
    inventing one at the point of use."""

    def sides(self, a, b):
        return product_against_bound(a, b)


# ── three importable shapes a file CAN name ──────────────────────────────────
#
# None is a plain `def`, all three were reported uncallable, and each one
# resolves to a module-level name here. A reproducer that reports "no
# execution result" for a target it could have executed is wrongly silent,
# which is a defect in the same family as a wrong result.


class ClassMethodTarget:
    """A classmethod: it HAS a ``__self__`` (the class) and is importable."""

    @classmethod
    def sides(cls, a, b):
        return product_against_bound(a, b)


class _CallableInstance:
    """The flax/equinox shape: a module-level callable instance. An
    instance does not inherit its class's ``__qualname__``."""

    def __call__(self, a, b):
        return product_against_bound(a, b)


CALLABLE_INSTANCE = _CallableInstance()
PARTIAL_TARGET = functools.partial(product_against_bound)
# two names for one object: resolution must pick the same one every run,
# or a file headed "GENERATED" is not reproducible from its own inputs
ALIASED_TARGET = _CallableInstance()
ZZ_SECOND_NAME_FOR_ALIASED = ALIASED_TARGET
