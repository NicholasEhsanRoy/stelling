# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The BY-NAME gate class, pinned.

A census constraint checks MEMBERSHIP: a primitive is in a registry or the
import raises. It cannot check a gate that keys on primitive NAMES, because
adding the primitive to every registry satisfies the census while leaving the
name tuple untouched. That is not hypothetical — registering `add_any` with the
same IEEE function object as `add` passed all seven census constraints and
produced a false discharge, because the FMA-contraction gate is a name tuple.

Two name gates guard the two ends of that one hazard:

    IEEE_PRODUCT_SOURCES     which outputs are product-derived (the SOURCE)
    IEEE_CONTRACTION_ADDENDS which additions a product may fuse into (the SINK)

Omission from either silently skips the hull. These tests are the enforcement
the census cannot provide.
"""
from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import propagate as P  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402

A = math.nextafter(1.0, 2.0)
M = -(A * A)


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _status(make_product, addend_prim="add"):
    """`product + (-fl(product))` in ieee mode: 0.0 under two roundings,
    4.93e-32 under one fused one. A transfer that models only the two-rounding
    answer discharges `<= 0.0`; a hulled one cannot.

    `add_any` is reached through `jax.grad`, not through a private-module bind.
    That is both the project's import rule and the more faithful test: `add_any`
    is jax's cotangent accumulator, so reverse-mode is how a user meets it.
    """
    if addend_prim == "add_any":
        def build():
            x = any_array((1,), "float64", (A, A))

            def loss(v):
                # two contributions to one cotangent -> jax accumulates them
                # with add_any, and the A* term makes the accumuland a product
                return (A * jnp.sum(v * x) + M * jnp.sum(v))[()]

            g = jax.grad(loss)(jnp.ones((1,), jnp.float64))
            return (assert_(g <= 0.0),)
    else:
        def build():
            x = any_array((1,), "float64", (A, A))
            out = make_product(x) + jnp.asarray([M])
            return (assert_(out <= 0.0),)

    cj = transcribe(jax.make_jaxpr(build)())
    prims = {str(e.primitive) for e in cj.jaxpr.eqns}
    assert addend_prim in prims, (
        f"fixture did not produce {addend_prim!r}; it produced {sorted(prims)} "
        f"— the probe would be testing a different program"
    )
    r = P.propagate(cj, semantics="ieee")
    return r.obligations[0].status


@pytest.mark.parametrize("addend", ["add", "add_any"])
def test_every_contraction_addend_is_hulled_against_the_fused_rounding(addend):
    """`add_any` was missing here and it was a FALSE DISCHARGE, not a latent
    one: measured under jit, HLO `multiply_add_fusion`, executed
    4.930380657631324e-32 against a box of [0.0, 0.0]."""
    assert addend in P.IEEE_CONTRACTION_ADDENDS
    assert _status(lambda x: x * jnp.asarray([A]), addend) == "unknown", (
        f"{addend!r} discharged an obligation that is FALSE at the single "
        f"point of its declared box under XLA's fused multiply-add"
    )


def test_the_probe_can_actually_see_the_defect_it_checks_for():
    """Anti-vacuity. Remove `add` from the addend tuple and the same query must
    discharge — otherwise these tests pass for a reason unrelated to the gate,
    and a real regression would pass too."""
    orig = P.IEEE_CONTRACTION_ADDENDS
    P.IEEE_CONTRACTION_ADDENDS = tuple(p for p in orig if p != "add")
    try:
        assert _status(lambda x: x * jnp.asarray([A]), "add") == "discharged"
    finally:
        P.IEEE_CONTRACTION_ADDENDS = orig
    assert _status(lambda x: x * jnp.asarray([A]), "add") == "unknown"


PRODUCTS = {
    "mul": lambda x: x * jnp.asarray([A]),
    "dot_general": lambda x: jax.lax.dot_general(
        x, jnp.asarray([A]), (((0,), (0,)), ((), ()))),
    "integer_pow": lambda x: x ** 2,
}


@pytest.mark.parametrize("prim", sorted(PRODUCTS))
def test_every_product_source_is_tainted_or_declines(prim):
    """THE CANARY, and it is written to fail rather than to pass quietly.

    `dot_general` and `integer_pow` produce products XLA contracts — measured,
    `bitcast_add_fusion` and `multiply_add_fusion`, both executing to
    4.930380657631324e-32. They were absent from the taint sources and that was
    LATENT, not safe: neither has an ieee transfer, so both decline to ⊤ before
    the taint is consulted. Protection by an unrelated decline is what the
    scatter bar's unreachability was, and it stops being protection the moment
    the missing rule is added.

    So this asserts the DISJUNCTION. If someone gives one of these an ieee
    transfer and does not taint it, the decline disappears, the box narrows to
    the two-rounding answer, and this test fails naming the primitive.
    """
    assert prim in P.IEEE_PRODUCT_SOURCES, (
        f"{prim!r} produces a product XLA may fuse into a following add; it "
        f"must taint its output or the contraction hull cannot fire"
    )
    assert _status(PRODUCTS[prim]) == "unknown"


def test_the_two_gates_are_constants_not_inline_literals():
    """Both ends must be importable, because the tests that cover them derive
    from these names rather than repeating a hand-written list beside them —
    which is how the sink gate came to be tested nowhere for `add_any`."""
    assert isinstance(P.IEEE_CONTRACTION_ADDENDS, (tuple, frozenset))
    assert isinstance(P.IEEE_PRODUCT_SOURCES, (tuple, frozenset))
    assert "mul" in P.IEEE_PRODUCT_SOURCES
    assert {"add", "sub", "add_any"} <= set(P.IEEE_CONTRACTION_ADDENDS)
