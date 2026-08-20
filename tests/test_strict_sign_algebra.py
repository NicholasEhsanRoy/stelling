# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE STRICT-SIGN CERTIFICATE IS A THEOREM OF ℝ, PINNED AGAINST THE FLOATS
IT IS NOT A THEOREM OF — the B5 follow-up's first item, scheduled by audit
0.2.0 B8a (item 7).

`propagate._Propagator._strict_sign_out` mints "this value is certainly
nonzero, of this sign" by composing rules that are theorems of the ordered
field ℝ: its nonzero elements are closed under `*` and `/`, and its positive
cone under `+`. No magnitude enters any of them, which is why the table
reads no box endpoint and why outward rounding cannot defeat it.

binary64 is not that field. Its nonzero values are NOT closed under `*`, and
the measurement below is the witness: a chain the table certifies negative
whose float64 value is exactly `-0.0`. That is not a defect of the table —
real mode judges *in exact real arithmetic over the declared sets*, and the
ieee face never calls this function (`0 if ieee else ...` at its one call
site). It is the fact the argument rests on, and it belongs in the suite so
that a lowering change reddens a test instead of a log.
"""

from __future__ import annotations

import math

import pytest

X = -1e-120
"""Small enough that the cube underflows, large enough that the square does
not: `1e-240` is a normal binary64, `1e-360` is below even the subnormals."""


def test_a_certified_nonzero_chain_underflows_to_negative_zero_in_binary64():
    """REDDENS ON A LOWERING CHANGE. Every step is nonzero in ℝ; the third
    is `-0.0` in binary64, sign bit set."""
    square = X * X
    cube = square * X
    assert square != 0.0, "the SQUARE must stay nonzero, or the chain is not one"
    assert square == 1e-240
    assert cube == 0.0, "the CUBE is what underflows"
    assert math.copysign(1.0, cube) == -1.0, (
        "and it underflows to -0.0, not +0.0 — the sign survives, the "
        "nonzero-ness does not"
    )
    # ... and in ℝ it is nothing of the kind
    from fractions import Fraction

    exact = Fraction(X) ** 3
    assert exact != 0
    assert exact < 0


def test_jax_lowers_the_same_chain_to_the_same_negative_zero():
    """The same measurement through jax's own float64, because the claim is
    about the program stelling was pointed at and not about python."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        x = jnp.float64(X)
        cube = float(x * x * x)
    finally:
        jax.config.update("jax_enable_x64", old)
    assert cube == 0.0
    assert math.copysign(1.0, cube) == -1.0


def test_the_table_certifies_that_chain_negative_under_real_semantics():
    """The other half of the pair: over a declared box every point of which
    cubes to `-0.0` in binary64, the real-mode table certifies the chain
    STRICTLY NEGATIVE — and `div` is licensed by it.

    Measured on `aabb58d` and unchanged by B8a: with a strict assume
    excluding the zero, the chain is `a` (-1) -> `a*a` (+1) -> `(a*a)*a`
    (-1) -> `1.0/that` (-1), while every point of that declared box cubes
    to `-0.0` in binary64.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from stelling import propagate as P
    from stelling.harness import any_array, assert_, assume, trace

    def h():
        a = any_array((), jnp.float64, (-1e-100, -1e-200))
        # THE certificate's one source for a declared input: a STRICT assume
        # excluding the zero (`_classify_cmp`). The declared box alone mints
        # nothing — a box is an over-approximation, and `-1e-200` is not
        # "certainly nonzero" for that reason.
        assume(a < -1e-200)
        c = a * a * a
        return assert_(1.0 / c <= -1.0)

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        cj = trace(h)
    finally:
        jax.config.update("jax_enable_x64", old)

    p = P._Propagator("constrain")
    p.run(cj.jaxpr, list(cj.consts), [])
    muls = [e for e in cj.jaxpr.eqns if e.primitive == "mul"]
    divs = [e for e in cj.jaxpr.eqns if e.primitive == "div"]
    assert len(muls) == 2 and len(divs) == 1
    assert p.strict_sign[muls[0].outvars[0].id] == 1, "a*a is certified +"
    assert p.strict_sign[muls[1].outvars[0].id] == -1, "(a*a)*a is certified -"
    assert p.strict_sign[divs[0].outvars[0].id] == -1

    # and the same walk under ieee mints NOTHING: the short-circuit at the
    # call site is the boundary between a theorem of ℝ and a claim about
    # the program's floats
    q = P._Propagator("constrain", "ieee")
    q.run(cj.jaxpr, list(cj.consts), [])
    assert q.strict_sign == {}, q.strict_sign
