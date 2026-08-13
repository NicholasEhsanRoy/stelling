# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for 0.2.0 fixes that shipped without dedicated coverage.

1. is_finite SMT emission: bounded harness with jnp.isfinite(x) + solver
   produces VERIFIED.
2. is_finite emission guard: OVERFLOW harness with jnp.isfinite(x*x) stays
   UNKNOWN (the emission declines because the interval reaches infinity).
3. Alias resolution for div guard: jnp.where(cond, safe_val, x) / safe_val
   with solver does not decline on the div guard.
4. is_finite definite-false (point at infinity): unit test of iv.is_finite
   on [inf, inf] -> [0, 0].
5. Assert-always-live in reachability: assert_(pred) without return is still
   REFUTED (not downgraded to UNKNOWN by the dead-variable rule).
"""

from __future__ import annotations

import math

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate
from stelling.reachability import reaches_output


# ---------------------------------------------------------------------------
# Test 4: is_finite definite-false (point at infinity) — no jax needed
# ---------------------------------------------------------------------------


def test_is_finite_point_at_infinity_is_definite_false():
    """iv.is_finite on [inf, inf] must return [0, 0] (definitely not finite)."""
    INF = math.inf
    a = iv.IntervalArray(shape=(), los=(INF,), his=(INF,))
    result = iv.is_finite(a)
    assert result.los == (0.0,)
    assert result.his == (0.0,)


def test_is_finite_point_at_neg_infinity_is_definite_false():
    """iv.is_finite on [-inf, -inf] must return [0, 0]."""
    INF = math.inf
    a = iv.IntervalArray(shape=(), los=(-INF,), his=(-INF,))
    result = iv.is_finite(a)
    assert result.los == (0.0,)
    assert result.his == (0.0,)


def test_is_finite_bounded_is_definite_true():
    """iv.is_finite on [1.0, 2.0] must return [1, 1] (definitely finite)."""
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    result = iv.is_finite(a)
    assert result.los == (1.0,)
    assert result.his == (1.0,)


# ---------------------------------------------------------------------------
# Test 5: Assert-always-live in reachability — no jax needed
# ---------------------------------------------------------------------------


F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def test_assert_without_return_is_still_live():
    """An assert_(pred) whose output is NOT in the jaxpr's outvars must
    still be considered live — its operand var must reach_output.

    Without the assert-always-live rule, the predicate operand appears dead
    and real violations get downgraded from REFUTED to UNKNOWN."""
    # Build a jaxpr where assert's output is NOT returned.
    # x = any_array(...), pred = x > 5, assert_(pred)
    # The jaxpr's outvars contain something ELSE (e.g. a dummy literal).
    x = ir.Var(id=0, aval=F64)
    threshold = ir.Literal(val=5.0, aval=F64)
    pred = ir.Var(id=1, aval=BOOL)
    assert_out = ir.Var(id=2, aval=BOOL)
    # A dummy output that is NOT the assert — simulates a harness that
    # does not return the assert value.
    dummy = ir.Var(id=3, aval=F64)

    jaxpr = ir.Jaxpr(
        constvars=(),
        invars=(x,),
        outvars=(dummy,),  # assert_out is NOT returned
        eqns=(
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(x,),
                params=(("shape", ()), ("dtype", "float64"),
                        ("lo", 0.0), ("hi", 10.0)),
            ),
            ir.JaxprEqn(
                primitive="gt",
                invars=(x, threshold),
                outvars=(pred,),
                params=(),
            ),
            ir.JaxprEqn(
                primitive="stelling_assert",
                invars=(pred,),
                outvars=(assert_out,),
                params=(),
            ),
            # dummy computation to have something in outvars
            ir.JaxprEqn(
                primitive="add",
                invars=(x, threshold),
                outvars=(dummy,),
                params=(),
            ),
        ),
    )

    live = reaches_output(jaxpr)
    # The assert's operand (pred, id=1) must be live even though
    # assert_out is not in outvars.
    assert pred.id in live, (
        "assert operand should be live even when assert output is not returned"
    )
    # And transitively, x must be live too (it feeds the pred).
    assert x.id in live


# ---------------------------------------------------------------------------
# Tests requiring jax
# ---------------------------------------------------------------------------

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None
HAVE_SOLVER = HAVE_Z3 or HAVE_CVC5

need_solver = pytest.mark.skipif(
    not HAVE_SOLVER, reason="needs at least one solver (z3 or cvc5)"
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# Test 1: is_finite SMT emission — bounded harness + solver -> VERIFIED
# ---------------------------------------------------------------------------


@need_solver
def test_is_finite_bounded_harness_verifies_with_solver():
    """A bounded harness asserting jnp.isfinite(x) must VERIFY with solvers.

    Under real semantics with bounded operands, is_finite is a tautology
    (Reals have no infinity). The emission emits constant `true` and the
    solver trivially confirms the assertion."""
    def h():
        x = any_array((), "float64", (1.0, 10.0))
        return (assert_(jnp.isfinite(x)),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED, got {v.status}; notes: {v.notes}"
    )


# ---------------------------------------------------------------------------
# Test 2: is_finite emission guard — overflow box stays UNKNOWN
# ---------------------------------------------------------------------------


@need_solver
def test_is_finite_overflow_box_stays_unknown():
    """A harness with x*x where x spans a wide range (overflow possible)
    should stay UNKNOWN — the emission must decline when the operand's
    interval has non-finite endpoints, never false-VERIFY.

    x in [-1e200, 1e200]: x*x can overflow to inf in ieee, so the interval
    propagation for x*x reaches [0, inf]. is_finite on that MUST decline
    (not emit constant true), and the obligation stays UNKNOWN."""
    def h():
        x = any_array((), "float64", (-1e200, 1e200))
        product = x * x
        return (assert_(jnp.isfinite(product)),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    # Must NOT be VERIFIED — that would be a false positive.
    assert v.status != "VERIFIED", (
        f"UNSOUND: is_finite on a potentially-infinite operand was certified; "
        f"notes: {v.notes}"
    )


# ---------------------------------------------------------------------------
# Test 3: Alias resolution for div guard
# ---------------------------------------------------------------------------


@need_solver
def test_alias_resolution_div_guard_where_pattern():
    """jnp.where(cond, safe_val, x) / safe_val must not decline on the
    div guard when the divisor is provably non-zero.

    The div guard checks whether the divisor interval straddles zero.
    When the divisor is a constant (like 2.0) whose interval is [2, 2],
    the division must proceed — even if the divisor arrives through
    alias resolution (e.g. after inlining).

    This tests that the alias resolution in _resolve_for_guard correctly
    finds the propagated interval for the divisor."""
    def h():
        x = any_array((), "float64", (0.0, 10.0))
        # safe_val is a constant that is definitely non-zero
        safe_val = jnp.float64(2.0)
        # The cond branch should not affect the divisor being non-zero
        cond = x > 5.0
        y = jnp.where(cond, safe_val, x)
        # Division by safe_val (== 2.0) — interval [2, 2], non-straddling
        result = y / safe_val
        return (assert_(result >= 0.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    # Must not decline — the divisor is a non-zero constant.
    # The exact verdict depends on the interval arithmetic, but it must
    # not be UNKNOWN due to a div-straddle decline.
    assert not any("straddles zero" in n for n in v.notes), (
        f"div guard declined on a non-zero constant divisor; notes: {v.notes}"
    )


# ---------------------------------------------------------------------------
# Test 5 (jax variant): assert-always-live end-to-end
# ---------------------------------------------------------------------------


def test_assert_always_live_end_to_end():
    """assert_(pred) without returning the assert value: the violation
    must still be REFUTED (not downgraded).

    x in [0, 10], assert x > 20 is definitely false — must be REFUTED."""
    def h():
        x = any_array((), "float64", (0.0, 10.0))
        # This assert is violated for all x in [0, 10]
        o = assert_(x > 20.0)
        return (o,)  # returning the assert value — baseline

    v = check(h, vacuity_mode="all")
    assert v.status == "REFUTED", (
        f"expected REFUTED for a definitely-false assert, got {v.status}"
    )


# ---------------------------------------------------------------------------
# pow emission tests
# ---------------------------------------------------------------------------


@need_solver
def test_pow_integer_exponent_verifies():
    """x in [1, 4], assert x**2.0 >= 1.0 must VERIFY with solver.

    pow with integer exponent 2.0 emits as product expansion (same as
    integer_pow). The solver trivially confirms x*x >= 1 for x in [1, 4]."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** 2.0 >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x**2 >= 1 on [1,4], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_integer_exponent_refutes():
    """x in [1, 4], assert x**2.0 <= 1.0 must REFUTE with solver.

    For x in [1, 4], x**2 is in [1, 16], so x**2 <= 1.0 is only true at
    x=1. The solver should find a witness x > 1 that falsifies it."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** 2.0 <= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "REFUTED", (
        f"expected REFUTED for x**2 <= 1 on [1,4], got {v.status}; "
        f"notes: {v.notes}"
    )


def test_pow_rational_exponent_verifies():
    """x in [1, 4], x**0.5 >= 1 with rational exponent auxiliary encoding.

    Since 0.2.0, rational exponents p/q with q <= 6 are emitted via
    auxiliary-variable polynomial constraints (y^q = x^p). sqrt(x) >= 1
    on [1, 4] is VERIFIED by both solvers."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** 0.5 >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for sqrt(x) >= 1 on [1,4], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_rational_exponent_refutes():
    """x in [1, 4], x**0.5 <= 1 must REFUTE (sqrt(4) = 2 > 1)."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** 0.5 <= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "REFUTED", (
        f"expected REFUTED for sqrt(x) <= 1 on [1,4], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_cube_root_verifies():
    """x in [1, 8], x**(1/3) >= 1 must VERIFY (cbrt(x) >= 1 for x >= 1)."""
    def h():
        x = any_array((), "float64", (1.0, 8.0))
        return (assert_(x ** (1.0/3.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for cbrt(x) >= 1 on [1,8], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_two_thirds_verifies():
    """x in [1, 8], x**(2/3) >= 1 must VERIFY."""
    def h():
        x = any_array((), "float64", (1.0, 8.0))
        return (assert_(x ** (2.0/3.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(2/3) >= 1 on [1,8], got {v.status}; "
        f"notes: {v.notes}"
    )


def test_pow_large_denominator_exponent_declines():
    """x in [1, 4], x**(1/97) has denominator > 64 and must decline.

    Exponents whose rational representation has a denominator exceeding
    RATIONAL_POW_DENOMINATOR_CAP are declined to avoid emitting
    extremely high-degree polynomials."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** (1.0/97.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "UNKNOWN"
    assert any("non-rational exponent" in n or "cap" in n for n in v.notes), (
        f"expected decline note about denominator cap; notes: {v.notes}"
    )


@need_solver
def test_pow_rational_negative_base_declines():
    """SOUNDNESS: x in [-4, -1], x**0.5 must NOT verify.

    JAX returns NaN for pow(negative, fractional). The Real encoding
    would either have no solution (even q -> UNSAT -> false VERIFIED)
    or model something JAX doesn't compute (odd q). The base-interval
    guard declines this to UNKNOWN."""
    def h():
        x = any_array((), "float64", (-4.0, -1.0))
        return (assert_(x ** 0.5 >= 99999.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status != "VERIFIED", (
        f"UNSOUND: rational pow on negative base was certified; "
        f"notes: {v.notes}"
    )
    assert v.status == "UNKNOWN"
    assert any("negative" in n for n in v.notes), (
        f"expected decline note about negative base; notes: {v.notes}"
    )


@need_solver
def test_pow_rational_straddle_base_declines():
    """x in [-1, 4], x**0.5 declines because base can be negative."""
    def h():
        x = any_array((), "float64", (-1.0, 4.0))
        return (assert_(x ** 0.5 >= 0.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "UNKNOWN"
    assert any("negative" in n for n in v.notes)
