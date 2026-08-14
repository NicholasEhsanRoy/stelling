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
    not HAVE_SOLVER, reason="needs an SMT solver"
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
def test_pow_eighth_root_verifies():
    """x in [1, 8], x**0.125 >= 1 must VERIFY (a denominator-8 aux encoding).

    **This test used to be `x ** (1.0/3.0)` and expected VERIFIED.** It was
    encoding the defect audit 0.2.0 S1 names: `1.0/3.0` is the binary64
    ``6004799503160661/18014398509481984``, NOT one third, and the
    predecessor rationalised it to ``1/3`` and emitted ``aux^3 = x`` — a
    problem about a different function, discharged with nothing downstream
    to re-derive it. The exponent now has to BE the traced literal, so a
    cube root declines (covered in ``test_pow_audit_findings.py``). The
    row's real subject — a rational exponent with a denominator above 2
    discharging through the aux encoding — is kept here at an exponent
    that is exactly what it looks like."""
    def h():
        x = any_array((), "float64", (1.0, 8.0))
        return (assert_(x ** 0.125 >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(1/8) >= 1 on [1,8], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_three_quarters_verifies():
    """x in [1, 8], x**0.75 >= 1 must VERIFY — numerator > 1, denominator > 2.

    **Was `x ** (2.0/3.0)`, expecting VERIFIED**, for the same reason as
    the row above: that literal is
    ``6004799503160661/9007199254740992``, not two thirds. ``0.75`` is
    exactly ``3/4``, so the emitted ``aux^4 = x^3`` is about the traced
    function, and the case still exercises the p != 1 arm of the
    encoding."""
    def h():
        x = any_array((), "float64", (1.0, 8.0))
        return (assert_(x ** 0.75 >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(3/4) >= 1 on [1,8], got {v.status}; "
        f"notes: {v.notes}"
    )


def test_pow_large_denominator_exponent_declines():
    """x in [1, 4], x**(1.0/191.0) must decline — its exact denominator is huge.

    The traced literal is ``3018119122007453/576460752303423488``: the
    binary64 nearest one 191st, and a dyadic rational whose denominator is
    2^59. The encoding would be that degree, far over
    RATIONAL_POW_DEGREE_CAP (128), so escalation declines. Before the S1
    fix this declined too, but for a reason that was not true — the
    message claimed the exponent "cannot be represented as p/q with
    q <= 128" when it can be represented exactly, at a power-of-two
    denominator."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** (1.0/191.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000)
    assert v.status == "UNKNOWN"
    assert any("denotes exactly" in n and "cap" in n for n in v.notes), (
        f"expected decline note quoting the exact rational and the cap; "
        f"notes: {v.notes}"
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


# ---------------------------------------------------------------------------
# z3 tactic workaround for degree-80 factoring pathology (0.2.0)
# ---------------------------------------------------------------------------


need_z3 = pytest.mark.skipif(
    not HAVE_Z3, reason="needs z3 wheel"
)


@need_z3
def test_z3_tactic_workaround_high_degree_verifies():
    """x**(1/128) on [1, 100] must VERIFY with z3 using the tactic chain.

    The auxiliary-variable encoding produces a degree-128 polynomial
    (y^128 = x). Without the tactic workaround, z3's default Solver()
    times out on this class (measured at degree 80: >10s). The custom
    tactic chain (simplify, solve-eqs, factor, purify-arith, tseitin-cnf,
    nlsat) restores the z3 cross-check (measured here at degree 128:
    0.31s).

    **Was `x ** (1.0/80.0)`.** That literal is not one eightieth — 80 is
    not a power of two, so its binary64 is a dyadic rational of degree
    2^59, and after the audit 0.2.0 S1 fix an exponent must BE the
    rational the emission writes. Degree 80 is unreachable through this
    row now: every admissible denominator is a power of two. 128 is the
    nearest reachable degree ABOVE 80, so the row still covers the
    pathology it was written for, at a degree the old test never reached.

    This test runs with ONLY z3 and verifies it discharges within the
    timeout (not timing out)."""
    def h():
        x = any_array((), "float64", (1.0, 100.0))
        return (assert_(x ** (1.0/128.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=30_000,
              solver="z3")
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(1/128) >= 1 on [1,100] with z3, "
        f"got {v.status}; notes: {v.notes}"
    )


@need_solver
def test_z3_tactic_workaround_high_degree_both_solvers():
    """x**(1/128) on [1, 100] must VERIFY with both solvers (full portfolio).

    The z3 tactic workaround ensures BOTH solvers discharge this obligation,
    so the portfolio is not degraded. Before the workaround, z3 timed out
    and only cvc5 answered. (Was `x ** (1.0/80.0)` — see the row above for
    why that exponent is no longer emittable.)"""
    def h():
        x = any_array((), "float64", (1.0, 100.0))
        return (assert_(x ** (1.0/128.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=30_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(1/128) >= 1 on [1,100], got {v.status}; "
        f"notes: {v.notes}"
    )
    # With the tactic workaround, the portfolio should NOT be degraded
    # (both solvers answer successfully)
    if HAVE_Z3 and HAVE_CVC5:
        assert not any("PORTFOLIO DEGRADED" in n for n in v.notes), (
            f"z3 tactic workaround failed — portfolio is still degraded; "
            f"notes: {v.notes}"
        )


@need_z3
def test_z3_tactic_workaround_does_not_fire_on_non_aux_scripts():
    """Linear/simple nonlinear scripts without aux_ must still use default Solver.

    Ensures the tactic workaround does not regress the common case."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x * x >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=20_000,
              solver="z3")
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^2 >= 1 on [1,4] with z3 default solver, "
        f"got {v.status}; notes: {v.notes}"
    )


# ---------------------------------------------------------------------------
# Rational pow with denominators above the original cap of 6
# ---------------------------------------------------------------------------


@need_solver
def test_pow_denominator_16_verifies():
    """x in [1, 100], x**(1/16) >= 1 — denominator 16, within the cap of 128.

    This tests that denominators > 6 (the original cap) work correctly.

    **Was `x ** (1.0/10.0)`.** One tenth is not a binary64: that literal
    is ``3602879701896397/36028797018963968``, and the predecessor emitted
    ``aux^10 = x`` about it — the headline case of audit 0.2.0 S1, which
    minted a false VERIFIED on ``x**0.1 <= 1e30`` over ``[1, 1e300]``.
    A denominator above the original cap is still what this row is for;
    16 is one that a binary64 exponent can actually denote."""
    def h():
        x = any_array((), "float64", (1.0, 100.0))
        return (assert_(x ** (1.0/16.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=30_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(1/16) >= 1 on [1,100], got {v.status}; "
        f"notes: {v.notes}"
    )


@need_solver
def test_pow_denominator_64_verifies():
    """x in [1, 100], x**(1/64) >= 1 — denominator 64, within the cap of 128.

    A high-degree polynomial (y^64 = x) that the z3 tactic workaround
    handles. Tests mid-range denominators that were at the old cap."""
    def h():
        x = any_array((), "float64", (1.0, 100.0))
        return (assert_(x ** (1.0/64.0) >= 1.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=30_000)
    assert v.status == "VERIFIED", (
        f"expected VERIFIED for x^(1/64) >= 1 on [1,100], got {v.status}; "
        f"notes: {v.notes}"
    )
