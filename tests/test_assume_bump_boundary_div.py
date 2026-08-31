# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""0.2.0 precision pair: ieee assume-bump and boundary-aware division.

Feature A: In ieee mode, `assume(x > 0)` narrows to
`[min_positive_for_format, hi]` rather than `[0, hi]`. This is EXACT in
ieee mode because there is no representable float between 0 and the
format's smallest positive value.

Feature B: When the divisor has zero at exactly one boundary (not spanning
both signs), compute a meaningful result instead of declining. This makes
`assume(b > 0)` followed by `a / b` produce useful bounds in both modes.

Hand-built IR throughout -- no jax needed.
"""

from __future__ import annotations

import math
import os
import pathlib
import struct

import pytest

from stelling.coverage import DEFAULT_TRANSPARENT
from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate, interval_env

INF = math.inf
# Smallest positive float32 subnormal: 2^(emin - p + 1) = 2^(-126-24+1) = 2^(-149)
F32_MIN_POSITIVE = math.ldexp(1.0, -149)
# Smallest positive float64 subnormal: 2^(-1022-53+1) = 2^(-1074)
F64_MIN_POSITIVE = math.ldexp(1.0, -1074)

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi, dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)
        )
    )


# --- Test 1: assume(b > 0) + a/b > 0 in ieee float32 -> VERIFIED -----------

def _assume_div_f32_query(bound: float):
    """assume(b > bound) + assert_(a/b > 0) over a in [1, 10], b in [-10, 10],
    all float32."""
    a = var(0, F32)
    b = var(1, F32)
    pred_assume = var(2, BOOL)
    assume_out = var(3, BOOL)
    q_out = var(4, F32)
    pred = var(5, BOOL)
    out = var(6, BOOL)
    return close(
        [
            any_eqn(a, 1.0, 10.0, dtype="float32"),
            any_eqn(b, -10.0, 10.0, dtype="float32"),
            eqn("gt", [b, lit(bound, F32)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0, F32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_assume_gt0_div_ieee_f32_no_longer_decides_and_this_is_the_price():
    """**A COVERAGE LOSS, recorded as one — audit 0.2.0 S10.**

    This asserted `discharged`: the ieee assume-bump narrows `b` to
    `[2**-149, 10]`, which excludes zero, and boundary-aware division then
    gave `[a_lo/10, +inf]`.

    The bump lands on the format's smallest SUBNORMAL, which is inside the
    subnormal band by construction, so the DAZ haze immediately hulls the
    divisor back to `[0, 10]` (measured: `_elt_haze_fmt(2**-149, 10, 2**-126)
    == (0.0, 10.0)`). The kernel is then handed a divisor box containing
    zero — and cannot tell WHICH zero, because `_elt_haze_fmt` hulls with the
    positive literal `0.0` and an endpoint carries no sign bit. Under IEEE
    `a / -0.0` is `-inf` for positive `a`, so the old `[positive, +inf]`
    excluded an attainable value.

    The DAZ flush of a positive subnormal does produce `+0.0` on the targets
    measured — so the old answer was right for a reason the domain does not
    carry. Restoring it needs the zero's sign IN THE DOMAIN, threaded through
    every kernel that can produce or consume one; inferring it here from the
    haze's own provenance would put a sign bit on a value only some producers
    set, which is the half-done version S10 exists to warn about.

    **The row is not dead**, and the companion below is the boundary: an
    assume whose bound is ABOVE the format's subnormal band keeps its
    tightening, because no haze puts zero back.
    """
    p = propagate(_assume_div_f32_query(0.0), semantics="ieee")
    assert p.obligations[0].status == "unknown", (
        f"expected unknown after S10; got {p.obligations[0].status}"
    )
    assert any("narrowed x1 (IR var 1)" in n for n in p.notes), p.notes


def test_assume_above_the_subnormal_band_still_divides_in_ieee_f32():
    """The other side of the same boundary: `assume(b > 1e-30)` in float32
    narrows to a divisor the haze leaves alone (1e-30 is far above float32's
    smallest normal, 2**-126), so no zero re-enters and the quotient is
    bounded away from 0. Boundary-aware division is withdrawn only where the
    divisor box actually reaches zero."""
    p = propagate(_assume_div_f32_query(1e-30), semantics="ieee")
    assert p.obligations[0].status == "discharged", (
        f"expected discharged, got {p.obligations[0].status}; notes: {p.notes}"
    )


# --- Test 2: assume(b > 0) + a/b > 0 in real mode -> VERIFIED ---------------

def test_assume_gt0_div_positive_real():
    """assume(b > 0) + assert_(a/b > 0) in real mode: boundary-aware
    division sees b = [0, 10] and computes [a_lo/10, +inf], which is > 0
    when a_lo > 0."""
    a, b = var(0), var(1)
    pred_assume = var(2, BOOL)
    assume_out = var(3, BOOL)
    q_out, pred, out = var(4), var(5, BOOL), var(6, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0),
            any_eqn(b, -10.0, 10.0),
            # assume(b > 0)
            eqn("gt", [b, lit(0.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # a / b
            eqn("div", [a, b], q_out),
            # assert a/b > 0
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "discharged", (
        f"expected discharged, got {p.obligations[0].status}; notes: {p.notes}"
    )


# --- Test 3: assume(b > 0) + a/b < 100 in real mode -> UNKNOWN ---------------

def test_assume_gt0_div_upper_bound_real_unknown():
    """assume(b > 0) + assert_(a/b < 100) in real mode: upper bound is
    +inf (b can approach 0), so the comparison is indeterminate."""
    a, b = var(0), var(1)
    pred_assume = var(2, BOOL)
    assume_out = var(3, BOOL)
    q_out, pred, out = var(4), var(5, BOOL), var(6, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0),
            any_eqn(b, -10.0, 10.0),
            # assume(b > 0)
            eqn("gt", [b, lit(0.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # a / b
            eqn("div", [a, b], q_out),
            # assert a/b < 100
            eqn("lt", [q_out, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "unknown", (
        f"expected unknown, got {p.obligations[0].status}; notes: {p.notes}"
    )


# --- Test 4: Division by [0, 0] still declines --------------------------------

def test_div_literal_zero_still_declines():
    """Division by [0, 0] (literal zero): still declines."""
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 5.0),
            any_eqn(b, 0.0, 0.0),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "unknown"


# --- Test 5: Division by [-5, 5] (true straddle) still declines ---------------

def test_div_true_straddle_still_declines():
    """Division by [-5, 5] (true straddle with lo < 0 < hi): still declines
    with the existing straddle message."""
    a, b = var(0), var(1)
    q_out, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 5.0),
            any_eqn(b, -5.0, 5.0),
            eqn("div", [a, b], q_out),
            eqn("gt", [q_out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "unknown"
    assert any("straddles zero" in n for n in p.notes)


# --- Test 6: Float32 ieee assume(b > 0) narrows to [min_subnormal, hi] --------

def test_ieee_f32_assume_gt0_bumps_to_min_positive():
    """In ieee float32 mode, assume(b > 0) narrows to
    [float32_min_subnormal, hi], not [0, hi]. Verified via the propagation
    note which reports the narrowed interval."""
    b = var(0, F32)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(b, -10.0, 10.0, dtype="float32"),
            eqn("gt", [b, lit(0.0, F32)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            eqn("ge", [b, lit(0.0, F32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
    # The propagation note reports the narrowed interval; verify the bump
    narrowing_notes = [n for n in p.notes if "narrowed x0 (IR var 0)" in n]
    assert narrowing_notes, f"expected a narrowing note, got: {p.notes}"
    note = narrowing_notes[0]
    # Should contain the min_positive value, NOT 0
    assert str(F32_MIN_POSITIVE) in note, (
        f"expected {F32_MIN_POSITIVE} in narrowing note, got: {note}"
    )
    assert "[0.0," not in note and "[0," not in note, (
        f"lo should NOT be 0 in ieee mode, got: {note}"
    )


# --- Test 7: Real mode assume(b > 0) still narrows to [0, hi] ----------------

def test_real_mode_assume_gt0_no_bump():
    """In real mode, assume(b > 0) narrows to [0, hi] (no ULP bump --
    sound overapproximation since reals exist between 0 and any positive
    float). Verified via interval_env with constrain mode."""
    b = var(0)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(b, -10.0, 10.0),
            eqn("gt", [b, lit(0.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            eqn("ge", [b, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    # Use interval_env with constrain mode to read the narrowed interval
    env = interval_env(query, assume_mode="constrain")
    b_box = env.get(0)
    assert b_box is not None
    # In real mode: the closed half-space for gt is [0, inf], meet with
    # [-10, 10] gives [0, 10]. No bump.
    assert b_box.los[0] == 0.0, (
        f"expected lo = 0.0 in real mode, got {b_box.los[0]}"
    )
    assert b_box.his[0] == 10.0


# --- Additional boundary-aware division tests --------------------------------

def test_boundary_div_negative_dividend_positive_boundary():
    """a in [-5, -1], b = [0, 2]: boundary-aware division gives [-inf, -1/2]."""
    a = iv.IntervalArray(shape=(), los=(-5.0,), his=(-1.0,))
    b = iv.IntervalArray(shape=(), los=(0.0,), his=(2.0,))
    result = iv.boundary_div(a, b)
    assert result.los[0] == -INF
    # upper bound: ahi/bhi = -1/2 (rounded outward)
    assert result.his[0] <= -0.5 + 1e-15  # approximately -0.5


def test_boundary_div_negative_boundary_divisor():
    """a in [2, 4], b = [-3, 0]: boundary-aware division gives [-inf, 2/(-3)]."""
    a = iv.IntervalArray(shape=(), los=(2.0,), his=(4.0,))
    b = iv.IntervalArray(shape=(), los=(-3.0,), his=(0.0,))
    result = iv.boundary_div(a, b)
    assert result.los[0] == -INF
    # upper bound: alo/blo = 2/(-3) (rounded outward)
    assert result.his[0] <= -2.0/3.0 + 1e-15


# --- General ieee bump: works for all k values --------------------------------

def test_assume_gt_negative_bumps_correctly():
    """assume(x > -5) in ieee float64: narrows to [nextafter(-5, inf), hi],
    which is approximately -4.999...9. assert(x > -1) is still UNKNOWN
    because the interval includes values in [-5+eps, -1]."""
    x = var(0)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(x, -10.0, 10.0),
            # assume(x > -5)
            eqn("gt", [x, lit(-5.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # assert(x > -1): still indeterminate (values in (-5, -1] exist)
            eqn("gt", [x, lit(-1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
    assert p.obligations[0].status == "unknown", (
        f"expected unknown (values in (-5, -1] exist), "
        f"got {p.obligations[0].status}; notes: {p.notes}"
    )
    # Verify the narrowing note shows the bumped value (not -5.0 exactly)
    narrowing_notes = [n for n in p.notes if "narrowed x0 (IR var 0)" in n]
    assert narrowing_notes, f"expected a narrowing note, got: {p.notes}"
    note = narrowing_notes[0]
    # The bumped lo should be nextafter(-5, inf) ≈ -4.999999999999999
    assert "-5.0," not in note, (
        f"lo should be bumped past -5.0 in ieee mode, got: {note}"
    )
    assert "-4.9999999999" in note, (
        f"expected bumped lo near -5, got: {note}"
    )


def test_assume_gt_positive_bumps_in_f32():
    """assume(x > 1.0) in ieee float32: narrows to [1 + 2^-23, hi].
    assert(x >= 1 + 2^-23) should verify."""
    import math
    next_f32_above_1 = 1.0 + 2**-23  # = 1.0000001192092896
    x = var(0, F32)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(x, 0.5, 2.0, dtype="float32"),
            # assume(x > 1.0)
            eqn("gt", [x, lit(1.0, F32)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # assert(x >= next_f32_above_1): should verify since lo = next_f32_above_1
            eqn("ge", [x, lit(next_f32_above_1, F32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
    assert p.obligations[0].status == "discharged", (
        f"expected discharged (x >= 1+2^-23 after bump), "
        f"got {p.obligations[0].status}; notes: {p.notes}"
    )


def test_assume_lt_negative_bumps_in_f64():
    """assume(x < -5) in ieee float64: narrows to [lo, nextafter(-5, -inf)].
    assert(x < -5) should then verify (since hi < -5)."""
    x = var(0)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(x, -10.0, 10.0),
            # assume(x < -5)
            eqn("lt", [x, lit(-5.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # assert(x < -5): should verify since hi = nextafter(-5, -inf) < -5
            eqn("lt", [x, lit(-5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
    assert p.obligations[0].status == "discharged", (
        f"expected discharged (hi < -5 after bump), "
        f"got {p.obligations[0].status}; notes: {p.notes}"
    )


def test_real_mode_never_bumps_nonzero():
    """In real mode, assume(x > -5) narrows to [-5, 10] — NO bump,
    because reals exist between -5 and nextafter(-5, inf)."""
    x = var(0)
    pred_assume = var(1, BOOL)
    assume_out = var(2, BOOL)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    query = close(
        [
            any_eqn(x, -10.0, 10.0),
            # assume(x > -5)
            eqn("gt", [x, lit(-5.0)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # assert(x >= -5): should verify in real mode (lo = -5, -5 >= -5)
            eqn("ge", [x, lit(-5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "discharged"
    # Verify the narrowing is [-5, 10] (no bump)
    env = interval_env(query, assume_mode="constrain")
    x_box = env.get(0)
    assert x_box is not None
    assert x_box.los[0] == -5.0, (
        f"expected lo = -5.0 in real mode, got {x_box.los[0]}"
    )


# --- Precondition guards: boundary_div rejects invalid divisors ---------------


def test_boundary_div_rejects_point_at_zero():
    """boundary_div([1,2], [0,0]) raises IntervalError, not ZeroDivisionError."""
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    b = iv.IntervalArray(shape=(), los=(0.0,), his=(0.0,))
    with pytest.raises(iv.IntervalError, match="point-at-zero"):
        iv.boundary_div(a, b)


def test_boundary_div_rejects_true_straddle():
    """boundary_div([1,2], [-1,1]) raises IntervalError, not wrong result."""
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    b = iv.IntervalArray(shape=(), los=(-1.0,), his=(1.0,))
    with pytest.raises(iv.IntervalError, match="true straddle"):
        iv.boundary_div(a, b)


def test_boundary_div_valid_positive_boundary():
    """boundary_div([1,2], [0,5]) still works (valid one-sided boundary)."""
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    b = iv.IntervalArray(shape=(), los=(0.0,), his=(5.0,))
    result = iv.boundary_div(a, b)
    # a >= 0 and b = [0, 5]: result is [1/5, +inf]
    assert result.his[0] == INF
    assert result.los[0] >= 0.2 - 1e-15  # approximately 1/5


def test_boundary_div_valid_negative_boundary():
    """boundary_div([1,2], [-5,0]) still works (valid one-sided boundary)."""
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    b = iv.IntervalArray(shape=(), los=(-5.0,), his=(0.0,))
    result = iv.boundary_div(a, b)
    # a >= 0 and b = [-5, 0]: result is [-inf, 1/(-5)]
    assert result.los[0] == -INF
    assert result.his[0] <= -0.2 + 1e-15  # approximately -1/5


# --- a CONSTANT operand carries its own strict sign -------------------------
#
# `read_strict_sign` used to answer 0 for every literal, on a docstring
# rationale that reasoned only about a literal DIVISOR. But `_strict_sign_out`
# reads EVERY operand of every rule, so a literal COEFFICIENT zeroed the whole
# chain through it: `0.5 * sum(x*x)`, `2.0 * x`, `x / 2.0` and the `/n` inside
# `jnp.mean` all lost the certificate on the coefficient. `_literal_strict_sign`
# answers from the literal's own decoded value instead.

from fractions import Fraction  # noqa: E402
from itertools import product  # noqa: E402

from stelling.propagate import _int_bracket, _literal_strict_sign  # noqa: E402


def any_eqn_shaped(out, lo, hi, shape, dtype="float64"):
    """`any_eqn`, but for a non-scalar declaration."""
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def _sign_of(val, aval=F64):
    return _literal_strict_sign(lit(val, aval))


def test_literal_strict_sign_reads_a_plain_nonzero_scalar():
    """The point of the fix: a nonzero finite literal answers its sign."""
    assert _sign_of(0.5) == 1
    assert _sign_of(2.0) == 1
    assert _sign_of(-2.0) == -1
    assert _sign_of(1e-300) == 1
    assert _sign_of(-1e-300) == -1


def test_literal_strict_sign_drops_zero_and_nonfinite():
    """Zero is not signed, and neither is anything non-finite.

    ``inf`` is nonzero but breaks the very rules that consume this fact:
    ``a / inf = 0``, so a certificate minted off an infinite operand would
    claim NONZERO of a value that is zero.
    """
    assert _sign_of(0.0) == 0
    assert _sign_of(-0.0) == 0
    assert _sign_of(INF) == 0
    assert _sign_of(-INF) == 0
    assert _sign_of(math.nan) == 0


def _saturating_int_document(dtype: str) -> dict:
    """A DOCUMENT whose one literal is ``10**400`` under ``dtype``.

    Built by taking a WELL-FORMED query through `to_dict` and editing the
    single value, so every other key is whatever the encoder actually
    writes rather than what this test guessed it writes — the literal
    cannot be constructed under an integer dtype any more, which is the
    whole subject below.
    """
    a = ir.Aval(kind="ShapedArray", shape=(), dtype=dtype)
    pred, out = var(1, BOOL), var(2, BOOL)
    doc = close(
        [
            eqn("gt", [lit(3, a), lit(1, a)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    ).to_dict()
    written = doc["jaxpr"]["eqns"][0]["invars"][0]
    assert written["k"] == "lit" and written["val"] == 3, written
    written["val"] = 10 ** 400
    return doc


def test_a_saturating_int_literal_is_REFUSED_at_the_from_dict_door():
    """THE RELOCATION, and the honest reading of what it can assert.

    This test used to be
    `test_literal_strict_sign_saturating_int_is_not_finite_and_drops`,
    superseded by
    ::test_a_saturating_int_literal_is_REFUSED_at_the_from_dict_door
    and ::test_the_strict_sign_DROP_on_a_saturating_int_is_STILL_REACHABLE
    between them. It built ``ir.Literal(10**400, Aval(dtype="int64"))``
    DIRECTLY,
    because its subject is that the strict-sign machinery drops a literal
    whose `_int_bracket` saturates to ``(maxf, inf)``. Component LIT
    (`ir._literal_range_problem`) makes that input unconstructible under
    `int64`: 10**400 is not an int64, and the door now says so. The
    principal ruled the test moves to the `from_dict` door with a crafted
    document, because adversarial IR belongs there anyway.

    **`from_dict` IS NOT A SOFTER DOOR, AND THAT IS WHAT THIS ASSERTS.**
    It builds `ir.Literal` objects, so it runs the same `__post_init__`;
    a document carrying an out-of-range int64 literal is REFUSED rather
    than admitted, and the relocated test therefore names THE DOOR'S
    REFUSAL. That is a different claim from the one it used to make, and
    where the original claim went is the next test, which drives the drop
    through the route the rules leave open.
    """
    doc = _saturating_int_document("int64")
    with pytest.raises(ir.TranscriptionError) as exc:
        ir.ClosedJaxpr.from_dict(doc)
    msg = str(exc.value)
    assert "int64's range [-9223372036854775808, 9223372036854775807]" in msg
    assert "would store as" in msg, msg
    # the door's own sentence, which claims exactly this reach
    assert "refused at construction" in msg
    assert "trace, from_dict, or direct construction" in msg
    # ...and the same document under a dtype nothing claims a range for
    # loads, which is what makes the next test's route a real one
    ir.ClosedJaxpr.from_dict(_saturating_int_document("key<fry>"))


def test_the_strict_sign_DROP_on_a_saturating_int_is_STILL_REACHABLE():
    """WHERE THE ORIGINAL CLAIM WENT — established by measurement, not by
    argument, which is what SPEC-LIT asked for.

    The range check makes NO CLAIM on an unrecognised dtype string, the
    same posture `ir._load_itemsize` takes when a dtype code does not name
    a size. jax's extended dtypes really do spell themselves that way —
    ``key<fry>``, which `stelling._tripwire.prop_guard` already names — so
    a literal under one still constructs, still reaches
    :func:`stelling.propagate._int_bracket`, and still saturates. **The
    drop path is LIVE, not dead**, so it keeps a driven test rather than
    a note saying where its coverage went.

    Every link is asserted in order, so the test fails at the link that
    breaks rather than at the end: the door admits the literal;
    `_int_bracket` saturates it to an endpoint of ``inf``;
    `_literal_strict_sign` drops it; and the finite counterparts under
    the SAME aval still answer their sign, which is what makes the drop
    the finiteness guard firing rather than the dtype being unreadable.
    """
    huge = 10 ** 400
    odd = ir.Aval(kind="ShapedArray", shape=(), dtype="key<fry>")
    maxf = math.nextafter(math.inf, 0.0)
    assert ir._literal_range_problem(huge, odd) == (None, None)
    assert _int_bracket(huge) == (maxf, INF)
    assert _int_bracket(-huge) == (-INF, -maxf)
    assert _sign_of(huge, odd) == 0
    assert _sign_of(-huge, odd) == 0
    assert _sign_of(3, odd) == 1
    assert _sign_of(-3, odd) == -1
    # ...and the int64 half of the original test, which is still
    # constructible because 3 and -3 are int64 values
    i64 = ir.Aval(kind="ShapedArray", shape=(), dtype="int64")
    assert _sign_of(3, i64) == 1
    assert _sign_of(-3, i64) == -1


def _f64_array(values):
    return ir.Array(
        dtype="<f8",
        shape=(len(values),),
        data=struct.pack(f"<{len(values)}d", *values),
    )


def test_literal_strict_sign_needs_EVERY_element_same_sign():
    """An ARRAY literal is signed only when all of it is — not the first cell.

    The fact this mints is quantified over every element (see
    `_strict_sign_out`), so a mixed-sign array must drop it. Reading only
    ``los[0]`` would pass the first two of these and mint a false NONZERO
    for the rest.
    """
    aval3 = ir.Aval(kind="ShapedArray", shape=(3,), dtype="float64")
    assert _sign_of(_f64_array([1.0, 2.0, 3.0]), aval3) == 1
    assert _sign_of(_f64_array([-1.0, -2.0, -3.0]), aval3) == -1
    # first element positive, a later one is not
    assert _sign_of(_f64_array([1.0, 2.0, -3.0]), aval3) == 0
    assert _sign_of(_f64_array([1.0, 0.0, 3.0]), aval3) == 0
    # first element negative, a later one is not
    assert _sign_of(_f64_array([-1.0, -2.0, 3.0]), aval3) == 0
    assert _sign_of(_f64_array([-1.0, 0.0, -3.0]), aval3) == 0
    # a single non-finite cell disqualifies the whole array
    assert _sign_of(_f64_array([1.0, INF, 3.0]), aval3) == 0


def test_literal_strict_sign_drops_an_undecodable_literal():
    """A dtype with no zero-dep decoder keeps answering 0, and does not raise."""
    bad = ir.Array(dtype="<c16", shape=(1,), data=b"\x00" * 16)
    aval = ir.Aval(kind="ShapedArray", shape=(1,), dtype="complex128")
    assert _literal_strict_sign(lit(bad, aval)) == 0
    # the NaN-sentinel spelling: a literal of a type with no interval meaning
    assert _literal_strict_sign(
        lit("not-a-number", ir.Aval(kind="ShapedArray", shape=(), dtype="float64"))
    ) == 0


def test_literal_strict_sign_drops_a_size_zero_literal():
    """A size-0 literal certifies nothing about "every element".

    Both all() quantifiers are vacuously true over an empty box, which
    would mint a sign for a value that has none.
    """
    aval0 = ir.Aval(kind="ShapedArray", shape=(0,), dtype="float64")
    assert _sign_of(_f64_array([]), aval0) == 0


def test_a_signed_literal_can_never_reach_the_div_boundary_gate():
    """The divisor case is not weakened, ENUMERATED rather than asserted.

    `_t_div` consults the strict-sign gate only when
    `iv.straddles_zero(divisor)` — some element with ``lo <= 0 <= hi``. A
    literal that answers +-1 has ``lo > 0`` (resp. ``hi < 0``) for EVERY
    element, so it cannot straddle. Hence no literal divisor's newly-minted
    sign can change any `div` outcome: the only literal divisors that reach
    the gate are the ones that still answer 0.
    """
    aval3 = ir.Aval(kind="ShapedArray", shape=(3,), dtype="float64")
    scalars = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-300, -1e-300,
               1e300, -1e300, INF, -INF, math.nan]
    cases = [(lit(v), F64) for v in scalars]
    cases += [
        (lit(_f64_array(list(t)), aval3), aval3)
        for t in product([0.0, 1.0, -1.0, INF], repeat=3)
    ]
    from stelling.propagate import _value_to_interval

    checked = undecodable = 0
    for atom, aval in cases:
        atom = lit(atom.val, aval)
        sign = _literal_strict_sign(atom)
        try:
            box = _value_to_interval(atom.val, aval.shape)
        except (iv.IntervalError, ir.TranscriptionError):
            # an undecodable literal answers 0 and never reaches the gate
            assert sign == 0
            undecodable += 1
            continue
        if sign != 0:
            assert not iv.straddles_zero(box), (
                f"literal {atom.val!r} answered sign={sign} AND straddles "
                f"zero — it would reach the div boundary gate"
            )
        checked += 1
    # 13 scalars, of which NaN raises at decode ("NaN endpoint in interval
    # arithmetic") before the finiteness guard is even consulted, plus every
    # 4**3 array over {0, 1, -1, inf}
    assert undecodable == 1, undecodable
    assert checked == len(scalars) - 1 + 4 ** 3 == 76, checked


# --- the four queries the literal sign newly admits, as hand-built IR --------


def _coeff_query(build, *, lo=0.0, hi=2.0, n=4):
    """`assume(x > 0)`; `assert_(1 / f(x) > 0)` with f built by `build`.

    `build(nxt, eqns, x)` returns the var holding the divisor.
    """
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    xa = ir.Aval(kind="ShapedArray", shape=(n,), dtype="float64") if n else F64
    x = var(0, xa)
    eqns = [any_eqn_shaped(x, lo, hi, (n,) if n else ())]
    pred_assume = nxt(ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool")
                      if n else BOOL)
    assume_out = nxt(ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool")
                     if n else BOOL)
    eqns.append(eqn("gt", [x, lit(0.0)], pred_assume))
    eqns.append(eqn("stelling_assume", [pred_assume], assume_out))
    d = build(nxt, eqns, x)
    q = nxt()
    eqns.append(eqn("div", [lit(1.0), d], q))
    pred = nxt(BOOL)
    out = nxt(BOOL)
    eqns.append(eqn("gt", [q, lit(0.0)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    return close(eqns, [out]), eqns


def _build_sumsq(nxt, eqns, x):
    sq = nxt(x.aval)
    eqns.append(eqn("mul", [x, x], sq))
    s = nxt()
    eqns.append(eqn("reduce_sum", [sq], s, [("axes", (0,))]))
    return s


def _build_half_sumsq(nxt, eqns, x):
    s = _build_sumsq(nxt, eqns, x)
    h = nxt()
    eqns.append(eqn("mul", [lit(0.5), s], h))
    return h


def _build_mean_sq(nxt, eqns, x):
    """`jnp.mean(x*x)` = reduce_sum(x*x) / 4.0 -- the /n is the literal."""
    s = _build_sumsq(nxt, eqns, x)
    m = nxt()
    eqns.append(eqn("div", [s, lit(4.0)], m))
    return m


def _build_two_x(nxt, eqns, x):
    t = nxt()
    eqns.append(eqn("mul", [lit(2.0), x], t))
    return t


def _build_x_half(nxt, eqns, x):
    t = nxt()
    eqns.append(eqn("div", [x, lit(2.0)], t))
    return t


COEFF_QUERIES = {
    "0.5*sum(x*x)": (_build_half_sumsq, 4),
    "mean(x*x)": (_build_mean_sq, 4),
    "2.0*x": (_build_two_x, 0),
    "x/2.0": (_build_x_half, 0),
}


@pytest.mark.parametrize("name", sorted(COEFF_QUERIES))
def test_literal_coefficient_no_longer_kills_the_certificate(name):
    """REDDENS ON REVERT of `_literal_strict_sign`.

    With `read_strict_sign` answering 0 for literals these four are UNKNOWN
    with `div` declining "REACHES zero at a boundary" — while the same
    query without the coefficient is VERIFIED. Nothing about the divisor
    changed; only the coefficient did.
    """
    build, n = COEFF_QUERIES[name]
    query, _ = _coeff_query(build, n=n)
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "discharged", (
        f"{name}: {p.obligations[0].status} — {p.obligations[0].detail}"
    )


def test_the_uncoefficiented_control_was_already_green():
    """The comparison the test above rests on: no coefficient, VERIFIED."""
    query, _ = _coeff_query(_build_sumsq, n=4)
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "discharged"


def test_a_negative_literal_coefficient_flips_the_sign():
    """`assume(x>0)`; `1/(-2.0*x) < 0` — the -1 arm of the literal rule."""
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    x = var(0, F64)
    eqns = [any_eqn(x, 0.0, 2.0, "float64")]
    pa, ao = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("gt", [x, lit(0.0)], pa))
    eqns.append(eqn("stelling_assume", [pa], ao))
    t = nxt()
    eqns.append(eqn("mul", [lit(-2.0), x], t))
    q = nxt()
    eqns.append(eqn("div", [lit(1.0), t], q))
    pred, out = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("lt", [q, lit(0.0)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    p = propagate(close(eqns, [out]), semantics="real")
    assert p.obligations[0].status == "discharged"


# --- the losses that must STAY lost -----------------------------------------


def test_a_literal_ZERO_coefficient_still_drops_the_certificate():
    """`0.0 * x` is zero everywhere; `1/(0.0*x)` must not verify."""
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    x = var(0, F64)
    eqns = [any_eqn(x, 0.0, 2.0, "float64")]
    pa, ao = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("gt", [x, lit(0.0)], pa))
    eqns.append(eqn("stelling_assume", [pa], ao))
    t = nxt()
    eqns.append(eqn("mul", [lit(0.0), x], t))
    q = nxt()
    eqns.append(eqn("div", [lit(1.0), t], q))
    pred, out = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("gt", [q, lit(0.0)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    p = propagate(close(eqns, [out]), semantics="real")
    assert p.obligations[0].status != "discharged"


def test_sub_still_breaks_the_chain_with_a_literal_present():
    """`1/(sum(x*x) - 8.0)`: the false-VERIFIED shape. `sub` stays out.

    The literal `8.0` is now signed, which is exactly the ingredient that
    could have re-opened this had the fix been applied to `sub` too.
    """
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    xa = ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64")
    ba = ir.Aval(kind="ShapedArray", shape=(2,), dtype="bool")
    x = var(0, xa)
    eqns = [any_eqn_shaped(x, 0.0, 2.0, (2,))]
    pa, ao = nxt(ba), nxt(ba)
    eqns.append(eqn("gt", [x, lit(0.0)], pa))
    eqns.append(eqn("stelling_assume", [pa], ao))
    sq = nxt(xa)
    eqns.append(eqn("mul", [x, x], sq))
    s = nxt()
    eqns.append(eqn("reduce_sum", [sq], s, [("axes", (0,))]))
    d = nxt()
    eqns.append(eqn("sub", [s, lit(8.0)], d))
    q = nxt()
    eqns.append(eqn("div", [lit(1.0), d], q))
    pred, out = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("lt", [q, lit(0.0)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    p = propagate(close(eqns, [out]), semantics="real")
    assert p.obligations[0].status != "discharged"


def test_no_assume_still_declines_even_with_a_literal_coefficient():
    """A declared [0,1] divisor with NO assume: the literal cannot rescue it."""
    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    x = var(0, F64)
    eqns = [any_eqn(x, 0.0, 1.0, "float64")]
    t = nxt()
    eqns.append(eqn("mul", [lit(2.0), x], t))
    q = nxt()
    eqns.append(eqn("div", [lit(1.0), t], q))
    pred, out = nxt(BOOL), nxt(BOOL)
    eqns.append(eqn("gt", [q, lit(0.0)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    p = propagate(close(eqns, [out]), semantics="real")
    assert p.obligations[0].status != "discharged"


# --- the exact-Fraction semantic check over the ASSUMED region --------------


_EXACT_RULES = {
    "mul": lambda a, b: [x * y for x, y in zip(a, b)],
    "div": lambda a, b: [x / y for x, y in zip(a, b)],
    "add": lambda a, b: [x + y for x, y in zip(a, b)],
    "sub": lambda a, b: [x - y for x, y in zip(a, b)],
    # the 0.3.0 census additions that are elementwise binaries. `max`/`min`
    # are here as VALUES; the rule's asymmetry (one certified operand is
    # enough) is a claim about the rule, and this table is what decides
    # whether that claim is true of the numbers.
    "max": lambda a, b: [x if x > y else y for x, y in zip(a, b)],
    "min": lambda a, b: [x if x < y else y for x, y in zip(a, b)],
    # comparisons are EVALUATED rather than skipped, because `select_n`
    # needs its selector's value. They were skipped when the only
    # comparison in these queries was the assume's own predicate.
    "gt": lambda a, b: [Fraction(int(x > y)) for x, y in zip(a, b)],
    "lt": lambda a, b: [Fraction(int(x < y)) for x, y in zip(a, b)],
    "ge": lambda a, b: [Fraction(int(x >= y)) for x, y in zip(a, b)],
    "le": lambda a, b: [Fraction(int(x <= y)) for x, y in zip(a, b)],
}

# The UNARY half of the same table. Added when the boundary dial's
# generated search began drawing `neg`, `square` and `abs` — all three are
# members of `propagate._STRICT_SIGN_PRIMITIVES`, so all three can mint a
# certificate this oracle then has to be able to check. Exact rationals
# throughout: `x * x` is exact in `Fraction`, so `square` needs no float.
# The census branch then added the ROUTING members below — `copy`,
# `stop_gradient`, `reshape`, `squeeze` — which are the identity on a
# C-order-flat value, and `sqrt`, which is exact or refuses.
_EXACT_UNARY = {
    "neg": lambda a: [-x for x in a],
    "abs": lambda a: [abs(x) for x in a],
    "square": lambda a: [x * x for x in a],
    "copy": lambda a: list(a),
    "stop_gradient": lambda a: list(a),
    "reshape": lambda a: list(a),      # C-order flat: the elements are the same
    "squeeze": lambda a: list(a),      # ditto — only size-1 axes go away
}


def _exact_sqrt(q):
    """The exact rational square root, or a refusal.

    This evaluator is an EXACT witness or it is nothing: a floating
    `math.sqrt` here would put the propagator's own rounding question back
    into the instrument that is supposed to be independent of it. So the
    queries that exercise the `sqrt` rule are built to square first, and a
    non-square operand raises rather than being approximated."""
    if q < 0:
        raise AssertionError(f"sqrt of a negative rational: {q}")
    n, d = q.numerator, q.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    if rn * rn != n or rd * rd != d:
        raise AssertionError(
            f"sqrt({q}) is irrational and this evaluator is exact or it "
            f"raises — build the query so the operand is a perfect square"
        )
    return Fraction(rn, rd)



def _exact_eval(eqns, point, env=None):
    """Evaluate the arithmetic spine of a built query in exact Fractions.

    `point` is the list of Fractions bound to var 0. Returns
    `var id -> list[Fraction]`, values held FLAT in C order. Only the
    primitives these queries use are implemented; anything else raises, so
    a query that grows a new primitive cannot silently skip its own check.

    **WHAT THIS EVALUATOR DOES NOT REACH.** It is one-dimensional and
    C-order-flat: `concatenate`, `slice` and `split` below are implemented
    for rank-1 operands along axis 0, which is what the queries here build,
    and a rank-2 operand would be evaluated WRONG rather than refused. That
    is why the census's INDEXING members — `gather`, `scatter`,
    `scatter-add`, `dynamic_slice`, `dynamic_update_slice` — are not
    checked here at all: their admitted forms are row shapes over rank >= 1
    operands with dimension-number params, and re-implementing that
    geometry in this evaluator would be a second copy of the thing under
    test. They are checked end-to-end against EXECUTED jax instead, in
    `tests/test_strict_sign_census.py`.

    **IT DESCENDS A TRANSPARENT WRAPPER**, because the boundary dial gave
    the propagator a way to certify a value on the far side of one and
    an oracle that stopped at the wrapper could not check that
    certificate at all. The body is evaluated into the SAME `env`, so an
    inner var id that collided with an outer one would silently clobber
    it — the callers here allocate inner ids from a disjoint block and
    the collision is CHECKED rather than assumed (the assertion below
    fires on a builder that forgets).

    `env` is the recursion's own parameter and is not part of the
    caller's interface: pass `point` and leave it alone.

    **WHAT IT DOES NOT REACH.** It has no `cond` arm. A `cond` is a
    choice, and evaluating one exactly needs the predicate, which needs
    the comparison primitives this evaluator deliberately skips. So a
    certificate carried INTO a cond branch is not checked by this oracle
    — it is checked by the verdict-level tests in
    `tests/test_boundary_dial.py`, and the direction that could mint a
    false VERIFIED (carrying one OUT of a cond) is refused by the
    propagator and driven red there against a deliberately broken build.
    """
    if env is None:
        env = {0: list(point)}

    def val(atom):
        if isinstance(atom, ir.Literal):
            return [Fraction(atom.val)]
        return env[atom.id]

    for e in eqns:
        prim = e.primitive
        if prim in ("stelling_any", "stelling_assume", "stelling_assert",
                    "stelling_nonvacuity"):
            continue

        def val(atom):
            if isinstance(atom, ir.Literal):
                return [Fraction(atom.val)]
            return env[atom.id]
        if prim in DEFAULT_TRANSPARENT:
            body = next(v for k, v in e.params if k == "jaxpr")
            for inner_in, atom in zip(body.jaxpr.invars, e.invars):
                assert inner_in.id not in env, (
                    f"inner invar {inner_in.id} is already bound in this "
                    f"env: the builder reused an outer var id inside a "
                    f"{prim!r} body and this oracle would clobber it"
                )
                env[inner_in.id] = val(atom)
            _exact_eval(body.jaxpr.eqns, None, env)
            for out, inner_out in zip(e.outvars, body.jaxpr.outvars):
                env[out.id] = val(inner_out)
            continue
        out = e.outvars[0].id
        if prim == "reduce_sum":
            env[out] = [sum(val(e.invars[0]), Fraction(0))]
            continue
        if prim == "sqrt":
            env[out] = [_exact_sqrt(x) for x in val(e.invars[0])]
            continue
        if prim in _EXACT_UNARY:
            env[out] = _EXACT_UNARY[prim](val(e.invars[0]))
            continue
        if prim == "select_n":
            which = val(e.invars[0])
            cases = [val(a) for a in e.invars[1:]]
            n = len(cases[0])
            if len(which) == 1:
                which = which * n
            env[out] = [cases[int(which[i])][i] for i in range(n)]
            continue
        if prim == "concatenate":
            assert int(dict(e.params).get("dimension", 0)) == 0
            flat = []
            for a in e.invars:
                flat.extend(val(a))
            env[out] = flat
            continue
        if prim == "slice":
            pr = dict(e.params)
            src = val(e.invars[0])
            lo = tuple(pr["start_indices"])[0]
            hi = tuple(pr["limit_indices"])[0]
            st = (tuple(pr["strides"]) or (1,))[0] if pr.get("strides") else 1
            env[out] = list(src[lo:hi:st])
            continue
        if prim == "split":
            src = val(e.invars[0])
            assert int(dict(e.params).get("axis", 0)) == 0
            off = 0
            for v in e.outvars:
                k = 1
                for d in v.aval.shape:
                    k *= d
                env[v.id] = list(src[off:off + k])
                off += k
            continue
        if prim in _EXACT_RULES:
            a, b = val(e.invars[0]), val(e.invars[1])
            if len(a) == 1:
                a = a * len(b)
            if len(b) == 1:
                b = b * len(a)
            env[out] = _EXACT_RULES[prim](a, b)
            continue
        raise AssertionError(f"no exact rule for {prim!r}")
    return env


def _assumed_points(n, lo, hi, steps):
    """Points of the ASSUMED region: every coordinate in (lo, hi], x > 0.

    `lo` is 0 for these queries, so the assume `x > 0` makes the region the
    HALF-OPEN (0, hi]; the sampled grid deliberately includes both a value
    adjacent to the excluded 0 and the closed upper endpoint.
    """
    base = [Fraction(hi) * Fraction(k, steps) for k in range(1, steps + 1)]
    base = [b for b in base if b > 0]
    base.append(Fraction(1, 10 ** 12))  # right up against the excluded zero
    if n == 0:
        return [[b] for b in base]
    return [list(t) for t in product(base, repeat=n)]


# --- the 0.3.0 census additions, in the same shape -------------------------
#
# Each build routes the divisor through ONE new rule so that a wrong rule
# shows up as a certified var whose exact value has the other sign. The
# INDEXING members of the routing class are deliberately not here; see
# `_exact_eval`'s scope paragraph for why and for where they are checked.


def _build_sub_opposite(nxt, eqns, x):
    """`x - (-x)` = 2x. The OPPOSITE-sign arm of the new `sub` rule; the
    same-sign arm has its own (still-refusing) test above."""
    nx = nxt(x.aval)
    eqns.append(eqn("neg", [x], nx))
    d = nxt(x.aval)
    eqns.append(eqn("sub", [x, nx], d))
    return d


def _build_sqrt_of_square(nxt, eqns, x):
    """`sqrt(x*x)`. Squared first so the exact evaluator can take the root
    without leaving the rationals."""
    sq = nxt(x.aval)
    eqns.append(eqn("mul", [x, x], sq))
    r = nxt(x.aval)
    eqns.append(eqn("sqrt", [sq], r))
    return r


def _build_max_one_sided(nxt, eqns, x):
    """`max(x, -5.0)`. ONE certified operand — the asymmetry. If the rule
    took `min`'s arm it would certify -1 here and the value is x > 0."""
    d = nxt(x.aval)
    eqns.append(eqn("max", [x, lit(-5.0)], d))
    return d


def _build_min_both_sided(nxt, eqns, x):
    """`min(x, 5.0)`, both operands certified +1."""
    d = nxt(x.aval)
    eqns.append(eqn("min", [x, lit(5.0)], d))
    return d


def _build_select_both_cases(nxt, eqns, x):
    """`select_n(x > 1, x, 2x)`. Both CASES certified; the selector is a
    bool and carries nothing, which is the operand split under test."""
    pred = nxt(BOOL)
    eqns.append(eqn("gt", [x, lit(1.0)], pred))
    two = nxt(x.aval)
    eqns.append(eqn("mul", [lit(2.0), x], two))
    d = nxt(x.aval)
    eqns.append(eqn("select_n", [pred, x, two], d))
    return d


def _build_routing_chain(nxt, eqns, x):
    """`sum(reshape(slice(concat(x*x, x*x))))` — the routing agreement rule
    through four members at once, on a rank-1 operand."""
    n = x.aval.shape[0]
    sq = nxt(x.aval)
    eqns.append(eqn("mul", [x, x], sq))
    cat_aval = ir.Aval(kind="ShapedArray", shape=(2 * n,), dtype="float64")
    cat = nxt(cat_aval)
    eqns.append(eqn("concatenate", [sq, sq], cat, [("dimension", 0)]))
    sl_aval = ir.Aval(kind="ShapedArray", shape=(n,), dtype="float64")
    sl = nxt(sl_aval)
    eqns.append(eqn(
        "slice", [cat], sl,
        [("start_indices", (1,)), ("limit_indices", (1 + n,)),
         ("strides", None)],
    ))
    rs = nxt(sl_aval)
    eqns.append(eqn("reshape", [sl], rs,
                    [("new_sizes", (n,)), ("dimensions", None)]))
    s = nxt()
    eqns.append(eqn("reduce_sum", [rs], s, [("axes", (0,))]))
    return s


# name -> (build, declaration size, grid steps, THE RULES THIS CASE DRIVES).
#
# The fourth field is not decoration. Without it a case whose new rule
# minted NOTHING still passed: `x` itself is always certified, so `signs`
# is non-empty whatever the divisor did, and the check would have been
# green over a rule it never reached. The test asserts a certified var
# produced by each named primitive, which is what makes the extension
# DRIVEN rather than merely present.
SEMANTIC_CASES = {
    "0.5*sum(x*x)": (_build_half_sumsq, 4, 3, ("mul", "reduce_sum")),
    "mean(x*x)": (_build_mean_sq, 4, 3, ("div", "reduce_sum")),
    "2.0*x": (_build_two_x, 0, 40, ("mul",)),
    "x/2.0": (_build_x_half, 0, 40, ("div",)),
    "sum(x*x)": (_build_sumsq, 4, 3, ("reduce_sum",)),
    # the 0.3.0 census additions
    "x-(-x)": (_build_sub_opposite, 0, 40, ("sub", "neg")),
    "sqrt(x*x)": (_build_sqrt_of_square, 0, 40, ("sqrt",)),
    "max(x,-5)": (_build_max_one_sided, 0, 40, ("max",)),
    "min(x,5)": (_build_min_both_sided, 0, 40, ("min",)),
    "select(x>1,x,2x)": (_build_select_both_cases, 0, 40, ("select_n",)),
    "sum(slice(cat(x*x,x*x)))": (
        _build_routing_chain, 3, 4, ("concatenate", "slice", "reshape"),
    ),
}


@pytest.mark.parametrize("name", sorted(SEMANTIC_CASES))
def test_strict_sign_certificate_is_TRUE_at_every_assumed_point(name):
    """The certificate is a semantic claim; check it semantically.

    For every var the propagator recorded a strict sign for, evaluate the
    query in EXACT `Fraction` arithmetic at points of the assumed region
    and confirm every element really has that sign. Exact rationals, so no
    rounding can mask a violation, and no interval reasoning is reused —
    this is an independent witness, not a restatement of the propagator.
    """
    from stelling.propagate import _Propagator

    build, n, steps, driven = SEMANTIC_CASES[name]
    query, eqns = _coeff_query(build, n=n)
    p = _Propagator("constrain")
    p.run(query.jaxpr, list(query.consts), [])
    signs = dict(p.strict_sign)
    assert signs, f"{name}: nothing was certified, the check would be vacuous"
    for prim in driven:
        produced = {
            e.outvars[0].id for e in eqns
            if e.primitive == prim and e.outvars
        }
        assert produced, f"{name}: no {prim!r} equation in this query"
        assert produced & set(signs), (
            f"{name}: no {prim!r} output was certified, so this case does "
            f"not exercise the {prim!r} rule and would pass with that rule "
            f"deleted"
        )

    points = _assumed_points(n, 0.0, 2.0, steps)
    assert points, "no sample points"
    checks = 0
    for point in points:
        env = _exact_eval(eqns, point)
        for vid, sgn in signs.items():
            if vid not in env:  # a bool/predicate var carries no arithmetic
                continue
            for cell in env[vid]:
                assert (cell > 0) if sgn > 0 else (cell < 0), (
                    f"{name}: var {vid} was certified sign={sgn} but is "
                    f"{cell} at assumed point {point}"
                )
                checks += 1
    assert checks > 0, f"{name}: no var-point check ran"
    if os.environ.get("STELLING_FUZZ_REPORT"):
        print(f"FUZZREPORT {name} points={len(points)} checks={checks} "
              f"signed_vars={len(signs)}")


def test_the_semantic_check_catches_a_certificate_that_is_false():
    """POSITIVE CONTROL: the check above fails when the certificate lies.

    **THIS DOCSTRING READ "Adding `sub` to the rules is the known break".**
    That stopped being true at 0.3.0: `sub` IS in the rules now, under the
    OPPOSITE-sign condition (`a > 0`, `b < 0` gives `a - b > 0`), and the
    census records it. What this control installs is not "sub" but the
    SAME-sign version of it — the rule the 0.2.0 comment was right to
    refuse — so it is now the per-rule mutation for `sub`, in the same
    shape as the five in `MUTATIONS` below. The query is `Σx² − 8`, the
    original false VERIFIED, and both its operands are certified `+1`, so
    the shipped rule answers 0 and the mutant answers `+1` for a value that
    is <= 0 at points of the assumed region. If this control ever stops
    finding violations, the check above is no longer checking anything.
    """
    from stelling import propagate as pm
    from stelling.propagate import _Propagator

    counter = [0]

    def nxt(aval=F64):
        counter[0] += 1
        return var(counter[0], aval)

    xa = ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64")
    ba = ir.Aval(kind="ShapedArray", shape=(2,), dtype="bool")
    x = var(0, xa)
    eqns = [any_eqn_shaped(x, 0.0, 2.0, (2,))]
    pa, ao = nxt(ba), nxt(ba)
    eqns.append(eqn("gt", [x, lit(0.0)], pa))
    eqns.append(eqn("stelling_assume", [pa], ao))
    sq = nxt(xa)
    eqns.append(eqn("mul", [x, x], sq))
    s = nxt()
    eqns.append(eqn("reduce_sum", [sq], s, [("axes", (0,))]))
    d = nxt()
    eqns.append(eqn("sub", [s, lit(8.0)], d))
    out = nxt(BOOL)
    eqns.append(eqn("stelling_assert", [ao], out))
    query = close(eqns, [out])

    real_out = pm._Propagator._strict_sign_out

    def with_sub(self, e, params, ins):
        if e.primitive == "sub":
            sgn = [self.read_strict_sign(a) for a in e.invars]
            return sgn[0] if len(sgn) == 2 and sgn[0] == sgn[1] else 0
        return real_out(self, e, params, ins)

    monkey = pm._STRICT_SIGN_PRIMITIVES | {"sub"}
    old_set = pm._STRICT_SIGN_PRIMITIVES
    pm._STRICT_SIGN_PRIMITIVES = monkey
    pm._Propagator._strict_sign_out = with_sub
    try:
        p = _Propagator("constrain")
        p.run(query.jaxpr, list(query.consts), [])
        signs = dict(p.strict_sign)
        assert signs.get(d.id) == 1, (
            "the control did not even mint the false certificate"
        )
        failures = cells = 0
        for point in _assumed_points(2, 0.0, 2.0, 4):
            env = _exact_eval(eqns, point)
            for cell in env[d.id]:
                cells += 1
                if not cell > 0:
                    failures += 1
        assert failures > 0, (
            "the positive control found NO violation — the semantic check "
            "is not actually checking the certificate"
        )
        if os.environ.get("STELLING_FUZZ_REPORT"):
            print(f"FUZZREPORT control failures={failures} of {cells} cells")
    finally:
        pm._Propagator._strict_sign_out = real_out
        pm._STRICT_SIGN_PRIMITIVES = old_set


# --- per-rule mutation: every 0.3.0 rule's WRONG version, caught -------------
#
# The control above is `sub`'s. These five are the rest of the algebraic
# additions. Each installs a deliberately wrong version of ONE rule, on a
# query built so that the wrong version mints a certificate the exact
# rationals contradict — and asserts BOTH halves: that the mutant really
# minted it (a control that does not fire is not a control) and that the
# semantic evaluator finds the violation.
#
# The INDEXING members of the routing class (`gather`, `scatter`,
# `scatter-add`, `dynamic_slice`, `dynamic_update_slice`) are NOT here.
# `_exact_eval` is rank-1 and C-order-flat by construction, so it cannot
# evaluate their row geometry; their mutations are driven end-to-end
# against executed jax in `tests/test_strict_sign_census.py`.


def _bare_query(build, *, lo=0.0, hi=2.0, n=0):
    """`assume(x > 0)`; `assert_(f(x) > 0)`, with NO division.

    The mutation controls do not need `div` — they need a certified var and
    an exact value for it — and a divisor that can be zero would make the
    evaluator raise before it could report the violation."""
    counter = [0]

    def nxt(a=F64):
        counter[0] += 1
        return var(counter[0], a)

    xa = ir.Aval(kind="ShapedArray", shape=(n,), dtype="float64") if n else F64
    ba = ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool") if n else BOOL
    x = var(0, xa)
    eqns = [any_eqn_shaped(x, lo, hi, (n,) if n else ())]
    pa, ao = nxt(ba), nxt(ba)
    eqns.append(eqn("gt", [x, lit(0.0)], pa))
    eqns.append(eqn("stelling_assume", [pa], ao))
    d = build(nxt, eqns, x)
    out = nxt(BOOL)
    eqns.append(eqn("stelling_assert", [ao], out))
    return close(eqns, [out]), eqns, d


def _build_sqrt_of_zero(nxt, eqns, x):
    """`sqrt(0.0 * x)` — exactly 0, and `sqrt` is the primitive under test."""
    z = nxt(x.aval)
    eqns.append(eqn("mul", [lit(0.0), x], z))
    r = nxt(x.aval)
    eqns.append(eqn("sqrt", [z], r))
    return r


def _build_max_neg_x(nxt, eqns, x):
    """`max(-x, x)` = x > 0. A `-1` and a `+1` operand, which is exactly
    where `max` and `min` disagree."""
    nx = nxt(x.aval)
    eqns.append(eqn("neg", [x], nx))
    m = nxt(x.aval)
    eqns.append(eqn("max", [nx, x], m))
    return m


def _build_min_neg_x(nxt, eqns, x):
    """`min(-x, x)` = -x < 0. The mirror."""
    nx = nxt(x.aval)
    eqns.append(eqn("neg", [x], nx))
    m = nxt(x.aval)
    eqns.append(eqn("min", [nx, x], m))
    return m


def _build_select_x_or_neg(nxt, eqns, x):
    """`select_n(x > 1, x, -x)` — the two cases DISAGREE, so the shipped
    rule mints nothing and a rule reading only the first case mints +1."""
    pred = nxt(BOOL if x.aval.shape == () else
               ir.Aval(kind="ShapedArray", shape=x.aval.shape, dtype="bool"))
    eqns.append(eqn("gt", [x, lit(1.0)], pred))
    nx = nxt(x.aval)
    eqns.append(eqn("neg", [x], nx))
    d = nxt(x.aval)
    eqns.append(eqn("select_n", [pred, x, nx], d))
    return d


def _build_cat_x_and_neg(nxt, eqns, x):
    """`concatenate(x, -x)` — operands disagree, so agreement mints nothing
    and an ANY-instead-of-ALL rule mints the first operand's sign."""
    n = x.aval.shape[0]
    nx = nxt(x.aval)
    eqns.append(eqn("neg", [x], nx))
    cat = nxt(ir.Aval(kind="ShapedArray", shape=(2 * n,), dtype="float64"))
    eqns.append(eqn("concatenate", [x, nx], cat, [("dimension", 0)]))
    return cat


# prim -> (wrong rule, build, declaration size, grid steps, what it is)
MUTATIONS = {
    "sqrt": (
        lambda sgn, e, params, ins: 1,
        _build_sqrt_of_zero, 0, 20,
        "sqrt certifies +1 without requiring a certified-positive operand; "
        "sqrt(0) is 0",
    ),
    "max": (
        lambda sgn, e, params, ins: (
            -1 if -1 in sgn else (1 if sgn == [1, 1] else 0)
        ),
        _build_max_neg_x, 0, 20,
        "max takes MIN's arm: one certified-negative operand certifies the "
        "max, and max(-x, x) is x > 0",
    ),
    "min": (
        lambda sgn, e, params, ins: (
            1 if 1 in sgn else (-1 if sgn == [-1, -1] else 0)
        ),
        _build_min_neg_x, 0, 20,
        "min takes MAX's arm: one certified-positive operand certifies the "
        "min, and min(-x, x) is -x < 0",
    ),
    "select_n": (
        lambda sgn, e, params, ins: sgn[1] if len(sgn) > 1 else 0,
        _build_select_x_or_neg, 0, 20,
        "select_n reads only the FIRST case instead of requiring every case "
        "to agree",
    ),
    "concatenate": (
        lambda sgn, e, params, ins: next((v for v in sgn if v), 0),
        _build_cat_x_and_neg, 3, 6,
        "the routing rule takes ANY nonzero operand sign instead of "
        "requiring all of them to agree",
    ),
}


def _run_with_mutant(prim, rule, build, n):
    """Install `rule` for `prim` only, run the query, return
    (certified table, eqns, divisor var)."""
    from stelling import propagate as pm

    real = pm._Propagator._strict_sign_out

    def mutated(self, e, params, ins):
        if e.primitive == prim:
            return rule([self.read_strict_sign(a) for a in e.invars],
                        e, params, ins)
        return real(self, e, params, ins)

    pm._Propagator._strict_sign_out = mutated
    try:
        query, eqns, d = _bare_query(build, n=n)
        p = pm._Propagator("constrain")
        p.run(query.jaxpr, list(query.consts), [])
        return dict(p.strict_sign), eqns, d
    finally:
        pm._Propagator._strict_sign_out = real


@pytest.mark.parametrize("prim", sorted(MUTATIONS))
def test_a_wrong_version_of_each_new_rule_is_CAUGHT(prim):
    """PER-RULE MUTATION. A rule whose wrong version passes the tests is a
    rule with no test."""
    rule, build, n, steps, why = MUTATIONS[prim]
    signs, eqns, d = _run_with_mutant(prim, rule, build, n)
    assert signs.get(d.id), (
        f"{prim}: the mutant ({why}) did not even mint the false "
        f"certificate, so this control demonstrates nothing"
    )
    sgn = signs[d.id]
    failures = cells = 0
    for point in _assumed_points(n, 0.0, 2.0, steps):
        env = _exact_eval(eqns, point)
        for cell in env[d.id]:
            cells += 1
            if not ((cell > 0) if sgn > 0 else (cell < 0)):
                failures += 1
    assert failures > 0, (
        f"{prim}: the mutant ({why}) certified sign={sgn} and the exact "
        f"evaluator found NO violating point in {cells} cells — the "
        f"semantic check would not have caught this rule"
    )
    if os.environ.get("STELLING_FUZZ_REPORT"):
        print(f"FUZZREPORT mutation {prim} failures={failures} of {cells}")


@pytest.mark.parametrize("prim", sorted(MUTATIONS))
def test_the_SHIPPED_rule_answers_DIFFERENTLY_and_TRULY_on_that_query(prim):
    """The other half of the battery, and it is two assertions.

    **A mutation is only a mutation where the two rules DISAGREE**, so the
    shipped rule's answer on the mutant's own query must differ from the
    mutant's — otherwise the control above is demonstrating a property of
    the query rather than of the rule.

    And whatever the shipped rule does answer must be TRUE at every point,
    by the same exact evaluator. Note this is NOT always "answers 0": on
    `max(-x, x)` the shipped rule certifies `+1` and is right to (the value
    IS `x > 0`), and on `min(-x, x)` it certifies `-1`. An earlier draft of
    this test asserted "the shipped rule mints nothing on these queries"
    and FAILED on exactly those two — the asymmetric rule is stronger than
    a decline, which is the whole reason it was written.
    """
    from stelling import propagate as pm

    rule, build, n, steps, why = MUTATIONS[prim]
    mutant_signs, _me, _md = _run_with_mutant(prim, rule, build, n)
    query, eqns, d = _bare_query(build, n=n)
    p = pm._Propagator("constrain")
    p.run(query.jaxpr, list(query.consts), [])
    shipped = p.strict_sign.get(d.id, 0)
    assert shipped != mutant_signs.get(d.id, 0), (
        f"{prim}: the shipped rule and the mutant ({why}) agree on this "
        f"query, so the control above tests the query and not the rule"
    )
    if not shipped:
        return
    checks = 0
    for point in _assumed_points(n, 0.0, 2.0, steps):
        env = _exact_eval(eqns, point)
        for cell in env[d.id]:
            checks += 1
            assert ((cell > 0) if shipped > 0 else (cell < 0)), (
                f"{prim}: the SHIPPED rule certified sign={shipped} but the "
                f"value is {cell} at assumed point {point}"
            )
    assert checks, f"{prim}: no point was checked"


# --- the disclosure must name the wrapper people actually write -------------


def _changelog_text():
    """CHANGELOG.md with runs of whitespace collapsed.

    The file is hard-wrapped, so a claim to be matched must be matched
    against the unwrapped text or the test is really testing line breaks.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    raw = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    return " ".join(raw.split())


_SUBJAXPR_DISCLOSURE_OPENS = (
    "- **The certificate does not cross a sub-jaxpr boundary"
)


def _subjaxpr_disclosure():
    """THE PARAGRAPH, not the file.

    Audit 0.2.0 B8a, item 7 (a B5 follow-up scheduled there). The check
    below used to search the WHOLE of CHANGELOG.md for each member's name,
    and `jit` and `remat2` are named all over a changelog for a jax tool —
    11 and 1 times respectively outside this bullet — so for those two the
    assertion passed on any file that mentioned them ANYWHERE. A gate that
    cannot fail is not a gate, and this one guards the sentence a user
    reads to learn that their `jit` silently costs them the certificate.

    IT WAS NOT BLIND TO EVERYTHING, and an earlier version of this
    docstring said it was. Driven on `aabb58d`: deleting this bullet
    outright REDDENED the whole-file check, because `custom_jvp_call` and
    `custom_vjp_call` occur 0 times elsewhere in the file. What the
    whole-file scope could not see was a bullet that still existed with a
    member DROPPED from it — and for `jit`, the member the disclosure
    exists to name, that is exactly the edit a rewrite makes.

    The slice is the markdown BULLET: from its opening marker to the next
    top-level `- ` bullet. Both ends are asserted, so a rewrite that moves
    the disclosure reddens here rather than passing on the surrounding
    prose. The membership check narrows further still, to the ENUMERATION
    inside it — see :func:`_transparent_enumeration` for why the bullet
    alone is not enough either.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    raw = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    start = raw.find(_SUBJAXPR_DISCLOSURE_OPENS)
    assert start != -1, (
        f"the sub-jaxpr disclosure bullet is gone from CHANGELOG.md "
        f"(looked for {_SUBJAXPR_DISCLOSURE_OPENS!r})"
    )
    end = raw.find("\n- ", start + len(_SUBJAXPR_DISCLOSURE_OPENS))
    assert end != -1, "the disclosure bullet does not end at another bullet"
    return " ".join(raw[start:end].split())


_ENUMERATION_OPENS = "`stelling.coverage.DEFAULT_TRANSPARENT` = "


def _transparent_enumeration():
    """THE ENUMERATION, not the bullet — audit 0.2.0 B8a FIXUP, item 3.

    Scoping to the bullet did not achieve item 7's own stated purpose.
    DRIVEN on `8772ced` and on `aabb58d` alike, by removing one member
    from the `DEFAULT_TRANSPARENT = …` list and asking the bullet-scoped
    check:

        remove `custom_jvp_call`  -> REDDENS
        remove `custom_vjp_call`  -> REDDENS
        remove `remat2`           -> REDDENS
        remove `jit`              -> PASSES

    `jit` is named 5 times in this bullet and only ONE of them is the
    enumeration, so dropping it from the list leaves 4 occurrences in the
    surrounding prose and the check finds one of those instead. The member
    the disclosure exists to name was the one member the check could not
    hold — the same hole one scope in, and the stated purpose ("blind for
    exactly the member the disclosure exists to name") was still not met.

    The enumeration is the span between `DEFAULT_TRANSPARENT` = and the
    em-dash that ends the list. Every member reddens on removal from it.
    """
    paragraph = _subjaxpr_disclosure()
    start = paragraph.find(_ENUMERATION_OPENS)
    assert start != -1, (
        f"the disclosure bullet no longer enumerates the frozenset "
        f"(looked for {_ENUMERATION_OPENS!r})"
    )
    start += len(_ENUMERATION_OPENS)
    end = paragraph.find("—", start)
    assert end != -1, "the enumeration does not end at an em-dash"
    return paragraph[start:end].strip()


def test_the_disclosure_names_EVERY_transparent_wrapper():
    """REDDENS ON REVERT of the CHANGELOG list, and on drift in either file.

    The published disclosure said "a transparent wrapper (`remat`,
    `custom_jvp`)" while `DEFAULT_TRANSPARENT` also contains `jit` — the
    one member essentially every jax user writes, and the only one most
    will ever hit. Naming the two rare members and omitting the universal
    one understates the limitation's cost to the reader who is paying it.
    Pinned to the code so the next member added to the frozenset cannot
    quietly fall out of the prose (B5 follow-up audit).

    SCOPED TO THE ENUMERATION, not to the paragraph and not to the file —
    see :func:`_subjaxpr_disclosure` and :func:`_transparent_enumeration`
    for what each wider scope could not detect. The bullet scope was blind
    for `jit`, which is the member this test exists for.

    The correspondence is checked in BOTH directions: a member missing
    from the prose understates the limitation, and a name in the prose
    that is no longer in the frozenset tells the reader they are paying a
    cost they are not.
    """
    enumeration = _transparent_enumeration()
    for member in sorted(DEFAULT_TRANSPARENT):
        assert f"`{member}`" in enumeration, (
            f"DEFAULT_TRANSPARENT member {member!r} is not named in the "
            f"`DEFAULT_TRANSPARENT = ...` enumeration of CHANGELOG.md's "
            f"sub-jaxpr disclosure; the frozenset is "
            f"{sorted(DEFAULT_TRANSPARENT)} and the enumeration reads "
            f"{enumeration!r}"
        )
    named = {t.strip().strip("`") for t in enumeration.split(",")}
    assert named == set(DEFAULT_TRANSPARENT), (
        f"the enumeration and the frozenset disagree: prose names "
        f"{sorted(named)}, code has {sorted(DEFAULT_TRANSPARENT)}"
    )
    assert (
        "`jit` is the one that matters in practice" in _subjaxpr_disclosure()
    ), "the disclosure no longer says which member the reader will hit"


def test_the_disclosure_scope_can_actually_fail():
    """POSITIVE CONTROL for the scoping, and a measurement of what each
    wider scope could and could not see.

    Occurrences of ``\u0060{member}\u0060`` on this tree, MEASURED:

        member             whole file   bullet   enumeration
        `jit`                      16        5             1
        `remat2`                    2        1             1
        `custom_jvp_call`           1        1             1
        `custom_vjp_call`           1        1             1

    Two edits, and the check must survive neither:

    * DELETE THE BULLET. The whole-file scope reddens — on the two
      `custom_*_call`s, which occur nowhere else — so the claim that it
      "passed on a file from which this disclosure had been deleted
      outright" was false. It stays blind for `jit` and `remat2`, which
      is a real hole but a different one.
    * DROP ONE MEMBER FROM THE ENUMERATION, the bullet still in place.
      This is what a rewrite actually does, and the BULLET scope passes
      for `jit`: four other occurrences of it survive inside the same
      bullet. Only the enumeration scope reddens for every member.
    """
    paragraph = _subjaxpr_disclosure()
    enumeration = _transparent_enumeration()
    whole_file = _changelog_text()
    assert enumeration in paragraph and paragraph in whole_file
    assert len(paragraph) < len(whole_file) / 10, (
        "the 'paragraph' is most of the file, so scoping bought nothing"
    )

    # -- edit 1: the whole bullet deleted
    without = whole_file.replace(paragraph, "")
    survives = {m for m in DEFAULT_TRANSPARENT if f"`{m}`" in without}
    assert "jit" in survives, (
        "the control rests on `jit` being named elsewhere; if it is not, "
        "the whole-file check was sound for the member that matters and "
        "this scoping needs a different argument"
    )
    assert survives == {"jit", "remat2"}, sorted(survives)
    # ... so the whole-file check DID redden on this edit, via the other two
    assert not {"custom_jvp_call", "custom_vjp_call"} & survives
    # ... and the scoped checks cannot pass on that text: the anchor is gone
    assert _SUBJAXPR_DISCLOSURE_OPENS not in without

    # -- edit 2: one member dropped from the enumeration, bullet intact
    blind_for_the_bullet = set()
    for member in sorted(DEFAULT_TRANSPARENT):
        token = f"`{member}`"
        assert token in enumeration, member
        cut = enumeration.replace(token + ", ", "", 1).replace(
            ", " + token, "", 1
        )
        assert token not in cut, member
        # the ENUMERATION scope reddens for every member — that is the fix
        if token in paragraph.replace(enumeration, cut):
            blind_for_the_bullet.add(member)
    assert blind_for_the_bullet == {"jit"}, (
        f"the bullet scope is blind for {sorted(blind_for_the_bullet)}; the "
        f"item this test came from exists because it was blind for `jit`"
    )


def _jit_wrapped_sumsq_query(*, assume_inside):
    """`assume(x>0)`; `1 / jit(lambda v: sum(v*v))(x) > 0`, hand-built.

    `assume_inside` moves the assume into the wrapper body, which is the
    other direction the certificate could have crossed.
    """
    xa = ir.Aval(kind="ShapedArray", shape=(4,), dtype="float64")
    ba = ir.Aval(kind="ShapedArray", shape=(4,), dtype="bool")
    inner_x = var(100, xa)
    inner_eqns = []
    if assume_inside:
        ip, io = var(101, ba), var(102, ba)
        inner_eqns += [
            eqn("gt", [inner_x, lit(0.0)], ip),
            eqn("stelling_assume", [ip], io),
        ]
    inner_sq, inner_s = var(103, xa), var(104, F64)
    inner_eqns += [
        eqn("mul", [inner_x, inner_x], inner_sq),
        eqn("reduce_sum", [inner_sq], inner_s, [("axes", (0,))]),
    ]
    body = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(inner_x,), outvars=(inner_s,),
            eqns=tuple(inner_eqns),
        )
    )

    x = var(0, xa)
    eqns = [any_eqn_shaped(x, 0.0, 2.0, (4,))]
    if not assume_inside:
        pa, ao = var(1, ba), var(2, ba)
        eqns += [
            eqn("gt", [x, lit(0.0)], pa),
            eqn("stelling_assume", [pa], ao),
        ]
    s, q, pred, out = var(3, F64), var(4, F64), var(5, BOOL), var(6, BOOL)
    eqns += [
        eqn("jit", [x], s, [("jaxpr", body)]),
        eqn("div", [lit(1.0), s], q),
        eqn("gt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


@pytest.mark.parametrize("assume_inside", [False, True])
@pytest.mark.parametrize("boundary", ["opaque", "transparent"])
def test_the_certificate_crosses_a_jit_boundary_IFF_THE_DIAL_SAYS_SO(
    boundary, assume_inside
):
    """**THIS TEST WAS CALLED
    ``test_the_certificate_does_not_cross_jit_in_either_direction`` AND
    ITS CLAIM WAS UNCONDITIONAL.** That test is now
    ``::test_the_certificate_crosses_a_jit_boundary_IFF_THE_DIAL_SAYS_SO``,
    which is this one. Its docstring read:

        The measurement the disclosure now reports, pinned.

        Not a bug — a fresh table per sub-jaxpr is what stops a
        branch-local assume licensing anything outside its branch. It is
        a COST, and the point of the finding is that the cost was
        disclosed under the names of two wrappers nobody writes.

    Every word of that is still true, and it is now true of ONE POSITION
    of a dial rather than of the analysis. The old body is the
    ``boundary="opaque"`` half below, unchanged in what it asserts — and
    that half is the regression guard that the DEFAULT did not move,
    which is the acceptance criterion the boundary dial was built to.
    The half that is new is ``"transparent"``.

    The transparent half does not assert ``discharged`` as a literal: it
    asserts EQUALITY WITH THE UNWRAPPED CONTROL
    (``::test_the_uncrossed_jit_query_is_green_without_the_wrapper``,
    whose query is rebuilt here), because the wrapped and unwrapped
    queries are the same arithmetic and a dial that made them agree for
    some other reason would not be worth having. If the control ever
    stops being green this half stops asserting a verdict and starts
    asserting an agreement, which is the claim that survives.
    """
    query = _jit_wrapped_sumsq_query(assume_inside=assume_inside)
    p = propagate(query, semantics="real", boundary=boundary)
    unwrapped, _ = _coeff_query(_build_sumsq, n=4)
    control = propagate(unwrapped, semantics="real", boundary=boundary)
    if boundary == "opaque":
        assert p.obligations[0].status != "discharged", (
            "the certificate crossed a jit boundary UNDER THE DEFAULT — "
            "the default has moved, which is the one thing the boundary "
            "dial may not do"
        )
        assert any(
            "REACHES zero at a boundary" in str(n) for n in p.notes
        ) or (
            "REACHES zero at a boundary" in str(p.obligations[0].detail)
        ), (p.notes, p.obligations[0].detail)
        assert p.boundary_crossings == 0, p.boundary_crossings
        assert p.boundary == "opaque"
        return
    assert p.obligations[0].status == control.obligations[0].status, (
        f"the jit-wrapped query is {p.obligations[0].status!r} while the "
        f"same arithmetic without the wrapper is "
        f"{control.obligations[0].status!r}; under boundary='transparent' "
        f"a boundary that carries the certificate must make the two agree "
        f"— {p.obligations[0].detail}"
    )
    assert p.obligations[0].status == "discharged", (
        f"the CONTROL is not green either, so this parametrisation is "
        f"asserting agreement between two UNKNOWNs: "
        f"{control.obligations[0].detail}"
    )
    assert p.boundary_crossings > 0, (
        "the verdict moved but nothing was recorded as having crossed"
    )


def test_the_uncrossed_jit_query_is_green_without_the_wrapper():
    """The control: identical arithmetic, no `jit`, VERIFIED.

    Without this the test above would pass for any reason at all.
    """
    query, _ = _coeff_query(_build_sumsq, n=4)
    p = propagate(query, semantics="real")
    assert p.obligations[0].status == "discharged"


def test_the_literal_sign_is_REAL_MODE_ONLY():
    """The invariant that makes reading the raw decoded box correct.

    `_literal_strict_sign` reads the literal's UN-HAZED value. Under ieee
    that box would be lying: DAZ flushes a literal like every other value
    (`_Propagator.read` hazes it), so a tiny positive literal's runtime
    value IS zero on a flush-to-zero target — S10's own lesson. Reading
    the raw box is only correct because neither call path can run under
    ieee: `_strict_sign_out` is short-circuited by `0 if ieee else`, and
    `div`'s `in_signs` argument is passed from the `elif` arm that the
    `if ieee` arm precedes. Asserted here rather than left to the reader
    to re-derive from two distant call sites.
    """
    query, _ = _coeff_query(_build_half_sumsq, n=4)
    from stelling.propagate import _Propagator

    real = _Propagator("constrain")
    real.run(query.jaxpr, list(query.consts), [])
    assert real.strict_sign, "real mode should certify something here"

    ieee = _Propagator("constrain")
    ieee.semantics = "ieee"
    ieee.run(query.jaxpr, list(query.consts), [])
    assert not ieee.strict_sign, (
        f"ieee mode wrote the strict-sign table: {ieee.strict_sign}. "
        f"Nothing may write or read it under a flush-to-zero semantics."
    )


# --- the same fact for a CONSTVAR, which is how array constants arrive ------
#
# A scalar constant traces to a Literal; an ARRAY constant traces to a
# CONSTVAR, which is a Var and so reads the assume-written table, where it
# is absent. Measured: `jnp.array([1.,2.,3.,4.]) * x` dropped the chain
# while the scalar `2.0 * x` kept it. The decline message promises the
# chain survives "nonzero finite constants", and a message that is true of
# scalars and false of arrays is the same claim-divergence class this
# finding is about — so the certificate is written for a constvar too, from the same
# `_box_strict_sign`, because a constvar's box IS its value.


def _const_coeff_query(values, *, assert_gt=True):
    """`assume(x > 0)`; `assert_(1 / sum(W * x * x) <cmp> 0)`, W a constvar."""
    n = len(values)
    xa = ir.Aval(kind="ShapedArray", shape=(n,), dtype="float64")
    ba = ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool")
    w = var(90, xa)
    x = var(0, xa)
    pa, ao = var(1, ba), var(2, ba)
    sq, wsq, s = var(3, xa), var(4, xa), var(5, F64)
    q, pred, out = var(6, F64), var(7, BOOL), var(8, BOOL)
    eqns = [
        any_eqn_shaped(x, 0.0, 2.0, (n,)),
        eqn("gt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
        eqn("mul", [x, x], sq),
        eqn("mul", [w, sq], wsq),
        eqn("reduce_sum", [wsq], s, [("axes", (0,))]),
        eqn("div", [lit(1.0), s], q),
        eqn("gt" if assert_gt else "lt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    closed = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(w,), invars=(), outvars=(out,), eqns=tuple(eqns)
        ),
        consts=(_f64_array(list(values)),),
    )
    return closed


@pytest.mark.parametrize(
    "values,assert_gt,want",
    [
        ([1.0, 2.0, 3.0, 4.0], True, "discharged"),
        ([-1.0, -2.0, -3.0, -4.0], False, "discharged"),
        ([1.0, 0.0, 3.0, 4.0], True, None),   # a zero element
        ([1.0, -2.0, 3.0, 4.0], True, None),  # mixed sign
        ([1.0, INF, 3.0, 4.0], True, None),   # non-finite
    ],
)
def test_an_array_CONSTVAR_carries_the_same_certificate(values, assert_gt, want):
    """REDDENS ON REVERT of the constvar half of the fix.

    The mixed-sign row is not a conservatism artifact: with
    `W = [1, -2, 3, 4]` the sum `Σ wᵢxᵢ²` really can be zero over the
    assumed region, so a certificate there would be FALSE. Whole-array
    quantification is what refuses it.
    """
    p = propagate(_const_coeff_query(values, assert_gt=assert_gt),
                  semantics="real")
    if want == "discharged":
        assert p.obligations[0].status == "discharged", (
            f"{values}: {p.obligations[0].status}"
        )
    else:
        assert p.obligations[0].status != "discharged", values


def test_a_PRE_BOXED_constvar_gets_no_certificate():
    """A const already handed over as an IntervalArray is a BOX, not a value.

    Its provenance is unknown — that branch is why `nan` is set True for it
    under ieee — so `[2, 3]` there means "somewhere in [2,3]", which is a
    range and not a constant, and it must not mint a sign.
    """
    xa = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
    w = var(90, xa)
    closed = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(w,), invars=(), outvars=(w,), eqns=()
        ),
        consts=(iv.IntervalArray(shape=(), los=(2.0,), his=(3.0,)),),
    )
    from stelling.propagate import _Propagator

    p = _Propagator("constrain")
    p.run(closed.jaxpr, list(closed.consts), [])
    assert p.strict_sign == {}, (
        f"a pre-boxed const minted {p.strict_sign}; that box is a RANGE of "
        f"unknown provenance, not a value"
    )
