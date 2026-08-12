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

import pytest

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

def test_assume_gt0_div_positive_ieee_f32():
    """assume(b > 0) + assert_(a/b > 0) in ieee float32: the bump
    excludes zero from b's interval, and boundary-aware division gives
    [positive, +inf], which is definitely > 0."""
    a = var(0, F32)
    b = var(1, F32)
    pred_assume = var(2, BOOL)
    assume_out = var(3, BOOL)
    q_out = var(4, F32)
    pred = var(5, BOOL)
    out = var(6, BOOL)
    query = close(
        [
            any_eqn(a, 1.0, 10.0, dtype="float32"),
            any_eqn(b, -10.0, 10.0, dtype="float32"),
            # assume(b > 0)
            eqn("gt", [b, lit(0.0, F32)], pred_assume),
            eqn("stelling_assume", [pred_assume], assume_out),
            # a / b
            eqn("div", [a, b], q_out),
            # assert a/b > 0
            eqn("gt", [q_out, lit(0.0, F32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
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
    narrowing_notes = [n for n in p.notes if "narrowed var 0" in n]
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
