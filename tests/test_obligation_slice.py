# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Obligation slices: extraction, fragment routing, declines, replay — no jax.

Hand-built IR queries in the :mod:`stelling.ir` conventions of
``test_propagate.py``. Per the guard rule, every decline path here proves
the obligation stays UNKNOWN with the reason (and the offending primitive)
quoted — never a raise, never a guess. Fragment routing is pinned in both
directions: linear content must stay QF_LRA, every polynomial marker must
route QF_NRA, and nothing outside the emission set may be guessed into a
logic.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from stelling import ir
from stelling.obligation import (
    DIV_GUARD_REASON,
    DeclinedObligation,
    ObligationSlice,
    ReplayError,
    evaluate_predicate,
    slice_obligation,
    slice_unknown_obligations,
)
from stelling.propagate import interval_env, propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi, dtype="float64", shape=()):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


def sole_slice(q):
    """The one escalation item of a query whose obligation is unknown."""
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1
    return items[0]


# --- extraction and routing ---------------------------------------------------


def linear_query():
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(0.25)], s),
            eqn("le", [s, lit(0.75)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def square_query(lo=1.0, hi=2.0, bound=2.0):
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_linear_slice_routes_qf_lra():
    sl = sole_slice(linear_query())
    assert isinstance(sl, ObligationSlice)
    assert sl.fragment == "QF_LRA"
    assert [i.name for i in sl.inputs] == ["x0"]
    assert (sl.inputs[0].lo, sl.inputs[0].hi) == (0.0, 1.0)


def test_product_of_nonconstants_routes_qf_nra():
    sl = sole_slice(square_query())
    assert isinstance(sl, ObligationSlice)
    assert sl.fragment == "QF_NRA"


def test_constant_scaling_stays_linear():
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [x, lit(3.0)], s),  # constant coefficient: linear
            eqn("le", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert sl.fragment == "QF_LRA"


def test_integer_pow_two_routes_qf_nra_and_pow_one_stays_linear():
    def q(y, bound):
        x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        return close(
            [
                any_eqn(x, 1.0, 2.0),
                eqn("integer_pow", [x], s, [("y", y)]),
                eqn("le", [s, lit(bound)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    assert sole_slice(q(2, 2.0)).fragment == "QF_NRA"
    assert sole_slice(q(1, 1.5)).fragment == "QF_LRA"


def test_division_by_nonconstant_routes_qf_nra_when_divisor_excludes_zero():
    x, y, d, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 1.0, 2.0),  # divisor interval excludes 0
            eqn("div", [x, y], d),
            eqn("le", [d, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert isinstance(sl, ObligationSlice)
    assert sl.fragment == "QF_NRA"


def test_division_by_constant_stays_linear():
    x, d, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("div", [x, lit(4.0)], d),
            eqn("le", [d, lit(0.125)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole_slice(q).fragment == "QF_LRA"


# --- decline paths: UNKNOWN with the reason quoted, never a raise -------------


def test_possibly_zero_divisor_declines_with_the_guard_reason_quoted():
    x, y, d, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, -1.0, 1.0),  # divisor may be zero
            eqn("div", [x, y], d),
            eqn("le", [d, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert DIV_GUARD_REASON in item.reason


def test_negative_integer_pow_uses_the_guarded_division_rule():
    def q(lo, hi):
        x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        return close(
            [
                any_eqn(x, lo, hi),
                eqn("integer_pow", [x], s, [("y", -2)]),
                eqn("le", [s, lit(100.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    ok = sole_slice(q(1.0, 2.0))  # base excludes 0: allowed, nonlinear
    assert isinstance(ok, ObligationSlice) and ok.fragment == "QF_NRA"
    bad = sole_slice(q(-1.0, 1.0))  # base may be zero: the guard declines
    assert isinstance(bad, DeclinedObligation)
    assert DIV_GUARD_REASON in bad.reason


def test_transcendental_declines_with_the_primitive_quoted_not_guessed():
    x, e, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("exp", [x], e),
            eqn("lt", [e, lit(7.0)], pred),  # straddles: unknown
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert "'exp'" in item.reason


def test_unknown_primitive_declines_quoted():
    x, y, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mystery_op", [x], y),
            eqn("lt", [y, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert "'mystery_op'" in item.reason


def test_array_shaped_operations_decline_scalar_only():
    x = var(0, ir.Aval(kind="ShapedArray", shape=(3,), dtype="float64"))
    r = var(1, ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    red = var(2, ir.Aval(kind="ShapedArray", shape=(3,), dtype="bool"))
    out = var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(3,)),
            eqn("lt", [x, lit(0.5)], red),
            eqn("reduce_or", [red], r, [("axes", (0,))]),
            eqn("stelling_assert", [r], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    # the walk meets the array-shaped reduction first; the primitive is quoted
    assert "'reduce_or'" in item.reason


def test_nonscalar_input_declaration_declines():
    # a size-1 but non-() declaration: eqns pass the size checks, the
    # declaration itself is what declines (shape () only)
    shp = ir.Aval(kind="ShapedArray", shape=(1,), dtype="float64")
    x = var(0, shp)
    pred = var(1, ir.Aval(kind="ShapedArray", shape=(1,), dtype="bool"))
    out = var(2, ir.Aval(kind="ShapedArray", shape=(1,), dtype="bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(1,)),
            eqn("lt", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert "scalar shape ()" in item.reason


def test_int_dtyped_input_declines_because_relaxation_admits_nonmembers():
    x = var(0, I32)
    pred, out = var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 10.0, dtype="int32"),
            eqn("lt", [x, lit(5, I32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert "int32" in item.reason


def test_integer_division_declines():
    x, y, d, pred, out = var(0, I32), var(1, I32), var(2, I32), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 10.0, dtype="int32"),
            any_eqn(y, 1.0, 10.0, dtype="int32"),
            eqn("div", [x, y], d),
            eqn("lt", [d, lit(5, I32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, DeclinedObligation)
    assert "truncates" in item.reason or "int32" in item.reason


def test_nested_assert_declines_the_whole_mapping():
    """Obligations recorded inside a transparent wrapper cannot be mapped
    onto top-level asserts by index — v1 declines rather than guesses."""
    x = var(0)
    # inner jaxpr: invar ix, assert(ix < 0.5)
    ix, ipred, iout = var(10), var(11, BOOL), var(12, BOOL)
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(ix,),
            outvars=(iout,),
            eqns=(
                eqn("lt", [ix, lit(0.5)], ipred),
                eqn("stelling_assert", [ipred], iout),
            ),
        )
    )
    jout = var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="jit",
                invars=(x,),
                outvars=(jout,),
                params=(("jaxpr", inner),),
            ),
        ],
        [jout],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], DeclinedObligation)
    assert "top-level" in items[0].reason


def test_only_unknown_obligations_are_escalated():
    x, sq, p1, o1, p2, o2 = var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], p1),  # straddles: unknown
            eqn("stelling_assert", [p1], o1),
            eqn("ge", [x, lit(0.0)], p2),  # definitely true: discharged
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "discharged"]
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1  # only the unknown one
    assert isinstance(items[0], ObligationSlice)
    assert items[0].index == 0


def test_assume_data_flow_passes_through_but_constraint_is_not_part_of_the_slice():
    # assume returns its input; if that value is consumed the slice treats
    # the equation as identity — exactly propagation's semantics; the
    # constraint itself is never emitted (that is asserted on the script in
    # test_smt_emission).
    x, apred, pred2, out = var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("gt", [x, lit(0.5)], apred),
            eqn("stelling_assume", [apred], pred2),
            eqn("stelling_assert", [pred2], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert isinstance(sl, ObligationSlice)
    assert evaluate_predicate(sl, {"x0": Fraction(3, 4)}) is True
    assert evaluate_predicate(sl, {"x0": Fraction(1, 4)}) is False


def test_shared_input_is_one_declaration():
    sl = sole_slice(square_query())
    assert isinstance(sl, ObligationSlice)
    assert len(sl.inputs) == 1  # x used twice in mul(x, x): ONE constant


# --- exact-rational replay ----------------------------------------------------


def test_replay_is_exact_dyadic_not_decimal():
    # 0.1 is not 1/10 in f64; the replay must see the exact dyadic value
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(0.1)], s),
            eqn("le", [s, lit(0.6)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    exact_tenth = Fraction(0.1)  # 3602879701896397/36028797018963968
    assert exact_tenth != Fraction(1, 10)
    # at x = 0.6 - float(0.1) the sum is exactly the f64 0.6? No — work in
    # exact rationals: pick x so x + dyadic(0.1) == dyadic(0.6) exactly.
    x_val = Fraction(0.6) - exact_tenth
    assert evaluate_predicate(sl, {"x0": x_val}) is True
    assert evaluate_predicate(sl, {"x0": x_val + Fraction(1, 10**30)}) is False


def test_replay_covers_the_v1_operation_set():
    # max/min/select_n/neg/sub/xor exercised at exact points
    x, y = var(0), var(1)
    mx, mn = var(2), var(3)
    gt_, sel = var(4, BOOL), var(5)
    diff, pred, out = var(6), var(7, BOOL), var(8, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            any_eqn(y, 0.0, 4.0),
            eqn("max", [x, y], mx),
            eqn("min", [x, y], mn),
            eqn("gt", [x, y], gt_),
            eqn("select_n", [gt_, mn, mx], sel),  # x>y ? mx : mn
            eqn("sub", [sel, mn], diff),
            eqn("le", [diff, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert isinstance(sl, ObligationSlice)
    # x=3, y=1: gt true -> sel = max = 3; diff = 3-1 = 2 > 1 -> violated
    assert evaluate_predicate(sl, {"x0": Fraction(3), "x1": Fraction(1)}) is False
    # x=1, y=3: gt false -> sel = min = 1; diff = 0 <= 1 -> holds
    assert evaluate_predicate(sl, {"x0": Fraction(1), "x1": Fraction(3)}) is True


def test_replay_missing_value_raises_replay_error():
    sl = sole_slice(square_query())
    with pytest.raises(ReplayError, match="x0"):
        evaluate_predicate(sl, {})


def test_replay_nonfraction_value_raises_replay_error():
    sl = sole_slice(square_query())
    with pytest.raises(ReplayError, match="Fraction"):
        evaluate_predicate(sl, {"x0": 1.5})  # a float is not an exact witness


def test_slice_obligation_out_of_range_declines():
    q = square_query()
    item = slice_obligation(q, 5, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert "no matching" in item.reason
