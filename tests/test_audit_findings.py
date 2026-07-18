# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the 2026-07-18 adversarial soundness audit.

Each test is one of the audit's verified failure constructions
(design/soundness-audit.md), kept as the permanent falsifier for its fix.
Hand-built IR (no jax needed) except where the audit's construction was a
real trace.
"""

from __future__ import annotations

import math
import struct

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
I64 = ir.Aval(kind="ShapedArray", shape=(), dtype="int64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def any_eqn(out, lo, hi, dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=outvars, eqns=tuple(eqns))
    )


def branch(in_id, out_id, eqns):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(in_id),),
            outvars=(var(out_id),),
            eqns=tuple(eqns),
        )
    )


def cond_eqn(index_var, operand, outvars, branches):
    return ir.JaxprEqn(
        primitive="cond",
        invars=(index_var, operand),
        outvars=outvars,
        params=(("branches", tuple(branches)),),
    )


# --- audit finding 1: value-changing conversions must not pass through ------


def test_value_changing_conversion_falls_to_top():
    # int32 {5} -> bool would collapse to 1; passing 5 through produced a
    # false violated-over-set on `y <= 2`
    x, y, pred, out = var(0, I32), var(1, BOOL), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 5.0, 5.0, "int32"),
            ir.JaxprEqn(
                primitive="convert_element_type",
                invars=(x,),
                outvars=(y,),
                params=(("new_dtype", "bool"),),
            ),
            ir.JaxprEqn(
                primitive="lt",
                invars=(y, ir.Literal(val=2.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"  # ⊤, not a wrong definite
    assert any("convert_element_type" in n for n in p.notes)


def test_narrowing_float_conversion_falls_to_top():
    x, y, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.1, 0.1),
            ir.JaxprEqn(
                primitive="convert_element_type",
                invars=(x,),
                outvars=(y,),
                params=(("new_dtype", "float32"),),  # rounds 0.1 upward
            ),
            ir.JaxprEqn(
                primitive="le",
                invars=(y, ir.Literal(val=0.1, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    assert propagate(q).obligations[0].status == "unknown"  # was: discharged


def test_exact_conversions_still_pass_through():
    x, y, pred, out = var(0, BOOL), var(1, I32), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0, "bool"),
            ir.JaxprEqn(
                primitive="convert_element_type",
                invars=(x,),
                outvars=(y,),
                params=(("new_dtype", "int32"),),
            ),
            ir.JaxprEqn(
                primitive="le",
                invars=(y, ir.Literal(val=1.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    assert propagate(q).obligations[0].status == "discharged"


# --- audit finding 2: cond's out-of-range convention is default-LAST --------


def add_eqn(a, b_lit, out):
    return ir.JaxprEqn(
        primitive="add", invars=(a, ir.Literal(val=b_lit, aval=F64)), outvars=(out,)
    )


def cond_query(idx_lo, idx_hi):
    """branches: b0 = v+100, b1(last) = v+0; x in [1,2]; assert y >= 100."""
    idx, x, y, pred, out = var(0, I32), var(1), var(2), var(3, BOOL), var(4, BOOL)
    b0 = branch(10, 11, [add_eqn(var(10), 100.0, var(11))])
    b1 = branch(20, 21, [add_eqn(var(20), 0.0, var(21))])
    return close(
        [
            any_eqn(idx, idx_lo, idx_hi, "int32"),
            any_eqn(x, 1.0, 2.0),
            cond_eqn(idx, x, (y,), [b0, b1]),
            ir.JaxprEqn(
                primitive="ge",
                invars=(y, ir.Literal(val=100.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )


def test_cond_negative_straddle_includes_last_branch():
    # index in [-1, 0]: -1 runs the LAST branch (verified jax semantics), so
    # the join must include b1 and the assert cannot discharge
    assert propagate(cond_query(-1.0, 0.0)).obligations[0].status != "discharged"


def test_cond_definite_negative_takes_last_branch():
    # index == -1 -> last branch only (b1: y in [1,2]) -> definitely < 100
    assert propagate(cond_query(-1.0, -1.0)).obligations[0].status == "violated-over-set"


def test_cond_definite_in_range_still_exact():
    assert propagate(cond_query(0.0, 0.0)).obligations[0].status == "discharged"


def test_cond_infinite_index_joins_all_without_crashing():
    assert propagate(cond_query(-math.inf, math.inf)).obligations[0].status != "discharged"


# --- audit finding 4: untaken branches count as unreached -------------------


def test_untaken_branch_equations_are_unreached_not_invisible():
    idx, x, y, pred, out = var(0, I32), var(1), var(2), var(3, BOOL), var(4, BOOL)
    b0 = branch(10, 11, [add_eqn(var(10), 100.0, var(11))])
    b1 = branch(
        20,
        21,
        [ir.JaxprEqn(primitive="mystery_op", invars=(var(20),), outvars=(var(21),))],
    )
    q = close(
        [
            any_eqn(idx, 0.0, 0.0, "int32"),
            any_eqn(x, 1.0, 2.0),
            cond_eqn(idx, x, (y,), [b0, b1]),
            ir.JaxprEqn(
                primitive="ge",
                invars=(y, ir.Literal(val=100.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"  # b0 taken, exact
    assert p.coverage.unreached == 1  # b1's mystery_op is in the denominator
    assert p.coverage.fraction_known < 1.0


# --- audit finding 3: int64 array constants above 2**53 bracket -------------


def test_large_int64_array_constant_is_bracketed():
    big = 2**53 + 1
    arr = ir.Array(dtype="<i8", shape=(), data=struct.pack("<q", big))
    base = ir.Literal(val=2**53, aval=I64)
    d, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            ir.JaxprEqn(
                primitive="sub",
                invars=(ir.Literal(val=arr, aval=I64), base),
                outvars=(d,),
            ),
            ir.JaxprEqn(
                primitive="lt",
                invars=(d, ir.Literal(val=0.5, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    # true difference is 1, so `< 0.5` is false; the widened brackets must
    # at minimum refuse to discharge (was: false VERIFIED from point decode)
    assert propagate(q).obligations[0].status != "discharged"


# --- audit finding 5: ⊤ selectors degrade, never crash ----------------------


def test_select_n_top_selector_joins_all():
    x, y = iv.point(10.0), iv.point(20.0)
    j = iv.select_n(iv.top(()), [x, y])
    assert (j.los[0], j.his[0]) == (10.0, 20.0)


def test_huge_python_int_literal_saturates_instead_of_crashing():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="add",
                invars=(x, ir.Literal(val=2**2000, aval=F64)),
                outvars=(var(3),),
            ),
            ir.JaxprEqn(
                primitive="gt",
                invars=(var(3), ir.Literal(val=0.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    assert propagate(q).obligations[0].status == "discharged"  # > maxfloat > 0


# --- audit finding 6: branch scopes are isolated ----------------------------


def test_cross_branch_read_raises_instead_of_reading_stale_scope():
    idx, x, y, out = var(0, I32), var(1), var(2), var(4, BOOL)
    b0 = branch(30, 31, [add_eqn(var(30), 100.0, var(31))])
    # b1 reads var 31 — b0's internal — as a free variable
    b1 = branch(20, 21, [add_eqn(var(31), 0.0, var(21))])
    q = close(
        [
            any_eqn(idx, -math.inf, math.inf, "int32"),  # both branches run
            any_eqn(x, 1.0, 2.0),
            cond_eqn(idx, x, (y,), [b0, b1]),
            ir.JaxprEqn(
                primitive="ge",
                invars=(y, ir.Literal(val=0.0, aval=F64)),
                outvars=(var(3, BOOL),),
            ),
            ir.JaxprEqn(
                primitive="stelling_assert", invars=(var(3, BOOL),), outvars=(out,)
            ),
        ],
        (out,),
    )
    with pytest.raises(ir.TranscriptionError):
        propagate(q)


# --- audit finding 7: join/select_n refuse shape mismatches ------------------


def test_join_and_select_n_refuse_shape_mismatch():
    a = iv.point(1.0)
    b = iv.from_values((2,), [1.0, 2.0])
    with pytest.raises(iv.IntervalError):
        iv.join([a, b])
    with pytest.raises(iv.IntervalError):
        iv.select_n(iv.point(0.0), [a, b])
