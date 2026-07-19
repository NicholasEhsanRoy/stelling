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


# --- audit-gate findings (any_pytree build): posture escapes, fixed ----------


def test_nan_literal_degrades_to_top_not_crash():
    # the ubiquitous NaN-sentinel pattern: a legal constant outside the ℝ
    # domain must bind ⊤ with a note, never kill the analysis
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="add",
                invars=(x, ir.Literal(val=math.nan, aval=F64)),
                outvars=(s,),
            ),
            ir.JaxprEqn(
                primitive="lt",
                invars=(s, ir.Literal(val=1.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("outside the domain" in n for n in p.notes)


def test_undecodable_dtype_const_degrades_to_top_not_crash():
    f16 = ir.Aval(kind="ShapedArray", shape=(), dtype="float16")
    arr = ir.Array(dtype="<f2", shape=(), data=b"\x00\x3c")  # f16 1.0
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="add",
                invars=(x, ir.Literal(val=arr, aval=f16)),
                outvars=(s,),
            ),
            ir.JaxprEqn(
                primitive="lt",
                invars=(s, ir.Literal(val=10.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("outside the domain" in n for n in p.notes)


def test_zero_size_structural_ops_run_without_phantom_elements():
    # _coords yielded a phantom coordinate for zero-size shapes; the
    # IndexError bypassed the decline channel (audit-gate finding 2)
    z = iv.from_values((0,), [])
    b = iv.broadcast_in_dim(z, (2, 0), (1,))
    assert b.shape == (2, 0) and b.los == ()
    s = iv.slice_(iv.from_values((3,), [1.0, 2.0, 3.0]), (3,), (3,), None)
    assert s.shape == (0,) and s.los == ()


def test_unhandled_transfer_form_degrades_to_top_not_crash():
    # a legal jax form the domain doesn't cover must DECLINE — ⊤ with the
    # reason noted — never kill the analysis (degrade-don't-crash: the
    # refusal is caught at the propagation layer). The audit's original
    # vehicle (scalar which, array cases) became a *registered* select_n
    # form in the pytree-probe round, so the vehicle here is now bitwise
    # integer `and`, which the bool-only logic transfer declines; the
    # property under test — IntervalError from a transfer becomes a noted
    # ⊤, never a crash — is unchanged.
    x, y, z, pred, out = (
        var(0, I32), var(1, I32), var(2, I32), var(3, BOOL), var(4, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 7.0, "int32"),
            any_eqn(y, 0.0, 7.0, "int32"),
            ir.JaxprEqn(primitive="and", invars=(x, y), outvars=(z,)),
            ir.JaxprEqn(
                primitive="le",
                invars=(z, ir.Literal(val=10.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(
                primitive="stelling_assert", invars=(pred,), outvars=(out,)
            ),
        ],
        (out,),
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"  # ⊤ decays the verdict soundly
    assert any("declined" in n for n in p.notes)
    assert p.coverage.unknown >= 1


# --- second audit, finding 4-B: float->int guard is strict at +2**(n-1) ------


def test_float_to_int_boundary_value_falls_to_top():
    # exactly 2**31 passed the old inclusive guard, but int32 cannot hold it
    # (jax clamps to 2**31-1; numpy wraps) — a false VERIFIED at the boundary
    x, y, pred, out = var(0), var(1, I32), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 2147483648.0, 2147483648.0),
            ir.JaxprEqn(
                primitive="convert_element_type",
                invars=(x,),
                outvars=(y,),
                params=(("new_dtype", "int32"),),
            ),
            ir.JaxprEqn(
                primitive="ge",
                invars=(y, ir.Literal(val=2147483648.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    assert propagate(q).obligations[0].status == "unknown"  # was: discharged


def test_float_to_int64_boundary_also_strict():
    # for int64 the float `bound - 1` rounds back to `bound`, so only a
    # strict upper check is sound at 2**63
    x, y, pred, out = var(0), var(1, I64), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 2.0**63, 2.0**63),
            ir.JaxprEqn(
                primitive="convert_element_type",
                invars=(x,),
                outvars=(y,),
                params=(("new_dtype", "int64"),),
            ),
            ir.JaxprEqn(
                primitive="ge",
                invars=(y, ir.Literal(val=0.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    assert propagate(q).obligations[0].status == "unknown"


# --- second audit, finding 3: select_n clamps out-of-range (unlike cond) -----


def test_select_n_negative_selector_clamps_to_first_case():
    cases = [iv.point(10.0), iv.point(20.0), iv.point(30.0)]
    # definite -1: jax's measured lax.select_n clamps to case 0 (NOT cond's
    # default-last); the old fallback picked 30.0 — a false VERIFIED
    r = iv.select_n(iv.from_bounds((), -1.0, -1.0), cases)
    assert (r.los[0], r.his[0]) == (10.0, 10.0)
    # entirely below range: still case 0
    r2 = iv.select_n(iv.from_bounds((), -2.0, -1.0), cases)
    assert (r2.los[0], r2.his[0]) == (10.0, 10.0)
    # straddling below and in-range: clamp target already included
    r3 = iv.select_n(iv.from_bounds((), -1.0, 0.0), cases)
    assert (r3.los[0], r3.his[0]) == (10.0, 10.0)
    # above range clamps to the last case
    r4 = iv.select_n(iv.from_bounds((), 5.0, 5.0), cases)
    assert (r4.los[0], r4.his[0]) == (30.0, 30.0)
    # the asymmetry with cond is real, measured, and deliberate: cond's
    # out-of-range convention (default-last) is tested in cond_query above


# --- second audit, findings 4-A and 1/2: the registered ℝ-semantics gap ------
#
# These two constructions are CORRECT under the `semantics="real"` dial
# (the stamp's own disclaimer: a predicate can hold in ℝ and fail in
# floats) and are pinned here as MARKERS of the gap: over ℝ, (x+x)*0 = 0 and
# any real is ≤ +∞, so both discharge; in IEEE both are NaN-possible and do
# not. THE DIAL EVENT HAPPENED: `propagate(..., semantics="ieee")` is the
# second dial position, and under it both shapes flip exactly as predicted —
# the conscious rewrites these markers demanded are the ieee companions
# `test_ieee_marker_overflow_times_zero_does_not_discharge` and
# `test_ieee_marker_top_output_leq_inf_does_not_discharge` in
# tests/test_ieee_semantics.py, which assert the flips. The real-mode tests
# below stay byte-identical and keep pinning the ℝ side of the dial. The
# ⊤-widening vacuity guard still fences the r ≤ ∞ shape out of any count
# (it discharges under ⊤, hence tautological).


def test_R_gap_marker_overflow_times_zero_discharges_in_R():
    x, s, z, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 1e308, 1.7e308),
            ir.JaxprEqn(primitive="add", invars=(x, x), outvars=(s,)),
            ir.JaxprEqn(
                primitive="mul",
                invars=(s, ir.Literal(val=0.0, aval=F64)),
                outvars=(z,),
            ),
            ir.JaxprEqn(
                primitive="lt",
                invars=(z, ir.Literal(val=1.0, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    # ℝ: (x+x)·0 = 0 < 1 — true. IEEE: inf·0 = NaN, NaN < 1 — false.
    # The ieee companion asserting the flip:
    # test_ieee_semantics.test_ieee_marker_overflow_times_zero_does_not_discharge
    assert propagate(q).obligations[0].status == "discharged"


def test_R_gap_marker_top_output_leq_inf_discharges_and_is_tautological():
    x, r, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(primitive="mystery_loop", invars=(x,), outvars=(r,)),
            ir.JaxprEqn(
                primitive="le",
                invars=(r, ir.Literal(val=math.inf, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,)),
        ],
        (out,),
    )
    # ⊤ = [−∞, ∞] contains every REAL; r ≤ +∞ is an ℝ-tautology, and it
    # discharges even though r fell to ⊤ — which also means it survives
    # ⊤-widening, so the vacuity guard voids any count that leaned on it.
    # The ieee companion asserting the flip (⊤ under ieee is maybe-NaN):
    # test_ieee_semantics.test_ieee_marker_top_output_leq_inf_does_not_discharge
    assert propagate(q).obligations[0].status == "discharged"
