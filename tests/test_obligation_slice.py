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


def two_unknown_obligations_query():
    """Two top-level asserts, both left `unknown` by propagation.

    Exists for the per-obligation containment half of
    `test_slice_unknown_obligations_CANNOT_RAISE_from_its_OWN_body`: with
    one obligation a net around the whole function and a net around one
    obligation are indistinguishable, which is precisely the distinction
    audit 0.2.0 M17 is about.
    """
    x = var(0)
    sq, p0, o0 = var(1), var(2, BOOL), var(3, BOOL)
    cu, p1, o1 = var(4), var(5, BOOL), var(6, BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], p0),
            eqn("stelling_assert", [p0], o0),
            eqn("mul", [sq, x], cu),
            eqn("le", [cu, lit(4.0)], p1),
            eqn("stelling_assert", [p1], o1),
        ],
        [o0, o1],
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
    # The bound is chosen to sit INSIDE the propagated interval on the
    # allowed branch: integer_pow now has an interval transfer (x ∈ [1, 2],
    # y = -2 propagates to exactly [0.25, 1.0]), so a slack bound like 100.0
    # would be discharged outright and never reach escalation at all. This
    # test is about the EMISSION guard, so it keeps the obligation unknown.
    def q(lo, hi, bound):
        x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        return close(
            [
                any_eqn(x, lo, hi),
                eqn("integer_pow", [x], s, [("y", -2)]),
                eqn("le", [s, lit(bound)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    ok = sole_slice(q(1.0, 2.0, 0.5))  # base excludes 0: allowed, nonlinear
    assert isinstance(ok, ObligationSlice) and ok.fragment == "QF_NRA"
    bad = sole_slice(q(-1.0, 1.0, 100.0))  # base may be zero: guard declines
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


def test_nonscalar_input_declaration_now_emits_per_element_names():
    # array-emission build: a non-() static declaration is one SMT constant
    # per element, named x{k}_{i} (flat C-order); the old scalar-shape-only
    # decline is retired. A (1,)-shaped declaration is an array of one
    # element and gets the element naming, not the legacy scalar x{k}.
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
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    assert [i.name for i in item.inputs] == ["x0_0"]
    assert item.inputs[0].shape == (1,) and item.inputs[0].element == 0
    assert (item.inputs[0].lo, item.inputs[0].hi) == (0.0, 1.0)


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


def test_an_index_past_the_START_declines_rather_than_raising():
    """AUDIT 0.2.0 S12, second half. A negative index WITHIN range has always
    been Python indexing from the end and still is; one PAST the start used to
    raise a raw `IndexError` out of a function documented never to raise on a
    legal query, and reached the VERIFIED bar's whole-query fallback through
    `verdict._bar_scope`'s outer `except` instead of through the decline
    channel. The range test is two-sided now."""
    q = square_query()  # exactly one top-level assert
    env = interval_env(q)
    assert not isinstance(slice_obligation(q, -1, env), DeclinedObligation)
    for index in (-2, -7):
        item = slice_obligation(q, index, env)
        assert isinstance(item, DeclinedObligation), index
        assert "no matching" in item.reason


def test_slice_obligation_CANNOT_RAISE_an_unhandled_exception(monkeypatch):
    """THE GUARD NET, driven by injection — the only way to drive it, and
    saying so is the point.

    `slice_obligation`'s contract is "never raises on legal queries", and its
    caller (`stelling.solvers.escalate`) catches `_Decline` and nothing else —
    worse, it iterates `slice_unknown_obligations` in the `for` HEADER, outside
    its own per-obligation `except Exception`, so anything escaping here takes
    every other obligation's verdict with it. Audit 0.2.0 S12 reached that
    through a `dot_general` whose contracted extents disagreed: the plan
    indexed off the end of the constant operand and raised `IndexError`.

    That route is closed at its root (the shared shape oracle), so NOTHING
    currently constructable reaches the net — which is exactly why it has to be
    driven by injection rather than by a query, and why a test that waited for
    a real exception would be a test that never runs. What is asserted is the
    posture: an unexpected exception becomes a DECLINE, quoted, naming the
    exception class and saying *internal error* in those words, so a stelling
    defect reads as a stelling defect and not as an undecided obligation.
    """
    import stelling.obligation as OB

    def boom(self, index, assert_eqn):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(OB._Slicer, "slice", boom)
    q = square_query()
    item = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item, DeclinedObligation), (
        "the exception escaped slice_obligation"
    )
    assert "internal error" in item.reason
    assert "IndexError" in item.reason
    assert "tuple index out of range" in item.reason
    # and it escapes the plural entry point no more than the singular one:
    # that is the call site whose exception took the whole query with it
    p = propagate(q)
    (plural,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(plural, DeclinedObligation)
    assert "internal error" in plural.reason


def test_slice_unknown_obligations_CANNOT_RAISE_from_its_OWN_body(monkeypatch):
    """THE NET AROUND THE ASSOCIATION CHECK — audit 0.2.0 B6/M17′, and this
    is a different net from the one above.

    The test above injects into `_Slicer.slice`, which sits INSIDE
    `slice_obligation`'s own `try`. So it cannot see an exception raised by
    `slice_unknown_obligations`'s own body — the position lookup, the
    one-to-one check, the `source_info` comparison — and the M17 fix put a
    fresh raise exactly there: `tuple(eqns[pos].source_info)` on IR carrying
    a non-tuple. `escalate` and `refine_propagation` both iterate this
    function in the `for` HEADER, outside their per-obligation nets, so that
    raise escaped the whole call.

    Injected here rather than driven by a query BECAUSE the source_info
    route is now closed at its root (`_frames` is total), and a net that
    only ever runs on one closed route is a net nobody has driven. What is
    asserted is the posture, per obligation: a DECLINE, quoted, naming the
    exception class, saying *internal error* in those words — and, crucially,
    the SIBLING obligation still gets its own answer. A net around the whole
    function would have failed that second assertion, and answering a
    per-obligation question with a whole-query outcome is the defect M17
    exists to have fixed.
    """
    import stelling.obligation as OB

    q = two_unknown_obligations_query()
    p = propagate(q)
    assert len([o for o in p.obligations if o.status == "unknown"]) == 2, (
        "the fixture no longer has two unknown obligations; the containment "
        "half of this test would measure nothing"
    )

    calls = {"n": 0}
    real = OB._frames

    def boom(v):
        # fail for the FIRST obligation only, inside the association check
        # and outside `slice_obligation` entirely
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("'int' object is not iterable")
        return real(v)

    monkeypatch.setattr(OB, "_frames", boom)
    items = OB.slice_unknown_obligations(q, p, interval_env(q))

    assert len(items) == 2, items
    first, second = items
    assert isinstance(first, DeclinedObligation), (
        "the exception escaped slice_unknown_obligations"
    )
    assert "internal error" in first.reason, first.reason
    assert "TypeError" in first.reason, first.reason
    assert "'int' object is not iterable" in first.reason, first.reason
    assert not isinstance(second, DeclinedObligation), (
        "the sibling obligation declined with it: the net is around the "
        "whole function rather than around one obligation"
    )


def test_a_non_tuple_source_info_declines_instead_of_raising():
    """The route that actually escaped, driven by IR rather than injection.

    `ir.ClosedJaxpr` is a public dataclass and `SOUNDNESS.md` names
    hand-built IR as in scope (`from_dict` coerces at its own door and
    nowhere else). With an `int` where the frames go, `4d793cf` raised
    `TypeError: 'int' object is not iterable` out of `escalate`, where
    `dee8bc2` returned a verdict for every obligation.

    The decline says the association could not be CHECKED — deliberately
    not that the two disagreed. "traced at 7 but records 7" is what the
    disagreement sentence would print here, and it reads as the tool
    contradicting itself.
    """
    q = square_query()
    eqns = tuple(
        ir.JaxprEqn(
            primitive=e.primitive,
            invars=e.invars,
            outvars=e.outvars,
            params=e.params,
            effects=e.effects,
            source_info=7,  # an int, not a tuple of frames
        )
        if e.primitive == "stelling_assert"
        else e
        for e in q.jaxpr.eqns
    )
    bad = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=q.jaxpr.constvars,
            invars=q.jaxpr.invars,
            outvars=q.jaxpr.outvars,
            eqns=eqns,
        ),
        consts=q.consts,
    )
    p = propagate(bad)
    (item,) = slice_unknown_obligations(bad, p, interval_env(bad))
    assert isinstance(item, DeclinedObligation), item
    assert "not a list of source frames" in item.reason, item.reason
    assert "cannot be CHECKED" in item.reason, item.reason
    assert "internal error" not in item.reason, (
        "the total comparison is the repair; the net is only its backstop"
    )


# ── the totality claims, DRIVEN (audit 0.2.0 B6 re-audit, R5 / R7) ────────


def test_frames_is_total_on_a_list_that_will_not_iterate():
    """R5. `_frames`' docstring says TOTAL and says that not raising is "the
    structural guarantee". It was not: `isinstance(v, list)` is a claim
    about the TYPE, and a `list` subclass whose `__iter__` raises satisfies
    it and then raises inside `tuple(v)`.

    A value that will not iterate is not a frame list either, so it reads as
    `None` — "this association cannot be CHECKED" — by exactly the same
    reasoning an `int` does. Asserted through the helper AND end to end,
    because the point is that the net is not what answers.

    **THE END-TO-END HALF NOW MEASURES A DIFFERENT THING, AND SAYS SO** —
    audit 0.2.0 B6 audit 6. `ir`'s canonicalization door replaces
    `source_info` with an exact `tuple` at construction, reading the list's
    payload through `list.__getitem__` rather than asking it to iterate, so
    an EQUATION can no longer carry a `Hostile` at all: the route that used
    to reach `_frames` with one is shut one layer earlier. That is a
    strictly better place for it and it is not a substitute for this
    helper's totality, because `_frames` also reads
    `ObligationReport.source_info`, which is not an `ir` dataclass field
    and goes through no door. So the helper is still driven directly, and
    the end-to-end half now asserts what is true: an equation's frames are
    canonical, and a NON-frame element (which canonicalization does not
    make a frame) still declines rather than raising."""
    import stelling.obligation as OB

    class Hostile(list):
        def __iter__(self):
            raise RuntimeError("this list refuses to iterate")

    assert isinstance(Hostile([1, 2]), list)
    assert OB._frames(Hostile(["a"])) is None
    # the other half of the helper's totality, and the one the door cannot
    # take away: an `ObligationReport`'s frames go through no `ir` door
    assert OB._frames(3) is None

    q = square_query()

    def _rebuild(source_info):
        eqns = tuple(
            ir.JaxprEqn(
                primitive=e.primitive, invars=e.invars, outvars=e.outvars,
                params=e.params, effects=e.effects,
                source_info=source_info,
            )
            if e.primitive == "stelling_assert" else e
            for e in q.jaxpr.eqns
        )
        return ir.ClosedJaxpr(
            jaxpr=ir.Jaxpr(
                constvars=q.jaxpr.constvars, invars=q.jaxpr.invars,
                outvars=q.jaxpr.outvars, eqns=eqns,
            ),
            consts=q.consts,
        )

    # 1. the door reads the hostile list's payload rather than iterating
    #    it, so the equation carries an ordinary frame tuple
    fine = _rebuild(Hostile(["somewhere.py:1"]))
    (asserted,) = [e for e in fine.jaxpr.eqns
                   if e.primitive == "stelling_assert"]
    assert asserted.source_info == ("somewhere.py:1",), asserted.source_info
    assert type(asserted.source_info) is tuple

    # 2. and a `source_info` that is not a frame list at all is still a
    #    DECLINE and not a raise. Canonicalization settles the TYPE of what
    #    is stored, never whether it means anything: an `int` is an exact
    #    `int` and is carried, and `_frames` reads it as `None` — the
    #    original M17′ row, reached through the door rather than around it.
    bad = _rebuild(3)
    assert [e.source_info for e in bad.jaxpr.eqns
            if e.primitive == "stelling_assert"] == [3]
    p = propagate(bad)
    (item,) = slice_unknown_obligations(bad, p, interval_env(bad))
    assert isinstance(item, DeclinedObligation), item
    assert "not a list of source frames" in item.reason, item.reason
    assert "internal error" not in item.reason, (
        "the net answered, which means `_frames` or its caller raised — the "
        "totality claim is still a docstring assertion rather than a property"
    )


def test_the_net_around_the_association_cannot_itself_raise(monkeypatch):
    """R7. A net that re-raises while composing its own message is not a
    net: the escape costs every sibling obligation's verdict exactly as the
    original raise would have.

    Three of its four reads can raise on a hostile object — `str(e)` runs
    the exception's own `__str__`, and `getattr(o, name, default)` returns
    the default only for `AttributeError`. All three are driven here at
    once, and the sibling obligation is asserted to survive."""
    import stelling.obligation as OB

    class Nasty(Exception):
        def __str__(self):
            raise RuntimeError("even my message refuses to be read")

    q = two_unknown_obligations_query()
    p = propagate(q)
    unknown = [o for o in p.obligations if o.status == "unknown"]
    assert len(unknown) == 2

    class HostileObligation:
        """The first obligation, with every field the handler reads made
        hostile — but still answering `top_level_eqn_pos`, so it reaches
        `_decide` and the raise happens inside the netted body."""

        def __init__(self, real):
            self._real = real

        status = "unknown"

        @property
        def top_level_eqn_pos(self):
            return self._real.top_level_eqn_pos

        @property
        def index(self):
            raise RuntimeError("index refuses to be read")

        @property
        def source_info(self):
            raise RuntimeError("source_info refuses to be read")

    real = OB._frames
    calls = {"n": 0}

    def boom(v):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Nasty()
        return real(v)

    monkeypatch.setattr(OB, "_frames", boom)
    hostile = HostileObligation(unknown[0])
    import dataclasses

    mixed = dataclasses.replace(
        p, obligations=(hostile,) + tuple(p.obligations[1:])
    )
    items = OB.slice_unknown_obligations(q, mixed, interval_env(q))

    assert len(items) == 2, items
    first, second = items
    assert isinstance(first, DeclinedObligation), (
        "the handler re-raised and took the whole call with it"
    )
    assert "internal error" in first.reason, first.reason
    assert "Nasty" in first.reason, first.reason
    assert "<unreadable message>" in first.reason, first.reason
    assert first.index == -1 and first.source_info == ()
    assert not isinstance(second, DeclinedObligation), (
        "the sibling declined with it, so the net is not per obligation"
    )


def test_the_claimants_count_is_read_ONCE(monkeypatch):
    """The count in the guard and the count in the sentence it prints must
    be the same read. `claimants.get(pos, 0)` followed by `claimants[pos]`
    were two, and the second raised `KeyError` whenever the key was absent —
    turning the intended "N obligations claim top-level assert position P"
    into "internal error: KeyError: 3".

    The absent case is reachable because `pos` here is a SECOND read of
    `o.top_level_eqn_pos`; `claimants` was built from a first one. An
    obligation that does not answer the same way twice is what this drives.
    """
    import stelling.obligation as OB

    q = square_query()
    p = propagate(q)
    (real,) = [o for o in p.obligations if o.status == "unknown"]
    pos = real.top_level_eqn_pos
    assert pos is not None

    class Drifting:
        status = "unknown"
        index = real.index
        source_info = real.source_info

        def __init__(self):
            self._n = 0

        @property
        def top_level_eqn_pos(self):
            # first read (building `claimants`) answers None; the second
            # (inside `_decide`) answers the real position, so `claimants`
            # has no entry for it
            self._n += 1
            return None if self._n == 1 else pos

    import dataclasses

    mixed = dataclasses.replace(p, obligations=(Drifting(),))
    (item,) = OB.slice_unknown_obligations(q, mixed, interval_env(q))
    assert isinstance(item, DeclinedObligation), item
    assert "0 obligations claim top-level assert position" in item.reason, (
        item.reason
    )
    assert "internal error" not in item.reason, item.reason
    assert "KeyError" not in item.reason, item.reason
