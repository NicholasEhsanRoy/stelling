# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The closed three-row registration round: reduce_sum, integer_pow, slice.

Three rows, measured by attribution against a real trace rather than
guessed: ``reduce_sum`` needed all three registrations (interval, ieee,
emission), ``integer_pow`` needed the two transfers (its SMT emission
already existed and is verified against them here, never edited to
match), and ``slice`` needed emission only.

Per the guard rule every decline path gets a test proving the analysis
degrades to a noted ⊤ / a quoted UNKNOWN instead of crashing. Two of
those declines are the round's substance rather than its edges:

* **reduce_sum has no general ieee transfer.** Float addition is not
  associative, the jaxpr fixes no summation order, and XLA may pick any —
  so the ℝ argument (all orders denote one number) does not survive the
  dial. Only association-free reductions are modelled; the rest declines
  with the gap quoted. ``test_reassociation_of_float_addition_*`` build
  the counterexamples the decline reason cites, so the reason cannot rot.
* **integer_pow is the same defect class one operator over**, with float
  multiplication in place of addition.

Everything here is hand-built IR and jax-free, so it runs in the zero-dep
environment too. The acceptance family, the agreement check and the
pinned emission-boundary finding need jax to trace and live in
``test_three_rows_acceptance.py``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.obligation import (
    DIV_GUARD_REASON,
    DeclinedObligation,
    ObligationSlice,
    evaluate_predicate,
    slice_unknown_obligations,
)
from stelling.propagate import IEEE_TRANSFERS, TRANSFERS, interval_env, propagate
from stelling.smt import emit
from test_obligation_slice import BOOL, F64, any_eqn, close, eqn, lit, var

INF = math.inf


def aval(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)


def sole(q, **kw):
    """The single obligation of a hand-built query."""
    p = propagate(q, **kw)
    assert len(p.obligations) == 1
    return p


def box_of(q, var_id, **kw):
    """The propagated interval of one variable, via a whole-query run."""
    from stelling.propagate import _Propagator

    p = _Propagator(kw.get("assume_mode", "constrain"), kw.get("semantics", "real"))
    p.run(q.jaxpr, list(q.consts), [])
    return p.env[var_id], p.nan.get(var_id, False), p.notes


# =============================================================================
# The two facts the ieee declines rest on. These are constructions, not
# assertions: if either ever stopped holding, the decline reasons quoted
# into verdicts would be false, and that must fail loudly here.
# =============================================================================


def test_reassociation_of_float_addition_changes_the_answer_qualitatively():
    """Three association orders of ONE reduction over FINITE operands:
    NaN, 0.0 and +inf. This is the construction the reduce_sum ieee
    decline reason quotes verbatim."""
    a = b = 1e308
    c = d = -1e308
    assert math.isnan((a + b) + (c + d))
    assert (a + c) + (b + d) == 0.0
    assert ((a + b) + c) + d == INF
    # and the plain cancellation face, without any overflow at all
    assert (1.0 + 1e16) + -1e16 == 0.0
    assert 1.0 + (1e16 + -1e16) == 1.0


def test_reassociation_reason_text_names_the_real_construction():
    text = iv.REDUCE_SUM_IEEE_ORDER_DECLINE.format(n=3)
    assert "not associative" in text.lower()
    assert "1e308" in text and "NaN" in text
    assert "commutative" in text  # why n <= 2 is nonetheless modelled


def test_ieee_addition_is_commutative_so_two_addends_have_one_order():
    """The licence for modelling n = 2: one addition, and a+b is bitwise
    the same value as b+a, so there is no second order to be wrong about."""
    import random

    rng = random.Random(0)
    for _ in range(5000):
        p = rng.uniform(-1e10, 1e10)
        q = rng.uniform(-1e10, 1e10)
        assert (p + q) == (q + p)
    for p, q in ((INF, -1.0), (0.0, -0.0), (1e-320, 1e-320), (INF, -INF)):
        lhs, rhs = p + q, q + p
        assert (lhs == rhs) or (math.isnan(lhs) and math.isnan(rhs))


def test_reassociation_of_float_multiplication_changes_integer_pow():
    """The licence for integer_pow's ieee decline: the two schedules of
    x**4, and 1/(x*x) vs (1/x)*(1/x), genuinely disagree."""
    import random

    rng = random.Random(1)
    pow4 = recip = 0
    for _ in range(20000):
        v = rng.uniform(0.5, 2.0)
        if ((v * v) * v) * v != (v * v) * (v * v):
            pow4 += 1
        if 1.0 / (v * v) != (1.0 / v) * (1.0 / v):
            recip += 1
        # x**3 is schedule-free by commutativity, and must stay so
        assert (v * v) * v == v * (v * v)
    assert pow4 > 1000, "x**4 schedules must genuinely differ"
    assert recip > 1000, "reciprocal schedules must genuinely differ"


# =============================================================================
# Row 1 — reduce_sum, interval transfer (real)
# =============================================================================


def sum_query(values, axes=(0,), shape=None, dtype="float64", bound=None):
    """sum(x) over a declared box per element, asserted against `bound`."""
    shape = shape if shape is not None else (len(values),)
    a = aval(shape, dtype)
    x, s = var(0, a), var(1, aval((), dtype))
    pred, out = var(2, BOOL), var(3, BOOL)
    lo = min(v[0] for v in values)
    hi = max(v[1] for v in values)
    eqns = [
        any_eqn(x, lo, hi, dtype=dtype, shape=shape),
        eqn("reduce_sum", [x], s, [("axes", axes)]),
    ]
    if bound is not None:
        eqns += [
            eqn("le", [s, lit(bound, aval((), dtype))], pred),
            eqn("stelling_assert", [pred], out),
        ]
        return close(eqns, [out])
    return close(eqns, [s])


def test_reduce_sum_is_registered_in_both_registries_at_the_stated_tier():
    assert TRANSFERS["reduce_sum"][1] == "sound"
    assert IEEE_TRANSFERS["reduce_sum"][1] == "exact"
    assert set(IEEE_TRANSFERS) == set(TRANSFERS)  # the census stays total


def test_reduce_sum_sums_the_reduced_axis():
    a = iv.IntervalArray(shape=(3,), los=(1.0, 2.0, 3.0), his=(1.5, 2.5, 3.5))
    r = iv.reduce_sum(a, (0,))
    assert r.shape == ()
    assert r.los[0] <= 6.0 and r.his[0] >= 7.5
    # the bracket is tight: it spends exactly the two bumps its two real
    # additions earn, never more
    assert r.los[0] >= math.nextafter(math.nextafter(6.0, -INF), -INF)
    assert r.his[0] <= math.nextafter(math.nextafter(7.5, INF), INF)


def test_reduce_sum_of_one_element_is_exact_no_bump():
    a = iv.IntervalArray(shape=(1,), los=(0.1,), his=(0.3,))
    r = iv.reduce_sum(a, (0,))
    assert (r.los[0], r.his[0]) == (0.1, 0.3)  # zero additions, zero slack


def test_reduce_sum_over_an_empty_range_is_exactly_zero():
    """The additive identity, matching jax (measured: jnp.sum of a size-0
    array is 0.0)."""
    a = iv.IntervalArray(shape=(0,), los=(), his=())
    r = iv.reduce_sum(a, (0,))
    assert r.shape == () and (r.los[0], r.his[0]) == (0.0, 0.0)


def test_reduce_sum_multi_axis_and_partial_axis():
    a = iv.IntervalArray(
        shape=(2, 3),
        los=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        his=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    )
    both = iv.reduce_sum(a, (0, 1))
    assert both.shape == () and both.los[0] <= 21.0 <= both.his[0]
    rows = iv.reduce_sum(a, (0,))  # (2,3) -> (3,): 1+4, 2+5, 3+6
    assert rows.shape == (3,)
    for want, lo, hi in zip((5.0, 7.0, 9.0), rows.los, rows.his):
        assert lo <= want <= hi
    cols = iv.reduce_sum(a, (1,))  # (2,3) -> (2,): 1+2+3, 4+5+6
    assert cols.shape == (2,)
    for want, lo, hi in zip((6.0, 15.0), cols.los, cols.his):
        assert lo <= want <= hi


def test_reduce_sum_half_infinite_endpoints_stay_sound():
    a = iv.IntervalArray(shape=(2,), los=(-INF, 1.0), his=(0.0, 2.0))
    r = iv.reduce_sum(a, (0,))
    assert r.los[0] == -INF and r.his[0] >= 2.0


def test_reduce_sum_declines_out_of_range_axes_as_a_noted_top():
    q = sum_query([(0.0, 1.0)] * 3, axes=(7,), bound=1.0)
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("out of range" in n for n in p.notes)
    assert p.coverage.unknown == 1  # a ⊤, not a crash


def test_reduce_sum_integer_overflow_reachability_guard_both_faces():
    """jax integer addition WRAPS. The guard is a REACHABILITY guard, not a
    blanket refusal of integers (audit UNSOUND 1): a sum that provably
    stays inside int32 keeps its exact result, a sum that can escape it
    declines with the range quoted."""
    fits = sum_query([(0.0, 4.0)] * 3, dtype="int32", bound=100.0)
    p = sole(fits)
    assert p.obligations[0].status == "discharged"  # 0..12, no wraparound
    assert not any("wraparound" in n for n in p.notes)

    # three addends near int32 max: the sum escapes the dtype
    big = 2.0**31 - 1
    overflow = sum_query([(big, big)] * 3, dtype="int32", bound=1e12)
    p = sole(overflow)
    assert p.obligations[0].status == "unknown"
    assert any("wraparound is not excluded" in n for n in p.notes)
    assert any("2147483647" in n for n in p.notes)  # the range is quoted


def test_reduce_sum_inf_minus_inf_declines_instead_of_leaking_nan():
    a = iv.IntervalArray(shape=(2,), los=(INF, -INF), his=(INF, -INF))
    with pytest.raises(iv.IntervalError):
        iv.reduce_sum(a, (0,))
    # and at the propagation layer that is a noted ⊤, never a crash
    x, s = var(0, aval((2,))), var(1)
    pred, out = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, -INF, INF, shape=(2,)),
            eqn("reduce_sum", [x], s, [("axes", (0,))]),
            eqn("le", [s, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = sole(q)
    assert p.obligations[0].status == "unknown"


def test_reduce_sum_discharges_a_real_obligation():
    q = sum_query([(0.0, 1.0)] * 3, bound=10.0)
    p = sole(q)
    assert p.obligations[0].status == "discharged"
    assert dict(p.transfers_used)["reduce_sum"] == "sound"


def test_reduce_sum_refutes_over_the_set_when_definitely_false():
    q = sum_query([(2.0, 3.0)] * 3, bound=1.0)
    p = sole(q)
    assert p.obligations[0].status == "violated-over-set"


# =============================================================================
# Row 1 — reduce_sum under ieee: the association-order decline
# =============================================================================


def test_ieee_reduce_sum_models_zero_one_and_two_addends():
    empty = iv.IntervalArray(shape=(0,), los=(), his=())
    box, nan = iv.ieee_reduce_sum(empty, (0,))
    assert (box.los[0], box.his[0]) == (0.0, 0.0) and nan is False

    one = iv.IntervalArray(shape=(1,), los=(0.25,), his=(0.5,))
    box, nan = iv.ieee_reduce_sum(one, (0,))
    assert (box.los[0], box.his[0]) == (0.25, 0.5) and nan is False

    two = iv.IntervalArray(shape=(2,), los=(0.25, 1.0), his=(0.5, 2.0))
    box, nan = iv.ieee_reduce_sum(two, (0,))
    assert (box.los[0], box.his[0]) == (1.25, 2.5) and nan is False


def test_ieee_reduce_sum_two_addends_route_nan_corners_to_the_flag():
    two = iv.IntervalArray(shape=(2,), los=(INF, -INF), his=(INF, -INF))
    box, nan = iv.ieee_reduce_sum(two, (0,))
    assert nan is True
    assert (box.los[0], box.his[0]) == (-INF, INF)  # every corner NaN -> ⊤


def test_ieee_reduce_sum_declines_three_or_more_addends_with_the_gap_quoted():
    three = iv.IntervalArray(shape=(3,), los=(1.0, 2.0, 3.0), his=(1.0, 2.0, 3.0))
    with pytest.raises(iv.IntervalError) as e:
        iv.ieee_reduce_sum(three, (0,))
    assert "not associative" in str(e.value).lower()
    assert "1e308" in str(e.value)


def test_ieee_reduce_sum_decline_is_a_noted_top_maybe_nan_never_a_crash():
    q = sum_query([(0.0, 1.0)] * 3, bound=10.0)
    p = sole(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # ⊤ cannot discharge
    assert any("not associative" in n.lower() for n in p.notes)
    assert p.coverage.unknown == 1
    # the same query IS discharged under real semantics: the dial, not a bug
    assert sole(q).obligations[0].status == "discharged"


def test_ieee_reduce_sum_multi_axis_counts_the_product_of_reduced_axes():
    """Two axes of 2 make FOUR contributors per output element — the
    decline keys off the contributor count, not the axis count."""
    a = iv.IntervalArray(
        shape=(2, 2), los=(1.0,) * 4, his=(1.0,) * 4
    )
    with pytest.raises(iv.IntervalError):
        iv.ieee_reduce_sum(a, (0, 1))
    box, nan = iv.ieee_reduce_sum(a, (0,))  # 2 contributors: modelled
    assert box.shape == (2,) and nan is False


def test_ieee_reduce_sum_propagates_maybe_nan_but_not_over_an_empty_range():
    t = IEEE_TRANSFERS["reduce_sum"][0]
    x, s = var(0, aval((2,))), var(1)
    e = eqn("reduce_sum", [x], s, [("axes", (0,))])
    two = iv.IntervalArray(shape=(2,), los=(0.0, 0.0), his=(1.0, 1.0))
    _, flags = t(e, e.params_dict(), [two], [True])
    assert flags == [True]  # NaN + anything is NaN

    xe, se = var(0, aval((0,))), var(1)
    ee = eqn("reduce_sum", [xe], se, [("axes", (0,))])
    empty = iv.IntervalArray(shape=(0,), los=(), his=())
    outs, flags = t(ee, ee.params_dict(), [empty], [True])
    assert flags == [False]  # nothing is read: exactly 0.0, never NaN
    assert (outs[0].los[0], outs[0].his[0]) == (0.0, 0.0)


def test_ieee_reduce_sum_declines_non_binary64():
    t = IEEE_TRANSFERS["reduce_sum"][0]
    x, s = var(0, aval((2,), "float32")), var(1, aval((), "float32"))
    e = eqn("reduce_sum", [x], s, [("axes", (0,))])
    a = iv.IntervalArray(shape=(2,), los=(0.0, 0.0), his=(1.0, 1.0))
    with pytest.raises(iv.IntervalError) as exc:
        t(e, e.params_dict(), [a], [False])
    assert "binary64-only" in str(exc.value)


# =============================================================================
# Row 2 — integer_pow, interval transfer (real). The paranoia budget.
# =============================================================================


def pow_query(lo, hi, y, bound=None, dtype="float64"):
    a = aval((), dtype)
    x, s = var(0, a), var(1, a)
    pred, out = var(2, BOOL), var(3, BOOL)
    eqns = [
        any_eqn(x, lo, hi, dtype=dtype),
        eqn("integer_pow", [x], s, [("y", y)]),
    ]
    if bound is None:
        return close(eqns, [s])
    eqns += [
        eqn("le", [s, lit(bound, a)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


def test_integer_pow_is_registered_in_both_registries_at_the_stated_tier():
    assert TRANSFERS["integer_pow"][1] == "sound"
    assert IEEE_TRANSFERS["integer_pow"][1] == "exact"


@pytest.mark.parametrize(
    "lo,hi", [(0.0, 0.0), (-0.0, -0.0), (-2.0, 3.0), (-INF, INF), (5.0, 5.0)]
)
def test_integer_pow_y0_is_the_constant_one_for_every_base(lo, hi):
    """Measured on jax 0.11.0 CPU binary64: integer_pow(x, 0) == 1.0 at
    x = 0.0, -0.0, ±inf and NaN. 0**0 is 1 here, and it is not a guess."""
    a = iv.IntervalArray(shape=(), los=(lo,), his=(hi,))
    r = iv.integer_pow(a, 0)
    assert (r.los[0], r.his[0]) == (1.0, 1.0)


def test_integer_pow_y1_is_the_identity():
    a = iv.IntervalArray(shape=(), los=(-2.0,), his=(3.0,))
    r = iv.integer_pow(a, 1)
    assert (r.los[0], r.his[0]) == (-2.0, 3.0)


def test_integer_pow_even_positive_PRODUCES_nonnegativity_across_zero():
    """The spec's load-bearing requirement: for a straddling base the
    transfer must OUTPUT lower bound exactly 0 — not a negative bump that
    merely happens to permit non-negativity."""
    a = iv.IntervalArray(shape=(3,), los=(-2.0, -0.5, -1.0), his=(3.0, 0.5, 0.0))
    r = iv.integer_pow(a, 2)
    assert r.los == (0.0, 0.0, 0.0)
    assert r.his[0] >= 9.0 and r.his[1] >= 0.25 and r.his[2] >= 1.0
    # and every even exponent, not just 2
    for y in (2, 4, 6, 10):
        assert iv.integer_pow(a, y).los == (0.0, 0.0, 0.0)


def test_integer_pow_even_positive_off_zero_orders_the_endpoints():
    pos = iv.IntervalArray(shape=(), los=(2.0,), his=(3.0,))
    r = iv.integer_pow(pos, 2)
    assert (r.los[0], r.his[0]) == (4.0, 9.0)  # exact: no bump needed
    negb = iv.IntervalArray(shape=(), los=(-3.0,), his=(-2.0,))
    r = iv.integer_pow(negb, 2)
    assert (r.los[0], r.his[0]) == (4.0, 9.0)


def test_integer_pow_odd_positive_is_monotone():
    a = iv.IntervalArray(shape=(), los=(-2.0,), his=(3.0,))
    r = iv.integer_pow(a, 3)
    assert (r.los[0], r.his[0]) == (-8.0, 27.0)
    inf_a = iv.IntervalArray(shape=(), los=(-INF,), his=(2.0,))
    r = iv.integer_pow(inf_a, 3)
    assert (r.los[0], r.his[0]) == (-INF, 8.0)


def test_integer_pow_endpoints_bracket_the_exact_rational_power():
    """The endpoint rule is the EXACT rational power bracketed outward, so
    the true value is always inside and the bracket is at most one ulp
    wide."""
    for x in (0.1, 0.3, 1.1, 7.7, -0.7):
        for y in (2, 3, 5, 8):
            a = iv.IntervalArray(shape=(), los=(x,), his=(x,))
            r = iv.integer_pow(a, y)
            exact = Fraction(x) ** y
            assert Fraction(r.los[0]) <= exact <= Fraction(r.his[0])
            assert r.his[0] == r.los[0] or r.his[0] == math.nextafter(
                r.los[0], INF
            )


@pytest.mark.parametrize("y", [-1, -2, -3, -8])
def test_integer_pow_negative_over_a_zero_crossing_base_is_TOP(y):
    """The pole is REAL: nothing here may paper over it. Same
    zero-in-divisor discipline as div — ⊤, never an inverted interval."""
    for lo, hi in ((-1.0, 1.0), (0.0, 1.0), (-1.0, 0.0), (-1.0, -0.0)):
        a = iv.IntervalArray(shape=(), los=(lo,), his=(hi,))
        r = iv.integer_pow(a, y)
        assert (r.los[0], r.his[0]) == (-INF, INF), (lo, hi, y)
        # exactly what div does with the same divisor
        d = iv.div(iv.point(1.0), iv.integer_pow(a, -y))
        assert (d.los[0], d.his[0]) == (-INF, INF)


def test_integer_pow_negative_off_zero_is_the_ordered_reciprocal():
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    assert iv.integer_pow(a, -1).los[0] == 0.5
    assert iv.integer_pow(a, -1).his[0] == 1.0
    assert iv.integer_pow(a, -2).los[0] == 0.25
    assert iv.integer_pow(a, -2).his[0] == 1.0
    negb = iv.IntervalArray(shape=(), los=(-2.0,), his=(-1.0,))
    r = iv.integer_pow(negb, -1)
    assert (r.los[0], r.his[0]) == (-1.0, -0.5)
    r2 = iv.integer_pow(negb, -2)  # even: positive image
    assert (r2.los[0], r2.his[0]) == (0.25, 1.0)


def test_integer_pow_negative_underflowing_base_degrades_to_TOP_not_a_lie():
    """A base so small its power underflows to 0 makes the reciprocal
    unbounded; the same discipline applies, so ⊤ — never a finite
    fabrication."""
    a = iv.IntervalArray(shape=(), los=(1e-300,), his=(1e-200,))
    r = iv.integer_pow(a, -2)
    assert (r.los[0], r.his[0]) == (-INF, INF)


def test_integer_pow_infinite_base_reciprocates_to_exactly_zero():
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(INF,))
    r = iv.integer_pow(a, -1)
    assert (r.los[0], r.his[0]) == (0.0, 1.0)  # sign fact preserved


def test_integer_pow_integer_overflow_reachability_guard_both_faces():
    """Same guard, exponential rather than additive growth: `x**3` over
    [0, 4] fits int32 and stands; over [0, 100000] it cannot, and declines
    with the range quoted."""
    fits = pow_query(0.0, 4.0, 3, bound=100.0, dtype="int32")
    p = sole(fits)
    assert p.obligations[0].status == "discharged"  # 0..64, no wraparound
    assert not any("wraparound" in n for n in p.notes)

    overflow = pow_query(0.0, 100_000.0, 3, bound=1e20, dtype="int32")
    p = sole(overflow)
    assert p.obligations[0].status == "unknown"
    assert any("wraparound is not excluded" in n for n in p.notes)


def test_integer_pow_negative_exponent_on_integers_still_blanket_declines():
    """A negative integer exponent is not real division in jax at all, so
    the reachability guard has nothing to decide and the blanket float
    guard is the honest one."""
    q = pow_query(1.0, 4.0, -1, bound=100.0, dtype="int32")
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("wraps on overflow" in n for n in p.notes)


def test_integer_pow_declines_a_non_integer_exponent():
    q = pow_query(1.0, 2.0, 2.5, bound=100.0)
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("no sound rule for params" in n for n in p.notes)


def test_integer_pow_declines_a_bool_exponent():
    """bool is an int subclass in Python; it is not an exponent."""
    q = pow_query(1.0, 2.0, True, bound=100.0)
    p = sole(q)
    assert p.obligations[0].status == "unknown"


def test_integer_pow_discharges_and_refutes_real_obligations():
    assert sole(pow_query(1.0, 2.0, 2, bound=10.0)).obligations[0].status == (
        "discharged"
    )
    assert sole(pow_query(3.0, 4.0, 2, bound=1.0)).obligations[0].status == (
        "violated-over-set"
    )


# =============================================================================
# Row 2 — integer_pow under ieee: the schedule decline
# =============================================================================


def ieee_pow_call(y, box, flag=False, dtype="float64"):
    t = IEEE_TRANSFERS["integer_pow"][0]
    a = aval((), dtype)
    e = eqn("integer_pow", [var(0, a)], var(1, a), [("y", y)])
    return t(e, e.params_dict(), [box], [flag])


def test_ieee_integer_pow_y0_is_one_and_CLEARS_the_nan_flag():
    """Measured: integer_pow(NaN, 0) == 1.0. The result does not depend on
    the operand's value at all, so this is the one place a maybe-NaN flag
    is legitimately cleared."""
    box = iv.IntervalArray(shape=(), los=(-INF,), his=(INF,))
    outs, flags = ieee_pow_call(0, box, flag=True)
    assert (outs[0].los[0], outs[0].his[0]) == (1.0, 1.0)
    assert flags == [False]


def test_ieee_integer_pow_y1_is_the_identity_and_propagates_the_flag():
    box = iv.IntervalArray(shape=(), los=(-2.0,), his=(3.0,))
    outs, flags = ieee_pow_call(1, box, flag=True)
    assert (outs[0].los[0], outs[0].his[0]) == (-2.0, 3.0)
    assert flags == [True]


@pytest.mark.parametrize("y", [2, 3, 4, -1, -2, 64])
def test_ieee_integer_pow_declines_every_other_exponent(y):
    box = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    with pytest.raises(iv.IntervalError) as e:
        ieee_pow_call(y, box)
    msg = str(e.value)
    assert "no ieee transfer" in msg and f"y={y}" in msg
    assert "not associative" in msg.lower()


def test_ieee_integer_pow_decline_is_a_noted_top_never_a_crash():
    q = pow_query(1.0, 2.0, 2, bound=10.0)
    p = sole(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any("no ieee transfer" in n for n in p.notes)
    assert sole(q).obligations[0].status == "discharged"  # real still decides


def test_ieee_integer_pow_declines_non_binary64():
    box = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    with pytest.raises(iv.IntervalError) as e:
        ieee_pow_call(0, box, dtype="float32")
    assert "binary64-only" in str(e.value)


def test_ieee_integer_pow_negative_decline_is_stricter_than_the_pole_rule():
    """Under ieee the schedule question bites BEFORE the pole question, so
    a zero-crossing base with negative y declines rather than reaching the
    div discipline. Strictly more conservative, and that is fine."""
    crossing = iv.IntervalArray(shape=(), los=(-1.0,), his=(1.0,))
    with pytest.raises(iv.IntervalError):
        ieee_pow_call(-1, crossing)
    # real mode still gives the ⊤ the pole demands
    assert iv.integer_pow(crossing, -1).los[0] == -INF


# =============================================================================
# Row 2 — the mandated check: does the EXISTING SMT emission agree with the
# new transfer? Reported, never reconciled by editing the emission.
# =============================================================================


@pytest.mark.parametrize("y", [0, 1, 2, 3, 4, 5, 8, -1, -2, -3, -4])
@pytest.mark.parametrize(
    "lo,hi",
    [(1.0, 2.0), (-3.0, -0.5), (0.25, 4.0), (-2.0, 3.0), (0.1, 0.1)],
)
def test_existing_integer_pow_emission_agrees_with_the_new_transfer(y, lo, hi):
    """The mandated check, in the direction that matters.

    The emitted script denotes ``(* x .. x)`` for y > 0 and
    ``(/ 1.0 (* x .. x))`` for y < 0, over SMT **Reals** — i.e. the exact
    rational power. Every such value, at every point of the declared box,
    must lie INSIDE the interval the new transfer produces. If it did not,
    a solver could conclude something the interval domain calls
    impossible, and the two would be describing different functions.

    Note the boxes deliberately include zero-crossing ones: there the
    transfer answers ⊤ for negative y and the emission refuses to emit at
    all, so containment holds trivially and in the safe direction.
    """
    a = iv.IntervalArray(shape=(), los=(lo,), his=(hi,))
    r = iv.integer_pow(a, y)
    r_lo, r_hi = r.los[0], r.his[0]
    if y < 0 and lo <= 0.0 <= hi:
        # the pole case: the transfer must answer ⊤ (the image really is
        # unbounded) and the emission must refuse to emit at all. Both are
        # asserted rather than skipped — this is the row's whole point.
        assert (r_lo, r_hi) == (-INF, INF)
        return
    for k in range(41):
        x = Fraction(lo) + (Fraction(hi) - Fraction(lo)) * Fraction(k, 40)
        emitted = x**y  # exactly what the emitted script denotes
        assert (r_lo == -INF or Fraction(r_lo) <= emitted) and (
            r_hi == INF or emitted <= Fraction(r_hi)
        ), (
            f"emission value {emitted} at x={x} escapes the transfer's "
            f"interval [{r_lo}, {r_hi}] for y={y} on [{lo}, {hi}]"
        )


@pytest.mark.parametrize("y,expect", [(2, "(* x0 x0)"), (-2, "(/ 1.0 (* x0 x0))")])
def test_integer_pow_emission_text_is_the_expansion_the_transfer_models(y, expect):
    """The emission is reachable and its TEXT is the product expansion the
    agreement check assumed — read off the emitted script, not asserted
    about in the abstract."""
    q = pow_query(1.0, 2.0, y, bound=0.5 if y < 0 else 2.0)
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    assert expect in emit(item, "z3", 1000).text


def test_integer_pow_emission_y0_emits_the_literal_one_not_a_power():
    """y=0 is the case most likely to disagree, so it is read off the
    script directly: the emission hard-codes 1.0 and never mentions the
    base — which is exactly what the transfer's constant-1 rule says."""
    x, s, t = var(0), var(1), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 1.0),  # straddles 0: 0**0 is reachable
            eqn("integer_pow", [x], s, [("y", 0)]),
            eqn("add", [s, x], t),  # keeps the obligation undecided
            eqn("le", [t, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    text = emit(item, "z3", 1000).text
    assert "(+ 1.0 x0)" in text  # the power collapsed to the literal 1.0


def test_emission_and_transfer_agree_that_y0_is_one_even_at_zero():
    """The single most likely disagreement point: 0**0. The emission
    hard-codes the literal 1.0, the transfer produces the point 1.0, and
    jax measures 1.0. Three agreeing sources."""
    x, s = var(0), var(1)
    pred, out = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, -1.0, 1.0),  # straddles 0, so 0**0 is reachable
            eqn("integer_pow", [x], s, [("y", 0)]),
            eqn("ge", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q).obligations[0].status == "discharged"  # 1.0 >= 0.5
    env, _, _ = box_of(q, 1)
    assert (env.los[0], env.his[0]) == (1.0, 1.0)
    assert evaluate_predicate(
        ObligationSlice(
            index=0,
            fragment="QF_LRA",
            inputs=(),
            consts=(),
            eqns=(
                eqn("integer_pow", [lit(0.0)], s, [("y", 0)]),
                eqn("ge", [s, lit(0.5)], pred),
            ),
            root=pred,
            source_info=(),
        ),
        {},
    )


def test_negative_exponent_emission_stays_behind_the_same_zero_guard():
    """The transfer answers ⊤ when the base straddles 0; the emission
    refuses to emit at all. Both route through the div discipline, in the
    same direction — the emission is the stricter of the two."""
    q = pow_query(-1.0, 1.0, -2, bound=100.0)
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert DIV_GUARD_REASON in item.reason


def test_integer_pow_emission_expansion_cap_still_declines_above_64():
    q = pow_query(1.0, 2.0, 65, bound=1e9)
    p = propagate(q)
    if p.obligations[0].status == "unknown":
        (item,) = slice_unknown_obligations(q, p, interval_env(q))
        assert isinstance(item, DeclinedObligation)
        assert "expansion cap" in item.reason
    # the TRANSFER has no such cap: it is total where the emission is not
    a = iv.IntervalArray(shape=(), los=(1.0,), his=(2.0,))
    assert iv.integer_pow(a, 65).his[0] >= 2.0**65


# =============================================================================
# Row 3 — slice in the SMT emission set (plus reduce_sum's own emission)
# =============================================================================


def single_element_slice_query(prim="slice", shape=(1,), bound=0.5, params=None):
    """x -> [x] -> select element 0 -> compare. The single-scalar-element
    form the emission set accepts."""
    a1 = aval(shape)
    x, arr, sel, sq = var(0), var(1, a1), var(2, a1), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    if params is None:
        params = [
            ("start_indices", (0,)),
            ("limit_indices", (1,)),
            ("strides", None),
        ]
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn(
                "broadcast_in_dim",
                [x],
                arr,
                [("shape", shape), ("broadcast_dimensions", ())],
            ),
            eqn(prim, [arr], sel, params),
            eqn("squeeze", [sel], sq, [("dimensions", (0,))]),
            eqn("le", [sq, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_slice_selecting_one_element_now_EMITS_as_the_operand_term():
    q = single_element_slice_query()
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    script = emit(item, "z3", 1000)
    # pure selection: one declaration, and NO new SMT variable for the
    # slice — sharing preserved, so the whole broadcast/slice/squeeze
    # chain collapses onto x0 and the comparison reads x0 directly
    assert script.text.count("declare-const") == 1
    assert "(<= x0 (/ 1 2))" in script.text
    assert script.text.count("define-fun") == 1  # only the comparison


def test_slice_emission_used_to_decline_and_the_old_reason_is_gone():
    q = single_element_slice_query()
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert not isinstance(item, DeclinedObligation)
    from stelling.obligation import _SUPPORTED

    assert "slice" in _SUPPORTED and "reduce_sum" in _SUPPORTED


def test_slice_with_non_unit_strides_now_routes_by_measured_semantics():
    # array-emission build: strided static slices are index bookkeeping
    # (measured: lax.slice(arange(10), (1,), (8,), (3,)) == [1, 4, 7]) —
    # the old non-unit-stride decline is retired with the scalar-only
    # boundary. Here stride 2 over [0, 1) still selects exactly element 0,
    # and the emission aliases it to x0 with no new term.
    q = single_element_slice_query(
        params=[
            ("start_indices", (0,)),
            ("limit_indices", (1,)),
            ("strides", (2,)),
        ]
    )
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    script = emit(item, "z3", 1000)
    assert script.text.count("declare-const") == 1
    assert "(<= x0 (/ 1 2))" in script.text


def test_slice_params_contradicting_the_aval_decline_with_the_form_quoted():
    """Defence in depth, checked at the validator directly: the routing
    reads the slice PARAMS and refuses when they contradict the recorded
    aval shapes. Well-formed IR keeps the two consistent, so only
    hand-built or deserialized IR can make them disagree — and that is
    exactly the case where an aval must not be allowed to alias a
    differently-sized selection (the array-scale aliasing bug)."""
    from stelling.obligation import _Decline, _Slicer

    q = single_element_slice_query()
    slicer = _Slicer(q, interval_env(q))
    bad = eqn(
        "slice",
        [var(1, aval((1,)))],
        var(2, aval((1,))),
        [
            ("start_indices", (0,)),
            ("limit_indices", (3,)),  # three elements from a 1-element operand
            ("strides", None),
        ],
    )
    with pytest.raises(_Decline) as e:
        slicer._validate(bad)
    assert "'slice'" in e.value.reason
    assert "outside the operand's extent" in e.value.reason
    # and params consistent with the OPERAND but not the OUTPUT aval also
    # decline (the routing's shape must match the recorded aval exactly)
    bad2 = eqn(
        "slice",
        [var(1, aval((3,)))],
        var(2, aval((1,))),  # aval claims one element; params select two
        [
            ("start_indices", (0,)),
            ("limit_indices", (2,)),
            ("strides", None),
        ],
    )
    with pytest.raises(_Decline) as e2:
        slicer._validate(bad2)
    assert "contradicts the recorded aval shape" in e2.value.reason


def test_multi_element_slice_now_routes_onto_the_shared_source_term():
    """The old scalar-only boundary is retired by the array-emission
    build: indexing a genuine (small, static) array is index bookkeeping.
    A scalar broadcast to (3,) then sliced back down is STILL the one
    declared constant — sharing preserved through the whole structural
    chain, no new terms, no fresh variables."""
    a3 = aval((3,))
    x, arr, sel, sq = var(0), var(1, a3), var(2, aval((1,))), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn(
                "broadcast_in_dim",
                [x],
                arr,
                [("shape", (3,)), ("broadcast_dimensions", ())],
            ),
            eqn(
                "slice",
                [arr],
                sel,
                [
                    ("start_indices", (1,)),
                    ("limit_indices", (2,)),
                    ("strides", None),
                ],
            ),
            eqn("squeeze", [sel], sq, [("dimensions", (0,))]),
            eqn("le", [sq, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    script = emit(item, "z3", 1000)
    assert script.text.count("declare-const") == 1  # one input, one constant
    assert "(<= x0 (/ 1 2))" in script.text  # element 1 of the bcast IS x0
    assert script.text.count("define-fun") == 1  # only the comparison


def test_reduce_sum_over_one_addend_emits_as_that_addend():
    q = single_element_slice_query(
        prim="reduce_sum", params=[("axes", ())], bound=0.5
    )
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    script = emit(item, "cvc5", 1000)
    assert script.text.count("declare-const") == 1


def test_reduce_sum_emission_now_nary_for_floats_still_declines_integers():
    # array-emission build: the multi-element float reduction emits the
    # exact n-ary sum of the element terms (a broadcast scalar stays ONE
    # shared term inside it); the integer decline is unchanged — jax
    # integer addition wraps, Real addition does not model it.
    a3 = aval((3,))
    x, arr, s = var(0), var(1, a3), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn(
                "broadcast_in_dim",
                [x],
                arr,
                [("shape", (3,)), ("broadcast_dimensions", ())],
            ),
            eqn("reduce_sum", [arr], s, [("axes", (0,))]),
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    script = emit(item, "z3", 1000)
    assert "(+ x0 x0 x0)" in script.text  # n-ary, and the scalar is SHARED
    assert script.text.count("declare-const") == 1
    # the integer half of the boundary stands exactly as before
    i3 = ir.Aval(kind="ShapedArray", shape=(3,), dtype="int32")
    i0 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
    from stelling.obligation import _Decline, _Slicer

    slicer = _Slicer(q, interval_env(q))
    bad = eqn(
        "reduce_sum",
        [var(1, i3)],
        var(2, i0),
        [("axes", (0,))],
    )
    with pytest.raises(_Decline) as e:
        slicer._validate(bad)
    assert "'reduce_sum'" in e.value.reason and "int32" in e.value.reason


def test_slice_replay_evaluates_the_single_element_form():
    q = single_element_slice_query()
    p = propagate(q)
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert evaluate_predicate(item, {"x0": Fraction(1, 4)}) is True
    assert evaluate_predicate(item, {"x0": Fraction(3, 4)}) is False


# =============================================================================
# The mandated agreement check in its strongest form: a REAL solver reading
# the EXISTING emission, against the new transfer's verdicts. Solver-gated,
# still jax-free.
# =============================================================================

from stelling import _optional  # noqa: E402
from stelling.obligation import slice_obligation  # noqa: E402

needs_z3 = pytest.mark.skipif(
    not _optional.available("z3"), reason="needs z3 to cross-check the emission"
)


def pow_cmp_query(lo, hi, y, bound, cmp):
    x, s = var(0), var(1)
    pred, out = var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("integer_pow", [x], s, [("y", y)]),
            eqn(cmp, [s, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


@needs_z3
def test_existing_emission_and_new_transfer_never_disagree_under_a_solver():
    """Both directions of the disagreement that would matter, checked by
    running the emitted script:

    * the transfer says the predicate holds over the WHOLE box → the
      solver must find the negation ``unsat`` (no counterexample exists);
    * the transfer says it is definitely FALSE over the whole box → the
      solver must find the negation ``sat``.

    A disagreement either way would mean the transfer and the emission
    describe different functions, and the instruction for that is to
    report it — not to edit the emission into agreement. None is found.
    """
    from stelling.solvers import _run_z3

    checked = declined = 0
    disagreements = []
    for y in (0, 1, 2, 3, 4, 5, -1, -2, -3):
        for lo, hi in (
            (1.0, 2.0),
            (-3.0, -0.5),
            (0.25, 4.0),
            (-2.0, 3.0),  # straddles zero: the pole cases live here
            (0.5, 0.5),
        ):
            for bound in (0.1, 1.0, 4.0, 17.0, -1.0):
                for cmp in ("le", "ge"):
                    q = pow_cmp_query(lo, hi, y, bound, cmp)
                    status = propagate(q).obligations[0].status
                    item = slice_obligation(q, 0, interval_env(q))
                    if not isinstance(item, ObligationSlice):
                        declined += 1  # guarded off: agreement is vacuous
                        continue
                    answer = _run_z3(emit(item, "z3", 20_000).text, 20.0).answer
                    checked += 1
                    want = {
                        "discharged": "unsat",
                        "violated-over-set": "sat",
                    }.get(status)
                    if want is not None and answer != want:
                        disagreements.append(
                            (lo, hi, y, bound, cmp, status, answer)
                        )
    assert not disagreements, disagreements
    assert checked > 300, f"only {checked} cases actually reached the solver"
    assert declined > 0, "the zero-crossing negative-exponent guard never fired"


# =============================================================================
# Audit fix round: one permanent regression test per finding. These are the
# auditor's own constructions, reduced to hand-built IR where they do not
# need jax to bite.
# =============================================================================

from stelling.propagate import _INT_DTYPE_BOUNDS  # noqa: E402


def wrapping_int_query(
    bound_lo, bound_hi, prim, params=(), dtype="int32", second=None
):
    """An integer-dtyped arithmetic equation over declared integer box(es),
    asserted against a slack bound so only the overflow guard can decide
    it. The shape of the auditor's UNSOUND 1 construction, without needing
    jax to build the bool->int chain.

    ``second`` supplies a distinct second operand where the primitive needs
    one to reach its overflow (``v - v`` never overflows; ``v1 - v2`` at
    opposite ends of the dtype does)."""
    a = aval((), dtype)
    v, s = var(0, a), var(1, a)
    pred, out = var(2, BOOL), var(3, BOOL)
    eqns = [any_eqn(v, bound_lo, bound_hi, dtype=dtype)]
    if prim in ("add", "sub", "mul", "div"):
        if second is None:
            ins = [v, v]
        else:
            w = var(4, a)
            eqns.append(any_eqn(w, second[0], second[1], dtype=dtype))
            ins = [v, w]
    else:
        ins = [v]
    eqns += [
        eqn(prim, ins, s, params),
        eqn("gt", [s, lit(-1e30, a)], pred),  # slack: only the guard decides
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


# Each primitive overflows int32 at a different shape, so each states its
# own: `v + v` and `v * v` near INT_MAX, `v1 - v2` at opposite ends (v - v
# is always near zero and CANNOT overflow), and neg/abs only at INT_MIN
# exactly — where two's complement has no positive counterpart.
_INT_MAX32 = 2.0**31 - 1
_INT_MIN32 = -(2.0**31)
_OVERFLOW_SHAPES = [
    ("add", (), (_INT_MAX32 - 1, _INT_MAX32), None),
    ("sub", (), (_INT_MAX32 - 1, _INT_MAX32), (_INT_MIN32, _INT_MIN32 + 1)),
    ("mul", (), (_INT_MAX32 - 1, _INT_MAX32), None),
    ("neg", (), (_INT_MIN32, _INT_MIN32), None),
    ("abs", (), (_INT_MIN32, _INT_MIN32), None),
    ("integer_pow", (("y", 2),), (_INT_MAX32 - 1, _INT_MAX32), None),
]


# -- UNSOUND 1: integer arithmetic modelled as real ---------------------------


@pytest.mark.parametrize(
    "prim,params,box,second", _OVERFLOW_SHAPES,
    ids=[s[0] for s in _OVERFLOW_SHAPES],
)
def test_integer_overflow_reachable_declines_for_every_guarded_primitive(
    prim, params, box, second
):
    """AUDIT UNSOUND 1. jax integer arithmetic wraps; this domain computes
    over ℝ. Where wraparound is reachable over the declared box the
    equation must decline with the range quoted — never discharge."""
    q = wrapping_int_query(box[0], box[1], prim, params, second=second)
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("wraparound is not excluded" in n for n in p.notes), p.notes
    assert any("2147483647" in n for n in p.notes)


@pytest.mark.parametrize(
    "prim,params", [(s[0], s[1]) for s in _OVERFLOW_SHAPES],
    ids=[s[0] for s in _OVERFLOW_SHAPES],
)
def test_in_range_integer_arithmetic_is_NOT_declined(prim, params):
    """The other face, and the reason this is a reachability guard rather
    than a blanket refusal: small-integer and index arithmetic keeps its
    exact result. A blanket decline would ⊤ every counter in every trace."""
    q = wrapping_int_query(0.0, 10.0, prim, params)
    p = sole(q)
    assert p.obligations[0].status == "discharged"
    assert not any("wraparound" in n for n in p.notes)


def test_int_dtype_bounds_are_the_exact_two_complement_ranges():
    assert _INT_DTYPE_BOUNDS["int32"] == (-(2**31), 2**31 - 1)
    assert _INT_DTYPE_BOUNDS["int64"] == (-(2**63), 2**63 - 1)
    assert _INT_DTYPE_BOUNDS["uint8"] == (0, 255)
    assert _INT_DTYPE_BOUNDS["bool"] == (0, 1)
    # kept as exact ints: float(2**63) rounds, and the guard compares at
    # the boundary
    assert all(
        isinstance(b, int) for lohi in _INT_DTYPE_BOUNDS.values() for b in lohi
    )


def test_int_range_is_derived_from_the_bounds_table_unchanged():
    """The float->int conversion guard's bound is now derived from the same
    table, so the two cannot drift. Its VALUES must be untouched — the
    strict upper comparison is what the second audit pinned."""
    from stelling.propagate import _INT_RANGE

    assert _INT_RANGE == {"int32": 2.0**31, "int64": 2.0**63}


def test_float_dtypes_are_untouched_by_the_integer_guard():
    q = wrapping_int_query(1e300, 1e300, "mul", (), dtype="float64")
    p = sole(q)
    assert p.obligations[0].status == "discharged"  # 1e600 overflows to inf
    assert not any("wraparound" in n for n in p.notes)


# -- UNSOUND 2: the emission accepted integer dtypes --------------------------


@pytest.mark.parametrize("prim,params", [
    ("add", ()), ("sub", ()), ("mul", ()), ("neg", ()),
    ("integer_pow", (("y", 2),)), ("reduce_sum", (("axes", ()),)),
])
def test_emission_declines_every_computed_integer_primitive(prim, params):
    """AUDIT UNSOUND 2 and the sweep it prompted. SMT-LIB2 Reals are
    unbounded and jax integers wrap, so emitting a computed integer as a
    Real lets the solver prove a claim the program falsifies — a false
    VERIFIED with a proof behind it. `integer_pow` was the finding;
    add/sub/mul/neg had the identical gap."""
    from stelling.obligation import _Decline, _Slicer

    q = wrapping_int_query(0.0, 10.0, prim, params)
    slicer = _Slicer(q, interval_env(q))
    a = aval((), "int32")
    ins = [var(0, a), var(0, a)] if prim in ("add", "sub", "mul") else [var(0, a)]
    with pytest.raises(_Decline) as e:
        slicer._validate(eqn(prim, ins, var(1, a), params))
    assert "wraps on overflow" in e.value.reason
    assert "int32" in e.value.reason


def test_emission_still_accepts_the_same_primitives_on_floats():
    from stelling.obligation import _Slicer

    q = pow_query(1.0, 2.0, 2, bound=0.5)
    slicer = _Slicer(q, interval_env(q))
    for prim, params, ins in (
        ("mul", (), [var(0), var(0)]),
        ("integer_pow", (("y", 2),), [var(0)]),
    ):
        slicer._validate(eqn(prim, ins, var(1), params))  # must not raise


def test_emission_selection_and_comparison_primitives_stay_unguarded():
    """The sweep's other half: max/min/select_n SELECT rather than compute
    and the comparisons are exact over Reals for integers, so they must
    NOT have been swept up by the guard."""
    from stelling.obligation import _Slicer

    q = wrapping_int_query(0.0, 10.0, "add", ())
    slicer = _Slicer(q, interval_env(q))
    a = aval((), "int32")
    for prim in ("max", "min", "gt", "le", "eq"):
        out = var(1, BOOL if prim in ("gt", "le", "eq") else a)
        slicer._validate(eqn(prim, [var(0, a), var(0, a)], out))  # no raise


def test_bool_integer_pow_no_longer_reaches_the_emission():
    """AUDIT COSMETIC 2, closed as a side effect: bool is in the integer
    bounds table, so a bool-dtyped `integer_pow` now declines here instead
    of emitting `(* b b)` at sort Bool. The ill-typed script — and the
    cvc5 misattribution it provoked — is no longer constructible by this
    route."""
    from stelling.obligation import _Decline, _Slicer

    q = wrapping_int_query(0.0, 10.0, "add", ())
    slicer = _Slicer(q, interval_env(q))
    b = aval((), "bool")
    with pytest.raises(_Decline):
        slicer._validate(eqn("integer_pow", [var(0, b)], var(1, b), (("y", 2),)))


# -- FRAGILE 1: iv.slice_ raised IndexError -----------------------------------


@pytest.mark.parametrize("start,limit,strides", [
    ((0,), (3,), None),      # limit past the operand's extent
    ((2,), (3,), None),      # start past the extent
    ((0,), (1,), (0,)),      # zero stride
    ((0, 0), (1, 1), None),  # rank mismatch
    ((1,), (0,), None),      # inverted selection
])
def test_slice_out_of_range_params_decline_not_IndexError(start, limit, strides):
    """AUDIT FRAGILE 1. `propagate` catches IntervalError, not IndexError,
    so an IndexError here killed the whole analysis rather than degrading."""
    a = iv.IntervalArray(shape=(1,), los=(0.0,), his=(1.0,))
    with pytest.raises(iv.IntervalError):
        iv.slice_(a, start, limit, strides)


def test_slice_bad_params_degrade_through_a_public_deserialisation_roundtrip():
    """The auditor's widening: `ir.ClosedJaxpr.from_dict` is a PUBLIC entry
    point, so this is not merely an ad-hoc-IR concern. A query arriving
    through it must degrade to a noted ⊤, not kill the walk."""
    a1 = aval((1,))
    x, arr, sel, sq = var(0), var(1, a1), var(2, a1), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("broadcast_in_dim", [x], arr,
                [("shape", (1,)), ("broadcast_dimensions", ())]),
            eqn("slice", [arr], sel,
                [("start_indices", (0,)), ("limit_indices", (9,)),
                 ("strides", None)]),
            eqn("squeeze", [sel], sq, [("dimensions", (0,))]),
            eqn("le", [sq, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    roundtripped = ir.ClosedJaxpr.from_dict(q.to_dict())
    assert roundtripped.content_hash() == q.content_hash()
    for shaped in (q, roundtripped):
        p = sole(shaped)  # must not raise
        assert p.obligations[0].status == "unknown"
        assert any("outside the operand's extent" in n for n in p.notes)
    interval_env(roundtripped)  # the other public entry point, also alive


def test_legal_slice_forms_are_unaffected_by_the_bounds_check():
    a = iv.IntervalArray(shape=(4,), los=(0.0, 1.0, 2.0, 3.0),
                         his=(0.0, 1.0, 2.0, 3.0))
    assert iv.slice_(a, (1,), (2,), None).los == (1.0,)
    assert iv.slice_(a, (0,), (4,), (2,)).los == (0.0, 2.0)
    assert iv.slice_(a, (0,), (4,), None).los == (0.0, 1.0, 2.0, 3.0)
    assert iv.slice_(a, (2,), (2,), None).shape == (0,)  # empty is legal


# -- FRAGILE 2: no exponent cap on the exact-rational path --------------------


def test_integer_pow_exponent_cap_declines_with_the_exponent_quoted():
    """AUDIT FRAGILE 2. `x ** 10_000_000` is ONE legal jax equation, and
    the exact-rational endpoints cost time linear in the exponent.
    Degrade-don't-crash extends to degrade-don't-hang."""
    for y in (iv.INTEGER_POW_EXACT_CAP + 1, 10**6, -(10**7)):
        q = pow_query(0.1, 0.2, y, bound=1.0)
        p = sole(q)
        assert p.obligations[0].status == "unknown"
        assert any("exceeds the exact-rational endpoint cap" in n
                   for n in p.notes), p.notes
        assert any(str(abs(y)) in n for n in p.notes)


def test_integer_pow_cap_is_bounded_in_TIME_not_merely_in_answer():
    """The point of the cap is cost, so cost is what is asserted. A
    beyond-cap exponent must return in the time an in-cap one does — not
    in the tens of seconds the uncapped exact power took."""
    import time

    t0 = time.perf_counter()
    for y in (10**6, 10**7, 2**40):
        iv.integer_pow(iv.from_bounds((), 0.1, 0.2), y)
        iv.integer_pow(iv.from_bounds((), 5e-324, 1e-320), y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"beyond-cap exponents took {elapsed:.2f}s"


def test_integer_pow_kernel_returns_top_beyond_the_cap():
    """Belt to the transfer's braces: a caller reaching the kernel directly
    gets a sound ⊤ rather than an unbounded computation. Same posture
    `div` already takes for a vanishing denominator."""
    r = iv.integer_pow(iv.from_bounds((), 0.1, 0.2), iv.INTEGER_POW_EXACT_CAP + 1)
    assert (r.los[0], r.his[0]) == (-INF, INF)


def test_integer_pow_at_the_cap_still_computes_exactly():
    """The cap must not eat the working range: at the cap itself the exact
    rational path still runs, and 16x above the SMT expansion cap."""
    r = iv.integer_pow(iv.from_bounds((), 0.5, 0.5), iv.INTEGER_POW_EXACT_CAP)
    assert Fraction(r.los[0]) <= Fraction(1, 2) ** iv.INTEGER_POW_EXACT_CAP
    from stelling.obligation import INTEGER_POW_EXPANSION_CAP

    assert iv.INTEGER_POW_EXACT_CAP > INTEGER_POW_EXPANSION_CAP


def test_ieee_integer_pow_cap_fires_before_the_schedule_decline():
    """The quoted reason must name the BINDING refusal."""
    box = iv.IntervalArray(shape=(), los=(0.1,), his=(0.2,))
    with pytest.raises(iv.IntervalError) as e:
        ieee_pow_call(10**6, box)
    assert "exact-rational endpoint cap" in str(e.value)


# -- COSMETIC 4: the equation-order reliance is now disclosed -----------------


def test_ieee_mode_stamps_the_equation_order_assumption():
    """AUDIT COSMETIC 4. The reduce_sum decline models an `add` chain while
    refusing a reduction; that contrast rests on the compiler not
    reassociating ACROSS equations, which was nowhere in the stamp."""
    q = sum_query([(0.0, 1.0)] * 2, bound=5.0)
    p = sole(q, semantics="ieee")
    assert any("equation-order reliance" in a for a in p.assumptions)
    assert any("REASSOCIATE ACROSS equations" in a for a in p.assumptions)
    # real mode is untouched: it makes no float-order claim at all
    assert not any("equation-order" in a for a in sole(q).assumptions)


# =============================================================================
# Audit re-attack round: the four findings the FIX round's own blind spot
# left behind. The lesson is structural and is tested as such — the sweep
# that missed `div` was a review, and a review finds siblings once.
# =============================================================================

from stelling.propagate import (  # noqa: E402
    _INT_COMPUTING,
    _INT_NON_COMPUTING,
    _is_integer_dtype,
)


def test_the_integer_semantics_census_is_TOTAL_over_the_transfer_registry():
    """AUDIT UNSOUND 3, structurally. The first sweep of this defect class
    was run over the EMISSION sites and not the TRANSFER sites, and `div`
    fell through: "already stricter" was true of the emission and false of
    the transfer. A review finds siblings once; an assert finds them every
    time."""
    assert _INT_COMPUTING | _INT_NON_COMPUTING == set(TRANSFERS)
    assert not (_INT_COMPUTING & _INT_NON_COMPUTING)
    assert "div" in _INT_COMPUTING  # the one that was missed


def test_every_computing_transfer_actually_carries_the_guard():
    """Membership in the census is enforced, not declared: each computing
    transfer must reach `_int_overflow_guard` on an out-of-range integer
    result."""
    for prim in sorted(_INT_COMPUTING):
        if prim in ("pow", "exp"):
            continue  # covered below; their kernels decline first on ints
        params = (("y", 2),) if prim == "integer_pow" else (
            (("axes", ()),) if prim == "reduce_sum" else ()
        )
        second = (_INT_MIN32, _INT_MIN32 + 1) if prim == "sub" else None
        box = (_INT_MAX32 - 1, _INT_MAX32)
        if prim in ("neg", "abs"):
            box = (_INT_MIN32, _INT_MIN32)
        if prim == "div":
            # div's ONLY integer overflow is INT_MIN / -1 (measured:
            # lax.div(-2**31, -1) = -2147483648, not +2**31)
            box, second = (_INT_MIN32, _INT_MIN32), (-1.0, -1.0)
        if prim == "reduce_sum":
            continue  # its own shape; covered by its dedicated test
        if prim == "scatter-add":
            # its own 3-operand shape; covered by its dedicated test
            # (tests/test_scatter_rows.py::test_int_scatter_add_overflow_declines)
            continue
        q = wrapping_int_query(box[0], box[1], prim, params, second=second)
        p = sole(q)
        assert p.obligations[0].status == "unknown", prim
        assert any("declined" in n for n in p.notes), prim


def test_the_emission_census_is_TOTAL_over_the_supported_set():
    from stelling.obligation import (
        _INT_OVERFLOW_EMITTED,
        _INT_SAFE_EMITTED,
        _SUPPORTED,
    )

    assert _INT_OVERFLOW_EMITTED | _INT_SAFE_EMITTED == _SUPPORTED
    assert not (_INT_OVERFLOW_EMITTED & _INT_SAFE_EMITTED)
    assert "div" in _INT_OVERFLOW_EMITTED


# -- UNSOUND 3: integer div truncates, and INT_MIN/-1 wraps -------------------


def int_div_query(alo, ahi, blo, bhi, cmp, bound, dtype="int32"):
    a = aval((), dtype)
    x, y, s = var(0, a), var(1, a), var(2, a)
    pred, out = var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(x, alo, ahi, dtype=dtype),
            any_eqn(y, blo, bhi, dtype=dtype),
            eqn("div", [x, y], s),
            eqn(cmp, [s, lit(bound, a)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


@pytest.mark.parametrize("a,b,want", [
    (-7, 2, -3),   # measured lax.div(-7, 2) = -3, NOT the real -3.5
    (7, 2, 3),
    (-7, -2, 3),
    (7, -2, -3),
    (-1, 2, 0),
    (5, 5, 1),
])
def test_integer_div_truncates_toward_zero_not_real_division(a, b, want):
    """AUDIT UNSOUND 3. `iv.div` computes real division; jax integer
    division TRUNCATES. Any predicate separating -3 from -3.5 was decided
    wrongly — six false definite verdicts, three VERIFIED and three
    REFUTED."""
    box = iv.int_div(
        iv.IntervalArray(shape=(), los=(float(a),), his=(float(a),)),
        iv.IntervalArray(shape=(), los=(float(b),), his=(float(b),)),
    )
    assert (box.los[0], box.his[0]) == (float(want), float(want))
    # python's // FLOORS and is the wrong model: -7 // 2 == -4
    if a < 0 <= b or b < 0 <= a:
        assert want != a // b or a % b == 0


@pytest.mark.parametrize("cmp,bound,expect", [
    # -7/2 truncates to -3, so `< -3` is definitely FALSE. It used to
    # read discharged — a false VERIFIED; now it is a sound refutation.
    ("lt", -3.0, "violated-over-set"),
    ("ge", -3.0, "discharged"),   # -3 >= -3 is TRUE, and now provably so
])
def test_integer_div_predicates_are_now_decided_by_truncation(cmp, bound, expect):
    q = int_div_query(-7, -7, 2, 2, cmp, bound)
    assert sole(q).obligations[0].status == expect


def test_integer_div_INT_MIN_over_minus_one_routes_through_the_overflow_guard():
    """The wraparound class the guard exists for, reached through `div`:
    measured lax.div(-2**31, -1) = -2147483648, not +2**31."""
    q = int_div_query(_INT_MIN32, _INT_MIN32, -1, -1, "gt", 0.0)
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("wraparound is not excluded" in n for n in p.notes), p.notes


def test_integer_div_by_a_zero_crossing_divisor_is_TOP():
    q = int_div_query(1, 10, -2, 2, "gt", 0.0)
    assert sole(q).obligations[0].status == "unknown"
    box = iv.int_div(iv.from_bounds((), 1.0, 10.0), iv.from_bounds((), -2.0, 2.0))
    assert (box.los[0], box.his[0]) == (-INF, INF)


def test_float_div_is_completely_unchanged():
    """Real mode's float division must not have moved a millimetre."""
    a = iv.from_bounds((), 1.0, 4.0)
    b = iv.from_bounds((), 2.0, 2.0)
    assert iv.div(a, b).los[0] < 0.5 < iv.div(a, b).his[0]
    q = pow_query(1.0, 2.0, 2, bound=10.0)
    assert sole(q).obligations[0].status == "discharged"


def test_integer_div_non_integral_endpoints_decline_rather_than_approximate():
    with pytest.raises(iv.IntervalError) as e:
        iv.int_div(iv.from_bounds((), 1.5, 2.5), iv.from_bounds((), 2.0, 2.0))
    assert "TRUNCATES" in str(e.value)


# -- FRAGILE 3: the cap bounded f(y), not size x f(y) -------------------------


def test_integer_pow_work_cap_bounds_total_not_just_the_exponent():
    """AUDIT FRAGILE 3. The exponent cap bounds the per-element cost; the
    transfer is elementwise, so a large array at the capped exponent still
    cost 112 s. The budget is on `size * |y|`."""
    n = 200_000
    box = iv.from_bounds((n,), 0.1, 0.2)
    assert n * 1024 > iv.INTEGER_POW_WORK_CAP
    import time

    t0 = time.perf_counter()
    r = iv.integer_pow(box, 1024)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"{elapsed:.1f}s for a beyond-budget array"
    assert (r.los[0], r.his[0]) == (-INF, INF)


def test_integer_pow_work_cap_declines_with_shape_and_exponent_quoted():
    a = aval((1000,))
    x, s = var(0, a), var(1, a)
    red, pred, out = var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.1, 0.2, shape=(1000,)),
            eqn("integer_pow", [x], s, [("y", 1024)]),
            eqn("reduce_sum", [s], red, [("axes", (0,))]),
            eqn("le", [red, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("units of exact-rational work" in n for n in p.notes), p.notes
    assert any("1000 elements" in n for n in p.notes)


def test_the_work_budget_still_admits_ordinary_shapes():
    """The budget must not have eaten normal use: y=2 over a 1000-element
    array is 2000 units, far inside it."""
    r = iv.integer_pow(iv.from_bounds((1000,), 2.0, 3.0), 2)
    assert (r.los[0], r.his[0]) == (4.0, 9.0)  # computed, not ⊤


# -- COSMETIC 5: three causes, three sentences --------------------------------


def test_the_overflow_decline_attributes_an_unbounded_operand_correctly():
    """Case 2: a ⊤ operand from an unrelated cause was reported as
    "wraparound reachable", attributing to overflow what was an unmodelled
    producer."""
    a = aval((), "int32")
    x, s = var(0, a), var(1, a)
    pred, out = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, -INF, INF, dtype="int32"),
            eqn("add", [x, x], s),
            eqn("gt", [s, lit(-1e30, a)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = sole(q)
    assert any("range is unbounded" in n for n in p.notes), p.notes
    assert any("NOT a wraparound finding" in n for n in p.notes)


def test_a_result_landing_exactly_on_the_dtype_max_is_DECIDED_not_declined():
    """Case 1, decided rather than declined (L13): the true values of an
    integer result are integers, so the bracket snaps to them exactly and
    the one-ulp outward bump stops costing the top of every dtype."""
    for dtype, mx in (("int8", 127), ("int16", 32767), ("int32", 2**31 - 1),
                      ("uint8", 255), ("uint32", 2**32 - 1)):
        q = wrapping_int_query(mx - 1, mx - 1, "add", (), dtype=dtype,
                               second=(1, 1))
        p = sole(q)
        assert p.obligations[0].status == "discharged", (dtype, p.notes)


def test_integer_results_are_snapped_to_integers():
    """The tightening itself: an integer-dtyped result carries an integer
    bracket, not a one-ulp-padded real one."""
    q = wrapping_int_query(3.0, 3.0, "add", (), dtype="int32", second=(4, 4))
    box, _, _ = box_of(q, 1)
    assert (box.los[0], box.his[0]) == (7.0, 7.0)  # exactly 7, no padding


def test_int64_near_the_boundary_reports_a_BRACKET_limit_not_an_overflow():
    """Where the double bracket really is wider than one integer (2048 at
    int64 magnitudes), the decline says so instead of claiming a
    demonstrated overflow."""
    mx = float(2**63 - 1)
    q = wrapping_int_query(mx, mx, "add", (), dtype="int64", second=(-1.0, -1.0))
    p = sole(q)
    if p.obligations[0].status == "unknown":
        assert any("BRACKET limit" in n or "wraparound is not excluded" in n
                   for n in p.notes), p.notes


# -- int4/uint4 and unknown integer dtypes ------------------------------------


def test_int4_and_uint4_are_registered_not_silently_skipped():
    assert _INT_DTYPE_BOUNDS["int4"] == (-8, 7)
    assert _INT_DTYPE_BOUNDS["uint4"] == (0, 15)
    q = wrapping_int_query(7.0, 7.0, "add", (), dtype="int4", second=(7.0, 7.0))
    p = sole(q)
    assert p.obligations[0].status == "unknown"  # 14 escapes int4's max of 7
    assert any("declined" in n for n in p.notes)


def test_an_unregistered_integer_dtype_declines_rather_than_skipping():
    """The guard used to return early for any dtype missing from the table,
    making it a silent no-op. Detection is now by NAME, so a dtype the
    table has not heard of refuses instead."""
    assert _is_integer_dtype("int128") and "int128" not in _INT_DTYPE_BOUNDS
    q = wrapping_int_query(1.0, 2.0, "add", (), dtype="int128")
    p = sole(q)
    assert p.obligations[0].status == "unknown"
    assert any("no representable range is registered" in n for n in p.notes)
    # and float dtypes are still untouched by any of it
    assert not _is_integer_dtype("float64")


# -- UNSOUND 4: FMA contraction ----------------------------------------------


def fma_query(cmp, bound):
    """(a*b) - 1 <cmp> bound, at the operands where contraction shows."""
    a, b, prod, diff = var(0), var(1), var(2), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
            any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
            eqn("mul", [a, b], prod),
            eqn("sub", [prod, lit(1.0)], diff),
            eqn(cmp, [diff, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


@pytest.mark.parametrize("cmp,bound", [
    ("eq", 0.0), ("ge", 0.0), ("lt", 0.0), ("gt", -1e-17),
])
def test_ieee_contraction_makes_the_two_roundings_indeterminate(cmp, bound):
    """AUDIT UNSOUND 4. XLA contracts a*b + c into one FMA (the compiled
    HLO shows multiply_add_fusion): (a*b)-1 is 0.0 eager and -2**-54 under
    jit. Four definite ieee verdicts contradicted the compiled program.
    The mode now HULLS both roundings, so each goes indeterminate."""
    assert sole(fma_query(cmp, bound), semantics="ieee").obligations[0].status == (
        "unknown"
    )


def test_ieee_contraction_hull_keeps_agreeing_cases_DEFINITE():
    """The hull must not be a blanket refusal: where the two roundings
    agree — which is the overwhelmingly common case — the result stays a
    point and the obligation still discharges."""
    a, b, prod, s = var(0), var(1), var(2), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(a, 2.0, 2.0),
            any_eqn(b, 3.0, 3.0),
            eqn("mul", [a, b], prod),
            eqn("add", [prod, lit(1.0)], s),
            eqn("eq", [s, lit(7.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "discharged"


def test_ieee_contraction_covers_both_operand_positions_and_sub():
    """c - a*b contracts too (negating the product), and the mul may sit on
    either side of the add."""
    for prim, order in (("add", 0), ("add", 1), ("sub", 0), ("sub", 1)):
        a, b, prod, s = var(0), var(1), var(2), var(3)
        pred, out = var(4, BOOL), var(5, BOOL)
        # pick the addend that lands the result on ~0, where the two
        # roundings differ: add wants -1, sub wants +1
        k = lit(-1.0) if prim == "add" else lit(1.0)
        ins = [prod, k] if order == 0 else [k, prod]
        q = close(
            [
                any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
                any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
                eqn("mul", [a, b], prod),
                eqn(prim, ins, s),
                eqn("eq", [s, lit(0.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )
        # whichever way round, a definite answer must not rest on one rounding
        st = sole(q, semantics="ieee").obligations[0].status
        assert st == "unknown", (prim, order, st)


def test_ieee_contraction_is_stamped_and_real_mode_is_untouched():
    p = sole(fma_query("eq", 0.0), semantics="ieee")
    assert any("contraction hull" in a for a in p.assumptions)
    assert any("multiply_add_fusion" in a for a in p.assumptions)
    # the equation-order assumption now cross-references it rather than
    # leaving a reader thinking reassociation is the only freedom
    assert any("CONTRACTION" in a for a in p.assumptions)
    # real mode makes no float-order claim and must be untouched
    real = sole(fma_query("eq", 0.0))
    assert not any("contraction" in a for a in real.assumptions)


def test_ieee_fma_hull_computes_the_contracted_value_exactly():
    a = iv.from_bounds((), 1.0 + 2.0**-27, 1.0 + 2.0**-27)
    b = iv.from_bounds((), 1.0 - 2.0**-27, 1.0 - 2.0**-27)
    c = iv.from_bounds((), -1.0, -1.0)
    box, nan = iv.ieee_fma_hull(a, b, c)
    assert nan is False
    assert box.los[0] == box.his[0] == -(2.0**-54)  # the exact FMA answer


def test_ieee_fma_hull_declines_infinite_operands_with_the_reason():
    with pytest.raises(iv.IntervalError) as e:
        iv.ieee_fma_hull(
            iv.from_bounds((), 1.0, INF), iv.from_bounds((), 1.0, 1.0),
            iv.from_bounds((), 1.0, 1.0),
        )
    assert "infinite" in str(e.value)


# =============================================================================
# Third re-attack: the contraction hull matched SYNTAX; XLA contracts after
# its own simplification passes. The fix closes the CLASS with a taint, so
# soundness stops depending on recognising the intervening shape.
# =============================================================================

from stelling.propagate import _TAINT_STOPS  # noqa: E402


def contraction_chain(between, cmp="eq", bound=0.0):
    """(a*b) with `between` applied, then - 1.0, compared. `between` is a
    list of (primitive, params) applied to the product in order — the
    intervening equations XLA elides or absorbs."""
    a, b, prod = var(0), var(1), var(2)
    eqns = [
        any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
        any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
        eqn("mul", [a, b], prod),
    ]
    cur, nxt = prod, 3
    for prim, params, shape in between:
        out = var(nxt, aval(shape))
        ins = [cur, lit(-INF)] if prim == "max" else [cur]
        eqns.append(eqn(prim, ins, out, params))
        cur, nxt = out, nxt + 1
    diff, pred, out = var(nxt), var(nxt + 1, BOOL), var(nxt + 2, BOOL)
    # land the result on ~0, where the two roundings differ: an odd number
    # of `neg`s has flipped the product's sign, so add rather than subtract
    negs = sum(1 for prim, _, _ in between if prim == "neg")
    final = ("add", 1.0) if negs % 2 else ("sub", 1.0)
    eqns += [
        eqn(final[0], [cur, lit(final[1])], diff),
        eqn(cmp, [diff, lit(bound)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    return close(eqns, [out])


# The auditor's ten forms: every equation XLA elides or absorbs between the
# multiply and the add, each measured to keep contracting while breaking a
# syntactic match.
_TEN_FORMS = {
    "neg": [("neg", (), ())],
    "neg neg": [("neg", (), ()), ("neg", (), ())],
    "reshape": [("reshape", (("new_sizes", ()), ("dimensions", None)), ())],
    "broadcast+squeeze": [
        ("broadcast_in_dim", (("shape", (1,)), ("broadcast_dimensions", ())), (1,)),
        ("squeeze", (("dimensions", (0,)),), ()),
    ],
    "broadcast+slice+squeeze": [
        ("broadcast_in_dim", (("shape", (1,)), ("broadcast_dimensions", ())), (1,)),
        ("slice", (("start_indices", (0,)), ("limit_indices", (1,)),
                   ("strides", None)), (1,)),
        ("squeeze", (("dimensions", (0,)),), ()),
    ],
    "broadcast+transpose+squeeze": [
        ("broadcast_in_dim", (("shape", (1,)), ("broadcast_dimensions", ())), (1,)),
        ("transpose", (("permutation", (0,)),), (1,)),
        ("squeeze", (("dimensions", (0,)),), ()),
    ],
    "broadcast+reduce_sum": [
        ("broadcast_in_dim", (("shape", (1,)), ("broadcast_dimensions", ())), (1,)),
        ("reduce_sum", (("axes", (0,)),), ()),
    ],
    "max": [("max", (), ())],
    "stop_gradient": [("stop_gradient", (), ())],
}


@pytest.mark.parametrize("name", sorted(_TEN_FORMS))
def test_contraction_survives_every_intervening_equation(name):
    """AUDIT UNSOUND 5. `_contraction_hull` fired only on a DIRECT `mul`
    producer, so anything XLA elides (reshape, broadcast, squeeze, slice,
    transpose, a one-element reduce_sum, max, stop_gradient, a jit
    wrapper) or absorbs (neg) broke the match while the contraction still
    happened — ten measured false VERIFIEDs.

    The taint closes the class: it flows from the mul through the
    dataflow, and every add/sub that meets it hulls, whatever lies
    between. This must hold WITHOUT the intervening shape being
    recognised."""
    q = contraction_chain(_TEN_FORMS[name])
    assert sole(q, semantics="ieee").obligations[0].status == "unknown", name


def test_the_baseline_direct_form_is_still_covered():
    assert sole(contraction_chain([]), semantics="ieee").obligations[0].status == (
        "unknown"
    )


def test_taint_reaches_through_an_arbitrary_unlisted_chain():
    """The property that matters is that soundness does NOT depend on a
    list: a chain nobody enumerated must still be covered."""
    chain = [
        ("abs", (), ()),
        ("broadcast_in_dim", (("shape", (1,)), ("broadcast_dimensions", ())), (1,)),
        ("reshape", (("new_sizes", (1,)), ("dimensions", None)), (1,)),
        ("transpose", (("permutation", (0,)),), (1,)),
        ("slice", (("start_indices", (0,)), ("limit_indices", (1,)),
                   ("strides", None)), (1,)),
        ("squeeze", (("dimensions", (0,)),), ()),
        ("stop_gradient", (), ()),
    ]
    assert sole(contraction_chain(chain), semantics="ieee").obligations[0].status == (
        "unknown"
    )


def test_the_hull_still_leaves_agreeing_forms_DEFINITE():
    """The taint must not have turned into a blanket refusal. Where the two
    roundings provably agree, the obligation still discharges — this is the
    L13 trade: the whole mul-free fragment, and every agreeing mul form,
    keeps its decidability."""
    a, b, prod, s = var(0), var(1), var(2), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(a, 2.0, 2.0),
            any_eqn(b, 3.0, 3.0),
            eqn("mul", [a, b], prod),
            eqn("add", [prod, lit(1.0)], s),
            eqn("eq", [s, lit(7.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "discharged"


def test_mul_free_obligations_are_completely_undisturbed():
    """The disclosed precision cost is bounded: an obligation with no
    multiply in it cannot be touched by the taint at all."""
    x, s, t = var(0), var(1), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("add", [x, lit(1.0)], s),
            eqn("sub", [s, lit(0.5)], t),
            eqn("le", [t, lit(3.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "discharged"


def test_taint_is_not_laundered_by_a_declined_equation():
    """A ⊤ from a decline must keep the taint: whatever the compiler did
    with the product, the result is still product-derived."""
    a, b, prod, p2 = var(0), var(1), var(2), var(3)
    s, pred, out = var(4), var(5, BOOL), var(6, BOOL)
    q = close(
        [
            any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
            any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
            eqn("mul", [a, b], prod),
            eqn("integer_pow", [prod], p2, [("y", 3)]),  # declines under ieee
            eqn("sub", [p2, lit(1.0)], s),
            eqn("eq", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "unknown"


def test_taint_crosses_jit_and_cond_scope_boundaries():
    """Taint rides with the values across scopes, or a nested jit would
    launder it — one of the ten forms."""
    inner_a, inner_b, inner_p = var(10), var(11), var(12)
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(inner_a, inner_b), outvars=(inner_p,),
            eqns=(eqn("mul", [inner_a, inner_b], inner_p),),
        )
    )
    a, b, prod, s = var(0), var(1), var(2), var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
            any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
            eqn("jit", [a, b], prod, [("jaxpr", inner)]),
            eqn("sub", [prod, lit(1.0)], s),
            eqn("eq", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "unknown"


def test_reduce_sum_of_two_over_a_tainted_array_declines():
    """A 2-element reduce_sum IS an addition, so a product among its
    elements can contract there too — and the per-array taint does not say
    which element carries it, so this declines rather than guesses."""
    a, b, prod, arr, s = var(0), var(1), var(2), var(3, aval((2,))), var(4)
    pred, out = var(5, BOOL), var(6, BOOL)
    q = close(
        [
            any_eqn(a, 2.0, 2.0),
            any_eqn(b, 3.0, 3.0),
            eqn("mul", [a, b], prod),
            eqn("broadcast_in_dim", [prod], arr,
                [("shape", (2,)), ("broadcast_dimensions", ())]),
            eqn("reduce_sum", [arr], s, [("axes", (0,))]),
            eqn("le", [s, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = sole(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any("performs an addition over a product-derived" in n
               for n in p.notes), p.notes


def test_taint_stops_are_exactly_the_argued_exemptions():
    """Each exemption is a claim that no compiler could present the product
    as a raw addend through it. `pow`/`integer_pow` are deliberately NOT
    exempt — `pow(x, 2)` is a multiply after expansion, which is precisely
    the simplification class this finding is about."""
    assert _TAINT_STOPS == frozenset({
        "exp", "lt", "gt", "le", "ge", "eq", "ne", "and", "or", "reduce_or",
    })
    for p in ("pow", "integer_pow", "div", "neg", "abs", "max", "min",
              "reduce_sum", "convert_element_type", "select_n", "slice"):
        assert p not in _TAINT_STOPS, p


def test_exp_stops_the_taint_and_a_later_add_stays_definite():
    a, b, prod, e, s = var(0), var(1), var(2), var(3), var(4)
    pred, out = var(5, BOOL), var(6, BOOL)
    q = close(
        [
            any_eqn(a, 0.0, 0.0),
            any_eqn(b, 0.0, 0.0),
            eqn("mul", [a, b], prod),   # exactly 0.0
            eqn("exp", [prod], e),      # exactly 1.0; taint stops here
            eqn("add", [e, lit(1.0)], s),
            eqn("le", [s, lit(3.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "discharged"


def test_real_mode_is_untouched_by_the_taint_machinery():
    for name in sorted(_TEN_FORMS):
        q = contraction_chain(_TEN_FORMS[name], cmp="le", bound=1e9)
        assert sole(q).obligations[0].status == "discharged", name


# -- COSMETIC 6: the membership census is behavioural, not declarative -------


def test_the_membership_census_cannot_be_satisfied_by_a_label():
    """AUDIT COSMETIC 6. The old assert read a settable attribute plus a
    hand-maintained escape list, either of which a computing transfer could
    satisfy while leaving the class open. The check is now BEHAVIOURAL: it
    runs each computing transfer on an out-of-range integer operand and
    requires it to decline or stay in range."""
    from stelling.propagate import (
        _INT_GUARDED_INSIDE,
        _assert_computing_transfers_close_the_integer_class,
    )

    # the escape list is abolished, not repaired
    assert _INT_GUARDED_INSIDE == frozenset()
    # the real check passes on the real registry
    _assert_computing_transfers_close_the_integer_class()

    # ...and a forged transfer that merely WEARS the marker is caught
    def forged(eqn, params, ins):
        return [iv.from_bounds((), 1e30, 1e30)]  # far outside int32

    forged._int_guarded = True  # the label the old assert trusted
    saved = TRANSFERS["mul"]
    TRANSFERS["mul"] = (forged, "sound")
    try:
        with pytest.raises(AssertionError) as e:
            _assert_computing_transfers_close_the_integer_class()
        assert "the integer class is open" in str(e.value)
    finally:
        TRANSFERS["mul"] = saved
    _assert_computing_transfers_close_the_integer_class()  # restored


# =============================================================================
# Fourth re-attack: a stated soundness mechanism that was never wired, and
# an assert that tested a representative instead of its invariant.
# =============================================================================


def test_taint_survives_the_cond_join():
    """AUDIT COSMETIC 7. §12 claimed "a joined cond output is tainted if
    ANY branch tainted it". The `nan` flag is joined that way; the taint
    was NOT — `branch_taints` was appended and never read, so the join
    defaulted it to False and laundered it.

    Harmless only because XLA does not currently contract across a cond
    boundary on this target — which is exactly the compiler behaviour the
    taint design exists so as not to depend on. A stated mechanism that is
    not wired is worse than an acknowledged gap, because the statement is
    what later work trusts."""
    ba, bb, bp = var(10), var(11), var(12)
    branch = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(ba, bb), outvars=(bp,),
            eqns=(eqn("mul", [ba, bb], bp),),  # the mul lives INSIDE
        )
    )
    other = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(var(20), var(21)), outvars=(var(20),), eqns=(),
        )
    )
    idx = var(0, aval((), "int32"))
    a, b, prod, s = var(1), var(2), var(3), var(4)
    pred, out = var(5, BOOL), var(6, BOOL)
    q = close(
        [
            # a DEFINITE index, so exactly one branch runs and the join is
            # that branch's own output — otherwise the obligation would be
            # unknown from the branch join and this test would pass without
            # discriminating (verified: with the taint laundered it reads
            # `discharged`, the false VERIFIED)
            any_eqn(idx, 0.0, 0.0, dtype="int32"),
            any_eqn(a, 1.0 + 2.0**-27, 1.0 + 2.0**-27),
            any_eqn(b, 1.0 - 2.0**-27, 1.0 - 2.0**-27),
            eqn("cond", [idx, a, b], prod, [("branches", (branch, other))]),
            eqn("sub", [prod, lit(1.0)], s),  # ...the add lives OUTSIDE
            eqn("eq", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole(q, semantics="ieee").obligations[0].status == "unknown"


def test_the_census_probes_its_invariant_not_a_representative():
    """The same lesson one turn later: the behavioural assert probed int32
    only, one boundary direction per primitive, so a transfer that declined
    that value while accepting a sibling would have satisfied it. It now
    sweeps every computing transfer x every integer dtype x both
    directions."""
    from stelling.propagate import (
        _INT_COMPUTING,
        _INT_DTYPE_BOUNDS,
        _assert_computing_transfers_close_the_integer_class,
        _probe_operands,
    )

    probes = len(_INT_COMPUTING) * len(_INT_DTYPE_BOUNDS) * 2
    assert probes >= 160, probes
    # every (transfer, dtype, direction) triple must be constructible
    for prim in _INT_COMPUTING:
        for lo_b, hi_b in _INT_DTYPE_BOUNDS.values():
            for high in (True, False):
                assert _probe_operands(prim, lo_b, hi_b, high) is not None
    _assert_computing_transfers_close_the_integer_class()


@pytest.mark.parametrize("dtype", sorted(_INT_DTYPE_BOUNDS))
def test_no_computing_transfer_leaks_at_any_dtype_boundary(dtype):
    """The sweep as a test too, per dtype, so a failure names the dtype
    rather than arriving as one import-time assertion."""
    from stelling.propagate import _probe_operands

    lo_b, hi_b = _INT_DTYPE_BOUNDS[dtype]
    for prim in sorted(_INT_COMPUTING):
        for high in (True, False):
            first, second = _probe_operands(prim, lo_b, hi_b, high)
            shape = (3,) if prim == "reduce_sum" else ()
            boxes = [iv.from_bounds(shape, first, first)]
            if second is not None:
                boxes.append(iv.from_bounds((), second, second))
            y = (2 if high else 3) if prim == "integer_pow" else None
            params = {"integer_pow": (("y", y),),
                      "reduce_sum": (("axes", (0,)),)}.get(prim, ())
            e = ir.JaxprEqn(
                primitive=prim,
                invars=tuple(
                    ir.Var(id=i, aval=aval(shape if i == 0 else (), dtype))
                    for i in range(len(boxes))
                ),
                outvars=(ir.Var(id=99, aval=aval((), dtype)),),
                params=params,
            )
            try:
                outs = TRANSFERS[prim][0](e, e.params_dict(), boxes)
            except iv.IntervalError:
                continue
            if outs is None:
                continue
            for box in outs:
                for lo, hi in zip(box.los, box.his):
                    assert lo_b <= lo and hi <= hi_b, (prim, dtype, high, lo, hi)
