# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The interval domain: outward rounding is the soundness-critical surface.

A rounding bug here is a false VERIFIED — the project's own thesis defect —
so these tests check *containment* (the true real result lies inside the
computed bracket), not tightness.
"""

from __future__ import annotations

import math

import pytest

from stelling import interval as iv

INF = math.inf


def scalar(lo, hi):
    return iv.IntervalArray(shape=(), los=(float(lo),), his=(float(hi),))


def test_add_brackets_true_result_exactly():
    """add is exact-when-representable: the endpoints are the two
    representable NEIGHBOURS of the exact real sum, so the bracket is as
    tight as doubles allow.

    The old form of this test also asserted ``lo < fl(0.1)+fl(0.2) < hi`` —
    a bracket strictly around the *rounded double*. That assertion cannot
    survive a tight endpoint rule and should not: round-to-nearest returns
    one of the two neighbours of the exact sum, which are exactly the
    endpoints, so the float always lands ON a bound. It recorded an intent
    that was never achievable as a guarantee, because float containment
    does NOT compose — summing ``[2**53, 1, 1]`` left to right gives
    ``2**53`` while the exact total is ``2**53 + 2``, outside any
    exactly-summed bracket. Real-mode brackets bound REALS; the stamp and
    the module docstring both say so.
    """
    from fractions import Fraction

    r = iv.add(iv.point(0.1), iv.point(0.2))
    true = Fraction(0.1) + Fraction(0.2)
    assert Fraction(r.los[0]) <= true <= Fraction(r.his[0])
    # exactness, not mere containment: adjacent doubles straddling `true`
    assert math.nextafter(r.los[0], math.inf) == r.his[0]
    # and round-to-nearest lands on one of them
    assert (0.1 + 0.2) in (r.los[0], r.his[0])


def test_sub_directed():
    """Directed subtraction — lo = a.lo - b.hi, hi = a.hi - b.lo — and
    EXACT here: -0.5 and 1.5 are both representable, so the
    exact-when-representable rule returns them unwidened."""
    r = iv.sub(scalar(1.0, 2.0), scalar(0.5, 1.5))
    assert r.los[0] == -0.5 and r.his[0] == 1.5


@pytest.mark.parametrize(
    "a, b, contains",
    [
        ((2.0, 3.0), (4.0, 5.0), (8.0, 15.0)),
        ((-2.0, 3.0), (4.0, 5.0), (-10.0, 15.0)),
        ((-2.0, -1.0), (-5.0, -4.0), (4.0, 10.0)),
        ((-2.0, 3.0), (-5.0, 4.0), (-15.0, 12.0)),
    ],
)
def test_mul_sign_cases(a, b, contains):
    r = iv.mul(scalar(*a), scalar(*b))
    assert r.los[0] <= contains[0] and r.his[0] >= contains[1]


def test_mul_zero_times_infinity_is_zero_not_nan():
    # the 0·inf endpoint product follows the closed-interval convention
    # (contributes 0, never NaN); the outward bump may then widen the zero
    # endpoint one ulp below — sound containment, no NaN, no sign surprise
    r = iv.mul(scalar(0.0, 5294.0), scalar(0.078, INF))
    assert -5e-324 <= r.los[0] <= 0.0 and r.his[0] == INF


def test_mul_tiny_negative_times_infinity_widens_soundly():
    r = iv.mul(scalar(-5e-324, 5294.0), scalar(0.078, INF))
    assert r.los[0] == -INF and r.his[0] == INF


def test_div_zero_crossing_denominator_is_top():
    r = iv.div(scalar(1.0, 2.0), scalar(-1.0, 1.0))
    assert (r.los[0], r.his[0]) == (-INF, INF)


def test_div_sound_bracket():
    """[1,2]/[4,8] = [1/8, 1/2] exactly, and both endpoints are doubles, so
    the exact-when-representable rule returns them unwidened. The corners are
    computed as exact rationals, so no corner is inexact and the outward
    rounding never fires here."""
    r = iv.div(scalar(1.0, 2.0), scalar(4.0, 8.0))
    assert r.los[0] == 0.125 and r.his[0] == 0.5


def test_exp_brackets_libm():
    r = iv.exp(scalar(1.0, 2.0))
    assert r.los[0] < math.exp(1.0) and r.his[0] > math.exp(2.0)
    assert r.los[0] > 0.0


def test_exp_overflow_saturates_outward():
    r = iv.exp(scalar(1000.0, 1000.0))
    assert r.his[0] == INF
    assert r.los[0] == math.nextafter(INF, 0.0)  # maxfloat: a sound lower bound


def test_exp_neg_infinity_floor_is_zero():
    r = iv.exp(scalar(-INF, 0.0))
    assert r.los[0] == 0.0 and r.his[0] > 1.0


def test_nan_raises_never_propagates():
    with pytest.raises(iv.IntervalError):
        scalar(math.nan, 1.0)


def test_empty_interval_refused_at_construction():
    # an inverted declared set verifies everything vacuously; the domain
    # refuses it unconditionally, independent of the harness-level check
    with pytest.raises(iv.IntervalError):
        iv.from_bounds((), 5.0, 3.0)


def test_comparisons_three_valued():
    assert iv.lt(scalar(1.0, 2.0), scalar(3.0, 4.0)).los[0] == 1.0  # definite
    assert iv.lt(scalar(3.0, 4.0), scalar(1.0, 2.0)).his[0] == 0.0  # definite no
    r = iv.lt(scalar(1.0, 3.5), scalar(3.0, 4.0))
    assert (r.los[0], r.his[0]) == (0.0, 1.0)  # undecided, never guessed


def test_structural_ops_are_exact():
    """Pure data movement moves BOTH endpoints unchanged.

    Both endpoints are asserted deliberately. The earlier form of this test
    checked ``.los`` only, which meant an upper-endpoint widening in any of
    the four ops would pass a test named "are_exact" — and structural ops
    are exactly where such a widening would be silent, since they perform
    no arithmetic and so have no rounding to justify one.
    """
    a = iv.from_values((2, 2), [1.0, 2.0, 3.0, 4.0])
    s = iv.slice_(a, (0, 1), (2, 2), None)
    assert s.shape == (2, 1) and s.los == (2.0, 4.0) and s.his == (2.0, 4.0)
    q = iv.squeeze(s, (1,))
    assert q.shape == (2,) and q.los == (2.0, 4.0) and q.his == (2.0, 4.0)
    b = iv.broadcast_in_dim(iv.point(7.0), (3,), ())
    assert b.shape == (3,) and b.los == (7.0, 7.0, 7.0) and b.his == (7.0, 7.0, 7.0)
    c = iv.concatenate([iv.from_values((1,), [1.0]), iv.from_values((2,), [2.0, 3.0])], 0)
    assert c.shape == (3,) and c.los == (1.0, 2.0, 3.0) and c.his == (1.0, 2.0, 3.0)
    # non-degenerate boxes too: a point interval cannot distinguish an
    # endpoint that moved from one that was copied from the other side.
    w = iv.IntervalArray(shape=(2,), los=(-1.5, 0.25), his=(2.5, 0.75))
    assert iv.slice_(w, (0,), (1,), None).los == (-1.5,)
    assert iv.slice_(w, (0,), (1,), None).his == (2.5,)
    assert iv.concatenate([w, w], 0).los == (-1.5, 0.25, -1.5, 0.25)
    assert iv.concatenate([w, w], 0).his == (2.5, 0.75, 2.5, 0.75)
    assert iv.broadcast_in_dim(iv.IntervalArray(shape=(), los=(-3.0,), his=(4.0,)),
                               (2,), ()).his == (4.0, 4.0)


def test_scalar_literal_broadcasts_against_array():
    # The broadcast pairing is the subject; the endpoints are EXACT now that
    # `mul` takes the exact-rational route `add` and `div` already had (audit
    # 0.2.0 M16). This read `r.los[0] < 20.0 < r.his[0]` while the transfer
    # bumped every endpoint unconditionally.
    r = iv.mul(iv.from_values((2,), [2.0, 3.0]), iv.point(10.0))
    assert (r.los[0], r.his[0]) == (20.0, 20.0)
    assert (r.los[1], r.his[1]) == (30.0, 30.0)


def test_maximum_minimum_are_exact_monotone():
    m = iv.maximum(scalar(1.0, 3.0), scalar(2.0, 2.0))
    assert (m.los[0], m.his[0]) == (2.0, 3.0)  # max endpointwise
    n = iv.minimum(scalar(1.0, 3.0), scalar(2.0, 2.0))
    assert (n.los[0], n.his[0]) == (1.0, 2.0)
    # a clamp: max(x, floor) >= floor for any x
    clamp = iv.maximum(scalar(-INF, 5.0), scalar(0.5, 0.5))
    assert clamp.los[0] == 0.5  # floored


def test_join_is_the_hull():
    h = iv.join([scalar(1.0, 2.0), scalar(5.0, 7.0)])
    assert (h.los[0], h.his[0]) == (1.0, 7.0)


def test_select_n_definite_picks_straddle_joins():
    x, y = scalar(10.0, 10.0), scalar(20.0, 20.0)
    # which definitely 0 -> case 0
    assert iv.select_n(scalar(0.0, 0.0), [x, y]).los[0] == 10.0
    # which definitely 1 -> case 1
    assert iv.select_n(scalar(1.0, 1.0), [x, y]).los[0] == 20.0
    # which straddles {0,1} -> join
    j = iv.select_n(scalar(0.0, 1.0), [x, y])
    assert (j.los[0], j.his[0]) == (10.0, 20.0)


# --- pytree-probe registration round: new domain ops -------------------------


def test_abs_piecewise_exact():
    r = iv.abs_(iv.from_values((3,), [-2.0, 3.0, 0.0]))
    assert r.los == (2.0, 3.0, 0.0) and r.his == (2.0, 3.0, 0.0)  # no bump
    s = iv.abs_(scalar(-3.0, 2.0))  # straddles zero
    assert (s.los[0], s.his[0]) == (0.0, 3.0)
    n = iv.abs_(scalar(-5.0, -1.0))  # all negative: exact flip
    assert (n.los[0], n.his[0]) == (1.0, 5.0)
    p = iv.abs_(scalar(1.0, 4.0))  # all positive: identity
    assert (p.los[0], p.his[0]) == (1.0, 4.0)
    half = iv.abs_(scalar(-INF, 5.0))  # infinite endpoint
    assert (half.los[0], half.his[0]) == (0.0, INF)


def test_eq_three_valued():
    # definitely true ONLY when both operands are the same single point
    assert iv.eq(scalar(2.0, 2.0), scalar(2.0, 2.0)).los[0] == 1.0
    # disjoint: definitely false
    assert iv.eq(scalar(1.0, 2.0), scalar(3.0, 4.0)).his[0] == 0.0
    # identical *intervals* are not a point: unknown, never guessed
    r = iv.eq(scalar(1.0, 2.0), scalar(1.0, 2.0))
    assert (r.los[0], r.his[0]) == (0.0, 1.0)
    # a point inside a wider interval: unknown
    r2 = iv.eq(scalar(1.0, 1.0), scalar(0.0, 2.0))
    assert (r2.los[0], r2.his[0]) == (0.0, 1.0)
    # touching endpoints may or may not be equal: unknown
    r3 = iv.eq(scalar(1.0, 2.0), scalar(2.0, 3.0))
    assert (r3.los[0], r3.his[0]) == (0.0, 1.0)


def test_ne_is_the_negation_of_eq():
    assert iv.ne(scalar(1.0, 2.0), scalar(3.0, 4.0)).los[0] == 1.0  # disjoint
    assert iv.ne(scalar(2.0, 2.0), scalar(2.0, 2.0)).his[0] == 0.0  # same point
    r = iv.ne(scalar(1.0, 2.0), scalar(1.0, 2.0))
    assert (r.los[0], r.his[0]) == (0.0, 1.0)


T3, F3, U3 = (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)


def test_logical_and_kleene_table():
    table = {
        (T3, T3): T3, (T3, F3): F3, (T3, U3): U3,
        (F3, T3): F3, (F3, F3): F3, (F3, U3): F3,
        (U3, T3): U3, (U3, F3): F3, (U3, U3): U3,
    }
    for (x, y), want in table.items():
        r = iv.logical_and(scalar(*x), scalar(*y))
        assert (r.los[0], r.his[0]) == want, (x, y)


def test_logical_or_kleene_table():
    table = {
        (T3, T3): T3, (T3, F3): T3, (T3, U3): T3,
        (F3, T3): T3, (F3, F3): F3, (F3, U3): U3,
        (U3, T3): T3, (U3, F3): U3, (U3, U3): U3,
    }
    for (x, y), want in table.items():
        r = iv.logical_or(scalar(*x), scalar(*y))
        assert (r.los[0], r.his[0]) == want, (x, y)


def test_logical_ops_read_top_as_unknown():
    # ⊤ flowing in from an unregistered producer canonicalizes to unknown —
    # sound while the true values are booleans (the transfer's dtype guard)
    top = scalar(-INF, INF)
    r = iv.logical_and(top, scalar(*T3))
    assert (r.los[0], r.his[0]) == U3
    r2 = iv.logical_or(top, scalar(*T3))
    assert (r2.los[0], r2.his[0]) == T3  # true ∨ anything = true


def test_reduce_or_folds_three_valued_over_axes():
    # rows: (F, T, U) and (F, F, F)
    a = iv.IntervalArray(
        shape=(2, 3),
        los=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        his=(0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    )
    rows = iv.reduce_or(a, (1,))
    assert rows.shape == (2,)
    assert (rows.los, rows.his) == ((1.0, 0.0), (1.0, 0.0))  # T; F
    cols = iv.reduce_or(a, (0,))
    assert cols.shape == (3,)
    assert (cols.los, cols.his) == ((0.0, 1.0, 0.0), (0.0, 1.0, 1.0))  # F T U
    both = iv.reduce_or(a, (0, 1))
    assert both.shape == () and (both.los[0], both.his[0]) == T3


def test_reduce_or_empty_range_axes_are_definitely_false():
    empty = iv.IntervalArray(shape=(0,), los=(), his=())
    r = iv.reduce_or(empty, (0,))
    assert r.shape == () and (r.los[0], r.his[0]) == F3  # OR over nothing
    twoby0 = iv.IntervalArray(shape=(2, 0), los=(), his=())
    r2 = iv.reduce_or(twoby0, (1,))
    assert r2.shape == (2,) and r2.los == (0.0, 0.0) and r2.his == (0.0, 0.0)


def test_reduce_or_bad_axes_raise_interval_error():
    with pytest.raises(iv.IntervalError):
        iv.reduce_or(iv.from_values((2,), [0.0, 1.0]), (1,))


def test_reshape_is_flat_c_order_identity():
    a = iv.from_values((2, 3), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = iv.reshape(a, (3, 2))
    assert r.shape == (3, 2) and r.los == a.los and r.his == a.his
    flat = iv.reshape(a, (6,))
    assert flat.shape == (6,) and flat.los == a.los
    with pytest.raises(iv.IntervalError):
        iv.reshape(a, (4,))  # element count changes: refuse


def test_pow_brackets_positive_base_corners():
    from fractions import Fraction

    r = iv.pow_(scalar(3.0, 3.0), scalar(3.0, 3.0))
    assert Fraction(r.los[0]) < 27 < Fraction(r.his[0])  # bumped outward
    # decreasing in the exponent for base < 1
    r2 = iv.pow_(scalar(0.5, 0.5), scalar(1.0, 2.0))
    assert r2.los[0] < 0.25 and r2.his[0] > 0.5
    # negative exponents (monotone decreasing in the base)
    r3 = iv.pow_(scalar(2.0, 4.0), scalar(-1.0, -1.0))
    assert r3.los[0] < 0.25 and r3.his[0] > 0.5
    assert r3.los[0] > 0.0  # x > 0 gives x**y > 0: floored at 0, never below


def test_pow_overflow_saturates_outward_never_inf_inf():
    r = iv.pow_(scalar(10.0, 10.0), scalar(400.0, 400.0))
    assert r.his[0] == INF
    assert r.los[0] == math.nextafter(INF, 0.0)  # maxfloat: sound, finite


def test_pow_infinite_endpoints():
    r = iv.pow_(scalar(1.0, INF), scalar(2.0, 2.0))
    assert r.los[0] < 1.0 and r.his[0] == INF
    # base straddling 1 with an unbounded exponent covers [0, inf]
    r2 = iv.pow_(scalar(0.5, 2.0), scalar(-INF, INF))
    assert (r2.los[0], r2.his[0]) == (0.0, INF)


def test_pow_nonpositive_base_raises_interval_error():
    for lo, hi in [(-1.0, 2.0), (0.0, 2.0), (-3.0, -1.0)]:
        with pytest.raises(iv.IntervalError):
            iv.pow_(scalar(lo, hi), scalar(2.0, 2.0))


def test_select_n_scalar_selector_broadcasts_over_cases():
    lo_case = iv.from_values((2,), [1.0, 2.0])
    hi_case = iv.from_values((2,), [10.0, 20.0])
    take0 = iv.select_n(iv.point(0.0), [lo_case, hi_case])
    assert take0.shape == (2,) and take0.los == (1.0, 2.0)
    take1 = iv.select_n(iv.point(1.0), [lo_case, hi_case])
    assert take1.los == (10.0, 20.0)
    joined = iv.select_n(scalar(0.0, 1.0), [lo_case, hi_case])
    assert joined.los == (1.0, 2.0) and joined.his == (10.0, 20.0)


def test_select_n_other_shape_combos_still_refuse():
    a2 = iv.from_values((2,), [1.0, 2.0])
    a3 = iv.from_values((3,), [1.0, 2.0, 3.0])
    with pytest.raises(iv.IntervalError):
        iv.select_n(iv.from_values((2,), [0.0, 1.0]), [a2, a3])  # cases disagree
    with pytest.raises(iv.IntervalError):
        iv.select_n(iv.from_values((3,), [0.0, 0.0, 0.0]), [a2, a2])  # non-scalar mismatch


def test_rank_broadcasting_size1_and_missing_leading_dims():
    col = iv.IntervalArray(shape=(2, 1), los=(1.0, 10.0), his=(1.0, 10.0))
    row = iv.from_values((3,), [1.0, 2.0, 3.0])
    r = iv.add(col, row)
    assert r.shape == (2, 3)
    for got_lo, got_hi, want in zip(
        r.los, r.his, [2.0, 3.0, 4.0, 11.0, 12.0, 13.0]
    ):
        # every one of these sums is exactly representable, so the
        # exact-when-representable rule returns it with no widening at all
        assert got_lo == want == got_hi


def test_rank_broadcasting_applies_to_comparisons():
    col = iv.IntervalArray(shape=(2, 1), los=(0.0, 10.0), his=(0.0, 10.0))
    row = iv.from_values((3,), [1.0, 2.0, 3.0])
    r = iv.lt(col, row)
    assert r.shape == (2, 3)
    assert (r.los[:3], r.his[:3]) == ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))  # 0 < all
    assert (r.los[3:], r.his[3:]) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # 10 > all


def test_rank_broadcasting_zero_size_dimension():
    empty = iv.IntervalArray(shape=(0,), los=(), his=())
    one = iv.IntervalArray(shape=(1,), los=(5.0,), his=(5.0,))
    r = iv.add(empty, one)
    assert r.shape == (0,) and r.los == () and r.his == ()


def test_incompatible_shapes_raise_interval_error():
    with pytest.raises(iv.IntervalError):
        iv.add(
            iv.from_values((2,), [1.0, 2.0]), iv.from_values((3,), [1.0, 2.0, 3.0])
        )
