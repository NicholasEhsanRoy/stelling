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


def test_add_brackets_true_result_and_widens():
    r = iv.add(iv.point(0.1), iv.point(0.2))
    # the *exact* real sum of the two doubles lies strictly inside the
    # bracket — checked in exact rational arithmetic, not in floats
    from fractions import Fraction

    true = Fraction(0.1) + Fraction(0.2)
    assert Fraction(r.los[0]) < true < Fraction(r.his[0])
    assert r.los[0] < 0.1 + 0.2 < r.his[0]  # the rounded double too


def test_sub_directed():
    r = iv.sub(scalar(1.0, 2.0), scalar(0.5, 1.5))
    assert r.los[0] < -0.5 <= 0.5 < r.his[0]


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
    r = iv.div(scalar(1.0, 2.0), scalar(4.0, 8.0))
    assert r.los[0] < 0.125 and r.his[0] > 0.5


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
    a = iv.from_values((2, 2), [1.0, 2.0, 3.0, 4.0])
    s = iv.slice_(a, (0, 1), (2, 2), None)
    assert s.shape == (2, 1) and s.los == (2.0, 4.0)
    q = iv.squeeze(s, (1,))
    assert q.shape == (2,) and q.los == (2.0, 4.0)
    b = iv.broadcast_in_dim(iv.point(7.0), (3,), ())
    assert b.shape == (3,) and b.los == (7.0, 7.0, 7.0)
    c = iv.concatenate([iv.from_values((1,), [1.0]), iv.from_values((2,), [2.0, 3.0])], 0)
    assert c.shape == (3,) and c.los == (1.0, 2.0, 3.0)


def test_scalar_literal_broadcasts_against_array():
    r = iv.mul(iv.from_values((2,), [2.0, 3.0]), iv.point(10.0))
    assert r.los[0] < 20.0 < r.his[0] and r.los[1] < 30.0 < r.his[1]
