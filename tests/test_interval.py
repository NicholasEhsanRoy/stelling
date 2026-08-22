# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The interval domain: outward rounding is the soundness-critical surface.

A rounding bug here is a false VERIFIED — the project's own thesis defect —
so these tests check *containment* (the true real result lies inside the
computed bracket), not tightness.
"""

from __future__ import annotations

import inspect
import math
import re
import sys
from fractions import Fraction
from typing import NamedTuple

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


# ===========================================================================
# THE ENDPOINT DISCIPLINES: ENUMERATED FROM THE MODULE, DECLARED, AND DRIVEN
# ===========================================================================
#
# WHAT THIS REPLACES, AND WHY A STRING CHECK WAS NOT ENOUGH.
# `test_the_module_docstring_states_a_scope_and_the_counter_examples_hold`
# said, in its own docstring, *"the scope is what has to be re-decided, not
# the digits"*. Measured on `6387a34`, it was the digits and nothing else:
#
# * deleting EVERY WORD of the scope prose from `interval.__doc__` -- the
#   whole kinds-of-operation block and the "outward is a claim about the
#   module; tight is a claim about one operation" paragraph -- and replacing
#   it with "Some numbers that appear in this module:" plus the eight
#   endpoint reprs left the whole zero-dep suite green at a byte-identical
#   `2173 passed, 164 skipped`. Its only structural teeth were
#   `repr(endpoint) in doc`.
# * both of its `not in doc` guards passed BY ACCIDENT, and both banned
#   wordings were in the docstring at the time, differently spelled. `"one
#   deliberate ulp of slack per operation"` was there with a line break and
#   an indent inside it; `"Every arithmetic endpoint is **correctly
#   directed-rounded**"` was there with the bold moved from
#   `directed-rounded` to `Every`. So the guards detected neither phrase
#   they named, would have reddened on a pure re-wrap, and missed any
#   reworded universal: inserting *"All arithmetic endpoints are correctly
#   directed-rounded: when the exact result is representable, both endpoints
#   ARE it and nothing is bumped"* -- the exact falsehood the correction had
#   just removed, reworded -- gave `42 passed`.
# * adding a public `half()` to `stelling.interval` that rounds both
#   endpoints INWARD left the whole zero-dep suite green, so *"the invariant
#   a new operation added to this module has to preserve"* was enforced by
#   nobody.
# * reverting `mul` to the unconditional bump left the pin green, and its
#   tight-side measurements were read by nothing at all: `grep -rn "0.0625"
#   tests/` was empty.
#
# So the instrument is not a string check. The POPULATION IS READ OFF THE
# MODULE -- every public function `stelling.interval` defines -- and each
# one has to declare a discipline and carry a case that DRIVES it. A new
# operation forces a classification; an operation that stops rounding
# outward reddens; an operation whose tightness is loosened OR tightened
# reddens; and the docstring is then checked for what it CLAIMS about the
# operations that are not correctly directed-rounded, not for which digits
# it contains. Nine mutations were watched go red for it, one at a time and
# each from green: an inward-rounding `half()` both unclassified and
# classified, the false universal in two spellings, the scope prose gutted
# twice, `mul` loosened, `sqrt` tightened, and `add` no longer rounding
# outward at all. The commit that added this table lists every one of them
# with the message it produced.

CORRECTLY_ROUNDED = "correctly directed-rounded"
ONE_ULP_BUMPED = "unconditional one-ulp bump"
EXACT_PER_STEP = "exact per step, not per result"
NO_ROUNDING = "no rounding: the endpoints are exact"
NOT_OUTWARD = "outside the outward-ℝ claim"
NO_ENDPOINTS = "produces no interval endpoints"

# The disciplines a reader of `interval.__doc__` would guess WRONG, so the
# docstring has to name the operations that carry them. `NO_ROUNDING` and
# `NO_ENDPOINTS` are not here: they are the safe direction (a reader who
# assumes `slice_` rounds is merely over-cautious), and demanding that the
# docstring list all of them would turn it into the census it says it is not.
MUST_BE_NAMED_IN_THE_DOCSTRING = {
    ONE_ULP_BUMPED, EXACT_PER_STEP, NOT_OUTWARD,
}


class Op(NamedTuple):
    discipline: str
    why: str
    drive: object          # () -> list[tuple], shaped by the discipline
    doc_token: str = ""    # what the docstring must name; default: the name
    quoted: tuple = ()     # indices of the cases the scope prose QUOTES


def _f(x):
    """Exact rational value of a double."""
    return Fraction(x)


def _rounded_down(exact: Fraction) -> float:
    """Largest double <= exact. Computed HERE, not by the module."""
    f = float(exact)
    return f if Fraction(f) <= exact else math.nextafter(f, -math.inf)


def _rounded_up(exact: Fraction) -> float:
    """Smallest double >= exact. Computed HERE, not by the module."""
    f = float(exact)
    return f if Fraction(f) >= exact else math.nextafter(f, math.inf)


def _e(r):
    """The endpoints of a scalar-shaped result, as a plain pair."""
    return r.los[0], r.his[0]


def _at(r, i):
    """The endpoints of ONE element of a result, as a plain pair.

    A row that reads element 0 of a rank-1 result measures one shape. Some
    operations take a different path through a different shape -- see
    :func:`_scatter_debt_evidence`, which exists because that was not
    hypothetical.
    """
    return r.los[i], r.his[i]


def _ie(r):
    """The endpoints of an ieee kernel's ``(interval, made_nan)`` result."""
    return r[0].los[0], r[0].his[0]


def _vals(shape, xs):
    return iv.from_values(shape, [float(x) for x in xs])


_EPS = 2.0 ** -53


# --- the table ------------------------------------------------------------
#
# CASE SHAPES, by discipline:
#   CORRECTLY_ROUNDED / ONE_ULP_BUMPED / EXACT_PER_STEP
#       (label, lo, hi, exact_lo: Fraction, exact_hi: Fraction)
#       where [exact_lo, exact_hi] is the EXACT real image of the operation
#       on the case's operands.
#   NO_ROUNDING
#       (label, got_los, got_his, want_los, want_his)
#   NOT_OUTWARD
#       (label, lo, hi, rlo: Fraction, rhi: Fraction) where rlo <= R <= rhi
#       encloses the true real result R the module's universal claim would
#       have put inside [lo, hi]. The checker requires the bracket to miss
#       that enclosure entirely, so the exclusion is proved and not asserted.
#   NO_ENDPOINTS
#       (label, got, want)

DISCIPLINE: dict[str, Op] = {}


def _op(name, discipline, why, drive, doc_token="", quoted=()):
    DISCIPLINE[name] = Op(discipline, why, drive, doc_token, quoted)


# ---- correctly directed-rounded ------------------------------------------
#
# Two cases each, and the pair is the point: one whose exact image IS
# representable, where both endpoints must BE it (this is what reverting an
# operation to the unconditional bump breaks), and one whose exact image is
# not, where the endpoints must be its two neighbours (this is what an
# inward rounding breaks).

_op("add", CORRECTLY_ROUNDED,
    "exact Fraction endpoints, rounded outward only when inexact",
    lambda: [
        ("[0.25,0.5] + [0.25,0.5]", *_e(iv.add(scalar(0.25, 0.5),
                                               scalar(0.25, 0.5))),
         Fraction(1, 2), Fraction(1)),
        ("0.1 + 0.2", *_e(iv.add(iv.point(0.1), iv.point(0.2))),
         _f(0.1) + _f(0.2), _f(0.1) + _f(0.2)),
    ],
    quoted=(0,))

_op("sub", CORRECTLY_ROUNDED,
    "directed subtraction on exact Fraction corners",
    lambda: [
        ("[0.25,0.5] - [0.25,0.5]", *_e(iv.sub(scalar(0.25, 0.5),
                                               scalar(0.25, 0.5))),
         Fraction(-1, 4), Fraction(1, 4)),
        ("1 - 0.1", *_e(iv.sub(iv.point(1.0), iv.point(0.1))),
         Fraction(1) - _f(0.1), Fraction(1) - _f(0.1)),
    ],
    quoted=(0,))

_op("mul", CORRECTLY_ROUNDED,
    "four exact corner products, one directed rounding (audit 0.2.0 M16)",
    lambda: [
        ("[0.25,0.5] * [0.25,0.5]", *_e(iv.mul(scalar(0.25, 0.5),
                                               scalar(0.25, 0.5))),
         Fraction(1, 16), Fraction(1, 4)),
        ("0.1 * 0.1", *_e(iv.mul(iv.point(0.1), iv.point(0.1))),
         _f(0.1) * _f(0.1), _f(0.1) * _f(0.1)),
    ],
    quoted=(0,))

# The `[1,1] / [2,inf]` case is the HALF-INFINITE arm, and it is in this
# row because it belongs to this discipline and used not to be treated as
# though it did. The exactness gate asked about the whole operand quadruple,
# so one infinite endpoint dropped ALL FOUR corners onto the unconditional
# bump and this returned `(-5e-324, 0.5000000000000001)` for an image that
# is exactly `[0, 1/2]` -- a strictly-positive quotient with a negative
# lower endpoint. It fits this row's case shape because the infinity is in
# the DIVISOR, where `x / +-inf` is exactly 0 in the limit, so the image is
# finite at both ends and both ends are `Fraction`s; the arms whose image
# REACHES infinity cannot be written here at all and are driven by
# `test_a_half_infinite_operand_costs_only_the_corners_it_touches`, which
# can express an infinite endpoint and this case shape cannot.
_op("div", CORRECTLY_ROUNDED,
    "exact Fraction quotients on the non-straddling case, corner by corner",
    lambda: [
        ("[0.25,0.5] / [0.25,0.5]", *_e(iv.div(scalar(0.25, 0.5),
                                               scalar(0.25, 0.5))),
         Fraction(1, 2), Fraction(2)),
        ("1 / 3", *_e(iv.div(iv.point(1.0), iv.point(3.0))),
         Fraction(1, 3), Fraction(1, 3)),
        ("[1,1] / [2,inf] (finite corner beside an infinite one)",
         *_e(iv.div(iv.point(1.0), scalar(2.0, INF))),
         Fraction(0), Fraction(1, 2)),
    ],
    quoted=(0,))

# `0.7 ** 2` is here because `0.1 ** 3` alone left this row ONE-SIDED. Both
# of `_frac_bracket`'s inexact branches are reachable -- the nearest double
# can fall either side of the exact power -- and `0.1 ** 3` falls ABOVE, so
# it drove the `xf > fr` branch and nothing drove the other. Measured:
# flipping `xf < fr` to bump the wrong way gives
# `integer_pow([0.7,0.7], 2) = (0.4899999999999999, 0.48999999999999994)`,
# a bracket that EXCLUDES the exact real 0.7**2, with this whole file green
# at `dbde454`'s own 46 passed. (`tests/test_three_rows.py`'s
# `test_integer_pow_endpoints_bracket_the_exact_rational_power` did catch
# it, so it was never a tree-level hole -- but a row that drives one branch
# of a two-branch rounding is not driving the discipline it declares.)
_op("integer_pow", CORRECTLY_ROUNDED,
    "Fraction(x) ** n is the EXACT power; bumped only on the side that fell",
    lambda: [
        ("[0.5,0.5] ** 2", *_e(iv.integer_pow(scalar(0.5, 0.5), 2)),
         Fraction(1, 4), Fraction(1, 4)),
        ("0.1 ** 3 (the double falls ABOVE the exact power)",
         *_e(iv.integer_pow(iv.point(0.1), 3)),
         _f(0.1) ** 3, _f(0.1) ** 3),
        ("0.7 ** 2 (the double falls BELOW it)",
         *_e(iv.integer_pow(iv.point(0.7), 2)),
         _f(0.7) ** 2, _f(0.7) ** 2),
    ])

_op("boundary_div", CORRECTLY_ROUNDED,
    "div's exact route on the arm whose divisor misses zero; the "
    "zero-boundary arm is the ±inf endpoint convention and widens on purpose",
    lambda: [
        ("[1,1] / [4,8]", *_e(iv.boundary_div(scalar(1.0, 1.0),
                                              scalar(4.0, 8.0))),
         Fraction(1, 8), Fraction(1, 4)),
        ("1 / 3", *_e(iv.boundary_div(iv.point(1.0), iv.point(3.0))),
         Fraction(1, 3), Fraction(1, 3)),
        ("[1,1] / [2,inf] (finite corner beside an infinite one)",
         *_e(iv.boundary_div(iv.point(1.0), scalar(2.0, INF))),
         Fraction(0), Fraction(1, 2)),
    ])


# ---- unconditional one-ulp bump ------------------------------------------
#
# Every case here has a REPRESENTABLE exact image, so "unconditional" is
# what is being measured: a correctly-rounded operation would return it
# unwidened, and these return its two neighbours instead. Tighten one and
# this reddens; widen one and it reddens too.

_op("sqrt", ONE_ULP_BUMPED,
    "the ulp keeps it sound on a platform whose sqrt is merely faithful",
    lambda: [("sqrt([4,4])", *_e(iv.sqrt(scalar(4.0, 4.0))),
              Fraction(2), Fraction(2))],
    quoted=(0,))

_op("exp", ONE_ULP_BUMPED,
    "the ulp pays for the faithfully-rounded-libm assumption",
    lambda: [("exp([0,0])", *_e(iv.exp(scalar(0.0, 0.0))),
              Fraction(1), Fraction(1))],
    quoted=(0,))

_op("pow_", ONE_ULP_BUMPED,
    "the ulp pays for the faithfully-rounded-libm assumption, at four corners",
    lambda: [("pow_([2,2],[3,3])", *_e(iv.pow_(scalar(2.0, 2.0),
                                               scalar(3.0, 3.0))),
              Fraction(8), Fraction(8))],
    quoted=(0,))


# ---- exact per step, not per result --------------------------------------
#
# Two cases, and again the pair is the discipline. A fold whose every step
# is exact returns the exact total UNWIDENED -- that is what makes this not
# the unconditional bump. A fold of three or more contributors may put the
# representable exact total STRICTLY INSIDE both endpoints -- that is what
# makes it not correctly directed-rounded either.

def _fold_cases(f, label):
    two = f(_vals((2,), [1.0, 1.0]))
    three = f(iv.IntervalArray(shape=(3,), los=(1.0, _EPS, _EPS),
                               his=(1.0, _EPS, _EPS)))
    total = Fraction(1) + 2 * _f(_EPS)
    return [
        (f"{label} of [1, 1]", *_e(two), Fraction(2), Fraction(2)),
        (f"{label} of [1, 2**-53, 2**-53]", *_e(three), total, total),
    ]


_op("reduce_sum", EXACT_PER_STEP,
    "each accumulation step is exact-when-representable; the TOTAL need not "
    "be a neighbour of either endpoint",
    lambda: _fold_cases(lambda a: iv.reduce_sum(a, (0,)), "reduce_sum"),
    quoted=(1,))

_op("dot_general", EXACT_PER_STEP,
    "the per-term products take mul's exact corners and the accumulation is "
    "reduce_sum's; the contraction total is still not per-result",
    lambda: _fold_cases(
        lambda a: iv.dot_general(a, _vals(a.shape, [1.0] * a.shape[0]),
                                 (((0,), (0,)), ((), ()))),
        "dot_general"))


# THE MULTI-COLUMN CASES ARE NOT DECORATION. This row used to drive exactly
# one shape -- `(1,)`, one index, one contribution -- and the kernel branches
# on `rowsz`, so a fix applied only where `rowsz > 1` left the rank-1 shape
# on the unconditional bump. Measured: `scatter_add_rows` then carried TWO
# disciplines at once -- correctly directed-rounded for a multi-column row,
# the unconditional bump for a rank-1 one -- falsifying both this row and the
# docstring's flat claim for half the shapes, with that tree's whole zero-dep
# suite green at its own `2178 passed, 164 skipped`. CONTRIBUTIONS is the
# other axis a partial fix can hide behind, and it is now the axis that
# separates this row's two halves: one contribution is EXACT, three need not
# be. `test_the_scatter_add_rows_debt_is_the_one_the_code_owes` drives the
# same two axes with the coarser question -- exact total, or wider?
def _scatter_row_cases():
    one = iv.scatter_add_rows(_vals((1,), [1.0]), _vals((1,), [1.0]), [0])
    wide = iv.scatter_add_rows(_vals((2, 2), [1.0, 2.0, 3.0, 4.0]),
                               _vals((1, 2), [1.0, 2.0]), [1])
    costed = iv.scatter_add_rows(
        iv.IntervalArray(shape=(1,), los=(0.0,), his=(0.0,)),
        iv.IntervalArray(shape=(1,), los=(0.0,), his=(16.0,)), [0])
    # THE HALF THAT MAKES THIS EXACT-PER-STEP AND NOT CORRECTLY ROUNDED, and
    # it is `reduce_sum`'s own case folded through this kernel instead: three
    # contributions into one element, whose exact total IS representable and
    # is STRICTLY INSIDE both endpoints. Without it this row would be
    # indistinguishable from `CORRECTLY_ROUNDED`, which is a claim about the
    # RESULT and is false here for `n >= 3`.
    folded = iv.scatter_add_rows(
        iv.IntervalArray(shape=(1,), los=(0.0,), his=(0.0,)),
        iv.IntervalArray(shape=(3,), los=(1.0, _EPS, _EPS),
                         his=(1.0, _EPS, _EPS)), [0, 0, 0])
    total = Fraction(1) + 2 * _f(_EPS)
    return [
        ("[1] +=[1]", *_e(one), Fraction(2), Fraction(2)),
        ("[0,0] += [0,16]", *_e(costed), Fraction(0), Fraction(16)),
        ("[[1,2],[3,4]] row 1 += [1,2], column 0 (rowsz 2)",
         *_at(wide, 2), Fraction(4), Fraction(4)),
        ("[[1,2],[3,4]] row 1 += [1,2], column 1 (rowsz 2)",
         *_at(wide, 3), Fraction(6), Fraction(6)),
        ("[0] += [1] += [2**-53] += [2**-53]", *_e(folded), total, total),
    ]


_op("scatter_add_rows", EXACT_PER_STEP,
    "each accumulation step is reduce_sum's own -- `_add_lo`/`_add_hi`, "
    "exact-when-representable -- so ONE contribution returns the exact "
    "total unwidened and a fold of three need not. It bumped every step "
    "UNCONDITIONALLY until B23, which was the M16 defect one operation "
    "over and cost a verdict at the public API",
    _scatter_row_cases,
    quoted=(0, 4))


# ---- outside the outward-ℝ claim -----------------------------------------
#
# The `ieee_*` kernels serve `semantics="ieee"`, where what is bracketed is
# the FLOAT the compiled program computes and not a real, so they do NOT
# round outward. `meet` is outside for a different reason: it is an
# intersection, so it drops points on purpose. Each case proves the
# exclusion rather than asserting it -- the returned bracket has to miss a
# rational enclosure of the true real result entirely.

_SQRT2_LO = Fraction(14142135623730950, 10 ** 16)
_SQRT2_HI = Fraction(14142135623730951, 10 ** 16)


def _ieee_binary(f, label):
    return [(label, *_ie(f(iv.point(0.1), iv.point(0.2))),
             _f(0.1) + _f(0.2), _f(0.1) + _f(0.2))]


_op("ieee_add", NOT_OUTWARD, "native binary64 add: the float, not the real",
    lambda: [("ieee_add(0.1, 0.2)",
              *_ie(iv.ieee_add(iv.point(0.1), iv.point(0.2))),
              _f(0.1) + _f(0.2), _f(0.1) + _f(0.2))],
    doc_token="ieee_*")

_op("ieee_add_fmt", NOT_OUTWARD, "ieee_add with the format's subnormal band",
    lambda: [("ieee_add_fmt(0.1, 0.2)",
              *_ie(iv.ieee_add_fmt(iv.point(0.1), iv.point(0.2),
                                   iv.MIN_NORMAL)),
              _f(0.1) + _f(0.2), _f(0.1) + _f(0.2))],
    doc_token="ieee_*")

_op("ieee_sub", NOT_OUTWARD, "native binary64 sub: the float, not the real",
    lambda: [("ieee_sub(1, 0.1)",
              *_ie(iv.ieee_sub(iv.point(1.0), iv.point(0.1))),
              Fraction(1) - _f(0.1), Fraction(1) - _f(0.1))],
    doc_token="ieee_*")

_op("ieee_sub_fmt", NOT_OUTWARD, "ieee_sub with the format's subnormal band",
    lambda: [("ieee_sub_fmt(1, 0.1)",
              *_ie(iv.ieee_sub_fmt(iv.point(1.0), iv.point(0.1),
                                   iv.MIN_NORMAL)),
              Fraction(1) - _f(0.1), Fraction(1) - _f(0.1))],
    doc_token="ieee_*")

_op("ieee_mul", NOT_OUTWARD, "native binary64 mul: the float, not the real",
    lambda: [("ieee_mul(0.1, 0.1)",
              *_ie(iv.ieee_mul(iv.point(0.1), iv.point(0.1))),
              _f(0.1) * _f(0.1), _f(0.1) * _f(0.1))],
    doc_token="ieee_*")

_op("ieee_mul_fmt", NOT_OUTWARD, "ieee_mul with the format's subnormal band",
    lambda: [("ieee_mul_fmt(0.1, 0.1)",
              *_ie(iv.ieee_mul_fmt(iv.point(0.1), iv.point(0.1),
                                   iv.MIN_NORMAL)),
              _f(0.1) * _f(0.1), _f(0.1) * _f(0.1))],
    doc_token="ieee_*")

_op("ieee_div", NOT_OUTWARD, "native binary64 div: the float, not the real",
    lambda: [("ieee_div(1, 3)",
              *_ie(iv.ieee_div(iv.point(1.0), iv.point(3.0))),
              Fraction(1, 3), Fraction(1, 3))],
    doc_token="ieee_*")

_op("ieee_div_fmt", NOT_OUTWARD, "ieee_div with the format's subnormal band",
    lambda: [("ieee_div_fmt(1, 3)",
              *_ie(iv.ieee_div_fmt(iv.point(1.0), iv.point(3.0),
                                   iv.MIN_NORMAL)),
              Fraction(1, 3), Fraction(1, 3))],
    doc_token="ieee_*")

_op("ieee_sqrt", NOT_OUTWARD, "native binary64 sqrt: the float, not the real",
    lambda: [("ieee_sqrt(2)", *_ie(iv.ieee_sqrt(iv.point(2.0))),
              _SQRT2_LO, _SQRT2_HI)],
    doc_token="ieee_*")

_op("ieee_sqrt_fmt", NOT_OUTWARD,
    "ieee_sqrt with the format's subnormal band",
    lambda: [("ieee_sqrt_fmt(2)",
              *_ie(iv.ieee_sqrt_fmt(iv.point(2.0), iv.MIN_NORMAL)),
              _SQRT2_LO, _SQRT2_HI)],
    doc_token="ieee_*")

_op("ieee_reduce_sum", NOT_OUTWARD,
    "the association-free ieee fold: the float total, not the real one",
    lambda: [("ieee_reduce_sum([0.1, 0.2])",
              *_ie(iv.ieee_reduce_sum(_vals((2,), [0.1, 0.2]), (0,))),
              _f(0.1) + _f(0.2), _f(0.1) + _f(0.2))],
    doc_token="ieee_*")

_op("ieee_reduce_sum_fmt", NOT_OUTWARD,
    "ieee_reduce_sum with the format's subnormal band",
    lambda: [("ieee_reduce_sum_fmt([0.1, 0.2])",
              *_ie(iv.ieee_reduce_sum_fmt(_vals((2,), [0.1, 0.2]), (0,),
                                          iv.MIN_NORMAL)),
              _f(0.1) + _f(0.2), _f(0.1) + _f(0.2))],
    doc_token="ieee_*")

_op("ieee_fma_hull", NOT_OUTWARD,
    "the CONTRACTED a*b + c at native precision: one rounding, of the float",
    lambda: [("ieee_fma_hull(0.1, 0.1, 0)",
              *_ie(iv.ieee_fma_hull(iv.point(0.1), iv.point(0.1),
                                    iv.point(0.0))),
              _f(0.1) * _f(0.1), _f(0.1) * _f(0.1))])

_op("meet", NOT_OUTWARD,
    "an INTERSECTION: it drops points on purpose, and its own docstring "
    "opens \"No outward rounding, deliberately.\"",
    lambda: [("meet([0,2], [1,3])",
              *_e(iv.meet(iv.from_bounds((), 0.0, 2.0),
                          iv.from_bounds((), 1.0, 3.0))),
              Fraction(1, 2), Fraction(1, 2))])


# ---- no rounding at all --------------------------------------------------
#
# Selection, data movement, three-valued booleans and the constructors. Every
# endpoint is a value that was already there, so the answer is asserted
# EXACTLY: any widening at all reddens, which is the whole check these
# operations need. (`abs_`, `neg`, `maximum`, `minimum`, `hull`, `join` and
# `int_div` DO compute -- with negation, `min`/`max` and truncation over
# doubles, every one of which is exact, so there is no rounding to direct.)

def _nr(label, r, los, his):
    return (label, r.los, r.his,
            tuple(float(x) for x in los), tuple(float(x) for x in his))


_ROWS = _vals((3, 2), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
_B01 = _vals((2,), [0.0, 1.0])


def _no_rounding(name, why, drive, doc_token=""):
    _op(name, NO_ROUNDING, why, drive, doc_token)


_no_rounding("abs_", "negation and max of doubles are exact",
             lambda: [_nr("abs_([-2,3])", iv.abs_(scalar(-2.0, 3.0)),
                          [0.0], [3.0])])
_no_rounding("neg", "negation of a double is exact",
             lambda: [_nr("neg([-2,3])", iv.neg(scalar(-2.0, 3.0)),
                          [-3.0], [2.0])])
_no_rounding("maximum", "endpointwise max, monotone and exact",
             lambda: [_nr("max([1,3],[2,2])",
                          iv.maximum(scalar(1.0, 3.0), scalar(2.0, 2.0)),
                          [2.0], [3.0])])
_no_rounding("minimum", "endpointwise min, monotone and exact",
             lambda: [_nr("min([1,3],[2,2])",
                          iv.minimum(scalar(1.0, 3.0), scalar(2.0, 2.0)),
                          [1.0], [2.0])])
_no_rounding("hull", "the elementwise join: min of los, max of his",
             lambda: [_nr("hull([0,1],[2,3])",
                          iv.hull(scalar(0.0, 1.0), scalar(2.0, 3.0)),
                          [0.0], [3.0])])
_no_rounding("join", "the n-ary hull",
             lambda: [_nr("join([0,1],[2,3])",
                          iv.join([scalar(0.0, 1.0), scalar(2.0, 3.0)]),
                          [0.0], [3.0])])
_no_rounding("top", "⊤ is a constant",
             lambda: [_nr("top(())", iv.top(()), [-INF], [INF])])
_no_rounding("point", "a degenerate interval AT the value; no bump",
             lambda: [_nr("point(0.1)", iv.point(0.1), [0.1], [0.1])])
_no_rounding("from_bounds", "the declared bounds, verbatim",
             lambda: [_nr("from_bounds((2,), 1, 2)",
                          iv.from_bounds((2,), 1.0, 2.0),
                          [1.0, 1.0], [2.0, 2.0])])
_no_rounding("from_values", "degenerate intervals at the values, verbatim",
             lambda: [_nr("from_values((2,), [1, 2])",
                          _vals((2,), [1.0, 2.0]), [1.0, 2.0], [1.0, 2.0])])
_no_rounding("slice_", "pure data movement",
             lambda: [_nr("slice_", iv.slice_(_ROWS, (0, 1), (2, 2), None),
                          [2.0, 4.0], [2.0, 4.0])])
_no_rounding("squeeze", "pure data movement",
             lambda: [_nr("squeeze",
                          iv.squeeze(iv.slice_(_ROWS, (0, 1), (2, 2), None),
                                     (1,)),
                          [2.0, 4.0], [2.0, 4.0])])
_no_rounding("reshape", "flat storage is untouched",
             lambda: [_nr("reshape", iv.reshape(_ROWS, (2, 3)),
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
_no_rounding("transpose", "axis permutation",
             lambda: [_nr("transpose", iv.transpose(_ROWS, (1, 0)),
                          [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
                          [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])])
_no_rounding("broadcast_in_dim", "copies",
             lambda: [_nr("broadcast_in_dim",
                          iv.broadcast_in_dim(iv.point(7.0), (3,), ()),
                          [7.0, 7.0, 7.0], [7.0, 7.0, 7.0])])
_no_rounding("concatenate", "copies",
             lambda: [_nr("concatenate",
                          iv.concatenate([_vals((1,), [1.0]),
                                          _vals((2,), [2.0, 3.0])], 0),
                          [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])])
_no_rounding("stack", "copies onto a new axis",
             lambda: [_nr("stack",
                          iv.stack([scalar(1.0, 2.0), scalar(3.0, 4.0)], 0),
                          [1.0, 3.0], [2.0, 4.0])])
_no_rounding("take_rows", "row copies",
             lambda: [_nr("take_rows", iv.take_rows(_ROWS, [2, 0]),
                          [5.0, 6.0, 1.0, 2.0], [5.0, 6.0, 1.0, 2.0])])
_no_rounding("take_row_ranges", "the exact hull of the rows in range",
             lambda: [_nr("take_row_ranges",
                          iv.take_row_ranges(_ROWS, [(0, 2)]),
                          [1.0, 2.0], [5.0, 6.0])])
_no_rounding("dynamic_slice_hull", "the exact hull of the reachable slices",
             lambda: [_nr("dynamic_slice_hull",
                          iv.dynamic_slice_hull(_ROWS, ((0, 1), (0, 0)),
                                                (1, 2)),
                          [1.0, 2.0], [3.0, 4.0])])
_no_rounding("dynamic_update_slice_hull",
             "the exact hull of written-or-not, per element",
             lambda: [_nr("dynamic_update_slice_hull",
                          iv.dynamic_update_slice_hull(
                              _ROWS, _vals((1, 2), [9.0, 9.0]),
                              ((0, 1), (0, 0))),
                          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [9.0, 9.0, 9.0, 9.0, 5.0, 6.0])])
_no_rounding("select_n", "an exact pick, or the hull when `which` is unknown",
             lambda: [_nr("select_n",
                          iv.select_n(_B01, [_vals((2,), [10.0, 20.0]),
                                             _vals((2,), [30.0, 40.0])]),
                          [10.0, 40.0], [10.0, 40.0])])
_no_rounding("int_div", "truncation toward zero, computed exactly",
             lambda: [_nr("int_div(7, 2)",
                          iv.int_div(_vals((1,), [7.0]), _vals((1,), [2.0])),
                          [3.0], [3.0])])
_no_rounding("subnormal_haze",
             "an ieee-mode band widening: hulls with 0 exactly. It only ever "
             "widens, so it is not a counter-example to outwardness -- it is "
             "simply not ℝ arithmetic",
             lambda: [
                 _nr("normal band untouched",
                     iv.subnormal_haze(scalar(1.0, 2.0))[0], [1.0], [2.0]),
                 _nr("subnormal band hulled with 0",
                     iv.subnormal_haze(scalar(1e-320, 1e-310))[0],
                     [0.0], [1e-310]),
             ])
_no_rounding("subnormal_haze_fmt", "subnormal_haze at a declared min_normal",
             lambda: [
                 _nr("normal band untouched",
                     iv.subnormal_haze_fmt(scalar(1.0, 2.0),
                                           iv.MIN_NORMAL)[0], [1.0], [2.0]),
                 _nr("subnormal band hulled with 0",
                     iv.subnormal_haze_fmt(scalar(1e-320, 1e-310),
                                           iv.MIN_NORMAL)[0],
                     [0.0], [1e-310]),
             ])

# three-valued comparisons and Kleene logic: endpoints on {0.0, 1.0} only
_no_rounding("lt", "definite or undecided, never guessed",
             lambda: [_nr("lt definite",
                          iv.lt(scalar(1.0, 2.0), scalar(3.0, 4.0)),
                          [1.0], [1.0]),
                      _nr("lt undecided",
                          iv.lt(scalar(1.0, 3.5), scalar(3.0, 4.0)),
                          [0.0], [1.0])])
_no_rounding("le", "definite or undecided, never guessed",
             lambda: [_nr("le definite",
                          iv.le(scalar(1.0, 2.0), scalar(2.0, 4.0)),
                          [1.0], [1.0])])
_no_rounding("gt", "definite or undecided, never guessed",
             lambda: [_nr("gt definite",
                          iv.gt(scalar(3.0, 4.0), scalar(1.0, 2.0)),
                          [1.0], [1.0])])
_no_rounding("ge", "definite or undecided, never guessed",
             lambda: [_nr("ge definite",
                          iv.ge(scalar(3.0, 4.0), scalar(1.0, 2.0)),
                          [1.0], [1.0])])
_no_rounding("eq", "definitely true only for two equal POINT intervals",
             lambda: [_nr("eq definite",
                          iv.eq(iv.point(2.0), iv.point(2.0)), [1.0], [1.0]),
                      _nr("eq undecided",
                          iv.eq(scalar(1.0, 3.0), iv.point(2.0)),
                          [0.0], [1.0])])
_no_rounding("ne", "the negation of eq's logic",
             lambda: [_nr("ne definite",
                          iv.ne(iv.point(2.0), iv.point(3.0)), [1.0], [1.0])])
_no_rounding("logical_and", "Kleene AND",
             lambda: [_nr("false ∧ unknown",
                          iv.logical_and(_vals((1,), [0.0]),
                                         iv.IntervalArray(shape=(1,),
                                                          los=(0.0,),
                                                          his=(1.0,))),
                          [0.0], [0.0])])
_no_rounding("logical_or", "Kleene OR",
             lambda: [_nr("true ∨ unknown",
                          iv.logical_or(_vals((1,), [1.0]),
                                        iv.IntervalArray(shape=(1,),
                                                         los=(0.0,),
                                                         his=(1.0,))),
                          [1.0], [1.0])])
_no_rounding("logical_not", "Kleene NOT",
             lambda: [_nr("¬false", iv.logical_not(_vals((1,), [0.0])),
                          [1.0], [1.0])])
_no_rounding("reduce_or", "three-valued OR-fold",
             lambda: [_nr("reduce_or([false, true])",
                          iv.reduce_or(_B01, (0,)), [1.0], [1.0])])
_no_rounding("is_finite", "three-valued isfinite",
             lambda: [_nr("is_finite finite",
                          iv.is_finite(_vals((1,), [1.0])), [1.0], [1.0]),
                      _nr("is_finite undecided",
                          iv.is_finite(iv.IntervalArray(shape=(1,),
                                                        los=(-INF,),
                                                        his=(1.0,))),
                          [0.0], [1.0])])


# ---- produces no interval endpoints --------------------------------------
#
# Shape oracles, predicates and stamp builders. They are in the table because
# the POPULATION is read off the module: a function that produces no
# endpoints still has to be classified, so nothing can be added to this
# module without somebody saying which of these six things it is.

def _ne_(name, why, drive):
    _op(name, NO_ENDPOINTS, why, drive)


_ne_("check_shape", "the decline channel for shapes no jax program can carry",
     lambda: [("check_shape((2,3))", iv.check_shape((2, 3)), (2, 3))])
_ne_("dot_general_geometry",
     "THE index oracle; the SMT emission drives it too",
     lambda: [("out_shape",
               iv.dot_general_geometry((2,), (2,),
                                       (((0,), (0,)), ((), ()))).out_shape,
               ()),
              ("contracted_extents",
               iv.dot_general_geometry(
                   (2,), (2,), (((0,), (0,)), ((), ()))).contracted_extents,
               (2,))])
_ne_("straddles_zero", "a predicate over the operand's endpoints",
     lambda: [("straddles", iv.straddles_zero(iv.from_bounds((), -1.0, 1.0)),
               True),
              ("does not", iv.straddles_zero(iv.from_bounds((), 1.0, 2.0)),
               False)])
_ne_("target_flushes_subnormals", "a fact about the MEASURED target",
     lambda: [("float64", iv.target_flushes_subnormals("float64"), True)])
_ne_("ieee_endpoint_assumption", "a stamp string",
     lambda: [("opens with the endpoint claim",
               iv.ieee_endpoint_assumption().startswith(
                   "ieee endpoint arithmetic is native binary64"), True)])
_ne_("subnormal_indeterminacy_assumption", "a stamp string",
     lambda: [("opens with the band claim",
               iv.subnormal_indeterminacy_assumption().startswith(
                   "subnormal indeterminacy"), True)])
_ne_("measured_flush_clause", "the measured-target clause of that stamp",
     lambda: [("names the measured target",
               "jax 0.11.0 CPU" in iv.measured_flush_clause(), True)])


# --- the gates ------------------------------------------------------------

def _public_operations() -> set[str]:
    """Every public function ``stelling.interval`` DEFINES.

    Read off the module, never off a list in a docstring: a list somebody
    forgets to extend is exactly how the universal claim went wrong twice.
    ``__module__`` filters out the names imported into the module's
    namespace (``dataclass``, ``NamedTuple``), so an import cannot silently
    add a row.

    **AND ``isfunction`` IS THE SHAPE, NOT THE REACH.** It sees plain
    functions, module-level lambdas, aliases and ``functools.wraps``-wrapped
    functions. It does NOT see a callable OBJECT, a ``functools.partial``,
    or a function re-exported from another module -- all three of which can
    be public and can round inward. Measured at ``dbde454``: a public
    inward-rounding callable object plus a public inward-rounding
    ``functools.partial``, both added to ``stelling.interval``, left that
    tree's entire zero-dep suite green at its own ``2178 passed, 164
    skipped``. No such object exists here today
    -- the only public callables this filter skips are four classes
    (``IntervalArray``, ``DotGeneralGeometry``, ``IntervalError``,
    ``IndexOutOfBoundsError``) and three imports (``Fraction``,
    ``NamedTuple``, ``dataclass``) -- so the population is complete as
    written, and the claim that goes with it is about the FUNCTION shape.
    Widening this to partials and callable objects is a build, and is
    recorded in ``stelling-sweeps/SWEEP-CARRY-FORWARD.md`` rather than
    half-done here.
    """
    return {
        name for name in dir(iv)
        if not name.startswith("_")
        and inspect.isfunction(getattr(iv, name))
        and getattr(iv, name).__module__ == iv.__name__
    }


def test_every_public_operation_of_this_module_declares_a_discipline():
    """A new public FUNCTION forces a classification.

    Measured on ``6387a34``: adding a public ``half()`` to
    ``stelling.interval`` that rounds both endpoints INWARD left the entire
    zero-dep suite green. The module docstring called outward rounding *"the
    invariant a new operation added to this module has to preserve"* and
    nothing enforced it, in either direction -- neither that the new
    operation preserves it nor that anyone looked.

    The word FUNCTION is the scope and it is not decoration:
    :func:`_public_operations` enumerates by ``inspect.isfunction``, which
    is every operation this module has and is not every shape a public
    callable can take. What it does not reach is written there, with the
    measurement.
    """
    ops = _public_operations()
    assert ops, "the population is empty, so this whole file measures nothing"
    missing = sorted(ops - set(DISCIPLINE))
    assert not missing, (
        f"these public operations of stelling.interval declare no endpoint "
        f"discipline: {missing}. Add each to DISCIPLINE with the discipline "
        f"it carries AND a case that drives it. This is not paperwork: "
        f"`interval.__doc__` calls outward rounding the invariant a new "
        f"operation has to preserve, and this table is what makes that "
        f"sentence true of the operation you just added."
    )
    stale = sorted(set(DISCIPLINE) - ops)
    assert not stale, (
        f"DISCIPLINE classifies names stelling.interval no longer exposes: "
        f"{stale}. A row for an operation that is gone is a claim about "
        f"nothing."
    )


def test_every_declared_discipline_is_driven_against_the_running_code():
    """The declarations are MEASURED, one operation at a time.

    Aggregated rather than parametrised so that one red line lists every
    operation that moved: a discipline change is usually a change of rule,
    and seeing the whole set that shifted is what tells a reader whether
    they changed a rule or broke one.

    What each discipline is checked FOR is written beside its constant.
    Every discipline that says anything about TIGHTNESS is checked
    two-sidedly -- tightening an operation reddens exactly as loosening it
    does, because the claim being pinned is the RULE and not an upper bound
    on the width. ``NOT_OUTWARD`` is the exception and makes no tightness
    claim at all: what it pins is that the bracket misses a real, so an
    operation that BECOMES outward reddens and one that widens further does
    not.
    """
    bad: list[str] = []
    for name in sorted(DISCIPLINE):
        op = DISCIPLINE[name]
        try:
            cases = op.drive()
        except Exception as exc:                       # pragma: no cover
            bad.append(f"{name}: driving it raised {exc!r}")
            continue
        assert cases, f"{name} declares {op.discipline} and drives nothing"
        bad += _check(name, op, cases)
    assert not bad, (
        "the endpoint disciplines and the running code disagree:\n  "
        + "\n  ".join(bad)
        + "\n\nEither the operation changed -- in which case re-decide its "
        "row in DISCIPLINE, and `interval.__doc__` with it -- or it "
        "regressed. Do not retype the digits."
    )


def _check(name: str, op: Op, cases) -> list[str]:
    bad: list[str] = []
    d = op.discipline

    if d in (CORRECTLY_ROUNDED, ONE_ULP_BUMPED, EXACT_PER_STEP):
        exactly_returned = 0
        strictly_inside = 0
        for label, lo, hi, elo, ehi in cases:
            where = f"{name}: {label}"
            # the universal claim first, for every one of them
            if not (_f(lo) <= elo and ehi <= _f(hi)):
                bad.append(f"{where}: NOT OUTWARD -- the exact real image "
                           f"[{elo}, {ehi}] is not inside [{lo!r}, {hi!r}]")
                continue
            if _f(lo) == elo and _f(hi) == ehi:
                exactly_returned += 1
            if _f(lo) < elo and ehi < _f(hi):
                strictly_inside += 1
            if d == CORRECTLY_ROUNDED:
                want = (_rounded_down(elo), _rounded_up(ehi))
                if (lo, hi) != want:
                    bad.append(
                        f"{where}: declared {d} but returned "
                        f"({lo!r}, {hi!r}); the correctly directed rounding "
                        f"of [{elo}, {ehi}] is ({want[0]!r}, {want[1]!r})")
            elif d == ONE_ULP_BUMPED:
                want = (math.nextafter(_rounded_down(elo), -math.inf),
                        math.nextafter(_rounded_up(ehi), math.inf))
                if (lo, hi) != want:
                    bad.append(
                        f"{where}: declared {d} but returned "
                        f"({lo!r}, {hi!r}); one ulp outside [{elo}, {ehi}] "
                        f"is ({want[0]!r}, {want[1]!r})")
        if d == CORRECTLY_ROUNDED and not exactly_returned:
            bad.append(f"{name}: declared {d} and no case returns its exact "
                       f"image UNWIDENED, which is the half of the rule an "
                       f"unconditional bump breaks")
        if d == CORRECTLY_ROUNDED and exactly_returned == len(cases):
            bad.append(f"{name}: declared {d} and every case is exact, so "
                       f"nothing here measures the DIRECTION of its "
                       f"rounding. Drive one inexact case too")
        if d == ONE_ULP_BUMPED and not any(
                _f(_rounded_down(c[3])) == c[3] for c in cases):
            bad.append(f"{name}: declared {d} and no case has a "
                       f"REPRESENTABLE exact image, so nothing here "
                       f"measures that the bump is unconditional")
        if d == EXACT_PER_STEP and not exactly_returned:
            bad.append(f"{name}: declared {d} and no case folds EXACTLY, so "
                       f"nothing distinguishes it from an unconditional bump")
        if d == EXACT_PER_STEP and not strictly_inside:
            bad.append(f"{name}: declared {d} and no case puts a "
                       f"representable exact total STRICTLY inside both "
                       f"endpoints, so nothing distinguishes it from "
                       f"correctly directed rounding")

    elif d == NO_ROUNDING:
        for label, got_los, got_his, want_los, want_his in cases:
            if (got_los, got_his) != (want_los, want_his):
                bad.append(f"{name}: {label}: declared {d} and returned "
                           f"{(got_los, got_his)}, not "
                           f"{(want_los, want_his)}")

    elif d == NOT_OUTWARD:
        for label, lo, hi, rlo, rhi in cases:
            if not (_f(hi) < rlo or _f(lo) > rhi):
                bad.append(
                    f"{name}: {label}: declared OUTSIDE the outward-ℝ "
                    f"claim, "
                    f"and the bracket [{lo!r}, {hi!r}] does NOT miss the "
                    f"true real result's enclosure [{rlo}, {rhi}]. If this "
                    f"operation has become outward, move it -- and re-read "
                    f"the ℝ-mode carve-out in `interval.__doc__`, which "
                    f"says it is not")

    elif d == NO_ENDPOINTS:
        for label, got, want in cases:
            if got != want:
                bad.append(f"{name}: {label}: {got!r}, not {want!r}")

    else:                                              # pragma: no cover
        bad.append(f"{name}: unknown discipline {d!r}")
    return bad


def _drive_recording(name: str, op: Op):
    """Drive ``op`` with ``iv.<name>`` wrapped, and return what came back.

    The wrapper records only calls whose IMMEDIATE caller is this module, so
    an operation reached only from inside another operation -- ``dot_general``
    calls ``mul``, ``join`` calls ``hull`` -- does not count as its own row
    driving it. The attribute is restored in a ``finally``: nothing after
    this sees the wrapper, and in particular :func:`_public_operations`
    (which asks ``inspect.isfunction``) never does.
    """
    real = getattr(iv, name)
    seen: list[object] = []

    def recorder(*args, **kwargs):
        out = real(*args, **kwargs)
        if sys._getframe(1).f_globals.get("__name__") == __name__:
            seen.append(out)
        return out

    setattr(iv, name, recorder)
    try:
        cases = op.drive()
    finally:
        setattr(iv, name, real)
    return cases, seen


def _has_endpoints(value, depth: int = 0) -> bool:
    """True when ``value`` is, or carries, an :class:`iv.IntervalArray`.

    The ``ieee_*`` kernels return ``(interval, made_nan)`` and
    ``subnormal_haze`` returns a pair too, so a shallow walk of tuples and
    lists is what "produces endpoints" has to mean here.
    """
    if isinstance(value, iv.IntervalArray):
        return True
    if depth < 3 and isinstance(value, (tuple, list)):
        return any(_has_endpoints(v, depth + 1) for v in value)
    return False


def test_every_row_drives_the_operation_it_names():
    """A row's CASE and the operation its row is filed under are tied.

    THE HOLE THIS CLOSES, and it is the one under the reclassification
    route below. ``NO_ROUNDING`` and ``NO_ENDPOINTS`` cases compare
    hand-typed expectations -- most of the table, and that is fine for what
    they are -- but nothing tied a row to the operation it names. A row
    could have driven a different operation entirely, or none, and typed an
    answer beside it; the population gate would still count the name as
    classified and the drive would still be green.

    So the operation is WRAPPED for the length of its own row's drive and
    the row has to have called it. Two-sided on the endpoint question as
    well, because that is the other half of what a bucket claims:
    ``NO_ENDPOINTS`` means the operation returns no interval, which is read
    off the value it actually returned rather than believed, and every other
    discipline means it returns one.
    """
    bad: list[str] = []
    for name in sorted(DISCIPLINE):
        op = DISCIPLINE[name]
        _cases, seen = _drive_recording(name, op)
        if not seen:
            bad.append(
                f"{name}: its `drive` never calls `iv.{name}`, so whatever "
                f"it measures is not this operation"
            )
            continue
        produced = any(_has_endpoints(r) for r in seen)
        if op.discipline == NO_ENDPOINTS and produced:
            bad.append(
                f"{name}: declared {NO_ENDPOINTS!r} and returned an "
                f"IntervalArray. An operation with endpoints cannot be "
                f"parked in the bucket for operations without them: give it "
                f"the discipline its endpoints carry"
            )
        if op.discipline != NO_ENDPOINTS and not produced:
            bad.append(
                f"{name}: declared {op.discipline!r} and returned no "
                f"IntervalArray at all, so no endpoint discipline can be "
                f"true of it. It belongs in {NO_ENDPOINTS!r}"
            )
    assert not bad, (
        "these rows do not measure the operation they are filed under:\n  "
        + "\n  ".join(bad)
        + "\n\nA row is a claim about ONE operation. Drive that operation "
        "in its own `drive`, and let the bucket say what the value it "
        "returns actually is."
    )


def _doc_blocks() -> list[str]:
    """The docstring's paragraphs and bullets, at EVERY level, flattened.

    Split on blank lines and on a bullet marker -- ``* `` at column 0 and
    ``- `` at any indent -- so that an operation and the words naming its
    discipline have to be in the SAME block for the block to count as saying
    anything about it. Wrapping and emphasis are normalised away first, for
    the reason they always are here: both of the guards this file replaces
    were defeated by one of them.

    **THE SUB-BULLET MARKER WAS MISSING AND THE MISS WAS LOAD-BEARING.**
    Splitting on the top-level marker alone put the whole *kinds of
    operation* list -- four ``  - `` items, no blank lines between them --
    into ONE block, so the discipline words in the first item counted as
    standing beside the operation named in the fourth. Measured: with the
    rounding unchanged, deleting the discipline-list debt bullet for
    ``scatter_add_rows`` left this file green at ``dbde454``'s own
    ``46 passed``, because the surviving four-in-one block carried both
    halves. The gate's promise -- *"a mention in the place
    that says what the operation does"* -- was weaker than stated in exactly
    the region the debt lives in.
    """
    raw = re.split(r"(?m)\n\s*\n|^\* |^[ \t]+- ", iv.__doc__)
    return [re.sub(r"\s+", " ", b).replace("**", "").replace("`", "")
            for b in raw if b.strip()]


def _names(token: str, block: str) -> bool:
    if token.endswith("*"):                    # a family, e.g. `ieee_*`
        return re.search(re.escape(token), block) is not None
    return re.search(r"\b%s\b" % re.escape(token), block) is not None


def test_the_docstring_names_every_operation_a_reader_would_guess_wrong():
    """The scope prose is checked for WHAT IT CLAIMS.

    Measured on ``6387a34``: deleting every word of it and leaving the eight
    endpoint reprs behind left the whole zero-dep suite green. So the digits
    are not what holds this text down -- the operations are. Every operation
    whose discipline is one a reader would guess wrong (it bumps
    unconditionally, it is exact only per step, or it is outside the
    outward-ℝ claim altogether) has to be named in a block of that docstring
    that ALSO carries the words naming that discipline. Naming it anywhere
    would not do: the defect this is here for was an operation the text
    never mentioned, and the repair for that is not a mention, it is a
    mention in the place that says what the operation does. *What counts as
    "the place" is* :func:`_doc_blocks`, *and the marker it split on was
    coarser than this sentence for one commit -- see there, with the
    measurement.*

    That couples the discipline constants above to the docstring's own
    vocabulary, deliberately -- they are written to BE its vocabulary, and a
    docstring that stops using the words it classifies operations with has
    stopped classifying them.

    The safe directions are deliberately not required: ``NO_ROUNDING`` and
    ``NO_ENDPOINTS`` operations are absent from this requirement -- a reader
    who assumes ``slice_`` rounds is merely over-cautious -- because
    demanding that the docstring list every one of them would turn it into
    the census it says, correctly, that it is not. *(This read "all forty of
    them", which was wrong, in a batch whose subject is hand-typed counts.
    There is no count here now: the set is every ``NO_ROUNDING`` and
    ``NO_ENDPOINTS`` row of the table above, and the table is enumerated
    from the module on every run.)*

    **AND HERE IS THE LIMIT THIS GATE HAS, STATED RATHER THAN DISCOVERED
    AGAIN.** It is one-directional. It asks whether each operation that
    needs a classification IS MENTIONED beside the words naming the
    discipline it actually carries; it never asks whether a block that
    mentions an operation is telling the truth about it. **A FALSE
    CLASSIFICATION IS INVISIBLE HERE** -- an operation named in a block
    claiming a discipline it does not have satisfies this gate exactly as
    well as one named in the right block, and satisfies it even when both
    blocks are in the same docstring saying opposite things.

    Not hypothetical, and not old: ``interval.__doc__``'s ⊤-escapes bullet
    filed *"an infinite operand under ``mul``'s ``0·±inf = 0`` rule"* under
    a heading reading *"every one of them is wider"*, in the same commit
    whose *exact-when-representable* entry said the convention *"names a
    POINT, so it is exact and takes no slack"* -- and ``mul([0, 0],
    [1, inf])`` is ``(0.0, 0.0)``, not wider by anything. Two entries of
    one docstring contradicting each other, with this gate green, because
    ``mul`` is ``CORRECTLY_ROUNDED`` and so is not in
    :data:`MUST_BE_NAMED_IN_THE_DOCSTRING` at all -- it was never being
    asked about. Extending the gate to catch it would mean deciding, from
    prose, which operations a block is making a claim ABOUT rather than
    merely referring to, and that is a different instrument than this one.
    What the replacement bullet DID get is
    :func:`test_the_measurements_the_docstring_quotes_are_the_ones_it_would_get`:
    the saturation pair it now quotes is driven off :data:`_SATURATION`
    rather than typed beside it, so a digit that stops being true reddens
    there. That is a check on the DIGITS. It is not a check on the heading
    they sit under, and nothing here is -- which is the paragraph above,
    repeated where a reader editing the bullet will be standing.
    """
    blocks = _doc_blocks()
    unnamed = sorted(
        name for name, op in DISCIPLINE.items()
        if op.discipline in MUST_BE_NAMED_IN_THE_DOCSTRING
        and not any(_names(op.doc_token or name, b) and op.discipline in b
                    for b in blocks)
    )
    assert not unnamed, (
        f"`stelling.interval.__doc__` has no block that names these "
        f"operations beside the discipline they carry, and each one carries "
        f"a discipline a reader of that docstring would guess wrong: "
        f"{ {n: DISCIPLINE[n].discipline for n in unnamed} }.\n"
        f"Name it and say what it does. The reason this gate exists is that "
        f"`scatter_add_rows` was in none of the docstring's groups while "
        f"bumping unconditionally, which made two of that text's sentences "
        f"positively false -- and the pin at the time asked only that some "
        f"digits appear."
    )


# --- the half-infinite arms, which the case shape above cannot express ----
#
# `DISCIPLINE`'s CORRECTLY_ROUNDED rows carry the exact image as a pair of
# `Fraction`s, and `Fraction(inf)` raises -- so no row in that table can
# state the image of `mul([1, inf], [2, 3])`, which is `[2, inf]`. That is
# not a cosmetic limit: it is exactly the half of the input space where the
# exactness gate was applied to the whole operand QUADRUPLE rather than to
# one corner, and so it is exactly the half nothing in this file could see.
#
# Measured on `61de794`, with that file's `DISCIPLINE` table green:
#
#     mul([1, inf], [2, 3])      (1.9999999999999998, inf)   image [2, inf]
#     mul([0, inf], [0, 3])      (-5e-324, inf)              image [0, inf]
#     div([2, inf], [2, 2])      (0.9999999999999999, inf)   image [1, inf]
#     boundary_div([1,1],[0,inf])(-5e-324, inf)              image [0, inf]
#
# The second is audit 0.2.0 M16's own symptom -- a non-negative product's
# exactly-zero corner put BELOW zero -- inside `mul`, after M16's fix, in
# the half of the input space the fix's gate excluded. `dot_general`
# inherited it through `_mul_corners`.
#
# So the finite extremum is asserted to be the correctly directed rounding
# of the exact real one, and the infinite extremum is asserted to keep the
# saturation posture it has always had -- BOTH, because tightening the
# finite side is the fix and tightening the infinite side would be a
# different decision nobody has made.
#
# AND ONE MORE THING, WHICH THIS LIST DID NOT PIN WHEN IT WAS WRITTEN.
# `_mul_corner`'s last line rescues the `0 * +-inf = 0` corner as the exact
# rational `Fraction(0)` rather than the float `0.0` `_prod` hands back,
# *because the convention names a POINT and a point needs no slack*. Delete
# that rescue -- leave `return p` -- and `_extreme_down`/`_extreme_up` take
# the `_down`/`_up` saturation branch on a float zero and bump it off zero.
# Measured on `e3a6475` with the rescue deleted: the ENTIRE zero-dep suite
# stayed green at `2181 passed, 164 skipped`, and `mul([-inf,-1],[-3,0])`
# came back `(-5e-324, inf)` for an image whose infimum is exactly 0 --
# M16's own symptom again, sign-mirrored.
#
# **The two pins that existed were both MASKED, and the masking is the part
# worth reading.** `min`/`max` are FIRST-WINS on a tie, and `Fraction(0)`
# compares equal to a float `0.0`. In `mul([0,0],[1,inf])` the corner list
# is `[0*1, 0*inf, 0*1, 0*inf]` and in `mul([0,inf],[0,3])` -- the row
# directly above -- it is `[0*0, 0*3, inf*0, inf*3]`: in both, a genuinely
# exact `Fraction(0)` corner sits EARLIER than any convention corner, so
# `min`/`max` return the exact one and the mutant is invisible. Neither row
# is deleted, because each still pins what it was written for; what they do
# not pin is the rescue. A pin on THAT has to put the convention corner
# where no exact zero can be selected ahead of it, and the rows below do it
# in two different ways.

_HALF_INFINITE = [
    # (label, result, exact finite endpoint as a Fraction or None, side)
    ("mul([1,inf],[2,3])",
     lambda: iv.mul(scalar(1.0, INF), scalar(2.0, 3.0)), Fraction(2), "lo"),
    ("mul([0,inf],[0,3]) -- M16's zero corner",
     lambda: iv.mul(scalar(0.0, INF), scalar(0.0, 3.0)), Fraction(0), "lo"),
    # --- the `0 * +-inf = 0` corner AS the extremum, both signs ----------
    #
    # ORDER-DEPENDENT, and labelled so rather than trusted quietly: an
    # exact `Fraction(0)` corner IS in each list (`(-1)*0`), and what hands
    # the convention corner to `min`/`max` is that it sits at index 0 and
    # ties go to the first. These two are the witnesses the audit named.
    # They redden on the deleted rescue; they would stop reddening if the
    # corner enumeration order in `_mul_corners` ever changed, which is
    # exactly why the pair below is here too.
    ("mul([-inf,-1],[-3,0]) -- the 0*-inf corner IS the infimum",
     lambda: iv.mul(scalar(-INF, -1.0), scalar(-3.0, 0.0)),
     Fraction(0), "lo"),
    ("mul([-inf,-1],[0,3]) -- and the supremum, sign-mirrored",
     lambda: iv.mul(scalar(-INF, -1.0), scalar(0.0, 3.0)),
     Fraction(0), "hi"),
    # ORDER-INDEPENDENT. BOTH endpoints of the second operand are infinite,
    # so every one of the four corners is either a convention zero or a
    # signed infinity and there is NO exact-zero corner anywhere in the
    # list for a tie-break to reach. No enumeration order can mask these.
    ("mul([0,1],[inf,inf]) -- no exact-zero corner to tie-break against",
     lambda: iv.mul(scalar(0.0, 1.0), scalar(INF, INF)), Fraction(0), "lo"),
    ("mul([-1,0],[inf,inf]) -- ditto, sign-mirrored",
     lambda: iv.mul(scalar(-1.0, 0.0), scalar(INF, INF)), Fraction(0), "hi"),
    ("dot_general([0,1].[inf,inf]) -- the rescue, through _mul_corners",
     lambda: iv.dot_general(
         iv.IntervalArray(shape=(1,), los=(0.0,), his=(1.0,)),
         iv.IntervalArray(shape=(1,), los=(INF,), his=(INF,)),
         (((0,), (0,)), ((), ()))), Fraction(0), "lo"),
    ("mul([-inf,-1],[2,3])",
     lambda: iv.mul(scalar(-INF, -1.0), scalar(2.0, 3.0)),
     Fraction(-2), "hi"),
    ("mul([0.5,inf],[0.25,0.5])",
     lambda: iv.mul(scalar(0.5, INF), scalar(0.25, 0.5)),
     Fraction(1, 8), "lo"),
    ("mul([1,inf],[0.1,0.1]) -- inexact, so it must ROUND and not sit",
     lambda: iv.mul(scalar(1.0, INF), iv.point(0.1)), _f(0.1), "lo"),
    ("div([2,inf],[2,2])",
     lambda: iv.div(scalar(2.0, INF), iv.point(2.0)), Fraction(1), "lo"),
    ("div([-inf,-2],[2,2])",
     lambda: iv.div(scalar(-INF, -2.0), iv.point(2.0)), Fraction(-1), "hi"),
    ("boundary_div([1,1],[0,inf])",
     lambda: iv.boundary_div(iv.point(1.0), scalar(0.0, INF)),
     Fraction(0), "lo"),
    ("boundary_div([-1,-1],[0,inf])",
     lambda: iv.boundary_div(iv.point(-1.0), scalar(0.0, INF)),
     Fraction(0), "hi"),
    ("dot_general([0,inf].[0,3]) -- inherits _mul_corners",
     lambda: iv.dot_general(
         iv.IntervalArray(shape=(1,), los=(0.0,), his=(INF,)),
         iv.IntervalArray(shape=(1,), los=(0.0,), his=(3.0,)),
         (((0,), (0,)), ((), ()))), Fraction(0), "lo"),
]

# The saturation posture, asserted separately and in the same breath, so
# that "the finite side got tighter" cannot be read as "the infinite side
# did too". `_down(+inf)` is maxfloat and `_up(-inf)` is -maxfloat; an
# infinite endpoint is not a real and there is nothing there to represent.
_SATURATION = [
    ("mul([inf,inf],[2,3])",
     lambda: iv.mul(scalar(INF, INF), scalar(2.0, 3.0)),
     (sys.float_info.max, INF)),
    ("mul([-inf,-inf],[2,3])",
     lambda: iv.mul(scalar(-INF, -INF), scalar(2.0, 3.0)),
     (-INF, -sys.float_info.max)),
    ("add([inf,inf],[1,1])",
     lambda: iv.add(scalar(INF, INF), iv.point(1.0)),
     (sys.float_info.max, INF)),
]

# The `_SATURATION` rows whose result `interval.__doc__` QUOTES, by index,
# with the operation the quoting block has to name. `DISCIPLINE`'s `quoted`
# mechanism cannot reach these: a CORRECTLY_ROUNDED row states its exact
# image as a pair of `Fraction`s and `Fraction(inf)` raises, so a case with
# an infinite endpoint cannot be a row of that table at all. The digits are
# derived here for the same reason they are derived there -- the ⊤-escapes
# bullet names this measurement, and a measurement typed into prose beside
# a claim is how that bullet came to carry a false one.
_DOC_QUOTED_SATURATION = [(0, "mul")]


def test_a_half_infinite_operand_costs_only_the_corners_it_touches():
    """Exactness is a property of ONE CORNER; the gate asked about four.

    `mul`, `div` and `boundary_div` tested `_exactable(alo, ahi, blo, bhi)`
    and, on a single infinite endpoint, dropped ALL FOUR corners onto the
    unconditional `_down`/`_up` bump -- including corners that are two
    finite doubles and whose product or quotient is an exact real.
    `_add_lo`/`_add_hi` never had the defect, because they gate on their
    own two operands, which is why `add([1, inf], [2, 3])` already returned
    a lower endpoint of exactly 3.0 while `mul([1, inf], [2, 3])` returned
    1.9999999999999998.

    THE FINITE ENDPOINT IS THE CLAIM, and it is checked as the correctly
    directed rounding of the exact real -- not merely as "outward", which a
    bump also satisfies, and not merely as "equal to the exact value",
    which would be false for the deliberately inexact `0.1` case below.
    Both directions redden: a bump reddens, and so would an inward step.
    """
    bad = []
    for label, drive, exact, side in _HALF_INFINITE:
        r = drive()
        lo, hi = r.los[0], r.his[0]
        got = lo if side == "lo" else hi
        want = _rounded_down(exact) if side == "lo" else _rounded_up(exact)
        if got != want:
            bad.append(
                f"{label}: the FINITE {side} endpoint is {got!r}; the "
                f"correctly directed rounding of the exact real {exact} is "
                f"{want!r}. The bracket is ({lo!r}, {hi!r})")
        other = hi if side == "lo" else lo
        if other != (INF if side == "lo" else -INF):
            bad.append(
                f"{label}: the other endpoint is {other!r}, not the "
                f"infinity this case exists to place beside a finite one")
    for label, drive, want in _SATURATION:
        r = drive()
        if (r.los[0], r.his[0]) != want:
            bad.append(
                f"{label}: an INFINITE extremum returned "
                f"({r.los[0]!r}, {r.his[0]!r}), not {want}. That is the "
                f"saturation convention, and tightening it is a decision "
                f"nobody has made -- it is not part of the corner fix")
    assert not bad, (
        "a half-infinite operand is costing corners it does not touch:\n  "
        + "\n  ".join(bad)
        + "\n\nExactness is a property of the CORNER. Gate it on that "
        "corner's own two operands, as `_add_lo`/`_add_hi` always have."
    )


# --- the debt, anchored to the code and not to a label --------------------

def _scatter_debt_evidence():
    """``scatter_add_rows`` driven on ACCUMULATED elements whose exact real
    total is representable -- over every shape distinction its kernel
    makes, AND over the operand VALUES a fix can be keyed on instead.

    Only accumulated elements: an element no index writes to is a copy of
    the operand and is not bumped, by design, so counting one as evidence
    that the debt is paid would be reading the wrong element.

    THE SHAPE AXES are the two a partial fix can hide behind: ``rowsz``,
    which the kernel branches on, and the number of contributions folded
    into one element, which decides how many bumps that element spends.

    THE VALUE AXIS IS THE OTHER HALF, and it is here because the shape axes
    alone were walked through TWICE. Every endpoint of the seven shape
    cases is drawn from ``{0, 1, 2, 3, 4, 16}`` -- all non-negative, none
    larger than 16, ranks 1 and 2 only. Sign, rank and magnitude were held
    constant across the whole set, so a fix keyed on the operand's VALUES
    rather than on its shape returned the exact total for all seven, after
    which this gate reported the debt PAID and DEMANDED the row move to
    ``EXACT_PER_STEP`` and both docstring entries be deleted. Measured
    twice, each ending green at ``48 passed`` with the debt still owed:

    * the exact-``Fraction`` route taken only where every endpoint of the
      step is ``>= 0.0``. This is the shape a fixer would actually reach
      for, which is why it matters: the M16 story the debt entry itself
      tells is a zero corner going negative, ``sqrt`` already clamps at
      ``max(0.0, ...)`` and ``reduce_sum`` carries a nonnegative clamp, so
      *"take the exact route where the accumulation cannot cross zero"* is
      a natural fix rather than a contrived one. Meanwhile
      ``scatter_add_rows([-1], [-1])`` still returned
      ``(-2.0000000000000004, -1.9999999999999998)`` where ``(-2.0, -2.0)``
      is exact.
    * the same route taken only where ``len(a.shape) <= 2``, which leaves
      every rank-3 operand on the unconditional bump. A second and
      independent key, so the first was not one unlucky predicate.

    Neither is a SOUNDNESS defect -- both branches still round outward,
    which is why nothing else in the tree caught either -- and that is
    exactly the failure mode this gate exists for: the debt half paid, with
    the docstring deleted as though it were paid in full. So the cases
    below also carry negative and mixed-sign endpoints, an accumulation
    whose running total crosses zero, a rank the shape axes never reach,
    and a magnitude outside ``{0 ... 16}``.

    Each case records the OPERAND'S SHAPE as its last field, because those
    four dimensions -- sign, mixed sign, magnitude, rank -- are read back
    off this list in the gate rather than trusted to survive its next edit.
    The sign-crossing case is not one of the four: it is a second witness
    against the ``>= 0.0`` key, which the all-negative case already reddens.

    The question asked of each is coarser than the one :data:`DISCIPLINE`
    asks and is the one the debt is actually about: does the bracket
    contain nothing but the exact total, or is it wider?
    """
    one = iv.scatter_add_rows(_vals((1,), [1.0]), _vals((1,), [1.0]), [0])
    twice = iv.scatter_add_rows(_vals((1,), [0.0]),
                                _vals((2,), [1.0, 1.0]), [0, 0])
    wide = iv.scatter_add_rows(_vals((2, 2), [1.0, 2.0, 3.0, 4.0]),
                               _vals((1, 2), [1.0, 2.0]), [1])
    wide_twice = iv.scatter_add_rows(
        _vals((2, 2), [1.0, 2.0, 3.0, 4.0]),
        _vals((2, 2), [1.0, 2.0, 1.0, 2.0]), [1, 1])
    costed = iv.scatter_add_rows(
        iv.IntervalArray(shape=(1,), los=(0.0,), his=(0.0,)),
        iv.IntervalArray(shape=(1,), los=(0.0,), his=(16.0,)), [0])
    # -- the same two shape axes, driven on the values the seven above
    # hold constant. `neg` and `crossing` are rank 1 and `straddling` is
    # rank 2, all three carrying endpoints no case above reaches; `cube`
    # and `cube_twice` are rank 3, `rowsz` 4.
    neg = iv.scatter_add_rows(_vals((1,), [-1.0]), _vals((1,), [-1.0]), [0])
    crossing = iv.scatter_add_rows(_vals((1,), [1024.0]),
                                   _vals((2,), [-4096.0, 128.0]), [0, 0])
    straddling = iv.scatter_add_rows(
        iv.IntervalArray(shape=(2, 2), los=(1.0, 2.0, -3.0, 4.0),
                         his=(1.0, 2.0, 5.0, 4.0)),
        iv.IntervalArray(shape=(1, 2), los=(-1.0, -8.0),
                         his=(2.0, -8.0)), [1])
    cube = iv.scatter_add_rows(
        _vals((2, 2, 2), [1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0]),
        _vals((1, 2, 2), [-1.0, -2.0, 3.0, -16.0]), [1])
    cube_twice = iv.scatter_add_rows(
        _vals((2, 2, 2), [-1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        _vals((2, 2, 2), [-2.0, -4.0, -6.0, -8.0, 1.0, 3.0, 5.0, 7.0]),
        [0, 0])
    return [
        ("rowsz 1, one contribution: [1] += [1]",
         *_at(one, 0), Fraction(2), Fraction(2), one.shape),
        ("rowsz 1, two contributions: [0] += [1] += [1]",
         *_at(twice, 0), Fraction(2), Fraction(2), twice.shape),
        ("rowsz 2, one contribution: row 1 += [1,2], column 0",
         *_at(wide, 2), Fraction(4), Fraction(4), wide.shape),
        ("rowsz 2, one contribution: row 1 += [1,2], column 1",
         *_at(wide, 3), Fraction(6), Fraction(6), wide.shape),
        ("rowsz 2, two contributions: row 1 += [1,2] += [1,2], column 0",
         *_at(wide_twice, 2), Fraction(5), Fraction(5), wide_twice.shape),
        ("rowsz 2, two contributions: row 1 += [1,2] += [1,2], column 1",
         *_at(wide_twice, 3), Fraction(8), Fraction(8), wide_twice.shape),
        ("the costed case: [0,0] += [0,16]",
         *_at(costed, 0), Fraction(0), Fraction(16), costed.shape),
        ("NEGATIVE, rowsz 1, one contribution: [-1] += [-1]",
         *_at(neg, 0), Fraction(-2), Fraction(-2), neg.shape),
        ("SIGN-CROSSING at 4096 scale, rowsz 1, two contributions: "
         "[1024] += [-4096] += [128]",
         *_at(crossing, 0), Fraction(-2944), Fraction(-2944),
         crossing.shape),
        ("MIXED-SIGN element, rowsz 2, one contribution: "
         "row 1 [-3,5] += [-1,2], column 0",
         *_at(straddling, 2), Fraction(-4), Fraction(7), straddling.shape),
        ("SIGN-FLIPPING element, rowsz 2, one contribution: "
         "row 1 [4,4] += [-8,-8], column 1",
         *_at(straddling, 3), Fraction(-4), Fraction(-4), straddling.shape),
        ("RANK 3, rowsz 4, one contribution: row 1 += [-1,-2,3,-16], "
         "column 0 (-5 + -1)",
         *_at(cube, 4), Fraction(-6), Fraction(-6), cube.shape),
        ("RANK 3, rowsz 4, one contribution: row 1 += [-1,-2,3,-16], "
         "column 3 (8 + -16)",
         *_at(cube, 7), Fraction(-8), Fraction(-8), cube.shape),
        ("RANK 3, rowsz 4, two contributions: row 0 column 2 "
         "(-3 += -6 += 5)",
         *_at(cube_twice, 2), Fraction(-4), Fraction(-4), cube_twice.shape),
    ]


def test_the_scatter_add_rows_debt_is_the_one_the_code_owes():
    """The self-destructing debt, anchored to the RUNNING CODE.

    ``interval.__doc__`` carries two entries about ``scatter_add_rows`` --
    the **A DEBT** entry in the scope block, and the discipline-list bullet
    -- and both of them say, in as many words, to DELETE them when the
    exact-``Fraction`` route lands. A debt that says that has one failure
    mode above all others: becoming a stale carve-out, describing a defect
    the code no longer has, or surviving one the code still does.

    Everything else in this file reaches the docstring's debt entries
    through ``scatter_add_rows``'s ROW -- its declared discipline decides
    whether it is in :data:`MUST_BE_NAMED_IN_THE_DOCSTRING` at all -- and a
    declared discipline is a string somebody types. Measured, with the
    rounding unchanged: moving the row from ``ONE_ULP_BUMPED`` to
    ``NO_ROUNDING`` with a hand-typed endpoint pair left this file green at
    ``dbde454``'s own ``46 passed``, dropped the operation out of the
    must-be-named set, and with it out of the docstring requirement --
    after which BOTH debt entries could be deleted, still green. The debt was not defeated by
    disproving it; it was defeated by refiling it.

    So this gate does not read the row. It drives the operation and asks
    whether the bracket is the exact total or wider, and then requires the
    tree to agree with the answer, in three places at once: the row, the
    scope entry, and the discipline-list bullet. When the fix lands, all
    three come down together -- and this test goes red until they do, which
    is what a self-destructing debt has to do to be one.

    What it drives is BOTH the shapes the kernel branches on and the
    operand VALUES a fix can be keyed on instead. The second half is not
    hypothetical either: driven on the shapes alone, two independent
    value-keyed partial fixes reported the debt paid and took both
    docstring entries down with them, green at ``48 passed``, with
    ``scatter_add_rows([-1], [-1])`` still returning a bumped bracket. Both
    are recorded in :func:`_scatter_debt_evidence`, and the reach of the
    case set is checked below rather than trusted.
    """
    evidence = _scatter_debt_evidence()
    assert evidence, "no case is driven, so this gate measures nothing"

    # THE REACH OF THE CASE SET, READ OFF IT RATHER THAN TRUSTED. Seven
    # cases over the two SHAPE axes were satisfied whole by a fix keyed on
    # the operand's VALUES -- twice, on two independent keys, both recorded
    # in `_scatter_debt_evidence` -- and an edit that quietly trimmed this
    # list back to non-negative, small, rank-1-and-2 operands would re-open
    # that route with this gate still green and still demanding the entries
    # come down.
    #
    # It reads each case's EXACT total and its operand shape, never the
    # endpoints the module returned: those carry the bump, and the bump is
    # what makes the costed case's `-5e-324` look like a negative endpoint
    # while the debt is owed and stop looking like one the moment it is
    # paid. The reach of a case set is a property of the cases.
    exacts = [e for c in evidence for e in (c[3], c[4])]
    thin = sorted(k for k, reached in {
        "an exact total that is negative": any(e < 0 for e in exacts),
        "an element whose exact interval straddles zero":
            any(c[3] < 0 < c[4] for c in evidence),
        "a magnitude outside the {0 ... 16} the shape cases use":
            any(abs(e) > 16 for e in exacts),
        "an operand of rank 3 or more":
            any(len(c[5]) >= 3 for c in evidence),
    }.items() if not reached)
    assert not thin, (
        f"the cases this gate drives no longer reach {thin}, so a fix keyed "
        f"on the operand's VALUES rather than on its shape could satisfy "
        f"every one of them and this gate would then report the debt paid "
        f"and require the docstring entries to be deleted. Two such keys "
        f"are recorded in `_scatter_debt_evidence`, with what each one "
        f"left bumped."
    )

    exact = [c for c in evidence if _f(c[1]) == c[3] and _f(c[2]) == c[4]]
    wider = [c for c in evidence if c not in exact]

    assert not (exact and wider), (
        f"`scatter_add_rows` returns the EXACT representable total for some "
        f"shapes and a widened bracket for others, so it carries two "
        f"endpoint disciplines at once and no single row in DISCIPLINE can "
        f"be true of it:\n"
        f"  exact: {[c[0] for c in exact]}\n"
        f"  wider: {[c[0] for c in wider]}\n"
        f"A fix applied to one shape is not the debt paid; it is the "
        f"docstring's flat claim made false for the other half of them."
    )

    owed = bool(wider)
    declared = DISCIPLINE["scatter_add_rows"].discipline
    # TWO entries, and they are told apart by what each one carries. The
    # discipline-list bullet POINTS AT the scope entry by name -- "the entry
    # headed A DEBT in the scope block above" -- so a block that carries both
    # the words `A DEBT` and the discipline words is that bullet, not the
    # entry it points at. Keyed the other way, deleting the scope entry left
    # this gate green while the pointer went dangling: measured.
    blocks = _doc_blocks()
    scope = [b for b in blocks if _names("scatter_add_rows", b)
             and "A DEBT" in b and ONE_ULP_BUMPED not in b]
    listed = [b for b in blocks
              if _names("scatter_add_rows", b) and ONE_ULP_BUMPED in b]

    if owed:
        assert declared == ONE_ULP_BUMPED, (
            f"`scatter_add_rows` still returns {wider[0][1]!r} where the "
            f"exact total {wider[0][3]} is representable -- the "
            f"unconditional bump, measured -- and its row declares "
            f"{declared!r}. A row that files it anywhere else takes it out "
            f"of MUST_BE_NAMED_IN_THE_DOCSTRING and takes the docstring's "
            f"debt entries down with it, which is refiling the debt rather "
            f"than paying it."
        )
        assert scope, (
            "`scatter_add_rows` still bumps an exact representable total "
            "and `interval.__doc__` has no block that names it and says "
            "A DEBT. That entry carries the measurements and says who is "
            "fixing it; the cost is measured rather than described because "
            "a documented defect nothing drives is how M16 survived its own "
            "fix the first time."
        )
        assert listed, (
            f"`scatter_add_rows` still bumps an exact representable total "
            f"and no block of `interval.__doc__` names it beside the words "
            f"{ONE_ULP_BUMPED!r}. A reader of the discipline list would "
            f"read it off the wrong bullet."
        )
    else:
        assert declared != ONE_ULP_BUMPED, (
            "`scatter_add_rows` now returns the exact total on every driven "
            "shape, so the unconditional bump is gone and its row still "
            "declares it. `interval.__doc__` says where the row goes: read "
            "it off the *exact per step, not per result* bullet instead."
        )
        assert not scope and not listed, (
            f"THE DEBT IS PAID AND THE DOCSTRING STILL OWES IT. "
            f"`scatter_add_rows` returns the exact representable total on "
            f"every driven shape, and `interval.__doc__` still carries "
            f"{len(scope)} A DEBT block(s) and {len(listed)} discipline-list "
            f"block(s) about it. Both say to delete them when this lands. "
            f"Delete them, move the row to EXACT_PER_STEP, and flip "
            f"`tests/test_scatter_add_row_gates.py`'s debt test to VERIFIED "
            f"on both spellings."
        )


# Universal tightness claims, matched against a docstring with its
# whitespace collapsed and its markdown emphasis stripped. BOTH of the
# earlier guards were defeated by exactly those two things -- a line break
# inside the quoted phrase, and a bold marker moved one word to the left --
# so the normalisation is the fix and the patterns are the check. This is
# still a string check and cannot catch every rewording; what makes the
# false universal expensive to re-introduce is the gate ABOVE, which will
# not let the operations it would be false about go unnamed.
_UNIVERSAL_TIGHTNESS = [
    r"(every|all|each)\s+\w*\s*(arithmetic\s+)?endpoints?\s+(is|are)\s+"
    r"correctly\s+directed-rounded",
    r"one\s+deliberate\s+ulp\s+of\s+slack\s+per\s+operation",
    r"(every|all|each)\s+\w*\s*(arithmetic\s+)?endpoints?\s+"
    r"(is|are)\s+exact[- ]when[- ]representable",
]


def _flat_doc() -> str:
    """``interval.__doc__`` with its wrapping and its emphasis taken out.

    Both of the guards this replaces were defeated by exactly those two
    things, so both are normalised away before anything is matched.
    """
    return re.sub(r"\s+", " ", iv.__doc__).replace("**", "").replace("`", "")


def _doc_prose() -> str:
    """The flattened docstring with QUOTED spans removed.

    A retracted wording is kept in this docstring on purpose, and it is
    written inside double quotes; a claim the docstring MAKES is not. So the
    quotation marks are the cut -- what is between a pair of them is
    history, and what is left is what this module is claiming now.
    """
    return re.sub(r'"[^"]*"', " ", _flat_doc())


def _counter_examples() -> list[str]:
    """The operations a universal tightness claim would be false about."""
    return sorted(n for n, o in DISCIPLINE.items()
                  if o.discipline in (ONE_ULP_BUMPED, EXACT_PER_STEP))


def test_the_docstring_makes_no_universal_tightness_claim():
    """The claim that has been made twice, in two spellings, and is false.

    Measured on ``6387a34``, against the guards this replaces: *"one
    deliberate ulp of slack per operation"* and *"**Every** arithmetic
    endpoint is correctly directed-rounded"* were BOTH in the docstring
    while both ``not in doc`` guards passed -- the first defeated by a line
    break and an indent inside the quoted phrase, the second by the bold
    moving from ``directed-rounded`` to ``Every``. Re-inserting the exact
    falsehood, reworded as *"All arithmetic endpoints are correctly
    directed-rounded…"*, gave ``42 passed``.

    A quoted, attributed history of a wording IS allowed -- this docstring
    keeps both retracted sentences on purpose, because a correction with no
    record of what was corrected is how the same claim gets made a third
    time. What is not allowed is one MADE, so the QUOTATION MARK is the cut:
    :func:`_doc_prose` drops what is between a pair of them and the patterns
    are matched against what is left.

    **AND THAT IS A LIMIT, STATED RATHER THAN IMPLIED.** A false universal
    written inside quotation marks would not redden here, and neither would
    a rewording this pattern list does not anticipate -- no string check
    can promise otherwise, and the guards this replaces promised it by
    accident. What makes the claim expensive to re-introduce anyway is
    :func:`test_the_docstring_names_every_operation_a_reader_would_guess_wrong`:
    a universal that is false about the operations in
    :func:`_counter_examples` has to sit in a docstring that names every one
    of them beside the discipline contradicting it.
    """
    hits = [p for p in _UNIVERSAL_TIGHTNESS
            if re.search(p, _doc_prose(), re.I)]
    quoted = re.findall(r'It (?:read|was then replaced by) "', _flat_doc())
    assert len(quoted) >= 2, (
        "the docstring no longer quotes the two wordings it retracted. They "
        "are kept deliberately: the same claim has now been made twice, and "
        "a correction with no record of what it corrected is how it gets "
        "made a third time."
    )
    assert not hits, (
        f"`stelling.interval.__doc__` states a UNIVERSAL tightness claim: "
        f"{hits}. It is false -- {_counter_examples()} "
        f"are counter-examples, driven in this file. Tightness is "
        f"per-operation; say so, and let each operation's own docstring be "
        f"the authority."
    )


def _quotes_the_pair(name: str, lo: float, hi: float, blocks) -> bool:
    """True when some block of the docstring writes ``(lo, hi)`` -- BOTH
    endpoints, in one bracket, in that order -- and names ``name`` too.

    The pair and the block, rather than the endpoint and the file, and both
    halves of that are load-bearing. See
    :func:`test_the_measurements_the_docstring_quotes_are_the_ones_it_would_get`.
    """
    pair = re.compile(r"[(\[]\s*%s\s*,\s*%s\s*[)\]]"
                      % (re.escape(repr(lo)), re.escape(repr(hi))))
    token = DISCIPLINE[name].doc_token or name
    return any(pair.search(b) and _names(token, b) for b in blocks)


def test_the_measurements_the_docstring_quotes_are_the_ones_it_would_get():
    """The digits, kept honest and DERIVED.

    This is what the pin this file replaces did, and it was worth keeping --
    just not on its own. Every measurement quoted in the scope prose is
    produced by the table above rather than typed beside it, so an operation
    whose endpoints move reddens on the SCOPE and not only on the drive.
    ``mul``'s exact corner is in here: before this, ``grep -rn "0.0625"
    tests/`` was empty, and reverting ``mul`` to the unconditional bump left
    the whole pin green.

    **IT READ ENDPOINT BY ENDPOINT, AS A SUBSTRING OF THE WHOLE DOCSTRING,
    AND THAT MADE IT ONE-SIDED.** Widening an operation lengthens its
    reprs, which do not occur -- caught. TIGHTENING one SHORTENS them, and a
    shorter repr is very often already in the text, either as another
    operation's measurement or as a prefix of this one's. Measured twice,
    each with this gate GREEN and only the drive red:

    * ``sqrt`` tightened to return an exact root gives ``(2.0, 2.0)``, and
      ``"2.0"`` is in the docstring -- it is ``div``'s upper corner.
    * the ``scatter_add_rows`` debt PAID gives ``(0.0, 16.0)``, and
      ``"16.0"`` is a PREFIX of the debt entry's own
      ``16.000000000000004``.

    And this test's own docstring used to claim the opposite in as many
    words, while ``eb61aac`` -- the commit that added it -- records under
    ``MUT-5`` that tightening ``sqrt`` reddened the drive and nothing else.
    The claim and the measurement shipped in the same commit.

    So the match is a WHOLE PAIR in ONE BLOCK that also names the operation:
    ``(lo, hi)`` as the text writes it, beside the operation it is a
    measurement of. A shorter repr no longer helps, because both endpoints
    have to be right and they have to be right together, and a coincidence
    somewhere else in the text no longer counts, because it has to be in the
    block that names the operation.

    **THE SATURATION MEASUREMENT IS READ THE SAME WAY**, through
    :data:`_DOC_QUOTED_SATURATION`, because ``DISCIPLINE``'s own ``quoted``
    mechanism structurally cannot reach it: an infinite endpoint has no
    ``Fraction`` image, so no row of that table can state one. The
    ⊤-escapes bullet is where a false classification shipped -- it filed
    the ``0·±inf = 0`` convention, which widens by nothing, under a heading
    saying every entry is wider -- and the measurement that replaced it is
    derived here rather than typed there.

    *What that leg catches, measured, and what it does not.* Tightening the
    saturation posture so ``mul([inf, inf], [2, 3])`` returns
    ``(FMAX, FMAX)`` reddens it: no block writes that pair. **Deleting the
    pair from the ⊤-escapes bullet does NOT redden it** -- the same pair
    stands beside ``mul`` in the discipline-list bullet, and this gate is
    scoped to a block that names the OPERATION, not to one particular
    bullet. So the leg holds the DIGITS current across the whole docstring;
    it does not hold them in any one place. Driven both ways rather than
    assumed, because assuming it was how the two ``mul`` pins one file over
    came to be masked by a tie-break nobody had checked.
    """
    blocks = _doc_blocks()
    marked = {n: op.quoted for n, op in DISCIPLINE.items() if op.quoted}
    assert marked, "no case is marked as quoted, so this gate reads nothing"
    missing = []
    for name, indices in sorted(marked.items()):
        cases = DISCIPLINE[name].drive()
        for i in indices:
            label, lo, hi, _elo, _ehi = cases[i]
            if not _quotes_the_pair(name, lo, hi, blocks):
                missing.append(f"{name}: {label}: ({lo!r}, {hi!r})")
    assert _DOC_QUOTED_SATURATION, "the saturation leg reads nothing"
    for i, name in _DOC_QUOTED_SATURATION:
        label, drive, _want = _SATURATION[i]
        r = drive()
        lo, hi = r.los[0], r.his[0]
        if not _quotes_the_pair(name, lo, hi, blocks):
            missing.append(f"{name}: saturation: {label}: ({lo!r}, {hi!r})")
    assert not missing, (
        f"`stelling.interval.__doc__` scopes its tightness claim with "
        f"measurements this run does not reproduce -- no block of it writes "
        f"these pairs beside the operation they belong to: {missing}. "
        f"Re-decide the SCOPE; do not retype the digits."
    )
