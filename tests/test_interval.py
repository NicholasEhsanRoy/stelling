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

_op("div", CORRECTLY_ROUNDED,
    "exact Fraction quotients on the finite, non-straddling case",
    lambda: [
        ("[0.25,0.5] / [0.25,0.5]", *_e(iv.div(scalar(0.25, 0.5),
                                               scalar(0.25, 0.5))),
         Fraction(1, 2), Fraction(2)),
        ("1 / 3", *_e(iv.div(iv.point(1.0), iv.point(3.0))),
         Fraction(1, 3), Fraction(1, 3)),
    ],
    quoted=(0,))

_op("integer_pow", CORRECTLY_ROUNDED,
    "Fraction(x) ** n is the EXACT power; bumped only on the side that fell",
    lambda: [
        ("[0.5,0.5] ** 2", *_e(iv.integer_pow(scalar(0.5, 0.5), 2)),
         Fraction(1, 4), Fraction(1, 4)),
        ("0.1 ** 3", *_e(iv.integer_pow(iv.point(0.1), 3)),
         _f(0.1) ** 3, _f(0.1) ** 3),
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

_op("scatter_add_rows", ONE_ULP_BUMPED,
    "A DEBT, not a discipline: it folds like reduce_sum and its steps are "
    "bumped unconditionally, so the ulp buys nothing but endpoint "
    "representation -- the M16 defect one operation over. When the "
    "exact-Fraction route lands, this entry moves to EXACT_PER_STEP and the "
    "docstring's debt bullets are deleted",
    lambda: [
        ("[1] +=[1]",
         *_e(iv.scatter_add_rows(_vals((1,), [1.0]), _vals((1,), [1.0]), [0])),
         Fraction(2), Fraction(2)),
        ("[0,0] += [0,16]",
         *_e(iv.scatter_add_rows(
             iv.IntervalArray(shape=(1,), los=(0.0,), his=(0.0,)),
             iv.IntervalArray(shape=(1,), los=(0.0,), his=(16.0,)), [0])),
         Fraction(0), Fraction(16)),
    ],
    quoted=(0, 1))


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
# operations need. (`abs_`, `maximum`, `minimum`, `hull`, `join`, `meet`'s
# arithmetic-free siblings and `int_div` compute, but they compute with
# `min`/`max`/truncation over doubles, which are exact.)

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


# --- the four gates -------------------------------------------------------

def _public_operations() -> set[str]:
    """Every public function ``stelling.interval`` DEFINES.

    Read off the module, never off a list in a docstring: a list somebody
    forgets to extend is exactly how the universal claim went wrong twice.
    ``__module__`` filters out the names imported into the module's
    namespace (``dataclass``, ``NamedTuple``), so an import cannot silently
    add a row.
    """
    return {
        name for name in dir(iv)
        if not name.startswith("_")
        and inspect.isfunction(getattr(iv, name))
        and getattr(iv, name).__module__ == iv.__name__
    }


def test_every_public_operation_of_this_module_declares_a_discipline():
    """A new operation FORCES a classification.

    Measured on ``6387a34``: adding a public ``half()`` to
    ``stelling.interval`` that rounds both endpoints INWARD left the entire
    zero-dep suite green. The module docstring called outward rounding *"the
    invariant a new operation added to this module has to preserve"* and
    nothing enforced it, in either direction -- neither that the new
    operation preserves it nor that anyone looked.
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


def _doc_blocks() -> list[str]:
    """The docstring's paragraphs and top-level bullets, flattened.

    Split on blank lines AND on the top-level ``* `` marker, so that an
    operation and the words naming its discipline have to be in the SAME
    block for the block to count as saying anything about it. Wrapping and
    emphasis are normalised away first, for the reason they always are here:
    both of the guards this file replaces were defeated by one of them.
    """
    raw = re.split(r"(?m)\n\s*\n|^\* ", iv.__doc__)
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
    mention in the place that says what the operation does.

    That couples the discipline constants above to the docstring's own
    vocabulary, deliberately -- they are written to BE its vocabulary, and a
    docstring that stops using the words it classifies operations with has
    stopped classifying them.

    The safe directions are deliberately not required: ``NO_ROUNDING`` and
    ``NO_ENDPOINTS`` operations are absent from this requirement, because
    demanding that the docstring list all forty of them would turn it into
    the census it says, correctly, that it is not.
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


def test_the_measurements_the_docstring_quotes_are_the_ones_it_would_get():
    """The digits, kept honest and DERIVED.

    This is what the pin this file replaces did, and it was worth keeping --
    just not on its own. Every endpoint quoted in the scope prose is
    produced by the table above rather than typed beside it, so tightening
    one of these operations reddens on the SCOPE and not only on the digit.
    ``mul``'s exact corner is in here: before this, ``grep -rn "0.0625"
    tests/`` was empty, and reverting ``mul`` to the unconditional bump left
    the whole pin green.
    """
    doc = iv.__doc__
    marked = {n: op.quoted for n, op in DISCIPLINE.items() if op.quoted}
    assert marked, "no case is marked as quoted, so this gate reads nothing"
    missing = []
    for name, indices in sorted(marked.items()):
        cases = DISCIPLINE[name].drive()
        for i in indices:
            label, lo, hi, _elo, _ehi = cases[i]
            for endpoint in (lo, hi):
                if repr(endpoint) not in doc:
                    missing.append(f"{name}: {label}: {endpoint!r}")
    assert not missing, (
        f"`stelling.interval.__doc__` scopes its tightness claim with "
        f"endpoints this run does not reproduce: {missing}. Re-decide the "
        f"SCOPE; do not retype the digit."
    )
