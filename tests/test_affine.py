# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The affine (zonotope) refinement: forms, decisions, declines, gauge — no jax.

Hand-built IR queries in the :mod:`stelling.ir` conventions of
``test_obligation_slice.py``. Every known answer here is public
mathematics hand-checked in the test body; every decline path proves the
obligation stays byte-identical with the reason (and the offending
primitive) quoted; and the fidelity gauge measures the claimed behaviors
against a mutation battery (L21: discriminating power is measured, not
asserted). Pure-form and slice-level tests run in both venvs.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from unittest import mock

import pytest

import stelling.affine as affine
from stelling.affine import (
    AFFINE_SUPPORTED,
    DISCHARGED_BY_AFFINE,
    AffineForm,
    SymbolTable,
    concretization,
    decide_slice,
    declared_form,
    mul_forms,
    refine_propagation,
)
from stelling.fidelity import gauge
from stelling.obligation import ObligationSlice, slice_unknown_obligations
from stelling.propagate import TRANSFERS, interval_env, propagate
from stelling.verdict import make_verdict

from test_obligation_slice import BOOL, F64, any_eqn, close, eqn, lit, var

from stelling import ir

VERSIONS = dict(
    stelling_version="test", jax_version="none", precision_config="test"
)


def aval(shape, dt="float64"):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dt)


def avar(i, shape):
    return ir.Var(id=i, aval=aval(shape))


def sole_slice(q) -> ObligationSlice:
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], ObligationSlice)
    return items[0]


def refined(q):
    p = propagate(q)
    assert all(o.status == "unknown" for o in p.obligations), (
        "these cases must be interval-undecided, or affine proves nothing"
    )
    return refine_propagation(q, p)


# --- the registry and its totality discipline --------------------------------


def test_registry_is_subset_of_the_censused_transfers():
    assert AFFINE_SUPPORTED <= set(TRANSFERS)


def test_every_registry_name_has_a_dedicated_behavior_test():
    """The L22-style correspondence walk: a primitive may not sit in
    AFFINE_SUPPORTED without a behavior test of its own name."""
    for name in sorted(AFFINE_SUPPORTED):
        assert f"test_op_{name}" in globals(), (
            f"registry name {name!r} has no dedicated behavior test "
            f"test_op_{name}"
        )


# --- per-primitive behavior (one dedicated test per registry name) -----------


def _both_directions(build_g):
    """A query stating g >= 0 AND -g >= 0 (via le) for a slack ``g`` that
    is exactly 0 over the box — the exact-cancellation acceptance shape."""
    eqns, g = build_g()
    p1, p2, o1, o2 = var(90, BOOL), var(91, BOOL), var(92, BOOL), var(93, BOOL)
    return close(
        eqns
        + [
            eqn("ge", [g, lit(0.0)], p1),
            eqn("stelling_assert", [p1], o1),
            eqn("le", [g, lit(0.0)], p2),
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )


def _assert_exact_zero_discharges(q):
    rp, rep = refined(q)
    assert [o.status for o in rp.obligations] == ["discharged", "discharged"]
    for o in rp.obligations:
        assert o.detail == DISCHARGED_BY_AFFINE
    return rep


def test_op_add():
    # (x + y) - y - x is exactly 0; intervals lose the correlation
    def build():
        x, y, s, d1, g = var(0), var(1), var(2), var(3), var(4)
        return [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("add", [x, y], s),
            eqn("sub", [s, y], d1),
            eqn("sub", [d1, x], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "add" in rep.ops_used


def test_op_sub():
    # KA 1: x - x over [0, 1] — interval says [-1, 1]; affine says 0
    def build():
        x, g = var(0), var(1)
        return [any_eqn(x, 0.0, 1.0), eqn("sub", [x, x], g)], g

    _assert_exact_zero_discharges(_both_directions(build))


def test_op_neg():
    # neg(x) + x is exactly 0
    def build():
        x, n, g = var(0), var(1), var(2)
        return [
            any_eqn(x, -2.0, 3.0),
            eqn("neg", [x], n),
            eqn("add", [n, x], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "neg" in rep.ops_used


def test_op_mul():
    # KA 3, the load-bearing v1 capability: the commuted-product pair
    def build():
        x, y, p, q, g = var(0), var(1), var(2), var(3), var(4)
        return [
            any_eqn(x, 0.1, 0.3),
            any_eqn(y, 0.1, 0.3),
            eqn("mul", [x, y], p),
            eqn("mul", [y, x], q),
            eqn("sub", [p, q], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "mul" in rep.ops_used


def test_op_integer_pow():
    # x**2 - x*x is exactly 0: the y=2 self-product shares the canonical
    # residue symbol with the explicit product, so the two spellings cancel
    def build():
        x, sq, p, g = var(0), var(1), var(2), var(3)
        return [
            any_eqn(x, -1.0, 2.0),
            eqn("integer_pow", [x], sq, params=(("y", 2),)),
            eqn("mul", [x, x], p),
            eqn("sub", [sq, p], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "integer_pow" in rep.ops_used


def test_op_integer_pow_only_y2_is_modelled():
    x, cb, pr, ob = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("integer_pow", [x], cb, params=(("y", 3),)),
            eqn("ge", [cb, lit(-9.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    p = propagate(q)
    # interval discharges x**3 >= -9 over [0, 1] itself, so pose the
    # slice directly: the y=3 decline is the claim under test
    assert slice_unknown_obligations(q, p, interval_env(q)) == ()
    from stelling.obligation import slice_obligation

    sl = slice_obligation(q, 0, interval_env(q))
    assert isinstance(sl, ObligationSlice)
    d = decide_slice(sl)
    assert d.status == "declined"
    assert "'integer_pow' exponent y=3" in d.detail
    assert "only y=2" in d.detail


def test_op_reduce_sum():
    # sum(a) - sum(a): four shared elements, exactly 0
    def build():
        a = avar(0, (3,))
        s1, s2, g = var(1), var(2), var(3)
        return [
            any_eqn(a, 0.0, 1.0, shape=(3,)),
            eqn("reduce_sum", [a], s1, params=(("axes", (0,)),)),
            eqn("reduce_sum", [a], s2, params=(("axes", (0,)),)),
            eqn("sub", [s1, s2], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "reduce_sum" in rep.ops_used


def test_op_reshape():
    # sum over (2, 2) equals sum over its (4,) reshape
    def build():
        a = avar(0, (2, 2))
        r = avar(1, (4,))
        s1, s2, g = var(2), var(3), var(4)
        return [
            any_eqn(a, 0.0, 1.0, shape=(2, 2)),
            eqn("reshape", [a], r, params=(("new_sizes", (4,)), ("dimensions", None))),
            eqn("reduce_sum", [a], s1, params=(("axes", (0, 1)),)),
            eqn("reduce_sum", [r], s2, params=(("axes", (0,)),)),
            eqn("sub", [s1, s2], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "reshape" in rep.ops_used


def test_op_slice():
    # sum(a[0:2]) + sum(a[2:4]) - sum(a) is exactly 0
    def build():
        a = avar(0, (4,))
        h1, h2 = avar(1, (2,)), avar(2, (2,))
        s1, s2, sa, t, g = var(3), var(4), var(5), var(6), var(7)
        return [
            any_eqn(a, 0.0, 1.0, shape=(4,)),
            eqn("slice", [a], h1, params=(
                ("start_indices", (0,)), ("limit_indices", (2,)), ("strides", None))),
            eqn("slice", [a], h2, params=(
                ("start_indices", (2,)), ("limit_indices", (4,)), ("strides", None))),
            eqn("reduce_sum", [h1], s1, params=(("axes", (0,)),)),
            eqn("reduce_sum", [h2], s2, params=(("axes", (0,)),)),
            eqn("reduce_sum", [a], sa, params=(("axes", (0,)),)),
            eqn("add", [s1, s2], t),
            eqn("sub", [t, sa], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "slice" in rep.ops_used


def test_op_squeeze():
    # squeeze((1,) -> ()) preserves the element: s - a[0] via sum
    def build():
        a = avar(0, (1,))
        s = var(1)
        sa, g = var(2), var(3)
        return [
            any_eqn(a, 0.0, 1.0, shape=(1,)),
            eqn("squeeze", [a], s, params=(("dimensions", (0,)),)),
            eqn("reduce_sum", [a], sa, params=(("axes", (0,)),)),
            eqn("sub", [s, sa], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "squeeze" in rep.ops_used


def test_op_concatenate():
    # KA 6: the rolling-average chain — 0.5·(a + roll(a, 1)) sums to
    # sum(a) exactly (each element appears twice at coefficient 0.5);
    # interval loses every correlation on the way
    q = _ka6_query()
    rp, rep = refined(q)
    assert [o.status for o in rp.obligations] == ["discharged", "discharged"]
    for o in rp.obligations:
        assert o.detail == DISCHARGED_BY_AFFINE
    assert {"slice", "concatenate", "add", "mul", "reduce_sum"} <= set(
        rep.ops_used
    )


def test_op_stack():
    # sum(stack([x, y])) - (x + y) is exactly 0
    def build():
        x, y = var(0), var(1)
        st = avar(2, (2,))
        s, t, g = var(3), var(4), var(5)
        return [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("stack", [x, y], st, params=(("axis", 0),)),
            eqn("reduce_sum", [st], s, params=(("axes", (0,)),)),
            eqn("add", [x, y], t),
            eqn("sub", [s, t], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "stack" in rep.ops_used


def test_op_broadcast_in_dim():
    # sum(broadcast(x, (3,))) - 3·x is exactly 0: the three broadcast
    # elements SHARE x's symbol (one input, one ε — the sharing intervals
    # cannot express)
    def build():
        x = var(0)
        b = avar(1, (3,))
        s, t, g = var(2), var(3), var(4)
        return [
            any_eqn(x, 0.0, 1.0),
            eqn("broadcast_in_dim", [x], b, params=(
                ("shape", (3,)), ("broadcast_dimensions", ()))),
            eqn("reduce_sum", [b], s, params=(("axes", (0,)),)),
            eqn("mul", [x, lit(3.0)], t),
            eqn("sub", [s, t], g),
        ], g

    rep = _assert_exact_zero_discharges(_both_directions(build))
    assert "broadcast_in_dim" in rep.ops_used


# --- the known answers -------------------------------------------------------


def _ka1_query():
    x, g = var(0), var(1)
    p1, p2, o1, o2 = var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("sub", [x, x], g),
            eqn("ge", [g, lit(0.0)], p1),
            eqn("stelling_assert", [p1], o1),
            eqn("le", [g, lit(0.0)], p2),
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )


def test_ka1_x_minus_x_interval_straddles_affine_discharges():
    q = _ka1_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "unknown"]
    rp, rep = refine_propagation(q, p)
    assert [o.status for o in rp.obligations] == ["discharged", "discharged"]
    for o in rp.obligations:
        assert o.detail == DISCHARGED_BY_AFFINE
    assert rep.discharged == (0, 1)
    # the slack really is exactly [0, 0]: no unconditional bump enters
    sl = slice_unknown_obligations(q, p, interval_env(q))[0]
    d = decide_slice(sl)
    assert d.ranges == ((0.0, 0.0),)


def test_ka2_regrouped_sum_is_exactly_zero():
    def build():
        x, y, s, d, g = var(0), var(1), var(2), var(3), var(4)
        return [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("add", [x, y], s),
            eqn("sub", [s, x], d),
            eqn("sub", [d, y], g),
        ], g

    _assert_exact_zero_discharges(_both_directions(build))


def _ka3_query():
    x, y, p, q_, d1, d2 = var(0), var(1), var(2), var(3), var(4), var(5)
    b1, b2, o1, o2 = var(6, BOOL), var(7, BOOL), var(8, BOOL), var(9, BOOL)
    return close(
        [
            any_eqn(x, 0.1, 0.3),
            any_eqn(y, 0.1, 0.3),
            eqn("mul", [x, y], p),
            eqn("mul", [y, x], q_),
            eqn("sub", [p, q_], d1),
            eqn("sub", [q_, p], d2),
            eqn("ge", [d1, lit(0.0)], b1),
            eqn("stelling_assert", [b1], o1),
            eqn("ge", [d2, lit(0.0)], b2),
            eqn("stelling_assert", [b2], o2),
        ],
        [o1, o2],
    )


def test_ka3_commuted_products_cancel_exactly():
    """The load-bearing v1 capability, pinned hard: p = x·y and q = y·x
    over x, y ∈ [0.1, 0.3] straddle under intervals in BOTH directions
    and cancel exactly under the canonicalized product handling."""
    q = _ka3_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "unknown"]
    rp, _ = refine_propagation(q, p)
    assert [o.status for o in rp.obligations] == ["discharged", "discharged"]
    for o in rp.obligations:
        assert o.detail == DISCHARGED_BY_AFFINE


def test_ka3_form_level_commuted_products_are_identical():
    """mul(a, b) and mul(b, a) must be the IDENTICAL AffineForm — same
    coefficients, same residue symbol, same err."""
    t = SymbolTable(2)
    fx = declared_form(0.1, 0.3, 0)
    fy = declared_form(0.1, 0.3, 1)
    ax, ay = avar(0, ()), avar(1, ())
    kx = affine._operand_key(fx, ax, 0)
    ky = affine._operand_key(fy, ay, 0)
    p = mul_forms(t, fx, kx, fy, ky)
    q = mul_forms(t, fy, ky, fx, kx)
    assert p == q
    d = affine.sub_forms(p, q)
    assert d == AffineForm(center=0.0, coeffs=(), err=0.0)
    assert concretization(d) == (0.0, 0.0)


def _ka4_query():
    x, y, m = var(0), var(1), var(2)
    pr, ob = var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(x, -1.0, 1.0),
            any_eqn(y, -1.0, 1.0),
            eqn("mul", [x, y], m),
            eqn("ge", [m, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )


def test_ka4_true_straddle_stays_undecided_with_the_range_noted():
    q = _ka4_query()
    p = propagate(q)
    rp, rep = refine_propagation(q, p)
    # x·y over [-1,1]² truly straddles ([-1, 1] is the exact range at the
    # corners (±1, ∓1)): affine must remain undecided and fall through
    assert rp.obligations == p.obligations  # byte-identical fall-through
    assert rep.undecided == (0,)
    assert any(
        "did not separate" in n and "[-1.0, 1.0]" in n for n in rp.notes
    )


def _ka5_query():
    a, c, b = var(0), var(1), var(2)
    tr, tr2, ac, bb, det, rhs = var(3), var(4), var(5), var(6), var(7), var(8)
    pred, out = var(9, BOOL), var(10, BOOL)
    return close(
        [
            any_eqn(a, 1.0, 2.0),
            any_eqn(c, 1.0, 2.0),
            any_eqn(b, -0.5, 0.5),
            eqn("add", [a, c], tr),
            eqn("mul", [tr, tr], tr2),
            eqn("mul", [a, c], ac),
            eqn("mul", [b, b], bb),
            eqn("sub", [ac, bb], det),
            eqn("mul", [det, lit(10.125)], rhs),
            eqn("le", [tr2, rhs], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_ka5_honest_negative_conditioning_shape_stays_undecided():
    """The probe's Part 2/3 shape: (a+c)² ≤ (a·c − b·b)·10.125 over
    a, c ∈ [1, 2], b ∈ [−0.5, 0.5] is TRUE (a QF_NRA validity) but
    dependency-shaped AND quadratic past plain affine — v1 must NOT
    close it. Hand derivation of the affine range of g = rhs − tr²:
    tr = 3 + 0.5εa + 0.5εc; tr² = 9 + 3εa + 3εc ± 1 (residue rad² = 1);
    a·c = 2.25 + 0.75εa + 0.75εc ± 0.25; b·b = 0 ± 0.25;
    det = a·c − b·b; rhs = det·10.125 (exact dyadic scale), so
    g = 13.78125 + 4.59375εa + 4.59375εc − 1·s₁ + 2.53125·s₂ − 2.53125·s₃
    with |g| ≤ 13.78125 ± (4.59375·2 + 1 + 2.53125·2) = 13.78125 ± 15.25:
    [−1.46875, 29.03125], which straddles 0. If an implementation DOES
    close this, the residue accounting is wrong — re-derive by hand
    before believing it."""
    q = _ka5_query()
    p = propagate(q)
    rp, rep = refine_propagation(q, p)
    assert rp.obligations == p.obligations
    assert rep.undecided == (0,)
    assert any("[-1.46875, 29.03125]" in n for n in rp.notes)


def _ka6_query():
    a = avar(0, (4,))
    last, first3 = avar(1, (1,)), avar(2, (3,))
    rolled, s, avg = avar(3, (4,)), avar(4, (4,)), avar(5, (4,))
    s1, s2, d1, d2 = var(6), var(7), var(8), var(9)
    p1, p2, o1, o2 = var(10, BOOL), var(11, BOOL), var(12, BOOL), var(13, BOOL)
    return close(
        [
            any_eqn(a, 0.0, 1.0, shape=(4,)),
            eqn("slice", [a], last, params=(
                ("start_indices", (3,)), ("limit_indices", (4,)), ("strides", None))),
            eqn("slice", [a], first3, params=(
                ("start_indices", (0,)), ("limit_indices", (3,)), ("strides", None))),
            eqn("concatenate", [last, first3], rolled, params=(("dimension", 0),)),
            eqn("add", [a, rolled], s),
            eqn("mul", [s, lit(0.5)], avg),
            eqn("reduce_sum", [avg], s1, params=(("axes", (0,)),)),
            eqn("reduce_sum", [a], s2, params=(("axes", (0,)),)),
            eqn("sub", [s1, s2], d1),
            eqn("sub", [s2, s1], d2),
            eqn("ge", [d1, lit(0.0)], p1),
            eqn("stelling_assert", [p1], o1),
            eqn("ge", [d2, lit(0.0)], p2),
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )


def test_ka6_rolling_average_sum_identity():
    """Σ 0.5·(a_i + a_{i-1}) = Σ a_i exactly (every element contributes
    twice at coefficient 0.5, and 0.5 scaling is dyadic-exact) — the
    hand-derivable exact range of the difference is [0, 0]. Interval
    propagation loses the correlation entirely ([0,4] − [0,4])."""
    q = _ka6_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "unknown"]
    rp, _ = refine_propagation(q, p)
    assert [o.status for o in rp.obligations] == ["discharged", "discharged"]


def _ka7_query():
    x, y, d, g = var(0), var(1), var(2), var(3)
    pr, ob = var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 1.0, 2.0),
            eqn("div", [x, y], d),
            eqn("sub", [d, d], g),
            eqn("ge", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )


def test_ka7_div_declines_wholly_and_falls_through_byte_identical():
    q = _ka7_query()
    p = propagate(q)
    rp, rep = refine_propagation(q, p)
    assert rp.obligations == p.obligations  # byte-identical for the obligation
    assert rep.declined == (
        (0, "primitive 'div' is outside AFFINE_SUPPORTED"),
    )
    assert (
        "assert #0: affine refinement declined: primitive 'div' is outside "
        "AFFINE_SUPPORTED" in rp.notes
    )


def test_ka7_min_max_and_transcendentals_decline_quoted():
    def one(prim, params=()):
        x, m, g = var(0), var(1), var(2)
        pr, ob = var(3, BOOL), var(4, BOOL)
        ins = [x, lit(0.5)] if prim in ("max", "min") else [x]
        q = close(
            [
                any_eqn(x, 0.0, 1.0),
                eqn(prim, ins, m, params=params),
                eqn("sub", [m, m], g),
                eqn("ge", [g, lit(0.0)], pr),
                eqn("stelling_assert", [pr], ob),
            ],
            [ob],
        )
        return refine_propagation(q, propagate(q))[1]

    for prim in ("max", "min"):
        rep = one(prim)
        assert rep.declined[0][1] == (
            f"primitive '{prim}' is outside AFFINE_SUPPORTED"
        )
    # exp is outside even the emission set: the slice itself is
    # unavailable and the decline quotes the slicer's reason, primitive
    # named
    rep = one("exp")
    (idx, reason), = rep.declined
    assert "the obligation slice is unavailable" in reason
    assert "'exp'" in reason


def test_ka8_err_overflow_declines_honestly():
    x, y, m = var(0), var(1), var(2)
    pr, ob = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, -1e300, 1e300),
            any_eqn(y, -1e300, 1e300),
            eqn("mul", [x, y], m),
            eqn("ge", [m, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    p = propagate(q)
    rp, rep = refine_propagation(q, p)
    assert rp.obligations == p.obligations  # UNKNOWN preserved, honestly
    (idx, reason), = rep.declined
    assert "overflowed the double range" in reason


def test_ka8_huge_scale_linear_cancellation_still_exact():
    def build():
        x, g = var(0), var(1)
        return [any_eqn(x, 1e300, 2e300), eqn("sub", [x, x], g)], g

    _assert_exact_zero_discharges(_both_directions(build))


def test_ka8_subnormal_radii_containment_and_no_silent_widening():
    # x + y ≥ 2.5e-323 straddles under intervals AND under affine (truly
    # undecidable from the box); the slack ranges must still contain the
    # dense-sampled truth at subnormal scale
    x, y, s = var(0), var(1), var(2)
    pr, ob = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 5e-324, 2e-323),
            any_eqn(y, 5e-324, 2e-323),
            eqn("add", [x, y], s),
            eqn("ge", [s, lit(2.5e-323)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    sl = sole_slice(q)
    d = decide_slice(sl)
    assert d.status == "unknown"
    ((lo, hi),) = d.ranges
    thr = Fraction(2.5e-323)
    for xv in (Fraction(5e-324), Fraction(2e-323)):
        for yv in (Fraction(5e-324), Fraction(2e-323)):
            assert Fraction(lo) <= xv + yv - thr <= Fraction(hi)
    # ...and exact cancellation still works at subnormal scale
    def build():
        xx, g = var(0), var(1)
        return [any_eqn(xx, 5e-324, 2e-323), eqn("sub", [xx, xx], g)], g

    _assert_exact_zero_discharges(_both_directions(build))


def test_ka8_adversarial_magnitude_containment():
    # 1e300-scale coefficients through a linear chain: concretization
    # must still contain the exact corner values
    f = declared_form(1e300, 1.5e300, 0)
    g = declared_form(-1e300, 1e300, 1)
    s = affine.add_forms(f, g)
    lo, hi = concretization(s)
    for a in (Fraction(1e300), Fraction(1.5e300)):
        for b in (Fraction(-1e300), Fraction(1e300)):
            assert Fraction(lo) <= a + b <= Fraction(hi)


# --- decision-rule boundaries ------------------------------------------------


def test_strict_root_comparison_declines():
    x, g = var(0), var(1)
    pr, ob = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("sub", [x, x], g),
            eqn("gt", [g, lit(-1.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    rp, rep = refine_propagation(q, propagate(q))
    (idx, reason), = rep.declined
    assert "'gt'" in reason and "strict half-space" in reason
    assert rp.obligations == propagate(q).obligations


def test_eq_root_declines():
    x, g = var(0), var(1)
    pr, ob = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("sub", [x, x], g),
            eqn("eq", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    _, rep = refine_propagation(q, propagate(q))
    (idx, reason), = rep.declined
    assert "'eq'" in reason and "closed-half-space" in reason


def _refute_query():
    x, d, g = var(0), var(1), var(2)
    pr, ob = var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("sub", [x, x], d),
            eqn("sub", [d, lit(1.0)], g),
            eqn("ge", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )


def test_affine_set_level_refutation():
    """(x − x) − 1 ≥ 0 is definitely false at EVERY point (the slack is
    exactly −1); interval straddles ([−2, 0]). The violated-over-set
    direction is licensed by containment: affine range ⊆ (−∞, 0)."""
    q = _refute_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    rp, rep = refine_propagation(q, p)
    assert [o.status for o in rp.obligations] == ["violated-over-set"]
    assert "affine refinement" in rp.obligations[0].detail
    assert "no witness" in rp.obligations[0].detail
    assert rep.violated == (0,)
    v = make_verdict(q, rp, refinement=rep, **VERSIONS)
    assert v.status == "REFUTED"
    assert "Not a witness" in v.render()  # the set-level rendering class


def test_ieee_semantics_declines_wholly():
    q = _ka1_query()
    p = propagate(q, semantics="ieee")
    rp, rep = refine_propagation(q, p)
    assert rp.obligations == p.obligations
    assert all("semantics='ieee'" in r for _, r in rep.declined)
    assert all("models exact real arithmetic" in r for _, r in rep.declined)


def test_constrained_assume_declines_wholly():
    q = _ka1_query()
    p = propagate(q)
    constrained = dataclasses.replace(
        p, coverage=dataclasses.replace(p.coverage, constrained=1)
    )
    rp, rep = refine_propagation(q, constrained)
    assert all("constrained 1 assume(s)" in r for _, r in rep.declined)
    assert rp.obligations == constrained.obligations


def test_nothing_unknown_returns_propagation_unchanged():
    x, pr, ob = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("ge", [x, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["discharged"]
    rp, rep = refine_propagation(q, p)
    assert rp is p
    assert rep.attempted == ()


def test_infinite_declaration_bounds_decline():
    x, g = var(0), var(1)
    pr, ob = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, -math.inf, math.inf),
            eqn("sub", [x, x], g),
            eqn("ge", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    _, rep = refine_propagation(q, propagate(q))
    (idx, reason), = rep.declined
    assert "not finite" in reason and "bounded declared boxes" in reason


def test_array_root_elementwise_decision():
    """The obligation is the elementwise universal claim: an array root
    discharges only when EVERY element's slack range clears 0, and one
    definite element does not carry a mixed root to a definite status."""
    a = avar(0, (2,))
    d = avar(1, (2,))
    pr = ir.Var(id=2, aval=aval((2,), "bool"))
    ob = ir.Var(id=3, aval=aval((2,), "bool"))
    q = close(
        [
            any_eqn(a, 0.0, 1.0, shape=(2,)),
            eqn("sub", [a, a], d),
            eqn("ge", [d, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    rp, _ = refine_propagation(q, propagate(q))
    assert [o.status for o in rp.obligations] == ["discharged"]
    # mixed: element 0 is exactly 0 (true), element 1 truly straddles —
    # the universal claim must stay undecided
    x, y = var(0), var(1)
    z, m = avar(2, (1,)), avar(3, (1,))
    g2 = avar(4, (2,))
    d0 = var(5)
    pr2 = ir.Var(id=6, aval=aval((2,), "bool"))
    ob2 = ir.Var(id=7, aval=aval((2,), "bool"))
    q2 = close(
        [
            any_eqn(x, -1.0, 1.0),
            any_eqn(y, -1.0, 1.0),
            eqn("sub", [x, x], d0),
            eqn("broadcast_in_dim", [d0], z, params=(
                ("shape", (1,)), ("broadcast_dimensions", ()))),
            eqn("mul", [x, y], var(8)),
            eqn("broadcast_in_dim", [var(8)], m, params=(
                ("shape", (1,)), ("broadcast_dimensions", ()))),
            eqn("concatenate", [z, m], g2, params=(("dimension", 0),)),
            eqn("ge", [g2, lit(0.0)], pr2),
            eqn("stelling_assert", [pr2], ob2),
        ],
        [ob2],
    )
    p2 = propagate(q2)
    rp2, rep2 = refine_propagation(q2, p2)
    assert rp2.obligations == p2.obligations
    assert rep2.undecided == (0,)
    assert any("1/2 element(s) undecided" in n for n in rp2.notes)


# --- verdict assembly with the refinement record -----------------------------


def test_absence_line_names_both_layers_when_affine_decided():
    q = _ka1_query()
    rp, rep = refine_propagation(q, propagate(q))
    v = make_verdict(q, rp, refinement=rep, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.solver.reason == (
        "no solver invoked: every obligation was judged by outward-rounded "
        "interval arithmetic with affine (zonotope) refinement — 2 "
        "obligation(s) decided by the affine domain"
    )
    assert v.stamp.assumptions[-1].startswith("affine refinement enabled (")
    assert "12 primitives" in v.stamp.assumptions[-1]
    # the arithmetic line names the deciding abstraction (audit F4)
    assert v.stamp.arithmetic_mode == (
        "interval/f64/outward-1ulp (stelling.interval) + affine/zonotope "
        "refinement (stelling.affine, same outward kernel)"
    )


def test_absence_line_keeps_interval_alone_only_with_honest_suffix():
    q = _ka4_query()
    rp, rep = refine_propagation(q, propagate(q))
    v = make_verdict(q, rp, refinement=rep, **VERSIONS)
    assert v.stamp.solver.reason == (
        "no solver invoked: every obligation was judged by outward-rounded "
        "interval arithmetic alone (affine refinement was enabled and "
        "decided nothing: 1 obligation(s) attempted)"
    )
    # nothing decided: the arithmetic line stays interval-only (audit F4)
    assert v.stamp.arithmetic_mode == (
        "interval/f64/outward-1ulp (stelling.interval)"
    )


def test_saturation_is_sound_and_disclosed():
    """Audit F3: a center beyond the double range SATURATES through the
    exact-bracket kernel with the full distance in err (measured: the
    snap can never return a non-finite double), and the obligation's
    notes disclose it. The T9 shape: x + x over [1e308, 1.5e308]."""
    from stelling.obligation import slice_obligation

    x, s = var(0), var(1)
    pr, ob = var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1e308, 1.5e308),
            eqn("add", [x, x], s),
            eqn("ge", [s, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    # interval discharges this itself; pose the slice directly
    sl = slice_obligation(q, 0, interval_env(q))
    assert isinstance(sl, ObligationSlice)
    d = decide_slice(sl)
    assert d.status == "discharged" and d.saturated
    ((lo, hi),) = d.ranges
    assert lo == 5.953862697246314e307 and hi == math.inf
    for xv in (Fraction(1e308), Fraction(1.5e308)):
        assert Fraction(lo) <= 2 * xv  # containment held under saturation
    # pipeline-level: a DEFINITE affine decision that saturated carries
    # the per-obligation disclosure note
    u1, dd, g = var(1), var(4), var(5)
    pr2, ob2 = var(6, BOOL), var(7, BOOL)
    q2 = close(
        [
            any_eqn(x, 1e308, 1.5e308),
            eqn("add", [x, x], u1),
            eqn("sub", [u1, u1], dd),
            eqn("add", [dd, lit(1.5e308)], g),
            eqn("ge", [g, lit(0.0)], pr2),
            eqn("stelling_assert", [pr2], ob2),
        ],
        [ob2],
    )
    p2 = propagate(q2)
    assert [o.status for o in p2.obligations] == ["unknown"]
    rp2, _ = refine_propagation(q2, p2)
    assert [o.status for o in rp2.obligations] == ["discharged"]
    assert (
        "assert #0: endpoint computation saturated at the double range; "
        "the accounted err covers the excess" in rp2.notes
    )
    # ...and an unsaturated evaluation carries no such note
    rp1, _ = refine_propagation(_ka1_query(), propagate(_ka1_query()))
    assert not any("saturated" in n for n in rp1.notes)


def test_default_assembly_without_refinement_is_byte_identical():
    q = _ka4_query()
    p = propagate(q)
    assert make_verdict(q, p, **VERSIONS).render() == make_verdict(
        q, p, refinement=None, **VERSIONS
    ).render()
    assert make_verdict(q, p, **VERSIONS).stamp.solver.reason == (
        "no solver invoked: every obligation was judged by outward-rounded "
        "interval arithmetic alone"
    )


# --- the fidelity gauge (mandatory; L21) -------------------------------------
#
# Baseline: the evaluator on a battery covering the known answers above.
# Every mutation twists ONE claimed behavior; each must be CAUGHT by at
# least one gate; the expected residual is empty. Containment is checked
# against hand-written exact value functions at corner-heavy sample
# grids (duplicate-heavy shared-input cases included), compared as exact
# rationals so a one-ulp breach counts.


def _chain11_forms():
    """x folded onto itself ten times (11·x) over x ∈ [0, 0.1]: the
    centers and coefficients hit inexact float sums (odd multiples of
    0.05), so the accumulated err is tens of ulps — large enough that
    dropping it is visible PAST the final outward bracket (a sub-ulp
    shortfall would be re-covered by the concretization's own outward
    rounding; this case is chosen so the shortfall is not sub-ulp)."""
    x = affine.declared_form(0.0, 0.1, 0)
    s = x
    for _ in range(10):
        s = affine.add_forms(s, x)
    return s


# --- the audit-4 battery extension (F2): four mutation classes survived
# the first battery because no case exercised their paths — a threshold-
# tight non-dyadic chain (unbracketed concretization), an err>0 mul
# operand (element-blind cache keys; err-signed sub), and non-dyadic
# scale constants (scale-branch snap accounting). Each construction is
# adapted from the auditor's demonstrated-unsound variants.


def _chain01_query():
    """Ten inputs, each scaled by the non-dyadic 0.1 and chain-added,
    against the threshold 1.0: the true maximum is 10·fl(0.1) − 1 =
    +5.55e-17 — ABOVE the threshold by less than an ulp of the range, so
    an unbracketed float-summed concretization (hi = 0.0) excludes it."""
    eqns, xs, nid = [], [], 0
    for _ in range(10):
        x = var(nid)
        nid += 1
        eqns.append(any_eqn(x, 0.0, 1.0))
        xs.append(x)
    prev = None
    for x in xs:
        s = var(nid)
        nid += 1
        eqns.append(eqn("mul", [x, lit(0.1)], s))
        if prev is None:
            prev = s
        else:
            nxt = var(nid)
            nid += 1
            eqns.append(eqn("add", [prev, s], nxt))
            prev = nxt
    g = var(nid)
    nid += 1
    eqns.append(eqn("sub", [prev, lit(1.0)], g))
    pr, ob = var(nid, BOOL), var(nid + 1, BOOL)
    eqns.append(eqn("ge", [g, lit(0.0)], pr))
    eqns.append(eqn("stelling_assert", [pr], ob))
    return close(eqns, [ob])


def _a2_query():
    """An err>0 mul operand: (a + 0.3) has per-element snap err, so the
    product's residue symbols ride the IDENTITY keys (var id, element) —
    a cache key that drops the element index falsely cancels p[0] − p[1]
    to [0, 0] while the true range is [−1, 1]."""
    aa = avar(0, (2,))
    y = var(1)
    bb, yb, pp = avar(2, (2,)), avar(3, (2,)), avar(4, (2,))
    e0, e1 = avar(5, (1,)), avar(6, (1,))
    s0, s1, g = var(7), var(8), var(9)
    pr, ob = var(10, BOOL), var(11, BOOL)
    return close(
        [
            any_eqn(aa, 0.0, 1.0, shape=(2,)),
            any_eqn(y, -1.0, 1.0),
            eqn("add", [aa, lit(0.3)], bb),
            eqn("broadcast_in_dim", [y], yb, params=(
                ("shape", (2,)), ("broadcast_dimensions", ()))),
            eqn("mul", [bb, yb], pp),
            eqn("slice", [pp], e0, params=(
                ("start_indices", (0,)), ("limit_indices", (1,)), ("strides", None))),
            eqn("slice", [pp], e1, params=(
                ("start_indices", (1,)), ("limit_indices", (2,)), ("strides", None))),
            eqn("reduce_sum", [e0], s0, params=(("axes", (0,)),)),
            eqn("reduce_sum", [e1], s1, params=(("axes", (0,)),)),
            eqn("sub", [s0, s1], g),
            eqn("ge", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )


def _append_scale_chain(eqns, v0, nid):
    """Ten successive multiplications by the NON-dyadic literal 1.3: the
    scale branch's center/coefficient snaps round at every step, so the
    accounted err is several final-ulps — dropping the scale-branch snap
    accounting is visible past the outward bracket."""
    prev = v0
    for _ in range(10):
        nxt = var(nid)
        nid += 1
        eqns.append(eqn("mul", [prev, lit(1.3)], nxt))
        prev = nxt
    return prev, nid


def _scalechain_query():
    eqns = []
    x = var(0)
    eqns.append(any_eqn(x, 0.0, 0.7))
    t, nid = _append_scale_chain(eqns, x, 1)
    g = var(nid)
    nid += 1
    eqns.append(eqn("sub", [t, lit(6.0)], g))
    pr, ob = var(nid, BOOL), var(nid + 1, BOOL)
    eqns.append(eqn("ge", [g, lit(0.0)], pr))
    eqns.append(eqn("stelling_assert", [pr], ob))
    return close(eqns, [ob])


def _b4_query():
    """A sub whose OPERANDS carry err > 0 (two independent scale chains):
    treating err as signed on sub cancels the two equal err bounds to 0
    and loses several final-ulps of coverage — visible at the corners."""
    eqns = []
    x, y = var(0), var(100)
    eqns.append(any_eqn(x, 0.0, 0.7))
    eqns.append(any_eqn(y, 0.0, 0.7))
    t1, _ = _append_scale_chain(eqns, x, 1)
    t2, nid = _append_scale_chain(eqns, y, 101)
    g = var(nid + 100)
    eqns.append(eqn("sub", [t1, t2], g))
    pr, ob = var(nid + 101, BOOL), var(nid + 102, BOOL)
    eqns.append(eqn("ge", [g, lit(0.0)], pr))
    eqns.append(eqn("stelling_assert", [pr], ob))
    return close(eqns, [ob])


def _run_battery():
    """The measured outcomes the gates judge. Everything routed through
    the module namespace so the mutation patches apply."""
    out = {}
    for name, q, expect in (
        ("ka1", _ka1_query(), None),
        ("ka3", _ka3_query(), None),
        ("ka6", _ka6_query(), None),
    ):
        rp, _ = affine.refine_propagation(q, propagate(q))
        out[name] = tuple(o.status for o in rp.obligations)
    for name, q in (
        ("ka4", _ka4_query()),
        ("ka5", _ka5_query()),
        ("chain01", _chain01_query()),
        ("a2case", _a2_query()),
        ("scalechain", _scalechain_query()),
        ("b4case", _b4_query()),
    ):
        p = propagate(q)
        sl = slice_unknown_obligations(q, p, interval_env(q))[0]
        d = affine.decide_slice(sl)
        out[name] = (d.status,)
        out[name + "_ranges"] = d.ranges
    # the zero-edge: g = (x − x) − r over r ∈ [0, 1] has exact range
    # [−1, 0]: hi == 0 means the point g = 0 satisfies g ≥ 0, so neither
    # definite face may be claimed — the case must stay undecided
    x, r, d0, g = var(0), var(1), var(2), var(3)
    pr, ob = var(4, BOOL), var(5, BOOL)
    qz = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(r, 0.0, 1.0),
            eqn("sub", [x, x], d0),
            eqn("sub", [d0, r], g),
            eqn("ge", [g, lit(0.0)], pr),
            eqn("stelling_assert", [pr], ob),
        ],
        [ob],
    )
    rpz, _ = affine.refine_propagation(qz, propagate(qz))
    out["zero_edge"] = tuple(o.status for o in rpz.obligations)
    # element-sensitive routing: rolled[0] IS a[3] under the structural
    # conventions; g = sum(rolled[0:1]) − sum(a[3:4]) is exactly 0 only
    # when the routing is faithful (a sum-of-all-elements case would be
    # permutation-invariant and blind to misrouting)
    a = avar(0, (4,))
    last, first3 = avar(1, (1,)), avar(2, (3,))
    rolled, r0 = avar(3, (4,)), avar(4, (1,))
    sr, sa, gr = var(5), var(6), var(7)
    prr, obr = var(8, BOOL), var(9, BOOL)
    qr = close(
        [
            any_eqn(a, 0.0, 1.0, shape=(4,)),
            eqn("slice", [a], last, params=(
                ("start_indices", (3,)), ("limit_indices", (4,)), ("strides", None))),
            eqn("slice", [a], first3, params=(
                ("start_indices", (0,)), ("limit_indices", (3,)), ("strides", None))),
            eqn("concatenate", [last, first3], rolled, params=(("dimension", 0),)),
            eqn("slice", [rolled], r0, params=(
                ("start_indices", (0,)), ("limit_indices", (1,)), ("strides", None))),
            eqn("reduce_sum", [r0], sr, params=(("axes", (0,)),)),
            eqn("reduce_sum", [last], sa, params=(("axes", (0,)),)),
            eqn("sub", [sr, sa], gr),
            eqn("ge", [gr, lit(0.0)], prr),
            eqn("stelling_assert", [prr], obr),
        ],
        [obr],
    )
    rpq, _ = affine.refine_propagation(qr, propagate(qr))
    out["route0"] = tuple(o.status for o in rpq.obligations)
    rpr, _ = affine.refine_propagation(_refute_query(), propagate(_refute_query()))
    out["refute"] = tuple(o.status for o in rpr.obligations)
    # containment subjects: the additive chain and a two-var product
    s = _chain11_forms()
    out["chain11_range"] = affine.concretization(s)
    t = SymbolTable(2)
    fx = affine.declared_form(0.1, 0.3, 0)
    fy = affine.declared_form(0.1, 0.3, 1)
    ax, ay = avar(0, ()), avar(1, ())
    prod = affine.mul_forms(
        t,
        fx,
        affine._operand_key(fx, ax, 0),
        fy,
        affine._operand_key(fy, ay, 0),
    )
    out["mul_range"] = affine.concretization(prod)
    # pipeline equivalence on the declining case
    q7 = _ka7_query()
    p7 = propagate(q7)
    rp7, rep7 = affine.refine_propagation(q7, p7)
    out["div_identical"] = rp7.obligations == p7.obligations
    out["div_reason"] = rep7.declined[0][1] if rep7.declined else ""
    return out


def _grid(lo, hi, n=7):
    lof, hif = Fraction(lo), Fraction(hi)
    return [lof + (hif - lof) * k / (n - 1) for k in range(n)]


def _contains(rng, v):
    lo, hi = rng
    if math.isfinite(lo) and v < Fraction(lo):
        return False
    if math.isfinite(hi) and v > Fraction(hi):
        return False
    return True


def _gate_containment(out):
    lo, hi = out["chain11_range"]
    for xv in _grid(0.0, 0.1):
        if not Fraction(lo) <= 11 * xv <= Fraction(hi):
            return False
    lo, hi = out["mul_range"]
    for xv in _grid(0.1, 0.3):
        for yv in _grid(0.1, 0.3):
            if not Fraction(lo) <= xv * yv <= Fraction(hi):
                return False
    ((lo, hi),) = out["ka4_ranges"]
    for xv in _grid(-1.0, 1.0):
        for yv in _grid(-1.0, 1.0):
            if not Fraction(lo) <= xv * yv <= Fraction(hi):
                return False
    ((lo, hi),) = out["ka5_ranges"]
    for av in _grid(1.0, 2.0, 5):
        for cv in _grid(1.0, 2.0, 5):
            for bv in _grid(-0.5, 0.5, 5):
                gval = (av * cv - bv * bv) * Fraction(81, 8) - (av + cv) ** 2
                if not Fraction(lo) <= gval <= Fraction(hi):
                    return False
    # audit-4 extension: the four exact-value functions of the new
    # cases, sampled at the corners the surviving mutants shifted past
    (rng,) = (out["chain01_ranges"],)
    ones = [Fraction(1)] * 10
    pts = [[Fraction(0)] * 10, ones] + [
        [Fraction(1) if j == i else Fraction(0) for j in range(10)]
        for i in range(10)
    ]
    for p in pts:
        if not _contains(rng[0], sum(Fraction(0.1) * x for x in p) - 1):
            return False
    (rng,) = (out["a2case_ranges"],)
    for a0 in (Fraction(0), Fraction(1)):
        for a1 in (Fraction(0), Fraction(1)):
            for yv in (Fraction(-1), Fraction(1)):
                gval = (a0 + Fraction(0.3)) * yv - (a1 + Fraction(0.3)) * yv
                if not _contains(rng[0], gval):
                    return False
    r13 = Fraction(1.3) ** 10
    (rng,) = (out["scalechain_ranges"],)
    for xv in _grid(0.0, 0.7, 8):
        if not _contains(rng[0], xv * r13 - Fraction(6.0)):
            return False
    (rng,) = (out["b4case_ranges"],)
    for xv in (Fraction(0), Fraction(0.7)):
        for yv in (Fraction(0), Fraction(0.7)):
            if not _contains(rng[0], (xv - yv) * r13):
                return False
    return True


def _gate_exact_cancellation(out):
    return (
        out["ka1"] == ("discharged", "discharged")
        and out["ka3"] == ("discharged", "discharged")
        and out["ka6"] == ("discharged", "discharged")
        and out["route0"] == ("discharged",)
        and out["refute"] == ("violated-over-set",)
    )


def _gate_honest_nonclosure(out):
    # FAILS if a mutant claims closure of the true straddle (ka4), the
    # quadratic-past-affine shape (ka5), the zero-edge, or any of the
    # audit-4 threshold/err-carrying cases (all genuinely undecidable
    # from their boxes)
    return (
        out["ka4"] == ("unknown",)
        and out["ka5"] == ("unknown",)
        and out["zero_edge"] == ("unknown",)
        and out["chain01"] == ("unknown",)
        and out["a2case"] == ("unknown",)
        and out["scalechain"] == ("unknown",)
        and out["b4case"] == ("unknown",)
    )


def _gate_pipeline_equivalence(out):
    return out["div_identical"] and "'div'" in out["div_reason"]


def _mutants():
    orig_snap = affine._snap
    orig_routes = affine._route_structural
    orig_add = affine.add_forms
    orig_mul = affine.mul_forms

    def snap_unaccounted(fr):
        z, _, sat = orig_snap(fr)
        return z, Fraction(0), sat

    def uncanonical_product(self, key_a, key_b):
        key = (key_a, key_b)  # ORDERED pair: commuted products now get
        # distinct residue symbols — the uncanonicalized cache
        got = self._products.get(key)
        if got is None:
            got = self._next
            self._next += 1
            self._products[key] = got
        return got

    def residue_without_rad_rad(f, g, deltas):
        return (
            abs(Fraction(f.center)) * Fraction(g.err)
            + abs(Fraction(g.center)) * Fraction(f.err)
            + deltas  # rad(f)·rad(g) forgotten
        )

    def add_without_err(f, g):
        coefs = {s: Fraction(k) for s, k in f.coeffs}
        for s, k in g.coeffs:
            coefs[s] = coefs.get(s, Fraction(0)) + Fraction(k)
        return affine._snap_form(
            Fraction(f.center) + Fraction(g.center), coefs, Fraction(0)
        )

    def routes_reversed(eqn_):
        return list(reversed(orig_routes(eqn_)))

    # --- the audit-4 mutation classes (each demonstrated unsound by
    # exact-rational measurement before these battery cases existed) ---

    def concretization_inward_float_sum(f):
        # bare float endpoint arithmetic, no outward bracket: Σ|coef|
        # round-to-nearest can round DOWN and the endpoints can round in
        total = 0.0
        for _, k in f.coeffs:
            total += abs(k)
        total += f.err
        return f.center - total, f.center + total

    def key_ignores_element(f, atom, element):
        if f.err == 0.0:
            return (0, (f.center, f.coeffs))
        if isinstance(atom, ir.Var):
            return (1, atom.id)  # ELEMENT INDEX DROPPED: cross-element
            # residues falsely share one symbol and cancel
        raise affine._Decline("no identity")

    def sub_err_signed(f, g):
        coefs = {s: Fraction(k) for s, k in f.coeffs}
        for s, k in g.coeffs:
            coefs[s] = coefs.get(s, Fraction(0)) - Fraction(k)
        err = Fraction(f.err) - Fraction(g.err)  # SIGNED: bounds cancel
        if err < 0:
            err = Fraction(0)
        return affine._snap_form(
            Fraction(f.center) - Fraction(g.center),
            coefs,
            err,
            saturated=f.saturated or g.saturated,
        )

    def scale_drops_deltas(table, f, key_f, g, key_g):
        if affine._is_point(g) or affine._is_point(f):
            if affine._is_point(g):
                scale, other = Fraction(g.center), f
            else:
                scale, other = Fraction(f.center), g
            center = float(scale * Fraction(other.center))  # bare float
            items = []
            for s, k in other.coeffs:
                kk = float(scale * Fraction(k))  # snap delta dropped
                if kk != 0.0 and math.isfinite(kk):
                    items.append((s, kk))
            err = float(abs(scale) * Fraction(other.err))
            if not (math.isfinite(center) and math.isfinite(err)):
                raise affine._Decline("overflow in mutant")
            return affine.AffineForm(
                center=center, coeffs=tuple(sorted(items)), err=err
            )
        return orig_mul(table, f, key_f, g, key_g)

    return {
        "inward-rounding-on-a-coefficient-path": mock.patch.object(
            affine, "_snap", snap_unaccounted
        ),
        "uncanonicalized-product-cache": mock.patch.object(
            SymbolTable, "product_symbol", uncanonical_product
        ),
        "mul-residue-dropped": mock.patch.object(
            affine, "_mul_residual", residue_without_rad_rad
        ),
        "err-accumulation-dropped-on-add": mock.patch.object(
            affine, "add_forms", add_without_err
        ),
        "discharge-reads-lo-strictly": mock.patch.object(
            affine, "_element_true", lambda lo: lo > 0.0
        ),
        "violation-claims-hi-at-zero": mock.patch.object(
            affine, "_element_false", lambda hi: hi <= 0.0
        ),
        "structural-routing-reversed": mock.patch.object(
            affine, "_route_structural", routes_reversed
        ),
        "concretization-inward-float-sum": mock.patch.object(
            affine, "concretization", concretization_inward_float_sum
        ),
        "cache-key-ignores-element": mock.patch.object(
            affine, "_operand_key", key_ignores_element
        ),
        "err-signed-on-sub": mock.patch.object(
            affine, "sub_forms", sub_err_signed
        ),
        "scale-branch-drops-snap-deltas": mock.patch.object(
            affine, "mul_forms", scale_drops_deltas
        ),
    }


def test_fidelity_gauge_every_mutation_caught(capsys):
    gates = {
        "containment-vs-dense-sampling": _gate_containment,
        "exact-cancellation": _gate_exact_cancellation,
        "honest-nonclosure": _gate_honest_nonclosure,
        "pipeline-equivalence": _gate_pipeline_equivalence,
    }
    baseline = _run_battery()
    mutations = {}
    for name, ctx in _mutants().items():
        with ctx:
            mutations[name] = _run_battery()
    report = gauge(baseline, gates, mutations, residual={})
    assert report.residual == ()  # every mutation caught; nothing survives
    for _, catchers in report.caught_by:
        assert catchers
    with capsys.disabled():
        print()
        print(report.render())
