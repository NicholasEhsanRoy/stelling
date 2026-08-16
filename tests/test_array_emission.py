# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Bounded static-shape array emission: naming, sharing, routing, budget.

Hand-built IR (no jax) covering the array-emission build's contracts:

* element naming — ``x{k}_{i}`` inputs, ``t{id}_{i}`` terms, flat C-order,
  with the scalar forms byte-identical to the pre-array emission (two
  script hashes pinned);
* **sharing** — a broadcast scalar is ONE SMT constant everywhere it
  appears (fresh per-element variables would invent decorrelation: the
  aliasing bug at array scale, the single worst defect this build can
  have), and structural ops alias source terms rather than minting any;
* structural-op index routing, pinned against values measured on jax
  0.11.0 (the jax-backed differential battery lives in
  ``test_array_acceptance.py``);
* the exact n-ary ``reduce_sum`` (empty sum emits jax's measured 0.0);
* the per-element div guard, naming the straddling element;
* the single per-obligation element budget — the ONE gate that replaced
  the seven scalar-only sites — declining with the count and the budget
  quoted;
* transparent call descent for the computation (obligations stay
  top-level-only), including malformed-wrapper and id-collision declines;
* the universal elementwise assert (negated conjunction), the array
  replay, the witness validator per element, and the fake-transport
  array-witness paths (garbage, missing, out-of-box, empty models).
"""

from __future__ import annotations

import array as _arraymod
import hashlib
import struct
from fractions import Fraction

import pytest

from stelling import ir
from stelling.obligation import (
    DIV_GUARD_REASON,
    ELEMENT_BUDGET,
    DeclinedObligation,
    ObligationSlice,
    ReplayError,
    _group_reduce_sum,
    _pair_elementwise,
    _pair_select_n,
    _route_structural,
    evaluate_predicate,
    slice_obligation,
    slice_unknown_obligations,
    violating_elements,
    witness_is_valid,
)
from stelling.propagate import interval_env, propagate
from stelling.smt import emit
from test_obligation_slice import BOOL, F64, any_eqn, close, eqn, lit, var

F64_ARR = {
    n: ir.Aval(kind="ShapedArray", shape=(n,), dtype="float64")
    for n in (0, 1, 2, 3, 4, 6, 10)
}
BOOL_ARR = {
    n: ir.Aval(kind="ShapedArray", shape=(n,), dtype="bool")
    for n in (0, 1, 2, 3, 4, 6, 10)
}


def aval(shape, dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=tuple(shape), dtype=dtype)


def arr_lit(values, shape=None, dtype="<f8"):
    shape = (len(values),) if shape is None else tuple(shape)
    data = struct.pack(f"<{len(values)}d", *[float(v) for v in values])
    return ir.Literal(
        val=ir.Array(dtype=dtype, shape=shape, data=data),
        aval=aval(shape),
    )


def sole(q):
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1, items
    return items[0]


def sole_slice(q) -> ObligationSlice:
    item = sole(q)
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    return item


def vec_gt_query(n=3, lo=-1.0, hi=1.0, bound=0.0):
    """x: (n,) in [lo, hi]; assert x > bound elementwise (array assert)."""
    x = var(0, F64_ARR[n])
    pred = var(1, BOOL_ARR[n])
    out = var(2, BOOL_ARR[n])
    return close(
        [
            any_eqn(x, lo, hi, shape=(n,)),
            eqn("gt", [x, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


# --- element naming -----------------------------------------------------------


def test_array_input_declares_one_constant_per_element_flat_c_order():
    sl = sole_slice(vec_gt_query(3))
    assert [i.name for i in sl.inputs] == ["x0_0", "x0_1", "x0_2"]
    assert [i.element for i in sl.inputs] == [0, 1, 2]
    assert all(i.shape == (3,) for i in sl.inputs)
    # bounds broadcast to every element
    assert all((i.lo, i.hi) == (-1.0, 1.0) for i in sl.inputs)
    text = emit(sl, "z3", 100).text
    for name in ("x0_0", "x0_1", "x0_2"):
        assert f"(declare-const {name} Real)" in text
        assert f"(assert (<= (- 1.0) {name}))" in text
        assert f"(assert (<= {name} 1.0))" in text


def test_intermediate_terms_are_per_element_t_id_i():
    x = var(0, F64_ARR[2])
    s = var(1, F64_ARR[2])
    pred = var(2, BOOL_ARR[2])
    out = var(3, BOOL_ARR[2])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("add", [x, x], s),
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(define-fun t1_0 () Real (+ x0_0 x0_0))" in text
    assert "(define-fun t1_1 () Real (+ x0_1 x0_1))" in text
    assert "(define-fun t2_0 () Bool (<= t1_0 (/ 3 2)))" in text
    assert "(define-fun t2_1 () Bool (<= t1_1 (/ 3 2)))" in text


def test_mixed_scalar_and_array_declarations_keep_both_schemes():
    a, x = var(0), var(1, F64_ARR[2])
    s = var(2, F64_ARR[2])
    pred = var(3, BOOL_ARR[2])
    out = var(4, BOOL_ARR[2])
    q = close(
        [
            any_eqn(a, 0.0, 1.0),  # scalar: stays x0, byte-identical scheme
            any_eqn(x, 0.0, 1.0, shape=(2,)),  # array: x1_0, x1_1
            eqn("add", [x, a], s),
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert [i.name for i in sl.inputs] == ["x0", "x1_0", "x1_1"]


# --- sharing: THE fidelity rule ----------------------------------------------


def test_broadcast_scalar_is_one_constant_shared_by_every_element():
    """The single most important rule of this build: a scalar operand
    broadcast against an array is the SAME SMT constant in every element's
    body — one declare-const, referenced n times — never one fresh
    variable per element (which would decorrelate the elements and prove
    the wrong formula)."""
    a, x = var(0), var(1, F64_ARR[3])
    s = var(2, F64_ARR[3])
    pred = var(3, BOOL_ARR[3])
    out = var(4, BOOL_ARR[3])
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(x, 0.0, 1.0, shape=(3,)),
            eqn("mul", [a, x], s),  # scalar a broadcast across x
            eqn("le", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    # exactly four constants: a and the three elements of x — nothing fresh
    assert text.count("(declare-const") == 4
    assert text.count("(declare-const x0 Real)") == 1
    # the ONE constant a appears in every element's product
    assert "(define-fun t2_0 () Real (* x0 x1_0))" in text
    assert "(define-fun t2_1 () Real (* x0 x1_1))" in text
    assert "(define-fun t2_2 () Real (* x0 x1_2))" in text
    assert text.count("(* x0 x1_") == 3  # the SAME name in every body


def test_scalar_literal_broadcast_uses_the_same_literal_text_everywhere():
    x = var(0, F64_ARR[2])
    s = var(1, F64_ARR[2])
    pred = var(2, BOOL_ARR[2])
    out = var(3, BOOL_ARR[2])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("mul", [lit(0.5), x], s),
            eqn("le", [s, lit(0.25)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(define-fun t1_0 () Real (* (/ 1 2) x0_0))" in text
    assert "(define-fun t1_1 () Real (* (/ 1 2) x0_1))" in text


def test_select_n_scalar_selector_is_shared_across_elements():
    """Measured jax form: jnp.where(scalar_pred, arr, arr) traces to
    select_n with a SCALAR bool selector over array cases (inside its
    transparent jit). The one selector term must appear in every
    element's ite — sharing again."""
    c = var(0)
    x = var(1, F64_ARR[2])
    y = var(2, F64_ARR[2])
    w = var(3, BOOL)
    sel = var(4, F64_ARR[2])
    pred = var(5, BOOL_ARR[2])
    out = var(6, BOOL_ARR[2])
    q = close(
        [
            any_eqn(c, 0.0, 1.0),
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            any_eqn(y, 2.0, 3.0, shape=(2,)),
            eqn("gt", [c, lit(0.5)], w),
            eqn("select_n", [w, x, y], sel),
            eqn("le", [sel, lit(2.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(define-fun t4_0 () Real (ite t3 x2_0 x1_0))" in text
    assert "(define-fun t4_1 () Real (ite t3 x2_1 x1_1))" in text
    assert text.count("(define-fun t3 () Bool") == 1  # ONE selector term


# --- structural ops are index bookkeeping, not new terms ----------------------


def struct_eqn(prim, in_avals, out_aval, params):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(var(10 + i, a) for i, a in enumerate(in_avals)),
        outvars=(var(30, out_aval),),
        params=tuple(params),
    )


def test_routing_pinned_against_measured_jax_transpose():
    # measured (jax 0.11.0): lax.transpose(arange(6).reshape(2,3), (1,0))
    # ravels to [0, 3, 1, 4, 2, 5]
    e = struct_eqn(
        "transpose", [aval((2, 3))], aval((3, 2)), [("permutation", (1, 0))]
    )
    assert _route_structural(e) == [
        (0, 0), (0, 3), (0, 1), (0, 4), (0, 2), (0, 5)
    ]


def test_routing_pinned_against_measured_jax_broadcast_in_dim():
    # measured: bcast (2,)->(3,2) bd=(1,) ravels [1,2,1,2,1,2] on [1,2]
    e = struct_eqn(
        "broadcast_in_dim",
        [aval((2,))],
        aval((3, 2)),
        [("shape", (3, 2)), ("broadcast_dimensions", (1,))],
    )
    assert _route_structural(e) == [(0, 0), (0, 1)] * 3
    # measured: bcast (2,1)->(2,3) bd=(0,1) ravels [1,1,1,2,2,2]
    e2 = struct_eqn(
        "broadcast_in_dim",
        [aval((2, 1))],
        aval((2, 3)),
        [("shape", (2, 3)), ("broadcast_dimensions", (0, 1))],
    )
    assert _route_structural(e2) == [
        (0, 0), (0, 0), (0, 0), (0, 1), (0, 1), (0, 1)
    ]


def test_routing_pinned_against_measured_jax_strided_slice():
    # measured: lax.slice(arange(10), (1,), (8,), (3,)) == [1, 4, 7]
    e = struct_eqn(
        "slice",
        [aval((10,))],
        aval((3,)),
        [("start_indices", (1,)), ("limit_indices", (8,)), ("strides", (3,))],
    )
    assert _route_structural(e) == [(0, 1), (0, 4), (0, 7)]


def test_routing_pinned_against_measured_jax_concatenate_dim1():
    # measured: concat[(2,2) arange(4), (2,3) arange(10,16)] dim=1 ravels
    # [0, 1, 10, 11, 12, 2, 3, 13, 14, 15]
    e = ir.JaxprEqn(
        primitive="concatenate",
        invars=(var(10, aval((2, 2))), var(11, aval((2, 3)))),
        outvars=(var(30, aval((2, 5))),),
        params=(("dimension", 1),),
    )
    assert _route_structural(e) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (1, 2),
        (0, 2), (0, 3), (1, 3), (1, 4), (1, 5),
    ]


def test_routing_reshape_is_c_order_identity_and_dimensions_declines():
    e = struct_eqn(
        "reshape", [aval((2, 3))], aval((3, 2)),
        [("new_sizes", (3, 2)), ("dimensions", None)],
    )
    assert _route_structural(e) == [(0, i) for i in range(6)]
    from stelling.obligation import _Decline

    bad = struct_eqn(
        "reshape", [aval((2, 3))], aval((3, 2)),
        [("new_sizes", (3, 2)), ("dimensions", (1, 0))],
    )
    with pytest.raises(_Decline, match="permutes before reshaping"):
        _route_structural(bad)


def test_routing_squeeze_is_identity_on_flat_elements():
    e = struct_eqn(
        "squeeze", [aval((1, 3))], aval((3,)), [("dimensions", (0,))]
    )
    assert _route_structural(e) == [(0, 0), (0, 1), (0, 2)]


def test_reduce_sum_grouping_matches_measured_jax_partial_axes():
    # measured: jnp.sum(arange(24).reshape(2,3,4), axis=(0,2)) == [60,92,124]
    e = struct_eqn(
        "reduce_sum", [aval((2, 3, 4))], aval((3,)), [("axes", (0, 2))]
    )
    groups = _group_reduce_sum(e)
    assert [len(g) for g in groups] == [8, 8, 8]
    assert [sum(g) for g in groups] == [60, 92, 124]  # arange value == index


def test_concatenate_output_element_IS_its_source_element_term():
    x, y = var(0, F64_ARR[2]), var(1, F64_ARR[1])
    c = var(2, F64_ARR[3])
    pred = var(3, BOOL_ARR[3])
    out = var(4, BOOL_ARR[3])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            any_eqn(y, 0.0, 1.0, shape=(1,)),
            eqn("concatenate", [x, y], c, [("dimension", 0)]),
            eqn("gt", [c, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    # no term for the concatenate itself: its elements ARE x0_0, x0_1, x1_0
    assert "(define-fun t1_0 () Bool (> x0_0 (/ 1 2)))" not in text  # naming sanity
    assert "(define-fun t3_0 () Bool (> x0_0 (/ 1 2)))" in text
    assert "(define-fun t3_1 () Bool (> x0_1 (/ 1 2)))" in text
    assert "(define-fun t3_2 () Bool (> x1_0 (/ 1 2)))" in text
    assert text.count("(define-fun") == 3  # only the comparisons


def test_malformed_concatenate_declines_at_the_validator_never_misroutes():
    """Posed at the validator directly: a legal jax trace cannot produce
    these params, but ``ClosedJaxpr.from_dict`` can, and the emission must
    decline them quoted rather than mis-route elements. (NOTE, reported in
    the build report: the same malformed forms CRASH the untouchable
    interval-propagation layer with an IndexError before slicing is ever
    reached — a pre-existing degrade-don't-crash escape in a sibling this
    build may not modify — so these constructions cannot be posed through
    ``propagate()`` here.)"""
    from stelling.obligation import _Decline

    # rank-1 pieces cannot concatenate along dimension 1
    e = ir.JaxprEqn(
        primitive="concatenate",
        invars=(var(10, aval((2,))), var(11, aval((3,)))),
        outvars=(var(30, aval((2, 5))),),
        params=(("dimension", 1),),
    )
    with pytest.raises(_Decline, match="'concatenate'"):
        _route_structural(e)
    # off-axis extents disagreeing decline quoted, never a wrong read
    e2 = ir.JaxprEqn(
        primitive="concatenate",
        invars=(var(10, aval((2, 2))), var(11, aval((3, 3)))),
        outvars=(var(30, aval((5, 2))),),
        params=(("dimension", 0),),
    )
    with pytest.raises(_Decline, match="disagree off the concatenation"):
        _route_structural(e2)


# --- the exact n-ary reduce_sum ----------------------------------------------


def test_reduce_sum_emits_nary_plus_of_element_terms():
    x = var(0, F64_ARR[3])
    s = var(1)
    pred = var(2, BOOL)
    out = var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(3,)),
            eqn("reduce_sum", [x], s, [("axes", (0,))]),
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(define-fun t1 () Real (+ x0_0 x0_1 x0_2))" in text


def test_zero_size_reduce_sum_emits_jaxs_measured_zero_identity():
    # measured: jnp.sum of a size-0 array is 0.0 (and interval.reduce_sum
    # pins the same identity); the emission's empty sum is the literal 0.0
    a = var(0)
    x = var(1, F64_ARR[0])
    s = var(2)
    t = var(3)
    pred = var(4, BOOL)
    out = var(5, BOOL)
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(x, 0.0, 1.0, shape=(0,)),
            eqn("reduce_sum", [x], s, [("axes", (0,))]),
            eqn("add", [a, s], t),
            eqn("le", [t, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(define-fun t2 () Real 0.0)" in text
    assert "(+ x0 t2)" in text


# --- the per-element div guard -----------------------------------------------


def test_div_guard_declines_naming_the_straddling_element():
    x = var(0, F64_ARR[3])
    d = var(1, F64_ARR[3])
    y = var(2, F64_ARR[3])
    pred = var(3, BOOL_ARR[3])
    out = var(4, BOOL_ARR[3])
    # divisor = x + [1, -0.5, 3]: element intervals [1,2], [-0.5,0.5], [3,4] —
    # element 1 truly straddles zero, elements 0 and 2 are safely nonzero
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(3,)),
            eqn("add", [x, arr_lit([1.0, -0.5, 3.0])], d),
            eqn("div", [x, d], y),
            eqn("lt", [y, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert DIV_GUARD_REASON in item.reason
    assert "element 1" in item.reason  # the straddler is NAMED


def test_div_guard_passes_when_every_element_excludes_zero():
    x = var(0, F64_ARR[2])
    d = var(1, F64_ARR[2])
    y = var(2, F64_ARR[2])
    pred = var(3, BOOL_ARR[2])
    out = var(4, BOOL_ARR[2])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("add", [x, arr_lit([1.0, 2.0])], d),  # [1,2], [2,3]: nonzero
            eqn("div", [x, d], y),
            eqn("gt", [y, lit(0.25)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    text = emit(sl, "z3", 100).text
    assert "(/ x0_0 t1_0)" in text and "(/ x0_1 t1_1)" in text


def test_scalar_div_guard_decline_text_is_unchanged():
    # the pre-array wording, no element suffix, for the scalar form
    x, y, d = var(0), var(1), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, -1.0, 1.0),
            eqn("div", [x, y], d),
            eqn("le", [d, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert item.reason == f"'div': {DIV_GUARD_REASON}"


# --- the single budget gate ---------------------------------------------------


def test_over_budget_obligation_declines_with_count_and_budget_quoted():
    n = ELEMENT_BUDGET  # inputs n + comparison n = 2n > budget
    x = var(0, aval((n,)))
    pred = var(1, aval((n,), "bool"))
    out = var(2, aval((n,), "bool"))
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(n,)),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert str(2 * n) in item.reason  # the count
    assert str(ELEMENT_BUDGET) in item.reason  # the budget
    assert "budget" in item.reason


def test_at_budget_obligation_emits():
    n = ELEMENT_BUDGET // 2  # inputs n + comparison n = budget exactly
    x = var(0, aval((n,)))
    pred = var(1, aval((n,), "bool"))
    out = var(2, aval((n,), "bool"))
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(n,)),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert sl.element_terms == ELEMENT_BUDGET
    assert emit(sl, "z3", 100).text.count("(declare-const") == n


def test_element_terms_counts_inputs_plus_term_producing_outputs():
    sl = sole_slice(vec_gt_query(3))
    assert sl.element_terms == 6  # 3 input elements + 3 comparison elements
    # structural ops add nothing: broadcast/slice/concat routed slices count 0
    x = var(0, F64_ARR[2])
    b = var(1, aval((2, 2)))
    pred = var(2, aval((2, 2), "bool"))
    out = var(3, aval((2, 2), "bool"))
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(2,)),
            eqn(
                "broadcast_in_dim",
                [x],
                b,
                [("shape", (2, 2)), ("broadcast_dimensions", (1,))],
            ),
            eqn("gt", [b, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole_slice(q).element_terms == 2 + 4  # inputs + comparison only


# --- transparent call descent (computation only) ------------------------------


def roll_query(n=3, lo=-1.0, hi=1.0, bound=0.0, wrapper="jit"):
    """x (n,) declared; a hand-built transparent wrapper computes
    roll(x, -1) exactly as measured on jax 0.11.0 (_roll_static: two
    slices + concatenate); assert x + roll(x) > bound elementwise."""
    x = var(0, F64_ARR[n])
    rolled = var(1, F64_ARR[n])
    ix = var(10, F64_ARR[n])
    p1 = var(11, aval((n - 1,)))
    p2 = var(12, F64_ARR[1])
    ic = var(13, F64_ARR[n])
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(ix,),
            outvars=(ic,),
            eqns=(
                eqn(
                    "slice",
                    [ix],
                    p1,
                    [
                        ("start_indices", (1,)),
                        ("limit_indices", (n,)),
                        ("strides", (1,)),
                    ],
                ),
                eqn(
                    "slice",
                    [ix],
                    p2,
                    [
                        ("start_indices", (0,)),
                        ("limit_indices", (1,)),
                        ("strides", (1,)),
                    ],
                ),
                eqn("concatenate", [p1, p2], ic, [("dimension", 0)]),
            ),
        )
    )
    s = var(2, F64_ARR[n])
    pred = var(3, BOOL_ARR[n])
    out = var(4, BOOL_ARR[n])
    return close(
        [
            any_eqn(x, lo, hi, shape=(n,)),
            ir.JaxprEqn(
                primitive=wrapper,
                invars=(x,),
                outvars=(rolled,),
                params=(("jaxpr", inner),),
            ),
            eqn("add", [x, rolled], s),
            eqn("gt", [s, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_transparent_descent_inlines_the_roll_and_shares_the_inputs():
    sl = sole_slice(roll_query(3))
    text = emit(sl, "z3", 100).text
    # roll is pure routing: element i of rolled IS x0_{(i+1) % 3} — the
    # adds pair each element with its rolled neighbour, no fresh terms
    assert "(define-fun t2_0 () Real (+ x0_0 x0_1))" in text
    assert "(define-fun t2_1 () Real (+ x0_1 x0_2))" in text
    assert "(define-fun t2_2 () Real (+ x0_2 x0_0))" in text
    assert text.count("(declare-const") == 3


def test_transparent_descent_replays_through_the_wrapper():
    sl = sole_slice(roll_query(3))
    vals = {
        "x0_0": Fraction(1),
        "x0_1": Fraction(-1),
        "x0_2": Fraction(1),
    }
    # x + roll = [0, 0, 2]: elements 0 and 1 are not > 0 -> violated
    assert evaluate_predicate(sl, vals) is False
    assert violating_elements(sl, vals) == (0, 1)
    good = {
        "x0_0": Fraction(1),
        "x0_1": Fraction(1),
        "x0_2": Fraction(1),
    }
    assert evaluate_predicate(sl, good) is True
    assert violating_elements(sl, good) == ()


def test_malformed_wrapper_declines_with_the_form_quoted():
    x = var(0, F64_ARR[2])
    rolled = var(1, F64_ARR[2])
    s = var(2, F64_ARR[2])
    pred = var(3, BOOL_ARR[2])
    out = var(4, BOOL_ARR[2])
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(2,)),
            # a jit with NO sub-jaxpr param resists sound inlining
            ir.JaxprEqn(
                primitive="jit",
                invars=(x,),
                outvars=(rolled,),
                params=(("name", "mystery"),),
            ),
            eqn("add", [x, rolled], s),
            eqn("gt", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert "'jit'" in item.reason
    assert "resists sound descent" in item.reason


def test_variable_id_reuse_across_scopes_never_misshares():
    """An inner scope reusing an OUTER var id must not alias two different
    values — the array-scale aliasing bug.

    This used to assert the DECLINE, because the descent detected the
    collision and poisoned the query. That was sound but a ceiling, and the
    descent now RENUMBERS instead: every inner binding gets a fresh id, so
    the collision cannot occur and the query proceeds.

    The test therefore asserts the PROPERTY rather than the old mechanism —
    and asserts it observably. The jit call is passed ``w in [10, 12]`` while
    its inner invar deliberately carries the outer declaration's id 0, which
    belongs to ``x in [-1, 1]``. The two ranges are DISJOINT, so a
    mis-sharing is not a subtle numeric difference but a verdict flip.

    The bound is 11.0, which STRADDLES w's range so intervals cannot decide
    it and the obligation actually reaches the slice — the stage under test.
    A bound of 5.0 would be discharged outright and this test would exercise
    nothing (CONTRIBUTING.md: a probe reading a final verdict must assert
    something non-trivial). Under a mis-share to x in [-1, 1] the predicate
    would instead be definitely FALSE, decided by intervals, and no unknown
    obligation would reach the slice at all — so the two readings remain
    distinguishable in both directions.
    """
    n = 2
    x = var(0, F64_ARR[n])
    w = var(5, F64_ARR[n])
    rolled = var(1, F64_ARR[n])
    # inner invar deliberately reuses the OUTER declaration's id 0, while the
    # call is passed `w` — so the two readings are distinguishable by range
    ix = var(0, F64_ARR[n])
    ic = var(13, F64_ARR[n])
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(ix,),
            outvars=(ic,),
            eqns=(
                eqn(
                    "slice",
                    [ix],
                    ic,
                    [
                        ("start_indices", (0,)),
                        ("limit_indices", (n,)),
                        ("strides", (1,)),
                    ],
                ),
            ),
        )
    )
    pred = var(3, BOOL_ARR[n])
    out = var(4, BOOL_ARR[n])
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(n,)),
            any_eqn(w, 10.0, 12.0, shape=(n,)),
            ir.JaxprEqn(
                primitive="jit",
                invars=(w,),
                outvars=(rolled,),
                params=(("jaxpr", inner),),
            ),
            eqn("gt", [rolled, lit(11.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    # It must DECIDE, not decline: the collision is renumbered away.
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    # And it must have read `w`, not the outer id 0. The slice's inputs name
    # which declaration the obligation actually depends on.
    names = {i.var_id for i in item.inputs}
    assert names == {5}, (
        f"slice depends on declarations {sorted(names)}; expected only var 5 "
        f"(w). Depending on var 0 means the inner invar mis-shared the outer "
        f"declaration's id."
    )


def test_obligations_stay_top_level_only_nested_assert_still_declines():
    # unchanged mapping rule: an assert INSIDE a wrapper declines wholly
    x = var(0)
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
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], DeclinedObligation)
    assert "top-level" in items[0].reason


# --- the universal elementwise assert and its negation ------------------------


def test_array_assert_negates_the_conjunction_of_all_elements():
    text = emit(sole_slice(vec_gt_query(3)), "z3", 100).text
    assert "(assert (not (and t1_0 t1_1 t1_2)))" in text


def test_scalar_assert_negation_is_byte_identical_no_conjunction():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(assert (not t1))" in text
    assert "(and" not in text


# --- zero-size shapes: measured and matched -----------------------------------


def test_zero_size_assert_is_discharged_by_interval_matching_jnp_all():
    # measured on jax 0.11.0: jnp.all of a size-0 predicate is True (the
    # empty universal claim); interval propagation discharges it the same
    # way, so it never reaches escalation
    x = var(0, F64_ARR[0])
    pred = var(1, BOOL_ARR[0])
    out = var(2, BOOL_ARR[0])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(0,)),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["discharged"]
    assert "0 element(s)" in p.obligations[0].detail
    # nothing unknown -> nothing sliced
    assert slice_unknown_obligations(q, p, interval_env(q)) == ()
    # a DIRECT ask (bypassing propagation's discharge) declines, quoted —
    # emission never mints the vacuous proof obligation itself
    item = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert "zero elements" in item.reason
    assert "vacuously" in item.reason


# --- fragment routing at array scale ------------------------------------------


def test_array_products_route_qf_nra_and_scalar_coefficients_stay_lra():
    x = var(0, F64_ARR[2])
    s = var(1, F64_ARR[2])
    pred = var(2, BOOL_ARR[2])
    out = var(3, BOOL_ARR[2])
    q_nra = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("mul", [x, x], s),
            eqn("le", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole_slice(q_nra).fragment == "QF_NRA"
    q_lra = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("mul", [lit(3.0), x], s),
            eqn("le", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert sole_slice(q_lra).fragment == "QF_LRA"


# --- array replay and the witness validator -----------------------------------


def test_array_replay_needs_every_element_and_names_the_missing_one():
    sl = sole_slice(vec_gt_query(3))
    with pytest.raises(ReplayError, match="x0_1"):
        evaluate_predicate(
            sl, {"x0_0": Fraction(1), "x0_2": Fraction(1)}
        )


def test_witness_validator_membership_is_per_element_and_names_it():
    sl = sole_slice(vec_gt_query(3))
    problem = witness_is_valid(
        sl,
        {
            "x0_0": Fraction(-1, 2),
            "x0_1": Fraction(2),  # above hi = 1: out of box
            "x0_2": Fraction(0),
        },
    )
    assert problem is not None
    assert "x0_1" in problem and "above its declared upper bound" in problem


def test_witness_validator_accepts_a_real_array_refutation():
    sl = sole_slice(vec_gt_query(3))
    vals = {
        "x0_0": Fraction(1, 2),
        "x0_1": Fraction(-1, 2),  # violates x > 0
        "x0_2": Fraction(1, 2),
    }
    assert witness_is_valid(sl, vals) is None
    assert violating_elements(sl, vals) == (1,)


def test_witness_validator_rejects_a_nonviolating_array_point():
    sl = sole_slice(vec_gt_query(3))
    vals = {
        "x0_0": Fraction(1, 2),
        "x0_1": Fraction(1, 2),
        "x0_2": Fraction(1, 2),
    }
    problem = witness_is_valid(sl, vals)
    assert problem is not None and "TRUE" in problem


def test_multidim_violating_elements_are_flat_c_order_indices():
    x = var(0, aval((2, 2)))
    pred = var(1, aval((2, 2), "bool"))
    out = var(2, aval((2, 2), "bool"))
    q = close(
        [
            any_eqn(x, -1.0, 1.0, shape=(2, 2)),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    assert [i.name for i in sl.inputs] == ["x0_0", "x0_1", "x0_2", "x0_3"]
    vals = {
        "x0_0": Fraction(1),
        "x0_1": Fraction(-1),  # coord (0,1) -> flat 1
        "x0_2": Fraction(1),
        "x0_3": Fraction(-1),  # coord (1,1) -> flat 3
    }
    assert violating_elements(sl, vals) == (1, 3)


# --- array constants ----------------------------------------------------------


def test_array_constant_decodes_per_element_in_emission_and_replay():
    x = var(0, F64_ARR[2])
    s = var(1, F64_ARR[2])
    pred = var(2, BOOL_ARR[2])
    out = var(3, BOOL_ARR[2])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("add", [x, arr_lit([0.25, 0.75])], s),
            eqn("le", [s, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = sole_slice(q)
    text = emit(sl, "z3", 100).text
    assert "(+ x0_0 (/ 1 4))" in text
    assert "(+ x0_1 (/ 3 4))" in text
    # replay agrees elementwise: x = [0.5, 0.5] -> s = [0.75, 1.25]
    vals = {"x0_0": Fraction(1, 2), "x0_1": Fraction(1, 2)}
    assert evaluate_predicate(sl, vals) is False
    assert violating_elements(sl, vals) == (1,)


# --- scalar byte-identity: pinned script hashes -------------------------------

# sha256 of the exact scripts the PRE-ARRAY emission produced for two
# scalar obligations, computed at the pre-build HEAD and pinned: the array
# build must not change a single byte of any scalar emission.
PIN_Z3_SQUARE = (
    "61ffd0a0f0f381879f15e928b52689faf0907043230149e17daa488929c9a5e5"
)
PIN_CVC5_MIXED = (
    "f8520a4f9f43cbf8d5033c3e2e2e8229eed2c8b55b0f8d363ade5ddfcc7ce6d5"
)


def test_scalar_scripts_are_byte_identical_to_the_pre_array_emission():
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q1 = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    s1 = emit(sole_slice(q1), "z3", 750)
    assert s1.sha256 == PIN_Z3_SQUARE
    assert s1.sha256 == hashlib.sha256(s1.text.encode()).hexdigest()

    f1 = ir.Aval(kind="ShapedArray", shape=(1,), dtype="float64")
    a, b = var(0), var(1)
    arr, sel1, sqz = var(2, f1), var(3, f1), var(4)
    w, picked = var(5, BOOL), var(6)
    mx, dv, pw = var(7), var(8), var(9)
    red = var(10)
    pred2, out2 = var(11, BOOL), var(12, BOOL)
    q2 = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(b, 0.5, 2.0),
            eqn(
                "broadcast_in_dim",
                [a],
                arr,
                [("shape", (1,)), ("broadcast_dimensions", ())],
            ),
            eqn(
                "slice",
                [arr],
                sel1,
                [
                    ("start_indices", (0,)),
                    ("limit_indices", (1,)),
                    ("strides", None),
                ],
            ),
            eqn("squeeze", [sel1], sqz, [("dimensions", (0,))]),
            eqn("gt", [sqz, lit(0.5)], w),
            eqn("select_n", [w, a, b], picked),
            eqn("max", [picked, b], mx),
            eqn("div", [mx, b], dv),
            eqn("integer_pow", [dv], pw, [("y", 2)]),
            eqn("reduce_sum", [pw], red, [("axes", ())]),
            eqn("le", [red, lit(1.5)], pred2),
            eqn("stelling_assert", [pred2], out2),
        ],
        [out2],
    )
    s2 = emit(sole_slice(q2), "cvc5", 500)
    assert s2.sha256 == PIN_CVC5_MIXED


# --- elementwise pairing declines --------------------------------------------


def test_incompatible_shapes_decline_quoted():
    x = var(0, F64_ARR[2])
    y = var(1, F64_ARR[3])
    s = var(2, F64_ARR[3])
    pred = var(3, BOOL_ARR[3])
    out = var(4, BOOL_ARR[3])
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            any_eqn(y, 0.0, 1.0, shape=(3,)),
            eqn("add", [x, y], s),
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert "'add'" in item.reason


def test_select_n_array_selector_shape_mismatch_declines():
    w = var(0, BOOL_ARR[2])
    x = var(1, F64_ARR[3])
    y = var(2, F64_ARR[3])
    sel = var(3, F64_ARR[3])
    from stelling.obligation import _Decline

    e = ir.JaxprEqn(
        primitive="select_n",
        invars=(w, x, y),
        outvars=(sel,),
        params=(),
    )
    with pytest.raises(_Decline, match="selector shape"):
        _pair_select_n(e)


def test_pairing_general_rank_broadcast_matches_numpy_semantics():
    # (2,1) against (2,3): numpy broadcast (measured via np.broadcast_shapes
    # and the propagation's own _pair_elements) — a: [0,0,0,1,1,1], b: 0..5
    e = ir.JaxprEqn(
        primitive="add",
        invars=(var(0, aval((2, 1))), var(1, aval((2, 3)))),
        outvars=(var(2, aval((2, 3))),),
        params=(),
    )
    ia, ib = _pair_elementwise(e)
    assert ia == [0, 0, 0, 1, 1, 1]
    assert ib == [0, 1, 2, 3, 4, 5]


# --- fake-transport array witnesses: the dispatch paths per element -----------
#
# The existing loud/degrade contracts, now exercised at array scale through
# the external-binary transport against fake "solvers" (works in every
# venv, including zero-dep).

from stelling.solvers import (  # noqa: E402
    NO_USABLE_MODEL,
    EmissionInfidelityError,
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from test_solver_dispatch import VERSIONS, fake_solver  # noqa: E402


def escalate_vec(monkeypatch, fake_path, n=2, timeout_ms=2000):
    monkeypatch.setenv("STELLING_CVC5", fake_path)
    q = vec_gt_query(n, lo=-1.0, hi=1.0, bound=0.0)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    config = SolverConfig(timeout_ms=timeout_ms, only=("cvc5",))
    return q, p, escalate(q, p, config)


def test_fake_sat_array_model_becomes_refuted_naming_the_element(
    monkeypatch, tmp_path
):
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_0 () Real (/ 1 2))")\n'
        'print("  (define-fun x0_1 () Real (- (/ 1 4)))")\n'
        'print(")")',
        "cvc5-arraysat",
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    assert record.witness.values == (("x0_0", "1/2"), ("x0_1", "-1/4"))
    assert record.witness.violating_elements == (1,)
    assert "violating element(s) of the assert operand: 1" in record.detail
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    rendered = v.render()
    assert "x0_0 = 1/2" in rendered and "x0_1 = -1/4" in rendered
    assert "violating element(s) of the assert operand: 1" in rendered


def test_fake_sat_out_of_box_element_raises_emission_infidelity(
    monkeypatch, tmp_path
):
    # element 1 escapes the declared box [-1, 1]: the single validator's
    # membership conjunct must catch it LOUDLY, naming the element's input
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_0 () Real (- (/ 1 2)))")\n'
        'print("  (define-fun x0_1 () Real 2.0)")\n'
        'print(")")',
        "cvc5-outofbox",
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = vec_gt_query(2, lo=-1.0, hi=1.0, bound=0.0)
    p = propagate(q)
    with pytest.raises(EmissionInfidelityError) as info:
        escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    assert "x0_1" in str(info.value)


def test_fake_sat_missing_element_is_completed_from_its_own_bounds(
    monkeypatch, tmp_path
):
    # the model supplies only element 1 (a genuine violation); element 0
    # is a per-element don't-care completed from ITS OWN bounds, disclosed
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_1 () Real (- (/ 3 4)))")\n'
        'print(")")',
        "cvc5-partial",
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    values = dict(record.witness.values)
    assert values["x0_1"] == "-3/4"
    assert values["x0_0"] == "-1"  # completed from the declared lo
    assert any(
        "omitted x0_0 (don't-care)" in n and "completed with -1" in n
        for n in record.notes
    )
    # completion is itself in-box and the replay still decided: -1 also
    # violates x > 0, so both elements are named
    assert record.witness.violating_elements == (0, 1)


def test_fake_sat_no_elements_at_all_is_the_transport_failure_unknown(
    monkeypatch, tmp_path
):
    fake = fake_solver(
        tmp_path, 'print("sat")\nprint("(")\nprint(")")', "cvc5-empty"
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert NO_USABLE_MODEL in record.detail
    assert make_solver_verdict(q, p, esc, **VERSIONS).status == "UNKNOWN"


def test_fake_sat_garbage_element_value_degrades_quoted(monkeypatch, tmp_path):
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_0 () Real flarb)")\n'
        'print("  (define-fun x0_1 () Real (/ 1 2))")\n'
        'print(")")',
        "cvc5-garbageval",
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    # an unparseable arith value is exactly what nonrational means: the
    # witness is not independently replayable — UNKNOWN by policy
    assert record.outcome == "unknown"
    assert "not independently replayable" in record.detail


def test_fake_sat_conflicting_duplicate_element_definitions_degrade(
    monkeypatch, tmp_path
):
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_0 () Real (/ 1 2))")\n'
        'print("  (define-fun x0_0 () Real (/ 1 4))")\n'
        'print("  (define-fun x0_1 () Real (- 1.0))")\n'
        'print(")")',
        "cvc5-dupes",
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert any("conflicting duplicate" in n for n in record.notes)


# --- the transport model-screening fix (latent, exposed by this build) --------
#
# z3's sat models list the script's Bool define-funs among the model decls
# (valued as EXPRESSIONS), and the pre-fix z3 transport flagged every such
# entry "non-rational", so ANY sat model for a predicate containing a
# comparison (i.e. all of them) was degraded "not independently
# replayable" whenever z3's model was the one consulted. Latent since the
# first z3 transport — the QF_NRA acceptance paths consult cvc5's model
# first — and exposed by the first QF_LRA sat-with-witness path (the
# array FACE case, where z3 is primary). The cvc5 external-binary parser
# carried the same class (a Bool-sorted define-fun echo failed Fraction
# parsing and set the flag); the cvc5 wheel driver was already clean
# (iterates DECLARED terms only). Axis of the sweep: transport
# implementation (z3-wheel / cvc5-wheel-driver / cvc5-external-binary).


def test_binary_model_parser_skips_bool_definitions_keeps_arith_flag():
    from stelling.solvers import _model_values_from_text

    # a Bool define-fun echo must not poison replayability
    values, nonrational = _model_values_from_text(
        "(\n"
        "  (define-fun x0 () Real (/ 1 2))\n"
        "  (define-fun t1 () Bool true)\n"
        ")"
    )
    assert values == (("x0", "1/2"),)
    assert nonrational is False
    # a genuinely non-rational ARITH value still flags (never a guess)
    values2, nonrational2 = _model_values_from_text(
        "(\n"
        "  (define-fun x0 () Real (_ real_algebraic_number <...>))\n"
        ")"
    )
    assert nonrational2 is True


def test_fake_sat_with_bool_definition_echo_still_yields_the_witness(
    monkeypatch, tmp_path
):
    # end-to-end through the external-binary transport: the echoed Bool
    # definition is ignored, the witness stands
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0_0 () Real (/ 1 2))")\n'
        'print("  (define-fun x0_1 () Real (- (/ 1 2)))")\n'
        'print("  (define-fun t1_0 () Bool true)")\n'
        'print("  (define-fun t1_1 () Bool false)")\n'
        'print(")")',
        "cvc5-boolecho",
    )
    q, p, esc = escalate_vec(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    assert record.witness.violating_elements == (1,)


@pytest.mark.skipif(
    not __import__("stelling._optional", fromlist=["available"]).available("z3"),
    reason="needs the z3 wheel",
)
def test_z3_wheel_sat_model_with_definition_echoes_is_replayable():
    # the z3-wheel half of the fix, against the real z3. Measured: z3
    # lists the script's define-funs among the model decls, valued as
    # EXPRESSIONS — Bool echoes (`t2_0 -> x0_0 + x0 > 1/2`) AND Real
    # echoes (`t1_0 -> x0_0 + x0`); the first fix version handled only
    # the Bool sort and was caught by the FACE acceptance case itself
    # (the re-attack rule working as intended). Both sorts must be
    # skipped; the declared inputs' rational values must survive.
    from stelling.smt import emit as _emit
    from stelling.solvers import _run_z3

    a = var(0)
    x = var(1, F64_ARR[2])
    s = var(2, F64_ARR[2])  # Real-sorted intermediate: Real echoes
    pred = var(3, BOOL_ARR[2])  # Bool-sorted comparison: Bool echoes
    out = var(4, BOOL_ARR[2])
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("add", [x, a], s),
            eqn("ge", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    script = _emit(sole_slice(q), "z3", 5000)
    r = _run_z3(script.text, 30.0)
    assert r.answer == "sat"
    assert r.nonrational is False
    got = {name for name, _ in r.values}
    assert {"x0", "x1_0", "x1_1"} <= got


@pytest.mark.skipif(
    not __import__("stelling._optional", fromlist=["available"]).available("z3"),
    reason="needs the z3 wheel",
)
def test_z3_wheel_algebraic_witness_value_still_flags_nonrational():
    # the sqrt2 policy must survive the fix: an ALGEBRAIC witness value
    # (a root-obj) is precisely what nonrational means, and stays flagged
    from stelling.smt import emit as _emit
    from stelling.solvers import _run_z3

    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("ne", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    script = _emit(sole_slice(q), "z3", 15000)
    r = _run_z3(script.text, 60.0)
    assert r.answer == "sat"
    assert r.nonrational is True  # x = sqrt(2): not independently replayable


# --- fix round (first-contact audit F1-F4) ------------------------------------
#
# F1/F2: the budget gate binds BOTH quantities (element terms AND root
# conjuncts) and is consulted from static shape metadata BEFORE any
# O(#elements) slice work. F3/F4: the shared routing oracle
# (stelling.interval) validates the measured jax contract and declines
# jax-illegal forms instead of silently mis-routing (F3) or crashing the
# propagation walk (F4).

import time  # noqa: E402


def test_f1_structural_root_inflation_declines_quoting_both_quantities():
    # audit c07 finding (i): scalar predicate broadcast to a huge root —
    # 3 element terms, n root conjuncts; the gate must bind the conjunction
    n = 4 * ELEMENT_BUDGET
    a = var(0)
    p = var(1, BOOL)
    b = var(2, aval((n,), "bool"))
    out = var(3, aval((n,), "bool"))
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            eqn("le", [a, lit(0.5)], p),
            eqn(
                "broadcast_in_dim",
                [p],
                b,
                [("shape", (n,)), ("broadcast_dimensions", ())],
            ),
            eqn("stelling_assert", [b], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert "2 element terms" in item.reason  # the le + input
    assert f"{n} root conjuncts" in item.reason
    assert str(ELEMENT_BUDGET) in item.reason
    # an in-budget inflation still emits: the gate must not over-guard
    m = ELEMENT_BUDGET // 2
    b2 = var(2, aval((m,), "bool"))
    out2 = var(3, aval((m,), "bool"))
    q2 = close(
        [
            any_eqn(a, 0.0, 1.0),
            eqn("le", [a, lit(0.5)], p),
            eqn(
                "broadcast_in_dim",
                [p],
                b2,
                [("shape", (m,)), ("broadcast_dimensions", ())],
            ),
            eqn("stelling_assert", [b2], out2),
        ],
        [out2],
    )
    sl = sole_slice(q2)
    text = emit(sl, "z3", 100).text
    # the conjunction repeats the ONE shared term m times (sharing intact)
    assert text.count("t1") == m + 1  # the define, plus m conjunct references


def test_f1_reduce_sum_bodies_count_their_operand_elements():
    # the adjacent shape of the same class: a reduce_sum over a big
    # constant emits an n-ary body of n literals while minting ONE term —
    # its operand elements count toward the budget
    n = 2 * ELEMENT_BUDGET
    a = var(0)
    s = var(1)
    t = var(2)
    pred = var(3, BOOL)
    out = var(4, BOOL)
    q = close(
        [
            any_eqn(a, 0.0, 1.0),
            eqn(
                "reduce_sum",
                [arr_lit([0.5] * n)],
                s,
                [("axes", (0,))],
            ),
            eqn("add", [a, s], t),
            eqn("le", [t, lit(float(n) / 2 + 0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole(q)
    assert isinstance(item, DeclinedObligation)
    assert "element terms" in item.reason and str(ELEMENT_BUDGET) in item.reason


def test_f2_over_budget_decline_is_cheap_never_linear_in_elements():
    # audit c07 finding (ii): pre-fix, an over-budget decline at n=1.6M
    # cost 2.91 s and hundreds of MB (SliceInputs + routing boxes built
    # before the gate). Post-fix the gate reads static shape metadata
    # only. n=800k declined in ~1 ms when this was measured; the bound
    # is 100x slack against machine noise, and the PRE-fix cost at this
    # n (~1.4 s, interpolating the auditor's scaling) fails it clearly.
    n = 800_000
    x = var(0, aval((n,)))
    pred = var(1, aval((n,), "bool"))
    out = var(2, aval((n,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(n,)),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    env = interval_env(q)  # the propagation walk is O(n) by nature; the
    # DECLINE must not be — time slice_obligation alone
    t0 = time.perf_counter()
    item = slice_obligation(q, 0, env)
    dt = time.perf_counter() - t0
    assert isinstance(item, DeclinedObligation)
    assert f"{2 * n} element terms" in item.reason
    assert dt < 0.4, f"over-budget decline took {dt:.3f}s at n={n}"


def test_f3_oracle_rejects_jax_illegal_broadcast_in_dim_forms():
    # the ORACLE itself (stelling.interval) — shared by propagation,
    # emission, and replay — must decline every measured-illegal form:
    # jax 0.11.0 rejects each of these at lax level (c11 ground truth)
    import stelling.interval as ivv

    box2 = ivv.IntervalArray(shape=(2,), los=(0.0, 1.0), his=(0.0, 1.0))
    box22 = ivv.IntervalArray(
        shape=(2, 2), los=(0.0, 1.0, 2.0, 3.0), his=(0.0, 1.0, 2.0, 3.0)
    )
    with pytest.raises(ivv.IntervalError, match="length equal to the operand rank"):
        ivv.broadcast_in_dim(box2, (2,), ())  # bd too short: was a silent
        # every-element-aliases-element-0 misroute (dropped dependence)
    with pytest.raises(ivv.IntervalError, match="duplicates"):
        ivv.broadcast_in_dim(box22, (2, 2), (0, 0))  # silent diagonal
    with pytest.raises(ivv.IntervalError, match="outside the output rank"):
        ivv.broadcast_in_dim(box2, (2,), (5,))  # was a raw IndexError
    with pytest.raises(ivv.IntervalError, match="neither 1 nor the output"):
        ivv.broadcast_in_dim(box2, (3,), (0,))  # extent disagreement
    # and the LEGAL non-monotonic form still routes exactly as measured
    # jax (transpose-broadcast; lax accepts and traces bd=(1,0)):
    box23 = ivv.IntervalArray(
        shape=(2, 3),
        los=tuple(float(i) for i in range(6)),
        his=tuple(float(i) for i in range(6)),
    )
    got = ivv.broadcast_in_dim(box23, (3, 2), (1, 0))
    assert got.los == (0.0, 3.0, 1.0, 4.0, 2.0, 5.0)  # measured jax ravel


def test_f3_oracle_rejects_jax_illegal_squeeze_and_concatenate_forms():
    import stelling.interval as ivv

    box23 = ivv.IntervalArray(
        shape=(2, 3),
        los=tuple(float(i) for i in range(6)),
        his=tuple(float(i) for i in range(6)),
    )
    with pytest.raises(ivv.IntervalError, match="has size 2, not 1"):
        ivv.squeeze(box23, (0,))
    with pytest.raises(ivv.IntervalError, match="out of range"):
        ivv.squeeze(box23, (5,))  # was silently IGNORED (identity)
    with pytest.raises(ivv.IntervalError, match="not distinct"):
        ivv.squeeze(box23, (0, 0))
    box2 = ivv.IntervalArray(shape=(2,), los=(0.0, 1.0), his=(0.0, 1.0))
    with pytest.raises(ivv.IntervalError, match="out of bounds"):
        ivv.concatenate([box2, box2], 1)
    box33 = ivv.IntervalArray(
        shape=(3, 3),
        los=tuple(float(i) for i in range(9)),
        his=tuple(float(i) for i in range(9)),
    )
    box22 = ivv.IntervalArray(
        shape=(2, 2), los=(0.0, 1.0, 2.0, 3.0), his=(0.0, 1.0, 2.0, 3.0)
    )
    with pytest.raises(ivv.IntervalError, match="disagree off the concatenation"):
        ivv.concatenate([box22, box33], 0)  # was a silent wrong-shape read
    with pytest.raises(ivv.IntervalError, match="no operands"):
        ivv.concatenate([], 0)


def bad_bid_query(in_shape, out_shape, bd_params):
    x = var(0, aval(in_shape))
    b = var(1, aval(out_shape))
    pred = var(2, aval(out_shape, "bool"))
    out = var(3, aval(out_shape, "bool"))
    return close(
        [
            any_eqn(x, 0.0, 1.0, shape=in_shape),
            eqn("broadcast_in_dim", [x], b, bd_params),
            eqn("le", [b, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_f3_the_flip_query_no_longer_mints_a_refuted():
    # audit c11 B1: b := broadcast_in_dim(x, (2,), ()); assert b - x == 0.
    # The plain jax reading of the call is the identity (property holds);
    # pre-fix the shared misroute made b = [x0, x0] and REFUTED with a
    # "confirmed" witness. Post-fix: declined, UNKNOWN, quoted.
    x = var(0, aval((2,)))
    b = var(1, aval((2,)))
    d = var(2, aval((2,)))
    pred = var(3, aval((2,), "bool"))
    out = var(4, aval((2,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn(
                "broadcast_in_dim",
                [x],
                b,
                [("shape", (2,)), ("broadcast_dimensions", ())],
            ),
            eqn("sub", [b, x], d),
            eqn("eq", [d, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not crash; the transfer declines to a noted ⊤
    assert [o.status for o in p.obligations] == ["unknown"]
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], DeclinedObligation)
    assert "broadcast_in_dim" in items[0].reason
    assert "length equal to the operand rank" in items[0].reason


def test_f4_propagate_survives_malformed_structural_params():
    # audit c11 B3: pre-fix these crashed the whole propagation walk with
    # a raw IndexError / KeyError (the guard rule broken). Post-fix both
    # degrade: the oracle raises IntervalError (caught, noted ⊤) and the
    # transfers require params through the decline channel.
    q_range = bad_bid_query(
        (2,), (2,), [("shape", (2,)), ("broadcast_dimensions", (5,))]
    )
    p = propagate(q_range)  # was: IndexError
    assert [o.status for o in p.obligations] == ["unknown"]
    assert any("broadcast_in_dim" in n and "⊤" in n for n in p.notes)

    q_missing = bad_bid_query((2,), (2,), [("shape", (2,))])
    p2 = propagate(q_missing)  # was: KeyError
    assert [o.status for o in p2.obligations] == ["unknown"]
    assert any("missing its required param" in n for n in p2.notes)

    # the same class across the swept sibling transfers: reduce_sum and
    # concatenate with their required params missing
    x = var(0, aval((2,)))
    s = var(1)
    pred = var(2, BOOL)
    out = var(3, BOOL)
    q3 = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("reduce_sum", [x], s, []),  # no axes param
            eqn("le", [s, lit(1.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p3 = propagate(q3)
    assert [o.status for o in p3.obligations] == ["unknown"]
    assert any("missing its required param" in n for n in p3.notes)

    c = var(1, aval((4,)))
    pred4 = var(2, aval((4,), "bool"))
    out4 = var(3, aval((4,), "bool"))
    q4 = close(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("concatenate", [x, x], c, []),  # no dimension param
            eqn("le", [c, lit(0.5)], pred4),
            eqn("stelling_assert", [pred4], out4),
        ],
        [out4],
    )
    p4 = propagate(q4)
    assert [o.status for o in p4.obligations] == ["unknown"]
    assert any("missing its required param" in n for n in p4.notes)


# --- fix round 2 (re-attack R1/R2): shape nonnegativity -----------------------
#
# Measured jax 0.11.0: every concrete context rejects a negative extent
# (jnp.zeros((-2,-2)) raises "shape must have every element be
# nonnegative"; lax.reshape rejects negative new_sizes) while zero-size
# shapes are legal everywhere. A negative-extent shape names an
# uninhabited type — the declared set is EMPTY — and stelling's tuple
# paths and coordinate enumeration DISAGREED about it ((-2,-2): 4
# elements vs 1 coordinate), which minted a REFUTED-with-witness from a
# sum with three addends dropped (R1, public-API reachable). Guarded at
# BOTH layers: the declaration API refuses (loud, the (inf,inf)-sibling
# posture — jax-gated test in test_array_acceptance.py) and every
# consumption layer declines quoted.




def test_r1_interval_layer_refuses_negative_extents():
    import stelling.interval as ivv

    # the box itself is unconstructable (every transfer output flows
    # through this constructor)
    with pytest.raises(ivv.IntervalError, match="negative extent"):
        ivv.IntervalArray(
            shape=(-2, -2), los=(0.0,) * 4, his=(1.0,) * 4
        )
    with pytest.raises(ivv.IntervalError, match="negative extent"):
        ivv.from_bounds((-2, -2), 0.0, 1.0)
    with pytest.raises(ivv.IntervalError, match="negative extent"):
        ivv.top((-2,))
    # the coordinate enumeration refuses instead of yielding ONE
    # coordinate for a 4-element product (the R1 inconsistency)
    with pytest.raises(ivv.IntervalError, match="negative extent"):
        list(ivv._coords((-2, -2)))
    # zero-size shapes stay legal (do not over-guard — measured jax
    # constructs them; the vacuous-discharge convention is pinned above)
    assert list(ivv._coords((0, 3))) == []
    assert ivv.from_bounds((0,), 0.0, 1.0).size == 0


def test_r2_reshape_rejects_negative_new_sizes_despite_matching_product():
    import stelling.interval as ivv

    box4 = ivv.IntervalArray(
        shape=(4,), los=(0.0, 1.0, 2.0, 3.0), his=(0.0, 1.0, 2.0, 3.0)
    )
    # (-1)*(-4) == 4: the count check alone would accept (re-attack R2);
    # measured jax rejects ("reshape new_sizes must all be positive")
    with pytest.raises(ivv.IntervalError, match="negative"):
        ivv.reshape(box4, (-1, -4))
    # zero-size reshape stays legal (measured: lax accepts (0,)->(0,3))
    z = ivv.reshape(
        ivv.IntervalArray(shape=(0,), los=(), his=()), (0, 3)
    )
    assert z.shape == (0, 3) and z.size == 0







# --- fix round 4 (re-attack N1/N2): the read gate + decode predicates ---------








# --- fix round 6 (re-attack P1): the constvar route + the from_dict door ------








# --- the from_dict door (Part 2 of the P1 round) ------------------------------

# --- type-level supersession (the CI-readiness construction census) -----------
#
# The construction-path census (design/ci-readiness.md, Part A) found the
# well-formedness gates lived only at the two doors, leaving direct
# dataclass construction ungated — the I1 residual's route. Validation
# now runs in the ir dataclasses' own __post_init__, so the malformed IR
# the N/P/R-round regression tests used to build IS UNCONSTRUCTABLE: the
# raise moved from the pipeline/door to the constructor, strictly
# earlier, and every superseded test's intent (no wrong verdict, no raw
# crash) holds a fortiori. Superseded here (names kept for the record):
# test_r1_* (negative-shape queries), test_n1_* (lying-aval laundering),
# test_n2_* (string extents, payload lies), test_p1_* (refused constvar
# lies), test_from_dict_refuses_* (door refusals, now constructor
# refusals). The in-pipeline gates (read gate, screens, decoders) remain
# in the code as defence-in-depth behind the types.


def test_type_level_negative_extents_are_unconstructable():
    with pytest.raises(ir.TranscriptionError, match="negative shape extent"):
        aval((-2, -2))
    with pytest.raises(ir.TranscriptionError, match="negative shape extent"):
        ir.Array(dtype="<f8", shape=(-2,), data=b"\x00" * 16)
    # zero-size stays legal (do not over-guard; measured jax constructs it)
    assert aval((0, 3)).shape == (0, 3)
    assert ir.Array(dtype="<f8", shape=(0,), data=b"").shape == (0,)


def test_type_level_noninteger_extents_are_unconstructable():
    with pytest.raises(ir.TranscriptionError, match="non-integer shape extent"):
        ir.Aval(kind="ShapedArray", shape=("x",), dtype="float64")
    # index-able dims stay legal (bool dims measured consistent with jax)
    assert ir.Aval(kind="ShapedArray", shape=(True, 2), dtype="float64")


def test_type_level_payload_length_lies_are_unconstructable():
    for name, data in (
        ("truncated", struct.pack("<1d", 1.0)),
        ("oversized", struct.pack("<3d", 1.0, 2.0, 3.0)),
        ("empty", b""),
    ):
        with pytest.raises(ir.TranscriptionError, match="byte"):
            ir.Array(dtype="<f8", shape=(2,), data=data)
    # a correct payload constructs and still emits exactly (no over-guard)
    good_arr = ir.Literal(
        val=ir.Array(dtype="<f8", shape=(2,), data=struct.pack("<2d", 0.25, 0.75)),
        aval=aval((2,)),
    )
    good = close(
        [
            any_eqn(var(0, F64_ARR[2]), 0.0, 1.0, shape=(2,)),
            eqn("add", [var(0, F64_ARR[2]), good_arr], var(1, F64_ARR[2])),
            eqn("le", [var(1, F64_ARR[2]), lit(1.5)], var(2, BOOL_ARR[2])),
            eqn("stelling_assert", [var(2, BOOL_ARR[2])], var(3, BOOL_ARR[2])),
        ],
        [var(3, BOOL_ARR[2])],
    )
    p = propagate(good)
    items = slice_unknown_obligations(good, p, interval_env(good))
    assert isinstance(items[0], ObligationSlice)
    text = emit(items[0], "z3", 100).text
    assert "(+ x0_0 (/ 1 4))" in text and "(+ x0_1 (/ 3 4))" in text


def test_type_level_literal_aval_value_disagreement_is_unconstructable():
    with pytest.raises(ir.TranscriptionError, match="contradicts the recorded aval"):
        ir.Literal(
            val=ir.Array(dtype="<f8", shape=(2,), data=struct.pack("<2d", 1.0, 2.0)),
            aval=aval((3,)),
        )
    with pytest.raises(ir.TranscriptionError, match="non-scalar aval"):
        ir.Literal(val=0.5, aval=aval((3,)))


def test_type_level_declaration_param_aval_disagreement_is_unconstructable():
    # the I1 instance itself: stelling_any params say one shape, the
    # outvar aval says another — two self-descriptions of one declared
    # set, now required to agree at construction
    with pytest.raises(ir.TranscriptionError, match="contradicts the outvar aval"):
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((3,))),),
            params=(("shape", (2,)), ("dtype", "float64"), ("lo", 0.0), ("hi", 1.0)),
        )
    with pytest.raises(ir.TranscriptionError, match="negative shape extent"):
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((4,))),),
            params=(("shape", (-2, -2)), ("dtype", "float64"), ("lo", 0.0), ("hi", 1.0)),
        )
    with pytest.raises(ir.TranscriptionError, match="contradicts the outvar aval dtype"):
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((2,))),),
            params=(("shape", (2,)), ("dtype", "float32"), ("lo", 0.0), ("hi", 1.0)),
        )


def test_the_declaration_check_reads_the_EXTENTS_not_the_param_type():
    """AUDIT 0.2.0 B6 RE-AUDIT, UNSOUND-1 — the check above used to run only
    `if isinstance(shape, tuple)`, so a `list` shape param SKIPPED it
    entirely and the exact disagreement the test above forbids was
    constructible one bracket away. `_validate_param_value` recursed into
    tuples and not lists either, so nothing else read it. A validator that
    silently passes a param class it cannot read is not a validator for that
    class; the comparison is now on the extents, whatever holds them.

    Note what is NOT refused here, and deliberately: a declaration with no
    `shape` param at all. Hand-built IR legitimately omits params (see
    `ir._validate_required_params`), so absence stays blessed — and the
    slicer's own `_one_shape_per_value` is what stands behind it, which is
    exactly why this door is not the defect's soundness boundary. See
    `tests/test_aval_lie_both_faces.py` for the slicer half."""
    for holder in (list, tuple):
        with pytest.raises(
            ir.TranscriptionError, match="contradicts the outvar aval"
        ):
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(var(0, aval((3,))),),
                params=(("shape", holder([2])), ("dtype", "float64"),
                        ("lo", 0.0), ("hi", 1.0)),
            )
        # ... and AGREEING extents in the same holder are still accepted
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((3,))),),
            params=(("shape", holder([3])), ("dtype", "float64"),
                    ("lo", 0.0), ("hi", 1.0)),
        )
    # a param that is not a sequence of extents at all is REFUSED, not
    # skipped: two self-descriptions cannot be reconciled if one of them
    # cannot be read. `str`/`bytes` are sequences and are not shapes —
    # `tuple("34")` is a tuple of CHARACTERS, so coercing one would compare
    # something the declaration never said.
    # `memoryview` and `array.array` join them not by being enumerated but
    # by falling outside the POSITIVE rule (a tuple or a list) that
    # replaced the enumeration — audit 0.2.0 B6 audit 3. Both read as
    # `(52, 52)` through `tuple(...)`, the identical misread as `b"44"`,
    # and the door ACCEPTED the memoryview form before this.
    for bad in (3, None, "34", b"34", memoryview(b"\x03"),
                _arraymod.array("b", b"\x03")):
        with pytest.raises(
            ir.TranscriptionError, match="not a sequence of extents"
        ):
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(var(0, aval((3,))),),
                params=(("shape", bad), ("dtype", "float64"),
                        ("lo", 0.0), ("hi", 1.0)),
            )
    # absence stays legal, and is the form the slicer must cover alone
    ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(var(0, aval((3,))),),
        params=(("dtype", "float64"), ("lo", 0.0), ("hi", 1.0)),
    )


def test_the_load_walk_recurses_into_LIST_params_as_well_as_tuples():
    """AUDIT 0.2.0 B6 RE-AUDIT, UNSOUND-1, the second half of the door.

    `ir._validate_param_value` dispatched on `ClosedJaxpr`, `Jaxpr`,
    `Array`, `tuple` and `NamedTupleParam`. A `list` matched none of them,
    so anything a list param held reached the rest of the library
    unvalidated — which is the same omission, in the same function family,
    that let a `list` `shape` param past `_validate_decl_eqn`.

    Driven directly rather than through `from_dict`, and that is the honest
    scope: `_decode` never builds a bare list, so no serialized document
    reaches this arm today. It is closed because "the container type I
    happened to enumerate" is not a reason for a validator to stop looking,
    and the payload below is one only this walk can catch — a missing
    required param is a LOAD-path refusal that `JaxprEqn.__post_init__`
    deliberately does not make."""
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(var(0, aval((3,))),),
            outvars=(var(1, aval(())),),
            eqns=(
                ir.JaxprEqn(
                    primitive="reduce_sum",
                    invars=(var(0, aval((3,))),),
                    outvars=(var(1, aval(())),),
                    params=(),  # `axes` is a param jax supplies on every one
                ),
            ),
        ),
        consts=(),
    )
    # constructible: hand-built IR may omit params
    with pytest.raises(ir.TranscriptionError, match="missing param"):
        ir._validate_param_value(inner, "eqn.params['direct']")
    with pytest.raises(ir.TranscriptionError, match="missing param"):
        ir._validate_param_value((inner,), "eqn.params['in a tuple']")
    with pytest.raises(ir.TranscriptionError, match="missing param"):
        ir._validate_param_value([inner], "eqn.params['in a list']")
    with pytest.raises(ir.TranscriptionError, match="missing param"):
        ir._validate_param_value([[inner]], "eqn.params['nested']")


def test_type_level_const_pairing_disagreement_is_unconstructable():
    # the P1 constvar: a scalar const under a non-scalar constvar aval
    j = ir.Jaxpr(
        constvars=(var(0, aval((3,))),),
        invars=(),
        outvars=(var(2, aval((), "bool")),),
        eqns=(
            eqn("reduce_sum", [var(0, aval((3,)))], var(1), [("axes", (0,))]),
            eqn("stelling_assert", [var(1, BOOL)], var(2, BOOL)),
        ),
    )
    with pytest.raises(ir.TranscriptionError, match="non-scalar aval"):
        ir.ClosedJaxpr(jaxpr=j, consts=(0.5,))


def test_the_door_still_refuses_malformed_dicts():
    # a malformed DICT never constructs objects until _decode, so the
    # from_dict door is still a live, distinct gate: corrupt a valid
    # query's serialized form and the load must refuse loudly
    q = vec_gt_query(3)
    d = q.to_dict()
    import copy, json

    d_bad = json.loads(json.dumps(d))
    # find and corrupt the declaration's aval shape in the dict form
    corrupted = json.loads(json.dumps(d_bad).replace("[3]", "[-3]", 1))
    with pytest.raises(ir.TranscriptionError, match="negative shape extent"):
        ir.ClosedJaxpr.from_dict(corrupted)
    # the uncorrupted dict still loads, hash byte-identical
    assert ir.ClosedJaxpr.from_dict(d).content_hash() == q.content_hash()


# -- audit 0.2.0 B6 audit 3, F1 / F2 / F3: the door's own totality ----------


class _IndexRaises:
    def __init__(self, exc):
        self._exc = exc

    def __index__(self):
        raise self._exc

    def __repr__(self):
        return f"_IndexRaises({type(self._exc).__name__})"


class _IndexOK_ReprRaises:
    """A PERFECTLY WELL-FORMED extent whose `__repr__` refuses."""

    def __init__(self, k):
        self._k = k

    def __index__(self):
        return self._k

    def __repr__(self):
        raise RuntimeError("repr refuses")


class _TwoFacedExtent:
    """`__index__` answers `first` once and `then` on every later call."""

    def __init__(self, first, then):
        self._answers = [first, then]
        self.reads = 0

    def __index__(self):
        self.reads += 1
        return self._answers[0] if self.reads == 1 else self._answers[1]

    def __repr__(self):
        return f"_TwoFacedExtent(reads={self.reads})"


@pytest.mark.parametrize(
    "exc",
    [ValueError("index says no"), OverflowError("too big"),
     RuntimeError("some other refusal")],
    ids=["ValueError", "OverflowError", "RuntimeError"],
)
def test_the_door_refuses_whatever___index___raises(exc):
    """AUDIT 0.2.0 B6 AUDIT 3, F2 — `_load_extent_problem` caught only
    `TypeError`, and `operator.index` raises whatever `__index__` raises.

    So a `ValueError` or an `OverflowError` from one extent came out of
    `ir.Aval(...)` and `ir.JaxprEqn(...)` RAW — out of a public constructor,
    which `_validate_decl_eqn`'s own docstring names as "the very class this
    function is closing". The exception type an extent chooses is not a fact
    about whether the document is malformed."""
    with pytest.raises(ir.TranscriptionError, match="non-integer shape"):
        ir.Aval(kind="ShapedArray", shape=(_IndexRaises(exc),), dtype="f8")
    with pytest.raises(ir.TranscriptionError, match="non-integer shape"):
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((3,))),),
            params=(("shape", (_IndexRaises(exc),)), ("dtype", "float64"),
                    ("lo", 0.0), ("hi", 1.0)),
        )


def test_a_hostile___repr___cannot_raise_out_of_the_public_constructor():
    """AUDIT 0.2.0 B6 AUDIT 3, F3 — and the sharpest form of it, because
    THE DOCUMENT HERE IS WELL FORMED.

    `_load_check`'s message is an ARGUMENT, so it is composed on the passing
    path as well as the failing one. `_validate_decl_eqn` interpolated
    `{shape!r}` into it unguarded, so a declaration whose extent answers
    `__index__` with 4 — matching an outvar aval of `(4,)`, nothing at all
    wrong with it — raised `RuntimeError: repr refuses` out of
    `ir.JaxprEqn(...)`. A refusal that has not been decided on may not
    crash, and one that has may not crash either: both quotes go through
    `_safe_repr`."""
    ok = ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(var(0, aval((4,))),),
        params=(("shape", (_IndexOK_ReprRaises(4),)), ("dtype", "float64"),
                ("lo", 0.0), ("hi", 1.0)),
    )
    assert ok.primitive == "stelling_any"
    # and a genuinely contradictory one still refuses, with the placeholder
    # visible rather than a plausible value
    with pytest.raises(ir.TranscriptionError) as ei:
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((4,))),),
            params=(("shape", _ReprRaisesNotASequence()), ("dtype", "f8"),
                    ("lo", 0.0), ("hi", 1.0)),
        )
    assert "not a sequence of extents" in str(ei.value)
    assert "<unreadable>" in str(ei.value), str(ei.value)


class _ReprRaisesNotASequence:
    def __iter__(self):
        raise RuntimeError("will not iterate")

    def __repr__(self):
        raise RuntimeError("repr refuses")


def test_the_door_compares_the_extents_it_VALIDATED_not_a_second_read():
    """AUDIT 0.2.0 B6 AUDIT 3, F1, in `ir.py` — the same defect the slicer
    carried, in the function that is supposed to be the door in front of it.

    `_load_extent_problem` bound `k = operator.index(d)`, tested `k` and
    DISCARDED it. `_validate_decl_eqn` then compared the RAW param objects
    against the aval with `==`, and `_validate_array_value` re-read them
    with `int(d)` for its byte-length product — so what the door validated
    and what the door used were two different reads of one self-describing
    object. Two reads of an object that answers differently each time is
    the whole of this finding's mechanism.

    Bound, the comparison is `int`-to-`int` and no `__eq__`, `__int__` or
    later `__index__` can move it: an extent that says 4 to the guard is
    compared as 4, and a later -1 cannot be smuggled past."""
    d = _TwoFacedExtent(4, -1)
    with pytest.raises(ir.TranscriptionError) as ei:
        ir.JaxprEqn(
            primitive="stelling_any",
            invars=(),
            outvars=(var(0, aval((2,))),),
            params=(("shape", (d,)), ("dtype", "float64"),
                    ("lo", 0.0), ("hi", 1.0)),
        )
    # the extents the door quotes are the ones it read, not a later answer
    assert "shape param (4,) contradicts the outvar aval shape (2,)" in str(
        ei.value
    ), str(ei.value)
    assert d.reads == 1, (
        f"the door read the extent {d.reads} times; one read is the fix"
    )
    # the agreeing case is accepted on that same single read
    d2 = _TwoFacedExtent(2, -1)
    ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(var(0, aval((2,))),),
        params=(("shape", (d2,)), ("dtype", "float64"),
                ("lo", 0.0), ("hi", 1.0)),
    )
    assert d2.reads == 1


def test_the_byte_length_product_uses_the_extents_the_guard_validated():
    """The other half of F1 in `ir.py`: `_validate_array_value` validated
    `arr.shape` with `operator.index` and then computed its expected byte
    length with a SECOND, different conversion (`int(d)`). A two-faced
    extent was therefore length-checked against a number nobody validated."""
    d = _TwoFacedExtent(2, 7)
    a = ir.Array(dtype="<f8", shape=(d,), data=b"\x00" * 16)
    assert a.shape == (d,)  # the dataclass still records what it was given
    assert d.reads == 1, (
        f"the length check read the extent {d.reads} times; one read is the "
        f"fix — the second read was `int(d)` and could disagree"
    )
    # a genuine length lie is still refused, quoting the validated extents
    with pytest.raises(ir.TranscriptionError, match=r"expected 16"):
        ir.Array(dtype="<f8", shape=(_TwoFacedExtent(2, 7),), data=b"\x00" * 8)
