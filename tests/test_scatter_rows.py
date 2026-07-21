# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The scatter-add / stack census round, at the propagation and emission
layers — no jax, no libraries: hand-built IR only (the
``test_transfers.py`` conventions).

The round's defining hazard is the ACCUMULATE semantic: duplicate scatter
indices accumulate (measured on jax 0.11.0:
``zeros(3).at[[0,2,0,0]].add([1,10,100,1000])`` is ``[1101, 0, 10]``,
where the set form's last-wins answer is ``[1000, 0, 10]``) — so these
tests pin the accumulate value itself, the exactness of untouched
elements, every measured decline path (dynamic indices, out-of-range
indices, foreign dimension numbers, a combiner contradicting the
primitive name), the integer overflow-reachability guard, the ieee
censused refusal, and the emission/replay agreement of the same
semantics. The tracing-side binding of the real primitive names lives in
``test_scatter_gauge_jax.py`` (skipped without jax), alongside the
mandated fidelity gauge.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.obligation import (
    _INT_OVERFLOW_EMITTED,
    _STRUCTURAL,
    _SUPPORTED,
    DeclinedObligation,
    evaluate_predicate,
    slice_obligation,
    witness_is_valid,
)
from stelling.propagate import (
    IEEE_TRANSFERS,
    TRANSFERS,
    _INT_COMPUTING,
    _INT_NON_COMPUTING,
    interval_env,
    propagate,
)
from stelling.smt import emit

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def aval(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)


def var(i, a=F64):
    return ir.Var(id=i, aval=a)


def lit(v, a=F64):
    return ir.Literal(val=v, aval=a)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(
            ("shape", out.aval.shape),
            ("dtype", out.aval.dtype),
            ("lo", lo),
            ("hi", hi),
        ),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns)
        )
    )


def int_idx_lit(values, shape):
    """A static int32 index literal of the given shape (flat C-order)."""
    n = 1
    for d in shape:
        n *= d
    assert len(values) == n
    arr = ir.Array(dtype="<i4", shape=tuple(shape), data=struct.pack(f"<{n}i", *values))
    return lit(arr, aval(tuple(shape), "int32"))


def scatter_dn(uwd, extra=()):
    """The measured ScatterDimensionNumbers with the core fields pinned;
    ``extra`` appends/overrides fields for the decline tests."""
    fields = dict(
        (
            ("update_window_dims", tuple(uwd)),
            ("inserted_window_dims", (0,)),
            ("scatter_dims_to_operand_dims", (0,)),
            ("operand_batching_dims", ()),
            ("scatter_indices_batching_dims", ()),
        )
    )
    fields.update(dict(extra))
    return ir.NamedTupleParam(
        cls="ScatterDimensionNumbers", fields=tuple(fields.items())
    )


def add_combiner(base_id=900):
    """The measured update_jaxpr: two scalar invars, one add, one outvar."""
    a, b, c = var(base_id), var(base_id + 1), var(base_id + 2)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(a, b), outvars=(c,),
            eqns=(eqn("add", [a, b], c),),
        )
    )


def mul_combiner(base_id=900):
    a, b, c = var(base_id), var(base_id + 1), var(base_id + 2)
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(a, b), outvars=(c,),
            eqns=(eqn("mul", [a, b], c),),
        )
    )


def scatter_add_eqn(operand, indices, updates, out, uwd, *, combiner="add",
                    dn_extra=()):
    params = [("dimension_numbers", scatter_dn(uwd, dn_extra))]
    if combiner == "add":
        params.append(("update_jaxpr", add_combiner()))
        params.append(("update_consts", ()))
    elif combiner == "mul":
        params.append(("update_jaxpr", mul_combiner()))
        params.append(("update_consts", ()))
    # combiner=None: no update_jaxpr param at all (hand-built form)
    return eqn("scatter-add", [operand, indices, updates], out, params)


# --- registration and census totality ---------------------------------------


def test_rows_registered_with_tiers_and_censused():
    assert TRANSFERS["scatter-add"][1] == "sound"
    assert TRANSFERS["stack"][1] == "exact"
    # the ieee census is total (the import-time assert enforces it; bind
    # the two names here so the binding survives a refactor of the assert)
    assert "scatter-add" in IEEE_TRANSFERS and "stack" in IEEE_TRANSFERS
    # integer-semantics census: the accumulate COMPUTES (the add class),
    # stack routes (pure data movement)
    assert "scatter-add" in _INT_COMPUTING
    assert "stack" in _INT_NON_COMPUTING
    # emission census: same classification at the emission sites
    assert "scatter-add" in _SUPPORTED and "scatter-add" in _INT_OVERFLOW_EMITTED
    assert "stack" in _STRUCTURAL


# --- the accumulate semantic (the round's defining fact) ---------------------


def _dup_query(combiner="add"):
    """zeros(3).at[[0,2,0,0]].add([1,10,100,1000]) as hand IR: operand a
    point-declared (3,) zero box, updates four point-declared elements."""
    op, up, out, pred, ob = (
        var(0, aval((3,))), var(1, aval((4,))), var(2, aval((3,))),
        var(3, aval((3,), "bool")), var(4, aval((3,), "bool")),
    )
    ups = ir.Array(
        dtype="<f8", shape=(4,), data=struct.pack("<4d", 1.0, 10.0, 100.0, 1000.0)
    )
    return close(
        [
            any_eqn(op, 0.0, 0.0),
            eqn("stelling_any", [], up),  # placeholder replaced below
        ],
        [ob],
    ), ups  # (unused; kept for clarity of the constant)


def _accumulate_query(op_bounds, out_id=2):
    """operand (3,) declared to op_bounds; indices [[0],[2],[0],[0]];
    updates the point literal [1, 10, 100, 1000]; returns (closed, out var
    id). out = operand + [1101, 0(+nothing), 10] contributions."""
    op = var(0, aval((3,)))
    out = var(out_id, aval((3,)))
    ups = lit(
        ir.Array(
            dtype="<f8", shape=(4,),
            data=struct.pack("<4d", 1.0, 10.0, 100.0, 1000.0),
        ),
        aval((4,)),
    )
    idx = int_idx_lit([0, 2, 0, 0], (4, 1))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(op, *op_bounds),
            scatter_add_eqn(op, idx, ups, out, uwd=()),
            eqn("ge", [out, lit(-1e9)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    return closed, out.id


def test_duplicates_accumulate_not_replace():
    closed, oid = _accumulate_query((0.0, 0.0))
    env = interval_env(closed)
    box = env[oid]
    ulp = math.ulp(1101.0)
    # element 0: three contributions accumulate to 1101 (within the three
    # outward bumps its three real additions earn)
    assert box.los[0] <= 1101.0 <= box.his[0]
    assert box.his[0] - box.los[0] <= 8 * ulp
    # the set-form last-wins answer (1000), the drop-duplicates answer (1),
    # and the first-contribution-only answer are all EXCLUDED — the box
    # discriminates accumulate from every wrong combining semantic
    assert not box.los[0] <= 1000.0 <= box.his[0]
    assert not box.los[0] <= 1.0 <= box.his[0]
    # element 1: untouched — EXACT copy of the operand point, no bump
    assert (box.los[1], box.his[1]) == (0.0, 0.0)
    # element 2: one contribution
    assert box.los[2] <= 10.0 <= box.his[2]
    assert box.his[2] - box.los[2] <= 4 * math.ulp(10.0)


def test_obligation_discharges_and_refutes():
    # operand in [0, 1]: out[0] in [1101, 1102]-ish => >= 1100.5 discharges
    op = var(0, aval((3,)))
    out = var(2, aval((3,)))
    ups = lit(
        ir.Array(
            dtype="<f8", shape=(4,),
            data=struct.pack("<4d", 1.0, 10.0, 100.0, 1000.0),
        ),
        aval((4,)),
    )
    idx = int_idx_lit([0, 2, 0, 0], (4, 1))
    s0, pred, ob = var(3), var(4, BOOL), var(5, BOOL)

    def q(cmp, bound):
        return close(
            [
                any_eqn(op, 0.0, 1.0),
                scatter_add_eqn(op, idx, ups, out, uwd=()),
                eqn(
                    "slice", [out], var(6, aval((1,))),
                    (("start_indices", (0,)), ("limit_indices", (1,)),
                     ("strides", None)),
                ),
                eqn("reshape", [var(6, aval((1,)))], s0,
                    (("new_sizes", ()), ("dimensions", None))),
                eqn(cmp, [s0, lit(bound)], pred),
                eqn("stelling_assert", [pred], ob),
            ],
            [ob],
        )

    p = propagate(q("ge", 1100.5))
    assert p.obligations[0].status == "discharged"
    assert ("scatter-add", "sound") in p.transfers_used
    # definite violation: out[0] <= 1101 + 1 + slack can never reach 2000
    p2 = propagate(q("ge", 2000.0))
    assert p2.obligations[0].status == "violated-over-set"


def test_scalar_index_sugar_rank1_and_rank2():
    # rank-1: x.at[1].add(v) — indices (1,), updates ()
    x, v, out = var(0, aval((3,))), var(1), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 2.0, 2.0),
            any_eqn(v, 5.0, 5.0),
            scatter_add_eqn(x, int_idx_lit([1], (1,)), v, out, uwd=()),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    env = interval_env(closed)
    box = env[2]
    assert (box.los[0], box.his[0]) == (2.0, 2.0)  # untouched: exact
    assert box.los[1] <= 7.0 <= box.his[1]
    assert box.his[1] - box.los[1] <= 4 * math.ulp(7.0)
    assert (box.los[2], box.his[2]) == (2.0, 2.0)

    # rank-2: x.at[1].add(v) on (3, 2) — indices (1,), updates (2,),
    # update_window_dims (0,) (the measured sugar form)
    x2, v2, out2 = var(0, aval((3, 2))), var(1, aval((2,))), var(2, aval((3, 2)))
    pred2, ob2 = var(3, aval((3, 2), "bool")), var(4, aval((3, 2), "bool"))
    closed2 = close(
        [
            any_eqn(x2, 1.0, 1.0),
            any_eqn(v2, 3.0, 3.0),
            scatter_add_eqn(x2, int_idx_lit([1], (1,)), v2, out2, uwd=(0,)),
            eqn("ge", [out2, lit(0.0)], pred2),
            eqn("stelling_assert", [pred2], ob2),
        ],
        [ob2],
    )
    box2 = interval_env(closed2)[2]
    # row 1 (flat elements 2, 3) accumulated; rows 0 and 2 exact
    assert (box2.los[0], box2.his[0]) == (1.0, 1.0)
    assert box2.los[2] <= 4.0 <= box2.his[2]
    assert box2.los[3] <= 4.0 <= box2.his[3]
    assert (box2.los[4], box2.his[4]) == (1.0, 1.0)


def test_trailing_dims_rows_accumulate():
    # the segment_sum M-assembly shape: operand (2,2,2), indices (3,1)
    # [0,1,0], updates (3,2,2) — rows 0 and 2 both land on segment 0
    op = var(0, aval((2, 2, 2)))
    ups = var(1, aval((3, 2, 2)))
    out = var(2, aval((2, 2, 2)))
    pred, ob = var(3, aval((2, 2, 2), "bool")), var(4, aval((2, 2, 2), "bool"))
    closed = close(
        [
            any_eqn(op, 0.0, 0.0),
            any_eqn(ups, 1.0, 1.0),
            scatter_add_eqn(
                op, int_idx_lit([0, 1, 0], (3, 1)), ups, out, uwd=(1, 2)
            ),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    box = interval_env(closed)[2]
    # segment 0 (flat 0..3): two contributions of 1.0 each => 2.0
    for i in range(4):
        assert box.los[i] <= 2.0 <= box.his[i]
        assert box.his[i] - box.los[i] <= 6 * math.ulp(2.0)
        assert not box.los[i] <= 1.0 <= box.his[i]  # last-wins excluded
    # segment 1 (flat 4..7): one contribution => 1.0
    for i in range(4, 8):
        assert box.los[i] <= 1.0 <= box.his[i]


# --- the decline paths (censused refusals) -----------------------------------


def _declined(closed, needle):
    p = propagate(closed)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown >= 1
    assert any(needle in n for n in p.notes), p.notes
    return p


def test_dynamic_indices_refused_loudly():
    x, i, v, out = var(0, aval((3,))), var(1, aval((1,), "int32")), var(2), var(3, aval((3,)))
    pred, ob = var(4, aval((3,), "bool")), var(5, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(i, 0.0, 2.0),  # a TRACED index: interval, not a point
            any_eqn(v, 0.0, 1.0),
            scatter_add_eqn(x, i, v, out, uwd=()),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    _declined(closed, "not definite integers")


def test_out_of_range_index_refused_loudly():
    x, v, out = var(0, aval((3,))), var(1), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(v, 0.0, 1.0),
            scatter_add_eqn(x, int_idx_lit([5], (1,)), v, out, uwd=()),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    p = _declined(closed, "out of range")
    # the reason names the mode dependence (drop vs clamp), never guesses
    assert any("mode-dependent" in n for n in p.notes)


def test_foreign_configurations_decline_to_noted_top():
    x, out = var(0, aval((3,))), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))

    def q(sc_eqn):
        return close(
            [
                any_eqn(x, 0.0, 1.0),
                any_eqn(var(1, aval((2,))), 0.0, 1.0),
                sc_eqn,
                eqn("ge", [out, lit(0.0)], pred),
                eqn("stelling_assert", [pred], ob),
            ],
            [ob],
        )

    ups = var(1, aval((2,)))
    idx = int_idx_lit([0, 1], (2, 1))
    # (a) nonempty batching dims: outside every measured form
    _declined(
        q(scatter_add_eqn(x, idx, ups, out, uwd=(),
                          dn_extra=(("operand_batching_dims", (0,)),))),
        "no sound rule",
    )
    # (b) a combiner that contradicts the primitive name (mul, not add)
    _declined(q(scatter_add_eqn(x, idx, ups, out, uwd=(), combiner="mul")),
              "no sound rule")
    # (c) updates shape inconsistent with the index count
    bad_ups = var(1, aval((2,)))
    _declined(
        q(scatter_add_eqn(x, int_idx_lit([0, 1, 2], (3, 1)), bad_ups, out,
                          uwd=())),
        "no sound rule",
    )
    # (d) foreign update_window_dims for the rank
    _declined(q(scatter_add_eqn(x, idx, ups, out, uwd=(1,))), "no sound rule")


def test_absent_combiner_is_accepted_hand_built_form():
    # primitive name is the semantic authority; update_jaxpr=absent is the
    # hand-built form and must work (combiner=None omits the param)
    x, v, out = var(0, aval((2,))), var(1, aval((1,))), var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))
    closed = close(
        [
            any_eqn(x, 1.0, 1.0),
            any_eqn(v, 2.0, 2.0),
            scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=(),
                            combiner=None),
            eqn("ge", [out, lit(0.5)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    p = propagate(closed)
    assert p.obligations[0].status == "discharged"


def test_int_scatter_add_overflow_declines():
    """The dedicated integer-boundary shape referenced from
    test_three_rows.py's census-binding loop: the in-range integer
    accumulate keeps its exact (snapped) result; the boundary accumulate
    declines with the range quoted."""
    imax = 2**31 - 1

    def q(op_val, up_val):
        x = var(0, aval((1,), "int32"))
        v = var(1, aval((1,), "int32"))
        out = var(2, aval((1,), "int32"))
        pred, ob = var(3, aval((1,), "bool")), var(4, aval((1,), "bool"))
        return close(
            [
                any_eqn(x, float(op_val), float(op_val)),
                any_eqn(v, float(up_val), float(up_val)),
                scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=(),
                                combiner=None),
                eqn("eq", [out, lit(float(op_val + up_val))], pred),
                eqn("stelling_assert", [pred], ob),
            ],
            [ob],
        )

    # in range: the snapped exact integer result discharges an equality
    p = propagate(q(3, 4))
    assert p.obligations[0].status == "discharged"
    # wraparound reachable: declines, range quoted
    p2 = propagate(q(imax, imax))
    assert p2.obligations[0].status == "unknown"
    assert any("wraparound" in n and "declined" in n for n in p2.notes)


# --- stack -------------------------------------------------------------------


def test_stack_routes_axis0_and_axis1_exactly():
    a, b = var(0, aval((2,))), var(1, aval((2,)))
    s0, s1 = var(2, aval((2, 2))), var(3, aval((2, 2)))
    pred, ob = var(4, aval((2, 2), "bool")), var(5, aval((2, 2), "bool"))
    closed = close(
        [
            any_eqn(a, 1.0, 1.0),
            any_eqn(b, 3.0, 3.0),
            eqn("stack", [a, b], s0, (("axis", 0),)),
            eqn("stack", [a, b], s1, (("axis", 1),)),
            eqn("ge", [s0, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    env = interval_env(closed)
    # axis 0: [[a0, a1], [b0, b1]] = [1, 1, 3, 3]; axis 1: [[a0, b0],
    # [a1, b1]] = [1, 3, 1, 3] — pure routing, points stay points
    assert env[2].los == (1.0, 1.0, 3.0, 3.0)
    assert env[2].his == (1.0, 1.0, 3.0, 3.0)
    assert env[3].los == (1.0, 3.0, 1.0, 3.0)
    p = propagate(closed)
    assert p.obligations[0].status == "discharged"
    assert ("stack", "exact") in p.transfers_used


def test_stack_scalars_and_single_operand():
    a, b = var(0), var(1)
    s, s1 = var(2, aval((2,))), var(3, aval((1,)))
    pred, ob = var(4, aval((2,), "bool")), var(5, aval((2,), "bool"))
    closed = close(
        [
            any_eqn(a, 0.5, 0.5),
            any_eqn(b, 0.25, 0.25),
            eqn("stack", [a, b], s, (("axis", 0),)),
            eqn("stack", [a], s1, (("axis", 0),)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    env = interval_env(closed)
    assert env[2].los == (0.5, 0.25)
    assert env[3].los == (0.5,)
    assert propagate(closed).obligations[0].status == "discharged"


def test_stack_malformed_forms_decline_not_crash():
    a, b = var(0, aval((2,))), var(1, aval((3,)))
    s = var(2, aval((2, 2)))
    pred, ob = var(3, aval((2, 2), "bool")), var(4, aval((2, 2), "bool"))
    # mismatched shapes
    closed = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(b, 0.0, 1.0),
            eqn("stack", [a, b], s, (("axis", 0),)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    _declined(closed, "disagree")
    # out-of-bounds axis (from_dict-only: the traced param is normalized)
    a2, b2 = var(0, aval((2,))), var(1, aval((2,)))
    closed2 = close(
        [
            any_eqn(a2, 0.0, 1.0),
            any_eqn(b2, 0.0, 1.0),
            eqn("stack", [a2, b2], s, (("axis", 5),)),
            eqn("ge", [s, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    _declined(closed2, "out of bounds")


# --- ieee: the censused refusal and the structural passthrough ---------------


def test_ieee_scatter_add_censused_refusal():
    closed, _ = _accumulate_query((0.0, 0.0))
    p = propagate(closed, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any(
        "scatter-add has no ieee transfer" in n and "ACCUMULATE" in n
        for n in p.notes
    ), p.notes
    # the refusal is a censused coverage entry, not an omission
    assert any(
        name == "scatter-add" for name, _ in p.coverage.unknown_primitives
    )


def test_ieee_stack_passthrough_and_flag_riding():
    a, b = var(0), var(1)
    s = var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))
    closed = close(
        [
            any_eqn(a, 1.0, 1.0),
            any_eqn(b, 2.0, 2.0),
            eqn("stack", [a, b], s, (("axis", 0),)),
            eqn("ge", [s, lit(0.5)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    p = propagate(closed, semantics="ieee")
    assert p.obligations[0].status == "discharged"

    # the flag ride, at the censused entry itself: a maybe-NaN operand's
    # flag ORs onto every output of the passthrough (NaN moves like any
    # other routed value)
    t = IEEE_TRANSFERS["stack"][0]
    st_eqn = eqn("stack", [a, b], s, (("axis", 0),))
    boxes = [iv.point(1.0), iv.point(2.0)]
    outs, flags = t(st_eqn, {"axis": 0}, boxes, [False, True])
    assert flags == [True]
    outs2, flags2 = t(st_eqn, {"axis": 0}, boxes, [False, False])
    assert flags2 == [False]
    assert outs2[0].los == (1.0, 2.0)


# --- coverage accounting: the combiner sub-jaxpr stays in the denominator ----


def test_combiner_add_counts_known_never_vanishes():
    closed, _ = _accumulate_query((0.0, 0.0))
    p = propagate(closed)
    # 4 top-level eqns + the recorded combiner's inner add = 5, all known
    assert p.coverage.total == 5
    assert p.coverage.known == 5
    assert p.coverage.unreached == 0
    # the hand-built (no-combiner) form has one fewer equation, also 100%
    x, v, out = var(0, aval((2,))), var(1, aval((1,))), var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))
    closed2 = close(
        [
            any_eqn(x, 1.0, 1.0),
            any_eqn(v, 2.0, 2.0),
            scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=(),
                            combiner=None),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    p2 = propagate(closed2)
    assert p2.coverage.total == p2.coverage.known == 5


# --- emission, replay, and the witness validator -----------------------------


def _emission_query():
    """x (3,) and v (4,) declared; out = x.at[[0,2,0,0]].add(v); assert
    out <= 3.5 elementwise (violable: out[0] = x0+v0+v2+v3 reaches 4)."""
    x, v, out = var(0, aval((3,))), var(1, aval((4,))), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(v, 0.0, 1.0),
            scatter_add_eqn(x, int_idx_lit([0, 2, 0, 0], (4, 1)), v, out,
                            uwd=()),
            eqn("le", [out, lit(3.5)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    return closed


def test_scatter_add_emission_duplicates_are_separate_addends():
    closed = _emission_query()
    env = interval_env(closed)
    sl = slice_obligation(closed, 0, env)
    assert not isinstance(sl, DeclinedObligation), sl
    # out + operand + updates elements counted by the budget gate
    # (3 + 4 declared inputs) + (3 out + 3 operand-addends + 4
    # update-addends) + 3 comparison terms = 20
    assert sl.element_terms == 20
    script = emit(sl, "z3", 1000)
    # element 0: operand term + THREE separate duplicate addends
    assert "(+ x0_0 x1_0 x1_2 x1_3)" in script.text
    # element 2: operand term + one addend
    assert "(+ x0_2 x1_1)" in script.text
    # element 1 is untouched: NO defined term aliases it — its comparison
    # references the operand's input constant directly
    assert "(<= x0_1 " in script.text.replace("3.5", "(/ 7 2)")


def test_scatter_add_replay_agrees_with_hand_accumulate():
    closed = _emission_query()
    sl = slice_obligation(closed, 0, interval_env(closed))
    F = Fraction
    # x = [1, 1, 1], v = [1, 1, 1, 1]: out = [1+3, 1, 1+1] = [4, 1, 2];
    # 4 > 3.5 -> the universal claim is FALSE at this point
    vals = {
        "x0_0": F(1), "x0_1": F(1), "x0_2": F(1),
        "x1_0": F(1), "x1_1": F(1), "x1_2": F(1), "x1_3": F(1),
    }
    assert evaluate_predicate(sl, vals) is False
    # the single conjunctive witness validator accepts this in-box
    # violating point...
    assert witness_is_valid(sl, vals) is None
    # ...and a point where the accumulate stays under the bound holds
    vals_ok = dict(vals, x1_0=F(0), x1_2=F(0), x1_3=F(1, 2))
    # out[0] = 1 + 0 + 0 + 1/2 = 3/2 <= 7/2; out[2] = 2 <= 7/2
    assert evaluate_predicate(sl, vals_ok) is True
    assert witness_is_valid(sl, vals_ok) is not None
    # an out-of-box point is refused by the membership conjunct
    vals_oob = dict(vals, x0_0=F(2))
    assert "escapes the declared box" in witness_is_valid(sl, vals_oob)


# --- third-audit regressions (F1, F2, F3, F5, F6, F4d) -----------------------


def _oob_decline_query():
    """A scatter-add DECLINE (out-of-range index) carrying the recorded
    add combiner — the hand-IR case of the F1 accounting."""
    x, v, out = var(0, aval((3,))), var(1), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(v, 0.0, 1.0),
            scatter_add_eqn(x, int_idx_lit([3], (1, 1)), v, out, uwd=()),
            eqn("ge", [out, lit(-9.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )


def test_f1_decline_total_equals_static_measure():
    """F1(b): the live counter's denominator equals the outcome-blind
    static walk's on a declined scatter-add — the combiner's inner add
    counts unreached, never vanishes (before the fix: live total 5 vs
    static 6; 80% reported where the honest number is 67%)."""
    from stelling.coverage import measure

    closed = _oob_decline_query()
    p = propagate(closed)
    assert p.obligations[0].status == "unknown"
    static = measure(closed, known=set(TRANSFERS))
    assert p.coverage.total == static.total == 6
    assert p.coverage.unreached == 1  # the combiner's inner add
    assert p.coverage.known == 4
    # the same accounting an UNREGISTERED sibling gets: the denominator is
    # a function of the program, never of the outcome


def test_f1_real_vs_ieee_totals_equal_on_one_program():
    """F1(a), hand-IR face: the same scatter-add-carrying program reports
    the same equation total under both semantics dials (before the fix:
    real 6 vs ieee 5 — one equation of the PROGRAM disappeared under the
    dial, because ieee's whole-primitive refusal dropped the combiner)."""
    x, v, out = var(0, aval((2,))), var(1, aval((1,))), var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))
    closed = close(
        [
            any_eqn(x, 1.0, 1.0),
            any_eqn(v, 2.0, 2.0),
            scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=()),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    p_real = propagate(closed, semantics="real")
    p_ieee = propagate(closed, semantics="ieee")
    assert p_real.coverage.total == p_ieee.coverage.total == 6
    assert p_real.obligations[0].status == "discharged"
    assert p_ieee.obligations[0].status == "unknown"  # the censused refusal
    assert p_ieee.coverage.unreached == 1


def _unique_query(unique, idx_vals):
    x, v, out = var(0, aval((3,))), var(1, aval((len(idx_vals),))), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    sc = scatter_add_eqn(
        x, int_idx_lit(idx_vals, (len(idx_vals), 1)), v, out, uwd=()
    )
    sc = ir.JaxprEqn(
        primitive=sc.primitive, invars=sc.invars, outvars=sc.outvars,
        params=sc.params + (("unique_indices", unique),),
    )
    return close(
        [
            any_eqn(x, 1.0, 1.0),
            any_eqn(v, 2.0, 2.0),
            sc,
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )


def test_f2_unique_indices_promise_violated_declines():
    """F2: unique_indices=True with measured duplicates is a violated
    promise — implementation-defined, declined loudly at the transfer AND
    at the emission (never modelled as accumulate)."""
    closed = _unique_query(True, [0, 2, 0])
    p = propagate(closed)
    assert p.obligations[0].status == "unknown"
    assert any(
        "unique_indices=True" in n
        and "implementation-defined" in n
        and "declined" in n
        for n in p.notes
    ), p.notes
    d = slice_obligation(closed, 0, interval_env(closed))
    assert isinstance(d, DeclinedObligation)
    assert "unique_indices=True" in d.reason
    assert "implementation-defined" in d.reason


def test_f2_unique_indices_promise_kept_proceeds():
    """F2: unique_indices=True with actually-unique indices proceeds — the
    promise holds, every backend agrees, and the obligation discharges."""
    p = propagate(_unique_query(True, [0, 2]))
    assert p.obligations[0].status == "discharged"


def test_f2_default_with_duplicates_accumulates():
    """F2: unique_indices=False (the default) with duplicates keeps the
    shipped accumulate behavior, pinned by value."""
    closed = _unique_query(False, [0, 2, 0])
    env = interval_env(closed)
    box = env[2]
    # element 0: 1 + 2 + 2 = 5 accumulated
    assert box.los[0] <= 5.0 <= box.his[0]
    assert not box.los[0] <= 3.0 <= box.his[0]  # last-wins/collapse excluded
    assert propagate(closed).obligations[0].status == "discharged"


def test_f3_classification_census_probe_or_exempt():
    """F3: an unprobed, unexempted registered transfer refuses the
    integer-classification census at import — doctored in-process through
    the census callables' explicit arguments (never a reload hack). Both
    layers; stale exemptions refuse too."""
    from stelling.obligation import (
        _INT_SAFE_EMITTED_REASONS,
        _assert_emission_classification_censused,
    )
    from stelling.propagate import (
        _INT_NON_COMPUTING_EXEMPT,
        _assert_integer_classification_censused,
    )

    # the live registries pass (this is what import already ran)
    _assert_integer_classification_censused()
    _assert_emission_classification_censused()
    # a future two-edit misfiling: registered + non-computing, no reason
    with pytest.raises(AssertionError, match="phantom_cumsum"):
        _assert_integer_classification_censused(
            registered=set(TRANSFERS) | {"phantom_cumsum"}
        )
    # an empty reason is not a claim
    with pytest.raises(AssertionError, match="phantom_cumsum"):
        _assert_integer_classification_censused(
            registered=set(TRANSFERS) | {"phantom_cumsum"},
            exemptions={**_INT_NON_COMPUTING_EXEMPT, "phantom_cumsum": "  "},
        )
    # a stale exemption is a soundness claim about nothing
    with pytest.raises(AssertionError, match="ghost"):
        _assert_integer_classification_censused(
            exemptions={**_INT_NON_COMPUTING_EXEMPT, "ghost": "routing"},
        )
    # the emission layer, same pattern
    with pytest.raises(AssertionError, match="phantom_cumsum"):
        _assert_emission_classification_censused(
            supported=_SUPPORTED | {"phantom_cumsum"}
        )
    with pytest.raises(AssertionError, match="ghost"):
        _assert_emission_classification_censused(
            reasons={**_INT_SAFE_EMITTED_REASONS, "ghost": "routing"},
        )


def test_f5b_int64_index_narrowing_boundary_both_sides():
    """F5b: statically-in-range constant int64->int32 narrowing passes as
    an exact identity (the census contact is index data: the
    default-dtype at[].add sugar under x64); the boundary is exact on
    both sides — int32max passes, int32max+1 declines."""
    imax = float(2**31 - 1)

    def q(val):
        a = var(0, aval((), "int64"))
        b = var(1, aval((), "int32"))
        pred, ob = var(2, BOOL), var(3, BOOL)
        return close(
            [
                any_eqn(a, val, val),
                eqn(
                    "convert_element_type", [a], b,
                    (("new_dtype", "int32"), ("weak_type", False)),
                ),
                eqn("eq", [b, lit(val)], pred),
                eqn("stelling_assert", [pred], ob),
            ],
            [ob],
        )

    p = propagate(q(imax))
    assert p.obligations[0].status == "discharged"
    p2 = propagate(q(float(2**31)))
    assert p2.obligations[0].status == "unknown"
    assert any("convert_element_type" in n and "no sound rule" in n for n in p2.notes)
    # under ieee the same narrowing passes identically (exact integer
    # identity: no float semantics, no flush hazard)
    p3 = propagate(q(imax), semantics="ieee")
    assert p3.obligations[0].status == "discharged"


def test_f6_int64_magnitude_bracket_pinned_as_intended():
    """F6: 'in-range exact snapped results' is a magnitude-conditional
    claim — pinned as INTENDED behavior, not accident. At 2**62 the
    outward ulps span many integers: the in-range accumulate keeps the
    snapped bracket [2**62 - 512, 2**62 + 1024] and stays undecided;
    2**62 + 2**62 escapes the range and declines with it quoted."""
    big = 2**62

    def q(op_val, up_val):
        x = var(0, aval((1,), "int64"))
        v = var(1, aval((1,), "int64"))
        out = var(2, aval((1,), "int64"))
        pred, ob = var(3, aval((1,), "bool")), var(4, aval((1,), "bool"))
        return close(
            [
                any_eqn(x, float(op_val), float(op_val)),
                any_eqn(v, float(up_val), float(up_val)),
                scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=(),
                                combiner=None),
                eqn("eq", [out, lit(float(op_val))], pred),
                eqn("stelling_assert", [pred], ob),
            ],
            [ob],
        )

    closed = q(big, 0)
    p = propagate(closed)
    assert p.obligations[0].status == "unknown"  # bracket too wide for eq
    box = interval_env(closed)[2]
    assert (box.los[0], box.his[0]) == (float(big - 512), float(big + 1024))
    p2 = propagate(q(big, big))
    assert p2.obligations[0].status == "unknown"
    assert any("outside the representable range" in n for n in p2.notes)


def test_f4d_empty_updates_identity_exact():
    """F4(d): the N=0 form is the exact identity on the operand — no
    contribution, no bump. A direct unit test on the transfer path,
    hand-IR deliberately: MEASURED on jax 0.11.0, tracing
    ``x.at[jnp.zeros((0,), int32)].add(jnp.zeros((0,)))`` produces NO
    scatter-add equation at all (jax short-circuits the empty update),
    so this form is reachable only through hand-built/from_dict IR and a
    gauge battery case could never exercise it."""
    x, v0, out = var(0, aval((3,))), var(1, aval((0,))), var(2, aval((3,)))
    pred, ob = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 2.0, 2.0),
            any_eqn(v0, 0.0, 1.0),
            scatter_add_eqn(x, int_idx_lit([], (0, 1)), v0, out, uwd=()),
            eqn("eq", [out, lit(2.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    box = interval_env(closed)[2]
    assert box.los == (2.0, 2.0, 2.0) and box.his == (2.0, 2.0, 2.0)
    assert propagate(closed).obligations[0].status == "discharged"


def test_scatter_add_slice_declines_dynamic_indices_quoted():
    x, i, v, out = (
        var(0, aval((3,))), var(1, aval((1,), "int32")), var(2),
        var(3, aval((3,))),
    )
    pred, ob = var(4, aval((3,), "bool")), var(5, aval((3,), "bool"))
    closed = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(i, 0.0, 2.0),
            any_eqn(v, 0.0, 1.0),
            scatter_add_eqn(x, i, v, out, uwd=()),
            eqn("ge", [out, lit(0.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    d = slice_obligation(closed, 0, interval_env(closed))
    assert isinstance(d, DeclinedObligation)
    # F5a wording: the derivation-form phrasing, accurate for BOTH the
    # traced/dynamic case and a constant column routed through
    # non-structural ops (the negative-index normalization arithmetic)
    assert "not statically derivable through the supported derivation forms" in d.reason


def test_scatter_add_emission_declines_integer_dtype():
    x = var(0, aval((1,), "int32"))
    v = var(1, aval((1,), "int32"))
    out = var(2, aval((1,), "int32"))
    pred, ob = var(3, aval((1,), "bool")), var(4, aval((1,), "bool"))
    closed = close(
        [
            any_eqn(x, 0.0, 5.0),
            any_eqn(v, 0.0, 5.0),
            scatter_add_eqn(x, int_idx_lit([0], (1, 1)), v, out, uwd=(),
                            combiner=None),
            eqn("eq", [out, lit(True, aval((), "bool"))], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    d = slice_obligation(closed, 0, interval_env(closed))
    assert isinstance(d, DeclinedObligation)


def test_stack_emission_aliases_and_replays():
    a, b = var(0), var(1)
    s = var(2, aval((2,)))
    pred, ob = var(3, aval((2,), "bool")), var(4, aval((2,), "bool"))
    closed = close(
        [
            any_eqn(a, 0.0, 1.0),
            any_eqn(b, 0.0, 1.0),
            eqn("stack", [a, b], s, (("axis", 0),)),
            eqn("ge", [s, lit(-0.5)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )
    sl = slice_obligation(closed, 0, interval_env(closed))
    assert not isinstance(sl, DeclinedObligation), sl
    script = emit(sl, "cvc5", 1000)
    # stack emits NO terms: the comparisons reference x0 and x1 directly
    assert "(>= x0 " in script.text and "(>= x1 " in script.text
    assert "t2" not in script.text
    F = Fraction
    assert evaluate_predicate(sl, {"x0": F(0), "x1": F(1)}) is True
