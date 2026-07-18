# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The pytree-probe registration round, at the propagation layer — no jax.

Hand-built IR queries (the :mod:`stelling.ir` conventions of
``test_propagate.py``) exercising the nine new registry rows plus the
extended select_n / broadcasting forms: correctness in both definite
directions, tier and assumption stamping, and — per the guard rule — one
test per decline path proving the analysis degrades to a noted ⊤ instead
of crashing.
"""

from __future__ import annotations

import pytest

from stelling import ir
from stelling.propagate import TRANSFERS, propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")


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


def test_registration_round_rows_and_tiers():
    want = {
        "abs": "exact",
        "eq": "exact",
        "ne": "exact",
        "and": "exact",
        "or": "exact",
        "stop_gradient": "exact",
        "reshape": "exact",
        "pow": "sound-libm",
        "reduce_or": "exact",
    }
    for prim, tier in want.items():
        assert prim in TRANSFERS, prim
        assert TRANSFERS[prim][1] == tier, prim


# --- abs ---------------------------------------------------------------------


def _abs_query(bound, cmp):
    x, ax, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, -3.0, 2.0),
            eqn("abs", [x], ax),
            eqn(cmp, [ax, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_abs_discharges_and_refutes():
    p = propagate(_abs_query(3.0, "le"))  # |[-3,2]| = [0,3] <= 3
    assert p.obligations[0].status == "discharged"
    assert ("abs", "exact") in p.transfers_used
    # definite-FALSE direction: |x| < -1 holds nowhere
    p2 = propagate(_abs_query(-1.0, "lt"))
    assert p2.obligations[0].status == "violated-over-set"


# --- eq / ne -----------------------------------------------------------------


def _cmp_query(prim, xb, yb):
    x, y, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, *xb),
            any_eqn(y, *yb),
            eqn(prim, [x, y], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_eq_transfer_three_directions():
    # same single point: definitely true
    p = propagate(_cmp_query("eq", (2.0, 2.0), (2.0, 2.0)))
    assert p.obligations[0].status == "discharged"
    assert ("eq", "exact") in p.transfers_used
    # disjoint: definitely FALSE
    assert (
        propagate(_cmp_query("eq", (1.0, 2.0), (3.0, 4.0))).obligations[0].status
        == "violated-over-set"
    )
    # identical non-point intervals: unknown, never guessed
    assert (
        propagate(_cmp_query("eq", (1.0, 2.0), (1.0, 2.0))).obligations[0].status
        == "unknown"
    )


def test_ne_transfer_three_directions():
    assert (
        propagate(_cmp_query("ne", (1.0, 2.0), (3.0, 4.0))).obligations[0].status
        == "discharged"
    )
    # same single point: ne is definitely FALSE
    assert (
        propagate(_cmp_query("ne", (2.0, 2.0), (2.0, 2.0))).obligations[0].status
        == "violated-over-set"
    )
    assert (
        propagate(_cmp_query("ne", (1.0, 2.0), (1.0, 2.0))).obligations[0].status
        == "unknown"
    )


# --- and / or ----------------------------------------------------------------


def test_and_or_transfers_on_bools():
    a, b, c, out = var(0, BOOL), var(1, BOOL), var(2, BOOL), var(3, BOOL)
    q_or = close(
        [
            any_eqn(a, 1.0, 1.0),  # definitely true
            any_eqn(b, 0.0, 1.0),  # unknown
            eqn("or", [a, b], c),  # T or U = T
            eqn("stelling_assert", [c], out),
        ],
        [out],
    )
    p = propagate(q_or)
    assert p.obligations[0].status == "discharged"
    assert ("or", "exact") in p.transfers_used

    q_and = close(
        [
            any_eqn(a, 0.0, 0.0),  # definitely false
            any_eqn(b, 0.0, 1.0),
            eqn("and", [a, b], c),  # F and U = F: definitely FALSE
            eqn("stelling_assert", [c], out),
        ],
        [out],
    )
    p2 = propagate(q_and)
    assert p2.obligations[0].status == "violated-over-set"
    assert ("and", "exact") in p2.transfers_used


@pytest.mark.parametrize("prim", ["and", "or"])
def test_bitwise_integer_logic_declines_not_crashes(prim):
    x, y, z, pred, out = (
        var(0, I32),
        var(1, I32),
        var(2, I32),
        var(3, BOOL),
        var(4, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 7.0),
            any_eqn(y, 0.0, 7.0),
            eqn(prim, [x, y], z),  # bitwise integer form: no interval rule
            eqn("le", [z, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any(prim in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1
    assert prim not in dict(p.transfers_used)  # no tier claimed on a decline


# --- stop_gradient -----------------------------------------------------------


def test_stop_gradient_is_identity():
    x, sg, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("stop_gradient", [x], sg),
            eqn("ge", [sg, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert ("stop_gradient", "exact") in p.transfers_used


# --- reshape -----------------------------------------------------------------


def _reshape_query(dimensions):
    a23, a32, b32 = aval((2, 3)), aval((3, 2)), aval((3, 2), "bool")
    x, r, pred, out = var(0, a23), var(1, a32), var(2, b32), var(3, b32)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn(
                "reshape",
                [x],
                r,
                params=(
                    ("new_sizes", (3, 2)),
                    ("dimensions", dimensions),
                    ("sharding", None),
                ),
            ),
            eqn("le", [r, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_reshape_flat_identity_discharges():
    p = propagate(_reshape_query(None))
    assert p.obligations[0].status == "discharged"
    assert ("reshape", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_reshape_with_dimensions_declines_not_crashes():
    p = propagate(_reshape_query((1, 0)))  # permuting form: no rule
    assert p.obligations[0].status == "unknown"
    assert any("reshape" in n and "no sound rule" in n for n in p.notes)
    assert p.coverage.unknown == 1


# --- pow ---------------------------------------------------------------------


def _pow_query(xb, yb, bound, cmp):
    x, y, z, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(x, *xb),
            any_eqn(y, *yb),
            eqn("pow", [x, y], z),
            eqn(cmp, [z, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_pow_discharges_with_tier_and_assumption():
    p = propagate(_pow_query((1.0, 2.0), (2.0, 3.0), 8.001, "le"))  # max 2**3
    assert p.obligations[0].status == "discharged"
    assert ("pow", "sound-libm") in p.transfers_used
    assert any("pow" in a and "libm" in a for a in p.assumptions)


def test_pow_definite_false_direction():
    # base >= 1, exponent >= 1: x**y >= 1, so "< 0.5" is false everywhere
    p = propagate(_pow_query((1.0, 2.0), (1.0, 2.0), 0.5, "lt"))
    assert p.obligations[0].status == "violated-over-set"


def test_pow_literal_exponent():
    x, z, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 2.0, 3.0),
            eqn("pow", [x, lit(2.0)], z),
            eqn("le", [z, lit(9.001)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert propagate(q).obligations[0].status == "discharged"


def test_pow_overflowing_range_still_sound():
    # 10**[1, 400] overflows the double range at the top: the bracket
    # saturates to [_, inf] and "pow > 0" still discharges (maxfloat > 0)
    p = propagate(_pow_query((10.0, 10.0), (1.0, 400.0), 0.0, "gt"))
    assert p.obligations[0].status == "discharged"


@pytest.mark.parametrize("base", [(-1.0, 2.0), (0.0, 2.0), (-3.0, -1.0)])
def test_pow_nonpositive_base_declines_not_crashes(base):
    p = propagate(_pow_query(base, (2.0, 2.0), 100.0, "le"))  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("pow" in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1
    assert "pow" not in dict(p.transfers_used)
    assert not any("pow" in a for a in p.assumptions)  # no assumption claimed


# --- select_n: scalar selector ----------------------------------------------


def test_select_n_scalar_selector_now_registered():
    # the exact form the audit once used as its unhandled example:
    # scalar which, (1,)-shaped cases — now a registered form
    w = var(0, BOOL)
    arr, barr = aval((1,)), aval((1,), "bool")
    c0, c1, y, pred, out = (
        var(1, arr),
        var(2, arr),
        var(3, arr),
        var(4, barr),
        var(5, barr),
    )
    q = close(
        [
            any_eqn(w, 0.0, 1.0),
            any_eqn(c0, 1.0, 2.0),
            any_eqn(c1, 3.0, 4.0),
            eqn("select_n", [w, c0, c1], y),
            eqn("le", [y, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"  # join [1,4] <= 10
    assert p.coverage.unknown == 0


def test_select_n_mismatched_cases_decline_not_crash():
    w = var(0, BOOL)
    a1, a2, b1 = aval((1,)), aval((2,)), aval((1,), "bool")
    c0, c1, y, pred, out = (
        var(1, a1),
        var(2, a2),
        var(3, a1),
        var(4, b1),
        var(5, b1),
    )
    q = close(
        [
            any_eqn(w, 0.0, 1.0),
            any_eqn(c0, 1.0, 2.0),
            any_eqn(c1, 3.0, 4.0),
            eqn("select_n", [w, c0, c1], y),  # cases disagree on shape
            eqn("le", [y, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("select_n" in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1


# --- reduce_or ---------------------------------------------------------------


def _reduce_or_query(lo, hi, in_shape=(2, 3), axes=(1,), out_shape=(2,)):
    bin_, bout = aval(in_shape, "bool"), aval(out_shape, "bool")
    x, r, out = var(0, bin_), var(1, bout), var(2, bout)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("reduce_or", [x], r, params=(("axes", axes),)),
            eqn("stelling_assert", [r], out),
        ],
        [out],
    )


def test_reduce_or_three_directions():
    # all definitely true -> OR true -> discharged
    assert propagate(_reduce_or_query(1.0, 1.0)).obligations[0].status == "discharged"
    # all definitely false -> OR false -> the assert is definitely FALSE
    assert (
        propagate(_reduce_or_query(0.0, 0.0)).obligations[0].status
        == "violated-over-set"
    )
    # unknown elements -> unknown
    p = propagate(_reduce_or_query(0.0, 1.0))
    assert p.obligations[0].status == "unknown"
    assert ("reduce_or", "exact") in p.transfers_used


def test_reduce_or_empty_range_axis_is_definite_false():
    # OR over an empty reduction range is false — and no crash on size 0
    p = propagate(_reduce_or_query(0.0, 1.0, in_shape=(0,), axes=(0,), out_shape=()))
    assert p.obligations[0].status == "violated-over-set"


def test_reduce_or_non_bool_input_declines_not_crashes():
    iin, bout = aval((2,), "int32"), aval((), "bool")
    x, r, out = var(0, iin), var(1, bout), var(2, bout)
    q = close(
        [
            any_eqn(x, 0.0, 5.0),
            eqn("reduce_or", [x], r, params=(("axes", (0,)),)),
            eqn("stelling_assert", [r], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("reduce_or" in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1


# --- rank broadcasting for the elementwise binaries --------------------------


def test_rank_broadcast_add_discharges():
    a21, a13, a23, b23 = (
        aval((2, 1)),
        aval((1, 3)),
        aval((2, 3)),
        aval((2, 3), "bool"),
    )
    x, y, s, pred, out = var(0, a21), var(1, a13), var(2, a23), var(3, b23), var(4, b23)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 2.0),
            eqn("add", [x, y], s),  # (2,1) + (1,3) -> (2,3)
            eqn("le", [s, lit(3.001)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert p.coverage.unknown == 0


def test_rank_broadcast_size1_against_bigger():
    # the h_hard shape pair: (2,) vs (1,) — size-1 axis replicates
    a2, a1, b2 = aval((2,)), aval((1,)), aval((2,), "bool")
    x, y, s, pred, out = var(0, a2), var(1, a1), var(2, a2), var(3, b2), var(4, b2)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("mul", [x, y], s),
            eqn("le", [s, lit(1.001)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert p.coverage.unknown == 0


# --- unsigned literal decoding (regression) ----------------------------------


def test_uint64_mask_literal_read_declines_not_crashes():
    # Registration of `and`/`or`/`eq`/`ne` made the propagator READ the uint
    # mask literals of RNG plumbing that previously hid behind unknown
    # primitives; without unsigned decoders the read raised and killed the
    # whole analysis on a legal trace (found on the probe's h_hard). The
    # decline must be the bool-dtype guard's, after a successful read.
    import struct

    u64 = aval((), "uint64")
    mask = ir.Literal(
        val=ir.Array(dtype="<u8", shape=(), data=struct.pack("<Q", 2**63)),
        aval=u64,
    )
    x, z, pred, out = var(0, u64), var(1, u64), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 100.0),
            eqn("and", [x, mask], z),
            eqn("le", [z, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("and" in n and "declined" in n for n in p.notes)


def test_uint32_literal_decodes_to_its_value():
    import struct

    u32 = aval((), "uint32")
    five = ir.Literal(
        val=ir.Array(dtype="<u4", shape=(), data=struct.pack("<I", 5)),
        aval=u32,
    )
    x, pred, out = var(0, u32), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 5.0, 5.0),
            eqn("eq", [x, five], pred),  # same single point: definitely true
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert propagate(q).obligations[0].status == "discharged"


def test_incompatible_shapes_decline_not_crash():
    a2, a3, b_ = aval((2,)), aval((3,)), aval((2,), "bool")
    x, y, s, pred, out = var(0, a2), var(1, a3), var(2, a2), var(3, b_), var(4, b_)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("add", [x, y], s),  # (2,) vs (3,): not broadcastable
            eqn("le", [s, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("add" in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1
