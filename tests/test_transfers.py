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
    assert p.coverage.unknown == 1
    # The note used to read "no sound rule for params {...}". Strengthened, not
    # relaxed: the decline must name the PERMUTATION it declines and say what
    # the row does model, so this fails if the reason regresses to the generic
    # form or describes the wrong form.
    rn = [n for n in p.notes if "reshape" in n]
    assert rn, p.notes
    assert any("(1, 0)" in n and "PERMUTES" in n and "C-order" in n for n in rn), rn


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


# --- scatter: static-index x.at[k].set(v) (maddening heat-node round) --------
#
# The one allowed-by-census structural addition from the maddening HeatNode
# trace (the Dirichlet boundary writes). Exact form only; everything else
# declines to a noted ⊤.


def _scatter_set_params(**dim_overrides):
    fields = {
        "update_window_dims": (),
        "inserted_window_dims": (0,),
        "scatter_dims_to_operand_dims": (0,),
        "operand_batching_dims": (),
        "scatter_indices_batching_dims": (),
    }
    fields.update(dim_overrides)
    return (
        (
            "dimension_numbers",
            ir.NamedTupleParam(
                cls="ScatterDimensionNumbers", fields=tuple(fields.items())
            ),
        ),
        ("indices_are_sorted", True),
        ("mode", ir.EnumParam(cls="GatherScatterMode", member="FILL_OR_DROP")),
        ("unique_indices", True),
        ("update_consts", ()),
        ("update_jaxpr", None),
    )


def _scatter_query(idx_bounds, threshold, dim_overrides=None):
    """out = ([0,1]^3).at[idx].set(5.0); assert out <= threshold."""
    f3, i1, b3 = aval((3,)), aval((1,), "int32"), aval((3,), "bool")
    x, idx, y = var(0, f3), var(1, i1), var(2, f3)
    pred, out = var(3, b3), var(4, b3)
    idx_eqn = ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(idx,),
        params=(
            ("shape", (1,)),
            ("dtype", "int32"),
            ("lo", idx_bounds[0]),
            ("hi", idx_bounds[1]),
        ),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            idx_eqn,
            eqn(
                "scatter",
                [x, idx, lit(5.0)],
                y,
                _scatter_set_params(**(dim_overrides or {})),
            ),
            eqn("le", [y, lit(threshold)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_scatter_static_set_discharges_with_tier():
    # written element becomes [5,5], untouched keep [0,1]: all <= 5
    p = propagate(_scatter_query((1.0, 1.0), 5.0))
    assert p.obligations[0].status == "discharged"
    assert ("scatter", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_scatter_static_set_definite_false_direction():
    # the written element [5,5] <= 2 is definitely false; the untouched
    # elements [0,1] <= 2 are definitely true — a sound 1/3 refutation,
    # which also proves the write landed on exactly the indexed element
    p = propagate(_scatter_query((1.0, 1.0), 2.0))
    assert p.obligations[0].status == "violated-over-set"
    assert "1/3" in p.obligations[0].detail


def _scatter_declined(q, *frags):
    """The decline pinned WITH its accounting: a raised decline must count
    exactly as the returned None it replaced (status unknown, one ⊤,
    the primitive named in unknown_primitives, no tier recorded)."""
    p = propagate(q)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("scatter", 1),)
    assert "scatter" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'scatter' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_scatter_dynamic_index_declines_not_crashes():
    # a non-point index interval has no exact rule: noted ⊤, never a
    # guess. The note used to read "no sound rule for params {...}";
    # strengthened, not relaxed — it must print the span the right way
    # round.
    _scatter_declined(
        _scatter_query((0.0, 1.0), 5.0),
        "index spans [0.0, 1.0]",
        "not a single point",
    )


def test_scatter_out_of_range_index_declines_not_crashes():
    # FILL_OR_DROP drops, CLIP clamps — the wedge bug class; refuse to
    # guess. The failing comparison is printed with the true bound, and
    # the mode behaviours the note states are measured in
    # test_transfers_jax.py::
    # test_scatter_out_of_range_mode_behaviours_as_the_decline_states.
    _scatter_declined(
        _scatter_query((7.0, 7.0), 5.0),
        "index 7 is out of range",
        "0 <= 7 < 3 fails",
        "mode-dependent",
        "never guessed",
    )


def test_scatter_windowed_form_declines_not_crashes():
    # anything but the canonical single-element dimension numbers declines
    # via the shared oracle's general form failure: the note must print
    # the observed configuration AND the covered core (derived from
    # _SCATTER_SET_CORE, so the sentence cannot drift from the check)
    _scatter_declined(
        _scatter_query((1.0, 1.0), 5.0, {"update_window_dims": (0,)}),
        "operand (3,), indices (1,), updates ()",
        "update_window_dims=(0,)",  # the observed side
        "outside the covered static-index set row form",
        "update_window_dims=(), inserted_window_dims=(0,), "
        "scatter_dims_to_operand_dims=(0,)",  # the covered core
    )


def test_scatter_arity_decline_names_the_operand_count():
    # hand-IR-only: a traced scatter binds operand + indices + updates
    f3, i1 = aval((3,)), aval((1,), "int32")
    x, idx, y = var(0, f3), var(1, i1), var(2, f3)
    pred, out = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(idx, 1.0, 1.0),
            eqn("scatter", [x, idx], y, _scatter_set_params()),
            eqn("le", [y, lit(5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _scatter_declined(q, "binds 2 operand(s)")


def test_scatter_leftover_update_consts_decline_hand_ir():
    # hand-IR-only: combiner state without a combiner. A traced set form
    # carries update_consts=() (measured in test_transfers_jax.py::
    # test_scatter_apply_traces_as_the_combiner_decline_states).
    params = tuple(
        (k, ((2.0,) if k == "update_consts" else v))
        for k, v in _scatter_set_params()
    )
    f3, i1 = aval((3,)), aval((1,), "int32")
    x, idx, y = var(0, f3), var(1, i1), var(2, f3)
    pred, out = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(idx, 1.0, 1.0),
            eqn("scatter", [x, idx, lit(5.0)], y, params),
            eqn("le", [y, lit(5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _scatter_declined(
        q,
        "non-empty update_consts (2.0,)",
        "combiner state without a combiner",
    )


def test_scatter_missing_dimension_numbers_decline_hand_ir():
    # hand-IR-only: the traced form always records dimension_numbers
    params = tuple(
        (k, v) for k, v in _scatter_set_params() if k != "dimension_numbers"
    )
    f3, i1 = aval((3,)), aval((1,), "int32")
    x, idx, y = var(0, f3), var(1, i1), var(2, f3)
    pred, out = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(idx, 1.0, 1.0),
            eqn("scatter", [x, idx, lit(5.0)], y, params),
            eqn("le", [y, lit(5.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _scatter_declined(q, "no readable dimension_numbers", "got None")


def test_scatter_non_integral_point_index_declines_through_the_walk():
    """WALK-DRIVEN, via the int-dtype declaration route, labeled: a
    stelling_any with dtype 'int32' and lo = hi = 0.5 (or inf)
    PROPAGATES UNREFUSED — the stelling_any transfer has no
    dtype-storability guard, so the declared box rides the walk on an
    int32-aval var and the scatter row is the first place the fraction
    meets a rule that needs an integer (audit repair F3a: an earlier
    draft claimed this path unreachable through any walk and pinned it
    by direct call — measured false by exactly this route). A traced
    index array cannot carry these values, and the lying-literal and
    dtype-contradicting-declaration routes ARE refused at IR
    construction; this consistent-but-fractional declaration is the one
    that gets through. The declaration transfer's missing dtype guard is
    a pre-existing surface, observed and out of scope here."""

    def q(val):
        f3, i1 = aval((3,)), aval((1,), "int32")
        x, idx, y = var(0, f3), var(1, i1), var(2, f3)
        pred, out = var(3, aval((3,), "bool")), var(4, aval((3,), "bool"))
        return close(
            [
                any_eqn(x, 0.0, 1.0),
                any_eqn(idx, val, val),
                eqn("scatter", [x, idx, lit(5.0)], y, _scatter_set_params()),
                eqn("le", [y, lit(9.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    _scatter_declined(q(0.5), "the point 0.5", "not a finite integer")
    _scatter_declined(q(float("inf")), "the point inf")


# --- gather / transpose: static structural forms (MIME fvm round) ------------
#
# The two allowed-by-census structural additions from the MIME fvm
# laplacian trace: gather in its static-index leading-axis row form
# (phi[mesh.owner] with const topology indices) and transpose (reached
# inside the transparent jnp.linalg.inv jit). Exact forms only;
# everything else declines to a noted ⊤.

import struct  # noqa: E402  (zero-dep literal-array payloads)


def _arr_lit(values, shape):
    n = 1
    for d in shape:
        n *= d
    return lit(
        ir.Array(dtype="<f8", shape=shape, data=struct.pack(f"<{n}d", *values)),
        aval(shape),
    )


def _gather_row_params(operand_rank, trailing, **dim_overrides):
    fields = {
        "offset_dims": tuple(range(1, operand_rank)),
        "collapsed_slice_dims": (0,),
        "start_index_map": (0,),
        "operand_batching_dims": (),
        "start_indices_batching_dims": (),
    }
    fields.update(dim_overrides)
    return (
        (
            "dimension_numbers",
            ir.NamedTupleParam(
                cls="GatherDimensionNumbers", fields=tuple(fields.items())
            ),
        ),
        ("fill_value", None),
        ("indices_are_sorted", False),
        ("mode", ir.EnumParam(cls="GatherScatterMode", member="PROMISE_IN_BOUNDS")),
        ("slice_sizes", (1,) + trailing),
        ("unique_indices", False),
    )


def _gather_query(idx_bounds, threshold, dim_overrides=None):
    """y = [1.0, 5.0, 2.0][idx]; assert y <= threshold."""
    idx = var(0, aval((1, 1), "int32"))
    y = var(1, aval((1,)))
    pred, out = var(2, aval((1,), "bool")), var(3, aval((1,), "bool"))
    return close(
        [
            any_eqn(idx, *idx_bounds),
            eqn(
                "gather",
                [_arr_lit([1.0, 5.0, 2.0], (3,)), idx],
                y,
                _gather_row_params(1, (), **(dim_overrides or {})),
            ),
            eqn("le", [y, lit(threshold)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_gather_static_row_discharges_with_tier():
    # idx = 1 selects the 5.0 element exactly: [5,5] <= 6
    p = propagate(_gather_query((1.0, 1.0), 6.0))
    assert p.obligations[0].status == "discharged"
    assert ("gather", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_gather_static_row_definite_false_direction():
    # the selected element [5,5] <= 2 is definitely false; rows 0/2 would
    # pass — the 1/1 refutation proves the take landed on exactly row 1
    p = propagate(_gather_query((1.0, 1.0), 2.0))
    assert p.obligations[0].status == "violated-over-set"
    assert "1/1" in p.obligations[0].detail


def test_gather_rank2_row_form_lands_whole_row():
    # grad[mesh.owner] shape: rank-2 operand, full trailing block per row
    idx = var(0, aval((1, 1), "int32"))
    y = var(1, aval((1, 2)))
    pred, out = var(2, aval((1, 2), "bool")), var(3, aval((1, 2), "bool"))
    q = close(
        [
            any_eqn(idx, 1.0, 1.0),
            eqn(
                "gather",
                [_arr_lit([1.0, 2.0, 9.0, 9.0], (2, 2)), idx],
                y,
                _gather_row_params(2, (2,)),
            ),
            eqn("le", [y, lit(3.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "violated-over-set"
    assert "2/2" in p.obligations[0].detail  # both elements of row 1 are 9.0


def _gather_declined(q, *frags):
    """The decline pinned WITH its accounting: a raised decline must count
    exactly as the returned None it replaced (status unknown, one ⊤,
    the primitive named in unknown_primitives, no tier recorded)."""
    p = propagate(q)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("gather", 1),)
    assert "gather" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'gather' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_gather_dynamic_in_range_index_takes_the_hull():
    # CHANGED EXPECTATION, index-bounds round. This declined before: a
    # non-point index had "no exact rule". It has one — the hull over the
    # rows the declared index can reach — and the row data is [1, 5, 2], so
    # an index anywhere in [0, 2] reaches [1.0, 5.0].
    #
    # The hull is what makes this sound and the THRESHOLD is what proves it
    # is not merely ⊤: 6.0 discharges, 4.0 must stay undecided (5.0 is
    # reachable) and 0.5 must refute (1.0 is reachable). A rule returning ⊤
    # fails the first; one returning only the first reachable row's value
    # wrongly discharges the second.
    p = propagate(_gather_query((0.0, 2.0), 6.0))
    assert p.obligations[0].status == "discharged"
    assert dict(p.transfers_used)["gather"] == "exact"
    assert propagate(_gather_query((0.0, 2.0), 4.0)).obligations[0].status == (
        "unknown"
    )
    assert propagate(_gather_query((0.0, 2.0), 0.5)).obligations[0].status == (
        "violated-over-set"
    )


def test_gather_dynamic_index_hull_narrows_with_the_declared_range():
    # the hull is over the DECLARED range, not over the whole operand: an
    # index confined to [0, 1] cannot reach row 2, and one confined to
    # [2, 2] reaches only that row. Without this the test above would pass
    # against a transfer that always hulled every row of the operand.
    assert propagate(_gather_query((0.0, 0.0), 1.0)).obligations[0].status == (
        "discharged"  # row 0 alone is 1.0
    )
    assert propagate(_gather_query((2.0, 2.0), 2.0)).obligations[0].status == (
        "discharged"  # row 2 alone is 2.0
    )
    assert propagate(_gather_query((0.0, 1.0), 2.0)).obligations[0].status == (
        "unknown"  # rows 0..1 reach 5.0
    )


def test_gather_index_straddling_the_axis_declines_not_crashes():
    # CHANGED EXPECTATION: [1, 7] on a 3-row operand admits inputs that
    # index in range and inputs that do not. The in-range ones take a row;
    # the others take a clamped or filled element that is not the one
    # written. No box states that, so the transfer declines — the middle
    # case of the three, and the one that keeps this round from modelling
    # the clamp.
    _gather_declined(
        _gather_query((1.0, 7.0), 6.0),
        "index element 0 spans [1, 7]",
        "straddles the legal positions [0, 2]",
    )


def test_gather_out_of_range_index_is_a_finding_not_a_decline():
    # CHANGED EXPECTATION: index 7 into a 3-row operand is out of bounds
    # for EVERY input the declared set admits. That is a fact about the
    # program, not a gap in stelling, and it gets its own note — the old
    # wording ("out-of-range handling is mode-dependent … never guessed")
    # explained why stelling declines and never said the program indexes
    # out of bounds. The ACCOUNTING is deliberately unchanged: still ⊤,
    # still unknown, still never a REFUTED.
    p = propagate(_gather_query((7.0, 7.0), 6.0))
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("gather", 1),)
    assert "gather" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "OUT-OF-BOUNDS INDEX (definite)" in n)
    for f in (
        "'gather'",
        "index element 0 spans [7, 7]",
        "EVERY value in it is outside",
        "the legal positions are [0, 2]",
        "no input for which this index is in bounds",
    ):
        assert f in note, (f, note)


def test_gather_wholly_negative_index_is_a_finding_too():
    # the other side of the axis, and the side a from-the-end index lands
    # on when it runs off the front: jnp normalises u[-11] on a length-10
    # axis to -1 BEFORE the take, so a negative index arriving here is
    # already out of bounds and is not a Python from-the-end request.
    p = propagate(_gather_query((-4.0, -2.0), 6.0))
    assert p.obligations[0].status == "unknown"
    note = next(n for n in p.notes if "OUT-OF-BOUNDS INDEX (definite)" in n)
    assert "spans [-4, -2]" in note, note


def test_gather_batched_form_declines_not_crashes():
    # the batched form (jax 0.11's lu_solve pivots gather) is not covered:
    # the note must name the offending non-empty field with its value
    _gather_declined(
        _gather_query((1.0, 1.0), 6.0, {"operand_batching_dims": (0,)}),
        "non-empty dimension-number field(s)",
        "'operand_batching_dims': (0,)",
    )


def test_gather_arity_decline_names_the_operand_count():
    # hand-IR-only: a traced gather binds operand + indices
    idx = var(0, aval((1, 1), "int32"))
    y = var(1, aval((1,)))
    pred, out = var(2, aval((1,), "bool")), var(3, aval((1,), "bool"))
    q = close(
        [
            any_eqn(idx, 1.0, 1.0),
            eqn("gather", [idx], y, _gather_row_params(1, ())),
            eqn("le", [y, lit(6.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _gather_declined(q, "binds 1 operand(s)")


def test_gather_rank0_operand_decline_names_the_missing_axis():
    # ALSO traced-reachable: lax.gather on a scalar with empty dims traces
    # on jax 0.11.0 (audit repair — the first claim here was
    # "hand-IR-only", measured false; the traced pin lives in
    # test_transfers_jax.py::test_gather_rank0_operand_declines_traced).
    # This jax-free hand-IR pin is kept beside it.
    idx = var(0, aval((1, 1), "int32"))
    x0 = var(1, aval(()))
    y = var(2, aval((1,)))
    pred, out = var(3, aval((1,), "bool")), var(4, aval((1,), "bool"))
    q = close(
        [
            any_eqn(idx, 0.0, 0.0),
            any_eqn(x0, 0.0, 1.0),
            eqn("gather", [x0, idx], y, _gather_row_params(1, ())),
            eqn("le", [y, lit(6.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _gather_declined(q, "rank-0", "no leading axis to take rows from")


def test_gather_missing_dimension_numbers_decline_hand_ir():
    # hand-IR-only: the traced form always records dimension_numbers
    # (pinned in test_transfers_jax.py); here the param is simply absent
    idx = var(0, aval((1, 1), "int32"))
    y = var(1, aval((1,)))
    pred, out = var(2, aval((1,), "bool")), var(3, aval((1,), "bool"))
    q = close(
        [
            any_eqn(idx, 1.0, 1.0),
            eqn(
                "gather",
                [_arr_lit([1.0, 5.0, 2.0], (3,)), idx],
                y,
                (("slice_sizes", (1,)),),
            ),
            eqn("le", [y, lit(6.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    _gather_declined(q, "no readable dimension_numbers", "got None")


def test_gather_non_integral_and_non_finite_points_decline_hand_ir():
    # hand-IR-only THROUGH THE WALK: a traced gather's index array is an
    # integer array, so a fractional or infinite POINT interval cannot
    # reach the transfer from a traced program — but a hand declaration
    # routed as float64 can, and the real walk drives it here.
    def q(lo, hi):
        idx = var(0, aval((1, 1)))  # float64 declaration as index data
        y = var(1, aval((1,)))
        pred, out = var(2, aval((1,), "bool")), var(3, aval((1,), "bool"))
        return close(
            [
                any_eqn(idx, lo, hi),
                eqn(
                    "gather",
                    [_arr_lit([1.0, 5.0, 2.0], (3,)), idx],
                    y,
                    _gather_row_params(1, ()),
                ),
                eqn("le", [y, lit(6.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    # message CHANGED with the index-bounds round: the classifier reads a
    # RANGE now, so it prints the span rather than "the point", and it
    # separates the two refusals it used to fold together — a non-integral
    # endpoint and an infinite one fail for different reasons and now say
    # so. Both still decline, which is the part that matters: this layer is
    # handed bounds with no dtype, so it never rounds [0.5, 0.5] inward to
    # the integers it contains.
    _gather_declined(
        q(0.5, 0.5),
        "index element 0 spans [0.5, 0.5]",
        "endpoints are not integers",
        "does not round an index inward",
    )
    _gather_declined(
        q(float("inf"), float("inf")),
        "spans [inf, inf]",
        "is not finite",
    )


def _transpose_query(perm, threshold):
    """y = transpose([[1,1,1],[9,9,9]], perm); assert y <= threshold."""
    y = var(0, aval((3, 2)))
    pred, out = var(1, aval((3, 2), "bool")), var(2, aval((3, 2), "bool"))
    return close(
        [
            eqn(
                "transpose",
                [_arr_lit([1.0, 1.0, 1.0, 9.0, 9.0, 9.0], (2, 3))],
                y,
                (("permutation", perm),),
            ),
            eqn("le", [y, lit(threshold)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_transpose_discharges_and_refutes():
    p = propagate(_transpose_query((1, 0), 10.0))
    assert p.obligations[0].status == "discharged"
    assert ("transpose", "exact") in p.transfers_used
    # definite-FALSE direction: the second source row's three 9.0s land in
    # column 1 of the transpose — 3/6 refutation proves exact placement
    p2 = propagate(_transpose_query((1, 0), 3.0))
    assert p2.obligations[0].status == "violated-over-set"
    assert "3/6" in p2.obligations[0].detail


def test_transpose_malformed_permutation_declines_not_crashes():
    y = var(0, aval((3, 2)))
    pred, out = var(1, aval((3, 2), "bool")), var(2, aval((3, 2), "bool"))
    q = close(
        [
            eqn(
                "transpose",
                [_arr_lit([1.0] * 6, (2, 3))],
                y,
                (("permutation", (0, 0)),),
            ),
            eqn("le", [y, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("transpose" in n and "declined" in n for n in p.notes)


# --- split: named declines (silent-⊤ conversion) -----------------------------
#
# jax validates every one of these forms at trace time (measured in
# test_transfers_jax.py::test_split_malformed_params_cannot_be_traced), so
# no traced program reaches them: each decline path is driven here by
# hand-built IR through the REAL walk, and pins both the named reason —
# with its numbers bound to the right operands — and the accounting, which
# must be identical to the returned-None decline it replaced: status
# unknown, coverage.unknown == 1, the primitive in unknown_primitives.


def _meqn(prim, ins, outs, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=tuple(outs),
        params=tuple(params),
    )


def _split_query(params, n_out=2, n_in=1):
    """x = any((4,)); split(x) -> n_out pieces; assert piece0 <= 1.0."""
    x = var(0, aval((4,)))
    outs = tuple(var(1 + i, aval((2,))) for i in range(n_out))
    pred, ob = var(9, aval((2,), "bool")), var(10, aval((2,), "bool"))
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            _meqn("split", [x] * n_in, outs, params),
            eqn("le", [outs[0], lit(1.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )


def _split_declined(q, *frags):
    p = propagate(q)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("split", 1),)
    assert "split" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'split' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


_SPLIT_OK = (("sizes", (2, 2)), ("axis", 0))


def test_split_hand_ir_positive_control():
    # the declines below must not have closed the row: the well-formed
    # hand-built form still discharges exactly
    p = propagate(_split_query(_SPLIT_OK))
    assert p.obligations[0].status == "discharged"
    assert ("split", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_split_arity_decline_names_the_operand_count():
    # 3 operands, 2 outputs: the printed count must be the OPERAND count
    # (a mutant printing the output count would show 2 here)
    _split_declined(
        _split_query(_SPLIT_OK, n_in=3),
        "binds 3 operands",
        "no single array",
    )


def test_split_absent_params_decline_names_what_is_missing():
    _split_declined(_split_query((("axis", 0),)), "'sizes'", "absent or None")
    _split_declined(_split_query((("sizes", (2, 2)),)), "'axis'")
    note = _split_declined(_split_query(()), "'sizes' and 'axis'")
    # present-with-None is the same fact as absent for this row and is
    # named by the same branch
    _split_declined(
        _split_query((("sizes", None), ("axis", 0))), "'sizes'"
    )
    assert "never guessed" in note


def test_split_non_integer_params_decline_prints_them():
    _split_declined(
        _split_query((("sizes", ("a", "b")), ("axis", 0))),
        "do not read as integers",
        "sizes=('a', 'b')",
    )
    _split_declined(
        _split_query((("sizes", (2, 2)), ("axis", "q"))),
        "axis='q'",
    )


def test_split_axis_out_of_range_decline_prints_axis_and_rank():
    _split_declined(
        _split_query((("sizes", (2, 2)), ("axis", 5))),
        "axis 5 lies outside the operand's rank 1",
        "(operand shape (4,))",
    )


def test_split_negative_size_decline_prints_the_offenders():
    _split_declined(
        _split_query((("sizes", (5, -1)), ("axis", 0))),
        "sizes (5, -1)",
        "negative piece extents (-1,)",
    )


def test_split_non_partition_decline_prints_sum_and_extent():
    # the numbers must be bound the right way round: 5 is the SIZES sum,
    # 4 is the AXIS extent (a swapped mutant fails both fragments)
    _split_declined(
        _split_query((("sizes", (2, 3)), ("axis", 0))),
        "sizes (2, 3) sum to 5",
        "axis 0 has extent 4",
    )


def test_split_output_arity_disagreement_declines_and_never_binds():
    # params name two pieces, the equation binds one output. WITHOUT the
    # decline the walk would zip two pieces onto one outvar and DISCHARGE
    # — this test measures that admission as a red, so the decline is
    # load-bearing, not cosmetic.
    _split_declined(
        _split_query((("sizes", (2, 2)), ("axis", 0)), n_out=1),
        "name 2 pieces",
        "binds 1 output(s)",
    )


# --- unstack: named declines (silent-⊤ conversion) ---------------------------
#
# Same posture as split above: jax validates the axis at trace time and its
# abstract eval fixes the output count (measured in test_transfers_jax.py::
# test_unstack_malformed_forms_cannot_be_traced), so every decline path is
# hand-IR-only and driven here through the real walk, pinning the named
# reason with its numbers and the unchanged accounting.


def _unstack_query(params, n_out=3, n_in=1):
    """x = any((3, 2)); unstack(x, axis) -> n_out pieces; piece0 <= 1.0."""
    x = var(0, aval((3, 2)))
    outs = tuple(var(1 + i, aval((2,))) for i in range(n_out))
    pred, ob = var(9, aval((2,), "bool")), var(10, aval((2,), "bool"))
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            _meqn("unstack", [x] * n_in, outs, params),
            eqn("le", [outs[0], lit(1.0)], pred),
            eqn("stelling_assert", [pred], ob),
        ],
        [ob],
    )


def _unstack_declined(q, *frags):
    p = propagate(q)  # must not raise: declines never kill the walk
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives == (("unstack", 1),)
    assert "unstack" not in dict(p.transfers_used)
    note = next(n for n in p.notes if "'unstack' declined this form" in n)
    for f in frags:
        assert f in note, (f, note)
    return note


def test_unstack_hand_ir_positive_control():
    # the declines below must not have closed the row
    p = propagate(_unstack_query((("axis", 0),)))
    assert p.obligations[0].status == "discharged"
    assert ("unstack", "exact") in p.transfers_used
    assert p.coverage.unknown == 0


def test_unstack_arity_decline_names_the_operand_count():
    # 2 operands, 3 outputs: the printed count must be the OPERAND count
    _unstack_declined(
        _unstack_query((("axis", 0),), n_in=2),
        "binds 2 operands",
        "no single array",
    )


def test_unstack_absent_axis_declines_named():
    note = _unstack_declined(_unstack_query(()), "'axis'", "absent or None")
    assert "never guessed" in note
    _unstack_declined(_unstack_query((("axis", None),)), "'axis'")


def test_unstack_non_integer_axis_decline_prints_it():
    _unstack_declined(
        _unstack_query((("axis", "q"),)),
        "does not read as an integer",
        "axis='q'",
    )


def test_unstack_axis_out_of_range_decline_prints_axis_and_rank():
    _unstack_declined(
        _unstack_query((("axis", 5),)),
        "axis 5 lies outside the operand's rank 2",
        "(operand shape (3, 2))",
    )


def test_unstack_output_arity_disagreement_declines_and_never_binds():
    # axis 0 has extent 3 but only 2 outputs are bound. WITHOUT the decline
    # the walk would zip three pieces onto two outvars and DISCHARGE — the
    # M5-class mutation measures that admission red.
    _unstack_declined(
        _unstack_query((("axis", 0),), n_out=2),
        "yields 3 piece(s)",
        "binds 2 output(s)",
    )
