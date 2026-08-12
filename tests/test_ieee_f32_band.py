# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Re-attack U2: the binary64-only guard completed — hand-built IR, no jax.

The measured flush is PER-DTYPE: jax 0.11.0 CPU flushes float32
subnormals (0 < |x| < 2**-126 — NORMAL binary64 numbers, invisible to
the binary64 haze) in f32 comparisons and in the f32→f64 convert, while
float16 is not flushed on this target. The adjudicated fix declines
rather than models: under ieee, any comparison with a non-f64 FLOAT
operand declines, any convert whose SOURCE is a non-f64 float declines
(including the whitelisted f32→f64 / f16→f32 / f16→f64 entries, whose
value-preservation claim is a gradual-semantics fact measured false for
f32 under DAZ), and the assume machinery — which consumes comparison
equations directly, bypassing the comparison transfer — drops non-f64-
float comparisons inert with the gap quoted (neither narrowing, nor
certification, nor the unsatisfiable-precondition raise). The decline is
uniform across non-f64 float dtypes (f16 pinned as a decline, not as
target behavior). Integer/bool comparisons and all f64 behavior are
unchanged.
"""

from __future__ import annotations

import pytest

from stelling import ir
from stelling.propagate import UnsatisfiableAssumptionError, propagate

F32_SUB = 1e-45  # an f32 subnormal (f32-normal threshold is 2**-126)
F32_SUB2 = 1e-40

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
F16 = ir.Aval(kind="ShapedArray", shape=(), dtype="float16")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, av=F64):
    return ir.Var(id=i, aval=av)


def lit(v, av=F64):
    return ir.Literal(val=v, aval=av)


def any_eqn(out, lo, hi, dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=()):
    return ir.JaxprEqn(
        primitive=prim, invars=tuple(ins), outvars=(out,), params=tuple(params)
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


# --- the auditor's four G-faces: every one indefinite now ---------------------


def test_g_a_f32_subnormal_comparison_is_indefinite():
    # G-A: assert(x > 0), f32 x = 1e-45 — was VERIFIED; measured: False
    # Now handled parametrically: the float32 subnormal haze covers 1e-45,
    # widening to include 0, so gt(x, 0) is unknown (correct by soundness).
    x, pred, out = var(0, F32), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, F32_SUB, F32_SUB, dtype="float32"),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    # real mode unchanged (the ℝ reading of the represented value)
    assert propagate(q).obligations[0].status == "discharged"


def test_g_b_distinct_f32_subnormals_eq_is_indefinite():
    # G-B: assert(x == y), f32 x = 1e-45, y = 1e-40 — was wrong REFUTED;
    # measured: True (both flush to 0)
    x, y, pred, out = var(0, F32), var(1, F32), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, F32_SUB, F32_SUB, dtype="float32"),
            any_eqn(y, F32_SUB2, F32_SUB2, dtype="float32"),
            eqn("eq", [x, y], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # never a definite
    assert propagate(q).obligations[0].status == "violated-over-set"


def _convert_query(src_dtype, src_av, sub):
    x = var(0, src_av)
    z = var(1)
    pred, out = var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, sub, sub, dtype=src_dtype),
            eqn(
                "convert_element_type",
                [x],
                z,
                params=[("new_dtype", "float64")],
            ),
            eqn("gt", [z, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_g_c_f32_to_f64_convert_is_indefinite_for_subnormals():
    # G-C: assert(convert_f64(x) > 0) — the whitelisted convert carried
    # the flushed 0 into f64 dataflow past the haze. Was VERIFIED;
    # measured: False. Now handled parametrically: the f32 subnormal haze
    # widens the source value to include 0 before the conversion, so
    # gt(z, 0) is unknown (correct by soundness).
    q = _convert_query("float32", F32, F32_SUB)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    # real mode keeps the whitelist byte-identically
    assert propagate(q).obligations[0].status == "discharged"


def test_g_d_converted_value_into_f64_arithmetic_is_indefinite():
    # G-D: assert(convert_f64(x) + 1e-30 > 1e-30) — the converted value
    # then satisfied the f64-only arithmetic guard. Was VERIFIED;
    # measured: False. The convert now declines to ⊤-maybe-NaN, which
    # blocks the downstream discharge.
    x, z, s = var(0, F32), var(1), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, F32_SUB, F32_SUB, dtype="float32"),
            eqn(
                "convert_element_type",
                [x],
                z,
                params=[("new_dtype", "float64")],
            ),
            eqn("add", [z, lit(1e-30)], s),
            eqn("gt", [s, lit(1e-30)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"


# --- the decline is uniform: f16 declines too (not target behavior) -----------


def test_f16_subnormal_comparison_is_indefinite():
    # float16's subnormal band is (-2**-14, 2**-14) ~ (-6.1e-5, 6.1e-5).
    # 6e-8 is deep in the float16 subnormal band, so the haze widens it
    # to include 0, making gt(x, 0) unknown. Now handled parametrically.
    x, pred, out = var(0, F16), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 6e-8, 6e-8, dtype="float16"),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    # the f16→f64 convert also handles subnormals correctly
    q2 = _convert_query("float16", F16, 6e-8)
    p2 = propagate(q2, semantics="ieee")
    assert p2.obligations[0].status == "unknown"
    # real mode keeps both (ℝ semantics; f16 6e-8 is a represented value)
    assert propagate(q).obligations[0].status == "discharged"
    assert propagate(q2).obligations[0].status == "discharged"


# --- integer/bool comparisons unaffected (no flush hazard) --------------------


def test_integer_and_bool_comparisons_stay_definite_under_ieee():
    x, pred, out = var(0, I32), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 5.0, 5.0, dtype="int32"),
            eqn("gt", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"
    # bool operands: eq of a definitely-true predicate with itself
    b, pred2, out2 = var(0, BOOL), var(1, BOOL), var(2, BOOL)
    q2 = close(
        [
            any_eqn(b, 1.0, 1.0, dtype="bool"),
            eqn("eq", [b, lit(True, BOOL)], pred2),
            eqn("stelling_assert", [pred2], out2),
        ],
        [out2],
    )
    assert propagate(q2, semantics="ieee").obligations[0].status == "discharged"
    # and an int refutation face stays definite too
    x3, pred3, out3 = var(0, I32), var(1, BOOL), var(2, BOOL)
    q3 = close(
        [
            any_eqn(x3, 5.0, 5.0, dtype="int32"),
            eqn("lt", [x3, lit(0.0)], pred3),
            eqn("stelling_assert", [pred3], out3),
        ],
        [out3],
    )
    assert propagate(q3, semantics="ieee").obligations[0].status == "violated-over-set"


# --- the swept surface: assume classification ---------------------------------


def test_f32_band_assume_neither_raises_nor_narrows():
    # assume(x == y) with distinct declared f32 subnormal points: both
    # values lie in the f32 subnormal band and get hazed to include 0
    # (format-parametric haze), making them non-point intervals. The
    # assume then drops inert because both sides vary (relational domain
    # needed). Under DAZ both flush to 0, so the runtime comparison IS
    # true — dropping inert is sound under both readings.
    x, y, pred, aout = var(0, F32), var(1, F32), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, F32_SUB, F32_SUB, dtype="float32"),
            any_eqn(y, F32_SUB2, F32_SUB2, dtype="float32"),
            eqn("eq", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q, semantics="ieee")  # must not raise
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    # real mode keeps the loud oracle byte-identically (in ℝ the two
    # represented values genuinely differ)
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q)


def test_f32_assume_cannot_narrow_or_certify_under_ieee():
    # assume(ge(x_f32, k)) over a declared f32 box: narrowing would
    # build a certified precondition whose runtime region can be empty
    # under DAZ — it must stay inert, and downstream obligations are
    # judged over the un-narrowed box (unconditional, sound)
    x, pred, aout = var(0, F32), var(1, BOOL), var(2, BOOL)
    p2v, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, F32_SUB, F32_SUB2, dtype="float32"),
            eqn("ge", [x, lit(5e-41, F32)], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("lt", [x, lit(1e-44, F32)], p2v),
            eqn("stelling_assert", [p2v], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.coverage.constrained == 0 and p.coverage.inert == 1
    assert p.assumptions == tuple(
        a for a in p.assumptions if "precondition" not in a
    )  # no conditional claim was stamped
    # the obligation's own f32 comparison also declined: unknown
    assert p.obligations[0].status == "unknown"
    # real mode narrows as before
    assert propagate(q).coverage.constrained == 1


# --- the swept surface: cond index dtype --------------------------------------


def test_float_cond_index_joins_all_branches_under_ieee():
    # jax's cond index is always int32; a float index is out-of-contract
    # hand-built IR whose value could flush per-dtype — under ieee its
    # interval is untrusted and every branch is joined (never a definite
    # single-branch selection)
    def add_eqn(a, b_lit, out):
        return eqn("add", [a, lit(b_lit)], out)

    b0 = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(var(10),), outvars=(var(11),),
            eqns=(add_eqn(var(10), 100.0, var(11)),),
        )
    )
    b1 = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(), invars=(var(20),), outvars=(var(21),),
            eqns=(add_eqn(var(20), 0.0, var(21)),),
        )
    )
    idx, x, y = var(0, F32), var(1), var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(idx, 0.0, 0.0, dtype="float32"),
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(
                primitive="cond",
                invars=(idx, x),
                outvars=(y,),
                params=(("branches", (b0, b1)),),
            ),
            eqn("ge", [y, lit(100.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    # ieee: both branches join → [1, 102] → undecided, never definite
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    # real mode unchanged: definite index 0 picks branch 0 → discharged
    assert propagate(q).obligations[0].status == "discharged"


# --- f64 behavior byte-unchanged ----------------------------------------------


def test_f64_comparisons_and_converts_unchanged():
    # the acceptance point shape at normal magnitude still lands definite
    t, dt, s, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(t, 1.0, 1.0),
            any_eqn(dt, 1e-20, 1e-20),
            eqn("add", [t, dt], s),
            eqn("gt", [s, t], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "violated-over-set"
    # f64 identity convert still passes through under ieee
    x, z, pred2, out2 = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q2 = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn(
                "convert_element_type",
                [x],
                z,
                params=[("new_dtype", "float64")],
            ),
            eqn("ge", [z, lit(1.0)], pred2),
            eqn("stelling_assert", [pred2], out2),
        ],
        [out2],
    )
    assert propagate(q2, semantics="ieee").obligations[0].status == "discharged"
    # int→f64 conversion (non-float source) keeps its ieee handling
    xi, zi, pred3, out3 = var(0, I32), var(1), var(2, BOOL), var(3, BOOL)
    q3 = close(
        [
            any_eqn(xi, 3.0, 3.0, dtype="int32"),
            eqn(
                "convert_element_type",
                [xi],
                zi,
                params=[("new_dtype", "float64")],
            ),
            eqn("eq", [zi, lit(3.0)], pred3),
            eqn("stelling_assert", [pred3], out3),
        ],
        [out3],
    )
    assert propagate(q3, semantics="ieee").obligations[0].status == "discharged"
