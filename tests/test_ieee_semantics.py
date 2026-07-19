# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The semantics="ieee" dial position — hand-built IR, no jax.

The ieee mode judges obligations about the traced program's IEEE binary64
round-to-nearest execution: native float endpoints (no outward rounding),
overflow saturating to the VALUE ±inf, NaN-producing corners routed to a
per-array maybe-NaN flag, ⊤-is-maybe-NaN, and NaN-aware comparisons.
This file carries the per-transfer ieee census unit tests, the two
marker-flip companions (test_audit_findings.py points here), the
``t + dt > t`` acceptance shapes, the tightened-domain guard (guard 1),
and the ieee verdict stamping. Guard 2 (solver-escalation refusal) lives
in tests/test_ieee_escalation_guard.py; the exactness lift's routing test
in tests/test_exactness_lift.py.
"""

from __future__ import annotations

import math
import struct

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import (
    IEEE_TRANSFERS,
    TIGHTENED_DOMAIN_REAL_REFUSAL,
    TRANSFERS,
    UnsatisfiableAssumptionError,
    propagate,
)
from stelling.verdict import (
    SEMANTICS_IEEE,
    SEMANTICS_REAL,
    make_verdict,
)

INF = math.inf

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def aval(shape=(), dtype="float64"):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype)


def var(i, av=F64):
    return ir.Var(id=i, aval=av)


def lit(v, av=F64):
    return ir.Literal(val=v, aval=av)


def arr_lit(values, shape, dtype="<f8", av_dtype="float64"):
    n = 1
    for d in shape:
        n *= d
    fmt = {"<f8": "d", "<i4": "i"}[dtype]
    return lit(
        ir.Array(dtype=dtype, shape=shape, data=struct.pack(f"<{n}{fmt}", *values)),
        aval(shape, av_dtype),
    )


def any_eqn(out, lo, hi, shape=(), dtype="float64"):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", dtype), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=(), source_info=()):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(ins),
        outvars=(out,),
        params=tuple(params),
        source_info=tuple(source_info),
    )


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


def assert_query(build_eqns, pred_var, out_var):
    eqns = list(build_eqns)
    eqns.append(eqn("stelling_assert", [pred_var], out_var))
    return close(eqns, [out_var])


VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="n/a (hand-built f64 IR)",
)


# --- the mode dial itself -----------------------------------------------------


def _simple_query():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    return assert_query(
        [any_eqn(x, 1.0, 2.0), eqn("ge", [x, lit(0.0)], pred)], pred, out
    )


def test_semantics_values_are_validated():
    q = _simple_query()
    with pytest.raises(ValueError):
        propagate(q, semantics="float32")
    with pytest.raises(ValueError):
        propagate(q, semantics="")
    assert propagate(q).semantics == "real"
    assert propagate(q, semantics="real").semantics == "real"
    assert propagate(q, semantics="ieee").semantics == "ieee"


def test_real_default_path_is_unchanged():
    # the default and the explicit real mode produce the identical
    # Propagation — the dial's real position is the pre-dial behavior
    q = _simple_query()
    assert propagate(q) == propagate(q, semantics="real")
    p = propagate(q)
    assert p.obligations[0].status == "discharged"
    assert p.semantics == "real"


def test_every_registered_transfer_is_censused_for_ieee():
    # rule 6, pinned beyond the module-level assert (which -O would strip):
    # no registered transfer without an explicit ieee census entry, and no
    # ieee entry for an unregistered primitive
    assert set(IEEE_TRANSFERS) == set(TRANSFERS)


# --- guard 1: tightened domains require ieee ----------------------------------


def test_domain_interval_is_the_only_accepted_value():
    q = _simple_query()
    assert propagate(q, domain="interval").obligations[0].status == "discharged"
    for bad in ("affine", "zonotope", "", "INTERVAL"):
        with pytest.raises(ValueError):
            propagate(q, domain=bad)
        with pytest.raises(ValueError):
            propagate(q, domain=bad, semantics="ieee")


def test_tightened_domain_under_real_carries_the_rationale():
    # the can't-drift pin: the real-mode refusal text exists and says why
    # (tightening ℝ arithmetic without float semantics converts accidental
    # UNKNOWNs into false VERIFIEDs; tightened domains run only under ieee)
    q = _simple_query()
    with pytest.raises(ValueError) as exc:
        propagate(q, domain="affine", semantics="real")
    msg = str(exc.value)
    assert TIGHTENED_DOMAIN_REAL_REFUSAL in msg
    assert "false VERIFIEDs" in msg
    assert "semantics='ieee'" in msg
    # under ieee the same unregistered domain still refuses (nothing is
    # registered), but without the real-mode rationale
    with pytest.raises(ValueError) as exc2:
        propagate(q, domain="affine", semantics="ieee")
    assert "only registered domain" in str(exc2.value)
    assert TIGHTENED_DOMAIN_REAL_REFUSAL not in str(exc2.value)


# --- verdict stamping ---------------------------------------------------------


def test_ieee_verdict_stamps_ieee_and_drops_the_real_convention():
    q = _simple_query()
    p = propagate(q, semantics="ieee")
    v = make_verdict(q, p, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.semantics == SEMANTICS_IEEE
    assert "binary64" in v.stamp.semantics
    # the 0·∞ = 0 convention is a consequence of ℝ semantics and must NOT
    # ride in an ieee stamp
    assert not any("0*inf = 0" in a for a in v.stamp.assumptions)
    # the ieee endpoint assumption rides instead: native binary64
    # round-to-nearest endpoints, monotonicity of fl-rounded ops relied on
    assert any(
        "native binary64" in a and "monotonicity" in a
        for a in v.stamp.assumptions
    )
    assert "native-endpoints+maybe-nan" in v.stamp.arithmetic_mode
    assert "semantics: ieee" in v.render()


def test_real_verdict_stamp_is_byte_identical():
    q = _simple_query()
    v = make_verdict(q, propagate(q), **VERSIONS)
    assert v.stamp.semantics == SEMANTICS_REAL
    assert any("0*inf = 0" in a for a in v.stamp.assumptions)
    assert not any("native binary64" in a for a in v.stamp.assumptions)
    assert v.stamp.arithmetic_mode == "interval/f64/outward-1ulp (stelling.interval)"


def test_ieee_stamps_its_measured_precision_boundary_and_real_never_does():
    """The mode's known precision limit is disclosed, not implied.

    Measured (design/ieee-reexamination.md): the maybe-NaN flag is unioned
    across array elements at construction, so ieee mode can return UNKNOWN
    exactly where the real-mode verdict discharges. Sound (an
    over-approximated flag only blocks discharges) but real, so it rides
    in every ieee stamp — a non-green under ieee must be readable against
    the scope line rather than as a float finding.
    """
    q = _simple_query()
    ieee = make_verdict(q, propagate(q, semantics="ieee"), **VERSIONS)
    real = make_verdict(q, propagate(q), **VERSIONS)
    assert iv.IEEE_NAN_HYGIENE_SCOPE in ieee.stamp.assumptions
    assert "OUTSIDE what ieee mode can reproduce" in iv.IEEE_NAN_HYGIENE_SCOPE
    # it is a precision statement, never a soundness one
    assert "not a soundness limit" in iv.IEEE_NAN_HYGIENE_SCOPE
    # and it must never appear in a real-mode stamp
    assert iv.IEEE_NAN_HYGIENE_SCOPE not in real.stamp.assumptions
    assert not any("maybe-NaN flag is unioned" in a for a in real.stamp.assumptions)


# --- the ieee kernels (interval.py unit level) --------------------------------


def s(lo, hi):
    return iv.IntervalArray(shape=(), los=(float(lo),), his=(float(hi),))


def test_ieee_add_point_inputs_give_point_float_outputs():
    box, made_nan = iv.ieee_add(s(1.0, 1.0), s(1e-20, 1e-20))
    assert (box.los[0], box.his[0]) == (1.0, 1.0)  # fl(1 + 1e-20) IS 1.0
    assert made_nan is False
    box2, _ = iv.ieee_add(s(0.1, 0.1), s(0.2, 0.2))
    assert box2.los[0] == box2.his[0] == 0.1 + 0.2  # the float itself


def test_ieee_add_overflow_saturates_to_the_value_inf():
    box, made_nan = iv.ieee_add(s(1e308, 1.7e308), s(1e308, 1.7e308))
    assert (box.los[0], box.his[0]) == (INF, INF)  # fl(2e308) == inf
    assert made_nan is False  # inf is a VALUE; no NaN corner here


def test_ieee_add_inf_minus_inf_corner_sets_the_flag():
    box, made_nan = iv.ieee_add(s(-INF, 0.0), s(INF, INF))
    assert made_nan is True
    assert (box.los[0], box.his[0]) == (INF, INF)  # non-NaN corner hull
    # both-sides-⊤: flag plus full hull, never a NaN endpoint
    box2, made_nan2 = iv.ieee_add(s(-INF, INF), s(-INF, INF))
    assert made_nan2 is True
    assert (box2.los[0], box2.his[0]) == (-INF, INF)


def test_ieee_sub_same_shape():
    box, made_nan = iv.ieee_sub(s(0.0, INF), s(0.0, INF))
    assert made_nan is True  # inf − inf
    assert (box.los[0], box.his[0]) == (-INF, INF)


def test_ieee_mul_zero_times_inf_is_nan_not_the_real_convention():
    # the real-mode 0·∞ = 0 endpoint convention must NOT be reused
    box, made_nan = iv.ieee_mul(s(INF, INF), s(0.0, 0.0))
    assert made_nan is True
    # every corner is NaN: the non-NaN value set is empty; ⊤ is the hull
    assert (box.los[0], box.his[0]) == (-INF, INF)
    # interior zero against an endpoint inf is caught too (no NaN corner)
    box2, made_nan2 = iv.ieee_mul(s(-1.0, 1.0), s(INF, INF))
    assert made_nan2 is True
    assert (box2.los[0], box2.his[0]) == (-INF, INF)


def test_ieee_mul_finite_endpoints_are_native_no_outward_bump():
    box, made_nan = iv.ieee_mul(s(0.1, 0.1), s(0.1, 0.1))
    assert made_nan is False
    assert box.los[0] == box.his[0] == 0.1 * 0.1  # exactly fl(0.01)
    # the real transfer pads outward — the pair witnesses the difference
    real = iv.mul(s(0.1, 0.1), s(0.1, 0.1))
    assert real.los[0] < box.los[0] and real.his[0] > box.his[0]


def test_ieee_div_zero_denominator_is_top_and_nan_only_when_possible():
    # x/0 is ±inf (a value): ⊤ interval, no NaN unless 0/0 is possible
    box, made_nan = iv.ieee_div(s(1.0, 1.0), s(-1.0, 1.0))
    assert (box.los[0], box.his[0]) == (-INF, INF)
    assert made_nan is False  # numerator excludes 0: no 0/0
    box2, made_nan2 = iv.ieee_div(s(0.0, 1.0), s(0.0, 1.0))
    assert made_nan2 is True  # 0/0 reachable
    box3, made_nan3 = iv.ieee_div(s(1.0, INF), s(2.0, INF))
    assert made_nan3 is True  # inf/inf reachable
    assert (box3.los[0], box3.his[0]) == (0.0, INF)  # non-NaN corner hull


def test_ieee_div_clean_corners_are_native_exact():
    box, made_nan = iv.ieee_div(s(1.0, 2.0), s(2.0, 4.0))
    assert made_nan is False
    assert (box.los[0], box.his[0]) == (0.25, 1.0)  # no outward bump


# --- census: native endpoint exactness through propagation --------------------


def test_ieee_mul_point_result_is_the_float_point():
    # eq against the exact float product discharges under ieee (point
    # float outputs) and stays unknown under real (outward rounding)
    x, z, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.1, 0.1),
            eqn("mul", [x, x], z),
            eqn("eq", [z, lit(0.1 * 0.1)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"
    assert propagate(q).obligations[0].status == "unknown"  # ℝ: padded box


def test_ieee_add_point_result_is_the_float_point():
    x, z, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.1, 0.1),
            eqn("add", [x, lit(0.2)], z),
            eqn("eq", [z, lit(0.1 + 0.2)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"
    assert propagate(q).obligations[0].status == "unknown"


def test_ieee_sub_inf_minus_inf_blocks_discharge():
    # x − x over [0, inf]: the inf − inf corner is NaN-possible, so even
    # the ℝ-tautology s ≤ +inf does not discharge
    x, z, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, INF),
            eqn("sub", [x, x], z),
            eqn("le", [z, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"  # ℝ marker shape


def test_ieee_div_by_zero_straddling_interval_is_unknown():
    x, y, z, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = assert_query(
        [
            any_eqn(x, 1.0, 2.0),
            any_eqn(y, -1.0, 1.0),
            eqn("div", [x, y], z),
            eqn("le", [z, lit(1e300)], pred),
        ],
        pred,
        out,
    )
    # z reaches ±inf (division by zero is a value): le(z, 1e300) is not
    # definitely true, and not definitely false — unknown
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"


# --- census: comparisons under maybe-NaN --------------------------------------


def _flagged_inf_point_eqns():
    """z = (−inf..0) + (inf..inf): interval [inf, inf], maybe-NaN — the
    narrow-interval-with-flag construction the comparison census needs."""
    a, b, z = var(0), var(1), var(2)
    return [
        any_eqn(a, -INF, 0.0),
        any_eqn(b, INF, INF),
        eqn("add", [a, b], z),
    ], z


def test_maybe_nan_blocks_definite_true_for_falsified_comparisons():
    # z ∈ {inf} ∪ {NaN}: le(z, inf) holds for inf but NaN falsifies —
    # never definitely true
    eqns, z = _flagged_inf_point_eqns()
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(eqns + [eqn("le", [z, lit(INF)], pred)], pred, out)
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"


def test_maybe_nan_allows_definite_false():
    # lt(z, 5): the non-NaN part {inf} is definitely false, and NaN also
    # falsifies lt — the universal-falsity claim stands: REFUTED
    eqns, z = _flagged_inf_point_eqns()
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(eqns + [eqn("lt", [z, lit(5.0)], pred)], pred, out)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "violated-over-set"
    assert make_verdict(q, p, **VERSIONS).status == "REFUTED"


def test_ne_is_the_dual_nan_satisfies_it():
    eqns, z = _flagged_inf_point_eqns()
    # ne(z, 5): non-NaN part {inf} ≠ 5 definitely, and NaN ≠ 5 is TRUE —
    # definitely true despite the flag
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(eqns + [eqn("ne", [z, lit(5.0)], pred)], pred, out)
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"
    # ne(z, inf): the non-NaN part is the single point inf == inf, which
    # would be definitely false — but NaN ≠ inf is true, so definite-false
    # is blocked
    eqns2, z2 = _flagged_inf_point_eqns()
    pred2, out2 = var(3, BOOL), var(4, BOOL)
    q2 = assert_query(eqns2 + [eqn("ne", [z2, lit(INF)], pred2)], pred2, out2)
    assert propagate(q2, semantics="ieee").obligations[0].status == "unknown"


def test_eq_definite_false_survives_the_flag():
    eqns, z = _flagged_inf_point_eqns()
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(eqns + [eqn("eq", [z, lit(5.0)], pred)], pred, out)
    # {inf} vs 5 disjoint (definitely false) and NaN == 5 is false too
    assert propagate(q, semantics="ieee").obligations[0].status == "violated-over-set"


# --- census: the binary64-only guard on the arithmetic core -------------------


def test_f32_arithmetic_declines_with_the_gap_quoted():
    x, z = var(0, F32), var(1, F32)
    pred, out = var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 1.0, 2.0, dtype="float32"),
            eqn("add", [x, x], z),
            eqn("le", [z, lit(100.0)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # ⊤-maybe-NaN, never a definite
    assert any("binary64-only" in n for n in p.notes)
    assert p.coverage.unknown >= 1
    # the same query under real semantics still discharges (ℝ claim)
    assert propagate(q).obligations[0].status == "discharged"


def test_int_arithmetic_declines_under_ieee():
    # integer add can wrap; native float endpoints do not model it
    x, z = var(0, I32), var(1, I32)
    pred, out = var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 7.0, dtype="int32"),
            eqn("add", [x, x], z),
            eqn("le", [z, lit(100.0)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any("binary64-only" in n for n in p.notes)


# --- census: pow declines maybe-NaN operands ----------------------------------


def test_pow_declines_flagged_operands_with_the_gap():
    # pow(NaN, 0) = 1 escapes both the corner bracket and the flag —
    # measured on jax 0.11.0; the census declines rather than mismodels
    eqns, z = _flagged_inf_point_eqns()
    w, pred, out = var(3), var(4, BOOL), var(5, BOOL)
    q = assert_query(
        eqns
        + [
            eqn("pow", [lit(2.0), z], w),
            eqn("le", [w, lit(INF)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any("pow" in n and "maybe-NaN" in n for n in p.notes)


def test_pow_and_exp_keep_libm_brackets_when_nan_free():
    x, e, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("exp", [x], e),
            eqn("lt", [e, lit(8.0)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "discharged"
    assert ("exp", "sound-libm") in dict(p.transfers_used).items()
    assert any("libm" in a for a in p.assumptions)  # the assumption still rides


# --- census: structural ops move flags with the data --------------------------


def test_flag_rides_through_structural_ops():
    # mystery → ⊤-maybe-NaN → reshape → the flag survives the data
    # movement and still blocks the r ≤ +inf discharge
    x = var(0, aval((2,)))
    r = var(1, aval((2,)))
    r2 = var(2, aval((2, 1)))
    pred, out = var(3, aval((2, 1), "bool")), var(4, aval((2, 1), "bool"))
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0, shape=(2,)),
            eqn("mystery_loop", [x], r),
            eqn(
                "reshape",
                [r],
                r2,
                params=[("new_sizes", (2, 1)), ("dimensions", None)],
            ),
            eqn("le", [r2, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"  # the ℝ side


def test_scatter_takes_the_updates_flag():
    # a maybe-NaN update poisons the whole array (per-array flag)
    x = var(0, aval((3,)))
    u0, u = var(1), var(2)
    k = arr_lit([0], (1,), "<i4", "int32")
    znew = var(3, aval((3,)))
    pred, out = var(4, aval((3,), "bool")), var(5, aval((3,), "bool"))
    scatter_params = [
        (
            "dimension_numbers",
            ir.NamedTupleParam(
                cls="ScatterDimensionNumbers",
                fields=(
                    ("update_window_dims", ()),
                    ("inserted_window_dims", (0,)),
                    ("scatter_dims_to_operand_dims", (0,)),
                ),
            ),
        ),
        ("update_jaxpr", None),
    ]
    q = assert_query(
        [
            any_eqn(x, 1.0, 2.0, shape=(3,)),
            any_eqn(u0, 0.0, 1.0),
            eqn("mystery_op", [u0], u),  # ⊤-maybe-NaN scalar update
            eqn("scatter", [x, k, u], znew, params=scatter_params),
            eqn("le", [znew, lit(INF)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    # real mode: same shape discharges (⊤ element still ≤ +inf in ℝ)
    assert propagate(q).obligations[0].status == "discharged"


def test_top_is_maybe_nan_under_ieee():
    # rule 5 directly: an unknown primitive's output is ⊤-maybe-NaN, so a
    # NaN-falsified comparison over it never discharges (the marker
    # companion below is the named acceptance shape of this rule)
    x, r, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_op", [x], r),
            eqn("ge", [r, lit(-INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"


# --- census: max/min hull under maybe-NaN -------------------------------------


def test_minmax_flagged_operand_widens_to_the_hull_and_flags():
    # max(x, z) with z maybe-NaN: the backend's NaN ordering decides the
    # result (measured on jax 0.11.0: NaN propagates), so the transfer
    # covers every non-NaN outcome with the operand hull and flags the
    # result — ge(m, 0) must not discharge even though max(x∈[0,1], ·)
    # would be ≥ 0 under either NaN convention's non-NaN face... the NaN
    # face falsifies
    eqns, z = _flagged_inf_point_eqns()
    x, m, pred, out = var(3), var(4), var(5, BOOL), var(6, BOOL)
    q = assert_query(
        eqns
        + [
            any_eqn(x, 0.0, 1.0),
            eqn("max", [x, z], m),
            eqn("ge", [m, lit(0.0)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"


def test_minmax_nan_free_is_exact_under_ieee():
    x, y, m, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 2.0, 3.0),
            eqn("max", [x, y], m),
            eqn("ge", [m, lit(2.0)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"


# --- census: convert under ieee -----------------------------------------------


def test_exact_conversion_propagates_the_flag():
    # the f64 identity convert is exact for every representable value
    # INCLUDING NaN: the flag must ride through, blocking the downstream
    # discharge. (The f32→f64 whitelist entry no longer reaches this
    # path under ieee — it declines outright per re-attack U2, pinned in
    # tests/test_ieee_f32_band.py.)
    x = var(0)
    r = var(1)
    y = var(2)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_op", [x], r),  # ⊤-maybe-NaN f64
            eqn(
                "convert_element_type",
                [r],
                y,
                params=[("new_dtype", "float64")],
            ),
            eqn("le", [y, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"


def test_float_to_int_conversion_declines_maybe_nan_input():
    x, r, y = var(0), var(1), var(2, I32)
    pred, out = var(3, BOOL), var(4, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_op", [x], r),
            eqn(
                "convert_element_type",
                [r],
                y,
                params=[("new_dtype", "int32")],
            ),
            eqn("le", [y, lit(1e12)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert any("maybe-NaN" in n and "convert" in n for n in p.notes)


# --- census: assume under ieee ------------------------------------------------


def test_assume_narrows_under_ieee_and_certifies_declared_inputs():
    x, ap, aout, p2, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(2.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "discharged"
    assert p.coverage.constrained == 1
    assert not any("UNCERTIFIED" in n for n in p.notes)


def test_assume_clears_maybe_nan_on_the_constrained_target():
    # an assumed-true comparison excludes NaN: the flag on the target is
    # cleared (disclosed), so the subsequent le discharges — while the
    # uncertified-precondition machinery still withholds nothing more than
    # in real mode (the target is a transfer output: VERIFIED with the
    # may-be-vacuous disclosure)
    x, z, ap, aout, p2, out = (
        var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_op", [x], z),  # ⊤-maybe-NaN
            eqn("le", [z, lit(5.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [z, lit(6.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "discharged"
    assert any("cleared maybe-NaN" in n for n in p.notes)
    # the target was an over-approximated intermediate: still uncertified
    assert any("UNCERTIFIED" in n for n in p.notes)
    # without the clearing the discharge would be blocked — the real-mode
    # control on the same query (no flags at all) also discharges
    assert propagate(q).obligations[0].status == "discharged"


def test_assume_with_maybe_nan_bound_stays_inert_with_the_gap():
    # the reachable flagged-finite-point construction: concatenate a
    # declared point row with a mystery row (per-array flag True), gather
    # the point row back out — interval [3, 3], maybe-NaN. Using it as an
    # assume BOUND must drop: if the bound IS NaN the true assumed region
    # is empty, and the half-space would certify a vacuous precondition.
    from test_transfers import _gather_row_params  # the registered form

    x = var(0, aval((1,)))
    k = var(1, aval((1,)))
    m0, m = var(2, aval((1,))), var(3, aval((1,)))
    cat = var(4, aval((2,)))
    idx = arr_lit([0], (1, 1), "<i4", "int32")
    g = var(5, aval((1,)))
    ap, aout = var(6, aval((1,), "bool")), var(7, aval((1,), "bool"))
    p2, out = var(8, aval((1,), "bool")), var(9, aval((1,), "bool"))
    q = close(
        [
            any_eqn(x, 0.0, 4.0, shape=(1,)),
            any_eqn(k, 3.0, 3.0, shape=(1,)),
            any_eqn(m0, 0.0, 1.0, shape=(1,)),
            eqn("mystery_op", [m0], m),
            eqn("concatenate", [k, m], cat, params=[("dimension", 0)]),
            eqn("gather", [cat, idx], g, _gather_row_params(1, ())),
            eqn("ge", [x, g], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("lt", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    # the assume dropped (inert), with the maybe-NaN-bound gap quoted
    assert p.coverage.constrained == 0 and p.coverage.inert == 1
    assert any(
        "bound may be NaN" in n and "half-space" in n for n in p.notes
    )
    # x stayed un-narrowed: lt(x, 1) over [0, 4] is a straddle, and no
    # REFUTED was minted from the possibly-empty precondition
    assert p.obligations[0].status == "unknown"
    # the real-mode control narrows normally (gather output [3,3] is a
    # finite point with no flag concept)
    pr = propagate(q)
    assert pr.coverage.constrained == 1
    assert pr.obligations[0].status == "violated-over-set"


def test_unsatisfiable_assume_still_raises_under_ieee():
    # NaN also falsifies, so a definitely-false precondition is a harness
    # defect under ieee exactly as under real
    x, pred, aout = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("ge", [x, lit(2.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    with pytest.raises(UnsatisfiableAssumptionError):
        propagate(q, semantics="ieee")


# --- the marker companions (test_audit_findings.py points here) ---------------


def _marker_overflow_times_zero_query():
    x, z1, z2, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(x, 1e308, 1.7e308),
            eqn("add", [x, x], z1),
            eqn("mul", [z1, lit(0.0)], z2),
            eqn("lt", [z2, lit(1.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_ieee_marker_overflow_times_zero_does_not_discharge():
    # the (x+x)·0 construction: under ieee, fl(x+x) == inf over the whole
    # box (overflow saturates to the VALUE inf) and inf·0 is NaN — the
    # obligation must NOT discharge. The real-mode marker
    # (test_audit_findings.test_R_gap_marker_overflow_times_zero_
    # discharges_in_R) stays untouched and still discharges.
    q = _marker_overflow_times_zero_query()
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # NaN-possible: not discharged
    assert propagate(q).obligations[0].status == "discharged"  # the ℝ side
    v = make_verdict(q, p, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.semantics == SEMANTICS_IEEE


def _marker_top_leq_inf_query():
    x, r, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_loop", [x], r),
            eqn("le", [r, lit(INF)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_ieee_marker_top_output_leq_inf_does_not_discharge():
    # the r ≤ +∞ ⊤-construction: ⊤ under ieee is maybe-NaN, and NaN
    # falsifies ≤ — the tautology dies with the dial. The real-mode marker
    # (test_audit_findings.test_R_gap_marker_top_output_leq_inf_
    # discharges_and_is_tautological) stays untouched and still discharges.
    q = _marker_top_leq_inf_query()
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"  # the ℝ side


# --- the t + dt > t acceptance shapes -----------------------------------------


def _t_dt_query(dt_lo, dt_hi):
    t, dt, ss, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    return close(
        [
            any_eqn(t, 1.0, 1.0),
            any_eqn(dt, dt_lo, dt_hi),
            eqn("add", [t, dt], ss),
            eqn("gt", [ss, t], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_t_plus_dt_point_collapse_is_refuted_under_ieee():
    # diffrax#632's shape: t = 1.0, dt = 1e-20 — fl(t + dt) == t exactly,
    # so t + dt > t is DEFINITELY FALSE about the float execution
    q = _t_dt_query(1e-20, 1e-20)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "violated-over-set"
    v = make_verdict(q, p, **VERSIONS)
    assert v.status == "REFUTED"
    assert v.stamp.semantics == SEMANTICS_IEEE


def test_t_plus_dt_box_spanning_the_collapse_boundary_is_unknown():
    # dt ∈ [1e-20, 1e-15] spans the collapse boundary (ulp(1.0)/2): some
    # dt collapse, some do not — unknown, never a guess
    q = _t_dt_query(1e-20, 1e-15)
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"


def test_t_plus_dt_provably_above_ulp_still_verifies_under_ieee():
    # the mode models float, it does not refuse everything: dt ≥ ulp(1.0)
    # over the whole box ⟹ fl(t + dt) > t for every point — VERIFIED
    dt_lo = 3e-16
    assert dt_lo >= math.ulp(1.0)  # the premise the shape claims
    q = _t_dt_query(dt_lo, 1e-15)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "discharged"
    assert make_verdict(q, p, **VERSIONS).status == "VERIFIED"


# --- select_n / cond flag plumbing --------------------------------------------


def test_select_n_takes_the_or_of_case_flags():
    x, r, w, pred, out = (
        var(0), var(1), var(2), var(3, BOOL), var(4, BOOL),
    )
    which = var(5, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_op", [x], r),  # flagged case
            eqn("ge", [x, lit(0.5)], which),
            eqn("select_n", [which, x, r], w),
            eqn("le", [w, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    # with both cases NaN-free the same shape discharges
    y = var(6)
    q2 = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 2.0, 3.0),
            eqn("ge", [x, lit(0.5)], which),
            eqn("select_n", [which, x, y], w),
            eqn("le", [w, lit(3.0)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q2, semantics="ieee").obligations[0].status == "discharged"


def test_cond_joins_branch_flags():
    bx = var(20)
    bm = var(21)
    branch_flagged = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(bx,),
            outvars=(bm,),
            eqns=(eqn("mystery_op", [bx], bm),),
        )
    )
    by = var(30)
    branch_clean = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(by,), outvars=(by,), eqns=())
    )
    x, w = var(0), var(1)
    sel = var(2, I32)
    cout, pred, out = var(3), var(4, BOOL), var(5, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(w, 0.0, 1.0),
            eqn("convert_element_type", [w], sel, params=[("new_dtype", "int32")]),
            eqn(
                "cond", [sel, x], cout,
                params=[("branches", (branch_clean, branch_flagged))],
            ),
            eqn("le", [cout, lit(INF)], pred),
        ],
        pred,
        out,
    )
    # the flagged branch is possible: the join carries maybe-NaN
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    # a definite selector for the clean branch drops the flag
    q2 = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(w, 0.0, 0.0),
            eqn("convert_element_type", [w], sel, params=[("new_dtype", "int32")]),
            eqn(
                "cond", [sel, x], cout,
                params=[("branches", (branch_clean, branch_flagged))],
            ),
            eqn("le", [cout, lit(1.0)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q2, semantics="ieee").obligations[0].status == "discharged"


# --- transparent scopes carry flags -------------------------------------------


def test_flag_survives_a_transparent_jit_scope():
    jx = var(40)
    jm = var(41)
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(jx,),
            outvars=(jm,),
            eqns=(eqn("mystery_op", [jx], jm),),
        )
    )
    x, jout, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("jit", [x], jout, params=[("jaxpr", inner)]),
            eqn("le", [jout, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"


# --- literals -----------------------------------------------------------------


def test_nan_literal_is_top_maybe_nan_not_a_crash():
    # the NaN sentinel: a legal value under ieee — ⊤-maybe-NaN with the
    # note, never a crash, never a definite verdict
    x, z, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(math.nan)], z),
            eqn("le", [z, lit(INF)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("outside the domain" in n for n in p.notes)


# --- float-fidelity audit F1: the selector invariant is enforced --------------


def test_select_n_float_selector_declines_with_the_gap():
    # the auditor's c10 shape: a float64 selector with definite interval
    # [1, 1] whose value set is {1.0} ∪ {NaN} (via 0·inf → NaN, +1).
    # Before the guard the selector's flag was dropped (out flag = OR of
    # CASE flags only) and eq(y, 7) discharged; the dtype guard now
    # declines the un-tracable shape with the gap quoted.
    a, b, z, sel = var(0), var(1), var(2), var(3)
    c0, c1v, y = var(4), var(5), var(6)
    pred, out = var(7, BOOL), var(8, BOOL)
    q = assert_query(
        [
            any_eqn(a, 0.0, 0.0),
            any_eqn(b, 5.0, INF),
            eqn("mul", [a, b], z),  # {0} ∪ {NaN}: interval [0,0], flagged
            eqn("add", [z, lit(1.0)], sel),  # {1} ∪ {NaN}: [1,1], flagged
            any_eqn(c0, 5.0, 5.0),
            any_eqn(c1v, 7.0, 7.0),
            eqn("select_n", [sel, c0, c1v], y),
            eqn("eq", [y, lit(7.0)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # never discharged
    assert any(
        "select_n" in n and "selector dtype" in n for n in p.notes
    )
    assert p.coverage.unknown >= 1  # the decline is counted


def test_select_n_flagged_selector_declines_with_the_gap():
    # a maybe-NaN-flagged selector of a LEGAL dtype (bool ⊤ decline
    # artifact) no longer silently selects/joins: it declines with the
    # reason quoted — matching the and/or/reduce_or guard pattern
    x, r, w = var(0), var(1, BOOL), var(2)
    y = var(3)
    pred, out = var(4, BOOL), var(5, BOOL)
    q = assert_query(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mystery_pred", [x], r),  # bool ⊤, flagged
            any_eqn(y, 2.0, 3.0),
            eqn("select_n", [r, x, y], w),
            eqn("le", [w, lit(3.0)], pred),
        ],
        pred,
        out,
    )
    p = propagate(q, semantics="ieee")
    # before the guard: join [0, 3] → le(w, 3) discharged; now declined
    assert p.obligations[0].status == "unknown"
    assert any(
        "selector carries maybe-NaN" in n for n in p.notes
    )


def test_inf_literal_is_a_definite_non_nan_value():
    # inf is a VALUE, not NaN: eq(x, inf) over a declared [inf, inf] input
    # is definitely true under ieee
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = assert_query(
        [
            any_eqn(x, INF, INF),
            eqn("eq", [x, lit(INF)], pred),
        ],
        pred,
        out,
    )
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"
