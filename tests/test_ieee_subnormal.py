# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The subnormal haze (float-fidelity audit U1) — hand-built IR, no jax.

Measured jax 0.11.0 CPU binary64 flushes subnormals (FTZ on results, DAZ
on operands) in arithmetic, comparisons, and libm, while strict IEEE-754
keeps gradual underflow. Neither pure semantics is right for every
target, so ieee mode is sound for BOTH: any interval touching the open
subnormal band (-MIN_NORMAL, MIN_NORMAL) excluding {0} is hulled with 0
— at declared inputs and constants (DAZ flushes inputs), at every ieee
arithmetic result (FTZ), and at comparison operands before judging (DAZ
reaches comparisons). Subnormal-band outcomes are indeterminate, never
definite; outside the band nothing changes.

This file pins the audit's seven end-to-end faces (each was a false
VERIFIED or wrong REFUTED before the fix; each must now be indefinite),
the band-boundary pair, the DAZ-created-0/0 divisor interplay, the
stamped disclosure, and the strict-assume certification face of the same
indeterminacy.
"""

from __future__ import annotations

import math

from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate
from stelling.verdict import make_verdict

INF = math.inf
MIN_NORMAL = 2.0**-1022
LARGEST_SUBNORMAL = math.nextafter(MIN_NORMAL, 0.0)
TINY = 5e-324

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, av=F64):
    return ir.Var(id=i, aval=av)


def lit(v):
    return ir.Literal(val=v, aval=F64)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out):
    return ir.JaxprEqn(primitive=prim, invars=tuple(ins), outvars=(out,))


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=tuple(outvars), eqns=tuple(eqns))
    )


VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="n/a (hand-built f64 IR)",
)


def _one_op_cmp_query(x_lo, x_hi, op, cmp, bound, y_bounds=None):
    """x declared; z = op(x, x) (or just x when op is None); assert
    cmp(z, bound)."""
    x = var(0)
    eqns = [any_eqn(x, x_lo, x_hi)]
    nxt = 1
    if y_bounds is not None:
        y = var(nxt)
        nxt += 1
        eqns.append(any_eqn(y, *y_bounds))
    else:
        y = x
    if op is not None:
        z = var(nxt)
        nxt += 1
        eqns.append(eqn(op, [x, y], z))
    else:
        z = x
    pred, out = var(nxt, BOOL), var(nxt + 1, BOOL)
    eqns.append(eqn(cmp, [z, lit(bound)], pred))
    eqns.append(eqn("stelling_assert", [pred], out))
    return close(eqns, [out])


# --- the kernel/haze unit surface ---------------------------------------------


def s(lo, hi):
    return iv.IntervalArray(shape=(), los=(float(lo),), his=(float(hi),))


def test_subnormal_haze_unit():
    hazed, changed = iv.subnormal_haze(s(TINY, TINY))
    assert changed and (hazed.los[0], hazed.his[0]) == (0.0, TINY)
    hazed, changed = iv.subnormal_haze(s(-1e-320, -TINY))
    assert changed and (hazed.los[0], hazed.his[0]) == (-1e-320, 0.0)
    # identity: 0 already contained, or clear of the band
    for lo, hi in [(0.0, 4.0), (-1.0, 2.0), (1.0, 2.0), (0.0, 0.0),
                   (MIN_NORMAL, 1.0), (-1.0, -MIN_NORMAL), (INF, INF),
                   (-INF, INF), (0.0, TINY)]:
        box = s(lo, hi)
        hazed, changed = iv.subnormal_haze(box)
        assert not changed and hazed is box  # the identity, not a copy
        assert (hazed.los[0], hazed.his[0]) == (lo, hi)


def test_kernels_haze_operands_and_results():
    # FTZ face: a subnormal product hulls with 0
    box, made_nan = iv.ieee_mul(s(1e-160, 1e-160), s(1e-160, 1e-160))
    assert not made_nan
    assert (box.los[0], box.his[0]) == (0.0, 1e-160 * 1e-160)
    # DAZ face: a subnormal divisor may read 0 — the 0-in-divisor path
    # (⊤) fires, and subnormal/subnormal is a DAZ-created 0/0 (flagged)
    box, made_nan = iv.ieee_div(s(TINY, TINY), s(TINY, TINY))
    assert made_nan
    assert (box.los[0], box.his[0]) == (-INF, INF)
    # DAZ-amplification: subnormal × inf can be 0 × inf = NaN at runtime
    box, made_nan = iv.ieee_mul(s(1e-310, 2e-308), s(5.0, INF))
    assert made_nan
    assert box.los[0] == 0.0  # the flushed image joined the hull
    # cancellation to a subnormal from NORMAL operands still hazes (FTZ)
    box, made_nan = iv.ieee_sub(s(1.5 * MIN_NORMAL, 1.5 * MIN_NORMAL),
                                s(MIN_NORMAL, MIN_NORMAL))
    assert not made_nan
    assert box.los[0] == 0.0 and box.his[0] == 0.5 * MIN_NORMAL


def test_kernels_unchanged_outside_the_band():
    box, _ = iv.ieee_add(s(0.1, 0.1), s(0.2, 0.2))
    assert box.los[0] == box.his[0] == 0.1 + 0.2  # native point, no widening
    box, _ = iv.ieee_div(s(1.0, 2.0), s(2.0, 4.0))
    assert (box.los[0], box.his[0]) == (0.25, 1.0)
    box, made_nan = iv.ieee_add(s(1e308, 1.7e308), s(1e308, 1.7e308))
    assert (box.los[0], box.his[0]) == (INF, INF) and not made_nan


# --- the audit's seven faces: every one indefinite now ------------------------


def test_face_a_mul_underflow_gt_zero_is_indefinite():
    # F-A: assert(x*x > 0), x = 1e-160 — was VERIFIED; measured jax: False
    q = _one_op_cmp_query(1e-160, 1e-160, "mul", "gt", 0.0)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert make_verdict(q, p, **VERSIONS).status == "UNKNOWN"
    # the ℝ dial still discharges (gradual underflow is the ℝ-consistent
    # reading; real mode is untouched)
    assert propagate(q).obligations[0].status == "discharged"


def test_face_b_mul_underflow_eq_zero_is_indefinite():
    # F-B: assert(x*x == 0), x = 1e-160 — was wrong REFUTED; measured: True
    q = _one_op_cmp_query(1e-160, 1e-160, "mul", "eq", 0.0)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"  # never a definite
    assert propagate(q).obligations[0].status == "violated-over-set"


def test_face_c_subnormal_over_itself_is_daz_created_nan():
    # F-C: assert(x/x == 1), x = 5e-324 — was VERIFIED; measured: NaN, False
    q = _one_op_cmp_query(TINY, TINY, "div", "eq", 1.0)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"


def test_face_d_subnormal_gt_zero_comparison_is_indefinite():
    # F-D: assert(x > 0), x = 5e-324 — no arithmetic at all; DAZ reaches
    # the comparison. Was VERIFIED; measured: False.
    q = _one_op_cmp_query(TINY, TINY, None, "gt", 0.0)
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"  # ℝ side


def test_face_e_distinct_subnormals_eq_is_indefinite():
    # F-E: assert(x == y), x = 5e-324, y = 1e-320 — was wrong REFUTED;
    # measured: True (both flush to 0)
    x, y, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, TINY, TINY),
            any_eqn(y, 1e-320, 1e-320),
            eqn("eq", [x, y], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "violated-over-set"


def test_face_f_underflow_boundary_t_minus_dt_is_indefinite():
    # F-F: assert(t - dt < t), t = minnormal, dt = t/2 — the project's own
    # t+dt>t bug shape at the underflow boundary. Was VERIFIED; measured:
    # False (t - dt == t under flush).
    t, dt, ss, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(t, MIN_NORMAL, MIN_NORMAL),
            any_eqn(dt, MIN_NORMAL / 2, MIN_NORMAL / 2),
            eqn("sub", [t, dt], ss),
            eqn("lt", [ss, t], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"


def test_face_g_exp_flush_gt_zero_is_indefinite():
    # F-G: assert(exp(x) > 0), x = -720 — was VERIFIED; measured jax:
    # exp(-720) = 0.0 (IEEE: 2.03e-313); the 1-ulp libm bracket cannot
    # absorb a flush to 0, the haze can.
    x, e, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, -720.0, -720.0),
            eqn("exp", [x], e),
            eqn("gt", [e, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert propagate(q).obligations[0].status == "discharged"  # ℝ side


# --- the band boundary: just-subnormal hazes, just-normal does not ------------


def test_band_boundary_pair():
    # largest subnormal: hazed → indefinite
    q_sub = _one_op_cmp_query(LARGEST_SUBNORMAL, LARGEST_SUBNORMAL, None, "gt", 0.0)
    assert propagate(q_sub, semantics="ieee").obligations[0].status == "unknown"
    # smallest normal: outside the OPEN band → definite discharge stands
    q_norm = _one_op_cmp_query(MIN_NORMAL, MIN_NORMAL, None, "gt", 0.0)
    assert propagate(q_norm, semantics="ieee").obligations[0].status == "discharged"
    # negative mirror
    q_nsub = _one_op_cmp_query(-LARGEST_SUBNORMAL, -LARGEST_SUBNORMAL, None, "lt", 0.0)
    assert propagate(q_nsub, semantics="ieee").obligations[0].status == "unknown"
    q_nnorm = _one_op_cmp_query(-MIN_NORMAL, -MIN_NORMAL, None, "lt", 0.0)
    assert propagate(q_nnorm, semantics="ieee").obligations[0].status == "discharged"


# --- nothing changes outside the band -----------------------------------------


def test_acceptance_shapes_unchanged_outside_the_band():
    # the t + dt > t acceptance triple at normal magnitudes, re-pinned
    def tdt(dt_lo, dt_hi):
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

    assert propagate(tdt(1e-20, 1e-20), semantics="ieee").obligations[0].status == "violated-over-set"
    assert propagate(tdt(1e-20, 1e-15), semantics="ieee").obligations[0].status == "unknown"
    assert propagate(tdt(3e-16, 1e-15), semantics="ieee").obligations[0].status == "discharged"
    # native point exactness at normal magnitudes still holds
    q = _one_op_cmp_query(0.1, 0.1, "mul", "eq", 0.1 * 0.1)
    assert propagate(q, semantics="ieee").obligations[0].status == "discharged"


def test_zero_containing_declared_boxes_keep_their_certification():
    # [0, 4] touches the band but the haze is the identity (0 already in),
    # so the declaration stays exact and a nonstrict assume still
    # certifies — the fix must not blanket-uncertify ordinary boxes
    x, ap, aout, p2, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(0.9)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("le", [x, lit(0.5)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "violated-over-set"  # certified REFUTED
    assert not any("UNCERTIFIED" in n for n in p.notes)


# --- the strict-assume face of the same indeterminacy -------------------------


def test_band_only_strict_assume_region_stays_uncertified():
    # x declared [0, 5e-324] (haze-identity: contains 0 → still exact);
    # assume(x > 0): the strict region's only member is subnormal — on a
    # DAZ target it reads 0 and the region is EMPTY; under gradual it is
    # {5e-324}. Indeterminate: the definite violation downstream must be
    # withheld, never a REFUTED.
    x, ap, aout, p2, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, TINY),
            eqn("gt", [x, lit(0.0)], ap),
            eqn("stelling_assume", [ap], aout),
            eqn("ge", [x, lit(1.0)], p2),
            eqn("stelling_assert", [p2], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    assert p.obligations[0].status == "unknown"
    assert "WITHHELD" in p.obligations[0].detail
    # control: the same strict assume over a NORMAL-reaching box keeps
    # its certification (a flush-robust witness exists)
    q2 = close(
        [
            any_eqn(var(0), 0.0, 4.0),
            eqn("gt", [var(0), lit(0.0)], var(1, BOOL)),
            eqn("stelling_assume", [var(1, BOOL)], var(2, BOOL)),
            eqn("ge", [var(0), lit(10.0)], var(3, BOOL)),
            eqn("stelling_assert", [var(3, BOOL)], var(4, BOOL)),
        ],
        [var(4, BOOL)],
    )
    p2r = propagate(q2, semantics="ieee")
    assert p2r.obligations[0].status == "violated-over-set"


def test_band_constant_assume_bound_drops_with_the_true_reason():
    # a subnormal-band literal bound is hazed to a non-point: the assume
    # drops, and the disclosed reason names the band shape — never
    # "both sides vary" (the audit-F6 discipline)
    x, pred, aout = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 4.0),
            eqn("ge", [x, lit(1e-320)], pred),
            eqn("stelling_assume", [pred], aout),
        ],
        [aout],
    )
    p = propagate(q, semantics="ieee")
    assert p.coverage.inert == 1 and p.coverage.constrained == 0
    note = next(n for n in p.notes if "DROPPED" in n)
    assert "subnormal-band constant" in note
    assert "both sides vary" not in note
    # real mode still narrows on the same bound (ℝ semantics, untouched)
    pr = propagate(q)
    assert pr.coverage.constrained == 1


# --- the stamped disclosure ---------------------------------------------------


def test_subnormal_indeterminacy_assumption_rides_in_ieee_stamps_only():
    q = _one_op_cmp_query(1.0, 2.0, "mul", "le", 100.0)
    p = propagate(q, semantics="ieee")
    assert iv.SUBNORMAL_INDETERMINACY_ASSUMPTION in p.assumptions
    v = make_verdict(q, p, **VERSIONS)
    assert any("subnormal indeterminacy" in a for a in v.stamp.assumptions)
    assert any("jax 0.11.0 CPU" in a for a in v.stamp.assumptions)
    # real stamps never carry it
    pr = propagate(q)
    assert iv.SUBNORMAL_INDETERMINACY_ASSUMPTION not in pr.assumptions
    vr = make_verdict(q, pr, **VERSIONS)
    assert not any("subnormal" in a for a in vr.stamp.assumptions)
