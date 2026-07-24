# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The censused sqrt transfer row — hand-built IR, no jax.

sqrt is a correctly-rounded IEEE basic op (tier ``sound``, no libm demotion)
whose domain ``arg >= 0`` is the obligation: the real transfer is monotone,
outward-rounded, floors the lower endpoint at 0, and declines the below-0
box (the ``pow`` out-of-domain posture); the ieee transfer brackets the
float root exactly and routes a negative argument's NaN into the flag;
emission declines it as nonlinear (outside the supported set, exactly as
``exp``/``pow`` do). The fidelity gauge closes the row with a mutation
battery whose residual is empty.
"""

from __future__ import annotations

from fractions import Fraction
from math import inf, nextafter, sqrt as msqrt

import pytest

from stelling import fidelity
from stelling import interval as iv
from stelling import ir
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    slice_unknown_obligations,
)
from stelling.propagate import (
    IEEE_TRANSFERS,
    TRANSFERS,
    _INT_COMPUTING,
    interval_env,
    propagate,
)

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
I32 = ir.Aval(kind="ShapedArray", shape=(), dtype="int32")


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


def _sqrt_query(xb, bound, cmp, *, dtype=F64, semantics="real"):
    x, s, pred, out = var(0, dtype), var(1, dtype), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, *xb),
            eqn("sqrt", [x], s),
            eqn(cmp, [s, lit(bound, dtype)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    return propagate(q, semantics=semantics)


# --- registration / census ---------------------------------------------------


def test_sqrt_registered_row_and_tiers():
    assert "sqrt" in TRANSFERS and TRANSFERS["sqrt"][1] == "sound"
    assert "sqrt" in IEEE_TRANSFERS and IEEE_TRANSFERS["sqrt"][1] == "exact"
    # computing (it produces a new value on floats); the census probe runs it
    # at every integer boundary and it declines there via the float-only guard
    assert "sqrt" in _INT_COMPUTING


# --- real transfer: soundness, monotonicity, the lower floor -----------------


def test_sqrt_discharges_no_libm_assumption():
    p = _sqrt_query((4.0, 9.0), 3.001, "le")  # sqrt[4,9] = [2,3] <= 3.001
    assert p.obligations[0].status == "discharged"
    assert ("sqrt", "sound") in p.transfers_used
    # sqrt is correctly rounded, NOT a faithfully-rounded libm call: no libm
    # assumption may ride into the stamp (the exp/pow distinction)
    assert not any("sqrt" in a and "libm" in a for a in p.assumptions)


def test_sqrt_definite_false_direction():
    # sqrt[4,9] = [2,3]; "< 1" holds nowhere over the box
    p = _sqrt_query((4.0, 9.0), 1.0, "lt")
    assert p.obligations[0].status == "violated-over-set"


def test_sqrt_lower_floor_gives_nonnegativity():
    # sqrt[0,16] lower endpoint is EXACTLY 0 (floored), so "sqrt >= 0"
    # discharges — the non-negativity fact is produced, not bumped away
    p = _sqrt_query((0.0, 16.0), 0.0, "ge")
    assert p.obligations[0].status == "discharged"


def test_sqrt_endpoints_outward_and_exact_zero():
    b = iv.sqrt(iv.from_bounds((), 0.0, 16.0))
    assert b.los[0] == 0.0  # sqrt(0) exact, floored
    assert Fraction(b.his[0]) ** 2 >= Fraction(16)  # outward upper bracket
    inf_hi = iv.sqrt(iv.from_bounds((), 1.0, inf))
    assert inf_hi.his[0] == inf  # sqrt(inf) = inf


# --- real transfer: the domain obligation (arg >= 0) -------------------------


@pytest.mark.parametrize("xb", [(-1.0, 4.0), (-3.0, -1.0), (-0.5, 100.0)])
def test_sqrt_negative_domain_declines_not_crashes(xb):
    p = _sqrt_query(xb, 100.0, "le")  # must not raise
    assert p.obligations[0].status == "unknown"
    assert any("sqrt" in n and "declined" in n for n in p.notes)
    assert p.coverage.unknown == 1
    assert "sqrt" not in dict(p.transfers_used)  # no tier claimed on a decline


def test_sqrt_zero_lower_bound_is_in_domain():
    # the domain is CLOSED at 0: lo == 0 does NOT decline
    p = _sqrt_query((0.0, 4.0), 2.001, "le")
    assert p.obligations[0].status == "discharged"


def test_sqrt_integer_operand_declines_float_only():
    # jax's sqrt is float-only; an integer operand declines rather than
    # modelling a wrapping integer as a real
    p = _sqrt_query((1.0, 4.0), 100.0, "le", dtype=I32)
    assert p.obligations[0].status == "unknown"
    assert any("sqrt" in n and "declined" in n for n in p.notes)
    assert "sqrt" not in dict(p.transfers_used)


# --- ieee transfer: correctly-rounded root, negative arg -> maybe-NaN --------


def test_sqrt_ieee_discharges_clean_box():
    p = _sqrt_query((4.0, 9.0), 3.001, "le", semantics="ieee")
    assert p.semantics == "ieee"
    assert p.obligations[0].status == "discharged"
    assert ("sqrt", "exact") in p.transfers_used


def test_sqrt_ieee_negative_arg_blocks_discharge():
    # sqrt[-1,4] under ieee: the non-negative part brackets [0,2] but a
    # negative argument is NaN (maybe_nan), and a comparison over a maybe-NaN
    # operand is never definitely true — so the otherwise-true bound does NOT
    # discharge (sound: an over-approximated flag only blocks discharges)
    p = _sqrt_query((-1.0, 4.0), 100.0, "le", semantics="ieee")
    assert p.obligations[0].status == "unknown"


def test_ieee_sqrt_kernel_routes_nan_and_is_exact_on_squares():
    box, nan = iv.ieee_sqrt(iv.from_bounds((), -1.0, 4.0))
    assert (box.los[0], box.his[0]) == (0.0, 2.0) and nan is True
    box2, nan2 = iv.ieee_sqrt(iv.from_bounds((), 4.0, 4.0))
    assert box2.los[0] == box2.his[0] == 2.0 and nan2 is False  # exact root
    box3, nan3 = iv.ieee_sqrt(iv.from_bounds((), -9.0, -1.0))
    assert (box3.los[0], box3.his[0]) == (-inf, inf) and nan3 is True


# --- emission: nonlinear, declines like exp/pow (outside the supported set) ---


def test_sqrt_emission_declines_outside_supported_set():
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 4.0),
            eqn("sqrt", [x], s),
            eqn("lt", [s, lit(1.5)], pred),  # straddles [1,2]: unknown
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, DeclinedObligation)
    assert "'sqrt'" in item.reason
    assert "supported emission set" in item.reason
    assert not isinstance(item, ObligationSlice)


# --- fidelity gauge: measured discriminating power, residual empty -----------


def _baseline(lo, hi):
    try:
        b = iv.sqrt(iv.from_bounds((), lo, hi))
    except iv.IntervalError:
        return ("raise",)
    return (b.los[0], b.his[0])


def _inward(lo, hi):  # bumps INWARD (narrower than the true root bracket)
    if lo < 0.0:
        return ("raise",)
    return (
        nextafter(msqrt(lo), inf),
        nextafter(msqrt(hi), -inf) if hi != inf else inf,
    )


def _no_outward(lo, hi):  # native math.sqrt, no outward bump
    if lo < 0.0:
        return ("raise",)
    return (msqrt(lo), msqrt(hi) if hi != inf else inf)


def _no_lower_floor(lo, hi):  # drops the max(0, .) floor at 0
    if lo < 0.0:
        return ("raise",)
    rlo = nextafter(msqrt(lo), -inf) if lo != 0.0 else nextafter(0.0, -inf)
    return (rlo, nextafter(msqrt(hi), inf) if hi != inf else inf)


def _domain_returns_value(lo, hi):  # wrong out-of-domain: a value, not a raise
    d = lo if lo > 0.0 else 0.0
    return (
        max(0.0, nextafter(msqrt(d), -inf)),
        nextafter(msqrt(hi), inf) if hi != inf else inf,
    )


def _monotone_swap(lo, hi):  # swaps the endpoints (monotonicity break)
    r = _baseline(lo, hi)
    return r if r == ("raise",) else (r[1], r[0])


def _constant_zero(lo, hi):  # collapses the image to {0}
    return (0.0, 0.0)


_IN_DOMAIN = [(2.0, 9.0), (4.0, 16.0), (0.5, 2.0), (2.0, 2.0), (0.0, 16.0)]
_NEG = (-1.0, 4.0)
_ZERO = (0.0, 16.0)


def _in_domain_pairs(sub):
    out = []
    for lo, hi in _IN_DOMAIN:
        r = sub(lo, hi)
        if r == ("raise",):
            return None  # declined an in-domain probe: the gate fails
        out.append((lo, hi, r[0], r[1]))
    return out


def _lower_sound(sub):
    got = _in_domain_pairs(sub)
    if not got:
        return False
    return all(
        rlo >= 0.0 and Fraction(rlo) ** 2 <= Fraction(lo)
        for lo, hi, rlo, rhi in got
    )


def _upper_sound(sub):
    got = _in_domain_pairs(sub)
    if not got:
        return False
    return all(
        rhi == inf or Fraction(rhi) ** 2 >= Fraction(hi)
        for lo, hi, rlo, rhi in got
    )


def _ordered(sub):
    got = _in_domain_pairs(sub)
    if not got:
        return False
    return all(rlo <= rhi for _, _, rlo, rhi in got)


def _domain_raises(sub):
    return sub(*_NEG) == ("raise",)


def _lower_floor(sub):
    r = sub(*_ZERO)
    return r != ("raise",) and r[0] == 0.0


_GATES = {
    "lower_sound": _lower_sound,
    "upper_sound": _upper_sound,
    "ordered": _ordered,
    "domain_raises": _domain_raises,
    "lower_floor": _lower_floor,
}

_MUTATIONS = {
    "inward_rounding": _inward,
    "no_outward_bump": _no_outward,
    "no_lower_floor": _no_lower_floor,
    "domain_returns_value": _domain_returns_value,
    "monotone_swap": _monotone_swap,
    "constant_zero": _constant_zero,
}


def test_sqrt_fidelity_gauge_residual_empty():
    report = fidelity.gauge(_baseline, _GATES, _MUTATIONS, residual={})
    # every mutation is caught by at least one gate; nothing survives
    assert report.residual == ()
    caught = dict(report.caught_by)
    for name in _MUTATIONS:
        assert caught[name], f"{name} survived every gate — a measured hole"
    # the specific coverage the work order names
    assert "lower_sound" in caught["inward_rounding"]
    assert "domain_raises" in caught["domain_returns_value"]
    assert "ordered" in caught["monotone_swap"]
    assert "lower_floor" in caught["no_lower_floor"]
