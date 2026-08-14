# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Audit 0.2.0 S10 and M16: the sign of an IEEE zero, and `mul`'s bump.

**S10 (FALSE VERIFIED, all four formats).** `ieee_div`/`ieee_div_fmt` used to
tighten a divisor box touching zero at exactly one boundary: `[lo, 0]` with
`lo < 0` was read as *"the divisor approaches 0 from below"*, so `a/b -> +inf`
for `a <= 0` and the returned box excluded `-inf`. Under IEEE the divisor does
not APPROACH zero, it IS zero at that endpoint, and the sign of `x/0` comes
from `sign(x) XOR signbit(0)` — the sign bit of the zero. `+0.0 == 0.0`, so
`+0.0` is a value of `[lo, 0]`, and there `a/b` is `-inf`: a value of the
program that the box did not contain.

An interval endpoint cannot carry a sign bit, so **which boundary is zero is
not enough information to make the tightening**, and no test on the endpoints'
positions can repair it. The tightening is withdrawn under IEEE; a
zero-containing divisor divides to top, which is what v0.1.0 returned.

**The real-mode `boundary_div` keeps the tightening and is NOT wrong for the
same reason**, which the tests below pin as a DIFFERENCE rather than leaving
to the next reader's assumption: R has one zero and `a/0` is undefined there,
so a box need only cover `b != 0`, and `[2, inf)` does.

**M16.** `mul` was the only arithmetic transfer with no exact-rational path.
It bumped every endpoint outward unconditionally, so `[2,3]x[2,3]` boxed to
`[3.9999999999999996, 9.000000000000002]` for an image that is exactly
`[4, 9]`, and the exactly-zero corner of `[0,4]x[0,4]` bumped to `-5e-324` —
BELOW ZERO. That defeats `reduce_sum`'s nonnegative clamp, so a sum of
squares written `x*x` became a true straddle and the division that consumed
it declined, while `x**2` and `jnp.square(x)` verified: one real property,
three spellings, two verdicts. `mul` now takes the same `_exactable`/
`Fraction` route `add` and `div` already had.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import (
    _FLOAT_FORMATS,
    _ieee_format_min_normal,
    _ieee_round_box,
    propagate,
)

INF = math.inf
FMAX = 1.7976931348623157e308
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")
FORMAT_NAMES = ("float64", "float32", "float16", "bfloat16")


def s(lo, hi):
    return iv.from_bounds((), lo, hi)


def av(dtype):
    return ir.Aval(kind="ShapedArray", shape=(), dtype=dtype)


def var(i, a=None):
    return ir.Var(id=i, aval=a or av("float64"))


def lit(v, a=None):
    return ir.Literal(val=v, aval=a or av("float64"))


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
        jaxpr=ir.Jaxpr(invars=(), constvars=(), eqns=tuple(eqns), outvars=tuple(outvars)),
        consts=(),
    )


def ieee_div_any_format(a, b, name):
    """`ieee_div` for float64, `ieee_div_fmt` + the propagate layer's outward
    format rounding for the rest — the composition `_ieee_arith` performs."""
    fmt = _FLOAT_FORMATS[name]
    if name == "float64":
        return iv.ieee_div(a, b)
    box, made_nan = iv.ieee_div_fmt(a, b, _ieee_format_min_normal(fmt))
    return _ieee_round_box(box, fmt), made_nan


# =========================================================================
# S10 — a zero-containing divisor divides to top under IEEE
# =========================================================================

# Every divisor box that CONTAINS zero, in the shapes the withdrawn branch
# split on: zero at the upper boundary, at the lower boundary, the point at
# zero, and a true straddle. All four must be top now; before the fix the
# first two were tightened to a one-signed infinity.
ZERO_TOUCHING_DIVISORS = [
    (-1.0, 0.0),      # [lo, 0]  — S10's shape
    (0.0, 1.0),       # [0, hi]  — the mirror
    (-0.0, 1.0),      # a NEGATIVE zero as the endpoint: still contains zero
    (-1.0, -0.0),
    (0.0, 0.0),
    (-1.0, 1.0),
    (-5e-324, 0.0),
    (0.0, 5e-324),
]

DIVIDENDS = [(-2.0, -2.0), (2.0, 2.0), (-5.0, -1.0), (1.0, 5.0), (-1.0, 1.0)]


@pytest.mark.parametrize("name", FORMAT_NAMES)
@pytest.mark.parametrize("blo,bhi", ZERO_TOUCHING_DIVISORS)
@pytest.mark.parametrize("alo,ahi", DIVIDENDS)
def test_ieee_div_zero_containing_divisor_is_top(name, blo, bhi, alo, ahi):
    """No case split on WHERE the zero sits: containing zero is the whole
    condition, in every format. Before the fix, `[-1, 0]` with a negative
    dividend returned `[2.0, inf]`."""
    box, _ = ieee_div_any_format(s(alo, ahi), s(blo, bhi), name)
    assert (box.los[0], box.his[0]) == (-INF, INF), (
        f"{name}: [{alo},{ahi}] / [{blo},{bhi}] -> "
        f"[{box.los[0]}, {box.his[0]}], expected top"
    )


@pytest.mark.parametrize("name", FORMAT_NAMES)
def test_ieee_div_box_contains_the_infinity_the_zeros_sign_produces(name):
    """The measured escape, stated as containment.

    `-2.0 / +0.0 = -inf` and `-2.0 / -0.0 = +inf`; both zeros are values of
    `[-1, 0]`, so both infinities are values of the quotient. The old box
    `[2.0, inf]` held one of them.
    """
    box, _ = ieee_div_any_format(s(-2.0, -2.0), s(-1.0, 0.0), name)
    assert box.los[0] == -INF, f"{name}: box misses -inf (= -2.0 / +0.0)"
    assert box.his[0] == INF, f"{name}: box misses +inf (= -2.0 / -0.0)"

    mirror, _ = ieee_div_any_format(s(2.0, 2.0), s(0.0, 1.0), name)
    assert mirror.los[0] == -INF, f"{name}: mirror misses -inf (= 2.0 / -0.0)"
    assert mirror.his[0] == INF, f"{name}: mirror misses +inf (= 2.0 / +0.0)"


@pytest.mark.parametrize("dtype", FORMAT_NAMES)
def test_s10_harness_no_longer_discharges_in_any_format(dtype):
    """End to end through `propagate`, the audit's harness: `a = [-2,-2]`,
    `x = [-1, 0]`, `assert(a/x > 0)`. jax computes `-inf` at `x = +0.0`, so a
    discharge here is a FALSE VERIFIED. Was `discharged` in all four."""
    A = av(dtype)
    a, x, q, pred, out = var(0, A), var(1, A), var(2, A), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, -2.0, -2.0, dtype=dtype),
            any_eqn(x, -1.0, 0.0, dtype=dtype),
            eqn("div", [a, x], q),
            eqn("gt", [q, lit(0.0, A)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query, semantics="ieee")
    assert p.obligations[0].status != "discharged", (
        f"{dtype}: FALSE VERIFIED — jax gives -inf at x=+0.0. "
        f"detail: {p.obligations[0].detail}"
    )


def test_ieee_div_does_not_raise_on_an_infinite_dividend_over_a_zero_edge():
    """The withdrawn branch could also CRASH the analysis, not only mislead it.

    `[-inf, -inf] / [-inf, 0]` took the `bhi == 0` arm and computed
    `ahi / blo = -inf / -inf = NaN`, which `IntervalArray.__post_init__`
    rejects — an `IntervalError` out of a kernel whose contract is to degrade.
    Found by the containment sweep, which could not even finish against the
    pre-fix tree. Not a separate repair: returning top before any endpoint
    arithmetic happens is what removes it.
    """
    for name in FORMAT_NAMES:
        box, made_nan = ieee_div_any_format(s(-INF, -INF), s(-INF, 0.0), name)
        assert (box.los[0], box.his[0]) == (-INF, INF)
        assert made_nan is True  # inf/inf is a real NaN class here


@pytest.mark.parametrize("name", FORMAT_NAMES)
def test_ieee_div_still_tightens_when_the_divisor_excludes_zero(name):
    """The withdrawal is confined to zero-containing divisors. A divisor
    bounded away from zero still divides to a bounded box — otherwise the fix
    would have cost the whole primitive rather than one branch."""
    box, made_nan = ieee_div_any_format(s(1.0, 2.0), s(2.0, 4.0), name)
    assert box.los[0] >= 0.2 and box.his[0] <= 1.1, (
        f"{name}: [1,2]/[2,4] -> [{box.los[0]}, {box.his[0]}]"
    )
    assert made_nan is False


# =========================================================================
# S10 — the real-mode kernel is different ON PURPOSE
# =========================================================================


def test_real_boundary_div_covers_every_nonzero_real_in_the_divisor_box():
    """The claim that licenses the difference, verified rather than asserted.

    Over R there is ONE zero and `a/0` is undefined, so `boundary_div`'s
    obligation is to cover `a/b` for every real `b != 0` in the box. Checked
    in exact rational arithmetic at values crowding the zero endpoint, where
    the quotient diverges.
    """
    cases = [
        (-2.0, -2.0, -1.0, 0.0),
        (2.0, 2.0, 0.0, 1.0),
        (-5.0, -1.0, 0.0, 2.0),
        (2.0, 4.0, -3.0, 0.0),
        (1.0, 1.0, 0.0, 32.0),
    ]
    checked = 0
    for alo, ahi, blo, bhi in cases:
        r = iv.boundary_div(s(alo, ahi), s(blo, bhi))
        lo, hi = r.los[0], r.his[0]
        flo = Fraction(lo) if math.isfinite(lo) else None
        fhi = Fraction(hi) if math.isfinite(hi) else None
        xs = [Fraction(alo), Fraction(ahi), (Fraction(alo) + Fraction(ahi)) / 2]
        span = Fraction(bhi) - Fraction(blo)
        ys = [Fraction(blo), Fraction(bhi)]
        for k in (2, 10, 10**3, 10**9, 10**30, 10**300):
            ys += [Fraction(blo) + span / k, Fraction(bhi) - span / k]
        for x in xs:
            for y in ys:
                if y == 0 or not (Fraction(blo) <= y <= Fraction(bhi)):
                    continue
                q = x / y
                checked += 1
                assert flo is None or q >= flo, (
                    f"boundary_div([{alo},{ahi}],[{blo},{bhi}]) -> [{lo},{hi}] "
                    f"misses {float(x)}/{float(y)}"
                )
                assert fhi is None or q <= fhi, (
                    f"boundary_div([{alo},{ahi}],[{blo},{bhi}]) -> [{lo},{hi}] "
                    f"misses {float(x)}/{float(y)}"
                )
    assert checked > 100


def test_real_and_ieee_division_disagree_at_a_zero_boundary_on_purpose():
    """**Read this before making the two kernels agree.**

    Same operands, two arithmetics, two answers, and both are right:

    * `boundary_div([-2,-2], [-1,0]) = [2, inf)` — over R the divisor box
      holds one zero, `-2/0` is undefined, and every real `b != 0` in
      `[-1, 0]` gives `-2/b >= 2`. Nothing is excluded that exists.
    * `ieee_div([-2,-2], [-1,0]) = top` — over the floats the box holds TWO
      zeros, `-2/+0.0 = -inf` and `-2/-0.0 = +inf` are both values of the
      program, and no box narrower than top contains both.

    The difference is the arithmetic, not an inconsistency: signed zero is a
    float fact with no real counterpart. Making the ieee kernel agree with
    the real one re-opens audit 0.2.0 S10 (FALSE VERIFIED in four formats);
    making the real one agree with the ieee kernel would give up
    boundary-aware division for no soundness gain at all.
    """
    real = iv.boundary_div(s(-2.0, -2.0), s(-1.0, 0.0))
    assert (real.los[0], real.his[0]) == (2.0, INF)

    ieee, _ = iv.ieee_div(s(-2.0, -2.0), s(-1.0, 0.0))
    assert (ieee.los[0], ieee.his[0]) == (-INF, INF)

    # And the real box is NOT required to contain what IEEE computes at a
    # signed zero: that value has no real preimage in the box.
    assert real.los[0] != -INF


def test_real_mode_div_transfer_still_uses_boundary_div():
    """The real-mode dispatch is untouched: the same harness that must not
    discharge under ieee still discharges under R, and that is correct
    there."""
    a, x, q, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    query = close(
        [
            any_eqn(a, -2.0, -2.0),
            any_eqn(x, -1.0, 0.0),
            eqn("div", [a, x], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "discharged", (
        f"real-mode boundary division regressed: {p.obligations[0].detail}"
    )


# =========================================================================
# S10 — a standing containment sweep, signed zeros distinguished
# =========================================================================

_SWEEP_POOL = [
    -INF, -1e300, -1.0, -5e-324, -0.0, 0.0, 5e-324, 1.0, 1e300, INF,
]
_SWEEP_BOXES = [
    (lo, hi) for lo, hi in itertools.product(_SWEEP_POOL, repeat=2) if lo <= hi
]


def _float_points(lo, hi):
    """Values of the box, with `+0.0` and `-0.0` kept APART. They compare
    equal, so a generator that dedups on `==` sees one zero and never
    produces the input that made S10 visible."""
    cands = [lo, hi, -0.0, 0.0, -1.0, 1.0, -1e300, 1e300]
    out, seen = [], set()
    for c in cands:
        if not (lo <= c <= hi):
            continue
        key = (c, math.copysign(1.0, c)) if c == 0.0 else (c, 0.0)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


@pytest.mark.parametrize("name", FORMAT_NAMES)
def test_ieee_div_containment_sweep_over_adversarial_boxes(name):
    """Every returned box must contain every quotient the format can compute
    at points of the operand boxes — infinities and signed zeros included."""
    checked = 0
    for (alo, ahi) in _SWEEP_BOXES:
        for (blo, bhi) in _SWEEP_BOXES:
            box, made_nan = ieee_div_any_format(s(alo, ahi), s(blo, bhi), name)
            for x in _float_points(alo, ahi):
                for y in _float_points(blo, bhi):
                    v = _ieee_quotient(x, y)
                    checked += 1
                    if v is None:  # NaN
                        assert made_nan, (
                            f"{name}: NaN at {x!r}/{y!r} but made_nan=False"
                        )
                        continue
                    assert box.los[0] <= v <= box.his[0], (
                        f"{name}: [{alo},{ahi}]/[{blo},{bhi}] -> "
                        f"[{box.los[0]},{box.his[0]}] misses {v!r} at "
                        f"x={x!r} y={y!r}"
                    )
    assert checked > 5000


def _ieee_quotient(x, y):
    """`x / y` with IEEE's answers, in pure Python: returns None for NaN.

    Python raises on division by zero instead of returning an infinity, so
    the zero cases — the whole subject — are supplied from the standard:
    `+-finite/+-0 = +-inf` by XOR of the sign bits, `0/0` and `inf/inf` NaN.
    """
    xz, yz = x == 0.0, y == 0.0
    xinf, yinf = math.isinf(x), math.isinf(y)
    if (xz and yz) or (xinf and yinf):
        return None
    sign = math.copysign(1.0, x) * math.copysign(1.0, y)
    if yz:
        return math.copysign(INF, sign)
    if xinf:
        return math.copysign(INF, sign)
    if yinf:
        return math.copysign(0.0, sign)
    return x / y


# =========================================================================
# M16 — `mul` takes the exact-rational route its siblings already had
# =========================================================================


def test_mul_is_exact_when_the_corner_products_are_representable():
    """`[2,3] x [2,3]` has the exact image `[4, 9]`; the transfer used to
    return `[3.9999999999999996, 9.000000000000002]`."""
    r = iv.mul(s(2.0, 3.0), s(2.0, 3.0))
    assert (r.los[0], r.his[0]) == (4.0, 9.0)


def test_mul_zero_corner_no_longer_bumps_below_zero():
    """The consequence that mattered: `[0,4] x [0,4]` bumped its exactly-zero
    lower corner to `-5e-324`, and a negative floor is what defeats
    `reduce_sum`'s nonnegative clamp."""
    r = iv.mul(s(0.0, 4.0), s(0.0, 4.0))
    assert (r.los[0], r.his[0]) == (0.0, 16.0)
    assert not math.copysign(1.0, r.los[0]) < 0.0


def test_mul_now_matches_its_siblings_on_the_same_operands():
    """`add` and `div` return the exact endpoint when it is representable;
    `mul` was the only arithmetic transfer that did not."""
    a = s(2.0, 3.0)
    assert (iv.add(a, a).los[0], iv.add(a, a).his[0]) == (4.0, 6.0)
    assert (iv.div(a, a).los[0], iv.div(a, a).his[0]) == (2.0 / 3.0, 1.5)
    assert (iv.mul(a, a).los[0], iv.mul(a, a).his[0]) == (4.0, 9.0)


def test_reduce_sum_of_products_keeps_its_nonnegative_floor():
    """`sum(x*x)` over `x in [0,4]^2`: exactly `[0, 32]`. With the bump it was
    `[-1e-323, 32.00000000000001]`, a TRUE straddle — which is why the
    division that consumed it declined instead of reaching `boundary_div`."""
    X = iv.from_bounds((2,), 0.0, 4.0)
    r = iv.reduce_sum(iv.mul(X, X), (0,))
    assert (r.los[0], r.his[0]) == (0.0, 32.0)
    assert r.los == iv.reduce_sum(iv.integer_pow(X, 2), (0,)).los


def test_mul_exact_route_is_confined_to_finite_endpoints():
    """An infinite endpoint keeps the unconditional bump and the
    closed-interval `0 * +-inf = 0` convention: `Fraction(inf)` raises, and
    the convention is an endpoint rule rather than real arithmetic. Same
    confinement `add` and `div` use."""
    r = iv.mul(s(0.0, 0.0), s(1.0, INF))
    assert r.los[0] <= 0.0 <= r.his[0]
    r2 = iv.mul(s(2.0, INF), s(3.0, 4.0))
    assert r2.his[0] == INF
    assert r2.los[0] < 6.0  # bumped, not exact — the infinite-endpoint route


def test_mul_saturates_outward_at_overflow():
    """The exact product of two `1e300`s is outside binary64. Saturating
    outward is the sound answer and is what `_exact_down`/`_exact_up` already
    did for `add`."""
    r = iv.mul(s(1e300, 1e300), s(1e300, 1e300))
    assert r.his[0] == INF
    assert r.los[0] == FMAX


def test_mul_containment_and_exactness_on_a_battery():
    """Containment against the exact rational image, plus the sharper claim
    that the box IS the image whenever both extrema are representable."""
    pool = [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 4.0, 8.0]
    boxes = [(lo, hi) for lo, hi in itertools.product(pool, repeat=2) if lo <= hi]
    for (alo, ahi) in boxes:
        for (blo, bhi) in boxes:
            r = iv.mul(s(alo, ahi), s(blo, bhi))
            corners = [
                Fraction(x) * Fraction(y)
                for x in (alo, ahi)
                for y in (blo, bhi)
            ]
            lo_exact, hi_exact = min(corners), max(corners)
            assert Fraction(r.los[0]) <= lo_exact
            assert Fraction(r.his[0]) >= hi_exact
            # every endpoint here is a small dyadic, so the image endpoints
            # are representable and the box must be exactly the image
            assert Fraction(r.los[0]) == lo_exact
            assert Fraction(r.his[0]) == hi_exact


def test_ieee_mul_deliberately_keeps_the_native_float_product():
    """`ieee_mul` does NOT take the exact route, and this pins why.

    Under ieee the value the program has IS `fl(x*y)`; the native corner
    product already IS that value, so routing through `Fraction` would round
    a REAL product outward and manufacture slack where there is none. At
    overflow it would also be wrong in kind: two `1e300`s multiply to `inf`
    on the target, so the true image is the point `[inf, inf]`, while the
    exact route reports `[FMAX, inf]` and names a value the program cannot
    produce — which is exactly what real-mode `mul` correctly returns above.
    """
    box, made_nan = iv.ieee_mul(s(1e300, 1e300), s(1e300, 1e300))
    assert (box.los[0], box.his[0]) == (INF, INF)
    assert made_nan is False
    assert iv.mul(s(1e300, 1e300), s(1e300, 1e300)).los[0] == FMAX

    # And the ordinary case stays the float point, not a rational bracket.
    point, _ = iv.ieee_mul(s(0.1, 0.1), s(0.1, 0.1))
    assert point.los[0] == point.his[0] == 0.1 * 0.1


def test_mul_transfer_end_to_end_reaches_boundary_division():
    """The shape the 0.2.0 boundary-division row was added for: a
    sum-of-squares residual in the denominator. `sum(x*x)` for `x in [0,4]`
    must floor at 0, so the divisor has zero at ONE boundary and
    `boundary_div` decides `1/sum > 0`."""
    x, sq, tot, q, pred, out = (
        var(0, av("float64")),
        var(1),
        var(2),
        var(3),
        var(4, BOOL),
        var(5, BOOL),
    )
    x = ir.Var(id=0, aval=ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64"))
    sq = ir.Var(id=1, aval=ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64"))
    query = close(
        [
            ir.JaxprEqn(
                primitive="stelling_any",
                invars=(),
                outvars=(x,),
                params=(("shape", (2,)), ("dtype", "float64"), ("lo", 0.0), ("hi", 4.0)),
            ),
            eqn("mul", [x, x], sq),
            eqn("reduce_sum", [sq], tot, params=(("axes", (0,)),)),
            eqn("div", [lit(1.0), tot], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert p.obligations[0].status == "discharged", (
        f"the `x*x` spelling still cannot reach boundary division: "
        f"{p.obligations[0].detail}; notes {p.notes}"
    )


# =========================================================================
# The traced faces — jax present
# =========================================================================


def test_s10_jax_computes_the_infinity_the_old_box_excluded():
    """The measurement the finding rests on, kept as a test: in every format
    jax evaluates `-2.0 / x` at `x = +0.0` to `-inf`, so a VERIFIED for
    `a/x > 0` over `x in [-1, 0]` is false about the running program."""
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_
    from stelling.preconditions import check

    dtypes = {
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }
    for name, dt in dtypes.items():
        y = jnp.asarray(-2.0, dtype=dt) / jnp.asarray(0.0, dtype=dt)
        assert float(y) == -INF, f"{name}: expected -inf at +0.0, got {y}"

        def harness(_dt=name):
            a = any_array((), _dt, (-2.0, -2.0))
            x = any_array((), _dt, (-1.0, 0.0))
            z = any_array((), _dt, (0.0, 0.0))
            q = a / x
            assert_(q > z)
            return q

        v = check(harness, vacuity_mode="inputs-only", semantics="ieee")
        assert v.status != "VERIFIED", (
            f"{name}: FALSE VERIFIED — jax gives -inf at x=+0.0"
        )


def test_three_spellings_of_squared_reach_the_same_verdict():
    """`x*x`, `x**2` and `jnp.square(x)` are the same real property. The
    `mul` bump used to decide between them: `via_mul` came back UNKNOWN with
    a decline recommending `assume(divisor > 0)` — which the caller had
    already effectively done on the inputs."""
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_, assume
    from stelling.preconditions import check

    def via_mul():
        x = any_array((2,), jnp.float64, (0.0, 4.0))
        assume(x > 0.0)
        return assert_(1.0 / jnp.sum(x * x) > 0.0)

    def via_ipow():
        x = any_array((2,), jnp.float64, (0.0, 4.0))
        assume(x > 0.0)
        return assert_(1.0 / jnp.sum(x**2) > 0.0)

    def via_square():
        x = any_array((2,), jnp.float64, (0.0, 4.0))
        assume(x > 0.0)
        return assert_(1.0 / jnp.sum(jnp.square(x)) > 0.0)

    got = {
        h.__name__: check(h, vacuity_mode="inputs-only").status
        for h in (via_mul, via_ipow, via_square)
    }
    assert set(got.values()) == {"VERIFIED"}, got
