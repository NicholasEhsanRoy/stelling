# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Audit 0.2.0 S10, M16 and the B5 follow-ups: the sign of an IEEE zero,
`mul`'s bump, and what licenses dropping a divisor's zero.

**B5-1 (FALSE VERIFIED, real mode), and it is what M16 made reachable.**
With `mul` exact, `sum(x*x)` floors at exactly 0, so `sum(x*x) - c` became a
ONE-SIDED BOUNDARY divisor where it used to be a true straddle — and the
one-sided arm was the only one of `div`'s four zero-containing shapes that
did not decline. It called `boundary_div`, which silently drops `b = 0`.
The program does not: `1/(sum(x*x) - 8)` over `x in [0,2]^2` DISCHARGED
`q <= -0.125` while jax at `x = [2,2]` returns `+inf`.

The kernel is sound over `b != 0` (the sweep below re-measures that). What
was missing is the premise: `boundary_div` is reachable only when a strict
`assume` certifies the divisor is nonzero, carried to the division through
`mul`/`square`/`integer_pow`/`reduce_sum`/`dot_general`. Everything else
declines with the other three shapes.

**B5-2.** `dot_general` kept an inlined copy of `mul`'s corner rule that M16
did not convert; the two are one function now.

**B5-3.** The `NaN endpoint` raise removed from `ieee_div` was still live in
`boundary_div`, where it surfaced as a user-facing decline reason.

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
import importlib
import math
from fractions import Fraction

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import (
    _FLOAT_FORMATS,
    _ieee_format_min_normal,
    _ieee_format_min_positive,
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


@pytest.fixture
def _x64():
    """float64 for one test, RESTORED afterwards — the house pattern.

    Two tests below used to call `jax.config.update("jax_enable_x64", True)`
    inline with no restore. x64 is process-global in jax, so an unrestored
    set leaks into every test that runs after it in the session; the one
    that caught it on `main` was
    `test_transcribe.py::test_content_hash_stable_across_processes`, which
    hashes an in-process trace against a clean subprocess — parent f64,
    child f32, hashes differ. Invisible to anyone running with
    `JAX_ENABLE_X64=1` in the environment, because then the child inherits
    it too, and CI sets no such variable.

    Function-scoped and NOT autouse, unlike most modules': the rest of this
    file is hand-built IR that runs with no jax at all, so requesting the
    fixture is also this module's jax gate.
    """
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


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


@pytest.mark.parametrize("dtype", FORMAT_NAMES)
def test_the_ieee_zero_divisor_rule_is_emitted_not_only_documented(dtype):
    """`IEEE_ZERO_DIVISOR_TOP` reaches the reader the CHANGELOG sends to it.

    The constant is named and placed like `SCATTER_ADD_IEEE_DECLINE`, which
    is raised as user-facing text, but it was referenced only from
    docstrings — while `CHANGELOG.md` told a user "`interval.
    IEEE_ZERO_DIVISOR_TOP` says why". What they actually saw contradicted
    itself: the transfer returned ⊤ as an ORDINARY result, so it counted
    KNOWN and the undecided note said "none fell to ⊤ … compatible with a
    precision near-miss" about a `[-inf, +inf]` box, while the stamp's
    `top_despite_coverage` line named `div ×1` in the same verdict.

    Both halves are asserted: the prose is emitted, and the coverage counts
    the ⊤ so the two sentences agree (audit 0.2.0 B5-6).
    """
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
    assert any(iv.IEEE_ZERO_DIVISOR_TOP in n for n in p.notes), p.notes
    assert any("contains zero (after the subnormal haze)" in n for n in p.notes)
    assert p.coverage.unknown_primitives == (("div", 1),), p.coverage


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


def test_boundary_div_answers_inf_over_inf_instead_of_raising():
    """**The same crash class in the SIBLING kernel** (audit 0.2.0 B5-3).

    The claim recorded for `ieee_div` — "returning ⊤ before any endpoint
    arithmetic removes the `NaN endpoint` raise too" — was true of that
    kernel and false of the real-mode one, which was never changed.
    `_boundary_div_lo`/`_hi` fall to `_down(num/den)` when either operand is
    infinite, and `inf/inf` is NaN, so `boundary_div([inf, inf], [0, inf])`
    raised `IntervalError("NaN endpoint in interval arithmetic")`. The
    dispatcher catches it, so it never crashed `check` — it surfaced an
    INTERNAL INVARIANT STRING as the user-facing decline reason, out of a
    public entry point. `div`'s own `inf/inf` guard now runs first in both
    of `boundary_div`'s arms.
    """
    r = iv.boundary_div(s(INF, INF), s(0.0, INF))
    assert (r.los[0], r.his[0]) == (-INF, INF)
    r2 = iv.boundary_div(s(-INF, -INF), s(-INF, 0.0))
    assert (r2.los[0], r2.his[0]) == (-INF, INF)

    # exhaustive over the pool: no legal one-sided-boundary call raises
    raised = []
    for alo, ahi in _SWEEP_BOXES:
        for blo, bhi in _SWEEP_BOXES:
            if blo < 0.0 < bhi or (blo == 0.0 and bhi == 0.0):
                continue  # outside boundary_div's documented precondition
            try:
                iv.boundary_div(s(alo, ahi), s(blo, bhi))
            except iv.IntervalError as e:  # pragma: no cover - the finding
                raised.append(((alo, ahi), (blo, bhi), str(e)))
    assert raised == [], f"{len(raised)} box pairs still raise: {raised[:3]}"


def test_the_nan_endpoint_string_is_not_a_decline_reason_the_user_can_see():
    """The same defect through the PUBLIC entry point, which is what made it
    worth fixing: the dispatcher catches the `IntervalError`, so nothing
    crashed — it printed the domain's internal invariant text as the reason
    `div` declined.

    `a = [inf, inf]`, `assume(b > 0)` on `b = [0, inf]`: the certificate
    admits `boundary_div`, whose `[0, hi]` arm divides `alo / bhi` —
    `inf / inf`. Measured on the pre-fix tree, verbatim:

        'div' declined this form at ...: NaN endpoint in interval arithmetic

    Both halves are asserted: the string is gone, and the obligation is
    still undecided for the RIGHT reason (`inf/inf` really is ⊤).
    """
    a, b, q, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    pa, ao = var(5, BOOL), var(6, BOOL)
    query = close(
        [
            any_eqn(a, INF, INF),
            any_eqn(b, 0.0, INF),
            eqn("gt", [b, lit(0.0)], pa),
            eqn("stelling_assume", [pa], ao),
            eqn("div", [a, b], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(query)
    assert not any("NaN endpoint" in n for n in p.notes), p.notes
    assert not any("declined this form" in n for n in p.notes), p.notes
    assert p.obligations[0].status == "unknown"


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


_BOUNDARY_DIV_DIVIDENDS = [
    (-2.0, -2.0), (-5.0, -1.0), (-1.0, -1e-9), (-1e300, -1e-300),
    (0.0, 0.0), (0.0, 4.0), (1.0, 1.0), (2.0, 4.0), (1e-300, 1e300),
    (-3.0, 7.5), (-1.0, 0.0), (0.0, 1e-320),
]
_BOUNDARY_DIV_DIVISORS = [
    (0.0, 1.0), (0.0, 2.0), (0.0, 32.0), (0.0, 1e300), (0.0, 5e-324),
    (-1.0, 0.0), (-3.0, 0.0), (-1e300, 0.0), (-5e-324, 0.0), (-0.125, 0.0),
]
_BOUNDARY_DIV_CROWDING = (
    2, 3, 10, 100, 10**3, 10**6, 10**9, 10**30, 10**120, 10**300,
)

# The exact number of quotients the sweep below checks. Asserted rather than
# bounded, and quoted in SOUNDNESS.md: a figure in the log that no run in the
# tree reproduces is worth less than no figure (audit 0.2.0 B5-4 — the entry
# claimed 31,350 quotients over ten cases while the shipped test executed 195
# over five and asserted only `> 100`).
BOUNDARY_DIV_SWEEP_QUOTIENTS = 7560


def test_real_boundary_div_covers_every_nonzero_real_in_the_divisor_box():
    """The claim that licenses the difference, verified rather than asserted.

    Over R there is ONE zero and `a/0` is undefined, so `boundary_div`'s
    obligation is to cover `a/b` for every real `b != 0` in the box. Checked
    in exact rational arithmetic over 12 dividend boxes x 10
    one-sided-boundary divisor boxes, at values crowding the zero endpoint
    down to a relative offset of `1e-300` of the span, where the quotient
    diverges.

    **This does not license `boundary_div`'s REACHABILITY**, which is a
    separate question the transfer answers (audit 0.2.0 B5-1): the kernel is
    sound over `b != 0`, and whether `b != 0` holds is what the strict-assume
    certificate decides. A sweep of the kernel can never see that, which is
    exactly how a sound kernel came to sit under a false VERIFIED.
    """
    checked = 0
    for alo, ahi in _BOUNDARY_DIV_DIVIDENDS:
        for blo, bhi in _BOUNDARY_DIV_DIVISORS:
            r = iv.boundary_div(s(alo, ahi), s(blo, bhi))
            lo, hi = r.los[0], r.his[0]
            flo = Fraction(lo) if math.isfinite(lo) else None
            fhi = Fraction(hi) if math.isfinite(hi) else None
            xs = [
                Fraction(alo), Fraction(ahi),
                (Fraction(alo) + Fraction(ahi)) / 2,
            ]
            span = Fraction(bhi) - Fraction(blo)
            ys = [Fraction(blo), Fraction(bhi)]
            for k in _BOUNDARY_DIV_CROWDING:
                ys += [Fraction(blo) + span / k, Fraction(bhi) - span / k]
            for x in xs:
                for y in ys:
                    if y == 0 or not (Fraction(blo) <= y <= Fraction(bhi)):
                        continue
                    q = x / y
                    checked += 1
                    assert flo is None or q >= flo, (
                        f"boundary_div([{alo},{ahi}],[{blo},{bhi}]) -> "
                        f"[{lo},{hi}] misses {x}/{y}"
                    )
                    assert fhi is None or q <= fhi, (
                        f"boundary_div([{alo},{ahi}],[{blo},{bhi}]) -> "
                        f"[{lo},{hi}] misses {x}/{y}"
                    )
    assert checked == BOUNDARY_DIV_SWEEP_QUOTIENTS, (
        f"the sweep executed {checked} quotients; SOUNDNESS.md quotes "
        f"{BOUNDARY_DIV_SWEEP_QUOTIENTS}. Update both or neither."
    )


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


def test_real_mode_div_transfer_reaches_boundary_div_under_a_certificate():
    """The real-mode dispatch reaches `boundary_div`, and what gets it
    there is the strict assume, not the position of the zero.

    This construction used to omit the assume and still discharge, which
    was the B5-1 false VERIFIED in its smallest form: `x = 0` is a DECLARED
    value, ℝ has no `-2/0`, and the running program returns `-inf` — which
    is not `> 0`. With `assume(x < 0)` the zero is excluded by the
    precondition, the closed box is still `[-1, 0]` because an interval
    cannot hold an open bound, and dropping the endpoint drops nothing.
    """
    a, x, q, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    pa, ao = var(5, BOOL), var(6, BOOL)
    eqns = [
        any_eqn(a, -2.0, -2.0),
        any_eqn(x, -1.0, 0.0),
        eqn("lt", [x, lit(0.0)], pa),
        eqn("stelling_assume", [pa], ao),
        eqn("div", [a, x], q),
        eqn("gt", [q, lit(0.0)], pred),
        eqn("stelling_assert", [pred], out),
    ]
    p = propagate(close(eqns, [out]))
    assert p.obligations[0].status == "discharged", (
        f"real-mode boundary division regressed: {p.obligations[0].detail}"
    )

    # …and without the assume it declines, naming the reason.
    bare = propagate(close([eqns[0], eqns[1], *eqns[4:]], [out]))
    assert bare.obligations[0].status == "unknown"
    assert any("REACHES zero at a boundary" in n for n in bare.notes), bare.notes


# =========================================================================
# S10 — a standing containment sweep, signed zeros distinguished
# =========================================================================

def _fmt_max(name):
    """The largest finite value of a format: `(2 - 2**(1-p)) * 2**emax`."""
    p, _emin, emax = _FLOAT_FORMATS[name]
    return math.ldexp(2.0 - math.ldexp(1.0, 1 - p), emax)


def _pow2(e: int) -> Fraction:
    """``2**e`` as an exact rational, for either sign of ``e``."""
    return Fraction(2**e, 1) if e >= 0 else Fraction(1, 2**-e)


def _round_to_format(q: Fraction, name: str) -> float:
    """``q`` rounded to nearest-even in ``name``'s grid, overflowing to ±inf.

    The narrow-format oracle, in pure Python. The alternative was to divide
    in `numpy`/`ml_dtypes`, which would put an import gate — and a skip the
    inventory has to disclose — on the only test that drives narrow-format
    division. Validated against `numpy.float16/float32/float64` and
    `ml_dtypes.bfloat16` over 379,440 operand pairs (each format's pool plus
    300 pseudo-random in-format values, all four formats), 0 mismatches; the
    check is not shipped because those packages are the dependency this
    function exists to avoid.

    Not a general float printer: it takes an EXACT rational and returns the
    binary64 double that holds the format's answer (every value of all four
    formats is a binary64 value), which is what the boxes are compared
    against.
    """
    p, emin, emax = _FLOAT_FORMATS[name]
    if q == 0:
        return 0.0
    sign = -1 if q < 0 else 1
    a = -q if q < 0 else q
    # e = floor(log2(a)), from the bit lengths plus at most one correction
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while _pow2(e) > a:
        e -= 1
    while _pow2(e + 1) <= a:
        e += 1
    if e > emax:  # past the top binade: overflow, whatever the significand
        return sign * INF
    e = max(e, emin)  # subnormals share the minimum exponent's ulp
    shift = p - 1 - e  # ulp of the target binade is 2**-shift
    n = round(a * _pow2(shift))  # Fraction rounds half to EVEN — IEEE's rule
    try:
        val = sign * math.ldexp(float(n), -shift)
    except OverflowError:  # rounded up out of the top binade
        return sign * INF
    fmax = _fmt_max(name)
    if val > fmax:
        return INF
    if val < -fmax:
        return -INF
    return val


def _fmt_pool(name):
    """The adversarial value pool **for this format**.

    It used to be one binary64 pool for all four parametrizations, so the
    three narrow instances drove `ieee_div_fmt` with operands no float16 or
    bfloat16 program can hold (`5e-324`, `1e300`) — they exercised the
    binary64 arithmetic and the outward rounding, not the format. Each
    format now brings its OWN smallest subnormal, largest finite, and a
    mid-range magnitude, so a narrow instance is a narrow-format sweep.
    """
    fmax = _fmt_max(name)
    tiny = _ieee_format_min_positive(_FLOAT_FORMATS[name])
    mid = math.ldexp(1.0, _FLOAT_FORMATS[name][2] // 2)  # 2**(emax//2)
    return [
        -INF, -fmax, -mid, -1.0, -tiny, -0.0, 0.0, tiny, 1.0, mid, fmax, INF,
    ]


_SWEEP_POOL = _fmt_pool("float64")
_SWEEP_BOXES = [
    (lo, hi) for lo, hi in itertools.product(_SWEEP_POOL, repeat=2) if lo <= hi
]


def _float_points(lo, hi, pool):
    """Values of the box, with `+0.0` and `-0.0` kept APART. They compare
    equal, so a generator that dedups on `==` sees one zero and never
    produces the input that made S10 visible."""
    out, seen = [], set()
    for c in [lo, hi, *pool]:
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
    at points of the operand boxes — infinities and signed zeros included.

    **The pool and the oracle are both the format's now.** They used to be
    binary64's for all four parametrizations, so the three narrow instances
    fed `ieee_div_fmt` operands no float16 or bfloat16 program can hold
    (`5e-324`, `1e300`) and compared its box against a binary64 division —
    exercising the outward rounding and nothing of the format. The pool is
    each format's own extremes (`_fmt_pool`) and the finite/finite quotient
    is `_round_to_format(Fraction(x)/Fraction(y))`, the value the target
    ACTUALLY computes, arrived at without dividing in binary64 first
    (a double rounding is not the format's answer).
    """
    pool = _fmt_pool(name)
    boxes = [
        (lo, hi) for lo, hi in itertools.product(pool, repeat=2) if lo <= hi
    ]
    checked = 0
    for (alo, ahi) in boxes:
        for (blo, bhi) in boxes:
            box, made_nan = ieee_div_any_format(s(alo, ahi), s(blo, bhi), name)
            for x in _float_points(alo, ahi, pool):
                for y in _float_points(blo, bhi, pool):
                    checked += 1
                    v = _ieee_quotient(x, y, name)
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


def _ieee_quotient(x, y, name="float64"):
    """`x / y` in format `name`, with IEEE's answers, in pure Python:
    returns None for NaN.

    Python raises on division by zero instead of returning an infinity, so
    the zero cases — the whole subject — are supplied from the standard:
    `+-finite/+-0 = +-inf` by XOR of the sign bits, `0/0` and `inf/inf` NaN.
    The finite/finite case is the exact rational quotient rounded once into
    the target format, which for `float64` is the same value `x / y`
    produces and for the narrow formats is the value binary64 division
    cannot give.
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
    if xz:
        return math.copysign(0.0, sign)
    return _round_to_format(Fraction(x) / Fraction(y), name)


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


# =========================================================================
# B5-2 — `dot_general` follows `mul`'s rule again, because it IS `mul`'s rule
# =========================================================================


def _contract_1d(a_box, b_box):
    """`dot_general`'s 1-D contraction: the interval meaning of
    `jnp.dot(x, y)` for vectors."""
    return iv.dot_general(a_box, b_box, (((0,), (0,)), ((), ())))


def test_dot_general_no_longer_loses_the_floor_reduce_sum_keeps():
    """The M16 shape one level up (audit 0.2.0 B5-2).

    `dot_general` carried an INLINED COPY of `mul`'s four-corner rule and
    M16 converted only the original, so `jnp.sum(x*x)` floored at exactly 0
    while `jnp.dot(x, x)` floored at `-1e-323` — the same nonnegative clamp,
    defeated the same way, one level up. Measured before the fix:
    `(-1e-323, 32.00000000000001)` against `reduce_sum`'s `(0.0, 32.0)`.

    The two are the same call now (`interval._mul_corners`), so this asserts
    the boxes are IDENTICAL rather than merely both nonnegative — an
    equality a future divergence cannot satisfy by accident.
    """
    X = iv.from_bounds((2,), 0.0, 4.0)
    via_sum = iv.reduce_sum(iv.mul(X, X), (0,))
    via_dot = _contract_1d(X, X)
    assert (via_dot.los[0], via_dot.his[0]) == (0.0, 32.0)
    assert (via_dot.los, via_dot.his) == (via_sum.los, via_sum.his)


def test_dot_general_is_exact_when_the_contraction_is_representable():
    """A matmul of `[2,3]`-valued matrices: each output element is
    `2 x [4, 9]`, exactly `[8, 18]`. The bumped copy returned
    `[7.999999999999999, 18.000000000000004]`."""
    A = iv.from_bounds((2, 2), 2.0, 3.0)
    r = iv.dot_general(A, A, (((1,), (0,)), ((), ())))
    assert r.los == (8.0,) * 4 and r.his == (18.0,) * 4


def test_dot_general_containment_on_a_battery():
    """The containment evidence `mul` got, for the converted rule.

    A 1-D contraction's image is `sum_i x_i*y_i`, and with the operand
    elements independent (no index appears twice in one output element —
    the property the row rests on) the exact image endpoints are the sums of
    the per-term corner extrema. Checked in exact rational arithmetic over
    every ordered endpoint pair from an 8-value dyadic pool, and — because
    every value here is a small dyadic — the box must be EXACTLY the image,
    not merely contain it.
    """
    pool = [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 4.0, 8.0]
    boxes = [(lo, hi) for lo, hi in itertools.product(pool, repeat=2) if lo <= hi]
    checked = 0
    for (alo, ahi) in boxes:
        for (blo, bhi) in boxes:
            # two identical terms, so the image is 2x the single-term image
            a_box = iv.IntervalArray(shape=(2,), los=(alo, alo), his=(ahi, ahi))
            b_box = iv.IntervalArray(shape=(2,), los=(blo, blo), his=(bhi, bhi))
            r = _contract_1d(a_box, b_box)
            corners = [
                Fraction(x) * Fraction(y)
                for x in (alo, ahi)
                for y in (blo, bhi)
            ]
            lo_exact, hi_exact = 2 * min(corners), 2 * max(corners)
            checked += 1
            assert Fraction(r.los[0]) <= lo_exact
            assert Fraction(r.his[0]) >= hi_exact
            assert Fraction(r.los[0]) == lo_exact, (
                f"dot_general([{alo},{ahi}],[{blo},{bhi}]) lo "
                f"{r.los[0]} != {float(lo_exact)}"
            )
            assert Fraction(r.his[0]) == hi_exact, (
                f"dot_general([{alo},{ahi}],[{blo},{bhi}]) hi "
                f"{r.his[0]} != {float(hi_exact)}"
            )
    assert checked == 1296


def test_dot_general_keeps_the_bump_where_mul_does():
    """The confinement is shared too: an infinite endpoint takes the
    unconditional-bump route in both, because `Fraction(inf)` raises and the
    `0 * ±inf = 0` convention is an endpoint rule. One implementation, one
    boundary."""
    a_box = iv.IntervalArray(shape=(1,), los=(2.0,), his=(INF,))
    b_box = iv.IntervalArray(shape=(1,), los=(3.0,), his=(4.0,))
    r = _contract_1d(a_box, b_box)
    assert r.his[0] == INF
    assert r.los[0] < 6.0  # bumped, exactly as `mul` is on the same operands
    assert r.los[0] == iv.mul(a_box, b_box).los[0]


def test_ieee_mul_deliberately_keeps_the_native_float_product():
    """`ieee_mul` does NOT take the exact route, and this pins why.

    Under ieee the value the program has IS `fl(x*y)`; the native corner
    product already IS that value, so routing through `Fraction` would round
    a REAL product outward and manufacture slack where there is none.
    """
    box, made_nan = iv.ieee_mul(s(1e300, 1e300), s(1e300, 1e300))
    assert (box.los[0], box.his[0]) == (INF, INF)
    assert made_nan is False
    assert iv.mul(s(1e300, 1e300), s(1e300, 1e300)).los[0] == FMAX

    # And the ordinary case stays the float point, not a rational bracket.
    point, _ = iv.ieee_mul(s(0.1, 0.1), s(0.1, 0.1))
    assert point.los[0] == point.his[0] == 0.1 * 0.1


def test_the_overflow_argument_for_ieee_mul_proves_too_much():
    """**Why the reason above stops where it does** (audit 0.2.0 B5-6).

    The docstring used to add an overflow argument: two binary64 operands
    near `FMAX` multiply to `inf`, so the exact route's `[FMAX, inf]` would
    "name a value the program cannot compute". True of binary64 — and the
    row's own NARROW-format path already returns exactly that box, because
    the corners are computed in binary64 and only then rounded outward onto
    the narrow grid. Measured here on float32, so nobody re-derives the
    argument from the docstring and applies it to the sibling.

    Sound in both places: the box holds the value the target computes. It
    is the ARGUMENT that does not survive, and an argument that condemns
    the row next door cannot be this row's reason.
    """
    fmt = _FLOAT_FORMATS["float32"]
    f32max = _fmt_max("float32")
    box, made_nan = iv.ieee_mul_fmt(
        s(f32max, f32max), s(f32max, f32max), _ieee_format_min_normal(fmt)
    )
    box = _ieee_round_box(box, fmt)
    assert (box.los[0], box.his[0]) == (f32max, INF), (
        "the narrow path stopped naming FMAX; re-read the docstring's reason"
    )
    assert made_nan is False
    # the value float32 actually computes is `inf`, and the box holds it —
    # while ALSO holding `FMAX`, which is the half the old argument forbade
    assert box.his[0] == INF and box.los[0] < INF


def _sum_of_squares_query(with_assume: bool):
    """`1 / sum(x*x) > 0` over `x in [0, 4]^2`, with or without
    `assume(x > 0)`. Hand-built so the two differ in exactly one pair of
    equations."""
    VEC = ir.Aval(kind="ShapedArray", shape=(2,), dtype="float64")
    VBOOL = ir.Aval(kind="ShapedArray", shape=(2,), dtype="bool")
    x = ir.Var(id=0, aval=VEC)
    sq = ir.Var(id=1, aval=VEC)
    tot, q, pred, out = var(2), var(3), var(4, BOOL), var(5, BOOL)
    pa, ao = ir.Var(id=6, aval=VBOOL), ir.Var(id=7, aval=VBOOL)
    decl = ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(x,),
        params=(("shape", (2,)), ("dtype", "float64"), ("lo", 0.0), ("hi", 4.0)),
    )
    gate = (
        [
            eqn("gt", [x, ir.Literal(val=0.0, aval=ir.Aval(
                kind="ShapedArray", shape=(), dtype="float64"))], pa),
            eqn("stelling_assume", [pa], ao),
        ]
        if with_assume
        else []
    )
    return close(
        [
            decl,
            *gate,
            eqn("mul", [x, x], sq),
            eqn("reduce_sum", [sq], tot, params=(("axes", (0,)),)),
            eqn("div", [lit(1.0), tot], q),
            eqn("gt", [q, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_mul_transfer_end_to_end_reaches_boundary_division():
    """The shape the 0.2.0 boundary-division row was added for: a
    sum-of-squares residual in the denominator, with the `assume(x > 0)`
    the row's own description gives it.

    Two things have to hold at once. `sum(x*x)` must floor at exactly 0
    (M16 — with the bump the divisor was a true straddle and the division
    declined before `boundary_div` was reached), and the strict assume's
    exclusion of zero must SURVIVE `mul` and `reduce_sum` to the division
    (B5-1 — the closed box is `[0, 32]` either way, so the box alone
    cannot license dropping the endpoint).
    """
    p = propagate(_sum_of_squares_query(with_assume=True))
    assert p.obligations[0].status == "discharged", (
        f"the `x*x` spelling still cannot reach boundary division: "
        f"{p.obligations[0].detail}; notes {p.notes}"
    )


def test_the_same_sum_of_squares_declines_with_the_assume_removed():
    """**The attribution control for the test above**, and the B5-1 defect
    at the propagate layer: remove the one assume and the identical divisor
    box `[0, 32]` must now DECLINE. `x = [0, 0]` is a declared point, the
    divisor is exactly 0 there, and ℝ has no quotient at it.

    So `boundary_div`'s reachability tracks the CERTIFICATE, not the shape
    of the box — which is the whole content of the fix, since the two
    queries produce the same box.
    """
    p = propagate(_sum_of_squares_query(with_assume=False))
    assert p.obligations[0].status == "unknown"
    assert any("REACHES zero at a boundary" in n for n in p.notes), p.notes
    assert any("[0.0, 32.0]" in n for n in p.notes), p.notes


# =========================================================================
# The traced faces — jax present
# =========================================================================


def test_s10_jax_computes_the_infinity_the_old_box_excluded(_x64):
    """The measurement the finding rests on, kept as a test: in every format
    jax evaluates `-2.0 / x` at `x = +0.0` to `-inf`, so a VERIFIED for
    `a/x > 0` over `x in [-1, 0]` is false about the running program."""
    import jax  # noqa: F401  (the `_x64` fixture is the import gate)
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


def test_three_spellings_of_squared_reach_the_same_verdict(_x64):
    """`x*x`, `x**2`, `jnp.square(x)` and `jnp.dot(x, x)` are the same real
    property. The `mul` bump used to decide between them: `via_mul` came
    back UNKNOWN with a decline recommending `assume(divisor > 0)` — which
    the caller had already effectively done on the inputs.

    `via_dot` is the FOURTH spelling and it is here for audit 0.2.0 B5-2:
    `dot_general` carried an inlined copy of `mul`'s corner rule that M16
    did not convert, so the contraction kept the bump and lost the same
    zero floor. It shares `_mul_corners` now, and the row is that all four
    spellings agree.

    Every one of them needs `assume(x > 0)` to reach `boundary_div` at all
    (audit 0.2.0 B5-1): the divisor's box is `[0, S]` either way, and what
    licenses dropping the zero is the strict assume, not the shape of the
    box. The propagator carries the strictness across `mul`/`square`/
    `integer_pow`/`reduce_sum`/`dot_general` to the division — which is
    precisely what this test measures four ways.
    """
    import jax  # noqa: F401  (the `_x64` fixture is the import gate)
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

    def via_dot():
        x = any_array((2,), jnp.float64, (0.0, 4.0))
        assume(x > 0.0)
        return assert_(1.0 / jnp.dot(x, x) > 0.0)

    got = {
        h.__name__: check(h, vacuity_mode="inputs-only").status
        for h in (via_mul, via_ipow, via_square, via_dot)
    }
    assert set(got.values()) == {"VERIFIED"}, got


def test_a_sum_of_squares_residual_declines_without_an_assume(_x64):
    """**The false VERIFIED audit 0.2.0 B5-1 names, refuted against jax.**

    `mul`'s exactness fix (M16) makes `sum(x*x)` floor at exactly 0, so
    `sum(x*x) - 8` over `x in [0, 2]^2` boxes to `[-8, 0]` — a ONE-SIDED
    BOUNDARY where it used to be a true straddle. `boundary_div` then
    returned `(-inf, -0.125]` and `q <= -0.125` DISCHARGED, because the
    kernel drops `b = 0` from the image. The program does not: at
    `x = [2, 2]`, a point of the DECLARED box, jax computes `+inf`.

    Both halves are asserted here — the verdict is not definite, AND jax
    at the declared point falsifies what the definite verdict would have
    claimed — so the test cannot be satisfied by an UNKNOWN that arrives
    for some unrelated reason.
    """
    import jax  # noqa: F401  (the `_x64` fixture is the import gate)
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_
    from stelling.preconditions import check

    def residual():
        x = any_array((2,), jnp.float64, (0.0, 2.0))
        return assert_(1.0 / (jnp.sum(x * x) - 8.0) <= -0.125)

    v = check(residual, vacuity_mode="inputs-only")
    assert v.status == "UNKNOWN", (
        f"FALSE VERIFIED: {v.status}; jax returns +inf at x = [2, 2]"
    )
    assert any("REACHES zero at a boundary" in n for n in v.notes), v.notes

    at_zero = 1.0 / (jnp.sum(jnp.array([2.0, 2.0]) ** 2) - 8.0)
    assert float(at_zero) == INF
    assert not bool(at_zero <= -0.125), (
        "the point that refutes the old verdict no longer refutes it"
    )
    # and the shape is genuinely decidable elsewhere in the box, so the
    # UNKNOWN is about the dropped point and not about the whole obligation
    inside = 1.0 / (jnp.sum(jnp.array([1.0, 1.0]) ** 2) - 8.0)
    assert bool(inside <= -0.125)


# --- the S10 sweep table is now a run, not a paragraph ---------------------


def test_the_S10_sweep_table_reproduces_exactly():
    """The counts `SOUNDNESS.md` prints, produced by a shipped sweep.

    The S10 entry's table was an out-of-tree measurement: `grep` for its
    counts found nothing in `tests/` or `src/`, which left it the largest
    unverifiable numeric block in the file — three paragraphs above the
    one explaining why such blocks are a problem, and immediately after
    the sibling figure that had just been converted to an exact assert
    (audit 0.2.0 B5-4, then the follow-up finding that the larger table
    beside it needed the same treatment). `tests/ieee_containment_sweep.py` is
    that sweep, and this is the assertion behind the table.

    Exact equality, not a bound, and on all four columns: a sweep that
    quietly stops sampling half its grid still reports "0 failures".

    DRIVEN THROUGH ``run_all``, not through a second loop over ``ROWS``
    (audit 0.2.0 B8a, item 7). That helper was dead and documented the
    OPPOSITE column order from ``POST_FIX_ROWS`` two screens above it — in
    a module whose purpose is stopping numbers from drifting. Realigning it
    and then leaving it uncalled would have fixed the text and kept the
    trap; calling it here is what holds the two orders together.
    """
    sweep = importlib.import_module("ieee_containment_sweep")
    assert (sweep.POOL_SIZE, sweep.BOX_COUNT, sweep.BOX_PAIRS) == (
        len(sweep.POOL), len(sweep.BOXES), len(sweep.BOXES) ** 2
    ), "the grid moved; SOUNDNESS.md quotes its size"
    rows = sweep.run_all()
    assert {k for k in rows if len(k) == 2} == set(sweep.POST_FIX_ROWS), (
        "run_all and the quoted table cover different rows"
    )
    for (name, fmt), want in sweep.POST_FIX_ROWS.items():
        got = rows[(name, fmt)]
        misses = rows.get((name, fmt, "misses"), ())
        assert got == want, (
            f"{name}/{fmt}: sweep produced (failures, samples, nan, raised)"
            f"={got}, SOUNDNESS.md quotes {want}. Update both or neither."
            + ("\n  " + "\n  ".join(misses) if misses else "")
        )


def test_the_S10_sweep_CATCHES_the_defect_it_certifies_gone():
    """POSITIVE CONTROL. A battery that has never failed is not evidence.

    `prefix_ieee_div` is the kernel S10 replaced — the one that case-split
    on WHERE the zero sat and kept a boundary-aware tightening for the
    one-sided shapes. Sound over ℝ, wrong under IEEE, because `[0, hi]`
    holds both zeros and `x / -0.0` is the opposite infinity from
    `x / +0.0`. If this ever stops finding violations, the table above is
    certifying nothing.
    """
    sweep = importlib.import_module("ieee_containment_sweep")
    s, f, n, r, _misses = sweep.sweep_ieee(
        "div", "float64", kern=sweep.prefix_ieee_div
    )
    assert (f, s, n, r) == sweep.PRE_FIX_IEEE_DIV_F64, (
        f"pre-fix control produced {(f, s, n, r)}, SOUNDNESS.md quotes "
        f"{sweep.PRE_FIX_IEEE_DIV_F64}. Update both or neither."
    )
    assert f > 0, "the control found no violation at all"
