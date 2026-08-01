# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A DECLARED BOUND IS RECORDED EXACTLY OR REFUSED — for every spelling of
the value and every route into the declaration layer.

"Exactly" is the landed dtype-aware policy read value-first: the recorded
box must contain every declared dtype value (an endpoint may round only
where the shaved sliver holds no value of the dtype, and an admitted
narrowing triggers the exact raw-endpoint emptiness re-check), and the
DECISION is a function of (value, dtype) alone. The live defect this
closed: ``np.longdouble(2**53+1)`` as a point bound was ADMITTED recording
``[2**53, 2**53]`` — disjoint from the declared point — and
``preconditions.check`` returned VERIFIED for an assertion false at the
only declared point, while the python-int spelling of the identical
declaration was refused.

Spellings outside the classifier's closed family (str, complex, arbitrary
``__float__`` objects, jax arrays, ...) are covered by DEFAULT-DENY: they
are refused before any conversion, with the family named. Routes are
covered by delegation: every route ends in ``any_array``, which now
classifies the caller's own objects (see
``tests/test_declaration_routes_agree.py`` for the route-agreement rails).
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")
import numpy as np

from stelling import preconditions
from stelling.harness import any_array, any_pytree, trace

LD_53P1 = np.longdouble(2) ** 53 + 1  # exact in longdouble's 64-bit significand


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _decide(dtype, lo, hi, route="hand"):
    """(decision, message-or-recorded-params) through a declaration route."""
    if route == "hand":
        fn = lambda: (any_array((1,), dtype, (lo, hi)),)  # noqa: E731
    else:
        fn = lambda: any_pytree(np.zeros((1,), np.dtype(dtype)), (lo, hi))  # noqa: E731
    try:
        cj = jax.make_jaxpr(fn)()
        p = cj.eqns[0].params
        return "admit", (p["lo"], p["hi"])
    except (ValueError, TypeError) as e:
        return "refuse", str(e)


# -- the live defect, end to end ----------------------------------------------


@pytest.mark.parametrize("route", ["hand", "sugar"])
@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_the_longdouble_false_verified_reproducer_now_refuses(route, dtype):
    """The exact reproducer measured on the parent: admitted recording
    [2**53, 2**53] on both dtypes; on float64, ``x <= 2**53`` — false at
    the only declared point 2**53+1 — returned VERIFIED."""
    d, out = _decide(dtype, LD_53P1, LD_53P1, route)
    assert d == "refuse", (
        f"{dtype}/{route}: the longdouble point 2**53+1 was admitted "
        f"recording {out!r}"
    )
    # the refusal names the true cause per dtype: int64 loses declared
    # values to the narrowing; float64's declared point holds no float64
    if dtype == "int64":
        assert "NARROWS the declared set" in out
    else:
        assert "declare a set EMPTY under dtype" in out


@pytest.mark.parametrize("route", ["hand", "sugar"])
def test_the_false_verified_cannot_be_reproduced_through_check(route):
    """End to end: the harness that returned VERIFIED for a claim false at
    the only declared point now refuses inside preconditions.check (a
    harness-authoring defect stays loud — check() only swallows
    transcription gaps)."""

    def harness():
        from stelling.harness import any_array, any_pytree, assert_

        if route == "hand":
            x = any_array((1,), "float64", (LD_53P1, LD_53P1))
        else:
            x = any_pytree(np.zeros((1,), np.float64), (LD_53P1, LD_53P1))
        return assert_(x <= 2.0**53)

    with pytest.raises(ValueError, match="EMPTY under dtype"):
        preconditions.check(harness, vacuity_mode="inputs-only")


# -- the property: admitted means the declared dtype-set is contained ---------


@pytest.mark.parametrize("dtype,lo,hi", [
    ("float64", Decimal("0.1"), Decimal("0.2")),
    ("float64", Fraction(1, 3), Fraction(2, 3)),
    ("float64", np.longdouble("0.5"), np.longdouble("2.00000000000000000003")),
    ("int32", Decimal("0.5"), Decimal("100.5")),
    ("int64", LD_53P1, 2**60),
    ("float32", np.float16("0.1"), np.float32("0.2")),
])
def test_an_admitted_inexact_spelling_records_a_box_containing_every_declared_dtype_value(
    dtype, lo, hi
):
    """The recorded endpoint is the correctly-rounded binary64 image of the
    declared value (never a second-hand conversion), and any rounding is
    outward-or-loss-free: no value of the dtype between a declared endpoint
    and its recorded image, checked against an independent numpy oracle."""
    d, (r_lo, r_hi) = _decide(dtype, lo, hi)
    assert d == "admit"
    x_lo, x_hi = _exact(lo), _exact(hi)
    # recorded == correctly-rounded image of the declared value
    assert r_lo == _nearest_binary64(x_lo) and r_hi == _nearest_binary64(x_hi)
    # containment of the declared dtype-set: the smallest/largest dtype
    # values inside the DECLARED interval lie inside the RECORDED box
    dd = np.dtype(dtype)
    if dd.kind in "iu":
        first, last = math.ceil(x_lo), math.floor(x_hi)
        info = np.iinfo(dd)
        first, last = max(first, int(info.min)), min(last, int(info.max))
        assert r_lo <= first and last <= r_hi
    else:
        up, down = np.array(np.inf, dd), np.array(-np.inf, dd)
        c_lo = np.array(float(x_lo), dd)
        while Fraction(*np.float64(c_lo).as_integer_ratio()) < x_lo:
            c_lo = np.nextafter(c_lo, up)
        c_hi = np.array(float(x_hi), dd)
        while Fraction(*np.float64(c_hi).as_integer_ratio()) > x_hi:
            c_hi = np.nextafter(c_hi, down)
        assert r_lo <= float(c_lo) and float(c_hi) <= r_hi


def _exact(v):
    """Independent exact-value oracle for the spellings used above."""
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    if isinstance(v, (Decimal, Fraction)):
        return Fraction(v)
    return Fraction(*v.as_integer_ratio())  # float / np.floating


def _nearest_binary64(x: Fraction) -> float:
    lo_c = float(x)  # Fraction.__float__ is correctly rounded in CPython
    return lo_c


# -- default-deny: unknown spellings are refused, never converted -------------


class _Floatable:
    """The trap the old code fell into: an object whose float() succeeds.
    Silent acceptance-via-float() of an undecided spelling is the defect
    class; this must refuse."""

    def __float__(self):
        return 1.0

    def __repr__(self):
        return "_Floatable()"


@pytest.mark.parametrize("route", ["hand", "sugar"])
@pytest.mark.parametrize("bad", [
    "0.25",                      # str: decidable but refused by policy
    _Floatable(),                # arbitrary __float__ object
    complex(1, 0),               # complex, even with zero imaginary part
])
def test_unknown_spellings_are_refused_naming_the_family(route, bad):
    d, msg = _decide("float64", bad, 1.0, route)
    assert d == "refuse"
    assert "is not an accepted bound spelling" in msg
    assert "decimal.Decimal" in msg and "fractions.Fraction" in msg


# numpy's scalar lattice is not a number lattice: np.timedelta64 IS an
# np.integer subclass, so an isinstance test alone silently ADMITTED tick
# counts with their units discarded — for the generic form and every unit
# whose value int() cannot turn into a datetime.timedelta (as/fs/ps/ns and
# the calendar units M/Y) — and crashed with a bare TypeError on the
# timedelta-representable units (us through W) and NaT; measured per unit,
# blinded lens rounds 1-2. The family test now asks the dtype KIND, and
# every lattice stray (with its 0-d array) gets the family refusal on
# every route.
_LATTICE_STRAYS = [
    np.timedelta64(5, "s"),
    np.timedelta64("NaT", "s"),
    np.datetime64("2020-01-01"),
    np.str_("0.25"),
    np.bytes_(b"1"),
    np.void(b"\x00"),
    np.array(np.timedelta64(7, "D")),
    np.array(np.datetime64("2020-01-01")),
]
_STRAY_IDS = ["td64-s", "td64-NaT", "dt64", "str_", "bytes_", "void",
              "0d-td64", "0d-dt64"]


@pytest.mark.parametrize("route", ["hand", "sugar"])
@pytest.mark.parametrize("stray", _LATTICE_STRAYS, ids=_STRAY_IDS)
def test_numpy_lattice_strays_get_the_family_refusal_on_every_route(
    route, stray
):
    d, msg = _decide("float64", stray, 1.0, route)
    assert d == "refuse", (stray, msg)
    assert "is not an accepted bound spelling" in msg


def test_the_unitless_timedelta_reproducer_is_refused_not_recorded():
    """The lens's exact reproducer: np.timedelta64(5) (generic unit) was
    ADMITTED recording its tick count as an integer bound."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        td = np.timedelta64(5)
    for route in ("hand", "sugar"):
        d, msg = _decide("int64", td, td, route)
        assert d == "refuse", (route, msg)
        assert "is not an accepted bound spelling" in msg


def test_numpy_lattice_strays_refuse_at_contract_authoring_too():
    from stelling.contracts import _closed_range

    for stray in _LATTICE_STRAYS:
        with pytest.raises(ValueError,
                           match="not an accepted bound spelling"):
            _closed_range("t", "n", (0, stray))


def test_every_spelling_the_refusal_names_is_actually_accepted():
    """The refusal text lists the family; each listed member must reach a
    DECISION (admit here), not the family refusal — the message's claims,
    measured. The jax hint is measured too: np.asarray of a concrete jax
    scalar is an accepted 0-d array."""
    members = [
        (0, 1),                                  # python int
        (0.0, 1.0),                              # python float
        (False, True),                           # python bool
        (np.int32(0), np.uint8(1)),              # numpy integers
        (np.float16(0), np.longdouble(1)),       # numpy floatings
        (np.False_, np.True_),                   # numpy bools
        (Decimal(0), Decimal(1)),                # Decimal
        (Fraction(0), Fraction(1)),              # Fraction
        (np.array(0.0), np.array(1)),            # 0-d arrays
        (np.asarray(jax.numpy.asarray(0.0)), 1.0),  # the message's jax hint
    ]
    for lo, hi in members:
        d, out = _decide("float64", lo, hi)
        assert d == "admit", (lo, hi, out)
        assert out == (0.0, 1.0)


def test_a_self_referential_object_array_is_refused_not_a_crash():
    """The 0-d unwrap recurses once, onto a numpy scalar only: a 0-d object
    array containing an ndarray (itself included) must hit the family
    refusal, not RecursionError."""
    a = np.empty((), dtype=object)
    a[()] = a
    d, msg = _decide("float64", a, 1.0)
    assert d == "refuse" and "is not an accepted bound spelling" in msg


# -- the cannot-be-recorded class, value-keyed --------------------------------

_HUGE = [
    ("int", 10**400),
    ("Decimal", Decimal("1e400")),
    ("np.longdouble", np.longdouble("1e400")),
    ("Fraction", Fraction(10**400)),
    ("0d-longdouble-array", np.array(np.longdouble("1e400"))),
]


@pytest.mark.parametrize("tag,huge", _HUGE)
@pytest.mark.parametrize("dtype", ["float64", "int64"])
def test_a_finite_bound_past_binary64_refuses_on_every_spelling(dtype, tag, huge):
    """The phase1 escape, closed value-first: float(10**400) used to raise a
    bare OverflowError while np.longdouble('1e400') silently recorded inf.
    One refusal, keyed on the value's image, covers every spelling — as hi
    (would record +inf) and negated as lo (would record -inf)."""
    d, msg = _decide(dtype, 0, huge)
    assert d == "refuse" and "outside binary64's finite range" in msg
    d, msg = _decide(dtype, -huge if tag != "np.longdouble" else -huge, 0)
    assert d == "refuse" and "outside binary64's finite range" in msg


def test_the_cannot_store_message_claims_only_measured_truths():
    """Each of the message's claims with measurable content, measured:
    (1) the quoted range endpoints are the real binary64 extremes; (2) no
    direction word like 'exceeds' appears (the phase1 lens found one
    pointing the wrong way for a negative bound); (3) the quoted rounded
    image carries the image's actual sign per side; (4) the refused value
    really lies outside the quoted range; (5) it is nevertheless FINITE
    (the 'unbounded side you did not declare' clause); (6) both suggested
    respellings — an infinite bound, or one inside the range — are
    actually accepted. The remaining clause, 'the IR stores bounds as
    binary64', is pinned by the recorded-param type assertions in the
    recording tests."""
    import sys

    _, msg_hi = _decide("float64", 0, Decimal("1e400"))
    _, msg_lo = _decide("float64", Decimal("-1e400"), 0)
    for msg in (msg_hi, msg_lo):
        assert repr(sys.float_info.max) in msg              # (1)
        assert repr(-sys.float_info.max) in msg
        assert "exceeds" not in msg                         # (2)
    assert "(inf)" in msg_hi, msg_hi                        # (3)
    assert "(-inf)" in msg_lo, msg_lo
    assert "bound hi=" in msg_hi and "bound lo=" in msg_lo
    assert Fraction(Decimal("1e400")) > Fraction(sys.float_info.max)  # (4)
    assert Decimal("1e400").is_finite()                     # (5)
    d, out = _decide("float64", 0, float("inf"))            # (6)
    assert d == "admit" and out == (0.0, math.inf)
    d, out = _decide("float64", 0, sys.float_info.max)
    assert d == "admit" and out == (0.0, sys.float_info.max)


def test_a_huge_finite_point_is_not_called_an_infinite_point():
    """(1e400-ish, 1e400-ish) used to refuse as 'an infinite point has no
    members' — false for finite, distinct declared bounds (phase1 lens A,
    corollary). The refusal now states the storability cause, on every
    spelling including the python int whose float() used to crash."""
    for lo, hi in [
        (10**400, 10**400),
        (np.longdouble("1e400"), np.longdouble("1e401")),
        (Decimal("1e400"), Decimal("2e400")),
    ]:
        d, msg = _decide("float64", lo, hi)
        assert d == "refuse"
        assert "infinite point" not in msg, msg
        assert "outside binary64's finite range" in msg
    # while a genuinely infinite point still gets the infinite-point text
    d, msg = _decide("float64", np.longdouble("inf"), np.longdouble("inf"))
    assert d == "refuse" and "infinite point has no members" in msg


def test_an_infinite_bound_against_a_huge_finite_value_is_an_empty_set():
    """lo=+inf lies above every finite hi, however large: the declared set
    is empty even though both binary64 images are +inf and compare equal.
    (Images tie exactly here, so only the exact-value clauses can see it.)"""
    for lo, hi in [
        (float("inf"), 10**400),
        (float("inf"), Decimal("1e400")),
        (Decimal("-1e400"), float("-inf")),
    ]:
        d, msg = _decide("float64", lo, hi)
        assert d == "refuse" and "declare an empty set" in msg, (lo, hi, msg)


# -- non-finite spellings behave like their float spellings -------------------


def test_infinite_and_nan_spellings_match_the_float_spelling():
    ref_admit = _decide("float64", float("-inf"), float("inf"))
    assert ref_admit[0] == "admit" and ref_admit[1] == (-math.inf, math.inf)
    for lo, hi in [
        (np.longdouble("-inf"), np.longdouble("inf")),
        (Decimal("-Infinity"), Decimal("Infinity")),
    ]:
        d, out = _decide("float64", lo, hi)
        assert (d, out) == ref_admit, (lo, hi)
    for lo, hi in [
        (Decimal("NaN"), Decimal(1)),
        (np.longdouble("nan"), np.longdouble("nan")),
    ]:
        d, msg = _decide("float64", lo, hi)
        assert d == "refuse" and "declare an empty set" in msg


def test_the_image_is_computed_on_the_ratio_not_on_its_parts():
    """A Fraction is a VALUE, not a pair of convertibles. Two kills for
    the parts-wise mutant (float(n)/float(d)), one per part: the
    IRREDUCIBLE Fraction(10**400 + 1, 10**400) — numerator coprime to
    denominator, asserted, so construction cannot normalize it away —
    has the ordinary value 1 + 10**-400 with image exactly 1.0, and the
    mutant's float(10**400 + 1) raises the OverflowError the image
    helper maps to +inf, refusing an ordinary declaration as unstorable;
    Fraction(1, 10**400) is sub-subnormal, correctly rounding to 0.0,
    and the mutant's float(10**400) DENOMINATOR overflows the same way.
    (The first version's first declaration was Fraction(10**401,
    10**400), which normalizes to Fraction(10, 1) at construction — the
    stated overflow mechanism could never occur through it; blinded
    lens, repair round 1.)"""
    big = Fraction(10**400 + 1, 10**400)
    assert big.numerator == 10**400 + 1 and big.denominator == 10**400
    d, out = _decide("float64", 0, big)
    assert d == "admit" and out == (0.0, 1.0)
    d, out = _decide("float64", Fraction(1, 10**400), 1)
    assert d == "admit" and out == (0.0, 1.0)


@pytest.mark.parametrize("lo,hi", [
    (2**53 + 1, 2**53),
    (Decimal(2**53 + 1), Decimal(2**53)),
    (np.longdouble(2) ** 53 + 1, np.longdouble(2) ** 53),
])
def test_the_raw_order_inverted_blind_spot_stays_parent_parity(lo, hi):
    """(2**53+1, 2**53) is raw-order-inverted — lo > hi as declared
    values — but both binary64 images collapse onto 2**53 and the parent
    ADMITTED it (disclosed blind spot: the landing audit counted it
    parent-parity, out of scope). The ordering check therefore judges
    IMAGES deliberately, and this pins that, spelling-independently: the
    mutant 'improving' the check to exact raw order flips these to
    refuse and left the whole suite green before this test existed
    (blinded lens, repair round 1)."""
    for dtype in ("int64", "float64"):
        d, out = _decide(dtype, lo, hi)
        assert d == "admit" and out == (2.0**53, 2.0**53), (dtype, out)


def test_a_negative_zero_bound_records_the_negative_zero_the_parent_recorded():
    """Fraction has no signed zero, so classifying -0.0 through it
    recorded +0.0 where the parent recorded -0.0 — recorded params and
    the query content hash moved for the python-float spelling, a class
    disclosed as unmoved (blinded lens, round 1). Zero bounds now go
    through float(), which is exact and keeps the sign, on every family
    spelling that can write one; each spelling here was measured
    recording -0.0 on the parent, so this is parity for python
    float/np.float64 and the PICKED-and-pinned behavior for the moved
    classes (np.float32/np.longdouble/Decimal('-0'))."""
    for lo in (-0.0, np.float64("-0.0"), np.float32("-0.0"),
               np.longdouble("-0.0"), Decimal("-0")):
        cj = jax.make_jaxpr(
            lambda l=lo: (any_array((1,), "float64", (l, 1.0)),)
        )()
        p = cj.eqns[0].params
        assert p["lo"] == 0.0 and math.copysign(1.0, p["lo"]) == -1.0, lo
        assert type(p["lo"]) is float
    cj = jax.make_jaxpr(lambda: (any_array((1,), "float64", (0.0, 1.0)),))()
    assert math.copysign(1.0, cj.eqns[0].params["lo"]) == 1.0


def test_the_dtype_level_narrows_refusal_states_the_policy_for_fractional_bounds():
    """'the tool would reason about fewer values than you declared' is
    FALSE at dtype level for (Decimal('0.1'), Decimal('5.5')) on int64 —
    no int64 lies in the shaved sliver — while longdouble(2**54) + 1.5
    genuinely drops the int64 value 2**54 + 1 (both premises measured
    below). The fractional-bound text therefore states the dtype-level
    policy and asserts neither direction, and it must also hold for
    complex dtypes, which refuse by policy with no dropped-value claim
    either way. Integer-valued bounds keep the parent's byte-identical
    text, whose dropped-value claim is true for them: the bound itself
    is a declared dtype value the recorded box excludes."""
    d, msg = _decide("int64", Decimal("0.1"), Decimal("5.5"))
    assert d == "refuse"
    assert "NARROWS the declared interval" in msg
    assert "dtype-level policy" in msg
    assert "fewer values than you declared" not in msg
    assert "|bound| <= 2**53" not in msg
    # premise 1: the sliver (1/10, binary64-0.1) holds no integer
    assert Fraction(float(Decimal("0.1"))) > Fraction(1, 10)
    assert math.floor(float(Decimal("0.1"))) == 0 < 1 == math.ceil(Fraction(1, 10))
    # premise 2: a fractional bound CAN drop an int64 value
    ld = np.longdouble(2) ** 54 + np.longdouble(1.5)
    v = Fraction(*ld.as_integer_ratio())
    assert float(v) == 2.0**54 and v > 2**54 + 1  # sliver holds 2**54+1
    d2, msg2 = _decide("int64", 0, ld)
    assert d2 == "refuse" and "dtype-level policy" in msg2
    # complex reaches the same text
    d5, msg5 = _decide("complex128", Decimal("0.1"), Decimal("5.5"))
    assert d5 == "refuse" and "dtype-level policy" in msg5
    # integer-valued spellings keep the parent text, byte-for-byte
    # identical across spellings modulo the bound's own repr
    d3, msg3 = _decide("int64", 0, Decimal(2**53 + 1))
    assert d3 == "refuse" and "fewer values than you declared" in msg3
    d4, msg4 = _decide("int64", 0, 2**53 + 1)
    assert msg3.replace(repr(Decimal(2**53 + 1)), repr(2**53 + 1)) == msg4


def test_a_numpy_spelled_infinity_records_the_python_float_the_parent_recorded():
    """np.float64 subclasses float; passing its ±inf through unnormalized
    would leak np.float64 objects into the recorded params, where the
    parent recorded python floats (measured on main) — a type change the
    decision grid cannot see. The recording must be the python float."""
    cj = jax.make_jaxpr(
        lambda: (any_array((1,), "float64",
                           (np.float64("-inf"), np.float64("inf"))),)
    )()
    p = cj.eqns[0].params
    assert (p["lo"], p["hi"]) == (-math.inf, math.inf)
    assert type(p["lo"]) is float and type(p["hi"]) is float


# -- the NARROWS refusal, reached from new spellings, stays truthful ----------


@pytest.mark.parametrize("dtype,lo,hi,which", [
    ("int64", LD_53P1, 2**60, None),               # ld lo widens: admitted
    ("int64", 0, LD_53P1, "hi"),                   # ld hi narrows: refused
    ("int64", Decimal("0.1"), Decimal("0.2"), "lo"),  # fractional lo narrows
    ("uint64", Fraction(1, 10), 2**63, "lo"),
])
def test_the_narrows_refusal_claims_are_measured_for_new_spellings(
    dtype, lo, hi, which
):
    """Three claims in the message, each checked against an independent
    computation: the bound is genuinely not a binary64 ('is not
    representable'), the quoted recording is the correctly-rounded image
    ('would be recorded as X'), and the image genuinely lies strictly
    inside the declared interval ('NARROWS')."""
    d, out = _decide(dtype, lo, hi)
    if which is None:
        assert d == "admit"
        return
    # integer-valued bounds keep the parent's "declared set" text;
    # fractional ones state the dtype-level policy over the "declared
    # interval" (its own claims are pinned in
    # test_the_dtype_level_narrows_refusal_states_the_policy...)
    assert d == "refuse" and "NARROWS the declared" in out
    raw = {"lo": lo, "hi": hi}[which]
    x = _exact(raw)
    img = float(x)  # correctly rounded by CPython
    assert x != Fraction(img), "test bug: the bound IS a binary64"
    assert f"would be recorded as {img!r}" in out
    if which == "lo":
        assert Fraction(img) > x  # strictly inside: genuinely narrows
    else:
        assert Fraction(img) < x
    assert f"bound {which}=" in out


# -- the gap-edge re-check now covers integer dtypes --------------------------


def test_a_fractional_pair_rounding_onto_an_integer_is_refused_empty():
    """The class the old integer-dtype screen let through: both endpoints a
    hair below 1 round UP onto 1.0, the recorded box [1.0, 1.0] holds the
    int32 value 1, and the declared interval holds no integer at all — a
    VERIFIED over it would be vacuous. Exact re-judgment refuses, with the
    integer-branch message and truthful neighbours."""
    lo = Decimal("0.99999999999999999990")
    hi = Decimal("0.99999999999999999995")
    assert float(lo) == 1.0 and float(hi) == 1.0  # the premise, measured
    for dtype in ("int32", "int16", "uint8"):
        d, msg = _decide(dtype, lo, hi)
        assert d == "refuse", (dtype, msg)
        assert "declare a set EMPTY under dtype" in msg
        assert "and the interval contains none of them" in msg
        assert "0 below and 1 above" in msg  # exact neighbours, both true


def test_the_admit_boundary_of_the_integer_recheck_is_exact():
    """The declared interval's largest integer sits exactly at the declared
    hi: one int32 inhabitant, and the declaration must admit (the landing
    audit pinned the float twin's boundary the same way)."""
    d, out = _decide("int32", Decimal("0.99999999999999999990"), Decimal("1"))
    assert d == "admit" and out == (1.0, 1.0)


def test_a_raw_interval_above_the_dtype_range_is_refused_when_its_image_touches_it():
    """Raw endpoints wholly above int32's max, images rounding the lo back
    onto the max: recorded box inhabited, declared set empty — the clamp
    side of the integer re-check."""
    lo = Decimal("2147483647.0000000001")   # just above int32 max, image 2147483647.0
    hi = Decimal("2147483649.0000000001")   # image rounds down: narrows -> re-check
    assert float(lo) == 2147483647.0 and float(hi) == 2147483649.0
    d, msg = _decide("int32", lo, hi)
    assert d == "refuse" and "declare a set EMPTY under dtype" in msg
    assert "2147483647 below" in msg  # truthful neighbour, clamped to the range


def test_the_integer_recheck_clamps_absorb_infinite_endpoints():
    """An unbounded side plus a fractional NARROWING endpoint on the OTHER
    side reaches the exact integer re-check with a float infinity in one
    slot; the range clamps must answer through the dtype's own extremes
    (``math.ceil(-inf)``/``math.floor(inf)`` raise — the crash the clamps
    prevent, one clamp per side). The narrowing direction is measured in
    the premise asserts: a hi must round DOWN to narrow, a lo must round
    UP — the first version of this test paired -inf with an up-rounding hi
    (a WIDENING one), never reached the re-check, and its clamp mutant
    survived the suite. Both declarations are inhabited (the unbounded
    side sweeps in the whole dtype range), so both must ADMIT."""
    hi_narrows = Decimal("1.00000000000000000005")  # image 1.0 < value
    lo_narrows = Decimal("0.99999999999999999990")  # image 1.0 > value
    assert float(hi_narrows) == 1.0 and float(lo_narrows) == 1.0
    d, out = _decide("int8", float("-inf"), hi_narrows)
    assert d == "admit" and out == (-math.inf, 1.0)
    d, out = _decide("int8", lo_narrows, float("inf"))
    assert d == "admit" and out == (1.0, math.inf)


# -- zero-size arrays keep their storability posture --------------------------


def test_zero_size_shapes_still_refuse_unrecordable_and_narrowing_bounds():
    """The storability guard runs before the zero-size vacuous return (as it
    did for python ints) — BOTH its halves, the narrowing refusal and the
    cannot-be-recorded refusal (the first version of this test exercised
    only narrowing; repair round 1); the emptiness checks stay skipped (a
    zero-size array satisfies any bounds vacuously)."""
    try:
        jax.make_jaxpr(lambda: (any_array((0,), "int64", (0, LD_53P1)),))()
        raised = None
    except ValueError as e:
        raised = str(e)
    assert raised and "NARROWS the declared set" in raised
    try:
        jax.make_jaxpr(
            lambda: (any_array((0,), "int64", (0, Decimal("1e400"))),)
        )()
        raised = None
    except ValueError as e:
        raised = str(e)
    assert raised and "outside binary64's finite range" in raised
    # while the gap-edge EMPTY refusal does not apply at zero size
    cj = jax.make_jaxpr(
        lambda: (any_array((0,), "float64", (LD_53P1, LD_53P1)),)
    )()
    assert cj.eqns[0].params["lo"] == 2.0**53


# -- the contract-template route ----------------------------------------------


def test_all_three_templates_refuse_the_longdouble_point_at_trace():
    """The template route reaches any_array's decision: the same longdouble
    point that produced the false VERIFIED refuses when each template's
    harness is traced (authoring validates only what it can see eagerly;
    the storability decision is any_array's, on the caller's own values)."""
    from stelling import contracts

    pair = (LD_53P1, LD_53P1)
    c1 = contracts.conditioning_2x2("float64", pair, (1, 2), (0, 0), 8.0)
    c2 = contracts.conditioning_2x2_field((2, 2), "float64", pair, 8.0, None)
    c3 = contracts.coefficient_contrast((4,), "float64", pair, 10.0)
    for c in (c1, c2, c3):
        with pytest.raises(ValueError, match="EMPTY under dtype"):
            trace(c.harness)


def test_closed_range_refuses_unknown_spellings_at_authoring():
    from stelling.contracts import _closed_range

    with pytest.raises(ValueError, match="not an accepted bound spelling"):
        _closed_range("t", "n", ("0.25", "0.5"))
    with pytest.raises(ValueError, match="not an accepted bound spelling"):
        _closed_range("t", "n", (0, _Floatable()))


def test_closed_range_refuses_past_double_values_cleanly_at_authoring():
    """float(10**400) used to escape _closed_range as a bare OverflowError;
    the longdouble spelling validated through as inf and was then refused
    at trace with the wrong cause. Both now get one authoring-time
    ValueError whose text states the measured fact (finite value, no
    binary64 recording) rather than 'non-finite endpoint'."""
    import sys

    from stelling.contracts import _closed_range

    for huge in (10**400, Decimal("1e400"), np.longdouble("1e400")):
        with pytest.raises(ValueError) as exc:
            _closed_range("t", "n", (0, huge))
        msg = str(exc.value)
        assert "outside binary64's finite range" in msg
        assert "non-finite endpoint" not in msg
        assert repr(sys.float_info.max) in msg
    # genuinely infinite endpoints keep the non-finite message
    with pytest.raises(ValueError, match="non-finite endpoint"):
        _closed_range("t", "n", (0, Decimal("Infinity")))
    # and it still returns the caller's own values on the family
    lo, hi = _closed_range("t", "n", (Decimal("0.1"), Fraction(1, 2)))
    assert lo == Decimal("0.1") and hi == Fraction(1, 2)


def test_the_templates_accept_exact_family_spellings_end_to_end():
    """A Decimal envelope whose values are exact binary64s authors, traces
    and records exactly — the family is usable, not merely named."""
    from stelling import contracts

    c = contracts.coefficient_contrast(
        (4,), "float64", (Decimal("0.25"), Decimal("0.5")), 10.0
    )
    closed = trace(c.harness)
    (eq,) = [e for e in closed.jaxpr.eqns if e.primitive == "stelling_any"]
    p = eq.params_dict()
    assert (p["lo"], p["hi"]) == (0.25, 0.5)
