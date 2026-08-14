# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Permanent regressions for the 0.2.0 audit's `pow` findings (S1–S3, M7–M9).

Each construction from the pre-release audit of the rational-`pow` row,
re-derived in-repo:

* **S1** — the emitted exponent was a *rationalisation* of the traced
  binary64 literal (``limit_denominator(128)`` guarded by a binary64
  distance), so ``x ** 0.1`` was analysed as ``x ** (1/10)``: a different
  real function, discharged with nothing downstream to re-derive it. The
  exposure map, the headline end-to-end harness, and the tolerance half.
* **S2** — ``q == 1`` emitted a unary ``(* aux)``, which SMT-LIB2's
  ``Reals`` theory does not define; cvc5 1.3.4 segfaults on it and z3
  reads it as the operand.
* **S3** — the rational-exponent replay was ``float(base) ** exp`` under a
  verdict sentence claiming exact-rational arithmetic, and made the public
  ``check()`` RAISE on a correct emission.
* **M7** — nothing bounded the rational branch's NUMERATOR, so
  ``x ** 100.5`` (degree 201) was admitted while ``x ** 100`` declined at
  64.
* **M8** — the same float ``pow`` raised ``OverflowError`` out of the
  replay, which no handler caught.
* **M9** — a rational ``pow`` over a declaration-independent base was
  stamped ``QF_LRA`` while the emission wrote ``(* aux aux)``, and both
  backends refused the script.

Everything above the solver divider runs with no jax and no solver.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, getcontext
from fractions import Fraction

import pytest

from stelling import ir
from stelling import smt
from stelling.obligation import (
    DeclinedObligation,
    INTEGER_POW_EXPANSION_CAP,
    ObligationSlice,
    RATIONAL_POW_DEGREE_CAP,
    ReplayDeclined,
    _exact_integer_root,
    _exact_rational_power,
    pow_exponent_rational,
    rational_pow_problem,
    witness_is_valid,
)
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, sole_slice, var


# --- S1: the emitted rational must denote the traced literal ------------------


# The audit's exposure map (AUDIT-0.2.0-FINDINGS.md § S1), verbatim. Every
# DYADIC exponent is exactly what it looks like and stays admitted; every
# other one is a substitution and must decline.
DYADIC_ADMITTED = (0.5, 0.25, 0.125, 1.5, 2.5, 63.5, 1.0 / 128.0, 127 / 128)
NOT_THE_LITERAL = (
    0.1, 0.2, 0.7, 1 / 3, 2 / 3, 1 / 7, 3 / 7,
    0.5000000000001, 1.9999999999999, 2.0000000000001, 3.0000000000001,
    1e-13, 1e-15, math.sqrt(2.0), math.e, math.log(2.0),
)


@pytest.mark.parametrize("e", DYADIC_ADMITTED)
def test_s1_a_dyadic_exponent_is_admitted_and_emitted_exactly(e):
    """A binary64 that IS a small dyadic rational stays emittable, and the
    rational the emission writes is the literal's own exact value."""
    assert rational_pow_problem(e) is None, f"{e!r} should still be emittable"
    frac = pow_exponent_rational(e)
    assert float(frac) == e
    assert frac == Fraction(e)  # the exact value, not a nearby one


@pytest.mark.parametrize("e", NOT_THE_LITERAL)
def test_s1_an_exponent_that_is_not_its_own_rational_declines(e):
    """The heart of S1. Each of these binary64 values is a dyadic rational
    of enormous denominator; the predecessor replaced it with a nearby
    low-denominator rational and emitted about THAT. The exact value is
    what the guard now compares, so every one of them declines."""
    frac = Fraction(e)
    near = frac.limit_denominator(RATIONAL_POW_DEGREE_CAP)
    problem = rational_pow_problem(e)
    assert problem is not None, (
        f"{e!r} = {frac} was admitted; the predecessor emitted about "
        f"{near}, a different real"
    )
    assert str(frac.numerator) in problem and str(frac.denominator) in problem, (
        f"the decline must name the literal's EXACT rational value; got: "
        f"{problem}"
    )


def test_s1_the_decline_message_does_not_claim_the_exponent_is_irrepresentable():
    """The predecessor's message said ``cannot be represented as p/q with
    q <= 128``. That is false: it can, exactly, and for 0.1 the exact
    denominator is 2^55. A message that misnames the reason cannot be
    acted on."""
    problem = rational_pow_problem(0.1)
    assert "3602879701896397/36028797018963968" in problem
    assert "cannot be represented" not in problem
    # and it says which nearby rational would have been the substitution
    assert "1/10" in problem


def test_s1_the_binary64_distance_test_can_never_separate_these():
    """Why no threshold fixes the predecessor: the guard it used measures
    exactly 0.0 on the case that motivated this finding."""
    near = Fraction(0.1).limit_denominator(RATIONAL_POW_DEGREE_CAP)
    assert abs(float(near) - 0.1) == 0.0
    assert near != Fraction(0.1)
    assert abs(float(near) - float(Fraction(0.1))) == 0.0


def test_s1_every_admissible_denominator_is_a_power_of_two():
    """A structural consequence worth pinning: a binary64 IS a dyadic
    rational, so an admitted non-integer exponent always has q = 2^k with
    k >= 1. In particular ``q == 1`` — the S2 shape — is unreachable."""
    seen = set()
    for i in range(1, 4000):
        e = i / 256.0
        if e == int(e) or rational_pow_problem(e) is not None:
            continue
        q = pow_exponent_rational(e).denominator
        seen.add(q)
        assert q & (q - 1) == 0, f"{e!r} admitted with denominator {q}"
        assert q >= 2
    assert seen, "the sweep admitted nothing — it is not measuring anything"


# --- M7: the two caps bound one quantity, and both bind ----------------------


@pytest.mark.parametrize("e", (100.5, 1000.5, 100000.5, 1000000000000.5))
def test_m7_a_large_numerator_declines_like_a_large_denominator(e):
    """``x ** 100.5`` emits ``aux^2 = x^201``. Nothing bounded that before,
    so a strictly larger emitted polynomial was admitted where the smaller
    ``x ** 100`` declined at the integer cap of 64 — and
    ``x ** 1000000000000.5`` built a 600 KB script before dying of
    MemoryError."""
    frac = pow_exponent_rational(e)
    assert frac.denominator == 2  # these ARE exact; only the size declines
    problem = rational_pow_problem(e)
    assert problem is not None
    assert str(frac.numerator) in problem
    assert str(RATIONAL_POW_DEGREE_CAP) in problem


def test_m7_both_caps_bound_the_degree_of_an_emitted_polynomial():
    """The two constants are not comparable magnitudes of different
    things: each is a bound on the degree of a polynomial the row writes.
    Pinned so a future edit cannot reintroduce an unbounded side."""
    assert INTEGER_POW_EXPANSION_CAP > 0 and RATIONAL_POW_DEGREE_CAP > 0
    # the rational cap binds BOTH sides of aux^q = x^p
    assert rational_pow_problem(0.5 + RATIONAL_POW_DEGREE_CAP // 2) is not None
    assert rational_pow_problem(1.0 / (2 * RATIONAL_POW_DEGREE_CAP)) is not None
    # and the largest admissible degree really is admissible
    assert rational_pow_problem(1.0 / RATIONAL_POW_DEGREE_CAP) is None


# --- S2: no emitted term is ever a unary application of `*` ------------------


UNARY_STAR = re.compile(r"\(\*\s+[^()\s]+\s*\)")


def test_s2_repeated_product_is_well_formed_at_every_arity():
    assert smt._repeated_product("a", 0) == "1.0"
    assert smt._repeated_product("a", 1) == "a"
    assert smt._repeated_product("a", 2) == "(* a a)"
    assert smt._repeated_product("a", 5) == "(* a a a a a)"
    for n in range(0, 6):
        assert not UNARY_STAR.search(smt._repeated_product("aux_2", n))


def test_s2_a_unary_star_is_what_the_detector_detects():
    """The positive control for the scan below: without it, a scan that
    matched nothing would look like a passing test."""
    assert UNARY_STAR.search("(assert (= (* aux_2) 1.0))")
    assert not UNARY_STAR.search("(assert (= (* aux_2 aux_2) 1.0))")


def pow_query(exp, lo=1.0, hi=4.0, bound=1.5, cmp="le"):
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("pow", [x, lit(exp)], s),
            eqn(cmp, [s, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


@pytest.mark.parametrize(
    "e,bound",
    (
        (0.5, 1.5), (0.25, 1.2), (0.125, 1.1), (1.5, 1.5), (2.5, 1.5),
        (1.0, 1.5), (2.0, 1.5), (-1.0, 0.5), (3.0, 1.5),
    ),
)
def test_s2_no_emitted_pow_script_contains_a_unary_star(e, bound):
    item = sole_slice(pow_query(e, bound=bound))
    if isinstance(item, DeclinedObligation):
        pytest.fail(f"exponent {e!r} unexpectedly declined: {item.reason}")
    text = smt.emit(item, "z3", 5000).text
    assert not UNARY_STAR.search(text), (
        f"exponent {e!r} emitted a unary (* t):\n{text}"
    )


def test_s2_the_rational_branch_always_has_at_least_two_aux_factors():
    """``q == 1`` is what wrote ``(* aux)``. After S1 it is unreachable,
    and the emitted equation shows it: the left side is a genuine
    product."""
    item = sole_slice(pow_query(0.5))
    text = smt.emit(item, "z3", 5000).text
    assert "(* aux_1 aux_1)" in text, text


# --- S3 / M8: the replay is exact or it declines ------------------------------


def test_s3_exact_integer_root_is_exact_and_confirms_itself():
    assert _exact_integer_root(4, 2) == 2
    assert _exact_integer_root(0, 2) == 0
    assert _exact_integer_root(1, 128) == 1
    assert _exact_integer_root(2, 2) is None
    assert _exact_integer_root(1 << 768, 128) == 1 << 6
    assert _exact_integer_root(1 << 767, 128) is None
    # the operand a float root would get wrong: 2^110 + 1 is one above a
    # perfect square and rounds to it in binary64
    assert float((1 << 110) + 1) == float(1 << 110)
    assert _exact_integer_root((1 << 110) + 1, 2) is None


def test_s3_exact_rational_power_when_the_value_is_rational():
    assert _exact_rational_power(Fraction(9), 1, 2) == 3
    assert _exact_rational_power(Fraction(4), 3, 2) == 8
    assert _exact_rational_power(Fraction(1, 16), 3, 4) == Fraction(1, 8)
    assert _exact_rational_power(Fraction(0), 1, 2) == 0


def test_s3_exact_rational_power_declines_when_the_value_is_irrational():
    with pytest.raises(ReplayDeclined):
        _exact_rational_power(Fraction(2), 1, 2)
    with pytest.raises(ReplayDeclined):
        _exact_rational_power(Fraction((1 << 110) + 1, 1 << 108), 1, 2)


def test_s3_the_crashing_witness_now_replays_exactly_and_confirms():
    """The audit's crashing point. cvc5's model for ``x ** 0.5 <= 2.0``
    over a box starting just above 4 is a REAL violation and the emission
    (``aux^2 = x0, aux >= 0``) is exactly right — but ``float(w) ** 0.5``
    is ``2.0``, so the float replay called the predicate TRUE,
    ``witness_is_valid`` reported emission infidelity about a correct
    emission, and ``check()`` RAISED.

    This particular witness is ``((2^55 + 1) / 2^54)^2``, so its square
    root IS rational and the exact replay decides it outright — the
    refutation stands, no decline needed. Which is the point: the float
    was the only thing that could not do this arithmetic."""
    w = Fraction(1298074214633706979190218120232961, 324518553658426726783156020576256)
    assert w > 4  # strictly above 4, so sqrt(w) > 2 in the reals: a violation
    assert float(w) == 4.0  # and indistinguishable from 4 in binary64
    assert float(w) ** 0.5 == 2.0  # the float replay's answer, still wrong
    assert _exact_rational_power(w, 1, 2) == Fraction((1 << 55) + 1, 1 << 54) > 2
    item = sole_slice(pow_query(0.5, lo=4.0, hi=4.0000000000000009, bound=2.0))
    assert isinstance(item, ObligationSlice)
    assert witness_is_valid(item, {"x0": w}) is None  # a real refutation


def test_s3_an_irrational_witness_declines_instead_of_accusing_emit():
    """The other half of the same box, and the channel the fix turns on:
    ``4 + 2^-108`` is in the box and violates, but its square root is
    irrational. The replay must REFUSE — a ``ReplayDeclined`` the caller
    degrades to UNKNOWN — rather than return the emission-infidelity
    string, which is an accusation against a script that is correct."""
    w = Fraction((1 << 110) + 1, 1 << 108)
    assert 4 < w < 4.0000000000000009
    assert _exact_integer_root(w.numerator, 2) is None  # sqrt(w) is irrational
    item = sole_slice(pow_query(0.5, lo=4.0, hi=4.0000000000000009, bound=2.0))
    with pytest.raises(ReplayDeclined):
        witness_is_valid(item, {"x0": w})


def test_s3_a_rational_witness_still_replays_exactly():
    """The other direction, so the decline above is not the whole row:
    a witness whose exact value IS rational is decided, exactly."""
    item = sole_slice(pow_query(0.5, lo=1.0, hi=9.0, bound=2.0))
    assert witness_is_valid(item, {"x0": Fraction(9)}) is None  # 3 > 2: violated
    assert witness_is_valid(item, {"x0": Fraction(4)}) is not None  # 2 <= 2: holds


def test_m8_a_huge_rational_power_replays_without_overflowing():
    """``float(base) ** 1.5`` raised ``OverflowError`` on these operands
    and nothing caught it. Exact rational arithmetic has no overflow to
    raise."""
    base = Fraction(10) ** 250
    assert _exact_rational_power(base, 3, 2) == Fraction(10) ** 375
    with pytest.raises(ReplayDeclined):  # 10^249 is not a perfect square
        _exact_rational_power(Fraction(10) ** 249, 3, 2)


# --- M9: the fragment stamp follows the aux, not the base's dependence -------


def independent_base_pow_query(exp=0.5, bound=2.5):
    """A rational ``pow`` whose BASE descends from no declaration.

    ``c`` is a product of two literals, so it is declaration-independent
    and the predecessor's ``ins_dep[0]`` test was False — yet the emission
    still declares ``aux`` and writes ``aux^q = c^p``."""
    x, c, r, s, pred, out = (
        var(0), var(1), var(2), var(3), var(4, BOOL), var(5, BOOL),
    )
    return close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [lit(2.0), lit(2.0)], c),   # constant 4.0
            eqn("pow", [c, lit(exp)], r),          # aux^2 = 4.0
            eqn("add", [x, r], s),
            eqn("le", [s, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_m9_a_declaration_independent_rational_pow_is_stamped_nonlinear():
    item = sole_slice(independent_base_pow_query())
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    text = smt.emit(item, "z3", 5000).text
    assert "declare-const aux_" in text, text
    assert item.fragment == "QF_NRA", (
        f"the emission writes an aux product but the slice is stamped "
        f"{item.fragment}; both backends refuse such a script\n{text}"
    )
    assert "(set-logic QF_NRA)" in text


def test_m9_an_integer_pow_over_a_constant_base_stays_linear():
    """The control. Only the AUX encoding is unconditionally nonlinear; an
    integer exponent over a constant base folds to a numeral and QF_LRA is
    still the honest logic, which is what the predecessor's rule got right."""
    x, c, r, s, pred, out = (
        var(0), var(1), var(2), var(3), var(4, BOOL), var(5, BOOL),
    )
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("mul", [lit(2.0), lit(2.0)], c),
            eqn("pow", [c, lit(2.0)], r),
            eqn("add", [x, r], s),
            eqn("le", [s, lit(16.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = sole_slice(q)
    assert isinstance(item, ObligationSlice), getattr(item, "reason", item)
    assert item.fragment == "QF_LRA"
    assert "declare-const aux_" not in smt.emit(item, "z3", 5000).text


def test_m9_a_dependent_base_is_still_nonlinear_at_an_integer_exponent():
    assert sole_slice(pow_query(2.0, bound=3.0)).fragment == "QF_NRA"
    assert sole_slice(pow_query(1.0, bound=3.0)).fragment == "QF_LRA"


# --- end to end, with a solver ------------------------------------------------

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402

HAVE_SOLVER = (
    _optional.available("z3")
    or _optional.available("cvc5")
    or _optional.cvc5_binary() is not None
)
need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def exact_real_power(base: float, exponent: float) -> Decimal:
    """``base ** exponent`` at 120 significant digits, with both operands
    taken as the exact rationals their binary64 values denote. The
    independent oracle: no jax, no solver, no stelling."""
    getcontext().prec = 120
    b = Decimal(Fraction(base).numerator) / Decimal(Fraction(base).denominator)
    e = Decimal(Fraction(exponent).numerator) / Decimal(
        Fraction(exponent).denominator
    )
    return (e * b.ln()).exp()


@need_solver
def test_s1_the_headline_harness_no_longer_verifies():
    """``x in [1, 1e300]``, ``assert x**0.1 <= 1e30``.

    Both backends answered ``unsat`` about ``x^(1/10)`` and the verdict was
    VERIFIED, while the exact real value of the traced expression at the
    declared upper bound EXCEEDS the asserted bound. An ordinary exponent,
    a well-formed script, no degradation note."""
    T = 1e30
    truth = exact_real_power(1e300, 0.1)
    assert truth > Decimal(Fraction(T).numerator), (
        "the oracle no longer shows a violation; the test has stopped "
        "measuring what it was written for"
    )

    def h():
        x = any_array((), "float64", (1.0, 1e300))
        return (assert_(x ** 0.1 <= T),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=15_000)
    assert v.status != "VERIFIED", (
        f"UNSOUND: x**0.1 <= 1e30 certified over [1, 1e300] while the "
        f"exact real value at 1e300 is {truth:.6e}; notes: {v.notes}"
    )
    assert v.status == "UNKNOWN"
    assert any("3602879701896397/36028797018963968" in n for n in v.notes), (
        f"expected the decline to quote the exponent's exact value; "
        f"notes: {v.notes}"
    )


@need_solver
@pytest.mark.parametrize(
    "exp,lo,hi,bound",
    (
        (0.5000000000001, 1.0, 100.0, 10.0),      # q=2 half of S1
        (2.0000000000001, 1.0, 4.0, 16.0),        # S2's q==1 case
        (1e-13, 1.0, 1e300, 1.0000000000001),     # rationalised to 0/1
    ),
)
def test_s1_the_tolerance_half_no_longer_verifies(exp, lo, hi, bound):
    """The other admission route: exponents within 1e-12 of a small
    rational were snapped to it. Each of these was measured VERIFIED with
    a violation at the declared upper bound."""
    assert float(jnp_pow(hi, exp)) > bound, (
        "the construction no longer violates at the box maximum"
    )

    def h():
        x = any_array((), "float64", (lo, hi))
        return (assert_(x ** exp <= bound),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=15_000)
    assert v.status != "VERIFIED", (
        f"UNSOUND: x**{exp} <= {bound} certified over [{lo}, {hi}]; "
        f"notes: {v.notes}"
    )


def jnp_pow(x, e):
    return jnp.asarray(x, dtype=jnp.float64) ** e


@need_solver
def test_s2_the_unary_star_harness_declines_and_no_backend_is_blamed():
    """``x ** 2.0000000000001`` on [1, 4]: the rationalisation produced
    ``2/1`` and the emission wrote ``(assert (= (* aux_2) (* x0 x0)))``.
    cvc5's child process died with SIGSEGV, the parent reported a protocol
    violation, and the run attributed the failure to cvc5."""
    def h():
        x = any_array((), "float64", (1.0, 4.0))
        return (assert_(x ** 2.0000000000001 <= 16.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=10_000)
    assert v.status == "UNKNOWN", f"notes: {v.notes}"
    assert not any("protocol violation" in n for n in v.notes), (
        f"a stelling emission bug is still being reported as a backend "
        f"failure; notes: {v.notes}"
    )


@need_solver
def test_s3_the_replay_crash_harness_returns_a_verdict():
    """``check()``'s contract is that a transcription failure returns a
    Verdict, not an exception. This harness raised
    ``EmissionInfidelityError`` — from the replay, about an emission that
    was correct."""
    def h():
        x = any_array((), "float64", (4.0, 4.0000000000000009))
        return (assert_(x ** 0.5 <= 2.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=10_000)
    assert v.status in ("REFUTED", "UNKNOWN"), f"notes: {v.notes}"


@need_solver
def test_m8_the_overflow_harness_returns_a_verdict():
    """``x in [1e200, 1e250]``, ``x**1.5 <= 1e308``: the float replay
    raised ``OverflowError``, which is not in the replay's caught set, and
    the whole escalation was lost to it."""
    def h():
        x = any_array((), "float64", (1e200, 1e250))
        return (assert_(x ** 1.5 <= 1e308),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=15_000)
    assert v.status in ("REFUTED", "UNKNOWN"), f"notes: {v.notes}"
    assert not any("OverflowError" in n for n in v.notes), (
        f"the replay still overflows; notes: {v.notes}"
    )


@need_solver
def test_m9_the_jit_constant_base_harness_reaches_a_backend():
    """The audit's reachable form of M9: a constant base inside ``jit`` so
    jax cannot fold it eagerly. Stamped QF_LRA, the script carried
    ``(* aux aux)`` and BOTH backends refused it — the whole obligation
    lost, and the notes blamed the solvers."""
    def h():
        x = any_array((), "float64", (0.0, 1.0))

        @jax.jit
        def f(cc, d):
            return jnp.sum(cc ** 0.5) + d * 0.0

        return (assert_(x + f(jnp.array([4.0, 9.0]), x) <= 6.0),)

    v = check(h, vacuity_mode="inputs-only", solver_timeout_ms=15_000)
    assert not any("does not support nonlinear" in n for n in v.notes), (
        f"the slice is still stamped with a logic its own script "
        f"violates; notes: {v.notes}"
    )
    assert v.status == "VERIFIED", f"notes: {v.notes}"
