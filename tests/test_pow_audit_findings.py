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

Plus the follow-up repairs to the S3 fix itself, in their own section: the
``ReplayError`` / ``ReplayDeclined`` split decides WHO IS ACCUSED and
nothing downstream re-derives it, so the channel of each refusal is pinned
by type rather than by message text, its decline messages are pinned
against crashing on an unbounded model operand, and the dispatch layer's
``except ReplayDeclined`` is driven by the one transport shape that is not
preempted by the installed wheels.

Everything above the solver divider runs with no jax and no real solver.
"""

from __future__ import annotations

import math
import re
import sys
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
    ReplayError,
    SliceInput,
    _exact_integer_root,
    _exact_rational_power,
    evaluate_predicate,
    pow_exponent_rational,
    rational_pow_problem,
    witness_is_valid,
)
from stelling.propagate import propagate
from stelling.solvers import (
    EmissionInfidelityError,
    SolverConfig,
    _require_valid_refutation,
    escalate,
    make_solver_verdict,
    make_validated_witness,
)
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, sole_slice, var
from test_solver_dispatch import VERSIONS, fake_solver


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


def test_the_admitted_exponent_set_is_the_448_pairs_the_cap_comment_swept():
    """The admitted set is small enough to enumerate, and the cap's
    comment now reports a sweep over ALL of it rather than an extremum
    over a set nobody counted (the previous text named ``127/128`` "the
    worst admitted case"; ``105/128`` is 49% worse, and both were reached
    only by measuring the whole set).

    This pins the set the comment claims to have covered — 7 reachable
    denominators x 64 numerators = 448 pairs — so raising the cap, or
    admitting a non-dyadic exponent, cannot leave those figures stale and
    silent. Solver-free: it is the admission guard that is enumerated,
    not the sweep."""
    denominators = tuple(1 << k for k in range(1, 8))
    admitted: set[tuple[int, int]] = set()
    for q in denominators:
        for p in range(1, RATIONAL_POW_DEGREE_CAP + 1):
            if Fraction(p, q).denominator != q:
                continue  # not in lowest terms: a smaller-q pair, counted there
            e = p / q
            assert Fraction(e) == Fraction(p, q), (
                f"{p}/{q} is not the exact value of the literal {e!r}, so it "
                f"is not a pair the guard can admit"
            )
            assert rational_pow_problem(e) is None, (p, q, e)
            assert max(p, q) <= RATIONAL_POW_DEGREE_CAP
            admitted.add((p, q))
    assert sorted({q for _, q in admitted}) == list(denominators) == [
        2, 4, 8, 16, 32, 64, 128
    ]
    assert all(
        sum(1 for _, qq in admitted if qq == q) == 64 for q in denominators
    )
    assert len(admitted) == 448
    # and the pair the comment reports as the worst, plus the one it used
    # to name, are both in it — with 127/2 (degree 127) showing that the
    # cost the comment reports does NOT track the degree the cap bounds
    assert {(105, 128), (127, 128), (127, 2)} <= admitted


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


# --- S3's two channels stay two channels -------------------------------------
#
# `ReplayDeclined` was introduced by the S3 fix to separate two things the
# replay had been conflating. Which channel a refusal takes is a decision
# about WHO IS ACCUSED, and nothing downstream re-derives it: a `ReplayError`
# is an alarm about the EMISSION and reaches the caller as a raise, while a
# `ReplayDeclined` costs a REFUTED and lands as UNKNOWN. The two tests below
# pin the channel of the two refusals that are easiest to move by editing one
# word, because reverting either word leaves the rest of the suite green.


F32 = ir.Aval(kind="ShapedArray", shape=(), dtype="float32")


def narrowing_convert_slice():
    """A slice whose only real work is ``float64 -> float32``, built by
    hand because NOTHING IN THE PIPELINE WILL BUILD IT: the interval
    transfer refuses a value-changing conversion
    (``propagate._EXACT_CONVERSIONS``) and the admission guard declines
    the slice again on the same membership test
    (``obligation._Slicer._validate``). Constructing the slice directly is
    the honest way to say what is being simulated — a hole in those
    guards, nothing more — and it simulates exactly the hole the audit
    opened by neutering ``_validate``."""
    x, y, pred = var(0), var(1, F32), var(2, BOOL)
    return ObligationSlice(
        index=0,
        fragment="QF_LRA",
        inputs=(SliceInput(name="x0", var_id=0, lo=1.0, hi=2.0),),
        consts=(),
        eqns=(
            eqn("convert_element_type", [x], y,
                params=[("new_dtype", "float32")]),
            # 5.0, NOT the box's 2.0. The two must differ or the emission
            # assertion below cannot discriminate: with both at 2.0 the
            # substring it looks for also matches the box constraint line
            # `(assert (<= x0 2.0))`, which is emitted whatever the
            # conversion does — so the assertion passed even with the
            # identity replaced, and its failure message described
            # something it could not detect.
            eqn("le", [y, lit(5.0)], pred),
        ),
        root=pred,
        source_info=(),
    )


def test_a_value_changing_conversion_accuses_the_script():
    """The replay's refusal of a value-changing ``convert_element_type``
    is a ``ReplayError`` — an alarm about the emission — and never the
    ``ReplayDeclined`` subclass.

    The reason is on the emission side and is asserted here too: a
    non-bool ``convert_element_type`` emits as the IDENTITY on its
    operand, so a script carrying a ``float64 -> float32`` narrowing has
    the rounding simply ABSENT and states a different function from the
    one the harness computes. A witness reaching the replay through such
    a script is evidence the script was wrong, which is what
    ``ReplayError`` means; ``ReplayDeclined`` means the opposite — a
    correct emission the replay cannot finish — and routes to UNKNOWN
    without raising.

    THE PATH IS UNREACHABLE TODAY: two independent guards decline this
    slice before any solver sees it (see ``narrowing_convert_slice``).
    This test exists because that is exactly what makes the raise a
    tripwire — a demotion to ``ReplayDeclined`` costs nothing until the
    day a guard has a hole, and then it costs the alarm. Reverting the
    one word leaves the whole rest of the suite green."""
    sl = narrowing_convert_slice()
    with pytest.raises(ReplayError) as exc:
        evaluate_predicate(sl, {"x0": Fraction(3, 2)})
    assert not isinstance(exc.value, ReplayDeclined), (
        f"the value-changing-conversion refusal has been demoted to the "
        f"decline channel: it would now degrade to UNKNOWN instead of "
        f"raising, and the emission for this conversion is the identity — "
        f"so a witness here accuses the script, not the replay "
        f"({exc.value})"
    )

    # THE EMISSION FACT THE CHANNEL RESTS ON. The narrowing is not
    # approximated in the script, or flagged in it — it is absent: the
    # predicate is written directly over the declared input, so the
    # script says `x0 <= 5` about a program that says `f32(x0) <= 5`.
    # The bound is the predicate's, distinct from the box's, so this
    # matches the CONVERTED term's line and nothing else.
    script = smt.emit(sl, "z3", 5_000).text
    assert "(<= x0 5.0)" in script, (
        f"the emission no longer writes a narrowing convert_element_type as "
        f"the identity; re-derive which channel the replay's refusal belongs "
        f"to before changing it\n{script}"
    )


def test_a_decline_message_survives_an_unrenderable_operand():
    """A refusal message may not crash on the operand it is refusing.

    Both refusals in ``_exact_rational_power`` interpolate the base, and
    the base is a SOLVER MODEL VALUE — nothing bounds it. CPython raises
    ``ValueError`` on ``int`` -> ``str`` past
    ``sys.get_int_max_str_digits()`` (4300), so ``Fraction(3**10000, 2)``
    turned a clean decline into a crash out of ``evaluate_predicate`` and
    ``witness_is_valid``, both public. Same hazard as
    ``smt._renderable``, same posture: detect by attempting the
    conversion, report magnitude instead of digits, never mutate the
    process-global limit."""
    limit = sys.get_int_max_str_digits()
    huge = Fraction(3 ** 10000, 2)
    # over the cap, whatever the cap is set to (never computed by str())
    assert huge.numerator.bit_length() / math.log2(10) > limit
    with pytest.raises(ValueError):
        str(huge.numerator)  # the crash this message used to inherit

    # the irrational refusal: 3^10000 IS a perfect square, the denominator 2
    # is not, so this declines with both terms in the message
    with pytest.raises(ReplayDeclined) as irr:
        _exact_rational_power(huge, 1, 2)
    assert f"{huge.numerator.bit_length()}-bit integer" in str(irr.value)

    # the negative-base refusal, which short-circuits before the roots
    with pytest.raises(ReplayDeclined) as neg:
        _exact_rational_power(-huge, 1, 2)
    assert f"-<{huge.numerator.bit_length()}-bit integer>" in str(neg.value)

    # and through the public replay surface, which is what a caller sees
    x, root, pred = var(0), var(1), var(2, BOOL)
    sl = ObligationSlice(
        index=0,
        fragment="QF_NRA",
        inputs=(SliceInput(name="x0", var_id=0, lo=0.0, hi=float("inf")),),
        consts=(),
        eqns=(
            eqn("pow", [x, lit(0.5)], root),
            eqn("le", [root, lit(2.0)], pred),
        ),
        root=pred,
        source_info=(),
    )
    with pytest.raises(ReplayDeclined):  # a decline, NOT a ValueError
        evaluate_predicate(sl, {"x0": huge})

    assert sys.get_int_max_str_digits() == limit, (
        "the replay raised the process-global int->str limit; that is a "
        "library mutating a caller's interpreter (smt._renderable)"
    )


def test_the_box_escape_alarm_survives_an_unrenderable_model_value():
    """The LOUD alarm's own message may not crash on the model it accuses.

    ``witness_is_valid``'s box-membership conjunct interpolates the model
    value, and a model value is unbounded — the same ``int`` -> ``str``
    hazard as the decline messages above, but on the *worse* side of the
    channel line. A decline that crashes costs an UNKNOWN; THIS message is
    the diagnosis of "the emitted problem does not mean the obligation",
    so a ``ValueError`` here replaces the one explanation a reader gets
    with a traceback out of ``fractions.py`` and leaves an
    emission-infidelity report unattributable.

    The declared bound cannot reach the cap (it is ``Fraction`` of a
    binary64), so only the model value can trip this — which is why it
    outlived the decline-message pass: the hazard is on the operand
    nobody controls."""
    limit = sys.get_int_max_str_digits()
    huge = Fraction(3 ** 10000, 2)
    assert huge.numerator.bit_length() / math.log2(10) > limit

    x, pred = var(0), var(1, BOOL)
    sl = ObligationSlice(
        index=0,
        fragment="QF_LRA",
        inputs=(SliceInput(name="x0", var_id=0, lo=0.0, hi=1.0),),
        consts=(),
        eqns=(eqn("le", [x, lit(2.0)], pred),),
        root=pred,
        source_info=(),
    )
    # the model escapes the declared box [0, 1]: the emission-infidelity
    # conjunct, which must RETURN a string rather than raise
    why = witness_is_valid(sl, {"x0": huge})
    assert why is not None
    assert "escapes the declared box" in why
    assert f"{huge.numerator.bit_length()}-bit integer" in why
    assert "above its declared upper bound 1" in why  # the bound still renders

    assert sys.get_int_max_str_digits() == limit


def test_the_dispatch_layer_renders_a_model_value_through_the_same_renderer():
    """Rendering a solver model value is ONE discipline, and `solvers.py`
    is on the other side of the module boundary from it.

    ``witness_is_valid`` was made safe first, and that was not enough: the
    box-escape alarm assembled its diagnosis without raising and then
    ``_require_valid_refutation`` died on the very next statement,
    stringifying the same values to attach them to the exception. The
    success path had it too — a replay-confirmed REFUTED crashed while its
    ``Witness`` was being built. Both are the loud paths: one is the alarm
    that means the emitted problem does not mean the obligation, the other
    is a correct refutation.

    ``fraction_text`` is public for exactly this reason. Nothing else in
    ``src/`` imports a private name across modules, so the alternative to
    exporting it was a second renderer — and a second renderer is how the
    first one came to be missing here."""
    huge = Fraction(3 ** 10000, 2)
    x, pred = var(0), var(1, BOOL)
    eqns = (eqn("le", [x, lit(2.0)], pred),)
    bits = huge.numerator.bit_length()

    # the ALARM path: the model escapes [0, 1], so this must raise the
    # emission-infidelity error carrying the values, not a ValueError
    escaping = ObligationSlice(
        index=0, fragment="QF_LRA",
        inputs=(SliceInput(name="x0", var_id=0, lo=0.0, hi=1.0),),
        consts=(), eqns=eqns, root=pred, source_info=(),
    )
    with pytest.raises(EmissionInfidelityError) as exc:
        _require_valid_refutation(
            escaping, {"x0": huge}, solver_label="probe", script_text=""
        )
    assert exc.value.values == (("x0", f"<{bits}-bit integer>/2"),)

    # the SUCCESS path is NOT the same problem, and must not get the same
    # answer. `Witness.values` is DATA — `reproduce._point` parses it back
    # with `Fraction()` to re-execute the harness — so a summarised value
    # would name a different point and break that contract on another
    # module's public surface. Fail closed: decline, so the dispatch
    # degrades to UNKNOWN with the reason quoted (the caller's
    # `except ReplayDeclined`), rather than minting a REFUTED whose
    # witness is not the value the solver produced.
    unbounded = ObligationSlice(
        index=0, fragment="QF_LRA",
        inputs=(SliceInput(name="x0", var_id=0, lo=0.0, hi=float("inf")),),
        consts=(), eqns=eqns, root=pred, source_info=(),
    )
    with pytest.raises(ReplayDeclined) as declined:
        make_validated_witness(
            unbounded, {"x0": huge}, "probe",
            solver_label="probe", script_text="",
        )
    assert "cannot be recorded exactly" in str(declined.value)

    # and the refutation it declined to witness IS real — the decline is
    # about recording the point, not about the point being wrong
    assert witness_is_valid(unbounded, {"x0": huge}) is None

    # a renderable model still produces a witness whose values round-trip
    small = make_validated_witness(
        unbounded, {"x0": Fraction(7, 2)}, "probe",
        solver_label="probe", script_text="",
    )
    assert small.values == (("x0", "7/2"),)
    assert Fraction(small.values[0][1]) == Fraction(7, 2)


def test_a_declined_replay_reaches_its_handler_and_stays_unknown(
    monkeypatch, tmp_path
):
    """The dispatch layer's ``except ReplayDeclined`` (solvers.py) is
    REACHABLE, and this is the case that reaches it.

    On the `pow` row the two INSTALLED wheels preempt it: the model
    carries the ``aux`` constant, ``aux`` is algebraic exactly when the
    exact value is irrational, and both wheels flag an algebraic model
    value as ``nonrational`` a hundred lines earlier. What does NOT
    preempt it is a transport that reports a rational-looking model
    WITHOUT flagging — which is the shape of the external cvc5-binary
    transport, not installed here and never driven by a real algebraic
    model. The fake below is exactly that transport: ``sat`` with
    ``x0 = 2`` on ``x**0.5 <= 1.0`` over ``[1, 9]``. The point is a
    genuine violation (``sqrt 2 > 1``) and the emission ``aux^2 = x0,
    aux >= 0`` is exactly right, so the replay's refusal is about the
    REPLAY and must not become a raise, a REFUTED, or a fabricated
    witness — it is an UNKNOWN quoting why."""
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0 () Real 2.0)")\n'
        'print(")")',
        "cvc5-rational-but-irrational-root",
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = pow_query(0.5, lo=1.0, hi=9.0, bound=1.0)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "unknown", (
        f"a point the replay DECLINED became {record.outcome!r}: "
        f"{record.detail}"
    )
    assert "not independently replayable" in record.detail
    # ...through the DECLINE handler specifically, not the `nonrational`
    # branch above it: the reason quoted is the replay's own refusal, which
    # the `nonrational` branch never produces (it says "model contains a
    # non-rational value" and never names the operand)
    assert "is irrational" in record.detail, record.detail
    assert record.witness is None
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN" and v.witnesses == ()


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
