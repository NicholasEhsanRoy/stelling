# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""SMT-LIB2 emission: golden scripts, exact rationals, determinism — no jax.

The script is the interchange artifact and the core soundness surface:
these tests pin the emitted text byte-for-byte (golden), the exact-dyadic
literal discipline (never a decimal approximation), bound closure and the
half-infinite/⊤ special cases, predicate negation, sharing (one input =
one constant), the inertness of ``stelling_assume`` in the emitted
hypothesis, the per-solver option blocks (a solver is never invoked on
defaults; cvc5's ``nl-cov``/``nl-ext`` exclusivity is pinned both ways),
and the recorded script hash.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest

from stelling import ir
from stelling.obligation import ObligationSlice, slice_unknown_obligations
from stelling.propagate import interval_env, propagate
from stelling.smt import emit, rational
from test_obligation_slice import any_eqn, close, eqn, lit, var, BOOL, F64


def sole_slice(q) -> ObligationSlice:
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], ObligationSlice)
    return items[0]


def test_rational_literals_are_exact_dyadic_never_decimal():
    assert rational(Fraction(2)) == "2.0"
    assert rational(Fraction(-2)) == "(- 2.0)"
    assert rational(Fraction(1, 2)) == "(/ 1 2)"
    assert rational(Fraction(-1, 2)) == "(- (/ 1 2))"
    # 0.1 emits its exact f64 dyadic value, not the decimal it looks like
    assert rational(Fraction(0.1)) == "(/ 3602879701896397 36028797018963968)"
    # 10.125 = 81/8 exactly
    assert rational(Fraction(10.125)) == "(/ 81 8)"


def test_golden_script_z3_flavor():
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    script = emit(sole_slice(q), "z3", 750)
    assert script.text == (
        "; stelling escalation: obligation #0 (QF_NRA)\n"
        "(set-option :produce-models true)\n"
        "(set-option :timeout 750)\n"
        "(set-logic QF_NRA)\n"
        "(declare-const x0 Real)\n"
        "(assert (<= 1.0 x0))\n"
        "(assert (<= x0 2.0))\n"
        "(define-fun t1 () Real (* x0 x0))\n"
        "(define-fun t2 () Bool (<= t1 2.0))\n"
        "(assert (not t2))\n"
        "(check-sat)\n"
        "(get-model)\n"
    )
    assert script.options == ((":produce-models", "true"), (":timeout", "750"))
    assert script.sha256 == hashlib.sha256(script.text.encode()).hexdigest()
    assert ("smt2_sha256", script.sha256) in script.stamp_options()
    assert ("set-logic", "QF_NRA") in script.stamp_options()


def test_cvc5_nra_options_pin_coverings_and_disable_nl_ext():
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    script = emit(sole_slice(q), "cvc5", 500)
    assert "(set-option :nl-cov true)" in script.text
    assert "(set-option :nl-ext none)" in script.text  # never both engines
    assert "(set-option :tlimit 500)" in script.text
    assert "(set-option :produce-models true)" in script.text


def test_cvc5_lra_options_have_no_nonlinear_engine_options():
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("add", [x, lit(0.25)], s),
            eqn("le", [s, lit(0.75)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    script = emit(sole_slice(q), "cvc5", 500)
    assert script.logic == "QF_LRA"
    assert "nl-cov" not in script.text and "nl-ext" not in script.text
    assert script.options == ((":produce-models", "true"), (":tlimit", "500"))


def test_bounds_are_closed_and_halfinfinite_emits_only_the_finite_side():
    a, b, c, s1, s2, pred, out = (
        var(0), var(1), var(2), var(3), var(4), var(5, BOOL), var(6, BOOL),
    )
    q = close(
        [
            any_eqn(a, 0.0, math.inf),
            any_eqn(b, -math.inf, math.inf),
            any_eqn(c, -math.inf, 1.0),
            eqn("add", [a, b], s1),
            eqn("add", [s1, c], s2),
            eqn("le", [s2, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    bound_lines = [ln for ln in text.splitlines() if ln.startswith("(assert (<=")]
    # a: lower side only; b (⊤): no bound constraint at all; c: upper only
    assert bound_lines == ["(assert (<= 0.0 x0))", "(assert (<= x2 1.0))"]
    assert "(declare-const x1 Real)" in text  # ⊤ input is still declared
    assert "inf" not in text  # infinities never reach the script


def test_strictness_is_preserved_lt_vs_le():
    def q(cmp):
        x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
        return close(
            [
                any_eqn(x, 0.0, 1.0),
                eqn(cmp, [x, lit(0.5)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [out],
        )

    assert "(< x0 (/ 1 2))" in emit(sole_slice(q("lt")), "z3", 100).text
    assert "(<= x0 (/ 1 2))" in emit(sole_slice(q("le")), "z3", 100).text
    assert "(> x0 (/ 1 2))" in emit(sole_slice(q("gt")), "z3", 100).text
    assert "(>= x0 (/ 1 2))" in emit(sole_slice(q("ge")), "z3", 100).text


def test_query_asserts_the_negation_of_the_predicate():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(assert (not t1))" in text
    assert "(define-fun t1 () Bool (<= x0 (/ 1 2)))" in text


def test_sharing_is_preserved_one_input_one_constant():
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert text.count("(declare-const") == 1
    assert "(* x0 x0)" in text  # both uses are the same constant


def test_assume_constraint_is_never_emitted():
    # x > 0.9 is assumed; emitting it would silently strengthen the
    # hypothesis relative to the propagated (assume-inert) semantics.
    x, apred, ap2, pred, out = var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("gt", [x, lit(0.9)], apred),
            eqn("stelling_assume", [apred], ap2),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "0.9" not in text and str(Fraction(0.9).numerator) not in text
    # exactly two asserts: the two box bounds; plus the negated predicate
    assert text.count("(assert ") == 3


def test_max_min_select_emit_as_ite():
    x, y, mx, pred, out = var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            any_eqn(y, 0.0, 1.0),
            eqn("max", [x, y], mx),
            eqn("le", [mx, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = emit(sole_slice(q), "z3", 100).text
    assert "(ite (>= x0 x1) x0 x1)" in text


def test_select_n_case_order_false_then_true():
    # (a boolean *input* declaration would decline in v1, so the boolean
    # selector comes from a comparison)
    x2, y2, c, w2, sel2, pred2, out2 = (
        var(10), var(11), var(12), var(13, BOOL), var(14), var(15, BOOL), var(16, BOOL),
    )
    q = close(
        [
            any_eqn(x2, 0.0, 1.0),
            any_eqn(y2, 2.0, 3.0),
            any_eqn(c, 0.0, 1.0),
            eqn("gt", [c, lit(0.5)], w2),
            eqn("select_n", [w2, x2, y2], sel2),
            eqn("le", [sel2, lit(2.5)], pred2),
            eqn("stelling_assert", [pred2], out2),
        ],
        [out2],
    )
    text = emit(sole_slice(q), "z3", 100).text
    # select_n(which, on_false, on_true) must emit (ite which on_true on_false)
    assert "(ite t13 x1 x0)" in text


def test_integer_pow_expands_to_products():
    x, cube, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("integer_pow", [x], cube, [("y", 3)]),
            eqn("le", [cube, lit(4.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    assert "(* x0 x0 x0)" in emit(sole_slice(q), "z3", 100).text


def test_emission_is_deterministic_and_hash_stable():
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    first = emit(sole_slice(q), "cvc5", 250)
    second = emit(sole_slice(q), "cvc5", 250)
    assert first.text == second.text
    assert first.sha256 == second.sha256


def test_timeout_must_be_positive():
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 0.0, 1.0),
            eqn("le", [x, lit(0.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    with pytest.raises(ValueError):
        emit(sole_slice(q), "z3", 0)
    with pytest.raises(ValueError):
        emit(sole_slice(q), "nosuch", 100)
