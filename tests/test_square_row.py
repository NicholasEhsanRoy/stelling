# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The `square` EMISSION row — hand-built IR, no jax.

`square` already had an interval transfer (`propagate._t_square`, tier
``sound``) and an ieee transfer that declines. What it had no rule for was
the obligation face: a slice that traversed one declined with ``primitive
'square' is outside the supported emission set``, so a property genuinely
false over the declared box could not reach the solver and came back as an
honest-but-unactionable UNKNOWN.

This file pins the emission row and only the emission row: the four
registries `square` now appears in (read from the definitions), the
SELF-PRODUCT body it emits, the ``QF_NRA`` fragment that body puts the
problem in, the exact-rational replay that confirms a model, the
integer-dtype refusal, and the boolean-operand refusal. The end-to-end
verdicts, the containment against jax eager and under ``jit``, and the
mutation battery live in ``tests/test_square_row_gauge_jax.py`` — that file
needs jax and a solver; this one must run in the zero-dep arm.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import stelling.obligation as OB
import stelling.smt as SM
from stelling import ir
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    evaluate_predicate,
    slice_unknown_obligations,
    violating_elements,
    witness_is_valid,
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


def _query(xb, bound, cmp="le", *, dtype=F64, shape=()):
    """``assert_(square(x) - x <cmp> bound)`` over ``x`` declared on ``xb``.

    The subtraction is deliberate: ``square(x)`` alone against a constant is
    decided by interval propagation on many boxes, and an obligation the
    cheap layer settles never reaches the stage this file is about."""
    a = ir.Aval(kind="ShapedArray", shape=shape, dtype=dtype.dtype)
    scalar = ir.Aval(kind="ShapedArray", shape=(), dtype=dtype.dtype)
    b = ir.Aval(kind="ShapedArray", shape=shape, dtype="bool")
    x, s, d, pred, out = var(0, a), var(1, a), var(2, a), var(3, b), var(4, b)
    return close(
        [
            any_eqn(x, *xb),
            eqn("square", [x], s),
            eqn("sub", [s, x], d),
            eqn(cmp, [d, lit(bound, scalar)], pred),  # broadcast scalar bound
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _slice_of(q):
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1
    return items[0]


# --- registration / census, read from the definitions ------------------------


def test_square_is_registered_in_every_registry_the_row_needs():
    """Printed from the live sets, never recalled. Emission and replay must
    BOTH carry it (a primitive emittable but not replayable yields a witness
    nobody can independently confirm), and the int classification must be
    total over the emission set or the import raises."""
    assert "square" in OB._SUPPORTED
    assert "square" in OB._REPLAY_SUPPORTED
    assert "square" in OB._INT_OVERFLOW_EMITTED
    assert "square" not in OB._INT_SAFE_EMITTED  # it COMPUTES; it is guarded
    # the transfer face was already there; this row did not touch it
    assert TRANSFERS["square"][1] == "sound"
    assert IEEE_TRANSFERS["square"][1] == "exact"
    assert "square" in _INT_COMPUTING


def test_the_import_time_censuses_still_hold_with_square_in_them():
    """Both import-time raises are re-run over the live sets, so a future
    edit that adds an emittable primitive without classifying it fails here
    as well as at import.

    WHAT THE THIRD ASSERTION DOES NOT ESTABLISH — measured, because it
    reads like it establishes more than it does. ``_REPLAY_SUPPORTED`` and
    ``_SUPPORTED`` are the SAME expression over ``_ARITH``, so a primitive
    added via ``_ARITH`` — which is how `square` was added — joins both
    simultaneously and the equality can never fail for it. Measured: with
    the ``elif prim == "square"`` branch DELETED from ``_root_elements``,
    ``stelling.obligation`` still imports cleanly and this assertion still
    passes; the deletion is caught only behaviourally, by 12 failures
    across this file, ``test_square_row_gauge_jax.py`` and
    ``test_square_acceptance_jaxfluids.py``. So the replay coverage of this
    row is established by those probes, NOT by this line. Pre-existing and
    shared: the same blind spot covers every other ``_ARITH`` member. Left
    as found — closing it is a change to the census's own shape and needs
    its own round."""
    assert OB._INT_OVERFLOW_EMITTED | OB._INT_SAFE_EMITTED == OB._SUPPORTED
    assert not (OB._INT_OVERFLOW_EMITTED & OB._INT_SAFE_EMITTED)
    assert OB._REPLAY_SUPPORTED == OB._SUPPORTED
    OB._assert_emission_classification_censused()  # square is guarded, not excused
    # the blind spot, pinned rather than described: both names resolve
    # through _ARITH, so removing `square` from one removes it from both
    assert "square" in OB._ARITH
    assert OB._SUPPORTED >= OB._ARITH and OB._REPLAY_SUPPORTED >= OB._ARITH


def test_the_row_did_not_widen_the_emission_set_past_square():
    """A negative control on the registry itself: the adjacent nonlinear
    rows stay out, so "square emits" is a statement about square."""
    for prim in ("sqrt", "exp", "pow", "abs", "sign", "rem", "log"):
        assert prim not in OB._SUPPORTED, prim


# --- the slice: reach, and the fragment the body puts it in ------------------


def test_the_slice_is_reached_rather_than_declined():
    item = _slice_of(_query((-2.0, 3.0), 5.0))
    assert isinstance(item, ObligationSlice)
    assert not isinstance(item, DeclinedObligation)
    assert [e.primitive for e in item.eqns] == ["square", "sub", "le"]


def test_a_square_of_a_declared_input_is_nonlinear():
    """`square` has no exponent param, so unlike ``integer_pow`` there is no
    y in (0, 1) to fall back to a linear case with: a dependent operand is
    unconditionally QF_NRA."""
    assert _slice_of(_query((-2.0, 3.0), 5.0)).fragment == OB.QF_NRA


def _constant_operand_query(eqns_for_unop):
    """``<unop>(2.0 + 1.0) + x <= 10``, with the unop's operand a
    constant-only subtree so it does not reach a declaration.
    ``9 + [-2, 3] = [7, 12]`` straddles 10, so the cheap layer cannot
    settle it and the slice really is built."""
    x, c, s, d, pred, out = (
        var(0), var(1), var(2), var(3), var(4, BOOL), var(5, BOOL)
    )
    return close(
        [
            any_eqn(x, -2.0, 3.0),
            eqn("add", [lit(2.0), lit(1.0)], c),  # constant-only subtree
            eqns_for_unop(c, s),
            eqn("add", [s, x], d),
            eqn("le", [d, lit(10.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


_SELF_PRODUCT_SPELLINGS = {
    "square": lambda c, s: eqn("square", [c], s),
    "mul": lambda c, s: eqn("mul", [c, c], s),
    "integer_pow": lambda c, s: eqn("integer_pow", [c], s, (("y", 2),)),
}


def test_a_constant_square_is_stamped_QF_LRA_and_emits_a_NONLINEAR_term():
    """MEASURED, and it is not what this test used to claim.

    The premise that stood here — "a square whose operand does not reach a
    declaration mints no nonlinear term, exactly as a constant product
    does" — is FALSE, and it is false in both halves, which is the point.
    ``_Slicer._fragment`` marks `square` nonlinear only when
    ``ins_dep[0]``, while :func:`stelling.smt._square_body` emits the
    self-product unconditionally. So this slice is stamped ``QF_LRA`` and
    the script it ships contains ``(* t1 t1)``. The consequence is in the
    next test, and it is not cosmetic.

    NOT A DEFECT OF THIS ROW, and deliberately not fixed here. Measured on
    main (b11f051), where `square` has no emission row at all, the
    identical script comes out of ``mul`` on two constant operands and out
    of ``integer_pow`` at y=2 — same ``(set-logic QF_LRA)``, same
    ``(define-fun t2 () Real (* t1 t1))``. `square` joins a class it did
    not create; ``_fragment`` is shared code this row does not own, and
    changing it needs its own gauge and its own round. The three-spelling
    parity below is what keeps that claim honest: if `square` ever diverges
    from its two siblings here, this test says so."""
    texts = {}
    for name, mk in _SELF_PRODUCT_SPELLINGS.items():
        q = _constant_operand_query(mk)
        p = propagate(q)
        assert p.obligations[0].status == "unknown", name
        (item,) = slice_unknown_obligations(q, p, interval_env(q))
        assert item.fragment == OB.QF_LRA, name
        text = SM.emit(item, "z3", 30_000).text
        assert "(set-logic QF_LRA)" in text, name
        assert "(define-fun t2 () Real (* t1 t1))" in text, (name, text)
        texts[name] = text
    # one class, not three behaviours: the bodies are byte-identical
    assert len(set(texts.values())) == 1, sorted(texts)


def test_the_mislabelled_logic_silently_costs_a_portfolio_member():
    """The consequence, measured — this is why the test above is not a
    naming quibble.

    The portfolio's design is that agreement decides and disagreement is a
    loud error. A script whose declared logic excludes the arithmetic it
    contains never gets to disagree: z3 REFUSES it outright, is recorded
    ``failed``, and the verdict is then decided by cvc5 alone — a
    one-backend REFUTED carrying a two-backend portfolio's stamp, whose
    ``set-logic`` names a fragment the problem is not in.

    The loss IS disclosed — both invocations ride in the ledger with their
    reasons and z3's error is quoted verbatim in the notes — so nothing is
    hidden. But nothing is flagged either, and "how many backends actually
    answered" is not a field any consumer reads.

    Recorded, not repaired, and shared: `mul` and `integer_pow` do the same
    on main."""
    pytest.importorskip("z3")
    pytest.importorskip("cvc5")
    from stelling.solvers import SolverConfig, escalate

    q = _constant_operand_query(_SELF_PRODUCT_SPELLINGS["square"])
    p = propagate(q)
    (record,) = escalate(q, p, SolverConfig(timeout_ms=30_000)).records
    notes = " | ".join(record.notes)
    assert "z3 (wheel) answered failed" in notes, notes
    assert "logic does not support nonlinear arithmetic" in notes, notes
    assert "cvc5 (wheel) answered sat" in notes, notes
    # two invocations stamped, ONE of them an answer, and the detail names
    # the fragment the problem is not in
    assert {s.name for s in record.invocations} == {"z3", "cvc5"}
    assert record.outcome == "violated-witness"
    assert "(QF_LRA)" in record.detail, record.detail


def test_element_terms_counts_the_square_output_once_per_element():
    item = _slice_of(_query((-2.0, 3.0), 5.0, shape=(3,)))
    # 3 declared input elements + 3 square + 3 sub + 3 le
    assert item.element_terms == 12


# --- the emitted body: ONE term, twice ---------------------------------------


def test_the_emitted_body_is_the_self_product_of_one_term():
    """The whole content of the row. Two occurrences of the SAME SMT
    constant, never two constants: the primitive has one operand, so the
    two factors are one value, and that correlation is exactly what the
    interval leg cannot see and the solver can."""
    assert SM._square_body("x0") == "(* x0 x0)"
    text = SM.emit(_slice_of(_query((-2.0, 3.0), 5.0)), "z3", 1000).text
    assert "(define-fun t1 () Real (* x0 x0))" in text
    assert "(set-logic QF_NRA)" in text
    # exactly one declared constant — a two-constant emission would be a
    # strictly weaker problem admitting points the program cannot reach
    assert text.count("(declare-const") == 1


def test_the_emitted_script_is_the_negated_predicate_over_the_closed_box():
    text = SM.emit(_slice_of(_query((-2.0, 3.0), 5.0)), "z3", 1000).text
    assert "(assert (<= (- 2.0) x0))" in text
    assert "(assert (<= x0 3.0))" in text
    assert "(assert (not t3))" in text


def test_an_array_square_emits_one_body_per_element():
    text = SM.emit(_slice_of(_query((-2.0, 3.0), 5.0, shape=(3,))), "z3", 1000).text
    for i in range(3):
        assert f"(define-fun t1_{i} () Real (* x0_{i} x0_{i}))" in text


# --- replay: the exact-rational leg that promotes a model to a witness -------


def test_replay_reproduces_the_predicate_exactly():
    sl = _slice_of(_query((-2.0, 3.0), 5.0))
    (name,) = [i.name for i in sl.inputs]
    for q, expected in (
        (Fraction(-2), False),   # 4 + 2 = 6 > 5
        (Fraction(3), False),    # 9 - 3 = 6 > 5
        (Fraction(0), True),
        (Fraction(5, 2), True),  # 25/4 - 5/2 = 15/4 <= 5
    ):
        assert evaluate_predicate(sl, {name: q}) is expected, q


def test_replay_is_exact_where_a_float_replay_would_decide_the_other_way():
    """The replay's reason for existing, with a MAGNITUDE.

    ``(2**27 + 1)**2 = 18014398777917441`` needs 54 significand bits, so no
    binary64 holds it: the float square rounds DOWN by exactly 1. At the
    bound ``18014398643699712.0`` the exact rational says the predicate is
    FALSE and a float replay would say TRUE — so this point is decided by
    the last bit, and the row's replay decides it in the arithmetic that
    has the bit."""
    x, s, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    bound = 18014398777917440.0
    q = close(
        [
            any_eqn(x, 0.0, 2e8),
            eqn("square", [x], s),
            eqn("le", [s, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    sl = _slice_of(q)
    (name,) = [i.name for i in sl.inputs]
    big = Fraction(2**27 + 1)
    assert OB._square_value(big) == Fraction(18014398777917441)  # exact
    assert float(big) ** 2 == bound  # the float square, one low
    assert evaluate_predicate(sl, {name: big}) is False
    assert witness_is_valid(sl, {name: big}) is None  # a real refutation


def test_a_witness_is_validated_on_both_conjuncts():
    sl = _slice_of(_query((-2.0, 3.0), 5.0))
    (name,) = [i.name for i in sl.inputs]
    assert witness_is_valid(sl, {name: Fraction(3)}) is None  # in box, violates
    # in the box but does NOT violate: the violation conjunct must refuse
    assert witness_is_valid(sl, {name: Fraction(0)}) is not None
    # violates but is OUTSIDE the declared box: membership must refuse
    assert witness_is_valid(sl, {name: Fraction(10)}) is not None


def test_violating_elements_names_the_failing_elements_of_an_array_square():
    sl = _slice_of(_query((-2.0, 3.0), 5.0, shape=(3,)))
    names = [i.name for i in sl.inputs]
    values = dict(zip(names, (Fraction(3), Fraction(0), Fraction(-2))))
    assert violating_elements(sl, values) == (0, 2)
    assert evaluate_predicate(sl, values) is False


def test_replay_and_emission_agree_on_the_same_body():
    """EMISSION == REPLAY, at the expression level rather than the set
    level: both faces are one named seam each, and they compute the same
    function of one argument."""
    for v in (Fraction(-3), Fraction(0), Fraction(7, 5), Fraction(11, 3)):
        assert OB._square_value(v) == v * v
    assert SM._square_body("t9") == "(* t9 t9)"


# --- the refusals ------------------------------------------------------------


def test_an_integer_square_declines_rather_than_relaxing_onto_reals():
    """SMT-LIB2 Reals are unbounded and jax integer arithmetic wraps, so a
    Real emission of an integer square would let the solver prove a claim
    the program falsifies. The emission is stricter than the transfer here
    on purpose."""
    x, s, pred, out = var(0, I32), var(1, I32), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 2_000_000_000.0, 2_100_000_000.0),
            eqn("square", [x], s),
            eqn("ge", [s, lit(0, I32)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"  # the transfer refused it too
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert "'square' on dtype 'int32'" in item.reason
    assert "wraps on overflow" in item.reason


def test_a_two_operand_square_raises_where_the_docstring_promises_a_decline():
    """RECORDED, NOT REPAIRED, and shared.

    ``obligation``'s module docstring says "Declines never raise". A
    malformed `square` equation carrying TWO operands passes ``_validate``
    — nothing there checks a unop's arity — and then dies inside
    ``smt.emit`` with a bare ``ValueError: too many values to unpack
    (expected 1)`` out of ``_pair_elementwise``, which is not a decline and
    does not name the primitive.

    Measured on main (b11f051): ``integer_pow`` does the identical thing,
    with the byte-identical message, and `square` is not in the emission
    set there at all. So this row joins the class rather than creating it,
    and the class is ``neg``/``not``/``integer_pow``/`square` — every
    emitted unop.

    NOT REACHABLE BY TRACING: jax declares ``square_p`` a unop, so no
    traced program produces this equation; it takes hand-built or
    ``from_dict`` IR. That is why it is recorded here rather than fixed
    here — the fix belongs with the arity check for the whole unop class,
    which this row does not own."""
    x, y, s, pred, out = (
        var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    )
    q = close(
        [
            any_eqn(x, -2.0, 3.0),
            any_eqn(y, -2.0, 3.0),
            eqn("square", [x, y], s),  # TWO operands on a unop
            eqn("le", [s, lit(4.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert not isinstance(item, DeclinedObligation), (
        "_validate now rejects a 2-operand square — if that is deliberate, "
        "this record is stale and should be deleted"
    )
    with pytest.raises(ValueError, match="too many values to unpack"):
        SM.emit(item, "z3", 1000)


def test_a_boolean_square_declines():
    x, b, s, pred, out = (
        var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL)
    )
    q = close(
        [
            any_eqn(x, -2.0, 3.0),
            eqn("gt", [x, lit(0.0)], b),
            eqn("square", [b], s),
            eqn("eq", [s, lit(True, BOOL)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    if p.obligations[0].status != "unknown":
        pytest.skip("interval propagation decided it; no slice to validate")
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert "'square' on boolean operands" in item.reason
