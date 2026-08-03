# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Constant subtrees fold to numerals, and a lost backend is FLAGGED.

Two repairs of one defect, recorded together because the second is the
floor under the first.

**The defect.** A slice is stamped ``QF_LRA`` exactly when nothing in it
multiplies two declaration-DEPENDENT operands
(``obligation._Slicer._fragment``), so a product whose factors descend
only from literals is a constant and the decision problem really is
linear. The emission nonetheless shipped ``(* t1 t1)`` over a compound
``t1``, which is *syntactically* nonlinear, and z3's QF_LRA parser
refuses that outright. The obligation was then decided by cvc5 alone —
one backend answering under a two-backend stamp — and **nothing said
so**: the existing "portfolio degraded" note fired on ``len(ordered) ==
1``, a backend not INSTALLED, and degradation by REFUSAL leaves both
backends in ``ordered``.

The dangerous direction is the DISCHARGE. A ``sat`` becomes REFUTED only
through independent exact-rational replay of the model, so a lost backend
there costs a cross-check another mechanism still performs. An ``unsat``
is a universal claim over the whole declared box: nothing re-derives it,
so the second backend IS the redundancy.

**Repair 1, the cause** (``smt._fold_constant_elements``): a
real-arithmetic value fixed at emission time is emitted as a NUMERAL, so
the emitted text is as linear as the declared logic says the problem is
and both backends read it. Exact, not a simplification — Fraction
arithmetic under a fragment emitted for the Reals. The third option,
stamping ``QF_NRA`` instead, is measured WRONG here and pinned as such:
these problems are not nonlinear, and widening the logic makes a solvable
problem harder for nothing.

**Repair 2, the reporting** (``solvers.PORTFOLIO_SIZE``,
``ObligationEscalation.answered_by``, ``Verdict.solver_redundancy``): who
ANSWERED is now derived and carried, distinct from who was ASKED (the
stamps' contract, unchanged). Repair 2 is required regardless of repair
1, because a backend can still be lost for reasons the fold does not
touch — a crash, a timeout, an uninstalled solver — and the residue of a
conservative fold is exactly the case that must not be silent.

SCOPE — what these tests REACH: the emitted TEXT for the affected class
and its controls, checked against the real z3 parser under the declared
logic; the fold's arithmetic against jax's own eager and jit execution;
the dispatch layer's degradation reporting on both outcome directions and
both causes (absent backend, refusing backend); and end-to-end verdict
containment for a battery of constant-bearing harnesses. They do NOT
reach: the affine domain, ieee semantics, arrays past the small battery
here, the scatter rows, or any primitive outside the emission set.

Positive and negative controls throughout: a genuinely nonlinear slice
must STILL be QF_NRA and STILL emit the unfolded self-product (otherwise
"the fold fixed it" and "the fold ate the row" print the same page), and
a full-portfolio verdict must carry NO degradation note (otherwise the
flag means nothing).

MEASURED DISCRIMINATING POWER, this tree, both solvers installed. Each
mechanism was deleted in turn and the suite re-run; the counts are the
tests that CAUGHT the deletion, and every one of them is in this file —
nothing that existed before this round noticed any of these:

    the fold, disabled entirely                      24 caught
    the degradation notes + detail clause, disabled   2 caught
    the top-of-render PORTFOLIO DEGRADED line          3 caught
    Verdict.solver_redundancy, never populated         5 caught

Per-primitive wrong-fold mutations are a live battery below
(:data:`MUTATIONS`) rather than a recorded number, because a fold that
computes the wrong value is the one way this repair could mint a false
VERIFIED.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

import stelling.obligation as OB
import stelling.smt as SM
from stelling import solvers
from stelling.obligation import ObligationSlice, slice_unknown_obligations
from stelling.propagate import interval_env, propagate
from stelling.solvers import (
    PORTFOLIO_SIZE,
    SolverConfig,
    _Backend,
    _RawResult,
    escalate,
    make_solver_verdict,
)
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, var

VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="jax_enable_x64=True (hand-built f64 IR)",
)


def _slice_of(q) -> ObligationSlice:
    p = propagate(q)
    items = slice_unknown_obligations(q, p, interval_env(q))
    assert len(items) == 1 and isinstance(items[0], ObligationSlice), items
    return items[0]


# --- the affected class -------------------------------------------------------
#
# `x` is declared on [-2, 3]; a constant-only subtree computes 2.0 + 1.0 = 3.0
# into `c`; each case combines `c` (and sometimes `x`) into `s`; the assert is
# `s + x <= bound` with a bound the propagated interval STRADDLES, so the cheap
# layer cannot settle it and a slice is really built.

C = var(1)  # the compound constant: (+ 2.0 1.0)
S = var(2)  # the case's output
_CONST_SUBTREE = [eqn("add", [lit(2.0), lit(1.0)], C)]

# name -> (equations producing S, straddled bound, the `d = S + x` body the
# emission must produce). The third field is the byte-level pin, written out
# per case rather than derived, so a fold that computed the wrong number
# fails here and not only in the mutation battery.
CLASS = {
    # a self-product of a constant, in all three spellings jax can write it
    "square(c)": ([eqn("square", [C], S)], 10.0, "(+ 9.0 x0)"),
    "mul(c, c)": ([eqn("mul", [C, C], S)], 10.0, "(+ 9.0 x0)"),
    "integer_pow(c, 2)": (
        [eqn("integer_pow", [C], S, (("y", 2),))], 10.0, "(+ 9.0 x0)"
    ),
    # a constant SCALE on a declared input — affine, and the commonest
    # shape of all: a hyperparameter arriving through a jit boundary. The
    # product SURVIVES here (one factor is symbolic); what changed is that
    # the other factor is a numeral, which is what QF_LRA needs.
    "mul(c, x)": ([eqn("mul", [C, var(0)], S)], 10.0, "(* 3.0 x0)"),
    "mul(x, c)": ([eqn("mul", [var(0), C], S)], 10.0, "(* x0 3.0)"),
    # a constant DIVISOR, and the reciprocal spelling of one
    "div(x, c)": ([eqn("div", [var(0), C], S)], 2.0, "(/ x0 3.0)"),
    "div(c, c)": ([eqn("div", [C, C], S)], 2.0, "(+ 1.0 x0)"),
    "integer_pow(c, -1)": (
        [eqn("integer_pow", [C], S, (("y", -1),))], 2.0, "(+ (/ 1 3) x0)"
    ),
}

# The controls, in both directions. LINEAR-ALREADY: the same operations with
# BARE LITERAL operands, which never had a compound term to fold and were
# always accepted — if these ever change, the fold moved something it should
# not have. NONLINEAR: a genuine product of two dependent operands, which must
# stay QF_NRA and stay unfolded.
LINEAR_CONTROLS = {
    "mul(3.0, x)": ([eqn("mul", [lit(3.0), var(0)], S)], 10.0, "(* 3.0 x0)"),
    "div(x, 3.0)": ([eqn("div", [var(0), lit(3.0)], S)], 2.0, "(/ x0 3.0)"),
}
NONLINEAR_CONTROLS = {
    "square(x)": ([eqn("square", [var(0)], S)], 10.0, "(* x0 x0)"),
    "mul(x, x)": ([eqn("mul", [var(0), var(0)], S)], 10.0, "(* x0 x0)"),
    "integer_pow(x, 2)": (
        [eqn("integer_pow", [var(0)], S, (("y", 2),))], 10.0, "(* x0 x0)"
    ),
}


def _case_query(mid, bound, *, with_const=True):
    x = var(0)
    d, pred, out = var(90), var(91, BOOL), var(92, BOOL)
    return close(
        [any_eqn(x, -2.0, 3.0)]
        + (list(_CONST_SUBTREE) if with_const else [])
        + list(mid)
        + [
            eqn("add", [S, x], d),
            eqn("le", [d, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def _case(name):
    table = {**CLASS, **LINEAR_CONTROLS, **NONLINEAR_CONTROLS}
    mid, bound, _ = table[name]
    return _case_query(mid, bound, with_const=name in CLASS)


@pytest.mark.parametrize("name", sorted(CLASS))
def test_the_affected_class_is_LINEAR_and_stays_stamped_linear(name):
    """The measurement that decides which repair is right.

    Every member of the class is stamped ``QF_LRA``, and it is stamped
    that way because it IS linear: the only multiplication or division in
    it has a compile-time-constant operand. Stamping ``QF_NRA`` instead
    would be the third option the round rejected — it makes a solvable
    problem harder and buys nothing, because the problem was never
    nonlinear.

    Four primitives, six operand shapes: this is not one primitive's
    row."""
    assert _slice_of(_case(name)).fragment == OB.QF_LRA


@pytest.mark.parametrize("name", sorted(NONLINEAR_CONTROLS))
def test_negative_control_a_genuine_product_stays_QF_NRA_and_UNFOLDED(name):
    """The other direction, without which the parametrization above says
    nothing: a product of two DEPENDENT operands is genuinely nonlinear,
    must keep its ``QF_NRA`` stamp, and must keep emitting the
    self-product the interval leg cannot see."""
    item = _slice_of(_case(name))
    assert item.fragment == OB.QF_NRA
    assert NONLINEAR_CONTROLS[name][2] in SM.emit(item, "z3", 30_000).text


@pytest.mark.parametrize("name", sorted(CLASS))
def test_a_constant_subtree_emits_its_VALUE_and_no_term(name):
    """The fold, at the byte level, in both halves.

    ABSENCE: the constant subtree's output (``t1``) gets no ``define-fun``
    and is referenced nowhere, so no product or quotient can have a
    compound constant where the linear fragment needs a numeral.

    PRESENCE: the exact value the constant really has appears in its
    place. Pinning only the absence would pass for a fold that emitted the
    wrong number, and pinning only ``"3.0" in text`` would pass
    vacuously — ``3.0`` is also this box's upper bound."""
    text = SM.emit(_slice_of(_case(name)), "z3", 30_000).text
    defined = {
        line.split()[1] for line in text.splitlines()
        if line.startswith("(define-fun ")
    }
    assert "t1" not in defined, (defined, text)
    assert CLASS[name][2] in text, text


@pytest.mark.parametrize("name", sorted({**CLASS, **LINEAR_CONTROLS}))
def test_z3_parses_every_member_of_the_class_under_its_declared_logic(name):
    """The consequence, measured against the real parser rather than
    argued: the emitted script is accepted under the logic the slice
    stamped. Before the fold, every ``CLASS`` member here was rejected
    with 'logic does not support nonlinear arithmetic' and every
    ``LINEAR_CONTROLS`` member was accepted — the controls are the proof
    that this gate can fail."""
    z3 = pytest.importorskip("z3")
    text = SM.emit(_slice_of(_case(name)), "z3", 30_000).text
    s = z3.Solver()
    s.from_string(text)  # raises Z3Exception on a refused fragment
    assert str(s.check()) in ("sat", "unsat")


def test_the_fold_is_exact_not_a_rounding():
    """The fold's arithmetic is exact rationals, so a constant whose f64
    value is not a nice decimal keeps its exact dyadic value — the same
    discipline ``rational()`` has always enforced on a literal.

    ``0.1 + 0.2`` in binary64 is NOT 0.3, and the folded numeral must be
    the sum of the two exact dyadics, never the decimal it looks like."""
    x, c, s, d, pred, out = (
        var(0), var(1), var(2), var(3), var(4, BOOL), var(5, BOOL)
    )
    q = close(
        [
            any_eqn(x, -2.0, 3.0),
            eqn("add", [lit(0.1), lit(0.2)], c),
            eqn("mul", [c, x], s),
            eqn("add", [s, x], d),
            eqn("le", [d, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    text = SM.emit(_slice_of(q), "z3", 30_000).text
    exact = Fraction(0.1) + Fraction(0.2)
    assert exact != Fraction(3, 10)  # the premise of the test
    assert SM.rational(exact) in text, text
    assert SM.rational(Fraction(3, 10)) not in text, text


def test_the_fold_declines_what_it_cannot_do_exactly():
    """Conservative by construction, and the conservatism is measured
    rather than asserted: a division by a constant zero has no rational
    value, so :func:`smt._fold_constant_elements` returns ``None`` and the
    caller emits exactly as before. (The slice validator refuses such a
    slice first; the fold refuses independently rather than rely on
    that.)"""
    div0 = eqn("div", [lit(1.0), lit(0.0)], S)
    assert SM._fold_constant_elements(
        div0, [(Fraction(1),), (Fraction(0),)]
    ) is None
    # a NON-constant operand is likewise not folded
    assert SM._fold_constant_elements(
        eqn("mul", [lit(2.0), var(0)], S), [(Fraction(2),), None]
    ) is None
    # and a primitive outside the foldable set is not folded even when
    # every operand is constant
    assert SM._fold_constant_elements(
        eqn("le", [lit(2.0), lit(3.0)], var(3, BOOL)),
        [(Fraction(2),), (Fraction(3),)],
    ) is None
    # the positive control for all three: the same shape, foldable
    assert SM._fold_constant_elements(
        eqn("div", [lit(1.0), lit(4.0)], S), [(Fraction(1),), (Fraction(4),)]
    ) == (Fraction(1, 4),)


# mutation name -> (primitive, the constant-only equation that exercises its
# fold, a deliberately wrong rule, a bound the propagated interval straddles)
MUTATIONS = {
    "add-folds-as-sub": ("add", (2.0, 1.0), lambda a, b: a - b, 4.0),
    "mul-folds-as-add": ("mul", (2.0, 3.0), lambda a, b: a + b, 5.0),
    "div-folds-inverted": ("div", (6.0, 3.0), lambda a, b: b / a, 3.0),
    "min-folds-as-max": ("min", (2.0, 5.0), lambda a, b: max(a, b), 3.0),
    "max-folds-as-min": ("max", (2.0, 5.0), lambda a, b: min(a, b), 6.0),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_mutation_a_wrong_fold_is_CAUGHT_by_the_emitted_value(mutation, monkeypatch):
    """The battery under every assertion above, and the reason the fold's
    exactness is a measurement and not a promise.

    A fold is a second exact evaluator beside the replay's, and a wrong
    one would put a number in the script that the program never computes
    — the one way this repair could mint a false VERIFIED. Each mutation
    replaces the fold's rule for one primitive with a deliberately wrong
    one; the emitted numeral must change. A mutation that survives is a
    hole in this file, not a curiosity.

    The baseline is asserted too: the unmutated emission must carry the
    value the operation really has, so a mutation that changed the text
    for some unrelated reason cannot pass for a catch."""
    prim, (a, b), wrong, bound = MUTATIONS[mutation]
    x, c, d, pred, out = (
        var(0), var(1), var(2), var(3, BOOL), var(4, BOOL)
    )
    q = close(
        [
            any_eqn(x, -2.0, 3.0),
            eqn(prim, [lit(a), lit(b)], c),  # constant-only: foldable
            eqn("add", [c, x], d),
            eqn("le", [d, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    real = SM._fold_constant_elements
    right = real(q.jaxpr.eqns[1], [(Fraction(a),), (Fraction(b),)])
    baseline = SM.emit(_slice_of(q), "z3", 30_000).text
    assert SM.rational(right[0]) in baseline, (mutation, baseline)

    def mutated(eqn_, ins):
        if eqn_.primitive == prim and not any(v is None for v in ins):
            return (wrong(ins[0][0], ins[1][0]),)
        return real(eqn_, ins)

    monkeypatch.setattr(SM, "_fold_constant_elements", mutated)
    assert SM.emit(_slice_of(q), "z3", 30_000).text != baseline, mutation


# --- degradation reporting: who ANSWERED, not who was ASKED -------------------


def _fake_backend(name, answer, *, detail="", values=()):
    return _Backend(
        name=name,
        flavor=name,
        label=f"{name} (fake)",
        transport="fake in-test transport",
        transport_fn=lambda text, wall: _RawResult(
            answer=answer, version="9.9.9-fake", detail=detail,
            values=tuple(values),
        ),
        version_fn=lambda: "9.9.9-fake",
    )


def _escalate_with(monkeypatch, q, backends, missing=()):
    monkeypatch.setattr(
        solvers, "_backends_for", lambda config: (tuple(backends), tuple(missing))
    )
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    return p, escalate(q, p, SolverConfig(timeout_ms=1000))


def _degrading_query():
    """A slice interval propagation cannot settle, so it really escalates.
    ``x*x`` over [-2, 3] is [0, 9] by the transfer and straddles 4."""
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, -2.0, 3.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(4.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_a_discharge_on_a_full_portfolio_carries_NO_degradation(monkeypatch):
    """The negative control for everything below. Two backends, both
    answering unsat: the verdict is VERIFIED, ``solver_redundancy`` names
    both, and no degradation is claimed anywhere. Without this, a flag
    that fired unconditionally would look exactly like a working one."""
    q = _degrading_query()
    p, esc = _escalate_with(
        monkeypatch, q,
        [_fake_backend("z3", "unsat"), _fake_backend("cvc5", "unsat")],
    )
    (record,) = esc.records
    assert record.outcome == "discharged"
    # cvc5 first: the dispatch order puts it primary on QF_NRA
    assert set(record.answered_by) == {"z3 (fake)", "cvc5 (fake)"}
    assert len(record.answered_by) == PORTFOLIO_SIZE
    assert not any("portfolio degraded" in n for n in record.notes), record.notes
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.solver_redundancy == ((0, record.answered_by),)
    assert "PORTFOLIO DEGRADED" not in v.render()


def test_a_discharge_a_REFUSING_backend_lost_is_flagged_everywhere(monkeypatch):
    """The defect this round closed, in the shape that has no backstop.

    Both backends are installed, invoked and STAMPED; one of them fails on
    the script. The verdict is still VERIFIED — the fold does not change
    any verdict's truth and neither does the flag — but it is now
    distinguishable from a two-backend VERIFIED at four separate reading
    points, because a reader reaches a verdict through more than one of
    them."""
    q = _degrading_query()
    p, esc = _escalate_with(
        monkeypatch, q,
        [
            _fake_backend("z3", "failed", detail="logic does not support X"),
            _fake_backend("cvc5", "unsat"),
        ],
    )
    (record,) = esc.records
    assert record.outcome == "discharged"
    # the stamp still says TWO — that is its contract, and it is exactly
    # why it could not be the thing that reports this
    assert {s.name for s in record.invocations} == {"z3", "cvc5"}
    assert record.answered_by == ("cvc5 (fake)",)
    notes = " | ".join(record.notes)
    assert "portfolio degraded" in notes, notes
    assert "z3 (fake) was invoked and returned failed" in notes, notes
    assert "no backstop" in notes, notes

    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"  # (1) the verdict itself is unchanged
    assert v.solver_redundancy == ((0, ("cvc5 (fake)",)),)  # (2) counted
    rendered = v.render()
    assert "PORTFOLIO DEGRADED — assert #0 was decided by ONE solver" in rendered
    (ob,) = v.obligations  # (3) the obligation's own detail line
    assert f"PORTFOLIO DEGRADED: 1 of {PORTFOLIO_SIZE} backends" in ob.detail
    assert "a discharge has no replay backstop" in ob.detail
    assert "portfolio degraded" in " | ".join(v.notes)  # (4) the notes


def test_a_refutation_is_flagged_too_but_without_the_no_backstop_sentence(monkeypatch):
    """Both directions are reported, and they are reported DIFFERENTLY.

    A ``sat`` reaches REFUTED only through independent exact-rational
    replay, so the lost backend costs a cross-check another mechanism
    still performs. Saying "no backstop" here would be false, and a note
    a reader learns is sometimes false is a note they stop reading."""
    q = _degrading_query()  # x*x <= 4 over [-2, 3]: false at x = 3
    p, esc = _escalate_with(
        monkeypatch, q,
        [
            _fake_backend("z3", "timeout"),
            _fake_backend("cvc5", "sat", values=(("x0", "3"),)),
        ],
    )
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    assert record.answered_by == ("cvc5 (fake)",)
    notes = " | ".join(record.notes)
    assert "portfolio degraded" in notes, notes
    assert "no backstop" not in notes, notes
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    assert v.solver_redundancy == ((0, ("cvc5 (fake)",)),)
    # the TOP-OF-RENDER line, named exactly. A bare `"PORTFOLIO DEGRADED"
    # in v.render()` passes off the obligation's detail clause alone —
    # measured, by deleting the render line and watching this test stay
    # green — so the two surfaces are pinned separately.
    assert "PORTFOLIO DEGRADED — assert #0 was decided by ONE solver" in v.render()
    (ob,) = v.obligations
    assert f"PORTFOLIO DEGRADED: 1 of {PORTFOLIO_SIZE} backends" in ob.detail
    assert "no replay backstop" not in ob.detail, ob.detail


def test_an_UNDECIDED_obligation_is_not_reported_as_a_degraded_portfolio(monkeypatch):
    """The second negative control, on the other axis. Nothing decided
    this obligation, so there is no redundancy to be short of — listing it
    would put every UNKNOWN in the degraded column and make the count
    useless."""
    q = _degrading_query()
    p, esc = _escalate_with(
        monkeypatch, q,
        [_fake_backend("z3", "timeout"), _fake_backend("cvc5", "unknown")],
    )
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert record.answered_by == ()
    assert not any("portfolio degraded" in n for n in record.notes), record.notes
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.solver_redundancy == ()
    assert "PORTFOLIO DEGRADED" not in v.render()


def test_an_ABSENT_backend_reaches_the_same_reporting_as_a_refusing_one(monkeypatch):
    """The pre-existing cause, joined to the new surface. A backend that
    was never installed and a backend that refused the script are the same
    fact to a reader — one answer, not two — so they must arrive at the
    same place, whatever else they say about themselves."""
    q = _degrading_query()
    p, esc = _escalate_with(
        monkeypatch, q, [_fake_backend("cvc5", "unsat")], missing=("z3",),
    )
    (record,) = esc.records
    assert record.answered_by == ("cvc5 (fake)",)
    notes = " | ".join(record.notes)
    assert "portfolio degraded — only cvc5 (fake) ran" in notes, notes
    assert "z3 is not installed" in notes, notes
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.solver_redundancy == ((0, ("cvc5 (fake)",)),)
    assert "PORTFOLIO DEGRADED — assert #0 was decided by ONE solver" in v.render()


def test_a_configured_restriction_is_not_reported_as_a_missing_install(monkeypatch):
    """A reporting-honesty fix in the same note. ``SolverConfig.only`` is a
    supported configuration; the predecessor rendered every absence as
    "not installed", which sends a reader to install something they
    already have."""
    q = _degrading_query()
    monkeypatch.setattr(
        solvers, "_backends_for",
        lambda config: ((_fake_backend("cvc5", "unsat"),), ()),
    )
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000, only=("cvc5",)))
    notes = " | ".join(esc.records[0].notes)
    assert "z3 was excluded by SolverConfig.only=('cvc5',)" in notes, notes
    assert "not installed" not in notes, notes


# --- end to end: the battery, and containment against jax --------------------
#
# The constant subtree has to be TRACED, not computed eagerly, or there is no
# compound term to fold: `jnp.float64(2.0) + jnp.float64(1.0)` written at the
# top level of a harness is executed on the spot and arrives as one literal.
# The shape that traces it is the one real code has — hyperparameters crossing
# a `jit` boundary as arguments — so every entry below is `jax.jit(f)(2.0,
# 1.0, v)` with `c = a + b` inside.
#
# `+ (v - v)` rides on every entry for the reason the square row's own gauge
# uses: it is exactly zero in the program and [-1, 1] to the interval leg, so
# the cheap layer cannot settle the obligation and the slice really reaches a
# solver. A battery the interval leg discharges would measure nothing here.
#
# NOT IN THIS BATTERY, and not because they work: `v / c` and `c ** -1` DECLINE
# end-to-end for an unrelated, pre-existing reason — the divisor guard needs a
# top-level propagated interval for the divisor, and "a value produced inside a
# transparent call carries none". That is a limitation of the div guard's
# reach, not of this round; the div and reciprocal arms of the class are
# covered above on hand-built IR, where the interval env sees every var.

_BATTERY = [
    # (name, f(a, b, v), bound, expected-to-hold-everywhere)
    ("square(c)", lambda jnp, a, b, v: jnp.square(a + b), 9.5, True),
    ("square(c) refuted", lambda jnp, a, b, v: jnp.square(a + b), 8.5, False),
    ("c*c", lambda jnp, a, b, v: (a + b) * (a + b), 9.5, True),
    ("c**2", lambda jnp, a, b, v: (a + b) ** 2, 9.5, True),
    ("c*v", lambda jnp, a, b, v: (a + b) * v, 3.0, True),
    ("c*v refuted", lambda jnp, a, b, v: (a + b) * v, 2.0, False),
    ("max(c, v)", lambda jnp, a, b, v: jnp.maximum(a + b, v), 3.5, True),
    ("max(c, v) refuted", lambda jnp, a, b, v: jnp.maximum(a + b, v), 2.5, False),
    ("(c-v)*c", lambda jnp, a, b, v: (a + b - v) * (a + b), 9.5, True),
    ("(c-v)*c refuted", lambda jnp, a, b, v: (a + b - v) * (a + b), 8.5, False),
]
BOX = (0.0, 1.0)


def _battery_case(fn, bound):
    """``(harness, oracle)`` for one battery entry — ONE function, traced
    for the harness and executed for the oracle, so the containment check
    cannot be of a different program from the verified one."""
    import jax
    import jax.numpy as jnp

    from stelling.harness import any_array, assert_

    jitted = jax.jit(lambda a, b, v: (v - v) + fn(jnp, a, b, v))

    def harness():
        v = any_array((), "float64", BOX)
        return (assert_(jitted(jnp.float64(2.0), jnp.float64(1.0), v) <= bound),)

    def oracle(v):
        got = jitted(jnp.float64(2.0), jnp.float64(1.0), jnp.float64(v))
        return bool(got <= bound)

    return harness, oracle


def test_the_battery_is_contained_by_what_jax_actually_computes():
    """Verdict containment end to end over the affected class, with its
    denominator.

    Every entry must reach a DEFINITE verdict — an UNKNOWN here would mean
    the pipeline stopped reaching the solver — and every definite verdict
    must be consistent with jax's own execution of the same program:

    * a VERIFIED claims the predicate holds EVERYWHERE in the declared
      box, so at every sampled point the real jnp program must satisfy it;
    * a REFUTED must carry a witness that, executed through the real jnp
      program, actually violates.

    Sampling cannot PROVE a VERIFIED and this test does not claim it does.
    What it catches is the failure mode the fold could introduce — a
    numeral in the script that the program never computes — which shows up
    as a VERIFIED whose predicate is false at an ordinary point.

    The last assertion is the round's capability claim: with the fold in,
    no entry in this class costs a portfolio member any more."""
    pytest.importorskip("jax")
    pytest.importorskip("z3")
    pytest.importorskip("cvc5")
    import jax

    from stelling.preconditions import check

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        samples = [0.0, 0.125, 0.5, 0.875, 1.0]
        tally = {"VERIFIED": 0, "REFUTED": 0}
        for name, fn, bound, holds in _BATTERY:
            harness, oracle = _battery_case(fn, bound)
            v = check(
                harness, vacuity_mode="inputs-only", solver_timeout_ms=30_000
            )
            assert v.status in tally, (name, v.status, v.notes)
            tally[v.status] += 1
            # the expected direction, stated per entry so a battery that
            # silently became all-VERIFIED cannot pass
            assert (v.status == "VERIFIED") == holds, (name, v.status)
            if v.status == "VERIFIED":
                for s in samples:
                    assert oracle(s), (name, "VERIFIED but false at", s)
            else:
                assert v.witnesses, (name, "REFUTED with no witness", v.notes)
                for w in v.witnesses:
                    for _, exact in w.values:
                        assert not oracle(float(Fraction(exact))), (name, exact)
            # every obligation a solver decided got the FULL portfolio.
            # The non-emptiness is asserted FIRST: a battery the interval
            # leg had settled would leave the loop below with nothing to
            # iterate and make this whole test a measurement of nothing.
            assert v.solver_redundancy, (name, "nothing escalated", v.notes)
            for index, who in v.solver_redundancy:
                assert len(who) == PORTFOLIO_SIZE, (name, index, who, v.notes)
            assert "PORTFOLIO DEGRADED" not in v.render(), name
        assert tally["VERIFIED"] + tally["REFUTED"] == len(_BATTERY) == 10
        # both directions really occur, or "contained" would be vacuous
        assert tally == {"VERIFIED": 6, "REFUTED": 4}, tally
    finally:
        jax.config.update("jax_enable_x64", old)
