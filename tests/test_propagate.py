# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Forward propagation over hand-built IR — no jax needed.

The queries here are constructed directly in :mod:`stelling.ir`, which
keeps the interpreter's semantics testable without any tracer in the
loop (and proves the interpreter really does consume the IR, not jaxprs).
"""

from __future__ import annotations

import pytest

from stelling import ir
from stelling.propagate import propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


def var(i, aval=F64):
    return ir.Var(id=i, aval=aval)


def any_eqn(out, lo, hi):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", ()), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def assert_eqn(pred, out):
    return ir.JaxprEqn(primitive="stelling_assert", invars=(pred,), outvars=(out,))


def close(eqns, outvars):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(), outvars=outvars, eqns=tuple(eqns))
    )


def exp_lt_harness(bound: float) -> ir.ClosedJaxpr:
    """x ∈ [1, 2] ⊢ exp(x) < bound."""
    x, ex, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            ir.JaxprEqn(
                primitive="lt",
                invars=(ex, ir.Literal(val=bound, aval=F64)),
                outvars=(pred,),
            ),
            assert_eqn(pred, out),
        ],
        (out,),
    )


def test_discharges_a_true_obligation():
    p = propagate(exp_lt_harness(8.0))  # e^2 ≈ 7.389 < 8
    assert p.all_discharged
    assert p.obligations[0].status == "discharged"
    assert ("exp", "sound-libm") in p.transfers_used
    assert any("libm" in a for a in p.assumptions)
    assert p.coverage.unknown == 0


def test_undecidable_obligation_is_unknown_not_guessed():
    p = propagate(exp_lt_harness(7.0))  # e^2 ≈ 7.389 > 7 > e
    assert not p.all_discharged
    assert p.obligations[0].status == "unknown"


def test_definitely_false_obligation_reports_violated_over_set():
    p = propagate(exp_lt_harness(2.0))  # e^1 ≈ 2.718 > 2: false everywhere
    assert not p.all_discharged
    assert p.obligations[0].status == "violated-over-set"


def test_unknown_primitive_falls_to_top_and_is_counted():
    x, y, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(primitive="mystery_op", invars=(x,), outvars=(y,)),
            ir.JaxprEqn(
                primitive="lt",
                invars=(y, ir.Literal(val=100.0, aval=F64)),
                outvars=(pred,),
            ),
            assert_eqn(pred, out),
        ],
        (out,),
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"  # ⊤ decays the verdict, soundly
    assert p.coverage.unknown == 1
    assert p.coverage.unknown_primitives[0][0] == "mystery_op"


def test_no_obligations_is_not_discharged():
    x = var(0)
    p = propagate(close([any_eqn(x, 0.0, 1.0)], (x,)))
    assert not p.all_discharged  # a harness with no asserts verifies nothing


def test_mixed_violated_and_unknown_reports_violation():
    # one face definitely false + one undecided: the violation is definite
    # regardless of the other face — any_violated drives a REFUTED verdict
    x, e1, p1, o1, p2, o2 = var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(e1,)),
            ir.JaxprEqn(
                primitive="lt",
                invars=(e1, ir.Literal(val=2.0, aval=F64)),  # e^1 > 2: false
                outvars=(p1,),
            ),
            assert_eqn(p1, o1),
            ir.JaxprEqn(
                primitive="lt",
                invars=(e1, ir.Literal(val=7.0, aval=F64)),  # straddles
                outvars=(p2,),
            ),
            assert_eqn(p2, o2),
        ],
        (o1, o2),
    )
    p = propagate(q)
    assert p.any_violated
    assert [o.status for o in p.obligations] == ["violated-over-set", "unknown"]


def _assume_harness():
    x, pred, apred, ex, p2, out = var(0), var(1, BOOL), var(2, BOOL), var(3), var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(
                primitive="gt",
                invars=(x, ir.Literal(val=1.5, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(primitive="stelling_assume", invars=(pred,), outvars=(apred,)),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
            ir.JaxprEqn(
                primitive="lt",
                invars=(ex, ir.Literal(val=8.0, aval=F64)),
                outvars=(p2,),
            ),
            assert_eqn(p2, out),
        ],
        (out,),
    )


def test_inert_assume_is_counted_named_and_noted_never_hidden():
    # assume_mode="inert": the comparability control — drops every assume
    # exactly as the MVP did, and says so.
    p = propagate(_assume_harness(), assume_mode="inert")
    assert p.coverage.inert == 1
    assert p.coverage.inert_primitives == (("stelling_assume", 1),)
    assert p.coverage.constrained == 0
    assert p.dropped_constraints == 1
    assert "DROPPED" in p.coverage.summary()
    assert p.coverage.fraction_known < 1.0  # a drop cannot hide inside 100%
    assert any("DROPPED" in n for n in p.notes)
    assert p.all_discharged  # VERIFIED-with-drops stands (superset proved)
    assert "stelling_assume" not in dict(p.transfers_used)  # no tier claimed


def test_constrainable_assume_narrows_and_discloses_by_default():
    p = propagate(_assume_harness())  # assume_mode="constrain" is the default
    assert p.coverage.constrained == 1
    assert p.coverage.constrained_primitives == (("stelling_assume", 1),)
    assert p.coverage.inert == 0 and p.dropped_constraints == 0
    assert "CONSTRAINED" in p.coverage.summary()
    assert any("assume CONSTRAINED" in n for n in p.notes)
    assert any("the verdict holds where the precondition holds" in a for a in p.assumptions)
    assert p.all_discharged
    assert "stelling_assume" not in dict(p.transfers_used)  # still no tier claimed


def test_nonvacuity_checks_are_collected_separately():
    x, m, pred, om, out = var(0), var(1, BOOL), var(2, BOOL), var(3, BOOL), var(4, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(
                primitive="gt",
                invars=(x, ir.Literal(val=0.0, aval=F64)),
                outvars=(m,),
            ),
            ir.JaxprEqn(primitive="stelling_nonvacuity", invars=(m,), outvars=(om,)),
            ir.JaxprEqn(
                primitive="lt",
                invars=(x, ir.Literal(val=3.0, aval=F64)),
                outvars=(pred,),
            ),
            assert_eqn(pred, out),
        ],
        (out, om),
    )
    p = propagate(q)
    assert len(p.nonvacuity_checks) == 1
    assert p.nonvacuity_checks[0].status == "discharged"
    assert len(p.obligations) == 1  # membership conditions are not obligations
    assert p.all_discharged


def test_unbound_var_raises_instead_of_widening():
    x, y, out = var(0), var(7), var(8, BOOL)  # var 7 is never bound
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(primitive="exp", invars=(y,), outvars=(out,)),
        ],
        (out,),
    )
    with pytest.raises(ir.TranscriptionError):
        propagate(q)


def test_free_invars_are_refused():
    x = var(0)
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(x,), outvars=(x,), eqns=())
    )
    with pytest.raises(ir.TranscriptionError):
        propagate(q)
