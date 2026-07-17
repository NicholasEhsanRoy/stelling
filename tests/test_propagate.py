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


def test_free_invars_are_refused():
    x = var(0)
    q = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(constvars=(), invars=(x,), outvars=(x,), eqns=())
    )
    with pytest.raises(ir.TranscriptionError):
        propagate(q)
