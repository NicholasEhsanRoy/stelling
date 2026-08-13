# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reaches-output reachability conjunct.

Three cases per the design:
1. Violation on a variable that IS used in the output -> stays REFUTED
2. Violation on a dead variable (computed but not returned) -> UNKNOWN with note
3. Mixed: one reachable violation, one unreachable -> REFUTED, unreachable gets note
"""

from __future__ import annotations

import pytest

from stelling import ir
from stelling.propagate import propagate
from stelling.reachability import obligation_operand_ids, reaches_output
from stelling.verdict import make_verdict

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


# -- Unit tests for reaches_output -------------------------------------------


def test_reaches_output_simple_chain():
    """All vars in a linear chain to an output are live."""
    x, y, z = var(0), var(1), var(2)
    jaxpr = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(z,),
        eqns=(
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="exp", invars=(x,), outvars=(y,)
            ),
            ir.JaxprEqn(
                primitive="exp", invars=(y,), outvars=(z,)
            ),
        ),
    )
    live = reaches_output(jaxpr)
    assert 0 in live  # x
    assert 1 in live  # y
    assert 2 in live  # z


def test_reaches_output_dead_branch():
    """A variable not needed for any output is dead."""
    x, dead, y = var(0), var(1), var(2)
    jaxpr = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(y,),
        eqns=(
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(
                primitive="exp", invars=(x,), outvars=(dead,)
            ),
            ir.JaxprEqn(
                primitive="mul",
                invars=(x, ir.Literal(val=2.0, aval=F64)),
                outvars=(y,),
                params=(("out_dtype", None),),
            ),
        ),
    )
    live = reaches_output(jaxpr)
    assert 0 in live  # x (used by mul -> y)
    assert 1 not in live  # dead (exp output, unused)
    assert 2 in live  # y (the output)


def test_reaches_output_literal_output():
    """A jaxpr whose output is a Literal: no Var IDs from it."""
    x = var(0)
    jaxpr = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(ir.Literal(val=1.0, aval=F64),),
        eqns=(any_eqn(x, 0.0, 1.0),),
    )
    live = reaches_output(jaxpr)
    assert 0 not in live  # x is never used for the output


# -- Unit tests for obligation_operand_ids ------------------------------------


def test_obligation_operand_ids_basic():
    """Maps obligation indices to their assert invars."""
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    closed = close(
        [any_eqn(x, 0.0, 1.0), assert_eqn(pred, out)],
        (out,),
    )
    ids = obligation_operand_ids(closed)
    assert ids == [[1]]  # obligation #0's invar is var 1


# -- Integration tests: verdict assembly with reachability --------------------


def _make_verdict(closed):
    """Propagate and assemble a verdict from a hand-built query."""
    p = propagate(closed)
    return make_verdict(
        closed,
        p,
        stelling_version="test",
        jax_version="test",
        precision_config="jax_enable_x64=True",
    )


class TestReachableViolationStaysRefuted:
    """Case 1: violation on a live variable -> stays REFUTED."""

    def test_live_violation(self):
        # x in [1, 2], exp(x) < 2.0 is violated (e^1 > 2), and
        # the assert output IS returned.
        x, ex, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=2.0, aval=F64)),
                    outvars=(pred,),
                ),
                assert_eqn(pred, out),
            ],
            (out,),  # assert output IS in the jaxpr's outvars
        )
        v = _make_verdict(closed)
        assert v.status == "REFUTED"
        assert v.obligations[0].status == "violated-over-set"
        # No reachability note
        assert not any("does not reach any output" in n for n in v.notes)


class TestDeadViolationDowngradedToUnknown:
    """Case 2: violation on a dead variable -> UNKNOWN with note."""

    def test_dead_violation(self):
        # x in [1, 2], compute dead = exp(x), check dead < 2.0 (violated).
        # But the output is just x, not the assert output.
        x, ex, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=2.0, aval=F64)),
                    outvars=(pred,),
                ),
                assert_eqn(pred, out),
            ],
            (x,),  # output is x, NOT out -- the assert is dead
        )
        v = _make_verdict(closed)
        assert v.status == "UNKNOWN"
        assert v.obligations[0].status == "unknown"
        assert "dead variable" in v.obligations[0].detail
        assert any("does not reach any output" in n for n in v.notes)

    def test_dead_violation_note_names_obligation_index(self):
        # The note must say which obligation is unreachable.
        x, ex, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=2.0, aval=F64)),
                    outvars=(pred,),
                ),
                assert_eqn(pred, out),
            ],
            (x,),
        )
        v = _make_verdict(closed)
        assert any("obligation #0" in n for n in v.notes)


class TestMixedReachability:
    """Case 3: one reachable violation + one unreachable -> REFUTED, note on dead."""

    def test_mixed_live_and_dead_violations(self):
        # Two violations:
        # - First: assert on a live path (output includes its result)
        # - Second: assert on a dead path (output does NOT include its result)
        x = var(0)
        ex = var(1)
        pred1 = var(2, BOOL)
        out1 = var(3, BOOL)
        pred2 = var(4, BOOL)
        out2 = var(5, BOOL)

        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                # First assert: exp(x) < 2.0, violated
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=2.0, aval=F64)),
                    outvars=(pred1,),
                ),
                assert_eqn(pred1, out1),
                # Second assert: exp(x) < 1.0, also violated
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=1.0, aval=F64)),
                    outvars=(pred2,),
                ),
                assert_eqn(pred2, out2),
            ],
            (out1,),  # ONLY out1 reaches the output; out2 is dead
        )
        v = _make_verdict(closed)
        # Overall: REFUTED because out1's violation is reachable
        assert v.status == "REFUTED"
        # First obligation: live, stays violated
        assert v.obligations[0].status == "violated-over-set"
        # Second obligation: dead, downgraded to unknown
        assert v.obligations[1].status == "unknown"
        assert "dead variable" in v.obligations[1].detail
        # Note about the dead one
        assert any("obligation #1" in n for n in v.notes)
        # No note about the live one
        assert not any("obligation #0" in n for n in v.notes)


class TestNonViolationsUnaffected:
    """Discharged and unknown obligations are never touched by reachability."""

    def test_discharged_not_affected(self):
        # x in [1, 2], exp(x) < 8.0 -> discharged. Even if output is x only.
        x, ex, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(ex,)),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=8.0, aval=F64)),
                    outvars=(pred,),
                ),
                assert_eqn(pred, out),
            ],
            (x,),  # output is x, assert is "dead" -- but it's discharged
        )
        v = _make_verdict(closed)
        assert v.status == "VERIFIED"
        assert v.obligations[0].status == "discharged"
        # No reachability notes (not applicable to non-violations)
        assert not any("does not reach any output" in n for n in v.notes)
