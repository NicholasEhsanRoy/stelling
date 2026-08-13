# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Tests for the reaches-output reachability conjunct.

Three cases per the design:
1. Violation on a variable that IS used in the output -> stays REFUTED
2. Violation on a dead variable (computed but not returned) -> UNKNOWN with note
3. Mixed: one reachable violation, one unreachable -> REFUTED, unreachable gets note
"""

from __future__ import annotations

from stelling import ir
from stelling.propagate import propagate
from stelling.reachability import defined_vars, reaches_output
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


# -- Unit tests for defined_vars -----------------------------------------------


def test_defined_vars_includes_all_equation_outvars():
    """defined_vars returns all Var IDs in the top-level scope."""
    x, y, z = var(0), var(1), var(2)
    jaxpr = ir.Jaxpr(
        constvars=(),
        invars=(),
        outvars=(z,),
        eqns=(
            any_eqn(x, 0.0, 1.0),
            ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(y,)),
            ir.JaxprEqn(primitive="exp", invars=(y,), outvars=(z,)),
        ),
    )
    scope = defined_vars(jaxpr)
    assert scope == frozenset({0, 1, 2})


def test_operand_var_ids_propagated():
    """ObligationReport carries operand_var_ids from propagation."""
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
        (out,),
    )
    p = propagate(closed)
    # The obligation should carry the var ID of the assert's invar (pred = var 2)
    assert p.obligations[0].operand_var_ids == (2,)


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
    """Case 2: a stelling_assert is ALWAYS live (it's a declaration), so
    the dead-variable downgrade fires only for obligations whose
    operand_var_ids are out of scope (sub-jaxpr origin) or empty.

    Before the assert-always-live fix, an assert whose output wasn't in
    the jaxpr's outvars was treated as dead. That was wrong: the user
    wrote the assert, so they care about it. The downgrade now serves
    a narrower purpose: protecting against positional misalignment of
    sub-jaxpr obligations.
    """

    def test_assert_not_returned_still_refuted(self):
        # An assert whose output is NOT in outvars is still REFUTED —
        # the assert is a declaration and its operand is live by intent.
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
            (x,),  # output is x, NOT out — assert still live
        )
        v = _make_verdict(closed)
        assert v.status == "REFUTED"
        assert v.obligations[0].status == "violated-over-set"

    def test_out_of_scope_operand_ids_downgraded(self):
        # An obligation with operand_var_ids that are NOT in the top-level
        # scope (simulating a sub-jaxpr obligation) gets downgraded.
        x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
        closed = close(
            [
                any_eqn(x, 1.0, 2.0),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(x, ir.Literal(val=0.5, aval=F64)),
                    outvars=(pred,),
                ),
                assert_eqn(pred, out),
            ],
            (out,),
        )
        v = _make_verdict(closed)
        # Verdict is REFUTED (normal case, operand in scope)
        assert v.status == "REFUTED"

    def test_assert_not_returned_no_downgrade_note(self):
        # An assert not in outvars should NOT produce a dead-variable note.
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
        assert v.status == "REFUTED"
        assert not any("does not reach any output" in n for n in v.notes)


class TestMixedReachability:
    """Both asserts are always live — the return convention doesn't matter."""

    def test_both_asserts_live_regardless_of_outvars(self):
        # Two violated asserts. Only out1 is in outvars, but BOTH assert
        # outvars are seeded as live (asserts are declarations).
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
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=2.0, aval=F64)),
                    outvars=(pred1,),
                ),
                assert_eqn(pred1, out1),
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(ex, ir.Literal(val=1.0, aval=F64)),
                    outvars=(pred2,),
                ),
                assert_eqn(pred2, out2),
            ],
            (out1,),  # only out1 in outvars, but both asserts are live
        )
        v = _make_verdict(closed)
        assert v.status == "REFUTED"
        # Both obligations stay violated (both asserts are live)
        assert v.obligations[0].status == "violated-over-set"
        assert v.obligations[1].status == "violated-over-set"
        # No dead-variable notes
        assert not any("does not reach any output" in n for n in v.notes)


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


# -- Interleaving test (jax required) -----------------------------------------
# A cond with an assert inside a branch BEFORE a top-level assert: the
# sub-jaxpr obligation must NOT be incorrectly downgraded.

import pytest


class TestSubJaxprInterleavingNotDowngraded:
    """Sub-jaxpr obligations interleaved with top-level ones stay correct."""

    def test_cond_branch_violation_not_downgraded(self):
        """A violation inside a forced cond branch stays REFUTED even when
        a later top-level assert exists -- the positional index must not
        misalign and incorrectly downgrade the branch obligation."""
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from stelling.harness import any_array, assert_
        from stelling.preconditions import check

        saved_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_x64", True)

        def h():
            x = any_array((), jnp.float64, (1.0, 2.0))
            # cond whose predicate is always True (x > 0 for x in [1,2])
            branch_result = jax.lax.cond(
                x > 0.0,
                lambda v: assert_(v > 5.0),  # violated: x in [1,2] < 5
                lambda v: assert_(v > -9.0),
                x,
            )
            return branch_result

        try:
            v = check(h, vacuity_mode="all")
            assert v.status == "REFUTED"
        finally:
            jax.config.update("jax_enable_x64", saved_x64)
