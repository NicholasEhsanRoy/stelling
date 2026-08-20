# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE REACHES-OUTPUT CONJUNCT IS REMOVED, AND THIS IS THE PROPERTY THAT
REPLACES IT — audit 0.2.0 B8a, item 4 (M1).

`verdict._apply_reachability_conjunct` downgraded a violated obligation to
UNKNOWN when its `operand_var_ids` were in the top-level scope but not in
`reachability.reaches_output`'s live set. It could not fire on a dead
variable — `reaches_output` seeded every `stelling_assert`'s outvars, so the
reverse walk made every assert's INVARS live, and `operand_var_ids` IS the
assert equation's invars — and the TWO inputs that DID reach the downgrade
both silence a genuine REFUTED: a var-id collision between an inner
obligation's operand and a dead top-level id, and a top-level
`stelling_assert` with NO OUTVARS, for which both halves of `reaches_output`
are quantified over an empty tuple and light nothing.

So the file no longer unit-tests `reaches_output` and `defined_vars` (both
gone with the conjunct). What it tests is the standing property:
**no verdict path downgrades a violated obligation on reachability
grounds**, whatever the harness returns and whatever ids the scopes use.
`tests/test_reachability_solver_path.py` drives the same property through
`make_solver_verdict`.
"""

from __future__ import annotations

from stelling import ir
from stelling.propagate import propagate
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


# -- THE FINDING, DRIVEN ------------------------------------------------------


def _collision_query(*, with_dead_top_level_var: bool):
    """`jit{ assert x < 0.5 }` over `x in [1, 2]` — a genuine REFUTED — with
    an optional UNRELATED, UNREAD top-level equation whose outvar id equals
    the inner predicate's id.

    Two scopes, two variables, one number. The removed conjunct read the
    inner obligation's `operand_var_ids` against the TOP-LEVEL live set, so
    the presence of the dead equation — which changes nothing the assert
    depends on — decided the verdict.
    """
    COLLIDING = 7
    inner_x = var(10)
    inner_pred = var(COLLIDING, BOOL)
    inner_out = var(8, BOOL)
    inner = ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=(),
            invars=(inner_x,),
            outvars=(inner_x,),
            eqns=(
                ir.JaxprEqn(
                    primitive="lt",
                    invars=(inner_x, ir.Literal(val=0.5, aval=F64)),
                    outvars=(inner_pred,),
                ),
                assert_eqn(inner_pred, inner_out),
            ),
        )
    )
    x, dead, wrapped = var(0), var(COLLIDING), var(1)
    eqns = [any_eqn(x, 1.0, 2.0)]
    if with_dead_top_level_var:
        eqns.append(ir.JaxprEqn(primitive="exp", invars=(x,), outvars=(dead,)))
    eqns.append(
        ir.JaxprEqn(
            primitive="jit",
            invars=(x,),
            outvars=(wrapped,),
            params=(("jaxpr", inner),),
        )
    )
    return close(eqns, (wrapped,))


def test_an_unrelated_dead_equation_cannot_silence_a_violation():
    """REDDENS ON REVERT of the conjunct's removal.

    Measured on `aabb58d`, with the conjunct in place:

        dead top-level var 7 present: False  ->  STATUS: REFUTED
        dead top-level var 7 present: True   ->  STATUS: UNKNOWN
            note: "obligation #0 is violated but the violated variable does
                   not reach any output of the harness function"

    Same assert, same declared box, same violation; one added equation
    nothing reads. That is the only input that ever reached the downgrade.
    """
    verdicts = {}
    for with_dead in (False, True):
        closed = _collision_query(with_dead_top_level_var=with_dead)
        p = propagate(closed)
        assert [o.status for o in p.obligations] == ["violated-over-set"], (
            "the propagation itself must see the violation in both queries"
        )
        verdicts[with_dead] = make_verdict(
            closed,
            p,
            stelling_version="test",
            jax_version="none: hand-built IR",
            precision_config="jax_enable_x64=True (hand-built f64 IR)",
        )
    assert verdicts[False].status == "REFUTED"
    assert verdicts[True].status == "REFUTED", (
        "a dead top-level equation whose outvar id collides with an inner "
        "obligation's operand id silenced a genuine REFUTED"
    )
    for v in verdicts.values():
        assert not any("does not reach any output" in n for n in v.notes)
        assert v.obligations[0].status == "violated-over-set"


def _zero_outvar_assert_query(*, with_outvar: bool):
    """`x in [1, 2] |- x < 0.5`, the harness returning `x` — a genuine
    REFUTED — with the `stelling_assert` given an outvar or none.

    No collision, no sub-jaxpr, nothing dead. `reaches_output` seeded the
    live set with `for out in eqn.outvars` and walked backwards under
    `any(out.id in live for out in eqn.outvars)`; over `outvars=()` the
    first adds nothing and the second is False, so the predicate never
    became live while `operand_var_ids` was still squarely inside
    `defined_vars`. Jax always gives the assert an outvar; `from_dict`
    does not require one and round-trips a document without one.
    """
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            ir.JaxprEqn(
                primitive="lt",
                invars=(x, ir.Literal(val=0.5, aval=F64)),
                outvars=(pred,),
            ),
            ir.JaxprEqn(
                primitive="stelling_assert",
                invars=(pred,),
                outvars=(out,) if with_outvar else (),
            ),
        ],
        (x,),  # the assert's own outvar is the only thing that could make
    )      # the predicate live, which is the point of the case


def test_an_assert_without_an_outvar_cannot_silence_a_violation():
    """THE SECOND REACHABLE INPUT — audit 0.2.0 B8a FIXUP.

    The removal's argument was that the downgrade held "BY CONSTRUCTION"
    for any top-level assert. It held only for a top-level assert WITH AN
    OUTVAR. Measured on `aabb58d`, with the conjunct in place:

        assert WITH an outvar    ->  STATUS: REFUTED
        assert with ZERO outvars ->  STATUS: UNKNOWN
            obligation #0: violated-over-set -> unknown
            note: "obligation #0 is violated but the violated variable does
                   not reach any output of the harness function"

    Same declared box, same predicate, same violation, no collision and no
    inner scope — so this reached the downgrade on its own, and the
    strengthened case for the removal rests on it.
    """
    verdicts = {}
    for with_outvar in (True, False):
        closed = _zero_outvar_assert_query(with_outvar=with_outvar)
        p = propagate(closed)
        assert [o.status for o in p.obligations] == ["violated-over-set"], (
            "the propagation itself must see the violation in both queries"
        )
        assert p.obligations[0].operand_var_ids == (1,)
        verdicts[with_outvar] = make_verdict(
            closed,
            p,
            stelling_version="test",
            jax_version="none: hand-built IR",
            precision_config="jax_enable_x64=True (hand-built f64 IR)",
        )
    assert verdicts[True].status == "REFUTED"
    assert verdicts[False].status == "REFUTED", (
        "a top-level assert with no outvar silenced a genuine REFUTED"
    )
    for v in verdicts.values():
        assert not any("does not reach any output" in n for n in v.notes)
        assert v.obligations[0].status == "violated-over-set"


def test_the_zero_outvar_assert_is_a_document_from_dict_accepts():
    """The case above is only worth pinning if the IR it needs is IR the
    library will load. It round-trips through `to_dict`/`from_dict` with
    the empty `outvars` intact."""
    closed = _zero_outvar_assert_query(with_outvar=False)
    back = ir.ClosedJaxpr.from_dict(closed.to_dict())
    eqn = back.jaxpr.eqns[-1]
    assert eqn.primitive == "stelling_assert"
    assert eqn.outvars == ()


# -- THE RETURN CONVENTION DOES NOT MOVE A VERDICT ----------------------------


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


class TestAssertNotReturnedIsStillRefuted:
    """A `stelling_assert` is a DECLARATION, so leaving its output out of
    the jaxpr's outvars does not withdraw it and does not soften its
    verdict. `docs/harness-api.md` says the same thing to the user.

    This class's name and docstring used to describe a "dead-variable
    downgrade" that "fires only for obligations whose operand_var_ids are
    out of scope (sub-jaxpr origin) or empty". Neither of those two cases
    downgraded anything — both took the conjunct's fail-safe arm — so the
    sentence described a branch nothing could reach. What could reach it
    is the collision above.
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

    def test_operand_in_scope_and_violated_stays_refuted(self):
        # The ordinary case: the operand is a top-level id, it is live, and
        # the violation stands.
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


class TestTwoAssertsOneReturned:
    """Both asserts stand — the return convention does not matter."""

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
    """A discharged obligation is unaffected by what the harness returns."""

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
# sub-jaxpr obligation must reach the user as the violation it is.

import pytest


class TestSubJaxprInterleavingStaysRefuted:
    """Sub-jaxpr obligations interleaved with top-level ones stay correct."""

    def test_cond_branch_violation_stays_refuted(self):
        """A violation inside a forced cond branch stays REFUTED even when
        a later top-level assert exists."""
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
