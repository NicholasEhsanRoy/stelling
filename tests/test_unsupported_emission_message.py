# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The unsupported-primitive escalation decline states which gap it is.

docs/proposed-decline-messages.md #3: "primitive 'square' is outside the
supported emission set" named the primitive (the right half) but not
whether an interval row also exists, whether this is a policy refusal or
an unbuilt row, or what the reader should conclude — an external
evaluator saw it next to "square [sound]" in the same stamp's transfer
list and could not reconcile the two. The decline now derives the
interval-row fact and its tier from the LIVE registry
(stelling.propagate.TRANSFERS), so the sentence cannot drift from it,
and names the gap class (an unbuilt row) explicitly.

Message content only: both branches still decline, the DeclinedObligation
shape is unchanged, and the branch selection is bound to registry
membership. Hand-built IR — no jax needed.
"""

from __future__ import annotations

from stelling import ir
from stelling.obligation import (
    DeclinedObligation,
    ObligationSlice,
    slice_unknown_obligations,
)
from stelling.propagate import TRANSFERS, interval_env, propagate

F64 = ir.Aval(kind="ShapedArray", shape=(), dtype="float64")
BOOL = ir.Aval(kind="ShapedArray", shape=(), dtype="bool")


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


def _declined(prim, lo=1.0, hi=2.0, bound=2.5):
    """Drive an unknown obligation whose slice crosses ``prim`` and return
    the DeclinedObligation's reason plus the propagation it rode on."""
    x, t, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, lo, hi),
            eqn(prim, [x], t),
            eqn("lt", [t, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert p.obligations[0].status == "unknown"
    (item,) = slice_unknown_obligations(q, p, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert not isinstance(item, ObligationSlice)  # still declines
    return item.reason, p


def _declined_reason(prim):
    return _declined(prim)[0]


def test_interval_row_present_is_named_with_its_live_tier():
    # 'square' is the external evaluator's own case: an interval row
    # exists (and its tier is read from the registry, not narrated)
    assert "square" in TRANSFERS
    tier = TRANSFERS["square"][1]
    reason, p = _declined("square")
    # 'alone' may be claimed only because the run recorded no square-⊤
    assert "square" not in dict(p.coverage.unknown_primitives)
    assert "primitive 'square' is outside the supported emission set" in reason
    assert "an unbuilt row, not a policy refusal of the form" in reason
    assert (
        f"An interval transfer row for 'square' IS registered "
        f"(tier {tier!r}), so the gap is the solver-emission row alone"
        in reason
    )
    # the no-row sentence must not ride on this branch
    assert "no interval transfer row either" not in reason


def test_registered_row_that_declined_this_run_names_both_gaps():
    # blinded-lens audit R2: sqrt over a negative-lo box — the registered
    # interval row DECLINED the triggering form on this very run, so 'the
    # solver-emission row alone' would be false; both gaps are named,
    # bound to the run's own record
    assert "sqrt" in TRANSFERS
    tier = TRANSFERS["sqrt"][1]
    reason, p = _declined("sqrt", lo=-1.0, hi=4.0)
    assert dict(p.coverage.unknown_primitives) == {"sqrt": 1}  # it fell
    assert (
        f"An interval transfer row for 'sqrt' IS registered "
        f"(tier {tier!r}), but on this run the interval leg ALSO fell "
        f"to ⊤ at 'sqrt' (its transfer declined a form — see the "
        f"coverage line and the decline notes), so both legs have a gap "
        f"here" in reason
    )
    assert "solver-emission row alone" not in reason


def test_structurally_handled_primitive_is_not_accused_of_top():
    # blinded-lens audit R3: cond has no TRANSFERS row yet the propagator
    # joins its branches — coverage records NO ⊤, and the old sentence
    # asserted one that never happened. Measured on a real traced cond.
    import pytest

    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        from stelling.harness import any_array, assert_, trace

        def h():
            g = any_array((), "float64", (0.0, 1.0))
            x = any_array((), "float64", (1.0, 2.0))
            y = jax.lax.cond(g > 0.5, lambda v: v, lambda v: v + 100.0, x)
            return assert_(y <= 4.0)

        cj = trace(h)
        p = propagate(cj)
    finally:
        jax.config.update("jax_enable_x64", old)
    assert "cond" not in TRANSFERS
    assert p.coverage.unknown == 0  # the join really happened; no ⊤
    assert p.obligations[0].status == "unknown"
    items = slice_unknown_obligations(cj, p, interval_env(cj))
    reasons = [
        i.reason for i in items if isinstance(i, DeclinedObligation)
    ]
    (reason,) = [r for r in reasons if "'cond'" in r]
    assert (
        "It has no interval transfer row either, yet it did not fall to "
        "⊤ on this run: the interval leg handled it structurally (see "
        "the coverage line)" in reason
    )
    assert "propagated ⊤ for it" not in reason


def test_without_a_run_record_neither_direction_is_claimed():
    # slice_obligation without top_primitives has no run record: the
    # sentence must claim neither 'alone' nor 'both gaps'
    from stelling.obligation import slice_obligation

    x, t, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("square", [x], t),
            eqn("lt", [t, lit(2.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    item = slice_obligation(q, 0, interval_env(q))
    assert isinstance(item, DeclinedObligation)
    assert (
        "the coverage line records whether it covered this run's forms"
        in item.reason
    )
    assert "solver-emission row alone" not in item.reason
    assert "both legs have a gap" not in item.reason


def test_no_interval_row_names_the_top_and_points_at_coverage():
    assert "sin" not in TRANSFERS
    reason = _declined_reason("sin")
    assert "primitive 'sin' is outside the supported emission set" in reason
    assert "an unbuilt row, not a policy refusal of the form" in reason
    assert (
        "It has no interval transfer row either, so the interval leg "
        "propagated ⊤ for it (see the coverage line)" in reason
    )
    # the registered-row sentence must not ride on this branch
    assert "IS registered" not in reason and "tier" not in reason


def test_the_top_claim_is_true_of_the_propagation_it_describes():
    # the no-row branch claims the interval leg propagated ⊤ — measured
    # on the same query: the primitive is counted ⊤ by the instrument
    x, t, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("sin", [x], t),
            eqn("lt", [t, lit(2.5)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    assert dict(p.coverage.unknown_primitives) == {"sin": 1}
