# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The ⊤-widening vacuity library — one implementation, both registered
modes, zero-dep (hand-built IR; no jax required).

The library exists because the transform was convention-copied per
corpus harness and drifted once (L12); these tests pin the two
registered procedures against each other so the difference between them
stays exactly the one registered condition (points hold still under
``inputs-only``).
"""

from __future__ import annotations

import math

import pytest

from stelling import ir
from stelling.vacuity import widen

INF = math.inf


def _any_eqn(var_id, lo, hi, *, shape=()):
    v = ir.Var(id=var_id, aval=ir.Aval(kind="ShapedArray", shape=shape, dtype="float64"))
    return v, ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(v,),
        params=(("lo", lo), ("hi", hi), ("shape", shape), ("dtype", "float64")),
        effects=(),
        source_info=(),
    )


def _query(*bounds):
    """A minimal query: one stelling_any per (lo, hi) plus an assert on
    the first declaration (enough structure for the transform)."""
    eqns, vars_ = [], []
    for i, (lo, hi) in enumerate(bounds):
        v, e = _any_eqn(i, lo, hi)
        vars_.append(v)
        eqns.append(e)
    b = ir.Var(id=100, aval=ir.Aval(kind="ShapedArray", shape=(), dtype="bool"))
    eqns.append(ir.JaxprEqn(
        primitive="gt", invars=(vars_[0], ir.Literal(val=0.0, aval=vars_[0].aval)),
        outvars=(b,), params=(), effects=(), source_info=(),
    ))
    o = ir.Var(id=101, aval=b.aval)
    eqns.append(ir.JaxprEqn(
        primitive="stelling_assert", invars=(b,), outvars=(o,), params=(),
        effects=(), source_info=(),
    ))
    j = ir.Jaxpr(constvars=(), invars=(), outvars=(o,), eqns=tuple(eqns),
                 effects=(), debug_info=None)
    return ir.ClosedJaxpr(jaxpr=j, consts=())


def _bounds(closed):
    return [
        (dict(e.params)["lo"], dict(e.params)["hi"])
        for e in closed.jaxpr.eqns
        if e.primitive == "stelling_any"
    ]


def test_mode_all_widens_every_declaration_including_points():
    q = _query((0.0, 1.0), (2.5, 2.5))
    w = widen(q, mode="all")
    assert _bounds(w) == [(-INF, INF), (-INF, INF)]


def test_mode_inputs_only_holds_points_still():
    """The registered successor procedure: widening a threshold defeats
    the tautology detector, so point declarations stay."""
    q = _query((0.0, 1.0), (2.5, 2.5))
    w = widen(q, mode="inputs-only")
    assert _bounds(w) == [(-INF, INF), (2.5, 2.5)]


def test_modes_differ_exactly_on_points():
    q = _query((0.0, 1.0), (7.0, 7.0), (-3.0, 4.0))
    all_b = _bounds(widen(q, mode="all"))
    io_b = _bounds(widen(q, mode="inputs-only"))
    assert all_b == [(-INF, INF)] * 3
    assert io_b == [(-INF, INF), (7.0, 7.0), (-INF, INF)]


def test_mode_is_required_and_validated():
    q = _query((0.0, 1.0))
    with pytest.raises(TypeError):
        widen(q)  # positional-less mode: keyword required
    with pytest.raises(ValueError):
        widen(q, mode="everything")


def test_original_query_is_not_mutated():
    q = _query((0.0, 1.0))
    widen(q, mode="all")
    assert _bounds(q) == [(0.0, 1.0)]


def test_nested_declaration_fails_loudly():
    """A stelling_any below top level would escape the widening; the
    guard must refuse rather than silently under-widen."""
    v, inner = _any_eqn(0, 0.0, 1.0)
    sub = ir.Jaxpr(constvars=(), invars=(), outvars=(v,), eqns=(inner,),
                   effects=(), debug_info=None)
    outer_v = ir.Var(id=5, aval=v.aval)
    call = ir.JaxprEqn(
        primitive="jit", invars=(), outvars=(outer_v,),
        params=(("jaxpr", ir.ClosedJaxpr(jaxpr=sub, consts=())),),
        effects=(), source_info=(),
    )
    j = ir.Jaxpr(constvars=(), invars=(), outvars=(outer_v,), eqns=(call,),
                 effects=(), debug_info=None)
    q = ir.ClosedJaxpr(jaxpr=j, consts=())
    # A REAL exception, not an assert:  strips asserts, and
    # stripped this guard did not merely stop reporting -- the widened query
    # silently kept the nested declaration's original bounds and the verdict
    # then carried "envelope not load-bearing" about a load-bearing envelope.
    from stelling.vacuity import NestedDeclaration
    with pytest.raises(NestedDeclaration):
        widen(q, mode="all")
    assert not isinstance(NestedDeclaration("x"), AssertionError), (
        "an assert here vanishes under -O, which is how this defect shipped"
    )
