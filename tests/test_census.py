# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The registry-independent census instrument — all jax-free."""

from __future__ import annotations

from stelling import census, ir

F32 = ir.Aval(kind="ShapedArray", shape=(4,), dtype="float32")


def _var(i: int) -> ir.Var:
    return ir.Var(id=i, aval=F32)


def _jaxpr(eqns: tuple[ir.JaxprEqn, ...]) -> ir.Jaxpr:
    return ir.Jaxpr(constvars=(), invars=(_var(0),), outvars=(_var(1),), eqns=eqns)


def _eqn(primitive: str, *params: tuple[str, object]) -> ir.JaxprEqn:
    return ir.JaxprEqn(primitive=primitive, invars=(_var(0),), outvars=(_var(1),), params=tuple(params))


def _target_a() -> ir.ClosedJaxpr:
    inside_jit = _jaxpr((_eqn("gather"), _eqn("add")))
    inside_scan = _jaxpr((_eqn("gather"), _eqn("mul")))
    return ir.ClosedJaxpr(
        jaxpr=_jaxpr(
            (
                _eqn("add"),
                _eqn("jit", ("jaxpr", ir.ClosedJaxpr(jaxpr=inside_jit))),
                _eqn("scan", ("jaxpr", ir.ClosedJaxpr(jaxpr=inside_scan))),
            )
        )
    )


def _target_b() -> ir.ClosedJaxpr:
    return ir.ClosedJaxpr(jaxpr=_jaxpr((_eqn("add"), _eqn("scatter-add"))))


def test_counts_breadth_contexts_and_order():
    acc = census.CensusAccumulator()
    acc.add("a", _target_a())
    acc.add("b", _target_b())
    result = acc.freeze()

    assert result.targets == ("a", "b")
    assert result.total == 9  # 3 top in a + 2 inside jit + 2 inside scan + 2 in b

    stats = {p.name: p for p in result.primitives}
    # add: twice in a (top + inside jit), once in b — breadth 2, ranked first
    assert stats["add"].count == 3 and stats["add"].breadth == 2
    assert result.primitives[0].name == "add"
    assert dict(stats["add"].contexts) == {"top": 2, "transparent": 1}
    # gather: counted everywhere at every depth, regardless of registry
    assert stats["gather"].count == 2 and stats["gather"].breadth == 1
    assert dict(stats["gather"].contexts) == {"transparent": 1, "nested": 1}
    assert stats["gather"].wedge and stats["scatter-add"].wedge
    assert not stats["scan"].wedge
    # equal breadth ties break by count then name
    names = [p.name for p in result.primitives]
    assert names.index("gather") < names.index("mul")


def test_top_and_markdown():
    acc = census.CensusAccumulator()
    acc.add("a", _target_a())
    result = acc.freeze()
    assert result.top(2) == tuple(p.name for p in result.primitives[:2])
    table = result.markdown_table()
    assert "| `gather` | 1/1 | 2 | 0 | 1 | 1 | ✓ |" in table


def test_wedge_predicate():
    assert census.is_wedge_primitive("gather")
    assert census.is_wedge_primitive("scatter")
    assert census.is_wedge_primitive("scatter-add")
    assert census.is_wedge_primitive("dynamic_slice")
    assert not census.is_wedge_primitive("add")
