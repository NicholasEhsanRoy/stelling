# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The ⊤-coverage instrument — all jax-free."""

from __future__ import annotations

from stelling import coverage, ir

F32 = ir.Aval(kind="ShapedArray", shape=(4,), dtype="float32")


def _var(i: int) -> ir.Var:
    return ir.Var(id=i, aval=F32)


def _jaxpr(eqns: tuple[ir.JaxprEqn, ...]) -> ir.Jaxpr:
    return ir.Jaxpr(constvars=(), invars=(_var(0),), outvars=(_var(1),), eqns=eqns)


def _eqn(primitive: str, *params: tuple[str, object]) -> ir.JaxprEqn:
    return ir.JaxprEqn(primitive=primitive, invars=(_var(0),), outvars=(_var(1),), params=tuple(params))


def test_measure_buckets_and_unreached():
    inner_transparent = _jaxpr((_eqn("mul"), _eqn("gather")))
    inner_unknown = _jaxpr((_eqn("sin"), _eqn("cos")))
    top = ir.ClosedJaxpr(
        jaxpr=_jaxpr(
            (
                _eqn("add"),
                _eqn("jit", ("jaxpr", ir.ClosedJaxpr(jaxpr=inner_transparent))),
                _eqn("cond", ("branches", (ir.ClosedJaxpr(jaxpr=inner_unknown),))),
            )
        )
    )
    cov = coverage.measure(top, known={"add", "mul"})
    assert cov.total == 7
    assert cov.known == 2  # add, mul
    assert cov.transparent == 1  # jit
    assert cov.unknown == 2  # gather (reached via jit), cond
    assert cov.unreached == 2  # sin, cos — behind the unknown cond
    assert cov.unknown_primitives == (("cond", 1), ("gather", 1))
    assert 0 < cov.fraction_known < 1


def test_summary_is_a_quantity():
    cov = coverage.measure(
        ir.ClosedJaxpr(jaxpr=_jaxpr((_eqn("add"), _eqn("gather"), _eqn("gather")))),
        known={"add"},
    )
    text = cov.summary()
    assert "3 eqns" in text
    assert "1 known" in text
    assert "2 ⊤" in text
    assert "gather ×2" in text


def test_empty_query_is_fully_covered():
    cov = coverage.measure(ir.ClosedJaxpr(jaxpr=_jaxpr(())), known=set())
    assert cov.total == 0
    assert cov.fraction_known == 1.0


def test_counter_matches_measure_shape():
    counter = coverage.CoverageCounter()
    counter.record_known("add")
    counter.record_transparent("jit")
    counter.record_unknown("gather")
    counter.record_unknown("gather")
    counter.record_unknown("scatter")
    counter.record_unreached("sin")
    cov = counter.freeze()
    assert (cov.total, cov.known, cov.transparent, cov.unknown, cov.unreached) == (6, 1, 1, 3, 1)
    assert cov.unknown_primitives == (("gather", 2), ("scatter", 1))
