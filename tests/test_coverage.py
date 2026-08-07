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


# --- the canonical call-body accessor -----------------------------------
#
# jax-free on purpose: the SHAPES these assert are the two a real jax hands
# over (remat2's body is a bare Jaxpr on 0.10 and a ClosedJaxpr on 0.11),
# but the accessor's contract is about ir, so it is pinned without a jax.


def _body(eqns=(_eqn("mul"),)) -> ir.Jaxpr:
    return _jaxpr(eqns)


def test_call_body_reads_both_container_shapes():
    """The shape a wrapper's body arrives in is a fact about the jax series,
    not about the callee, so the accessor must not care."""
    closed = ir.ClosedJaxpr(jaxpr=_body(), consts=())
    assert coverage.call_body(_eqn("jit", ("jaxpr", closed))) is closed

    open_body = _body()
    got = coverage.call_body(_eqn("remat2", ("jaxpr", open_body)))
    assert isinstance(got, ir.ClosedJaxpr)
    assert got.jaxpr is open_body and got.consts == ()


def test_call_body_is_none_when_there_is_no_body():
    assert coverage.call_body(_eqn("jit", ("inline", False))) is None
    assert coverage.call_body(_eqn("jit")) is None


def test_call_body_prefers_a_closed_body_over_a_bare_one():
    """So that adding the bare-Jaxpr arm cannot change which param is chosen
    on any equation the ClosedJaxpr-only code already handled."""
    closed = ir.ClosedJaxpr(jaxpr=_body(), consts=())
    eqn = _eqn("jit", ("first", _body()), ("second", closed))
    assert coverage.call_body(eqn) is closed


def test_call_body_does_not_invent_consts_for_a_body_that_needs_them():
    """Closing a bare jaxpr over () is lossless only when it has no
    constvars. When it has them, the pair is deliberately inconsistent so
    that every caller's `len(constvars) == len(consts)` guard REFUSES the
    inline — the right answer for a body whose consts are not available."""
    needs_consts = ir.Jaxpr(
        constvars=(_var(9),), invars=(_var(0),), outvars=(_var(1),), eqns=(_eqn("mul"),)
    )
    got = coverage.call_body(_eqn("remat2", ("jaxpr", needs_consts)))
    assert len(got.jaxpr.constvars) == 1 and got.consts == ()
    assert len(got.jaxpr.constvars) != len(got.consts)
