# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The solver-escalation acceptance cases, end to end with real solvers.

The worked example with a known answer: a degree-2 polynomial obligation
in three bounded reals (trace → propagate → escalate → verdict). Harness A
(``b ∈ [-0.5, 0.5]``) straddles under interval propagation and must be
VERIFIED by the QF_NRA portfolio (cvc5 with coverings, plus z3); harness B
(``b ∈ [-1.4, 1.4]``) must come back REFUTED **with a replay-confirmed
witness rendered**. Also witnessed here: each solver alone (degraded
portfolio), the no-solver run (UNKNOWN with the absence stamp — the
propagation-only behavior, unchanged), and the non-rational-model policy
(sat at √2 only → UNKNOWN, never a guessed witness).

Skipped per availability: jax to trace; z3/cvc5 per test."""

from __future__ import annotations

from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_, nonvacuity, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.solvers import (  # noqa: E402
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import make_verdict  # noqa: E402

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None

need_both = pytest.mark.skipif(
    not (HAVE_Z3 and HAVE_CVC5), reason="needs both z3 and cvc5"
)

VERSIONS = dict(
    stelling_version="test",
    jax_version=jax.__version__,
    precision_config="jax_enable_x64=True",
)
CONFIG = SolverConfig(timeout_ms=30_000)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def harness(b_lo, b_hi):
    def h():
        a = any_array((), "float64", (1.0, 2.0))
        c = any_array((), "float64", (1.0, 2.0))
        b = any_array((), "float64", (b_lo, b_hi))
        tr = a + c
        det = a * c - b * b
        o = assert_(tr * tr <= det * 10.125)  # 10.125 = 81/8 exactly
        # nonvacuity: the point (a, b, c) = (1.5, 0.0, 1.5) is in the box
        ap = any_array((), "float64", (1.5, 1.5))
        bp = any_array((), "float64", (0.0, 0.0))
        cp = any_array((), "float64", (1.5, 1.5))
        n = [nonvacuity(ap >= 1.0), nonvacuity(ap <= 2.0),
             nonvacuity(bp >= -0.5), nonvacuity(bp <= 0.5),
             nonvacuity(cp >= 1.0), nonvacuity(cp <= 2.0)]
        return (o, *n)

    return h


def h_A():
    return harness(-0.5, 0.5)


def h_B():
    return harness(-1.4, 1.4)


def _predicate_violated(values: dict[str, str]) -> bool:
    """The acceptance obligation recomputed independently of stelling, in
    exact rationals: tr² <= det·(81/8) at (a, c, b) = (x0, x1, x2)."""
    a, c, b = (Fraction(values[k]) for k in ("x0", "x1", "x2"))
    tr2 = (a + c) ** 2
    rhs = (a * c - b * b) * Fraction(81, 8)
    return not (tr2 <= rhs)


def test_interval_alone_leaves_the_acceptance_obligation_unknown():
    p = propagate(trace(h_A()))
    assert [o.status for o in p.obligations] == ["unknown"]


@need_both
def test_acceptance_A_portfolio_verifies_with_both_stamps():
    cj = trace(h_A())
    p = propagate(cj)
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "discharged"
    assert len(record.invocations) == 2  # cvc5 AND z3, both stamped
    assert {s.name for s in record.invocations} == {"cvc5", "z3"}
    assert all(s.invoked for s in record.invocations)
    assert record.invocations[0].name == "cvc5"  # primary for QF_NRA
    for s in record.invocations:
        opts = dict(s.options)
        assert "smt2_sha256" in opts and opts["set-logic"] == "QF_NRA"
        assert opts[":produce-models"] == "true"
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 2
    assert v.stamp.nonvacuity.startswith("checked")


@need_both
def test_acceptance_B_refuted_with_replay_confirmed_witness():
    cj = trace(h_B())
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    witness = record.witness
    assert witness is not None
    # the solver may return any witness; whatever came back must genuinely
    # violate the predicate — recomputed here in exact rationals without
    # any stelling arithmetic in the loop
    assert _predicate_violated(dict(witness.values))
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    rendered = v.render()
    assert "witness for assert #0" in rendered
    for name, value in witness.values:
        assert f"{name} = {value}" in rendered
    assert "replay" in rendered
    assert "Not a witness" not in rendered  # that text is interval-only


@pytest.mark.parametrize(
    "solver_name",
    [
        pytest.param(
            "z3", marks=pytest.mark.skipif(not HAVE_Z3, reason="needs z3")
        ),
        pytest.param(
            "cvc5", marks=pytest.mark.skipif(not HAVE_CVC5, reason="needs cvc5")
        ),
    ],
)
def test_acceptance_A_each_solver_alone_degraded_portfolio(solver_name):
    cj = trace(h_A())
    p = propagate(cj)
    esc = escalate(cj, p, SolverConfig(timeout_ms=30_000, only=(solver_name,)))
    (record,) = esc.records
    assert record.outcome == "discharged"
    assert len(record.invocations) == 1
    assert record.invocations[0].name == solver_name
    assert any("portfolio degraded" in n for n in record.notes)
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"


def test_acceptance_A_no_solver_is_unknown_with_absence_stamp(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(_optional, "available", lambda name: False)
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: None)
    cj = trace(h_A())
    p = propagate(cj)
    esc = escalate(cj, p, SolverConfig(timeout_ms=1000))
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.solver.invoked is False
    # and the interval-only public path: byte-identical behavior, UNKNOWN
    plain = make_verdict(cj, p, **VERSIONS)
    assert plain.status == "UNKNOWN"
    assert plain.stamp.solver.invoked is False


@need_both
def test_sat_at_sqrt2_only_stays_unknown_by_policy():
    # ¬P is satisfiable only at x = √2: both solvers say sat, neither model
    # is rational, no witness is guessable — UNKNOWN by policy.
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x * x != 2.0)

    cj = trace(h)
    p = propagate(cj)
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(cj, p, CONFIG)
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert "not independently replayable" in record.detail
    v = make_solver_verdict(cj, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert len(v.stamp.solver) == 2  # both invocations still stamped
