# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""THE SOLVER PATH DOES NOT DOWNGRADE A VIOLATION ON REACHABILITY GROUNDS.

This file was written for the reaches-output conjunct, which
``make_solver_verdict`` called exactly as ``make_verdict`` did; the
docstring here promised that "a violated obligation whose predicate operand
does not reach any output of the harness function is downgraded to UNKNOWN"
on both paths. The conjunct is removed (audit 0.2.0 B8a, item 4 — it could
not fire on a dead variable and its one reachable input silenced a genuine
REFUTED; see ``tests/test_reachability_removed.py``), and every case below
already asserted the behaviour that survives it.

What is pinned here:
1. an un-returned assert violated via solver -> REFUTED, with the witness
2. a returned assert violated via solver -> REFUTED, with the witness
3. both verdict paths agree on the same query
4. a solver DISCHARGE on an un-returned assert -> VERIFIED
and, throughout, that no verdict carries a "does not reach any output" note.
"""

from __future__ import annotations

import stat
import sys

import pytest

from stelling import _optional, ir
from stelling.propagate import propagate
from stelling.solvers import (
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import make_verdict
from test_obligation_slice import BOOL, F64, any_eqn, close, eqn, lit, var

VERSIONS = dict(
    stelling_version="test",
    jax_version="none: hand-built IR",
    precision_config="jax_enable_x64=True (hand-built f64 IR)",
)


def dead_square_query(lo=1.0, hi=2.0, bound=2.0):
    """Like square_query but the output is x, NOT the assert output.

    The assert variable is dead: it never reaches any output of the
    harness function.  x in [lo, hi], x^2 <= bound is the obligation.
    """
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [x],  # output is x, NOT out -- the assert is dead
    )


def live_square_query(lo=1.0, hi=2.0, bound=2.0):
    """Like square_query with the output being the assert output.

    The assert variable IS live: it reaches the output.
    """
    x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    return close(
        [
            any_eqn(x, lo, hi),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(bound)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],  # output IS the assert output -- live
    )


def fake_solver(tmp_path, body: str, name: str):
    """A fake external 'solver' that runs body with the script on stdin."""
    text = (
        f"#!{sys.executable}\n"
        "import sys\n"
        'if "--version" in sys.argv[1:]:\n'
        '    print("This is cvc5 version 9.9.9-fake")\n'
        "    sys.exit(0)\n"
        "data = sys.stdin.read()\n"
        f"{body}\n"
    )
    path = tmp_path / name
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# x0 = 3/2 is a valid witness for x^2 <= 2: (3/2)^2 = 9/4 = 2.25 > 2
SAT_BODY = (
    'print("sat")\n'
    'print("(")\n'
    'print("  (define-fun x0 () Real (/ 3 2))")\n'
    'print(")")'
)


class TestAssertNotReturnedViolationSolverPath:
    """An assert is a declaration, so the solver path treats
    assert-not-returned exactly as assert-returned: REFUTED."""

    def test_assert_not_returned_solver_still_refuted(self, monkeypatch, tmp_path):
        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-dead")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = dead_square_query()
        p = propagate(q)
        assert p.obligations[0].status == "unknown"

        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        # The solver found the violation
        assert esc.records[0].outcome == "violated-witness"
        assert esc.records[0].witness is not None

        # Assert is live by intent — verdict is REFUTED (not UNKNOWN)
        v = make_solver_verdict(q, p, esc, **VERSIONS)
        assert v.status == "REFUTED"
        assert not any("does not reach any output" in n for n in v.notes)

    def test_assert_not_returned_matches_interval_path(
        self, monkeypatch, tmp_path
    ):
        """Both paths agree: assert-not-returned is still REFUTED."""
        x, sq, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
        violated_query = close(
            [
                any_eqn(x, 2.0, 3.0),
                eqn("mul", [x, x], sq),
                eqn("le", [sq, lit(2.0)], pred),
                eqn("stelling_assert", [pred], out),
            ],
            [x],  # output is x, not out — but assert is live by intent
        )
        p_interval = propagate(violated_query)
        v_interval = make_verdict(violated_query, p_interval, **VERSIONS)
        assert v_interval.status == "REFUTED"

        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-match")
        monkeypatch.setenv("STELLING_CVC5", fake)
        q = dead_square_query()
        p = propagate(q)
        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        v_solver = make_solver_verdict(q, p, esc, **VERSIONS)
        # Both paths agree: REFUTED
        assert v_solver.status == v_interval.status == "REFUTED"

    def test_solver_witness_preserved_on_refuted(self, monkeypatch, tmp_path):
        """Witnesses are preserved on the REFUTED verdict."""
        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-witness")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = dead_square_query()
        p = propagate(q)
        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        v = make_solver_verdict(q, p, esc, **VERSIONS)

        assert v.status == "REFUTED"
        assert len(v.witnesses) == 1
        assert v.witnesses[0].values == (("x0", "3/2"),)

    def test_dead_violation_solver_stamp_preserved(self, monkeypatch, tmp_path):
        """The solver stamp and all solver-specific fields are unchanged."""
        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-stamp")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = dead_square_query()
        p = propagate(q)
        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        v = make_solver_verdict(q, p, esc, **VERSIONS)

        # Solver stamp is present and records the invocation
        assert isinstance(v.stamp.solver, tuple)
        assert len(v.stamp.solver) == 1
        assert v.stamp.solver[0].invoked is True
        assert v.stamp.solver[0].name == "cvc5"


class TestLiveVariableViolationSolverPath:
    """A live-variable violation found by the solver stays REFUTED."""

    def test_live_violation_via_solver_stays_refuted(self, monkeypatch, tmp_path):
        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-live")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = live_square_query()
        p = propagate(q)
        assert p.obligations[0].status == "unknown"

        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        assert esc.records[0].outcome == "violated-witness"

        v = make_solver_verdict(q, p, esc, **VERSIONS)
        assert v.status == "REFUTED"
        assert v.obligations[0].status == "violated-witness"
        # No reachability downgrade note
        assert not any("does not reach any output" in n for n in v.notes)

    def test_live_violation_preserves_witness(self, monkeypatch, tmp_path):
        fake = fake_solver(tmp_path, SAT_BODY, "cvc5-sat-live-w")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = live_square_query()
        p = propagate(q)
        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        v = make_solver_verdict(q, p, esc, **VERSIONS)

        assert v.status == "REFUTED"
        assert len(v.witnesses) == 1
        assert v.witnesses[0].values == (("x0", "3/2"),)


class TestSolverDischargeUnaffected:
    """A solver-discharged obligation (unsat) is VERIFIED whatever the
    harness returns."""

    def test_unsat_on_unreturned_assert_stays_verified(self, monkeypatch, tmp_path):
        """An un-returned assert discharged by the solver is VERIFIED."""
        # x in [1, 2], x^2 <= 2.0: the interval path is "unknown" (x^2
        # straddles [1, 4] vs 2.0).  The fake solver says "unsat" (no
        # counterexample), so the obligation is discharged.  Even though
        # the output is x (assert is dead), VERIFIED must not be
        # downgraded -- reachability only applies to violations.
        fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat-dead")
        monkeypatch.setenv("STELLING_CVC5", fake)

        q = dead_square_query()  # x in [1, 2], x^2 <= 2.0, output=x
        p = propagate(q)
        assert p.obligations[0].status == "unknown"

        config = SolverConfig(timeout_ms=2000, only=("cvc5",))
        esc = escalate(q, p, config)
        assert esc.records[0].outcome == "discharged"

        v = make_solver_verdict(q, p, esc, **VERSIONS)
        assert v.status == "VERIFIED"
        # No reachability notes (only violations are affected)
        assert not any("does not reach any output" in n for n in v.notes)
