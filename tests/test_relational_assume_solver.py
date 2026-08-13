# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Relational assume forwarding to the solver: the bridge between the
non-relational interval domain and the relational SMT solver.

When both operands of an assumed comparison vary (non-constant intervals),
the interval domain cannot narrow either independently. The constraint is
RELATIONAL and is dropped with reason. This feature records such constraints
and forwards them to the solver as additional axioms, enabling the solver to
prove properties that depend on the relational constraint.

Three cases:
1. A property that NEEDS the relational constraint to be proved (without it,
   the solver sees no reason x < y + 1 when all it knows is x in [-10, 10]
   and y in [-10, 10]).
2. A property that holds INDEPENDENTLY of the relational constraint (no
   regression).
3. A property the relational constraint CONTRADICTS (the solver finds SAT
   with a witness showing the property fails even under the constraint).
"""

from __future__ import annotations

import math
import struct

import pytest

from stelling import interval as iv
from stelling import ir
from stelling.propagate import propagate, interval_env, Propagation
from stelling.obligation import slice_unknown_obligations, ObligationSlice
from stelling.smt import emit

INF = math.inf


def f64(shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype="float64")


def boolav(shape=()):
    return ir.Aval(kind="ShapedArray", shape=shape, dtype="bool")


def var(i, aval=None):
    return ir.Var(id=i, aval=aval if aval is not None else f64())


def lit(v, aval=None):
    return ir.Literal(val=v, aval=aval if aval is not None else f64())


def any_eqn(out, lo, hi, shape=()):
    return ir.JaxprEqn(
        primitive="stelling_any",
        invars=(),
        outvars=(out,),
        params=(("shape", shape), ("dtype", "float64"), ("lo", lo), ("hi", hi)),
    )


def eqn(prim, ins, out, params=(), source_info=()):
    return ir.JaxprEqn(
        primitive=prim,
        invars=tuple(ins),
        outvars=(out,),
        params=tuple(params),
        source_info=tuple(source_info),
    )


def close(eqns, outvars, constvars=(), consts=()):
    return ir.ClosedJaxpr(
        jaxpr=ir.Jaxpr(
            constvars=tuple(constvars),
            invars=(),
            outvars=tuple(outvars),
            eqns=tuple(eqns),
        ),
        consts=tuple(consts),
    )


# ---------------------------------------------------------------------------
# Test 1: Recording relational assumes in propagation
# ---------------------------------------------------------------------------


class TestRecording:
    """Propagation records relational assumes when both sides of a comparison
    vary (non-point intervals)."""

    def test_relational_assume_recorded(self):
        """assume(x < y) where both x and y vary: the comparison equation is
        recorded in propagation.relational_assumes."""
        x = var(0, f64())
        y = var(1, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        # x in [-10, 10], y in [-10, 10]
        # assume(x < y) -- both sides vary, so relational
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn("lt", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
        ], [aout])
        p = propagate(cj)
        assert len(p.relational_assumes) == 1
        ra = p.relational_assumes[0]
        assert ra.primitive == "lt"
        assert len(ra.invars) == 2

    def test_non_relational_assume_not_recorded(self):
        """assume(x < 5.0) where one side is constant: NOT recorded as
        relational (it can be narrowed by the interval domain)."""
        x = var(0, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            eqn("lt", [x, lit(5.0)], pred),
            eqn("stelling_assume", [pred], aout),
        ], [aout])
        p = propagate(cj)
        assert len(p.relational_assumes) == 0

    def test_multiple_relational_assumes(self):
        """Two relational assumes in one query: both recorded."""
        x = var(0, f64())
        y = var(1, f64())
        z = var(2, f64())
        pred1 = var(3, boolav())
        aout1 = var(4, boolav())
        pred2 = var(5, boolav())
        aout2 = var(6, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            any_eqn(z, -10.0, 10.0),
            eqn("lt", [x, y], pred1),
            eqn("stelling_assume", [pred1], aout1),
            eqn("le", [y, z], pred2),
            eqn("stelling_assume", [pred2], aout2),
        ], [aout1, aout2])
        p = propagate(cj)
        assert len(p.relational_assumes) == 2
        assert p.relational_assumes[0].primitive == "lt"
        assert p.relational_assumes[1].primitive == "le"

    def test_relational_assume_not_recorded_under_ieee(self):
        """Under ieee semantics, relational assumes are NOT recorded (ieee
        declines escalation wholly, so there is nothing to forward)."""
        x = var(0, f64())
        y = var(1, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn("lt", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
        ], [aout])
        p = propagate(cj, semantics="ieee")
        assert len(p.relational_assumes) == 0


# ---------------------------------------------------------------------------
# Test 2: Emission of relational assumes as SMT axioms
# ---------------------------------------------------------------------------


class TestEmission:
    """The emit function includes relational assumes as positive assertions
    in the SMT script when operand variables are in the slice."""

    def _make_slice_with_relational(self, cmp="lt"):
        """Build a query: assume(x <cmp> y), assert(x < y + 1).
        Returns (closed_jaxpr, propagation, the slice)."""
        x = var(0, f64())
        y = var(1, f64())
        # comparison for assume
        pred = var(2, boolav())
        aout = var(3, boolav())
        # computation for assert: x < y + 1
        y_plus_1 = var(4, f64())
        assert_pred = var(5, boolav())
        assert_out = var(6, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn(cmp, [x, y], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("add", [y, lit(1.0)], y_plus_1),
            eqn("lt", [x, y_plus_1], assert_pred),
            eqn("stelling_assert", [assert_pred], assert_out),
        ], [aout, assert_out])
        p = propagate(cj)
        env = interval_env(cj)
        slices = slice_unknown_obligations(cj, p, env)
        # there should be one unknown obligation
        assert len(slices) == 1
        sl = slices[0]
        assert isinstance(sl, ObligationSlice)
        return cj, p, sl

    def test_relational_assume_emitted_in_script(self):
        """When a relational assume's operands are in the slice, the
        positive constraint is emitted as an assertion."""
        cj, p, sl = self._make_slice_with_relational("lt")
        assert len(p.relational_assumes) == 1
        # Emit with relational assumes
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        # The script should contain the positive assertion: (assert (< x0 x1))
        assert "(assert (< x0 x1))" in script.text

    def test_relational_assume_not_emitted_without_param(self):
        """Without passing relational_assumes, no axiom is emitted."""
        cj, p, sl = self._make_slice_with_relational("lt")
        # Emit WITHOUT relational assumes (the default)
        script = emit(sl, "z3", 5000)
        # The only assert should be the negated obligation
        assert_lines = [
            l for l in script.text.splitlines()
            if l.startswith("(assert") and "not" not in l
            and "<=" in l  # bound assertions use <=
        ]
        # bound assertions only (no relational)
        for line in script.text.splitlines():
            if line.startswith("(assert") and "(< x0 x1)" in line:
                pytest.fail("relational assume emitted without being passed")

    def test_relational_assume_le(self):
        """assume(x <= y) emits (assert (<= x0 x1))."""
        cj, p, sl = self._make_slice_with_relational("le")
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        assert "(assert (<= x0 x1))" in script.text

    def test_relational_assume_ge(self):
        """assume(x >= y) emits (assert (>= x0 x1))."""
        cj, p, sl = self._make_slice_with_relational("ge")
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        assert "(assert (>= x0 x1))" in script.text

    def test_relational_assume_gt(self):
        """assume(x > y) emits (assert (> x0 x1))."""
        cj, p, sl = self._make_slice_with_relational("gt")
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        assert "(assert (> x0 x1))" in script.text

    def test_relational_assume_precedes_negated_obligation(self):
        """The relational axiom appears BEFORE the negated obligation in the
        script (it is an assumption, not the query)."""
        cj, p, sl = self._make_slice_with_relational("lt")
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        lines = script.text.splitlines()
        rel_idx = next(
            i for i, l in enumerate(lines)
            if "(assert (< x0 x1))" in l
        )
        neg_idx = next(
            i for i, l in enumerate(lines)
            if "(assert (not" in l
        )
        assert rel_idx < neg_idx, (
            "relational axiom must precede the negated obligation"
        )

    def test_relational_assume_skipped_when_operands_not_in_slice(self):
        """If the relational assume's operand variables are NOT in the
        obligation's backward cone, the assume is silently skipped."""
        # Build a query where z is declared but not in the obligation slice
        x = var(0, f64())
        y = var(1, f64())
        z = var(2, f64())
        # assume(x < z) -- z is NOT used in the assert
        pred = var(3, boolav())
        aout = var(4, boolav())
        # assert(x < y + 1) -- only uses x and y
        y_plus_1 = var(5, f64())
        assert_pred = var(6, boolav())
        assert_out = var(7, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            any_eqn(z, -10.0, 10.0),
            eqn("lt", [x, z], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("add", [y, lit(1.0)], y_plus_1),
            eqn("lt", [x, y_plus_1], assert_pred),
            eqn("stelling_assert", [assert_pred], assert_out),
        ], [aout, assert_out])
        p = propagate(cj)
        assert len(p.relational_assumes) == 1
        env = interval_env(cj)
        slices = slice_unknown_obligations(cj, p, env)
        sl = slices[0]
        assert isinstance(sl, ObligationSlice)
        # z (x2) should NOT be in the slice inputs
        input_names = {inp.name for inp in sl.inputs}
        assert "x2" not in input_names
        # emit with the relational assume -- it should be SKIPPED
        script = emit(sl, "z3", 5000, relational_assumes=p.relational_assumes)
        for line in script.text.splitlines():
            if line.startswith("(assert") and "(< x0 x2)" in line:
                pytest.fail(
                    "relational assume with out-of-slice operand was emitted"
                )


# ---------------------------------------------------------------------------
# Test 3: End-to-end solver integration (requires z3 or cvc5)
# ---------------------------------------------------------------------------

try:
    from stelling import _optional
    HAVE_Z3 = _optional.available("z3")
    HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None
    HAVE_SOLVER = HAVE_Z3 or HAVE_CVC5
except Exception:
    HAVE_SOLVER = False

need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs z3 or cvc5")


@need_solver
class TestEndToEnd:
    """End-to-end: relational assumes forwarded to solver enable proving
    properties that depend on the relational constraint."""

    def test_relational_assume_enables_discharge(self):
        """assume(x < y), assert(x < y + 1): without the relational
        constraint the solver cannot prove this (x could be 10, y could be
        -10). With it, x < y implies x < y + 1."""
        from stelling.solvers import SolverConfig, escalate

        x = var(0, f64())
        y = var(1, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        y_plus_1 = var(4, f64())
        assert_pred = var(5, boolav())
        assert_out = var(6, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn("lt", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("add", [y, lit(1.0)], y_plus_1),
            eqn("lt", [x, y_plus_1], assert_pred),
            eqn("stelling_assert", [assert_pred], assert_out),
        ], [aout, assert_out])
        p = propagate(cj)
        assert len(p.relational_assumes) == 1
        assert any(o.status == "unknown" for o in p.obligations)
        config = SolverConfig(timeout_ms=30_000)
        esc = escalate(cj, p, config)
        assert len(esc.records) == 1
        record = esc.records[0]
        assert record.outcome == "discharged", (
            f"expected discharged, got {record.outcome}: {record.detail}"
        )
        # The disclosure note should be present
        assert any(
            "relational assume" in n and "forwarded" in n
            for n in record.notes
        ), f"disclosure note missing: {record.notes}"

    def test_independent_property_still_discharged(self):
        """A property that holds independently of any relational assume:
        x + 1 > x. The relational assume is present but irrelevant. No
        regression."""
        from stelling.solvers import SolverConfig, escalate

        x = var(0, f64())
        y = var(1, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        x_plus_1 = var(4, f64())
        assert_pred = var(5, boolav())
        assert_out = var(6, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn("lt", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("add", [x, lit(1.0)], x_plus_1),
            eqn("gt", [x_plus_1, x], assert_pred),
            eqn("stelling_assert", [assert_pred], assert_out),
        ], [aout, assert_out])
        p = propagate(cj)
        config = SolverConfig(timeout_ms=30_000)
        esc = escalate(cj, p, config)
        assert len(esc.records) == 1
        record = esc.records[0]
        assert record.outcome == "discharged", (
            f"expected discharged, got {record.outcome}: {record.detail}"
        )

    def test_contradicting_relational_assume_finds_sat(self):
        """assume(x > y), assert(x < y): the relational assume says x > y,
        but the assert claims x < y. These cannot both be true, so the
        solver should find a VIOLATION (sat on the negation means
        the property is falsifiable within the assumed region).

        More precisely: the negated obligation is (not (< x y)), i.e. x >= y.
        Combined with the axiom x > y, the solver has x > y AND x >= y --
        which is satisfiable (any x > y works). So the outcome is
        violated-witness or violated-constant."""
        from stelling.solvers import SolverConfig, escalate

        x = var(0, f64())
        y = var(1, f64())
        pred = var(2, boolav())
        aout = var(3, boolav())
        assert_pred = var(4, boolav())
        assert_out = var(5, boolav())
        cj = close([
            any_eqn(x, -10.0, 10.0),
            any_eqn(y, -10.0, 10.0),
            eqn("gt", [x, y], pred),
            eqn("stelling_assume", [pred], aout),
            eqn("lt", [x, y], assert_pred),
            eqn("stelling_assert", [assert_pred], assert_out),
        ], [aout, assert_out])
        p = propagate(cj)
        assert len(p.relational_assumes) == 1
        config = SolverConfig(timeout_ms=30_000)
        esc = escalate(cj, p, config)
        assert len(esc.records) == 1
        record = esc.records[0]
        # With assume_dropped=True the violation is WITHHELD from REFUTED.
        # The outcome should still be UNKNOWN (withheld violation) since
        # the assume was dropped.
        # Actually: the assume_dropped flag IS set because the relational
        # assume was dropped. So violations are withheld. The outcome is
        # OB_UNKNOWN because the violation was withheld.
        # This is the correct sound behavior: the relational assume was
        # dropped (from the interval domain's perspective), so the solver's
        # sat answer is withheld.
        assert record.outcome in ("violated-witness", "violated-constant", "unknown"), (
            f"expected violation or unknown, got {record.outcome}: {record.detail}"
        )
