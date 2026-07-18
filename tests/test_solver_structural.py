# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Can't-drift tests for the structuralized audit invariants — no real solver.

Three invariants the escalation layer now enforces by structure, each with
the test that keeps the structure from drifting back into separable checks:

1. **The witness conjunction is one function.** A REFUTED-with-witness
   means "∃w: in_box(w) ∧ violates(w)"; both conjuncts are computed only
   by :func:`stelling.obligation.witness_is_valid`, and the dispatch
   path's only ``Witness(`` construction site is the factory
   (:func:`stelling.solvers.make_validated_witness`). AST-checked.
2. **The stamp is append-only, recorded at invocation.** Dispatch appends
   the fully-populated stamp BEFORE the transport runs, so a failure after
   the append leaves the stamp standing; the reason carries invocation
   context, never the answer; absence is derived from the empty ledger at
   a single point, never written by a degradation branch.
3. **The spawn counter is a runtime invariant.** The transport-entry
   boundary counts spawns; the dispatch site appends stamps; no shared
   helper updates both (AST-checked), and every escalated verdict is
   gated on their agreement — divergence in either direction raises
   :exc:`ProvenanceError` instead of emitting.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from stelling import _optional, solvers
from stelling.propagate import propagate
from stelling.solvers import (
    ProvenanceError,
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from test_obligation_slice import square_query
from test_solver_dispatch import (
    VERSIONS,
    escalate_square,
    fake_solver,
    two_fake_backends,
)
from test_verdict import invoked_stamp

SRC = pathlib.Path(solvers.__file__).resolve().parent
SOLVERS_TREE = ast.parse((SRC / "solvers.py").read_text(encoding="utf-8"))
OBLIGATION_TREE = ast.parse((SRC / "obligation.py").read_text(encoding="utf-8"))


def functions_calling(tree: ast.Module, callee: str) -> set[str]:
    """Names of the function defs whose bodies call ``callee`` (by bare
    name or as an attribute); ``<module>`` for module-level calls."""
    found: set[str] = set()
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Call(self, node: ast.Call) -> None:
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == callee:
                found.add(stack[-1] if stack else "<module>")
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def all_names(tree: ast.Module) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


# --- Change 1: the witness conjunction is structurally one --------------------


def test_witness_is_constructed_in_exactly_one_function():
    # the dispatch path's only Witness( construction site is the factory
    assert functions_calling(SOLVERS_TREE, "Witness") == {
        "make_validated_witness"
    }


def test_folded_conjunct_helpers_no_longer_exist_separately():
    # the membership helper was folded into the validator: gone from
    # solvers, absent everywhere as a separate module-level function
    assert not hasattr(solvers, "_box_escape")
    assert "_box_escape" not in all_names(SOLVERS_TREE)
    assert "_box_escape" not in all_names(OBLIGATION_TREE)
    # dispatch code never calls the replay engine directly for witness
    # acceptance: no call (or reference) anywhere in solvers.py …
    assert functions_calling(SOLVERS_TREE, "evaluate_predicate") == set()
    assert "evaluate_predicate" not in all_names(SOLVERS_TREE)
    # … and inside obligation.py the engine is called only by the validator
    assert functions_calling(OBLIGATION_TREE, "evaluate_predicate") == {
        "witness_is_valid"
    }


def test_both_refutation_shapes_route_through_the_one_validator():
    # the validator is reached only via the single refutation gate, and the
    # gate is used by exactly the factory (witness shape) and the
    # constants-only branch of dispatch — one code path decides
    # "the refutation is real" for both shapes
    assert functions_calling(SOLVERS_TREE, "witness_is_valid") == {
        "_require_valid_refutation"
    }
    assert functions_calling(SOLVERS_TREE, "_require_valid_refutation") == {
        "make_validated_witness",
        "_dispatch_obligation",
    }


# --- Change 2: the stamp is append-only, recorded at invocation ---------------


def test_stamp_reason_carries_context_never_the_answer(monkeypatch, tmp_path):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    (record,) = esc.records
    (stamp,) = record.invocations
    # the reason names the invocation: fragment, portfolio role, obligation
    assert "QF_NRA" in stamp.reason
    assert "assert #0" in stamp.reason
    # … and cannot know the answer: the stamp predates the result
    assert "answered" not in stamp.reason
    assert "unsat" not in stamp.reason
    assert "ms" not in stamp.reason
    # outcome and latency are recorded additively in the notes, after the run
    assert any("answered unsat in" in n and "ms" in n for n in record.notes)


def test_transport_failure_after_append_leaves_the_stamp_standing(
    monkeypatch, tmp_path
):
    # an exec failure (nonexistent binary) happens AFTER the append: the
    # ask was real and fully described, so the stamp stands — invoked=True
    # with the full option set — and the notes quote the failure
    monkeypatch.setenv("STELLING_CVC5", str(tmp_path / "does-not-exist"))
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "unknown"  # never a verdict from a failed run
    (stamp,) = record.invocations
    assert stamp.invoked is True
    opts = dict(stamp.options)
    assert "smt2_sha256" in opts and "set-logic" in opts
    assert any("transport failed" in n for n in record.notes)
    # the invocation was counted at the transport-entry boundary too
    assert esc.ledger.spawns == 1
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    # "one invocation then a failure" stamps one invocation, structurally
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 1


def test_absence_is_derived_from_the_empty_ledger_never_written(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(_optional, "available", lambda name: False)
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: None)
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000))
    assert esc.ledger.stamps == [] and esc.ledger.spawns == 0
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.solver.invoked is False
    assert "no solver invoked" in v.stamp.solver.reason


def test_solver_absent_is_called_only_at_the_single_derivation_point():
    # absence reasons are derived where the empty ledger is read — exactly
    # one call site in solvers.py, inside the verdict assembly
    assert functions_calling(SOLVERS_TREE, "solver_absent") == {
        "make_solver_verdict"
    }


def test_stamp_append_and_spawn_count_have_no_shared_updater():
    # Change-3 anti-correlation, pinned structurally: the function that
    # appends stamps and the function that counts spawns are disjoint
    appenders: set[str] = set()
    counters: set[str] = set()
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Call(self, node: ast.Call) -> None:
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "append"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "stamps"
            ):
                appenders.add(stack[-1] if stack else "<module>")
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            t = node.target
            if isinstance(t, ast.Attribute) and t.attr == "spawns":
                counters.add(stack[-1] if stack else "<module>")
            self.generic_visit(node)

    Visitor().visit(SOLVERS_TREE)
    assert appenders == {"_dispatch_obligation"}
    assert counters == {"run"}  # _Backend.run, the transport-entry boundary
    assert not (appenders & counters)


# --- Change 3: the provenance gate, both divergence directions ----------------


def test_spawn_without_increment_raises_provenance_error(monkeypatch, tmp_path):
    # (a) a backend that runs its transport WITHOUT the counter increment:
    # the spawn is recorded by stamp only — the verdict must never emit
    monkeypatch.setattr(
        solvers._Backend,
        "run",
        lambda self, ledger, script_text, wall_s: self.transport_fn(
            script_text, wall_s
        ),
    )
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    assert len(esc.ledger.stamps) == 1 and esc.ledger.spawns == 0
    with pytest.raises(ProvenanceError) as info:
        make_solver_verdict(q, p, esc, **VERSIONS)
    assert info.value.spawns == 0 and info.value.stamped == 1
    assert len(info.value.stamps) == 1
    assert "divergence" in str(info.value)


def test_append_without_transport_call_raises_provenance_error(
    monkeypatch, tmp_path
):
    # (b) an extra append with no transport call behind it — the ledger
    # says two asks, the boundary counted one: no verdict
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    assert len(esc.ledger.stamps) == 1 and esc.ledger.spawns == 1
    esc.ledger.stamps.append(invoked_stamp("cvc5"))
    with pytest.raises(ProvenanceError) as info:
        make_solver_verdict(q, p, esc, **VERSIONS)
    assert info.value.spawns == 1 and info.value.stamped == 2


def test_double_spawn_count_raises_provenance_error(monkeypatch, tmp_path):
    # (a'), the other direction: the boundary counted more spawns than
    # stamps were appended
    real_run = solvers._Backend.run

    def double_count(self, ledger, script_text, wall_s):
        ledger.spawns += 1  # a phantom extra spawn
        return real_run(self, ledger, script_text, wall_s)

    monkeypatch.setattr(solvers._Backend, "run", double_count)
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    assert len(esc.ledger.stamps) == 1 and esc.ledger.spawns == 2
    with pytest.raises(ProvenanceError):
        make_solver_verdict(q, p, esc, **VERSIONS)


def test_happy_path_portfolio_of_two_counts_two(monkeypatch, tmp_path):
    backends = two_fake_backends(tmp_path, 'print("unsat")', 'print("unsat")')
    monkeypatch.setattr(solvers, "_backends_for", lambda config: (backends, ()))
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000))
    assert len(esc.ledger.stamps) == 2 and esc.ledger.spawns == 2
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 2


def test_happy_path_degraded_portfolio_counts_one(monkeypatch, tmp_path):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    assert len(esc.ledger.stamps) == 1 and esc.ledger.spawns == 1
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 1


def test_happy_path_no_solver_counts_zero_with_derived_absence(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(_optional, "available", lambda name: False)
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: None)
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000))
    assert len(esc.ledger.stamps) == 0 and esc.ledger.spawns == 0
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.solver.invoked is False
