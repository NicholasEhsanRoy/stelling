# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Every sentence of the constrained-assume refusal is a measured mechanism.

docs/proposed-decline-messages.md #5: the old refusal was rated "10/10 as
an explanation, 0/10 as a next step" — and the practical effect it left
unstated is that adding a true hypothesis removes the solver you had. The
refusal now states the trade and the working form, and each added
sentence is pinned here against the mechanism it describes:

1. "it fired before any solver was looked for" — escalate() returns the
   refusal without ever reaching backend discovery;
2. "This refusal keyed only on the constraining assume being present ...
   removing the assume removes it (escalation is then attempted)" — the
   same query minus the assume escalates, with real invocations;
3. "a bound on a single declared input can be stated in the declaration
   itself ... narrows the same box without disabling escalation" — the
   declaration-stated bound produces the IDENTICAL narrowed box, and
   that query escalates.

Proposal delta (tree moved, recorded in the commit): the proposal's
draft said a single-input bounding precondition "does not disable
escalation" — false in this tree, where a constraining assume is exactly
what fires the refusal; the draft's relational example is DROPPED (not
constrained) here and never reaches this refusal. The intent — name the
trade, name what works — is implemented against the current mechanics.

Message content only: the refusal fires exactly where it fired, with the
same refusal-shaped escalation (zero invocations, empty ledger).
"""

from __future__ import annotations

import pytest

import stelling.solvers as sv
from stelling.propagate import interval_env, propagate
from stelling.solvers import (
    CONSTRAINED_ASSUME_REFUSAL,
    SolverConfig,
    escalate,
)
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, var
from test_solver_dispatch import constrained_square_query, fake_solver


def _declared_bound_query():
    """The declaration-side form of constrained_square_query: the x >= 1.2
    bound stated in the envelope itself, no assume."""
    x, sq, pred, out = var(0), var(3), var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(x, 1.2, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )


def test_refusal_fires_before_any_solver_is_looked_for(monkeypatch):
    # the sentence's claim, structurally: backend discovery must never be
    # reached on the constrained path — a discovery call fails the test
    def boom(config):
        raise AssertionError("backend discovery reached under a constrained assume")

    monkeypatch.setattr(sv, "_backends_for", boom)
    q = constrained_square_query()
    p = propagate(q)
    assert p.coverage.constrained == 1
    esc = escalate(q, p, SolverConfig(timeout_ms=2000))
    (record,) = esc.records
    assert CONSTRAINED_ASSUME_REFUSAL in record.detail
    assert record.invocations == ()
    assert esc.ledger.spawns == 0


def test_removing_the_assume_removes_the_refusal_and_escalation_runs(
    monkeypatch, tmp_path
):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    config = SolverConfig(timeout_ms=2000, only=("cvc5",))
    # with the assume: the refusal, nothing attempted
    qa = constrained_square_query()
    pa = propagate(qa)
    esc_a = escalate(qa, pa, config)
    assert CONSTRAINED_ASSUME_REFUSAL in esc_a.records[0].detail
    assert esc_a.ledger.spawns == 0
    # without the assume (the declared box, wide): escalation is ATTEMPTED
    # — a real invocation happens — and the refusal appears nowhere
    x, sq, pred, out = var(0), var(3), var(4, BOOL), var(5, BOOL)
    qb = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    pb = propagate(qb)
    assert pb.coverage.constrained == 0
    assert pb.obligations[0].status == "unknown"
    esc_b = escalate(qb, pb, config)
    assert esc_b.ledger.spawns > 0  # attempted, as the text promises
    assert not any(
        CONSTRAINED_ASSUME_REFUSAL in r.detail for r in esc_b.records
    )


def test_declaration_stated_bound_narrows_the_same_box_and_escalates(
    monkeypatch, tmp_path
):
    qa = constrained_square_query()
    qd = _declared_bound_query()
    # "narrows the same box": the constrained env box of x equals the
    # declaration-stated one, endpoint for endpoint
    env_a = interval_env(qa, assume_mode="constrain")
    env_d = interval_env(qd)
    assert (env_a[0].los, env_a[0].his) == (env_d[0].los, env_d[0].his)
    assert env_d[0].los == (1.2,) and env_d[0].his == (2.0,)
    # "without disabling escalation": the declared form is offered to a
    # solver — a real invocation happens, no refusal anywhere
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    pd = propagate(qd)
    assert pd.coverage.constrained == 0
    assert pd.obligations[0].status == "unknown"  # still interval-unknown
    esc = escalate(qd, pd, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    assert esc.ledger.spawns > 0
    assert not any(
        CONSTRAINED_ASSUME_REFUSAL in r.detail for r in esc.records
    )


def test_refusal_text_keeps_the_pinned_prefix_and_names_the_working_form():
    assert CONSTRAINED_ASSUME_REFUSAL.startswith("constrained assume present:")
    assert "fired before any solver was looked for" in CONSTRAINED_ASSUME_REFUSAL
    assert "removing the assume removes it" in CONSTRAINED_ASSUME_REFUSAL
    assert "the envelope passed to any_array" in CONSTRAINED_ASSUME_REFUSAL
    # the proposal's false-in-this-tree sentence must NOT have been copied
    assert "does not disable escalation" not in CONSTRAINED_ASSUME_REFUSAL
