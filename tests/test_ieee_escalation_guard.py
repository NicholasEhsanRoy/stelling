# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Guard 2: ieee-mode propagation refuses solver escalation — no real solver.

The SMT backends emit over the reals (QF_LRA/QF_NRA): escalating a
float-semantics obligation would prove the ℝ obligation under an
ieee-stamped claim. The invariant is enforced twice, anti-correlated
(SOUNDNESS.md: one invariant, two mechanisms): :func:`escalate` declines
every unknown obligation of an ieee propagation with the reason quoted
(zero invocations, absence derived from the empty ledger), and
:func:`make_solver_verdict` independently raises
:exc:`MispairedEscalationError` when an ieee propagation is paired with
an escalation carrying solver work (the mode-mixed caller bypass). Fake
external "solvers" via ``STELLING_CVC5`` make this runnable in every
venv, including zero-dep — the same instrument as
tests/test_solver_dispatch.py.
"""

from __future__ import annotations

import pytest

from stelling.propagate import propagate
from stelling.solvers import (
    IEEE_SEMANTICS_REFUSAL,
    MispairedEscalationError,
    SolverConfig,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import SEMANTICS_IEEE, SEMANTICS_REAL
from test_obligation_slice import square_query
from test_solver_dispatch import VERSIONS, fake_solver


def test_ieee_escalation_declines_every_unknown_with_zero_invocations(
    monkeypatch, tmp_path
):
    # a fake solver that WOULD answer unsat is installed and eager: if the
    # refusal failed to fire, the obligation would come back discharged —
    # an ℝ proof under an ieee claim.
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p = propagate(q, semantics="ieee")
    assert [o.status for o in p.obligations] == ["unknown"]
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "unknown"  # never discharged: nothing ran
    assert record.invocations == ()
    assert IEEE_SEMANTICS_REFUSAL in record.detail  # the reason, quoted
    assert esc.notes == (IEEE_SEMANTICS_REFUSAL,)
    # absence is DERIVED: the ledger never saw an append, never a spawn
    assert esc.ledger.stamps == [] and esc.ledger.spawns == 0


def test_ieee_honest_pairing_emits_unknown_with_the_ieee_stamp(
    monkeypatch, tmp_path
):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p = propagate(q, semantics="ieee")
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.semantics == SEMANTICS_IEEE
    assert v.stamp.solver.invoked is False  # absence from the empty ledger
    # the real convention line must not ride in the ieee stamp
    assert not any("0*inf = 0" in a for a in v.stamp.assumptions)
    assert any("native binary64" in a for a in v.stamp.assumptions)
    assert IEEE_SEMANTICS_REFUSAL in v.obligations[0].detail


def test_mode_mixed_pairing_raises_instead_of_stamping_an_r_proof(
    monkeypatch, tmp_path
):
    # the bypass the second mechanism exists for: escalate on the REAL
    # propagation (real invocation happens, fake unsat "discharges"),
    # then assemble the verdict against the IEEE propagation — before the
    # gate this would mint a VERIFIED-by-solver under an ieee stamp.
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p_real = propagate(q)  # semantics="real"
    p_ieee = propagate(q, semantics="ieee")
    cfg = SolverConfig(timeout_ms=2000, only=("cvc5",))
    esc = escalate(q, p_real, cfg)  # real solver work happens
    (record,) = esc.records
    assert record.outcome == "discharged" and record.invocations
    with pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(q, p_ieee, esc, **VERSIONS)
    msg = str(exc.value)
    assert "mispaired escalation" in msg
    assert "semantics='ieee'" in msg
    assert "refusing to emit" in msg
    # the honest pairings still emit:
    # (1) the real propagation with the escalation it produced
    v = make_solver_verdict(q, p_real, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.semantics == SEMANTICS_REAL
    # (2) the ieee propagation with its own refusal-shaped escalation
    esc_refused = escalate(q, p_ieee, cfg)
    v2 = make_solver_verdict(q, p_ieee, esc_refused, **VERSIONS)
    assert v2.status == "UNKNOWN" and v2.stamp.semantics == SEMANTICS_IEEE


def test_witness_carrying_escalation_also_refused_against_ieee(
    monkeypatch, tmp_path
):
    # same bypass, sat direction: a replayed witness refutes the ℝ claim;
    # under ieee the predicate at that point may hold in floats — the
    # witness must never be stamped against the ieee propagation.
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\nprint("  (define-fun x0 () Real (/ 3 2))")\nprint(")")',
        "cvc5-sat",
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p_real = propagate(q)
    p_ieee = propagate(q, semantics="ieee")
    cfg = SolverConfig(timeout_ms=2000, only=("cvc5",))
    esc = escalate(q, p_real, cfg)
    (record,) = esc.records
    assert record.witness is not None
    with pytest.raises(MispairedEscalationError):
        make_solver_verdict(q, p_ieee, esc, **VERSIONS)


def test_real_mode_escalation_is_untouched(monkeypatch, tmp_path):
    # §-integrity: the real path never mentions the ieee refusal
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "discharged"
    assert IEEE_SEMANTICS_REFUSAL not in esc.notes
    assert all(IEEE_SEMANTICS_REFUSAL not in r.detail for r in esc.records)
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.semantics == SEMANTICS_REAL
    assert any("0*inf = 0" in a for a in v.stamp.assumptions)


def test_ieee_with_nothing_unknown_never_reaches_the_refusal(
    monkeypatch, tmp_path
):
    # an ieee propagation that decided everything has nothing to escalate:
    # an empty escalation, and the assembled verdict stamps ieee
    from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, var

    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q, semantics="ieee")
    esc = escalate(q, p, SolverConfig(timeout_ms=1000, only=("cvc5",)))
    assert esc.records == ()
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.semantics == SEMANTICS_IEEE
    assert v.stamp.solver.invoked is False


# --- float-fidelity audit C1: the absence reason names the arithmetic ---------


def test_ieee_absence_reason_names_native_binary64(monkeypatch, tmp_path):
    from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, var

    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    x, pred, out = var(0), var(1, BOOL), var(2, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("ge", [x, lit(0.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    cfg = SolverConfig(timeout_ms=1000, only=("cvc5",))
    # ieee: the arithmetic that judged it was native binary64
    p_ieee = propagate(q, semantics="ieee")
    v = make_solver_verdict(q, p_ieee, escalate(q, p_ieee, cfg), **VERSIONS)
    assert "outward-rounded" not in v.stamp.solver.reason
    assert "native-binary64" in v.stamp.solver.reason
    # real: byte-identical pre-existing wording
    p_real = propagate(q)
    vr = make_solver_verdict(q, p_real, escalate(q, p_real, cfg), **VERSIONS)
    assert vr.stamp.solver.reason == (
        "no solver invoked: every obligation was decided by outward-rounded "
        "interval arithmetic alone; escalation had nothing to do"
    )


# --- float-fidelity audit C2: the symmetric semantics-pairing gate ------------


def test_escalation_records_the_semantics_it_was_produced_from(
    monkeypatch, tmp_path
):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    cfg = SolverConfig(timeout_ms=2000, only=("cvc5",))
    assert escalate(q, propagate(q), cfg).semantics == "real"
    assert (
        escalate(q, propagate(q, semantics="ieee"), cfg).semantics == "ieee"
    )


def test_reverse_mode_mix_raises_instead_of_misattributing(
    monkeypatch, tmp_path
):
    # the auditor's c7 (f) probe: a REAL propagation paired with the
    # refusal-shaped escalation produced from the ieee run of the same
    # query. Statuses would stay UNKNOWN (sound), but the obligation
    # details would quote the ieee refusal under a real stamp — the
    # symmetric gate refuses to emit instead.
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p_real = propagate(q)
    p_ieee = propagate(q, semantics="ieee")
    cfg = SolverConfig(timeout_ms=2000, only=("cvc5",))
    esc_ieee = escalate(q, p_ieee, cfg)  # refusal-shaped, semantics="ieee"
    with pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(q, p_real, esc_ieee, **VERSIONS)
    msg = str(exc.value)
    assert "semantics='real'" in msg and "semantics='ieee'" in msg
    assert "misattributed" in msg and "refusing to emit" in msg
    # honest pairings unaffected, both directions
    v = make_solver_verdict(q, p_ieee, esc_ieee, **VERSIONS)
    assert v.status == "UNKNOWN" and v.stamp.semantics == SEMANTICS_IEEE
    esc_real = escalate(q, p_real, cfg)
    vr = make_solver_verdict(q, p_real, esc_real, **VERSIONS)
    assert vr.stamp.semantics == SEMANTICS_REAL
