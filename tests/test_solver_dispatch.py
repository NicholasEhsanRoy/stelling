# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""Portfolio dispatch, transports, timeouts, disagreement — no real solver.

The external-binary transport is exercised against fake "solvers" (tiny
scripts written to tmp and reached via ``STELLING_CVC5``), which lets every
venv — including the zero-dep one — witness the dispatch contracts: a
timeout degrades to UNKNOWN and is never a VERIFIED; a sat/unsat
disagreement raises :exc:`SolverDisagreement` loudly; a sat whose model
does not replay raises :exc:`EmissionInfidelityError`; garbage output is
quoted and degrades, never becoming a verdict; absence is stamped as
absence with the install hint. The banked never-on-defaults instance
lands here too: the config's time limit is required, and every invocation
stamp carries the exact emitted option set plus the script hash.
"""

from __future__ import annotations

import stat
import sys

import pytest

from stelling import _optional, solvers
from stelling.propagate import propagate
from stelling.solvers import (
    INSTALL_HINT,
    TRANSPORT_CVC5_BINARY,
    EmissionInfidelityError,
    SolverConfig,
    SolverDisagreement,
    _Backend,
    _make_run_cvc5_binary,
    escalate,
    make_solver_verdict,
)
from stelling.verdict import make_verdict
from test_obligation_slice import BOOL, any_eqn, close, eqn, lit, square_query, var

VERSIONS = dict(
    stelling_version="test", jax_version="none: hand-built IR",
    precision_config="jax_enable_x64=True (hand-built f64 IR)",
)


def fake_solver(tmp_path, body: str, name: str, version_line: str | None = "This is cvc5 version 9.9.9-fake") -> str:
    """A fake external 'solver': answers --version, then runs ``body`` with
    the script text bound to ``data``."""
    head = ""
    if version_line is not None:
        head = (
            'if "--version" in sys.argv[1:]:\n'
            f"    print({version_line!r})\n"
            "    sys.exit(0)\n"
        )
    text = (
        f"#!{sys.executable}\n"
        "import sys\n"
        f"{head}"
        "data = sys.stdin.read()\n"
        f"{body}\n"
    )
    path = tmp_path / name
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def escalate_square(monkeypatch, fake_path, timeout_ms=2000):
    monkeypatch.setenv("STELLING_CVC5", fake_path)
    q = square_query()
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown"]
    config = SolverConfig(timeout_ms=timeout_ms, only=("cvc5",))
    return q, p, escalate(q, p, config)


# --- the config's own no-defaults discipline ---------------------------------


def test_time_limit_is_required_no_implicit_default():
    with pytest.raises(TypeError):
        SolverConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        SolverConfig(timeout_ms=0)
    with pytest.raises(ValueError):
        SolverConfig(timeout_ms=-5)
    with pytest.raises(ValueError):
        SolverConfig(timeout_ms=100, only=("mathematica",))


# --- absence: stamped, hinted, UNKNOWN ---------------------------------------


def test_no_solver_installed_unknown_with_absence_stamp(monkeypatch):
    monkeypatch.delenv("STELLING_CVC5", raising=False)
    monkeypatch.setattr(_optional, "available", lambda name: False)
    monkeypatch.setattr(_optional, "cvc5_binary", lambda: None)
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000))
    assert [r.outcome for r in esc.records] == ["unknown"]
    assert esc.invocations == ()
    assert INSTALL_HINT in esc.notes
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.solver.invoked is False  # absence recorded, never implied
    assert v.stamp.solver.reason
    # and the plain interval path is untouched by all of this
    assert make_verdict(q, p, **VERSIONS).status == "UNKNOWN"


# --- the external-binary transport against fakes ------------------------------


def test_fake_unsat_discharges_and_stamps_the_degraded_portfolio(monkeypatch, tmp_path):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    q, p, esc = escalate_square(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "discharged"
    assert len(record.invocations) == 1
    stamp = record.invocations[0]
    assert stamp.invoked is True
    assert stamp.name == "cvc5"
    assert stamp.version == "9.9.9-fake"
    assert stamp.transport == TRANSPORT_CVC5_BINARY
    opts = dict(stamp.options)
    assert opts[":produce-models"] == "true"
    assert opts[":tlimit"] == "2000"
    assert opts[":nl-cov"] == "true" and opts[":nl-ext"] == "none"
    assert "smt2_sha256" in opts and "set-logic" in opts
    assert any("portfolio degraded" in n for n in record.notes)
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 1


def test_fake_sat_with_replaying_witness_is_refuted_with_witness(monkeypatch, tmp_path):
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\nprint("  (define-fun x0 () Real (/ 3 2))")\nprint(")")',
        "cvc5-sat",
    )
    q, p, esc = escalate_square(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    assert record.witness is not None
    assert record.witness.values == (("x0", "3/2"),)
    assert "external-binary" in record.witness.produced_by
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    rendered = v.render()
    assert "x0 = 3/2" in rendered
    assert "replay" in rendered
    assert "Not a witness" not in rendered  # set-level text is interval-only


def test_fake_sat_with_nonviolating_model_raises_emission_infidelity(monkeypatch, tmp_path):
    # x0 = 1 satisfies x0^2 <= 2: a "witness" that does not witness. The
    # replay must catch it loudly — never a REFUTED, never a silent skip.
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\nprint("  (define-fun x0 () Real 1.0)")\nprint(")")',
        "cvc5-badsat",
    )
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = square_query()
    p = propagate(q)
    with pytest.raises(EmissionInfidelityError) as info:
        escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    assert info.value.obligation_index == 0
    assert ("x0", "1") in info.value.values
    assert "(check-sat)" in info.value.script_text


def test_fake_garbage_output_degrades_to_unknown_with_the_garbage_quoted(monkeypatch, tmp_path):
    fake = fake_solver(
        tmp_path, 'print("flarb the snozzberries")', "cvc5-garbage", version_line=None
    )
    q, p, esc = escalate_square(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "unknown"  # a parse failure is never a verdict
    assert any("flarb the snozzberries" in n for n in record.notes)
    assert len(record.invocations) == 1  # the invocation happened; stamped
    assert record.invocations[0].invoked is True
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"


def test_fake_sleeper_hits_the_wall_clock_guard_unknown_never_verified(monkeypatch, tmp_path):
    fake = fake_solver(tmp_path, "import time\ntime.sleep(3600)", "cvc5-sleep")
    q, p, esc = escalate_square(monkeypatch, fake, timeout_ms=200)
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert "never a VERIFIED" in record.detail
    (stamp,) = record.invocations
    # the stamp records the ask, appended before any result existed: the
    # reason carries invocation context only; the outcome (timeout) lives
    # in the notes, where results are recorded after the run
    assert stamp.invoked is True
    assert "assert #0" in stamp.reason and "timeout" not in stamp.reason
    assert any("timeout" in n for n in record.notes)
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"


def test_nonrational_model_stays_unknown_by_policy(monkeypatch, tmp_path):
    fake = fake_solver(
        tmp_path,
        'print("sat")\nprint("(")\n'
        'print("  (define-fun x0 () Real (_ real_algebraic_number <1*x^2 + (-2), (5/4, 3/2)>))")\n'
        'print(")")',
        "cvc5-algebraic",
    )
    q, p, esc = escalate_square(monkeypatch, fake)
    (record,) = esc.records
    assert record.outcome == "unknown"
    assert "not independently replayable" in record.detail
    assert make_solver_verdict(q, p, esc, **VERSIONS).status == "UNKNOWN"


# --- portfolio combination rules ----------------------------------------------


def two_fake_backends(tmp_path, body_a: str, body_b: str):
    path_a = fake_solver(tmp_path, body_a, "member-a")
    path_b = fake_solver(tmp_path, body_b, "member-b")
    a = _Backend(
        name="cvc5", flavor="cvc5", label="cvc5 (fake a)",
        transport=TRANSPORT_CVC5_BINARY,
        transport_fn=_make_run_cvc5_binary(path_a),
        version_fn=lambda: "9.9.9-fake",
    )
    b = _Backend(
        name="z3", flavor="z3", label="z3 (fake b)",
        transport=TRANSPORT_CVC5_BINARY,
        transport_fn=_make_run_cvc5_binary(path_b),
        version_fn=lambda: "9.9.9-fake",
    )
    return (a, b)


def test_disagreement_raises_loud_with_both_verdicts_options_and_scripts(
    monkeypatch, tmp_path
):
    backends = two_fake_backends(tmp_path, 'print("unsat")', 'print("sat")')
    monkeypatch.setattr(solvers, "_backends_for", lambda config: (backends, ()))
    q = square_query()
    p = propagate(q)
    with pytest.raises(SolverDisagreement) as info:
        escalate(q, p, SolverConfig(timeout_ms=2000))
    err = info.value
    assert sorted(answer for _, answer in err.verdicts) == ["sat", "unsat"]
    assert len(err.options) == 2 and all(opts for _, opts in err.options)
    assert len(err.scripts) == 2
    assert all("(check-sat)" in text for _, text in err.scripts)


def test_one_definitive_plus_one_unknown_definitive_stands(monkeypatch, tmp_path):
    backends = two_fake_backends(tmp_path, 'print("unsat")', 'print("unknown")')
    monkeypatch.setattr(solvers, "_backends_for", lambda config: (backends, ()))
    q = square_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000))
    (record,) = esc.records
    assert record.outcome == "discharged"
    assert len(record.invocations) == 2  # both invocations stamped
    assert any("returned unknown" in n for n in record.notes)
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple) and len(v.stamp.solver) == 2


def test_both_sat_agreement_uses_primary_witness(monkeypatch, tmp_path):
    model = 'print("sat")\nprint("(")\nprint("  (define-fun x0 () Real {})")\nprint(")")'
    backends = two_fake_backends(
        tmp_path, model.format("(/ 3 2)"), model.format("(/ 7 4)")
    )
    monkeypatch.setattr(solvers, "_backends_for", lambda config: (backends, ()))
    q = square_query()  # QF_NRA: the cvc5-flavored member is primary
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000))
    (record,) = esc.records
    assert record.outcome == "violated-witness"
    assert record.witness.values == (("x0", "3/2"),)  # the primary's model


# --- mixed obligations and rendering honesty ---------------------------------


def mixed_query():
    x, sq, p1, o1, p2, o2 = var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
    return close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], p1),  # straddles: unknown
            eqn("stelling_assert", [p1], o1),
            eqn("ge", [x, lit(0.0)], p2),  # definitely true
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )


def test_interval_decisions_are_not_redecided(monkeypatch, tmp_path):
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    q = mixed_query()
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    assert len(esc.records) == 1 and esc.records[0].index == 0
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.obligations[0].status == "discharged"
    assert "solver" in v.obligations[0].detail
    assert v.obligations[1].status == "discharged"
    assert "solver" not in v.obligations[1].detail  # interval's own discharge


def test_interval_refutation_keeps_set_level_wording(monkeypatch, tmp_path):
    # one obligation violated by interval, one unknown discharged by the
    # fake: verdict REFUTED, and the set-level "not a witness" wording
    # stays — it belongs to the interval refutation that is present.
    x, sq, p1, o1, p2, o2 = var(0), var(1), var(2, BOOL), var(3, BOOL), var(4, BOOL), var(5, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("mul", [x, x], sq),
            eqn("le", [sq, lit(2.0)], p1),  # unknown
            eqn("stelling_assert", [p1], o1),
            eqn("lt", [x, lit(0.0)], p2),  # definitely false: violated
            eqn("stelling_assert", [p2], o2),
        ],
        [o1, o2],
    )
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    p = propagate(q)
    assert [o.status for o in p.obligations] == ["unknown", "violated-over-set"]
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "REFUTED"
    assert "Not a witness" in v.render()  # the set-level refutation is real


def test_declined_obligation_travels_into_verdict_notes(monkeypatch, tmp_path):
    # an unsupported slice must reach the verdict as UNKNOWN with the
    # primitive quoted, even with a solver installed and eager
    fake = fake_solver(tmp_path, 'print("unsat")', "cvc5-unsat")
    monkeypatch.setenv("STELLING_CVC5", fake)
    x, e, pred, out = var(0), var(1), var(2, BOOL), var(3, BOOL)
    q = close(
        [
            any_eqn(x, 1.0, 2.0),
            eqn("exp", [x], e),
            eqn("lt", [e, lit(7.0)], pred),
            eqn("stelling_assert", [pred], out),
        ],
        [out],
    )
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=2000, only=("cvc5",)))
    (record,) = esc.records
    assert record.outcome == "unknown" and record.invocations == ()
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    assert v.stamp.solver.invoked is False  # no invocation to stamp
    assert any("'exp'" in n for n in v.notes)


def test_escalation_with_nothing_unknown_is_a_noop_with_absence_stamp(monkeypatch, tmp_path):
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
    p = propagate(q)
    esc = escalate(q, p, SolverConfig(timeout_ms=1000, only=("cvc5",)))
    assert esc.records == ()
    v = make_solver_verdict(q, p, esc, **VERSIONS)
    assert v.status == "VERIFIED"
    assert v.stamp.solver.invoked is False
    assert "interval" in v.stamp.solver.reason
