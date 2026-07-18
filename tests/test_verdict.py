# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The stamp contract — the banked can't-drift instance, landed.

Two were banked (`design/semantics-drift-probe.md` bookkeeping, SOUNDNESS
commitments): *never invoke a solver on defaults* (lands with the first
solver call — still banked) and *never emit a verdict without a complete
stamp* (lands with the first verdict — this file). Every field of
SOUNDNESS.md's contract must be populated; a missing field fails loudly at
construction rather than defaulting.
"""

from __future__ import annotations

import dataclasses

import pytest

from stelling import ir
from stelling.propagate import ObligationReport, propagate
from stelling.verdict import (
    SolverStamp,
    Stamp,
    StampError,
    Verdict,
    Witness,
    make_verdict,
    solver_absent,
)
from test_propagate import exp_lt_harness


def full_stamp_kwargs():
    return dict(
        stelling_version="0.1.0",
        jax_version="0.11.0",
        query_content_hash="deadbeef",
        arithmetic_mode="interval/f64/outward-1ulp",
        semantics="real (ℝ)",
        precision_config="jax_enable_x64=True",
        device_class="none: no concrete execution in this verdict",
        solver=solver_absent("interval arithmetic discharged everything"),
        nonvacuity="checked — 1 membership condition(s) definitely true",
        transfer_tiers=(("exp", "sound-libm"),),
        transfer_provenance=(("exp", "core"),),
        assumptions=("libm exp faithful",),
        coverage="4 eqns: 4 known (100%)",
    )


def test_every_contract_field_is_required():
    for missing in full_stamp_kwargs():
        kwargs = full_stamp_kwargs()
        del kwargs[missing]
        with pytest.raises(TypeError):
            Stamp(**kwargs)


def test_empty_fields_fail_loudly_instead_of_defaulting():
    for field in (
        "stelling_version",
        "jax_version",
        "query_content_hash",
        "arithmetic_mode",
        "semantics",
        "precision_config",
        "device_class",
        "nonvacuity",
        "coverage",
    ):
        kwargs = full_stamp_kwargs()
        kwargs[field] = ""
        with pytest.raises(StampError):
            Stamp(**kwargs)


def test_solver_absence_is_recorded_not_implied():
    with pytest.raises(StampError):
        solver_absent("")  # absence without a reason is not recorded absence
    from stelling.verdict import SolverStamp

    with pytest.raises(StampError):  # invoked without the emitted option set
        SolverStamp(
            invoked=True, reason="z3 on the residual goal",
            name="z3", version="4.13", transport="wheel", options=None,
        )
    with pytest.raises(StampError):  # absent but carrying solver fields
        SolverStamp(
            invoked=False, reason="not invoked",
            name="z3", version=None, transport=None, options=None,
        )


def test_first_verdict_carries_the_full_stamp():
    closed = exp_lt_harness(8.0)
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "VERIFIED"
    assert v.stamp.query_content_hash == closed.content_hash()
    assert v.stamp.solver.invoked is False and v.stamp.solver.reason
    assert ("exp", "sound-libm") in v.stamp.transfer_tiers
    assert all(origin == "core" for _, origin in v.stamp.transfer_provenance)
    rendered = v.render()
    for needle in (
        "VERIFIED",
        "solver: none",
        "assumes:",
        "coverage:",
        "query ",
        "semantics: real",
        "nonvacuity: UNCHECKED",
    ):
        assert needle in rendered
    # the ℝ-semantics consequence rides as a stamped assumption, always
    assert any("0*inf = 0" in a for a in v.stamp.assumptions)
    # a VERIFIED without checked nonvacuity says it may be vacuous
    assert any("vacuous" in n for n in v.notes)
    # the stamp is frozen: no post-hoc field surgery
    with pytest.raises(dataclasses.FrozenInstanceError):
        object.__delattr__  # placate linters; the real check:
        v.stamp.jax_version = "other"  # type: ignore[misc]


def test_definite_violation_is_refuted_set_level():
    closed = exp_lt_harness(2.0)
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "REFUTED"  # red is a fact, and now it has a verdict
    assert isinstance(v, Verdict)
    assert v.obligations[0].status == "violated-over-set"
    rendered = v.render()
    assert "not invariant as stated" in rendered
    assert "Not a witness" in rendered


def test_straddle_stays_unknown_not_refuted():
    closed = exp_lt_harness(7.0)  # e^2 ≈ 7.389 > 7 > e: undecided, not false
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert v.status == "UNKNOWN"  # our imprecision, never their counterexample


# --- the portfolio extension of the solver field ------------------------------


def invoked_stamp(name="z3", reason="portfolio member: answered unsat"):
    return SolverStamp(
        invoked=True,
        reason=reason,
        name=name,
        version="9.9",
        transport="wheel-bindings (smt2 text)",
        options=((":produce-models", "true"), (":timeout", "1000"),
                 ("set-logic", "QF_LRA"), ("smt2_sha256", "ab" * 32)),
    )


def test_portfolio_stamp_tuple_is_validated_not_defaulted():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = ()  # an empty tuple is not recorded absence
    with pytest.raises(StampError):
        Stamp(**kwargs)
    kwargs["solver"] = (invoked_stamp(), "not a stamp")
    with pytest.raises(StampError):
        Stamp(**kwargs)
    kwargs["solver"] = "z3 said so"  # a string is not a stamp either
    with pytest.raises(StampError):
        Stamp(**kwargs)


def test_portfolio_stamp_tuple_renders_every_invocation():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = (invoked_stamp("cvc5"), invoked_stamp("z3"))
    rendered = Stamp(**kwargs).render()
    assert "cvc5 9.9" in rendered and "z3 9.9" in rendered
    assert "2 invocation(s)" in rendered
    assert "smt2_sha256" in rendered


def test_single_invoked_stamp_still_renders_inline():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = invoked_stamp("z3")
    rendered = Stamp(**kwargs).render()
    assert "solver: z3 9.9 (wheel-bindings (smt2 text))" in rendered


def test_witness_requires_populated_fields():
    with pytest.raises(StampError):
        Witness(obligation_index=0, values=(), produced_by="z3", replay="ok")
    with pytest.raises(StampError):
        Witness(
            obligation_index=0, values=(("x0", "1/2"),), produced_by="", replay="ok"
        )


def test_witness_backed_refuted_renders_witness_not_the_set_level_text():
    kwargs = full_stamp_kwargs()
    kwargs["solver"] = (invoked_stamp("cvc5"),)
    v = Verdict(
        status="REFUTED",
        obligations=(
            ObligationReport(
                index=0,
                status="violated-witness",
                detail="violated at a concrete witness found by cvc5",
                source_info=(),
            ),
        ),
        stamp=Stamp(**kwargs),
        notes=(),
        witnesses=(
            Witness(
                obligation_index=0,
                values=(("x0", "3/2"), ("x1", "-11/8")),
                produced_by="cvc5 9.9 (wheel-bindings (smt2 text))",
                replay="confirmed by exact-rational replay",
            ),
        ),
    )
    rendered = v.render()
    assert "x0 = 3/2" in rendered
    assert "x1 = -11/8" in rendered and "-1.375" in rendered
    assert "produced by: cvc5" in rendered
    assert "replay: confirmed" in rendered
    assert "strictly stronger" in rendered
    assert "Not a witness" not in rendered  # interval-only wording


def test_interval_refuted_render_keeps_the_set_level_text():
    # unchanged contract, now conditional on an actual set-level violation
    closed = exp_lt_harness(2.0)
    v = make_verdict(
        closed,
        propagate(closed),
        stelling_version="0.1.0",
        jax_version="0.11.0",
        precision_config="jax_enable_x64=True",
    )
    assert "Not a witness" in v.render()
