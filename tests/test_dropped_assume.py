# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""A DROPPED assume must not license a refutation — F7's no-op half.

F7 withholds a definite violation when a constraining assume narrowed an
over-approximated variable. It is set INSIDE the narrowing branch, so an assume
that no-ops never reached it; and the solver-side refusal keys on a CONSTRAINED
assume being present, which a dropped one is not. Both protections were
conditioned on the assume having taken effect.

Measured before the fix: `assume(jnp.all(x >= 0))` over `x ∈ [-10, 10]^3`
asserting `sum(x) >= 0` returned REFUTED with the replay-confirmed witness
`[0, 0, -1]` — a counterexample that VIOLATES the dropped precondition.

The disposition is ONE-SIDED, matching F7: a drop means the query ran over a
SUPERSET, so a discharge still implies the intended set's discharge while a
witness may lie entirely outside it.
"""
from __future__ import annotations

import dataclasses

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import propagate as P  # noqa: E402
from stelling import solvers as S  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from _solver_gate import need_solver  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _reproducer():
    """The measured defect, verbatim."""
    x = any_array((3,), "float64", (-10.0, 10.0))
    assume(jnp.all(x >= 0.0))          # reduce_and -> no decidable box
    return (assert_(jnp.sum(x) >= 0.0),)


def _working_idiom():
    """Same meaning, arithmetic form — this one CONSTRAINS."""
    x = any_array((3,), "float64", (-10.0, 10.0))
    assume(jnp.sum(jnp.maximum(0.0 - x, 0.0)) <= 0.0)
    return (assert_(jnp.sum(x) >= 0.0),)


def _verified_under_drop():
    x = any_array((3,), "float64", (0.0, 10.0))
    assume(jnp.all(x >= 0.0))
    return (assert_(jnp.sum(x) >= -1.0),)


def _run(h):
    return check(h, vacuity_mode="inputs-only", solver_timeout_ms=60_000)


def test_the_reproducer_no_longer_refutes():
    v = _run(_reproducer)
    assert v.status != "REFUTED", (
        "a witness minted over a superset of the intended set was presented "
        "as a counterexample to the query the author wrote"
    )
    assert v.witnesses == ()


def test_a_verified_under_a_dropped_assume_is_STILL_RENDERED():
    """One-sided. VERIFIED over a superset implies VERIFIED over the subset,
    so suppressing it would be the over-firing the scatter bar did for its
    entire history."""
    assert _run(_verified_under_drop).status == "VERIFIED"


def test_the_kept_verified_still_discloses_that_the_assume_did_nothing():
    """A user who wrote an assume that no-opped believes it is load-bearing."""
    v = _run(_verified_under_drop)
    assert any("DROPPED" in n for n in v.notes)
    assert any("REFUTED WITNESS MAY VIOLATE" in n for n in v.notes)


def test_the_working_arithmetic_idiom_is_not_broken_by_the_fix():
    """Don't fix the bad path by breaking the good one."""
    v = _run(_working_idiom)
    assert any("CONSTRAINED" in n for n in v.notes)
    assert v.status != "REFUTED"


@pytest.mark.parametrize("semantics", ["real", "ieee"])
def test_both_semantics_modes_mark_and_withhold(semantics):
    """add_any's lesson: real-mode-only is a scope, not a pass."""
    cj = transcribe(jax.make_jaxpr(_reproducer)())
    p = P.propagate(cj, semantics=semantics)
    assert p.assume_dropped
    esc = S.escalate(cj, p, S.SolverConfig(timeout_ms=60_000))
    assert all(r.witness is None for r in esc.records)


@need_solver
def test_the_SOLVER_half_is_load_bearing_on_its_own():
    """0e's split mutation. The interval withhold and the solver withhold are
    separate consumers of one marking, and a test exercising only the interval
    path would pass with half the fix missing.

    Neutralising the marking at the SOLVER boundary alone must bring the
    witness back — otherwise this file is not testing what it claims.
    """
    cj = transcribe(jax.make_jaxpr(_reproducer)())
    p = P.propagate(cj)

    unmarked = dataclasses.replace(p, assume_dropped=False)
    esc_off = S.escalate(cj, unmarked, S.SolverConfig(timeout_ms=60_000))
    assert any(r.witness is not None for r in esc_off.records), (
        "with the solver-side marking off the defect must reappear; if it "
        "does not, the interval half is masking it and this test proves "
        "nothing about the solver half"
    )

    esc_on = S.escalate(cj, p, S.SolverConfig(timeout_ms=60_000))
    assert all(r.witness is None for r in esc_on.records)


def test_a_relational_inert_assume_still_escalates_normally():
    """The documented path the first attempt broke. A relational assume stays
    inert in constrain mode BY DESIGN — the drop over-approximates, so unsat
    over the superset still implies unsat over the subset. Declining
    escalation outright suppressed that too; the suite caught it.
    """
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        u = any_array((), "float64", (0.0, 1.0))
        w = any_array((), "float64", (0.0, 2.0))
        assume(u >= w)                      # relational: no box narrowing
        return (assert_(jnp.sum(x) >= 0.0),)

    v = _run(h)
    # the point is that escalation is not refused outright; the obligation
    # here discharges on intervals, and no drop-refusal note appears
    assert v.status == "VERIFIED"
