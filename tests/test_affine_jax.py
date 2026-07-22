# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The affine refinement end-to-end: traced harnesses through check().

The pipeline opt-in (``refine="affine"``, never on by default), the
stamp truth (refinement line, mechanism notes, the reworded solver-
absence line), the solver fall-through (escalation sees only what
affine left undecided), and the widen re-check parity (refined iff the
original refined). Skipped without jax; solver cases skip per
availability, exactly as ``test_solver_acceptance.py``."""

from __future__ import annotations

from fractions import Fraction

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

import stelling.affine  # noqa: E402
from stelling import _optional  # noqa: E402
from stelling.harness import any_array, assert_, trace  # noqa: E402
from stelling.affine import DISCHARGED_BY_AFFINE  # noqa: E402
from stelling.preconditions import check  # noqa: E402

HAVE_Z3 = _optional.available("z3")
HAVE_CVC5 = _optional.available("cvc5") or _optional.cvc5_binary() is not None

need_both = pytest.mark.skipif(
    not (HAVE_Z3 and HAVE_CVC5), reason="needs both z3 and cvc5"
)


@pytest.fixture(autouse=True, scope="module")
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def h_commuted():
    # KA 3, traced: the commuted-product pair
    x = any_array((), "float64", (0.1, 0.3))
    y = any_array((), "float64", (0.1, 0.3))
    p = x * y
    q = y * x
    return assert_(p - q >= 0.0), assert_(q - p >= 0.0)


def h_product():
    # KA 4, traced: a true straddle affine must NOT close
    x = any_array((), "float64", (-1.0, 1.0))
    y = any_array((), "float64", (-1.0, 1.0))
    return assert_(x * y >= 0.0)


def h_conditioning():
    # KA 5, traced: the probe's conditioning shape (quadratic past
    # plain affine; a QF_NRA validity)
    a = any_array((), "float64", (1.0, 2.0))
    c = any_array((), "float64", (1.0, 2.0))
    b = any_array((), "float64", (-0.5, 0.5))
    tr = a + c
    det = a * c - b * b
    return assert_(tr * tr <= det * 10.125)  # 10.125 = 81/8 exactly


def h_rolling():
    # KA 6, traced: rolling-average mass identity over shared inputs
    a = any_array((4,), "float64", (0.0, 1.0))
    rolled = jnp.concatenate([a[3:4], a[0:3]])
    avg = 0.5 * (a + rolled)
    return assert_(jnp.sum(avg) - jnp.sum(a) >= 0.0), assert_(
        jnp.sum(a) - jnp.sum(avg) >= 0.0
    )


def h_div():
    # KA 7, traced: a slice containing div declines wholly
    x = any_array((), "float64", (0.0, 1.0))
    y = any_array((), "float64", (1.0, 2.0))
    d = x / y
    return assert_(d - d >= 0.0)


def h_refute():
    # the affine set-level refutation: (x - x) - 1 is exactly -1
    x = any_array((), "float64", (0.0, 1.0))
    return assert_((x - x) - 1.0 >= 0.0)


# --- opt-in and validation ---------------------------------------------------


def test_refine_default_never_runs_and_is_byte_identical(monkeypatch):
    calls = []
    orig = stelling.affine.refine_propagation

    def counting(closed, prop):
        calls.append(1)
        return orig(closed, prop)

    monkeypatch.setattr(stelling.affine, "refine_propagation", counting)
    v_default = check(h_commuted, vacuity_mode="inputs-only")
    assert calls == []  # never on by default
    v_none = check(h_commuted, vacuity_mode="inputs-only", refine=None)
    assert calls == []
    assert v_default.render() == v_none.render()
    assert v_default.status == "UNKNOWN"
    # the old absence wording is untouched when the refinement is off
    assert v_default.stamp.solver.reason == (
        "no solver invoked: every obligation was judged by outward-rounded "
        "interval arithmetic alone"
    )
    assert not any("affine" in a for a in v_default.stamp.assumptions)


def test_unknown_refine_value_raises_eagerly_before_tracing():
    def exploding_harness():  # tracing this would raise something else
        raise AssertionError("harness must not be traced")

    with pytest.raises(ValueError, match="refine must be None or 'affine'"):
        check(exploding_harness, vacuity_mode="inputs-only", refine="zonotope")


# --- the known answers through the pipeline ----------------------------------


def test_traced_commuted_products_verify_with_the_mechanism_named():
    v = check(h_commuted, vacuity_mode="inputs-only", refine="affine")
    assert v.status == "VERIFIED"
    for o in v.obligations:
        assert o.status == "discharged"
        assert o.detail == DISCHARGED_BY_AFFINE
    assert v.stamp.solver.reason == (
        "no solver invoked: every obligation was judged by outward-rounded "
        "interval arithmetic with affine (zonotope) refinement — 2 "
        "obligation(s) decided by the affine domain"
    )
    aff_lines = [
        a
        for a in v.stamp.assumptions
        if a.startswith("affine refinement enabled")
    ]
    assert len(aff_lines) == 1
    assert "2 discharged" in aff_lines[0]
    assert "ops used:" in aff_lines[0]
    # the arithmetic line names the deciding abstraction (audit F4)
    assert v.stamp.arithmetic_mode.endswith(
        "+ affine/zonotope refinement (stelling.affine, same outward kernel)"
    )


def test_traced_rolling_average_verifies():
    v = check(h_rolling, vacuity_mode="inputs-only", refine="affine")
    assert v.status == "VERIFIED"
    assert [o.status for o in v.obligations] == ["discharged", "discharged"]


def test_traced_refutation_via_affine_with_solver_budget_offered():
    """Affine decides everything (a definite set-level refutation), so a
    supplied solver budget must be recorded as not needed, truthfully."""
    v = check(
        h_refute,
        vacuity_mode="inputs-only",
        refine="affine",
        solver_timeout_ms=30_000,
    )
    assert v.status == "REFUTED"
    assert v.stamp.solver.reason == (
        "no solver invoked: every obligation was decided by outward-rounded "
        "interval arithmetic with affine (zonotope) refinement — 1 "
        "obligation(s) decided by the affine domain; escalation had nothing "
        "to do"
    )
    assert "Not a witness" in v.render()  # the set-level refutation class


def test_traced_ka5_honest_unknown_without_solver():
    v = check(h_conditioning, vacuity_mode="inputs-only", refine="affine")
    assert v.status == "UNKNOWN"
    assert [o.status for o in v.obligations] == ["unknown"]
    assert any(
        "affine refinement ran and did not separate" in n for n in v.notes
    )
    assert "decided nothing: 1 obligation(s) attempted" in (
        v.stamp.solver.reason
    )


def test_traced_div_declines_and_obligations_match_refine_none():
    v_off = check(h_div, vacuity_mode="inputs-only")
    v_on = check(h_div, vacuity_mode="inputs-only", refine="affine")
    assert v_on.status == v_off.status == "UNKNOWN"
    assert v_on.obligations == v_off.obligations  # byte-identical outcome
    assert any(
        "affine refinement declined: primitive 'div' is outside "
        "AFFINE_SUPPORTED" in n
        for n in v_on.notes
    )
    assert not any("affine" in n for n in v_off.notes)


# --- solver fall-through -----------------------------------------------------


@need_both
def test_ka4_full_fall_through_affine_undecided_solver_refutes():
    """The full chain: interval straddles, affine runs and does not
    separate, the solver then REFUTES with a replay-confirmed witness."""
    v = check(
        h_product,
        vacuity_mode="inputs-only",
        refine="affine",
        solver_timeout_ms=30_000,
    )
    assert v.status == "REFUTED"
    assert len(v.witnesses) == 1
    w = v.witnesses[0]
    vals = dict(w.values)
    x, y = Fraction(vals["x0"]), Fraction(vals["x1"])
    assert x * y < 0  # independently recomputed violation
    assert -1 <= x <= 1 and -1 <= y <= 1
    assert any(
        "affine refinement ran and did not separate" in n for n in v.notes
    )
    # real invocations: the solver field carries stamps, and the
    # refinement line still records the attempt
    assert isinstance(v.stamp.solver, tuple)
    assert any(
        a.startswith("affine refinement enabled") and "0 discharged" in a
        for a in v.stamp.assumptions
    )


@need_both
def test_ka5_full_fall_through_affine_undecided_solver_verifies():
    v = check(
        h_conditioning,
        vacuity_mode="inputs-only",
        refine="affine",
        solver_timeout_ms=30_000,
    )
    assert v.status == "VERIFIED"
    assert isinstance(v.stamp.solver, tuple)  # the QF_NRA portfolio ran
    assert any(
        a.startswith("affine refinement enabled")
        for a in v.stamp.assumptions
    )


# --- widen re-check parity ---------------------------------------------------


def test_widen_recheck_runs_at_the_same_refinement_depth(monkeypatch):
    calls = []
    orig = stelling.affine.refine_propagation

    def counting(closed, prop):
        calls.append(closed)
        return orig(closed, prop)

    monkeypatch.setattr(stelling.affine, "refine_propagation", counting)
    # VERIFIED via affine: the widen re-check must run, and must refine
    # (the widened declarations then decline honestly — non-finite
    # bounds are outside the domain). The stamp states the MEASUREMENT
    # (audit F1) and discloses that the re-check ran weaker than the
    # original (interval-only here: affine declined the unbounded boxes
    # and no solver budget was offered).
    v = check(h_commuted, vacuity_mode="inputs-only", refine="affine")
    assert v.status == "VERIFIED"
    assert len(calls) == 2  # the original run AND the widen re-check
    vac = [a for a in v.stamp.assumptions if "vacuity checked" in a]
    assert vac == [
        "vacuity checked (mode=inputs-only): no obligation discharges "
        "with the declared bounds widened — under the mechanism(s) that "
        "ran, this VERIFIED was not re-derivable without the declared "
        "envelope. The re-check ran weaker than the original "
        "(interval-only: the affine refinement declines unbounded boxes "
        "by construction), so envelope-independence of the affine-decided "
        "obligation(s) was not measured; an explicit solver_timeout_ms "
        "measures it."
    ]
    assert "load-bearing" not in vac[0]  # the retired inference
    # UNKNOWN: no widen re-check, exactly one refinement
    calls.clear()
    v2 = check(h_conditioning, vacuity_mode="inputs-only", refine="affine")
    assert v2.status == "UNKNOWN"
    assert len(calls) == 1


def test_refine_off_interval_top_case_gets_the_measured_wording():
    """Audit F1, the pre-existing half: x*x >= 0 over [1, 2] is a range
    theorem, but the widened re-check's interval mul is ⊤ and cannot
    re-derive it — the old stamp claimed the envelope load-bearing,
    which is false; the new line states only the measurement. No affine
    involved: this is the deliberate refine-off wording change."""
    def h():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x * x >= 0.0)

    v = check(h, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    vac = [a for a in v.stamp.assumptions if "vacuity checked" in a]
    assert vac == [
        "vacuity checked (mode=inputs-only): no obligation discharges "
        "with the declared bounds widened — under the mechanism(s) that "
        "ran, this VERIFIED was not re-derivable without the declared "
        "envelope"
    ]
    # no affine participated: no reduced-power sentence rides
    assert "ran weaker" not in vac[0]


def test_contract_checker_passes_refine_through():
    from stelling import contracts

    c = contracts.conditioning_2x2(
        "float64", (1.0, 2.0), (1.0, 2.0), (-0.5, 0.5), 8.0
    )
    cv = contracts.check_contract(
        c, vacuity_mode="inputs-only", refine="affine"
    )
    assert cv.requires.status == "UNKNOWN"
    assert any(
        "affine refinement" in n for n in cv.requires.notes
    )
    assert any(
        a.startswith("affine refinement enabled")
        for a in cv.requires.stamp.assumptions
    )
    with pytest.raises(ValueError, match="refine must be None or 'affine'"):
        contracts.check_contract(c, vacuity_mode="inputs-only", refine="aff")
