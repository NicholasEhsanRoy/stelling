# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The contract layer: both faces of both templates, the probe-checked
known answers through the public API, the DECLARED-only ensures
invariant, and the counterfactual pins.

Every known answer of corpus/supply/la_contract_probe.py is reproduced
here through stelling's own pipeline (the probe's Z3-checked results are
ground truth). Counterfactual pins state, in comments, exactly which
plausible wrong implementation each would catch — each was measured to
flip the verdict when the wrong implementation is substituted.

This module needs jax (the module-level importorskip below skips the
WHOLE file at collection in a jax-less environment — measured); the
contract layer's genuinely jax-free tests live in
tests/test_contracts_nojax.py so they actually run in both venvs.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from stelling import contracts  # jax-free import must always work
from stelling.contracts import (
    ContractVerdict,
    ENSURES_DECLARED,
    EnsuresFace,
)


jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True, scope="module")
def _x64():
    # module-scoped and restored, following tests/test_preconditions.py
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


from stelling._jax_compat import trace  # noqa: E402
from stelling._optional import available  # noqa: E402
from stelling.contracts import (  # noqa: E402
    check_contract,
    coefficient_contrast,
    conditioning_2x2,
)
from stelling.coverage import sub_jaxprs  # noqa: E402
from stelling.verdict import SolverStamp  # noqa: E402

KAPPA = 8.0
RHS_COEFF = KAPPA + 1.0 / KAPPA + 2.0  # 10.125, exact in binary64

_HAVE_SOLVER = available("z3") or available("cvc5")
needs_solver = pytest.mark.skipif(
    not _HAVE_SOLVER, reason="no SMT solver installed"
)


def _t1(a_range, c_range, b_range, kappa=KAPPA):
    return conditioning_2x2("float64", a_range, c_range, b_range, kappa)


def _t2(shape, chi_range, bound=200.0):
    # the motivating construction: permeability mu = 1 + chi over a
    # declared susceptibility envelope — the caller's own transform
    return coefficient_contrast(
        shape, "float64", chi_range, bound, transform=lambda chi: 1.0 + chi
    )


# --- T1 known answer 1: a,c in [1,2], b in ±0.5, kappa=8 ---------------------
# Probe Part 2: Z3 proves the negation unsat. Probe Part 3: the interval
# evaluation of the same obligation straddles (dependency-shaped: a and c
# are shared between tr and det).


def test_t1_ka1_interval_only_is_unknown_with_straddle_quoted():
    cv = check_contract(
        _t1((1, 2), (1, 2), (-0.5, 0.5)), vacuity_mode="inputs-only"
    )
    assert cv.requires_status == "UNKNOWN"  # NOT VERIFIED, NOT REFUTED
    obs = cv.requires.obligations
    assert [o.status for o in obs] == [
        "discharged", "discharged", "discharged", "unknown",
    ]
    # the straddle is quoted, in numbers, on the verdict itself
    straddle = [n for n in cv.requires.notes if "straddles" in n]
    assert len(straddle) == 1
    assert "obligation #3" in straddle[0]
    assert "lhs in [" in straddle[0] and "rhs in [" in straddle[0]
    assert "solver_timeout_ms" in straddle[0]  # the no-budget hint
    # never-on defaults: no solver ran, and the stamp records the absence
    assert isinstance(cv.requires.stamp.solver, SolverStamp)
    assert not cv.requires.stamp.solver.invoked
    # the ensures face rides along unchanged, DECLARED
    assert cv.ensures_status == ENSURES_DECLARED


@needs_solver
def test_t1_ka1_with_solver_budget_is_verified_via_qf_nra():
    contract = _t1((1, 2), (1, 2), (-0.5, 0.5))
    cv = check_contract(
        contract, vacuity_mode="inputs-only", solver_timeout_ms=20000
    )
    assert cv.requires_status == "VERIFIED"
    ratio = cv.requires.obligations[3]
    assert ratio.status == "discharged"
    assert "solver escalation (QF_NRA)" in ratio.detail
    assert "unsat" in ratio.detail
    # every invocation is stamped; the fragment routed QF_NRA
    stamps = cv.requires.stamp.solver
    assert isinstance(stamps, tuple) and stamps
    assert all(s.invoked for s in stamps)
    assert any("QF_NRA" in s.reason for s in stamps)
    # the widen re-check ran at the same pipeline depth and could not
    # re-derive the VERIFIED (over (-inf, inf) nothing discharges) — the
    # stamp states that measurement, never the load-bearing inference
    # (audit-4 F1)
    assert any(
        "vacuity checked (mode=inputs-only)" in a
        and "not re-derivable without the declared envelope" in a
        for a in cv.requires.stamp.assumptions
    )
    # requires-VERIFIED is universal, not existential: the standard
    # may-be-vacuous nonvacuity note must ride (emptiness asymmetry)
    assert any("may be vacuous" in n for n in cv.requires.notes)
    # the ensures is the caller's object, untouched — no upgrade path
    assert cv.ensures is contract.ensures
    assert cv.ensures.status == ENSURES_DECLARED


# --- T1 known answer 2: b in ±1.4 — satisfiable, witness-backed --------------


@needs_solver
def test_t1_ka2_sliver_region_refuted_with_replayed_witness():
    cv = check_contract(
        _t1((1, 2), (1, 2), (-1.4, 1.4)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "REFUTED"
    assert cv.requires.witnesses  # a concrete witness, not just set-level
    for w in cv.requires.witnesses:
        # replay-confirmed by the existing validator before render
        assert "exact-rational replay" in w.replay
        vals = {name: Fraction(text) for name, text in w.values}
        a, c, b = vals["x0"], vals["x1"], vals["x2"]
        # box membership (declaration order: a, c, b; the b bounds are the
        # exact dyadic values of the declared floats, as emitted)
        assert 1 <= a <= 2 and 1 <= c <= 2
        assert Fraction(-1.4) <= b <= Fraction(1.4)
        # the violation, re-derived here in exact arithmetic
        det = a * c - b * b
        if w.obligation_index == 2:
            assert det < 0
        elif w.obligation_index == 3:
            assert (a + c) ** 2 > det * Fraction(str(RHS_COEFF))
        else:  # pragma: no cover - no other obligation may refute here
            raise AssertionError(f"unexpected witness {w.obligation_index}")


# --- T1 known answer 3: the probe Part 1 mesh point --------------------------
# F3's real per-cell normal matrix (one-face stencil + reg): a = 0.25+1e-30,
# b = 0, c = 1e-30 — cond_2 ≈ 2.5e29. The ratio conjunct fails definitively
# at this point; the interval path alone must refute, no solver.


def test_t1_ka3_degenerate_mesh_point_refuted_by_intervals_alone():
    a0 = 0.25 + 1e-30
    cv = check_contract(
        _t1((a0, a0), (1e-30, 1e-30), (0.0, 0.0)),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status == "REFUTED"
    ratio = cv.requires.obligations[3]
    assert ratio.status == "violated-over-set"
    assert "definitely false" in ratio.detail
    # no solver was needed, and the stamp records exactly that
    assert isinstance(cv.requires.stamp.solver, SolverStamp)
    assert not cv.requires.stamp.solver.invoked
    assert "no solver invoked" in cv.requires.stamp.solver.reason


# --- T1 counterfactual pins --------------------------------------------------


def test_t1_pin_reduction_constant_kappa_plus_inverse_plus_two():
    """Pins RHS_COEFF = kappa + 1/kappa + 2 (probe Part 2, exact).

    Counterfactual: the plausible wrong constant kappa + 2 (dropping the
    1/kappa term of the exact identity tr^2/det = r + 1/r + 2). At the
    point M = [[1, 0.775], [0.775, 1]], tr^2/det ≈ 10.0156 lies strictly
    between 10 (wrong coefficient, kappa=8) and 10.125 (right), so the
    correct implementation VERIFIES and a kappa+2 implementation REFUTES
    — measured: substituting kappa+2 flips this exact query to REFUTED
    via the interval path."""
    ratio = 4.0 / (1.0 - 0.775 * 0.775)
    assert 10.0 < ratio < 10.125  # the pin's bite, in numbers
    cv = check_contract(
        _t1((1, 1), (1, 1), (0.775, 0.775)), vacuity_mode="inputs-only"
    )
    assert cv.requires_status == "VERIFIED"


@needs_solver
def test_t1_pin_closed_inequality_at_the_conditioning_boundary():
    """Pins the closed (non-strict) posing of the ratio conjunct.

    Over a in [1, 8], c = 1, b = 0 the worst point is a = 8, where
    tr^2 = 81 EQUALS det * 10.125 = 81 exactly: the closed obligation
    holds everywhere with equality attained on the envelope boundary.
    Counterfactual: a strict posing (tr^2 < det*coeff) REFUTES — its
    negation is satisfiable exactly at a = 8 — measured: substituting <
    for <= flips this query to REFUTED with witness a=8, c=1, b=0."""
    assert (8.0 + 1.0) ** 2 == 8.0 * RHS_COEFF  # equality is exact
    cv = check_contract(
        _t1((1, 8), (1, 1), (0.0, 0.0)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "VERIFIED"
    assert "solver escalation (QF_NRA)" in cv.requires.obligations[3].detail


# --- T2 known answer 1: chi in [1e-6, 1e2], n=64, C=200 ----------------------
# mu = 1 + chi gives contrast ≈ 101: max <= C*min holds with
# interval-visible margin (independent element boxes — min/max attain box
# endpoints, no dependency problem), so the interval path decides it.


def test_t2_ka1_contrast_within_bound_verified_by_intervals_alone():
    cv = check_contract(
        _t2((64,), (1e-6, 1e2)), vacuity_mode="inputs-only"
    )
    assert cv.requires_status == "VERIFIED"
    assert [o.status for o in cv.requires.obligations] == [
        "discharged", "discharged",
    ]
    assert isinstance(cv.requires.stamp.solver, SolverStamp)
    assert not cv.requires.stamp.solver.invoked  # intervals decided it
    # requires-only contract: the verdict says so, in words
    assert cv.ensures is None
    assert cv.ensures_status == "none declared"
    assert any(
        "ensures face: none declared" in n for n in cv.requires.notes
    )
    assert "ensures: none declared" in cv.render()


# --- T2 known answer 2: chi in [1e-6, 1e5] — violated ------------------------


def test_t2_ka2_interval_only_is_honest_unknown_with_straddle():
    cv = check_contract(
        _t2((64,), (1e-6, 1e5)), vacuity_mode="inputs-only"
    )
    assert cv.requires_status == "UNKNOWN"
    straddle = [n for n in cv.requires.notes if "straddles" in n]
    assert len(straddle) == 1 and "obligation #1" in straddle[0]


@needs_solver
def test_t2_ka2_with_solver_refuted_witness_names_high_and_low_elements():
    cv = check_contract(
        _t2((64,), (1e-6, 1e5)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "REFUTED"
    (w,) = cv.requires.witnesses
    assert w.obligation_index == 1  # the contrast obligation
    assert "exact-rational replay" in w.replay  # replay-confirmed
    mus = {name: 1 + Fraction(text) for name, text in w.values}
    assert len(mus) == 64  # every element named
    hi_name = max(mus, key=mus.get)
    lo_name = min(mus, key=mus.get)
    # the witness names one high and one low element, and they violate
    assert hi_name != lo_name
    assert mus[hi_name] > 200 * mus[lo_name]
    # membership: every named element is inside the declared chi box —
    # whose endpoints are the exact dyadic values of the declared floats
    # (Fraction(1e-6) is slightly below decimal 1/10^6; the emission and
    # the witness validator both speak the float's exact value)
    for mu in mus.values():
        assert Fraction(1e-6) <= mu - 1 <= Fraction(1e5)


# --- T2 encoding pins --------------------------------------------------------


def test_t2_pin_division_free_and_reduction_free_query():
    """Pins the T2 encoding, structurally, on the traced query itself.

    Counterfactual (division): posing the contrast as max/min <= C emits
    a 'div' — besides emitting the division the spec forbids, it hits
    the escalation div-guard whenever the minimum may be zero. No 'div'
    may appear anywhere in the traced requires.

    Counterfactual (reduction): posing the extrema as jnp.max/jnp.min
    emits 'reduce_max'/'reduce_min', which have NO transfer or emission
    row — every escalation would decline and the violated known answer
    could never produce its witness. The fold encoding must keep the
    query inside the supported set: slice + binary max/min."""
    closed = trace(_t2((8,), (1e-6, 1e2)).harness)
    prims = set()
    stack = list(closed.jaxpr.eqns)
    while stack:
        eqn = stack.pop()
        prims.add(eqn.primitive)
        for sub in sub_jaxprs(eqn):
            stack.extend(sub.eqns)
    assert "div" not in prims
    assert "reduce_max" not in prims and "reduce_min" not in prims
    assert {"slice", "max", "min"} <= prims  # the fold is really there


@needs_solver
def test_t2_pin_exact_closed_extremum_encoding_at_the_contrast_boundary():
    """Pins the fold encoding's exactness and the closed contrast form.

    Over chi in [0, 199] with C = 200, the extreme field (one element at
    mu = 200, one at mu = 1) attains max = C*min EXACTLY; the closed
    obligation holds over the whole envelope with equality attained, so
    the correct implementation VERIFIES (by QF_LRA escalation — the
    interval comparison straddles at the shared endpoint 200).

    Counterfactuals measured to flip this query: (i) a strict posing
    (max < C*min) REFUTES — its negation is satisfiable exactly at the
    boundary field (witness mu_high = 200, mu_low = 1); (ii) a
    one-directional extremum encoding (a bound m >= every element
    instead of the exact selection) makes the negated obligation
    satisfiable by an unboundedly large m, minting a wrong REFUTED
    here."""
    cv = check_contract(
        _t2((4,), (0.0, 199.0)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "VERIFIED"
    assert "solver escalation (QF_LRA)" in cv.requires.obligations[1].detail


# --- T2: the budget decline is loud ------------------------------------------


def test_t2_over_budget_declines_loudly_with_both_numbers_quoted():
    """mu = 1+chi at n elements costs 4n element terms; n = 200 is over
    the 512 budget: the obligation must stay UNKNOWN with the decline
    quoting both quantities and the budget — never a silent
    approximation, never a hang."""
    cv = check_contract(
        _t2((200,), (1e-6, 1e5)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "UNKNOWN"
    declines = [
        n
        for n in cv.requires.notes
        if "over the per-obligation emission budget of 512" in n
    ]
    assert declines
    assert any("800 element terms" in n for n in declines)
    # the decline happened before any invocation: absence is stamped
    assert isinstance(cv.requires.stamp.solver, SolverStamp)
    assert not cv.requires.stamp.solver.invoked


# --- the two-faced render and the emptiness asymmetry ------------------------


@needs_solver
def test_contract_render_states_both_faces_in_words():
    cv = check_contract(
        _t1((1, 2), (1, 2), (-0.5, 0.5)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    text = cv.render()
    # the requires face: mechanized, standard stamp present
    assert "requires face (MECHANIZED" in text
    assert "== VERIFIED" in text
    assert "query " in text and "solver:" in text  # the standard stamp
    # the ensures face: DECLARED, conditional, derivation stamped
    assert "ensures face (DECLARED)" in text
    assert "declared, NOT checked" in text
    assert "holds only" in text and "where the requires holds" in text
    assert "conditional on the requires face:" in text
    assert "derivation:" in text
    assert "corpus/supply/la_contract_probe.py Part 4" in text
    # the norm-sensitivity statement itself (probe Part 4)
    assert "||M^-1 r||_2 <= (8.0 / ||M||_2) * ||r||_2" in text
    # the ensures line is stamped (append-only, on the standard stamp)
    assert any(
        "ensures face: DECLARED" in a
        for a in cv.requires.stamp.assumptions
    )


def test_emptiness_asymmetry_no_existential_phrasing_anywhere():
    """A requires-VERIFIED is universally quantified over the declared
    envelope and must NOT certify nonemptiness; the ensures must be
    vacuously true on an empty envelope and say so. No render, stamp
    line, or module docstring may phrase either face existentially."""
    cv = check_contract(
        _t1((1, 1), (1, 1), (0.0, 0.0)), vacuity_mode="inputs-only"
    )
    text = cv.render()
    for banned in ("there exist", "there is a well-conditioned",
                   "the envelope contains"):
        assert banned not in text
        assert banned not in (contracts.__doc__ or "")
    # the universal reading and the vacuous-truth disclosure, in words
    assert "for every point" in cv.ensures.statement
    assert "vacuously true" in cv.ensures.statement
    assert "does not assert that any such point exists" in cv.ensures.statement
    assert "vacuously true" in text
    assert "nonempty" in text  # the render names the asymmetry outright
    assert "stelling.exactness" in text  # and routes the caller


def test_ensures_face_survives_check_unchanged_and_unupgraded():
    contract = _t1((1, 1), (1, 1), (0.0, 0.0))
    cv = check_contract(contract, vacuity_mode="inputs-only")
    assert cv.requires_status == "VERIFIED"
    # the checker carries the caller's face through by identity: there is
    # no site at which an ensures status could have been minted
    assert cv.ensures is contract.ensures
    assert cv.ensures.status == ENSURES_DECLARED
    with pytest.raises(ValueError, match="DECLARED and nothing else"):
        dataclasses.replace(cv.ensures, status="VERIFIED")


def test_contract_verdict_exposes_no_combined_status():
    cv = check_contract(
        _t1((1, 1), (1, 1), (0.0, 0.0)), vacuity_mode="inputs-only"
    )
    assert isinstance(cv, ContractVerdict)
    # the two faces are separate surfaces by design; a combined `status`
    # would collapse DECLARED into the verdict statuses
    assert not hasattr(cv, "status")
    assert cv.requires_status == "VERIFIED"
    assert cv.ensures_status == ENSURES_DECLARED


# --- the transform convention (mirrors field_positive) -----------------------


def test_t2_tuple_transform_poses_the_pair_per_produced_value():
    def two_fields(chi):
        return 1.0 + chi, 2.0 + chi

    cv = check_contract(
        coefficient_contrast(
            (8,), "float64", (1e-6, 1e2), 200.0, transform=two_fields
        ),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status == "VERIFIED"
    assert len(cv.requires.obligations) == 4  # (positivity, contrast) x 2


# --- audit F2: the vacuity instrument on all-point envelopes -----------------


def test_point_envelope_vacuity_is_inert_not_falsely_load_bearing():
    """Under inputs-only, point declarations hold still, so an all-point
    envelope widens to the IDENTICAL query: the re-run proves nothing.
    The old path stamped 'discharges with the declared bounds widened to
    (-inf, inf)' — measured false (audit b_emptiness.py b6: mode='all'
    on the same query says load-bearing). The instrument must now stamp
    an honest inert line, run no re-run, and claim load-bearing in
    NEITHER direction; mode='all' widens the points for real and keeps
    the measuring path."""
    contract = _t1((1, 1), (1, 1), (0.775, 0.775))
    cv = check_contract(contract, vacuity_mode="inputs-only")
    assert cv.requires_status == "VERIFIED"
    inert = [
        a for a in cv.requires.stamp.assumptions
        if "vacuity instrument inert (mode=inputs-only)" in a
    ]
    assert len(inert) == 1
    assert "every declared input is a point interval" in inert[0]
    assert "mode='all'" in inert[0]
    # no load-bearing claim in either direction, and no false widen claim
    assert not any(
        "bounds widened" in a for a in cv.requires.stamp.assumptions
    )
    assert not any("load-bearing" in a for a in cv.requires.stamp.assumptions)
    assert not any("load-bearing" in n for n in cv.requires.notes)
    # mode='all' on the SAME query widens the points: the real
    # instrument ran at full power and failed to re-derive — the
    # universal wording states exactly that measurement (audit-4 F1)
    cv_all = check_contract(contract, vacuity_mode="all")
    assert cv_all.requires_status == "VERIFIED"
    assert any(
        "vacuity checked (mode=all)" in a
        and "not re-derivable without the declared envelope" in a
        for a in cv_all.requires.stamp.assumptions
    )


# --- audit F3: widen re-check solver invocations are stamped -----------------


@needs_solver
def test_widen_recheck_solver_invocations_are_stamped_and_tagged(monkeypatch):
    """The VERIFIED widen re-check runs at the same pipeline depth, so on
    the escalated path it makes real transport spawns the vacuity line
    relies on — measured 10 spawns vs 2 stamped before the fix (audit
    j_spawncount.py). Every invocation must now be stamped, the
    re-check's tagged apart from the original run's; and on the inert
    (all-point) path there is no re-run, so spawns == stamps with no
    tagged entries."""
    import stelling.solvers as sv

    real_run = sv._Backend.run
    count = {"spawns": 0}

    def counted(self, ledger, text, wall):
        count["spawns"] += 1
        return real_run(self, ledger, text, wall)

    monkeypatch.setattr(sv._Backend, "run", counted)

    cv = check_contract(
        _t1((1, 2), (1, 2), (-0.5, 0.5)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "VERIFIED"
    stamps = cv.requires.stamp.solver
    assert isinstance(stamps, tuple) and all(s.invoked for s in stamps)
    assert len(stamps) == count["spawns"]  # every spawn is in the stamp
    tagged = [
        s for s in stamps if s.reason.startswith("vacuity widen re-check: ")
    ]
    untagged = [s for s in stamps if s not in tagged]
    assert tagged and untagged  # both the original asks and the re-check's

    # the inert path (all-point envelope): no re-run, nothing tagged
    count["spawns"] = 0
    cv0 = check_contract(
        _t1((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv0.requires_status == "VERIFIED"
    stamps0 = cv0.requires.stamp.solver
    assert isinstance(stamps0, tuple)
    assert len(stamps0) == count["spawns"]
    assert not any(
        s.reason.startswith("vacuity widen re-check: ") for s in stamps0
    )
    assert any(
        "vacuity instrument inert" in a
        for a in cv0.requires.stamp.assumptions
    )


# --- audit F4/F5: stranger-shaped transforms ---------------------------------


def test_t2_empty_transform_refused_loudly():
    """transform=lambda chi: () used to yield a silent zero-obligation
    UNKNOWN with no note saying why (audit g_stranger.py g3a)."""
    with pytest.raises(ValueError, match="transform returned no values"):
        check_contract(
            coefficient_contrast((8,), "float64", (0.0, 1.0), 10.0,
                                 transform=lambda chi: ()),
            vacuity_mode="inputs-only",
        )


def test_t2_constant_transform_returns_produce_verdicts():
    """A transform returning a plain constant (or a tuple containing one)
    crashed raw in fold_extrema (AttributeError on .size — audit
    g_stranger.py g3b/g3c); the mirrored field_positive convention
    tolerates constant returns, so these now pose trivially-discharged
    obligations, with the vacuity instrument disclosing them."""
    cv = check_contract(
        coefficient_contrast((8,), "float64", (0.0, 1.0), 10.0,
                             transform=lambda chi: 3.0),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status == "VERIFIED"
    assert any("envelope not load-bearing" in n for n in cv.requires.notes)
    cv2 = check_contract(
        coefficient_contrast((8,), "float64", (0.0, 1.0), 10.0,
                             transform=lambda chi: (1.0 + chi, 2.0)),
        vacuity_mode="inputs-only",
    )
    assert cv2.requires_status == "VERIFIED"
    assert len(cv2.requires.obligations) == 4


# --- audit F7/F10: eager argument validation at the pipeline door ------------


def test_vacuity_mode_typo_raises_eagerly_on_unknown_bound_query():
    """The widen used to be the only validator and runs only on a
    VERIFIED: a typo'd mode rode green through UNKNOWN paths and would
    first explode on the day a VERIFIED occurred (audit g_stranger.py
    g2a/g2b). Now refused before tracing, with widen's own wording."""
    with pytest.raises(ValueError, match="widen mode must be one of"):
        check_contract(
            _t1((1, 2), (1, 2), (-0.5, 0.5)),  # UNKNOWN-bound query
            vacuity_mode="inputs_only",  # underscore typo
        )


def test_solver_timeout_ms_rejects_str_and_float():
    """'5000' parsed and 5000.9 silently truncated (audit h_misc H5): a
    stamped, reproducible budget must be exactly the int the caller
    wrote — or None."""
    contract = _t1((1, 1), (1, 1), (0.0, 0.0))
    with pytest.raises(TypeError, match="solver_timeout_ms must be an int"):
        check_contract(contract, vacuity_mode="inputs-only",
                       solver_timeout_ms="20000")
    with pytest.raises(TypeError, match="solver_timeout_ms must be an int"):
        check_contract(contract, vacuity_mode="inputs-only",
                       solver_timeout_ms=20000.9)
    with pytest.raises(TypeError, match="solver_timeout_ms must be an int"):
        check_contract(contract, vacuity_mode="inputs-only",
                       solver_timeout_ms=True)


# --- T3: conditioning_2x2_field — the family template -------------------------
# MIME-free by construction: every transform here is a hand construction.
# The known answers reuse the probe's Z3-checked ground truth (probe Parts
# 1-2) through the family template's two return conventions.

from stelling.contracts import conditioning_2x2_field  # noqa: E402


def _stack_last(*scalars):
    """Family assembly on supported primitives only: concatenate over
    reshaped single-element pieces (measured on jax 0.11.0: jnp.stack
    traces to a `stack` primitive with no transfer row)."""
    import jax.numpy as jnp

    return jnp.concatenate([jnp.reshape(s, (1,)) for s in scalars])


def _as_family(rows):
    """rows: list of (m00, m01, m10, m11) scalar tuples -> [N, 2, 2]."""
    import jax.numpy as jnp

    flat = jnp.concatenate([_stack_last(*r) for r in rows])
    return jnp.reshape(flat, (len(rows), 2, 2))


def test_t3_trace_time_refusals_are_loud():
    """Out-of-convention transform returns refuse at trace time with the
    convention quoted — never a silent zero-obligation UNKNOWN."""
    import jax.numpy as jnp

    with pytest.raises(ValueError, match="length 2"):
        check_contract(
            conditioning_2x2_field(
                (2,), "float64", (0.0, 1.0), 8.0, lambda t: (1.0, 2.0)
            ),
            vacuity_mode="inputs-only",
        )
    with pytest.raises(ValueError, match=r"trailing shape \(2, 2\)"):
        check_contract(
            conditioning_2x2_field(
                (3,), "float64", (0.0, 1.0), 8.0, lambda t: t
            ),
            vacuity_mode="inputs-only",
        )
    with pytest.raises(ValueError, match="zero matrices"):
        check_contract(
            conditioning_2x2_field(
                (2,), "float64", (0.0, 1.0), 8.0,
                lambda t: jnp.zeros((0, 2, 2)),
            ),
            vacuity_mode="inputs-only",
        )
    with pytest.raises(ValueError, match="neither an array"):
        check_contract(
            conditioning_2x2_field(
                (2,), "float64", (0.0, 1.0), 8.0, lambda t: 3.0
            ),
            vacuity_mode="inputs-only",
        )


def test_t3_triple_refuses_mixed_non_scalar_shapes():
    """Audit round 2, F6: a triple mixing non-scalar shapes ((2, 1) with
    (3,)) would silently outer-broadcast to a family the caller never
    posed — refused at trace time naming both shapes."""
    import jax.numpy as jnp

    def mixed(t):
        a = jnp.reshape(1.0 + t, (2, 1))  # (2, 1)
        b = jnp.concatenate([t, t[:1]])  # (3,)
        return a, b, 1.5

    with pytest.raises(ValueError, match=r"\(2, 1\).*\(3,\)"):
        check_contract(
            conditioning_2x2_field((2,), "float64", (0.0, 0.1), 8.0, mixed),
            vacuity_mode="inputs-only",
        )


def test_t3_triple_scalar_broadcast_still_works():
    """The documented convention: scalars broadcast against the one
    non-scalar shape (a over (2,), b and c scalar -> a (2,)-family)."""

    def tri(t):
        return 1.0 + t, 0.0 * t[0], 1.5 + t[0]

    cv = check_contract(
        conditioning_2x2_field((2,), "float64", (0.0, 0.1), 8.0, tri),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status == "VERIFIED"
    assert len(cv.requires.obligations) == 4
    assert all(o.status == "discharged" for o in cv.requires.obligations)
    # the broadcast is per-expression: obligations touching `a` (a >= 0,
    # det >= 0, the ratio) are judged over the broadcast 2 elements; the
    # scalar-only c >= 0 stays scalar
    details = [o.detail for o in cv.requires.obligations]
    assert "2 element(s)" in details[0]
    assert "1 element(s)" in details[1]
    assert "2 element(s)" in details[2] and "2 element(s)" in details[3]


def test_t3_triple_vs_array_paths_pose_the_same_conditioning():
    """The two return conventions on the same well-conditioned point
    family: the triple poses the four conjuncts (the caller's off-diagonal
    IS their symmetry assertion), the [..., 2, 2] array poses the same
    four PLUS the two symmetry obligations — indices 0-3 stable across
    both paths, 4-5 the symmetry pair.

    Measured seam, pinned here: even at an all-point envelope the
    symmetry pair STRADDLES when the off-diagonal is a computed product
    (0.0 * t propagates to the outward-rounded [-5e-324, 5e-324], and
    equal non-point boxes straddle <= in both directions) — the array
    path is interval-UNKNOWN where the triple path is interval-VERIFIED,
    and the pair needs the (trivial) solver step to close."""

    def triple(t):
        return 1.0 + t, 0.0 * t, 1.0 + t

    def family(t):
        a = 1.0 + t
        b = 0.0 * t
        return _as_family([(a, b, b, a)])[0]  # single [2, 2] matrix

    cv3 = check_contract(
        conditioning_2x2_field((), "float64", (0.0, 0.0), 8.0, triple),
        vacuity_mode="inputs-only",
    )
    assert cv3.requires_status == "VERIFIED"
    assert len(cv3.requires.obligations) == 4
    cv4 = check_contract(
        conditioning_2x2_field((), "float64", (0.0, 0.0), 8.0, family),
        vacuity_mode="inputs-only",
    )
    assert cv4.requires_status == "UNKNOWN"
    assert len(cv4.requires.obligations) == 6  # + the posed symmetry pair
    assert [o.status for o in cv4.requires.obligations] == (
        ["discharged"] * 4 + ["unknown"] * 2
    )
    straddles = [n for n in cv4.requires.notes if "straddles" in n]
    assert len(straddles) == 2  # the pair, quoted in numbers
    # both faces declare the same per-matrix ensures shape
    assert cv3.ensures_status == ENSURES_DECLARED
    assert "every matrix of the family" in cv4.ensures.statement


def test_t3_family_query_stays_on_supported_primitives():
    """Structural pin: the [..., 2, 2] extraction is slice+squeeze and the
    assembly is concatenate+reshape — no gather, no scatter, no stack may
    appear (gather/stack have no emission row; their appearance would send
    the family or its escalation to ⊤/decline)."""

    def family(t):
        a = 1.0 + t[0]
        b = t[1] * t[0]
        return _as_family([(a, b, b, a), (a, b, b, a)])

    closed = trace(
        conditioning_2x2_field((2,), "float64", (0.1, 0.2), 8.0, family).harness
    )
    prims = set()
    stack = list(closed.jaxpr.eqns)
    while stack:
        eqn = stack.pop()
        prims.add(eqn.primitive)
        for sub in sub_jaxprs(eqn):
            stack.extend(sub.eqns)
    assert "gather" not in prims
    assert "scatter" not in prims
    assert "stack" not in prims
    assert {"slice", "concatenate", "reshape", "mul"} <= prims


def test_t3_symmetric_family_symmetry_straddles_then_solver_discharges():
    """The posed symmetry pair on a symmetric-by-construction family over
    a non-point envelope: the off-diagonals are DISTINCT element terms
    (mul(t0, t1) vs mul(t1, t0)) with equal propagated boxes, and equal
    boxes straddle <= in both directions — interval-UNKNOWN, honestly.
    The straddle is quoted in numbers on the verdict."""

    def family(t):
        b01 = t[0] * t[1]
        b10 = t[1] * t[0]  # the transcribed-outer-product shape
        return _as_family([(1.5, b01, b10, 1.5)])

    contract = conditioning_2x2_field((2,), "float64", (0.1, 0.2), 8.0, family)
    cv = check_contract(contract, vacuity_mode="inputs-only")
    assert cv.requires_status == "UNKNOWN"
    assert [o.status for o in cv.requires.obligations] == [
        "discharged", "discharged", "discharged", "discharged",
        "unknown", "unknown",  # the symmetry pair, straddling
    ]
    straddles = [n for n in cv.requires.notes if "straddles" in n]
    assert len(straddles) == 2
    assert any("obligation #4" in n for n in straddles)
    assert any("obligation #5" in n for n in straddles)


@needs_solver
def test_t3_symmetry_pair_discharged_trivially_by_solver():
    def family(t):
        b01 = t[0] * t[1]
        b10 = t[1] * t[0]
        return _as_family([(1.5, b01, b10, 1.5)])

    cv = check_contract(
        conditioning_2x2_field((2,), "float64", (0.1, 0.2), 8.0, family),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "VERIFIED"
    for idx in (4, 5):
        ob = cv.requires.obligations[idx]
        assert ob.status == "discharged"
        assert "solver escalation" in ob.detail and "unsat" in ob.detail


@needs_solver
def test_t3_posed_symmetry_refutes_asymmetric_transform_with_witness():
    """The symmetry rule's bite: a deliberately asymmetric family (m10 =
    2*m01, undecidable by intervals since the boxes overlap) REFUTES with
    a replay-confirmed witness confirming the asymmetry. A template that
    silently read one off-diagonal would have VERIFIED this family."""

    def family(t):
        return _as_family([(1.0, t, 2.0 * t, 1.0)])

    cv = check_contract(
        conditioning_2x2_field((), "float64", (0.1, 0.2), 8.0, family),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "REFUTED"
    assert cv.requires.witnesses
    for w in cv.requires.witnesses:
        assert w.obligation_index == 5  # m10 <= m01 is the violated one
        assert "exact-rational replay" in w.replay
        (name, text), = [(n, v) for n, v in w.values]
        theta = Fraction(text)
        assert Fraction(0.1) <= theta <= Fraction(0.2)  # box membership
        assert 2 * theta > theta  # the asymmetry, re-derived exactly


def test_t3_identity_family_over_a_box_is_honestly_undecided():
    """transform=None: theta IS the family, its off-diagonals are
    independent declared elements — the family genuinely contains
    asymmetric matrices, so symmetry must NOT verify."""
    cv = check_contract(
        conditioning_2x2_field((2, 2), "float64", (1.0, 2.0), 8.0, None),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status in ("UNKNOWN", "REFUTED")
    assert cv.requires.obligations[4].status != "discharged"
    assert cv.requires.obligations[5].status != "discharged"


@needs_solver
def test_t3_identity_family_refuted_with_confirmed_witnesses():
    """The identity family over a box REFUTES under escalation: with all
    four entries independent it contains non-PSD members and asymmetric
    members alike, and whichever obligation the portfolio refutes first,
    every shown witness is replay-confirmed and inside the declared box.
    A witness on the symmetry pair names a concrete asymmetric member."""
    cv = check_contract(
        conditioning_2x2_field((2, 2), "float64", (1.0, 2.0), 8.0, None),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "REFUTED"
    assert cv.requires.witnesses
    for w in cv.requires.witnesses:
        assert w.obligation_index in (2, 3, 4, 5)
        assert "exact-rational replay" in w.replay
        vals = {name: Fraction(text) for name, text in w.values}
        assert all(1 <= v <= 2 for v in vals.values())
        if w.obligation_index in (4, 5):
            # flat C-order of (2, 2): x0_1 = M[0,1], x0_2 = M[1,0]
            assert vals["x0_1"] != vals["x0_2"]  # concrete asymmetric member


def test_t3_known_conditioning_refuted_by_intervals_at_the_mesh_point():
    """Probe Part 1 through the family template: the boundary-starved
    per-cell normal matrix (a = 0.25 + reg, b = 0, c = reg) as a
    two-member family — REFUTED on intervals alone, per matrix, both
    members, no solver."""
    reg = 1e-30

    def family(t):
        a = (0.25 + reg) + t
        b = 0.0 * t
        c = reg + t
        return _as_family([(a, b, b, c), (a, b, b, c)])

    cv = check_contract(
        conditioning_2x2_field((), "float64", (0.0, 0.0), 8.0, family),
        vacuity_mode="inputs-only",
    )
    assert cv.requires_status == "REFUTED"
    ratio = cv.requires.obligations[3]
    assert ratio.status == "violated-over-set"
    assert "definitely false for 2/2 element(s)" in ratio.detail
    assert isinstance(cv.requires.stamp.solver, SolverStamp)
    assert not cv.requires.stamp.solver.invoked


@needs_solver
def test_t3_known_conditioning_solver_verified_through_a_transform():
    """Probe Part 2's well-shaped region through a triple transform
    (a, c in [1, 2] via 1 + t, b in ±0.5 via t - 0.5 scaling): solver
    VERIFIED via QF_NRA, exactly as the point template's KA1."""

    def triple(t):
        return 1.0 + t[0], t[1] - 0.5, 1.0 + t[2]

    cv = check_contract(
        conditioning_2x2_field((3,), "float64", (0.0, 1.0), 8.0, triple),
        vacuity_mode="inputs-only",
        solver_timeout_ms=20000,
    )
    assert cv.requires_status == "VERIFIED"
    ratio = cv.requires.obligations[3]
    assert ratio.status == "discharged"
    assert "solver escalation (QF_NRA)" in ratio.detail
    assert len(cv.requires.obligations) == 4  # triple path: no symmetry pair


def test_t3_ensures_face_is_sealed_declared_and_per_matrix():
    contract = conditioning_2x2_field(
        (), "float64", (0.0, 0.0), 8.0, lambda t: (1.0 + t, 0.0 * t, 1.0 + t)
    )
    face = contract.ensures
    assert type(face) is EnsuresFace
    assert face.status == ENSURES_DECLARED
    assert "every matrix of the family" in face.statement
    assert "vacuously true" in face.statement
    assert "does not assert that any such point exists" in face.statement
    assert "corpus/supply/la_contract_probe.py Part 4" in face.derivation
    assert face.conditional_on == contract.requires_description
    for text in (face.statement, face.derivation, face.conditional_on):
        assert "\n" not in text and "\r" not in text
    with pytest.raises(ValueError, match="DECLARED and nothing else"):
        dataclasses.replace(face, status="VERIFIED")


# --- audit F9: the straddle hint promises the offer, not the outcome ---------


def test_straddle_hint_promises_offer_not_result():
    """At n=200 the old hint promised 'escalates exactly this obligation'
    while the escalation layer declines on budget and no solver runs
    (audit g_stranger.py g6). The hint now promises only the offer; the
    decline text owns the rest."""
    cv = check_contract(
        _t1((1, 2), (1, 2), (-0.5, 0.5)), vacuity_mode="inputs-only"
    )
    (note,) = [n for n in cv.requires.notes if "straddles" in n]
    assert "offers exactly this obligation to solver escalation" in note
    assert "escalates exactly this obligation" not in note
