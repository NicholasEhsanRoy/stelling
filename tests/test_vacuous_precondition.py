# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The admitted region a solver discharge rests on (audit 0.2.0 S7, S7', M5).

THE DEFECT. A relational ``assume`` is inert in the interval domain, so
``propagate._unsatisfiable`` — the oracle that refuses an empty declared set —
never sees it: that oracle meets a BOX with a half-space, and ``x < y`` is not
a half-space on either box. Since 0.2.0 the same assume is emitted to the
solver as a POSITIVE AXIOM. If the axiom set is unsatisfiable, on its own or
against the declared boxes, then ``boxes ∧ axioms ∧ ¬P`` is unsat for that
reason alone, FOR EVERY P: the obligation discharges, the verdict is VERIFIED,
and nothing checked whether the ``unsat`` came from the obligation or from an
empty precondition.

The asymmetry is what makes it a defect rather than a technicality: the
NON-RELATIONAL form of the identical mistake — ``dt ∈ [5,10]`` with
``assume(dt < 1.0)`` — raises ``UnsatisfiableAssumptionError`` and says
"harness defect; nothing was verified". 0.2.0's forwarding built a route
around that refusal, and fixing S5 (correct forwarding from behind a
transparent call) made the route MORE reachable, not less.

WHAT THESE TESTS PIN, in the order the repair reads:

1. the three shapes of the empty region — a mis-declared bound, an axiom
   cycle, and the inductive-step form — each REFUSE, in the same class and
   with the same sentence as the non-relational form;
2. the CONTROL, which is the whole point: a satisfiable relational assume
   whose VERIFIED must stay VERIFIED, and must stay CLEAN (no may-be-vacuous
   line). A repair that turned every relational-assume query into UNKNOWN
   would pass (1) and be worthless;
3. the emission property the check rests on — the admitted-region script is
   the obligation's own script minus EXACTLY the negated-obligation line;
4. the decision rule as a pure function, including the tie it refuses;
5. the disclosure: an undecided region stamps a may-be-vacuous line, and the
   vacuity sentence stops claiming substantiveness on such a run;
6. the two inductive-step over-claims (M5's conditional step, M4's
   positionally-mapped REFUTED note);
7. the SCOPE of both readings, which is audit B3 (below).

AUDIT B3, TWO FINDINGS ON THE REPAIR ABOVE.

**The check asked about the SLICE and the answer was read as being about the
QUERY.** ``_carry_assumes`` drops every relational assume whose operands fall
outside an obligation's backward cone, so the admitted-region script states a
SUBSET of the query's axioms. ``unsat`` on a subset is sound — an empty
relaxation proves the tighter set empty — but ``sat`` on one was read as
:data:`REGION_INHABITED`, "a point of the region". Measured: ``x, y, z ∈
[-10,10]``, ``assume(x<y); assume(y<z); assume(z<x)``, ``assert_(x - y <=
0.0)``. The assert's cone is ``{x, y}``, so the script states only ``x<y``;
``boxes ∧ (x<y)`` is satisfiable; VERIFIED, stamped CLEAN, over a precondition
that admits no point at all. The repair is one condition —
:data:`REGION_INHABITED` only when ``unaccounted_assumes`` is empty for this
obligation — and section 7 pins both directions of it.

**A per-obligation fact read as a whole-query one.** The stamped
``forwarded relational assume(s) on obligation(s) #k`` line names the
obligations it reaches; both readers asked ``any(... in assumptions)``, which
is the whole-query question. Measured, an interval refutation called
"conditional … judged over the propagated superset of the
precondition-narrowed set" on a run where nothing narrowed, and an inductive
step called "CONDITIONAL — NOT the inductive step" on a body whose bound
obligations were judged over the full declared box.
"""

from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

# THE BARE IDIOM ON PURPOSE. A custom ``reason=`` replaces pytest's standard
# "could not import 'jax'" message, and that message is what
# ``test_skip_inventory.py``'s ``_IMPORT_GATE`` matches to disclose the gate —
# a custom one reads as an UNDISCLOSED skip in every jax-less session, and
# reddens the inventory in exactly the environment nobody runs locally.
pytest.importorskip("jax")

import jax  # noqa: E402

from stelling import smt, solvers  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.inductive import check_inductive_step  # noqa: E402
from stelling.obligation import slice_unknown_obligations  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import (  # noqa: E402
    CONDITIONAL_ON_PRECONDITION,
    UNCERTIFIED_PRECONDITION_PREFIX,
    UnsatisfiableAssumptionError,
    conditional_on_precondition,
    interval_env,
    propagate,
)
from stelling.solvers import (  # noqa: E402
    REGION_EMPTY,
    REGION_INHABITED,
    REGION_NOT_ASKED,
    REGION_UNCERTIFIED,
    UNCERTIFIED_REGION_ASSUMPTION,
    _region_answer,
    relational_assume_assumption,
)

try:
    from stelling import _optional
    HAVE_SOLVER = (
        _optional.available("z3")
        or _optional.available("cvc5")
        or _optional.cvc5_binary() is not None
    )
except Exception:  # pragma: no cover - environment probe only
    HAVE_SOLVER = False

need_solver = pytest.mark.skipif(not HAVE_SOLVER, reason="needs an SMT solver")

TIMEOUT = 5000


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """This module declares float64 inputs, so it must ask for x64 ITSELF.

    Every harness here is `any_array((), "float64", …)`, and in a float32
    session those declarations TRUNCATE: the obligations decline, the
    escalation never runs, and the assertions fail on a
    `DeclinedObligation` rather than on anything this file is about. CI runs
    plain `pytest` with no `JAX_ENABLE_X64`, so a module that asks for
    nothing is a module that only passes on a developer's machine.

    A module-scoped fixture that SAVES AND RESTORES is the house pattern,
    and the restore is the load-bearing half: a bare module-scope
    `jax.config.update` runs at COLLECTION, before any test, and sets x64
    for the whole session — which is how a cross-process hash comparison in
    `test_transcribe.py` was broken by a module that never ran.
    """
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# ---------------------------------------------------------------------------
# the harnesses
# ---------------------------------------------------------------------------


def mis_declared_bound():
    """The ordinary typo, and the audit's headline reproducer.

    ``dt >= 5`` and ``dt_max <= 1``, so NO point of the declared box satisfies
    ``dt < dt_max``; and ``dt + dt_max >= 5`` everywhere in the box, so the
    assert is false at EVERY declared point. Deleting the assume gives
    REFUTED. With it, 0.2.0 gave VERIFIED.
    """
    dt = any_array((), "float64", (5.0, 10.0))
    dt_max = any_array((), "float64", (0.0, 1.0))
    assume(dt < dt_max)
    return assert_(dt + dt_max <= 1.0)


def mis_declared_bound_control():
    """The same query with the assume deleted — the refutation is real."""
    dt = any_array((), "float64", (5.0, 10.0))
    dt_max = any_array((), "float64", (0.0, 1.0))
    return assert_(dt + dt_max <= 1.0)


def axiom_cycle():
    """``x < y``, ``y < z``, ``z < x`` over ``[-10, 10]³``.

    Unsatisfiable with no help from the boxes at all — a strict order has no
    3-cycle — asserting ``x + y + z >= 100`` where the box maximum is 30.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    z = any_array((), "float64", (-10.0, 10.0))
    assume(x < y)
    assume(y < z)
    assume(z < x)
    return assert_(x + y + z >= 100.0)


def cone_split_cycle():
    """AUDIT B3's shape: the SAME 3-cycle, spread across obligation cones.

    ``assert_(x - y <= 0.0)`` has backward cone ``{x, y}``, so
    ``_carry_assumes`` carries ``x < y`` and skips the other two — their
    operands name nothing in this slice. The script the admitted-region check
    is asked about is therefore ``boxes ∧ (x < y)``, which is satisfiable, and
    reading that ``sat`` as "a point of the region" stamped a VERIFIED clean
    over an EMPTY precondition (:func:`_cone_split_admits_no_point` is the
    independent ground truth).

    Distinct from :func:`axiom_cycle`, where the assert's cone contains all
    three variables so the script states the whole contradiction and the
    ``unsat`` route already refuses. The difference between the two harnesses
    is one assert.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    z = any_array((), "float64", (-10.0, 10.0))
    assume(x < y)
    assume(y < z)
    assume(z < x)
    return assert_(x - y <= 0.0)


def cone_split_chain():
    """THE CONE-SPLIT CONTROL: the same shape with the cycle broken.

    ``x < y`` and ``y < z`` over the same boxes, the same
    ``assert_(x - y <= 0.0)``, the same skipped assume — everything the
    vacuous harness has except the emptiness. Its VERIFIED is substantive and
    must STAY VERIFIED.

    It is also where the repair's COST is measured. The region here really is
    inhabited (``(0, 1, 2)``), and nothing on this run establishes that: the
    script cannot state ``y < z`` — ``z`` is not in the cone — and the
    propagation's probe grid finds no point at which a STRICT chain is
    definitely true. So this VERIFIED is qualified, correctly and
    conservatively. :func:`cone_split_chain_certified` is the same shape one
    character different, and is not.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    z = any_array((), "float64", (-10.0, 10.0))
    assume(x < y)
    assume(y < z)
    return assert_(x - y <= 0.0)


def cone_split_chain_certified():
    """The cone-split control that stays CLEAN, and the reason the repair is
    not "any skipped assume, a caveat forever".

    ``<=`` where :func:`cone_split_chain` has ``<``. The propagation's
    non-emptiness probe finds a point of the declared set at which every
    assume of the query — including the one no script can state — is
    definitely true, and that certificate is a WHOLE-QUERY answer: it settles
    the emptiness question the per-obligation script cannot reach, so the
    extra solver call is skipped and no caveat is stamped.
    """
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    z = any_array((), "float64", (-10.0, 10.0))
    assume(x <= y)
    assume(y <= z)
    return assert_(x - y <= 0.0)


def satisfiable_assume():
    """THE CONTROL. ``x <= y`` is satisfiable over the declared boxes and the
    assert ``x - y <= 0`` is TRUE at every admitted point.

    This VERIFIED is substantive and must survive the repair untouched. It is
    also the shape a whole-feature withdrawal would break: refusing to
    discharge whenever relational axioms are present passes every emptiness
    test above and fails here.
    """
    x = any_array((), "float64", (0.0, 10.0))
    y = any_array((), "float64", (0.0, 10.0))
    assume(x <= y)
    return assert_(x - y <= 0.0)


def _first_slice(harness):
    q = trace(harness)
    p = propagate(q)
    items = list(slice_unknown_obligations(q, p, interval_env(q)))
    return next(s for s in items if getattr(s, "assumes", None))


def _cone_split_admits_no_point(n=21):
    """GROUND TRUTH for :func:`cone_split_cycle`, computed here, exactly.

    An ``n³`` grid of exact :class:`~fractions.Fraction` points spanning
    ``[-10, 10]³``, counting those satisfying ``x < y ∧ y < z ∧ z < x``. No
    float, no solver, no stelling: the emptiness this file's headline test
    rests on is a fact about the harness, and a fact about the harness must
    not be established by the tool under test. (It is also provable in one
    line — a strict order has no 3-cycle, since the three give ``x < x`` — and
    the grid is the measurement that catches a harness edited into a
    different shape.)
    """
    lo, hi = Fraction(-10), Fraction(10)
    pts = [lo + (hi - lo) * Fraction(i, n - 1) for i in range(n)]
    return sum(
        1
        for x in pts for y in pts for z in pts
        if x < y and y < z and z < x
    )


# ---------------------------------------------------------------------------
# 1. the empty region refuses, in the class the non-relational form uses
# ---------------------------------------------------------------------------


@need_solver
@pytest.mark.parametrize(
    "harness,n_axioms",
    [(mis_declared_bound, 1), (axiom_cycle, 3)],
    ids=["mis-declared-bound", "axiom-cycle"],
)
def test_an_unsatisfiable_forwarded_assume_refuses_instead_of_verifying(
    harness, n_axioms
):
    """Both shapes returned VERIFIED before this repair. Deleting the
    admitted-region check restores that, which is what makes this test
    load-bearing rather than descriptive."""
    with pytest.raises(UnsatisfiableAssumptionError) as e:
        check(harness, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    msg = str(e.value)
    # the SAME sentence the non-relational refusal ends with — one posture,
    # one class, one wording, so a caller who handles that form handles this
    assert "harness defect; nothing was verified" in msg
    assert "declared set as assumed is empty" in msg
    # ... and it says WHICH mechanism decided, and on what
    assert f"the {n_axioms} relational assume(s) forwarded" in msg
    assert "with the negated obligation removed" in msg


def test_the_cone_split_harness_really_does_admit_no_point():
    """The premise of every test below it, measured rather than asserted."""
    assert _cone_split_admits_no_point() == 0


def test_the_cone_split_slice_states_ONE_link_of_the_three():
    """WHY the sat is not an answer, at the emission. The slice carries one
    axiom and quotes a reason for each of the two it cannot state — so the
    script the region check is asked about describes a STRICT RELAXATION of
    the user's precondition, and no model of it is a point of that
    precondition."""
    sl = _first_slice(cone_split_cycle)
    assert len(sl.assumes) == 1
    assert len(sl.assumes_skipped) == 2
    assert all(
        "not in this obligation's backward cone" in r
        for r in sl.assumes_skipped
    )


@need_solver
def test_a_cone_split_empty_region_is_never_stamped_clean():
    """AUDIT B3, THE BLOCKING FINDING. The discharge stands — it is sound over
    the superset the solver ran on — and it may NOT read as substantive: the
    admitted-region check answered ``sat`` about one link of a 3-cycle, which
    settles nothing about the cycle.

    Reverting the one condition in :func:`solvers._region_answer` restores
    ``REGION_INHABITED`` here and reddens every assertion below.
    """
    v = check(
        cone_split_cycle, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT,
    )
    # the claim itself is sound and survives; what changes is what it says
    assert v.status == "VERIFIED"
    (ob,) = v.obligations
    assert ob.status == "discharged"
    assert "MAY BE VACUOUS" in ob.detail
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions
    # and the mechanism is named where it is TRUE — per obligation, because
    # the solver DID decide, over the wrong question
    mech = [n for n in v.notes if solvers.REGION_PARTIAL_MECHANISM in n]
    assert len(mech) == 1, v.notes
    # naming which conjuncts the check never saw, as the withholding does
    assert mech[0].count("[forwarded]") == 2, mech[0]
    assert solvers.REGION_UNDECIDED_MECHANISM not in mech[0]


@need_solver
def test_the_cone_split_control_with_the_cycle_broken_stays_VERIFIED():
    """THE CONTROL THAT MAKES THE REPAIR A REPAIR: same boxes, same assert,
    same skipped assume, satisfiable precondition — still VERIFIED. A repair
    that withdrew the discharge would pass every emptiness test above and be
    worthless."""
    v = check(
        cone_split_chain, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT,
    )
    assert v.status == "VERIFIED"
    (ob,) = v.obligations
    assert ob.status == "discharged"


@need_solver
def test_the_cone_split_cost_is_a_caveat_on_an_INHABITED_region():
    """THE PRICE, PINNED RATHER THAN ARGUED. On the satisfiable cone-split
    chain nothing establishes non-emptiness — the script cannot state ``y<z``
    and the probe grid finds no strict-chain point — so the VERIFIED is
    qualified. Measured over a 48-harness cone-split family: 8 of 28
    inhabited-region VERIFIEDs gain this caveat, 0 change verdict.

    Pinned because it is the cost a future whole-query admitted-region script
    would BUY BACK, and a silent change to it is a change to what a clean
    VERIFIED means."""
    v = check(
        cone_split_chain, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT,
    )
    (ob,) = v.obligations
    assert "MAY BE VACUOUS" in ob.detail
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions


@need_solver
def test_the_whole_query_certificate_still_clears_a_cone_split_run():
    """... and the repair is NOT "a skipped assume, a caveat forever". The
    propagation's non-emptiness probe answers the WHOLE-QUERY question, which
    is the question the per-obligation script cannot reach, so a cone-split
    run it certifies is clean AND pays no extra solver call."""
    q = trace(cone_split_chain_certified)
    p = propagate(q)
    assert p.region_inhabited is True
    v = check(
        cone_split_chain_certified, vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT,
    )
    assert v.status == "VERIFIED"
    (ob,) = v.obligations
    assert "MAY BE VACUOUS" not in ob.detail
    assert not any(
        a.startswith(UNCERTIFIED_PRECONDITION_PREFIX)
        for a in v.stamp.assumptions
    )
    assert all(
        "admitted-region check" not in s.reason
        for s in (v.stamp.solver if isinstance(v.stamp.solver, tuple) else ())
    )


@need_solver
def test_the_inductive_entry_point_stops_calling_a_cone_split_step_clean():
    """The same hole through the public inductive API. Bound obligation #1
    (``0.6(x-y)+0.6 <= 1``) discharges only because ``x < y`` was forwarded,
    and ``x < y < z < x`` admits no state at all — so the step is BOTH
    conditional and unestablished, and the verdict now says both.

    The residual gap is named in the assertion at the end: it is still a
    VERDICT and not a raise. Nothing here can prove the region empty, because
    no single obligation's script ever states more than one link of the cycle.
    """
    def body(state, consts):
        x, y, z = state["x"], state["y"], state["z"]
        assume(x < y)
        assume(y < z)
        assume(z < x)
        return {"x": 0.6 * (x - y) + 0.6, "y": 0.5 * y, "z": 0.5 * z}

    bounds = {k: ((-1.0, 1.0), "float64") for k in ("x", "y", "z")}
    v = check_inductive_step(body, bounds, solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED"
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions
    assert "MAY BE VACUOUS" in v.obligations[1].detail
    # M5's caveat fires too: this bound obligation IS one of the induction's
    assert v.notes[-1].startswith("inductive step CONDITIONAL")
    # ... and the sentence `docs/inductive-step.md` used to end on — "raises
    # rather than returning a verdict" — is still FALSE for this shape.
    assert v.status != "RAISED"


@need_solver
def test_the_control_without_the_assume_is_refuted_not_refused():
    """The refusal is about the PRECONDITION, not about the harness shape:
    the identical query minus the assume returns the honest REFUTED."""
    v = check(
        mis_declared_bound_control,
        vacuity_mode="all",
        solver_timeout_ms=TIMEOUT,
    )
    assert v.status == "REFUTED"


@need_solver
def test_the_relational_refusal_is_the_same_class_as_the_non_relational_one():
    """The asymmetry the audit named, closed: the box form and the relational
    form of one mistake now raise the same exception type."""
    def non_relational():
        dt = any_array((), "float64", (5.0, 10.0))
        assume(dt < 1.0)
        return assert_(dt <= 1.0)

    with pytest.raises(UnsatisfiableAssumptionError):
        check(non_relational, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    with pytest.raises(UnsatisfiableAssumptionError):
        check(
            mis_declared_bound, vacuity_mode="all", solver_timeout_ms=TIMEOUT
        )


@need_solver
def test_the_inductive_step_form_refuses_too():
    """S7'. ``x, y -> (x + y) * 10`` on ``[-1, 1]²`` under contradictory
    assumes returned VERIFIED with "the invariant is preserved by one step" —
    and from ``x = y = 0.5``, inside the invariant, one step gives 10.0."""
    def body(state, consts):
        x, y = state["x"], state["y"]
        assume(x < y)
        assume(y < x)
        return {"x": (x + y) * 10.0, "y": (x + y) * 10.0}

    bounds = {
        "x": ((-1.0, 1.0), "float64"),
        "y": ((-1.0, 1.0), "float64"),
    }
    with pytest.raises(UnsatisfiableAssumptionError):
        check_inductive_step(body, bounds, solver_timeout_ms=TIMEOUT)


@need_solver
def test_the_inductive_control_without_the_assumes_is_refuted():
    def body(state, consts):
        x, y = state["x"], state["y"]
        return {"x": (x + y) * 10.0, "y": (x + y) * 10.0}

    bounds = {
        "x": ((-1.0, 1.0), "float64"),
        "y": ((-1.0, 1.0), "float64"),
    }
    v = check_inductive_step(body, bounds, solver_timeout_ms=TIMEOUT)
    assert v.status == "REFUTED"


# ---------------------------------------------------------------------------
# 2. THE CONTROL: a satisfiable relational assume still VERIFIES, cleanly
# ---------------------------------------------------------------------------


@need_solver
def test_a_satisfiable_relational_assume_still_verifies():
    v = check(
        satisfiable_assume, vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT,
    )
    assert v.status == "VERIFIED"
    assert [o.status for o in v.obligations] == ["discharged"]


@need_solver
def test_a_satisfiable_relational_assume_carries_no_may_be_vacuous_line():
    """Not merely VERIFIED — CLEAN. A repair that stamped "may be vacuous" on
    every relational-assume discharge would leave the disclosure meaningless,
    which is the failure mode the stamp exists to avoid."""
    v = check(
        satisfiable_assume, vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT,
    )
    assert UNCERTIFIED_REGION_ASSUMPTION not in v.stamp.assumptions
    assert not any(
        a.startswith(UNCERTIFIED_PRECONDITION_PREFIX)
        for a in v.stamp.assumptions
    )


@need_solver
def test_a_satisfiable_relational_assume_says_the_verdict_is_conditional():
    """The solver was GIVEN the precondition, so the claim is about the
    assumed region — the same conditionality an interval narrowing already
    stamped, in the same words, for the mechanism that had none (M5's root)."""
    v = check(
        satisfiable_assume, vacuity_mode="inputs-only",
        solver_timeout_ms=TIMEOUT,
    )
    conditional = [
        a for a in v.stamp.assumptions if CONDITIONAL_ON_PRECONDITION in a
    ]
    assert conditional, v.stamp.assumptions
    assert "forwarded relational assume(s)" in conditional[0]
    assert "#0" in conditional[0]


@need_solver
def test_the_certificate_skips_the_extra_call_when_it_settles_the_question():
    """THE HYBRID HALF. The propagation's own probe found a point of the
    declared set satisfying every assume, so the emptiness question is
    already answered and no admitted-region script is emitted. Measured on
    the invocation count, not on a flag: the control pays nothing."""
    q = trace(satisfiable_assume)
    p = propagate(q)
    assert p.region_inhabited is True
    esc = solvers.escalate(q, p, solvers.SolverConfig(timeout_ms=TIMEOUT))
    assert esc.region_uncertified == ()
    # exactly the portfolio's own asks — no admitted-region invocation
    assert all(
        "admitted-region check" not in s.reason for s in esc.invocations
    ), [s.reason for s in esc.invocations]


# ---------------------------------------------------------------------------
# 3. the emission property the whole check rests on
# ---------------------------------------------------------------------------


def test_the_admitted_region_script_is_the_obligation_script_minus_one_line():
    """The comparison is only a comparison if the two texts agree about
    everything else: same declarations, same bounds, same definitions, same
    axioms, same logic, same options. Checked as a LINE DIFFERENCE rather
    than argued from the source, so an emission that grew a second
    divergence would show up here."""
    sl = _first_slice(mis_declared_bound)
    full = smt.emit(sl, "z3", TIMEOUT).text.splitlines()
    region = smt.emit(
        sl, "z3", TIMEOUT, states_obligation=False
    ).text.splitlines()
    # the region script's own header comment is the only ADDED line
    added = [ln for ln in region if ln not in full]
    assert len(added) == 1 and added[0].startswith("; admitted-region check:")
    removed = [ln for ln in full if ln not in region]
    assert len(removed) == 1, removed
    assert removed[0].startswith("(assert (not ")
    # and both still ask the same two questions of the solver
    assert full[-2:] == ["(check-sat)", "(get-model)"]
    assert region[-2:] == ["(check-sat)", "(get-model)"]


def test_the_admitted_region_script_keeps_every_forwarded_axiom():
    sl = _first_slice(axiom_cycle)
    region = smt.emit(sl, "z3", TIMEOUT, states_obligation=False)
    assert region.relational_assumes_emitted == 3
    # `(assert (< ` with the space is the STRICT comparison the three axioms
    # are; the declared bounds emit `(assert (<= …))` and must not be counted
    assert region.text.count("(assert (< ") == 3


def test_the_default_emission_is_byte_identical_to_before_the_flag():
    """Every existing caller emits the same script it emitted before: the
    flag defaults to the pre-existing behaviour and is keyword-only."""
    sl = _first_slice(satisfiable_assume)
    assert (
        smt.emit(sl, "z3", TIMEOUT).text
        == smt.emit(sl, "z3", TIMEOUT, states_obligation=True).text
    )
    p = inspect.signature(smt.emit).parameters["states_obligation"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True


# ---------------------------------------------------------------------------
# 4. the decision rule, as a pure function
# ---------------------------------------------------------------------------


def test_the_region_rule_reads_unsat_as_empty_and_sat_as_inhabited():
    assert _region_answer(
        frozenset({"unsat"}), accounts_for_every_assume=True
    ) == REGION_EMPTY
    assert _region_answer(
        frozenset({"sat"}), accounts_for_every_assume=True
    ) == REGION_INHABITED


def test_an_undecided_region_script_certifies_nothing():
    """Timeouts, `unknown`, a transport that never ran — none of them is an
    answer, and the honest reading of "nobody decided" is not "inhabited"."""
    for answers in (
        frozenset(),
        frozenset({"unknown"}),
        frozenset({"timeout"}),
        frozenset({"failed", "not-run"}),
    ):
        assert _region_answer(
            answers, accounts_for_every_assume=True
        ) == REGION_UNCERTIFIED


def test_a_lone_definitive_answer_still_decides():
    """A degraded portfolio does not silence the check: one backend's own
    ``unsat`` on its own script is the reading the refusal rests on, and it
    needs nothing believed across solvers."""
    assert _region_answer(
        frozenset({"unsat", "unknown"}), accounts_for_every_assume=True
    ) == REGION_EMPTY
    assert _region_answer(
        frozenset({"sat", "timeout"}), accounts_for_every_assume=True
    ) == REGION_INHABITED


def test_sat_over_a_PARTIAL_axiom_set_certifies_nothing():
    """AUDIT B3, as a pure function. A model of a relaxation is a point of the
    relaxation and of nothing tighter, so the honest status is the one that
    discloses — never the one that stamps clean."""
    for answers in (frozenset({"sat"}), frozenset({"sat", "timeout"})):
        assert _region_answer(
            answers, accounts_for_every_assume=False
        ) == REGION_UNCERTIFIED


def test_unsat_is_EMPTY_however_partial_the_axiom_set():
    """The other direction, and it is a different argument: the script's
    axioms are a SUBSET of the query's assumes, so a relaxation with no point
    at all proves the tighter set has none. Weakening this half would turn a
    detected harness defect back into a silent one."""
    assert _region_answer(
        frozenset({"unsat"}), accounts_for_every_assume=False
    ) == REGION_EMPTY
    assert _region_answer(
        frozenset({"unsat", "unknown"}), accounts_for_every_assume=False
    ) == REGION_EMPTY


def test_the_accounting_is_a_required_keyword():
    """No default, because the two directions rest on different arguments and
    a default would pick one of them for a caller who did not think about it.
    The measured defect is exactly what picking the permissive one looks
    like."""
    p = inspect.signature(_region_answer).parameters["accounts_for_every_assume"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty


def test_the_four_region_statuses_are_distinct():
    assert len({
        REGION_EMPTY, REGION_INHABITED, REGION_UNCERTIFIED, REGION_NOT_ASKED
    }) == 4


# ---------------------------------------------------------------------------
# 5. disclosure: the stamped line, and the vacuity sentence it qualifies
# ---------------------------------------------------------------------------


def test_the_may_be_vacuous_line_shares_the_uncertified_prefix():
    """The join the qualification is keyed on. Three mechanisms can leave a
    precondition unsettled and a consumer must see all of them, so they share
    a PREFIX rather than a membership list nobody remembers to extend."""
    assert UNCERTIFIED_REGION_ASSUMPTION.startswith(
        UNCERTIFIED_PRECONDITION_PREFIX
    )


def _force_unsettled_region(monkeypatch):
    """Drive the ONE route that survives the refusal.

    An EMPTY region never becomes a verdict — it raises — so the only way a
    VERIFIED can still rest on an unsettled precondition is a region script
    nobody decided. Both halves of that are forced explicitly rather than
    hunted for in a harness whose incidental shape happens to produce them:

    * the propagation's non-emptiness certificate is cleared, so the cheap
      hybrid path does not answer the question first (it fires on the natural
      satisfiable harnesses, which is the whole point of it);
    * the ADMITTED-REGION script — and only it; the obligation's own script
      is untouched, and the assertion below checks the discharge still
      happened — comes back ``unknown``.

    Forcing is honest here because the property under test is a DISCLOSURE
    rule keyed on a state, not the probability of reaching that state.
    """
    import dataclasses

    from stelling import propagate as _prop

    real_propagate = _prop.propagate

    def cleared(*a, **kw):
        return dataclasses.replace(
            real_propagate(*a, **kw), region_inhabited=False
        )

    monkeypatch.setattr(_prop, "propagate", cleared)

    real_run = solvers._Backend.run

    def run(self, ledger, script_text, wall_s):
        if "admitted-region check" in script_text:
            # the spawn is still counted: `run` IS the transport-entry
            # boundary and the provenance gate compares its count against the
            # stamps. A fake that skipped it would be testing a divergence
            # this repair does not produce.
            ledger.spawns += 1
            return solvers._RawResult(answer="unknown", version=self.version())
        return real_run(self, ledger, script_text, wall_s)

    monkeypatch.setattr(solvers._Backend, "run", run)
    return cleared


@need_solver
def test_an_undecided_region_discharges_but_says_it_may_be_vacuous(monkeypatch):
    """The discharge STANDS — it is sound, every admitted point satisfies the
    obligation — and it stops being clean. That is exactly SOUNDNESS.md's
    constraining-assume policy ("uncertified VERIFIEDs carry a stamped
    may-be-vacuous line"), which audit 0.2.0 S7 measured absent on this
    path."""
    cleared = _force_unsettled_region(monkeypatch)
    q = trace(satisfiable_assume)
    p = cleared(q)
    assert p.region_inhabited is False
    esc = solvers.escalate(q, p, solvers.SolverConfig(timeout_ms=TIMEOUT))
    assert esc.region_uncertified == (0,)
    (record,) = esc.records
    assert record.outcome == solvers.OB_DISCHARGED
    assert "MAY BE VACUOUS" in record.detail
    assert any("admitted-region check" in s.reason for s in esc.invocations), [
        s.reason for s in esc.invocations
    ]
    v = solvers.make_solver_verdict(
        q, p, esc,
        stelling_version="t", jax_version="t", precision_config="t",
    )
    assert v.status == "VERIFIED"
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions


@need_solver
def test_the_vacuity_sentence_stops_claiming_substantiveness_when_unsettled(
    monkeypatch
):
    """Audit 0.2.0 S7's stamp half, at the surface a reader reads.

    At ``vacuity_mode="all"`` the stamp said "no obligation discharges with
    the declared bounds widened — this VERIFIED was not re-derivable without
    the declared envelope", i.e. SUBSTANTIVE, on a run whose VERIFIED rested
    on a precondition no declared point satisfied. The measurement was real
    (widening makes the assume satisfiable again, so the negated obligation
    becomes sat); the reading it invites was not.
    """
    _force_unsettled_region(monkeypatch)
    v = check(satisfiable_assume, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED"
    assert UNCERTIFIED_REGION_ASSUMPTION in v.stamp.assumptions
    vac = [a for a in v.stamp.assumptions if a.startswith("vacuity ")]
    assert len(vac) == 1, v.stamp.assumptions
    assert "WHAT THIS MEASUREMENT DOES NOT SAY" in vac[0]
    assert "may be vacuously true of an empty precondition" in vac[0]


@need_solver
def test_a_settled_region_leaves_the_vacuity_sentence_unqualified():
    """The counterpart, and the reason the qualification means anything: on
    the same harness with the certificate in place, the sentence stands as
    written."""
    v = check(satisfiable_assume, vacuity_mode="all", solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED"
    vac = [a for a in v.stamp.assumptions if a.startswith("vacuity ")]
    assert len(vac) == 1, v.stamp.assumptions
    assert "WHAT THIS MEASUREMENT DOES NOT SAY" not in vac[0]


# ---------------------------------------------------------------------------
# 6. the inductive-step over-claims
# ---------------------------------------------------------------------------


def test_an_assume_in_an_inductive_body_makes_the_note_say_conditional():
    """M5. ``x -> 1.5x`` on ``[-1, 1]`` under ``|x| <= 0.5`` is VERIFIED, and
    iterating from the ADMITTED ``x = 0.4`` leaves ``[-1, 1]`` at step 3:
    0.4, 0.6, 0.9, 1.35. The claim is true and it is not the inductive step,
    because the successor need not re-enter the assumed sub-region."""
    def body(state, consts):
        x = state["x"]
        assume(x <= 0.5)
        assume(x >= -0.5)
        return {"x": 1.5 * x}

    v = check_inductive_step(body, {"x": ((-1.0, 1.0), "float64")})
    assert v.status == "VERIFIED"
    note = v.notes[-1]
    assert note.startswith("inductive step CONDITIONAL — NOT the inductive step")
    assert "need not re-enter the assumed sub-region" in note
    # the fix is named, not implied
    assert "state_bounds" in note
    # and the arithmetic is real: from the admitted 0.4, step 3 escapes
    xs = [0.4]
    for _ in range(4):
        xs.append(xs[-1] * 1.5)
    assert xs[3] > 1.0


def test_an_inductive_body_without_assumes_keeps_the_unconditional_note():
    """The caveat must fire on the runs it is about and on no others: a
    qualification printed on every VERIFIED is a qualification nobody
    reads."""
    def body(state, consts):
        return {"x": 0.5 * state["x"]}

    v = check_inductive_step(body, {"x": ((-1.0, 1.0), "float64")})
    assert v.status == "VERIFIED"
    note = v.notes[-1]
    assert note.startswith("inductive step: all state variables stay within")
    assert "CONDITIONAL" not in note


def test_the_refuted_note_names_the_variable_that_actually_escaped():
    """M4. Obligations were mapped to state variables POSITIONALLY, and the
    harness appends its bound checks AFTER tracing ``body`` — so any
    ``assert_`` the body declares shifts every index. Measured: body
    ``{"a": a + 10, "b": b * 0.5}`` on ``[-1, 1]²`` with one assert in the
    body named "b (below lower bound)" while ``b*0.5 ∈ [-0.5, 0.5]`` never
    escapes and ``a + 10 ∈ [9, 11]`` escapes above."""
    def body(state, consts):
        a, b = state["a"], state["b"]
        assert_(a <= 100.0)  # the body's own obligation: index 0
        return {"a": a + 10.0, "b": b * 0.5}

    bounds = {
        "a": ((-1.0, 1.0), "float64"),
        "b": ((-1.0, 1.0), "float64"),
    }
    v = check_inductive_step(body, bounds)
    assert v.status == "REFUTED"
    note = v.notes[-1]
    assert "Escaped: a (above upper bound)" in note
    assert "b (" not in note


def test_the_positional_map_is_unshifted_without_a_body_assert():
    """The offset is DERIVED, so the case it was always right for stays
    right: with no body obligation it is zero and nothing moves."""
    def body(state, consts):
        a, b = state["a"], state["b"]
        return {"a": a + 10.0, "b": b * 0.5}

    bounds = {
        "a": ((-1.0, 1.0), "float64"),
        "b": ((-1.0, 1.0), "float64"),
    }
    v = check_inductive_step(body, bounds)
    assert v.status == "REFUTED"
    assert "Escaped: a (above upper bound)" in v.notes[-1]


# ---------------------------------------------------------------------------
# 7. AUDIT B3: the SCOPE of a conditionality line
#
# Both mechanisms write `CONDITIONAL_ON_PRECONDITION` and they are scoped
# differently. An interval NARROWING moves the boxes every obligation is
# judged over, so its line is whole-query and names no obligation. A FORWARDED
# relational axiom reaches only the obligations whose scripts stated it, and
# says which. Both readers asked the whole-query question of both kinds.
# ---------------------------------------------------------------------------


def test_the_forwarded_line_names_its_obligations_and_the_reader_parses_them():
    """The round trip, pinned: producer and reader agree on the idiom, so a
    reworded scope cannot silently degrade every scoped read to whole-query.
    """
    line = relational_assume_assumption((1, 3))
    assert conditional_on_precondition([line], {1})
    assert conditional_on_precondition([line], {3})
    assert conditional_on_precondition([line], {0, 3})
    assert not conditional_on_precondition([line], {0})
    assert not conditional_on_precondition([line], {2, 4})
    assert not conditional_on_precondition([line], set())


def test_a_whole_query_line_bears_on_every_obligation():
    """The narrowing half, which had no scope and needs none: it moved the
    boxes, so every obligation of the run was judged over the narrowed set."""
    narrowing = (
        f"constrained assume at foo.py:1 (h): {CONDITIONAL_ON_PRECONDITION} "
        f"— narrowed var 3 to [0, 1]"
    )
    assert conditional_on_precondition([narrowing], {0})
    assert conditional_on_precondition([narrowing], {7})
    # ... and asking about no obligation is answered honestly, not by the
    # accident of a non-empty intersection
    assert not conditional_on_precondition([narrowing], set())


def test_an_unparseable_scope_falls_back_to_whole_query():
    """The failure direction of the parse is OVER-disclosure. A line whose
    scope cannot be read qualifies everything, which costs a caveat; reading
    it as scoped-to-nothing would drop a real one."""
    mangled = (
        f"forwarded relational assume(s) on obligations one and three: "
        f"{CONDITIONAL_ON_PRECONDITION} — ..."
    )
    assert conditional_on_precondition([mangled], {0})


def test_a_line_without_the_phrase_is_not_a_conditionality_line():
    assert not conditional_on_precondition(
        ["forwarded relational assume(s) on obligation(s) #0: something else"],
        {0},
    )


@need_solver
def test_an_interval_refutation_is_not_conditional_on_a_forwarded_axiom():
    """AUDIT B3, CONSUMER (a). One interval-refuted obligation, one escalated
    obligation with a forwarded axiom.

    Both clauses of the conditional wording were FALSE on this run: nothing
    narrowed (a relational assume is inert in the interval domain), and assert
    #0's own detail line — printed four lines below the sentence — says it was
    judged over the full declared box. A forwarded axiom lives in a SCRIPT,
    and a script exists only where the interval domain gave up, so it can
    never bear on a `violated-over-set`.
    """
    def mixed():
        x = any_array((), "float64", (0.0, 10.0))
        y = any_array((), "float64", (0.0, 10.0))
        assume(x <= y)
        assert_(x >= 20.0)            # interval-refuted over the declared box
        return assert_(x - y <= 0.0)  # escalates, and states the axiom

    v = check(mixed, vacuity_mode="inputs-only", solver_timeout_ms=TIMEOUT)
    assert v.status == "REFUTED"
    assert [o.status for o in v.obligations] == [
        "violated-over-set", "discharged"
    ]
    # the conditionality is real and stays stamped — it is just not about
    # the obligation the set-level sentence describes
    assert any(
        CONDITIONAL_ON_PRECONDITION in a for a in v.stamp.assumptions
    ), v.stamp.assumptions
    render = v.render()
    assert "set-level, conditional" not in render
    assert "not invariant as stated" in render
    # the sentence and the detail line four rows down now agree
    assert "over the declared box" in v.obligations[0].detail
    assert "precondition-narrowed set" not in render


@need_solver
def test_the_inductive_note_is_unconditional_when_the_BOUND_obligations_are():
    """AUDIT B3, CONSUMER (b). ``{x, y} -> {0.5x, 0.5y}`` on ``[-1, 1]²`` with
    ``assume(x < y)`` carried only into a body-assert's slice.

    The four bound obligations have single-variable cones, state no axiom, and
    were judged over the full declared box: ``|0.5·t| <= 0.5 <= 1`` everywhere,
    so the induction closes UNCONDITIONALLY. The note said the opposite — "NOT
    the inductive step … the invariant does NOT follow for all iterations" —
    which under-reports a real result on the strength of an obligation the
    sentence is not about.
    """
    def body(state, consts):
        x, y = state["x"], state["y"]
        assume(x < y)
        assert_(x - y <= 0.0)  # the body's own claim: index 0, conditional
        return {"x": 0.5 * x, "y": 0.5 * y}

    bounds = {
        "x": ((-1.0, 1.0), "float64"),
        "y": ((-1.0, 1.0), "float64"),
    }
    v = check_inductive_step(body, bounds, solver_timeout_ms=TIMEOUT)
    assert v.status == "VERIFIED"
    # obligation 0 is the body's, 1..4 are the harness's bound checks
    assert [o.status for o in v.obligations] == ["discharged"] * 5
    assert "solver escalation" in v.obligations[0].detail
    assert all(
        "definitely true" in o.detail for o in v.obligations[1:]
    ), [o.detail for o in v.obligations]
    # the conditionality is stamped, scoped to the obligation it reached
    assert relational_assume_assumption((0,)) in v.stamp.assumptions
    note = v.notes[-1]
    assert note.startswith("inductive step: all state variables stay within")
    assert "CONDITIONAL" not in note
