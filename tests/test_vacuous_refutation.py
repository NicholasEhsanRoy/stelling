# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""REFUTED over a superset of the assumed region, on the three paths that
still emitted it.

`tests/test_dropped_assume.py` pins F7's no-op half for an assume that
dropped WHOLLY. Three paths reached a definite REFUTED without ever passing
through it.

**A — a drop beside a narrowing.** `_assume_constrain` routes on whether
ANY conjunct narrowed. `assume((x >= -1.) & jnp.all(x >= 2.))` narrows on
the first conjunct — which is the DECLARED LOWER BOUND and narrows nothing,
the note says "already within the assumed region" — and drops the second.
Measured on `9efea6f` over `x ∈ [-1,1]^3` asserting `x > 5.`: **REFUTED,
witnesses=()**, over an assumed region that is EMPTY. Delete the redundant
conjunct and the same query returns UNKNOWN.

**B — the affine refinement.** It declines wholly on
`coverage.constrained`, which a DROPPED assume never raises. Measured on
`9efea6f`: `assume(jnp.all(x >= 2.))` asserting `x >= 5.` returns UNKNOWN
at `refine=None` (the interval leg judged the violation and withheld it)
and **REFUTED at `refine="affine"`** — the same violation re-minted from
the same declared boxes.

**D — a branch-scoped unsatisfiable assume.** Inside a possibly-untaken
`lax.cond` branch, `_unsatisfiable` must NOT raise: the assume is
branch-scoped and the other branch is real (audit F2). It degrades to a
branch-local vacuity note, appending to `vacuous` and NOT to `dropped` —
and the whole-drop guard read `if dropped or not vacuous:`, which is then
FALSE, so the note and the flags were gated together and NEITHER flag was
set. Measured at `9efea6f` AND at the branch tip `3afbf01`, `x ∈ [-1,1]^3`,
`cond(x[0] > 0, yes, no)` with `yes: assume(v >= 2.); assert_(v > 5.)`:
**REFUTED, witnesses=()** — identical to the same query with the assume
deleted, while the same assume at top level RAISES
`UnsatisfiableAssumptionError`. The assume changed nothing at all, and the
run's own note said "obligations in this branch MAY BE VACUOUS under the
branch's precondition" beside the word REFUTED. All three detection sites
reach it (empty meet, strict-boundary collapse, definitely-false constant
comparison). The mixed path's `if restricting or vacuous:` already had the
rule; the `else:` path was edited around and never received it.

All three are wrong for one reason: the assumed region is EMPTY, so the
conditional claim is vacuously TRUE. A definite violation over a superset
is a refutation only if the assumed region is non-empty, and a dropped
conjunct is exactly the part of the precondition whose satisfiability was
never established.

NOT fixed here, deliberately: an `assume` traced AFTER the `assert_` it
should constrain. That is a question about what an `assume` SCOPES OVER,
not about superset judging. The principal has since RULED it QUERY-scoped;
implementing that uniformly is a separate change with its own
pre-registration. This file only pins the current behaviour, on BOTH legs
— and the two legs do not agree: the affine guard keys on
`propagation.assume_dropped` (a whole-run flag, so query-scoped) while the
interval withhold reads `uncertified` at assert time (order-scoped). See
the C section.
"""
from __future__ import annotations

import dataclasses

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import affine as A  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume  # noqa: E402
from stelling.preconditions import check  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


# -- the harnesses -----------------------------------------------------------
#
# Every "EMPTY" comment below is an oracle result, not an argument: 50 000
# uniform samples over the declared box plus every corner plus a 21^3 grid
# satisfy the assume 0 times (scratchpad/PREREG_REF1.md).

def _plain_empty():
    """No redundant conjunct. Withheld even before this branch."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume(jnp.all(x >= 2.0))                    # region EMPTY
    return (assert_(x > 5.0),)


def _redundant_empty():
    """THE REPRODUCER. `x >= -1.` IS the declared lower bound."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume((x >= -1.0) & jnp.all(x >= 2.0))      # region EMPTY
    return (assert_(x > 5.0),)


def _jointly_empty():
    """Neither conjunct is definitely false ALONE; together they are empty.

    The discriminating case between "empty assumed region" and
    "definitely-FALSE dropped conjunct" being one defect or two.
    """
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume((x >= 0.5) & jnp.all(x <= 0.2))       # each satisfiable; region EMPTY
    return (assert_(x < -0.5),)


def _mixed_nonempty():
    """A genuinely narrowing dropped conjunct over a NON-empty region.

    REFUTED here is CORRECT — and it is withheld anyway, because nothing at
    the interval level establishes the region is non-empty. The cost, pinned
    so it cannot be lost silently.
    """
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume((x >= 0.0) & jnp.all(x >= 0.5))       # region [0.5, 1]^3, NON-empty
    return (assert_(x > 5.0),)


def _mixed_verified():
    """One-sided: a discharge over a superset still discharges."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume((x >= 0.0) & jnp.all(x >= 0.5))
    return (assert_(x > -5.0),)


def _harmless_relational():
    """The dropped conjunct is RELATIONAL and definitely TRUE: a ∈ [0,1],
    b ∈ [5,6], so `a <= b` holds at every point of the declared boxes.

    It restricts nothing, so its absence widens nothing, so the refutation
    is genuine — region [0,1]^3, non-empty; `a > 5.` false throughout.
    """
    a = any_array((3,), "float64", (0.0, 1.0))
    b = any_array((3,), "float64", (5.0, 6.0))
    assume((a >= 0.0) & (a <= b))
    return (assert_(a > 5.0),)


def _restricting_relational():
    """Same shape, boxes OVERLAPPING: `a <= b` is indeterminate, so the drop
    really does widen and the discriminant must NOT fire."""
    a = any_array((3,), "float64", (0.0, 10.0))
    b = any_array((3,), "float64", (5.0, 6.0))
    assume((a >= 0.0) & (a <= b))
    return (assert_(a > 50.0),)


def _affine_reachable_empty():
    """`>=` — the affine v1 rule covers CLOSED half-spaces; `>` declines,
    which is why the affine leg was dark to the `>` reproducer above."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume(jnp.all(x >= 2.0))                    # region EMPTY, assume DROPPED
    return (assert_(x >= 5.0),)


def _affine_reachable_verified():
    """Interval-undecided, affine-discharged: `x - x` concretizes to
    [-2, 2] as a box and to exactly 0 as an affine form, so the refinement
    is the leg that decides it — which is what makes this a test of the
    refinement's disposition rather than of the interval leg's."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    assume(jnp.all(x >= 2.0))                    # DROPPED
    return (assert_(x - x >= -0.5),)


def _assert_before_assume():
    """Decided at the INTERVAL leg (`x > 5.` is definitely false on the box)."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    o = assert_(x > 5.0)
    assume(jnp.all(x >= 2.0))
    return (o,)


def _assert_before_assume_affine():
    """The same ordering, decided at the AFFINE leg instead.

    `x - x` is [-2, 2] as a box and exactly 0 as an affine form, so the
    interval leg cannot decide it and the refinement is what judges it.
    Without this harness the ordering pin runs entirely on the interval
    leg and is blind to the half that actually moved on this branch.
    """
    x = any_array((3,), "float64", (-1.0, 1.0))
    o = assert_(x - x >= 0.5)
    assume(jnp.all(x >= 2.0))
    return (o,)


def _affine_control_no_assume():
    """The same obligation with no assume at all — the affine leg's own
    control, so a withhold cannot be confused with the leg going dark."""
    x = any_array((3,), "float64", (-1.0, 1.0))
    return (assert_(x - x >= 0.5),)


# -- D harnesses: the assume lives inside a possibly-untaken cond branch -----
#
# `lax.cond(x[0] > 0, yes, no, x)`. Oracle over 59 269 points (50 000 uniform
# in [-1,1]^3 + all 8 corners + a 21^3 grid): 29 337 take the `yes` branch,
# and of those the number satisfying the branch's assume is **0** for every
# construction below — the branch precondition is EMPTY, so every obligation
# inside `yes` is vacuously true and REFUTED is wrong. The control's assert
# is violated at all 29 337, so its REFUTED is right.


def _in_branch(assume_body, claim=lambda v: v > 5.0):
    """`yes` carries the assume and the obligation; `no` is inert filler."""
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))

        def yes(v):
            if assume_body is not None:
                assume_body(v)
            return assert_(claim(v))

        def no(v):
            return assert_(v > -5.0)

        return (jax.lax.cond(x[0] > 0.0, yes, no, x),)
    return h


# each of the three `_unsatisfiable` detection sites, reached under a branch
_branch_empty_meet = _in_branch(lambda v: assume(v >= 2.0))
_branch_strict_collapse = _in_branch(lambda v: assume(v > 1.0))
_branch_no_assume = _in_branch(None)                      # the control
_branch_verified = _in_branch(lambda v: assume(v >= 2.0), lambda v: v > -5.0)


def _branch_const_false():
    """The third site: a definitely-false CONSTANT comparison (both sides
    point intervals), which needs a degenerate declaration to reach."""
    k = any_array((), "float64", (1.0, 1.0))     # point box [1, 1]
    x = any_array((3,), "float64", (-1.0, 1.0))

    def yes(v):
        assume(k >= 2.0)                          # point vs point: def. FALSE
        return assert_(v > 5.0)

    def no(v):
        return assert_(v > -5.0)

    return (jax.lax.cond(x[0] > 0.0, yes, no, x),)


def _branch_mixed_vacuous():
    """narrowed NON-empty, `dropped` EMPTY, `vacuous` non-empty.

    The only shape that reaches `or vacuous` in `if restricting or vacuous:`
    on the mixed path — full-suite mutation found that operand uncovered.
    `v >= -1.` is the declared lower bound (narrows nothing, but routes the
    assume to the `if narrowed:` arm); `v > 1.` collapses onto the boundary.
    """
    x = any_array((3,), "float64", (-1.0, 1.0))

    def yes(v):
        assume((v >= -1.0) & (v > 1.0))           # branch region EMPTY
        return assert_(v > 5.0)

    def no(v):
        return assert_(v > -5.0)

    return (jax.lax.cond(x[0] > 0.0, yes, no, x),)


def _run(h, **kw):
    return check(h, vacuity_mode="inputs-only", **kw)


def _prop(h, **kw):
    return P.propagate(transcribe(jax.make_jaxpr(h)()), **kw)


# -- A: the drop beside a narrowing ------------------------------------------

def test_the_reproducer_no_longer_refutes_an_empty_region():
    v = _run(_redundant_empty)
    assert v.status == "UNKNOWN", (
        "x ∈ [-1,1] cannot satisfy x >= 2, so the assumed region is EMPTY "
        "and the implication is VACUOUSLY TRUE; REFUTED tells the author "
        "their correct program is broken"
    )
    assert v.witnesses == ()


def test_the_redundant_conjunct_no_longer_changes_the_verdict():
    """The whole defect in one assertion: adding a conjunct that narrows
    NOTHING must not move the verdict."""
    assert _run(_plain_empty).status == _run(_redundant_empty).status


def test_the_conjunct_really_is_redundant_by_the_propagator_s_own_note():
    """Guards the test above against being satisfied for the wrong reason:
    if `x >= -1.` ever started narrowing, the two harnesses would agree
    while the premise of the reproducer had quietly evaporated."""
    notes = "\n".join(_prop(_redundant_empty).notes)
    assert "already within the assumed region" in notes
    assert "DROPPED" in notes


def test_the_withhold_is_disclosed_not_silent():
    v = _run(_redundant_empty)
    assert any("UNCERTIFIED" in n for n in v.notes)
    assert any("WITHHELD from REFUTED" in n for n in v.notes)


def test_a_jointly_empty_region_moves_WITH_the_definitely_false_conjunct():
    """HANDOFF5 §15.1 lists "empty assumed region" and "definitely-FALSE
    dropped conjunct" as two defects. They are one: neither conjunct of
    `_jointly_empty` is definitely false on its own, and it withholds by the
    same edit at the same site."""
    assert _run(_jointly_empty).status == "UNKNOWN"
    assert _run(_redundant_empty).status == "UNKNOWN"


def test_a_mixed_drop_marks_the_propagation_the_way_a_whole_drop_does():
    """The marking, not just the verdict — `assume_dropped` is what the
    solver leg and the affine leg read, and on `9efea6f` it was False here
    while the note already said "a superset"."""
    p = _prop(_redundant_empty)
    assert p.assume_dropped is True
    assert p.coverage.constrained == 1
    assert p.coverage.dropped_conjuncts == 1


def test_a_discharge_under_a_mixed_drop_is_STILL_RENDERED():
    """One-sided, exactly as the whole-drop path is. Suppressing VERIFIED
    over a superset would be over-firing: it implies VERIFIED over the
    subset."""
    assert _run(_mixed_verified).status == "VERIFIED"


def test_the_cost_of_the_withhold_is_pinned_where_it_falls():
    """A legitimate REFUTED that this fix gives up. Nothing at the interval
    level establishes that `x >= 0.5` is satisfiable, so the refutation is
    withheld even though the region IS non-empty. Recorded as a cost, not
    hidden: if a later change recovers it soundly, this test says so."""
    assert _run(_mixed_nonempty).status == "UNKNOWN"


# -- the certainly-true discriminant, pinned on BOTH faces -------------------

def test_a_definitely_true_dropped_conjunct_still_refutes():
    """`a <= b` over disjoint boxes restricts nothing, so dropping it
    introduced no superset and the refutation is genuine."""
    assert _run(_harmless_relational).status == "REFUTED"
    p = _prop(_harmless_relational)
    assert p.coverage.dropped_conjuncts == 1, "the conjunct must really drop"
    assert p.assume_dropped is False, "and must NOT mark the run uncertified"


def test_the_drop_note_stops_saying_superset_when_it_is_not_one():
    """The disclosure has to distinguish the two, or a reader cannot tell
    the drop that kept the refutation from the drop that withheld it."""
    harmless = "\n".join(_prop(_harmless_relational).notes)
    assert "NOT a superset" in harmless
    restricting = "\n".join(_prop(_restricting_relational).notes)
    assert "— a superset" in restricting
    assert "NOT a superset" not in restricting


def test_an_indeterminate_dropped_conjunct_does_NOT():
    """The other face. Without this the discriminant could return True
    unconditionally and every test above still pass."""
    assert _run(_restricting_relational).status == "UNKNOWN"
    assert _prop(_restricting_relational).assume_dropped is True


def test_a_non_bool_and_is_refused_as_a_conjunction_and_says_so():
    """Replaces `test_the_discriminant_refuses_a_non_bool_operand`, which
    could not fail.

    That test named `_conjunct_certainly_true`'s dtype gate and then
    asserted `assume_dropped is True` — which on this harness comes from
    the whole-drop `else:` path, and that path never consults `harmless`
    at all. It stayed green under `_conjunct_certainly_true -> return
    True` and under deleting the dtype gate outright.

    The refusal that IS reachable, and IS load-bearing, lives one level up
    in `_classify_assumed_pred`: jax's `&` is bitwise, so a non-bool `and`
    is not the logical connective and must not be recursed into as one.
    Pinned by its disclosed REASON, because that is its whole effect —
    the verdict is withheld either way.

    The dtype gate inside `_conjunct_certainly_true` is unreachable in
    effect, and the second assertion here is why: jax promotes `bool &
    int32` to an int32 `and`, so a non-bool operand promotes the WHOLE
    tree and is refused at the top, never reaching the recursion where
    `harmless` is filled. If jax's promotion ever changed, this assertion
    is what would say so.
    """
    def h():
        i = any_array((3,), "int32", (1, 1))
        assume(i & i)                 # bit arithmetic on non-bool operands
        return (assert_(i > 5),)
    note = next(n for n in _prop(h).notes if "constraint DROPPED" in n)
    assert "'and' on non-bool operands is bit arithmetic, not conjunction" \
        in note

    def mixed():
        x = any_array((3,), "float64", (-1.0, 1.0))
        i = any_array((3,), "int32", (1, 1))
        assume((x >= 0.5) & (i & i))
        return (assert_(x > 5.0),)
    cj = transcribe(jax.make_jaxpr(mixed)())
    dtypes = [
        [v.aval.dtype for v in e.invars]
        for e in cj.jaxpr.eqns if e.primitive == "and"
    ]
    assert dtypes and all("bool" not in d for d in dtypes), (
        f"a non-bool operand must promote the whole `and` tree, which is "
        f"what makes _conjunct_certainly_true's dtype gate unreachable in "
        f"effect; jax produced {dtypes}"
    )
    assert _prop(mixed).coverage.constrained == 0


@pytest.mark.parametrize("semantics", ["real", "ieee"])
def test_both_semantics_modes_withhold_the_mixed_drop(semantics):
    """add_any's lesson, applied here: real-mode-only is a scope, not a
    pass."""
    p = _prop(_redundant_empty, semantics=semantics)
    assert p.assume_dropped is True
    assert all(o.status != "violated-over-set" for o in p.obligations)


# -- B: the affine re-mint ---------------------------------------------------

def test_the_affine_refinement_does_not_re_mint_a_withheld_violation():
    assert _run(_affine_reachable_empty, refine=None).status == "UNKNOWN"
    assert _run(_affine_reachable_empty, refine="affine").status == "UNKNOWN"


def test_the_affine_refusal_is_load_bearing_on_its_own():
    """The split mutation `test_dropped_assume` uses for the solver leg.
    Neutralise the marking at the AFFINE boundary alone: the wrong REFUTED
    must come back, or this file proves nothing about the affine half."""
    cj = transcribe(jax.make_jaxpr(_affine_reachable_empty)())
    p = P.propagate(cj)
    assert any(o.status == "unknown" for o in p.obligations)

    unmarked = dataclasses.replace(p, assume_dropped=False)
    off, _ = A.refine_propagation(cj, unmarked)
    assert any(o.status == "violated-over-set" for o in off.obligations), (
        "with the affine-side marking off the defect must reappear; if it "
        "does not, the interval half is masking it"
    )

    on, _ = A.refine_propagation(cj, p)
    assert all(o.status != "violated-over-set" for o in on.obligations)


def test_the_affine_refusal_is_ONE_SIDED():
    """A discharge the refinement reaches over a superset still implies the
    discharge over the intended set. `solvers.py` tried declining wholly on
    its own leg and reverted it; the affine leg must not repeat that."""
    _, rep = A.refine_propagation(
        transcribe(jax.make_jaxpr(_affine_reachable_verified)()),
        P.propagate(transcribe(jax.make_jaxpr(_affine_reachable_verified)())),
    )
    assert rep.discharged, (
        "the refinement must still be allowed to DISCHARGE under a dropped "
        "assume — only violations are withheld"
    )
    assert _run(_affine_reachable_verified, refine="affine").status == "VERIFIED"


def test_the_affine_withhold_says_why():
    _, rep = A.refine_propagation(
        transcribe(jax.make_jaxpr(_affine_reachable_empty)()),
        P.propagate(transcribe(jax.make_jaxpr(_affine_reachable_empty)())),
    )
    v = _run(_affine_reachable_empty, refine="affine")
    assert any("WITHHELD from REFUTED" in n for n in v.notes)
    assert rep.violated == ()


def test_an_unconstrained_refutation_is_untouched_by_either_guard():
    """Don't fix the bad path by breaking the good one: no assume at all,
    both legs must still refute."""
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return (assert_(x >= 5.0),)
    assert _run(h, refine=None).status == "REFUTED"
    assert _run(h, refine="affine").status == "REFUTED"


# -- D: the branch-scoped unsatisfiable assume -------------------------------

_DETECTORS = [
    ("empty-meet", _branch_empty_meet),
    ("strict-boundary-collapse", _branch_strict_collapse),
    ("definitely-false-constant", _branch_const_false),
]
_DETECTOR_IDS = [n for n, _ in _DETECTORS]


@pytest.mark.parametrize("name,h", _DETECTORS, ids=_DETECTOR_IDS)
def test_a_branch_scoped_unsatisfiable_assume_no_longer_refutes(name, h):
    """All three `_unsatisfiable` detection sites, each under a cond branch.

    Parametrised rather than written once because the three sites append to
    `vacuous` from three different places in `_classify_assumed_pred`, and a
    fix that reached only the one this branch happened to probe would leave
    the other two live.
    """
    assert _run(h).status == "UNKNOWN", (
        f"{name}: no point of the declared box that takes this branch "
        f"satisfies the branch's assume, so every obligation inside it is "
        f"vacuously true and REFUTED is wrong"
    )
    p = _prop(h)
    assert p.assume_dropped is True
    assert all(o.status != "violated-over-set" for o in p.obligations)


@pytest.mark.parametrize("name,h", _DETECTORS, ids=_DETECTOR_IDS)
def test_the_branch_vacuity_note_and_the_verdict_now_agree(name, h):
    """The defect's signature was a run that said both things at once: the
    F2 note "obligations in this branch may be vacuous" beside REFUTED.
    Whichever note fires, the obligation must be withheld."""
    p = _prop(h)
    assert any(
        "may be vacuous under the branch's precondition" in n
        or "constraint DROPPED" in n
        for n in p.notes
    )
    assert any("WITHHELD from REFUTED" in n for n in p.notes)


def test_the_branch_assume_now_changes_something():
    """The whole defect in one line: at `9efea6f` and at `3afbf01` these two
    verdicts were EQUAL, so the assume contributed nothing whatever."""
    assert _run(_branch_no_assume).status == "REFUTED"
    assert _run(_branch_empty_meet).status == "UNKNOWN"


def test_the_branch_control_still_refutes_on_both_legs():
    """Don't close the bad path by blunting the good one."""
    assert _run(_branch_no_assume, refine=None).status == "REFUTED"
    assert _run(_branch_no_assume, refine="affine").status == "REFUTED"
    assert _prop(_branch_no_assume).assume_dropped is False


def test_the_branch_withhold_is_ONE_SIDED():
    """A discharge inside the branch still discharges: VERIFIED over a
    superset implies VERIFIED over the subset, and the flag must not touch
    it. This is the direction the fix must NOT move."""
    v = _run(_branch_verified)
    assert v.status == "VERIFIED"
    assert _prop(_branch_verified).assume_dropped is True


def test_the_same_assume_at_top_level_still_RAISES():
    """The F2 degradation is what makes D possible, and it is correct: the
    branch-local case must stay a note, the whole-domain case a refusal.
    Pinned so a later fix cannot "close D" by re-raising under a branch."""
    def top():
        x = any_array((3,), "float64", (-1.0, 1.0))
        assume(x >= 2.0)
        return (assert_(x > 5.0),)
    with pytest.raises(P.UnsatisfiableAssumptionError):
        _prop(top)


def test_a_vacuous_conjunct_beside_a_narrowing_withholds():
    """`or vacuous` on the MIXED path — `restricting` is empty here (nothing
    was dropped at all), so that operand is the only thing setting the
    flags. Delete it and this returns REFUTED."""
    p = _prop(_branch_mixed_vacuous)
    assert p.coverage.constrained == 1, (
        "this harness is only a test of the MIXED path while a conjunct "
        "actually narrows; if that stops holding the test is testing "
        "something else"
    )
    assert p.assume_dropped is True
    assert _run(_branch_mixed_vacuous).status == "UNKNOWN"


# -- C: the ordering rule, and the two legs that do not yet agree on it ------
#
# The principal has RULED: an `assume` is QUERY-scoped — it constrains the
# whole query, not only the obligations traced after it. Implementing that
# uniformly is a separate change with its own pre-registration; this branch
# implements none of it and only pins what the tree does today, on BOTH legs,
# so that change lands loudly wherever it touches.
#
# What the tree does today is not one thing. The affine guard keys on
# `propagation.assume_dropped`, a whole-run flag with no order in it, while
# the interval withhold reads `uncertified` at assert time — so the affine
# leg is already query-scoped and the interval leg is still order-scoped.
# Measured (`refine=None` / `refine="affine"`):
#
#     assert-before-assume, interval-decided   REFUTED / REFUTED
#     assert-before-assume, affine-decided     UNKNOWN / UNKNOWN
#     the same, no assume at all (control)     UNKNOWN / REFUTED
#
# The affine row read UNKNOWN / **REFUTED** at `9efea6f`, so this branch DID
# move an ordering row (PREREG_REF1 C7 is false at the tip; see its
# re-scoring). The direction is REFUTED → UNKNOWN, and it is the direction
# the ruling wants, but it arrived as a side effect of the mechanism-B guard
# rather than as a decision.

_ORDERING_ROWS = [
    ("interval-leg/refine-none", _assert_before_assume, None, "REFUTED"),
    ("interval-leg/refine-affine", _assert_before_assume, "affine", "REFUTED"),
    ("affine-leg/refine-none", _assert_before_assume_affine, None, "UNKNOWN"),
    ("affine-leg/refine-affine", _assert_before_assume_affine, "affine",
     "UNKNOWN"),
]


@pytest.mark.parametrize(
    "name,h,refine,expected", _ORDERING_ROWS,
    ids=[r[0] for r in _ORDERING_ROWS],
)
def test_an_assume_after_the_assert_is_pinned_on_BOTH_legs(
    name, h, refine, expected
):
    """CURRENT BEHAVIOUR, not a claim that any of these four is right.

    The oracle says this region is EMPTY (no point of [-1,1]^3 satisfies
    `x >= 2`), so under the query-scoped ruling every REFUTED here is
    wrong and will change. The predecessor of this test ran only
    `refine=None` on the interval-decided harness — one of these four
    cells — and was blind to the affine cell that actually moved.
    """
    assert _run(h, refine=refine).status == expected


def test_the_two_legs_do_not_yet_agree_on_assume_ordering():
    """The inconsistency as a fact rather than as prose. Disclosed, not
    decided: resolving it is the forthcoming query-scoping change."""
    # the affine leg withholds under an assume traced AFTER the assert...
    assert _run(_assert_before_assume_affine, refine="affine").status == "UNKNOWN"
    # ...and it is the assume doing that, not the leg going dark
    assert _run(_affine_control_no_assume, refine="affine").status == "REFUTED"
    # the interval leg does not withhold on the same ordering
    assert _run(_assert_before_assume, refine=None).status == "REFUTED"
