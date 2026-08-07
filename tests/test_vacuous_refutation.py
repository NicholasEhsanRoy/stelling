# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""REFUTED over a superset of the assumed region, on the two paths that
still emitted it.

`tests/test_dropped_assume.py` pins F7's no-op half for an assume that
dropped WHOLLY. Two paths reached a definite REFUTED without ever passing
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

Both are wrong for one reason: the assumed region is EMPTY, so the
conditional claim is vacuously TRUE. A definite violation over a superset
is a refutation only if the assumed region is non-empty, and a dropped
conjunct is exactly the part of the precondition whose satisfiability was
never established.

NOT covered here, deliberately: an `assume` traced AFTER the `assert_` it
should constrain. That is a question about what an `assume` SCOPES OVER,
not about superset judging, and it is reserved to the principal
(`scratchpad/PREREG_REF1.md`, clause C7). `test_an_assume_after_the_assert
_is_the_reserved_ordering_question` pins the CURRENT behaviour so a ruling
either way lands loudly.
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
    x = any_array((3,), "float64", (-1.0, 1.0))
    o = assert_(x > 5.0)
    assume(jnp.all(x >= 2.0))
    return (o,)


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


def test_an_indeterminate_dropped_conjunct_does_NOT():
    """The other face. Without this the discriminant could return True
    unconditionally and every test above still pass."""
    assert _run(_restricting_relational).status == "UNKNOWN"
    assert _prop(_restricting_relational).assume_dropped is True


def test_the_discriminant_refuses_a_non_bool_operand():
    """An integer `and`'s box of [1,1] is the integer one, not truth."""
    def h():
        i = any_array((3,), "int32", (1, 1))
        assume(i & i)                 # bit arithmetic on non-bool operands
        return (assert_(i > 5),)
    p = _prop(h)
    assert p.assume_dropped is True, (
        "a non-bool `and` whose box is [1,1] must not be read as "
        "certainly-true; its [1,1] means the integer one"
    )


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


# -- C: reserved ------------------------------------------------------------

def test_an_assume_after_the_assert_is_the_reserved_ordering_question():
    """NOT a claim that this verdict is right.

    Nothing in `assume`'s or `assert_`'s docstring makes an assume
    order-scoped; forward-only narrowing is stated in `propagate`'s module
    docstring as a propagation fact. Whether an `assume` constrains the
    obligations traced BEFORE it is a semantics choice reserved to the
    principal (PREREG_REF1 C7), so this pass deliberately left it alone.

    The oracle says this region is EMPTY, so under a query-scoped reading
    this REFUTED is wrong. Pinned as CURRENT BEHAVIOUR so a ruling either
    way changes a test rather than sliding through.
    """
    assert _run(_assert_before_assume).status == "REFUTED"
