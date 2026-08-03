# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""One cause, three property-stating primitives, one hint.

``MEMBERSHIP_IDIOM_HINT`` was emitted at ONE call site — the dropped-assume
note — while the SAME ⊤ silently weakens the other two ways a property is
stated. Measured on this tree before the fix:

* ``assume(jnp.all(x >= 0))``      DROPPED, hint printed
* ``assert_(jnp.all(x >= 0))``     obligation ``unknown``, detail
                                   "undecided for 1/1 element(s)",
                                   **no propagation note at all**
* ``nonvacuity(jnp.all(...))``     check ``unknown``, stamp says
                                   "undecided — a membership condition could
                                   not be decided", **no note at all** — and
                                   not even :func:`verdict.undecided_cause_note`,
                                   which fires only on an undecided OBLIGATION,
                                   so a run whose obligations all discharge
                                   reports the gap in exactly one word.

The cause is a REGISTRY ASYMMETRY, pinned below: ``reduce_or`` has an interval
transfer in both registries and ``reduce_and`` has one in neither, so
``jnp.any`` decides and ``jnp.all`` does not.

**The hint's claims are the tests.** A hint that names a dead end is worse
than no hint — it is read by the one user already stuck — so every sentence
in the text is measured here: that all three named rewrites decide on all
three primitives, that all three make an ``assume`` CONSTRAIN rather than
DROP, and that they are still NOT interchangeable, which is the part the
text says nobody would guess: the two arithmetic forms narrow the
reduction's own intermediate and raise satisfiability-UNCERTIFIED (so a
definite violation is withheld from REFUTED), while the elementwise form
narrows the declared input and leaves REFUTED reachable.

Diagnostics only: this file also pins that the statuses on every emitting
path are exactly what they were before the hint reached them.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import _optional  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling import solvers as S  # noqa: E402
from stelling._jax_compat import transcribe  # noqa: E402
from stelling.harness import any_array, assert_, assume, nonvacuity  # noqa: E402
from stelling.preconditions import check  # noqa: E402

LO, HI = 0.0, 10.0

SEMANTICS = ["real", "ieee"]


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _run(h, semantics="real", assume_mode="constrain"):
    return P.propagate(
        transcribe(jax.make_jaxpr(h)()),
        assume_mode=assume_mode,
        semantics=semantics,
    )


def _hinted(p):
    return [n for n in p.notes if P.MEMBERSHIP_IDIOM_HINT in n]


# -- the cause ----------------------------------------------------------------


def test_the_registry_asymmetry_the_hint_asserts():
    """The hint's first sentence is a MEMBERSHIP fact about two registries,
    and a census can see it. Register a `reduce_and` row and the sentence
    "which has no interval transfer" becomes a lie printed at a user — so the
    row and the text move together or this fails."""
    assert "reduce_or" in P.TRANSFERS
    assert "reduce_or" in P.IEEE_TRANSFERS
    assert "reduce_and" not in P.TRANSFERS, (
        "a `reduce_and` interval transfer is registered, so "
        "MEMBERSHIP_IDIOM_HINT's 'which has no interval transfer' is false — "
        "rewrite the hint (and re-measure every rewrite it names) before "
        "landing the row"
    )
    assert "reduce_and" not in P.IEEE_TRANSFERS, (
        "same, for the ieee registry: the hint is emitted under both "
        "semantics and claims the gap unconditionally"
    )
    assert "no interval transfer" in P.MEMBERSHIP_IDIOM_HINT


# -- the three emitting paths -------------------------------------------------


def _assume_all():
    x = any_array((3,), "float64", (-10.0, 10.0))
    assume(jnp.all(x >= LO))
    return (assert_(jnp.sum(x) >= 0.0),)


def _assert_all():
    x = any_array((3,), "float64", (1.0, 9.0))
    return (assert_(jnp.all(x >= LO)),)


def _nonvacuity_all():
    x = any_array((3,), "float64", (LO, HI))
    pt = jnp.array([1.0, 2.0, 3.0])
    return (assert_(jnp.sum(x) >= 0.0), nonvacuity(jnp.all(pt >= LO)))


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_assume_path_still_hints(semantics):
    """The pre-existing site. It must not regress while the others gain it."""
    p = _run(_assume_all, semantics)
    assert p.assume_dropped
    assert len(_hinted(p)) == 1
    assert "DROPPED" in _hinted(p)[0]


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_assert_path_hints(semantics):
    """Before: `unknown`, "undecided for 1/1 element(s)", and silence."""
    p = _run(_assert_all, semantics)
    assert p.obligations[0].status == "unknown"
    notes = _hinted(p)
    assert len(notes) == 1, p.notes
    assert notes[0].startswith("obligation UNDECIDED at ")
    assert "'reduce_and'" in notes[0]


@pytest.mark.parametrize("semantics", SEMANTICS)
def test_the_nonvacuity_path_hints(semantics):
    """Before: the stamp's one word `undecided`, and nothing else — this face
    does not even reach the verdict's coverage-cause note."""
    p = _run(_nonvacuity_all, semantics)
    assert p.nonvacuity_checks[0].status == "unknown"
    notes = _hinted(p)
    assert len(notes) == 1, p.notes
    assert notes[0].startswith("nonvacuity condition UNDECIDED at ")


def test_one_query_stating_all_three_gets_one_note_per_undecided_face():
    def h():
        x = any_array((3,), "float64", (-10.0, 10.0))
        pt = jnp.array([1.0, 2.0, 3.0])
        assume(jnp.all(x >= LO))
        return (assert_(jnp.all(x <= HI)), nonvacuity(jnp.all(pt >= LO)))

    p = _run(h)
    notes = _hinted(p)
    assert len(notes) == 3, notes
    assert sum(n.startswith("assume constraint DROPPED") for n in notes) == 1
    assert sum(n.startswith("obligation UNDECIDED") for n in notes) == 1
    assert sum(n.startswith("nonvacuity condition UNDECIDED") for n in notes) == 1


# -- negative controls: the hint must not fire indiscriminately ---------------


def test_a_plain_straddle_gets_no_hint():
    """The commonest UNKNOWN of all. Coverage is complete; there is no
    reduction to delete, and naming one would send the reader nowhere."""
    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        return (assert_(jnp.sum(x) >= 0.0),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown == 0
    assert _hinted(p) == []


def test_a_top_upstream_of_the_predicate_gets_no_hint():
    """`reduce_max` has no transfer either — same ⊤, same undecided
    obligation, and `jnp.all` is not the thing to rewrite."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.max(x) >= 0.0),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown_primitives == (("reduce_max", 1),)
    assert _hinted(p) == []


def test_a_nonvacuity_top_upstream_of_the_predicate_gets_no_hint():
    def h():
        x = any_array((3,), "float64", (LO, HI))
        pt = jnp.array([1.0, 2.0, 3.0])
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(jnp.min(pt) >= LO))

    p = _run(h)
    assert p.nonvacuity_checks[0].status == "unknown"
    assert _hinted(p) == []


def test_a_predicate_that_IS_another_primitives_top_gets_no_hint():
    """The gate's own mutant. The two controls above leave the gate itself
    untested: a `reduce_max` ⊤ feeds a `ge` whose transfer runs, so the
    judged predicate is not an artifact ⊤ at all and a gate widened to "any
    ⊤" would still not fire. `jnp.logical_not` lowers to `not`, which has no
    transfer either, so HERE the predicate itself is the ⊤ — and the hint
    must still stay away, because deleting a `jnp.all` fixes nothing for
    this reader."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        return (assert_(jnp.logical_not(x >= LO)),)

    p = _run(h)
    assert p.obligations[0].status == "unknown"
    assert p.coverage.unknown_primitives == (("not", 1),)
    assert _hinted(p) == []


def test_a_nonvacuity_predicate_that_IS_another_primitives_top_gets_no_hint():
    def h():
        x = any_array((3,), "float64", (LO, HI))
        pt = jnp.array([1.0, 2.0, 3.0])
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(jnp.logical_not(pt < LO)))

    p = _run(h)
    assert p.nonvacuity_checks[0].status == "unknown"
    assert p.coverage.unknown_primitives == (("not", 1),)
    assert _hinted(p) == []


def test_reduce_and_that_is_not_the_judged_predicate_gets_no_hint():
    """The asymmetry's other half: `reduce_and` fell to ⊤ in this query, but
    the obligation is a plain comparison that decides. A hint keyed on the
    query containing `reduce_and` — rather than on the judged predicate BEING
    its ⊤ — would fire here, at a reader with nothing to fix."""
    def h():
        x = any_array((3,), "float64", (1.0, 2.0))
        y = jnp.where(jnp.all(x >= LO), x, x)
        return (assert_(jnp.sum(y) >= 0.0),)

    p = _run(h)
    assert ("reduce_and", 1) in p.coverage.unknown_primitives
    assert p.obligations[0].status == "discharged"
    assert _hinted(p) == []


def test_the_decided_faces_get_no_hint():
    """`jnp.any` lowers to `reduce_or`, which IS registered — the asymmetry,
    from the deciding side."""
    def h():
        x = any_array((3,), "float64", (1.0, 9.0))
        pt = jnp.array([1.0, 2.0, 3.0])
        return (assert_(jnp.any(x >= LO)), nonvacuity(pt >= LO))

    p = _run(h)
    assert p.obligations[0].status == "discharged"
    assert p.nonvacuity_checks[0].status == "discharged"
    assert _hinted(p) == []


# -- the hint's own claims ----------------------------------------------------
#
# Named in the text, so measured here. The parameter is the membership
# predicate "every element of v is >= LO", one spelling per row.

ELEMENTWISE = ("elementwise", lambda v: v >= LO)
HINGE = ("hinge", lambda v: jnp.sum(jnp.maximum(LO - v, 0.0)) <= 0.0)
COUNT = ("count", lambda v: jnp.sum((v < LO).astype(jnp.int32)) == 0)
REWRITES = [ELEMENTWISE, HINGE, COUNT]


def _ids(x):
    return x if isinstance(x, str) else ""


@pytest.mark.parametrize("name,form", REWRITES, ids=_ids)
def test_every_named_rewrite_decides_as_an_assert(name, form):
    def h():
        x = any_array((3,), "float64", (1.0, 9.0))
        return (assert_(form(x)),)

    p = _run(h)
    assert p.obligations[0].status == "discharged", p.obligations[0].detail
    assert p.coverage.unknown == 0


@pytest.mark.parametrize("name,form", REWRITES, ids=_ids)
def test_every_named_rewrite_decides_as_a_nonvacuity(name, form):
    def h():
        x = any_array((3,), "float64", (LO, HI))
        pt = jnp.array([1.0, 2.0, 3.0])
        return (assert_(jnp.sum(x) >= 0.0), nonvacuity(form(pt)))

    p = _run(h)
    assert p.nonvacuity_checks[0].status == "discharged"
    assert p.coverage.unknown == 0


@pytest.mark.parametrize("name,form", REWRITES, ids=_ids)
def test_every_named_rewrite_makes_an_assume_constrain(name, form):
    def h():
        x = any_array((3,), "float64", (-10.0, 10.0))
        assume(form(x))
        return (assert_(jnp.sum(x) >= 0.0),)

    p = _run(h)
    assert p.coverage.constrained == 1
    assert not p.assume_dropped


def _violation_under(form):
    """A definitely-false obligation under an assume spelled `form`."""
    def h():
        x = any_array((3,), "float64", (-10.0, 10.0))
        assume(form(x))
        return (assert_(jnp.sum(x) <= -100.0),)

    return _run(h)


def test_the_arithmetic_forms_narrow_an_intermediate_and_withhold_refuted():
    """The clause the text calls the part nobody would guess, half one.

    CONSTRAIN is not the whole story: these two narrow the reduction's own
    output, which is an over-approximated intermediate, so audit F7 stamps
    the precondition satisfiability-UNCERTIFIED and every definite violation
    judged under it is withheld from REFUTED."""
    for _, form in (HINGE, COUNT):
        p = _violation_under(form)
        assert any("UNCERTIFIED" in n for n in p.notes)
        assert p.obligations[0].status == "unknown"
        assert "WITHHELD from REFUTED" in p.obligations[0].detail


def test_the_elementwise_form_narrows_the_declaration_and_keeps_refuted():
    """Half two, and the reason the elementwise form is named first."""
    p = _violation_under(ELEMENTWISE[1])
    assert not any("UNCERTIFIED" in n for n in p.notes)
    assert p.obligations[0].status == "violated-over-set"


def test_only_the_elementwise_form_discharges_the_downstream_obligation():
    """The same difference from the VERIFIED side: a narrowing that lands on
    a dead intermediate buys the query nothing."""
    def under(form):
        def h():
            x = any_array((3,), "float64", (-10.0, 10.0))
            assume(form(x))
            return (assert_(jnp.sum(x) >= 0.0),)

        return _run(h).obligations[0].status

    assert under(ELEMENTWISE[1]) == "discharged"
    assert under(HINGE[1]) == "unknown"
    assert under(COUNT[1]) == "unknown"


# -- the hint reaches the reader ----------------------------------------------


def test_the_hint_survives_the_front_door_without_a_solver():
    v = check(_assert_all, vacuity_mode="inputs-only")
    assert v.status == "UNKNOWN"
    assert any(P.MEMBERSHIP_IDIOM_HINT in n for n in v.notes)


@pytest.mark.skipif(
    not (_optional.available("cvc5") or _optional.available("z3")),
    reason="no SMT backend installed",
)
def test_the_hint_survives_escalation_which_replaces_the_detail():
    """Why the hint is a NOTE and not the obligation detail. Measured: the
    escalation record's detail REPLACES the propagation's, so a detail-only
    hint would vanish for exactly the reader who paid for a solver."""
    cj = transcribe(jax.make_jaxpr(_assert_all)())
    p = P.propagate(cj)
    esc = S.escalate(cj, p, S.SolverConfig(timeout_ms=60_000))
    assert P.MEMBERSHIP_IDIOM_HINT not in esc.records[0].detail
    v = check(_assert_all, vacuity_mode="inputs-only", solver_timeout_ms=60_000)
    assert any(P.MEMBERSHIP_IDIOM_HINT in n for n in v.notes)


# -- diagnostics only ---------------------------------------------------------


@pytest.mark.parametrize(
    "harness,semantics,statuses,nonvac",
    [
        (_assume_all, "real", ["unknown"], []),
        (_assume_all, "ieee", ["unknown"], []),
        (_assert_all, "real", ["unknown"], []),
        (_assert_all, "ieee", ["unknown"], []),
        (_nonvacuity_all, "real", ["discharged"], ["unknown"]),
        # ieee's reduce_sum declines above two elements, so THIS row's
        # obligation was already unknown for a reason of its own — pinned as
        # measured, not normalized away
        (_nonvacuity_all, "ieee", ["unknown"], ["unknown"]),
    ],
)
def test_the_hint_moves_no_verdict(harness, semantics, statuses, nonvac):
    """The statuses on every emitting path, pinned at what they were before
    the hint reached them."""
    p = _run(harness, semantics)
    assert [o.status for o in p.obligations] == statuses
    assert [c.status for c in p.nonvacuity_checks] == nonvac
