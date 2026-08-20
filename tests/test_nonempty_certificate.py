# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The non-emptiness certificate: what it may do, and what it may never do.

A run whose assume state does not certify a set-level refutation withholds
every definite violation from REFUTED, query-wide. The withholding has
exactly one ground — the assumed region MAY BE EMPTY, in which case every
obligation is vacuously true. Exhibit one point of the declared set at
which every assume of the query holds and the ground is gone.

This file pins the three claims that make that safe:

1. **ONE-SIDED.** A failed search is not a proof of emptiness, and leaves
   the run BYTE-IDENTICAL to what it was — same statuses, same notes,
   same details, no disclosure that a search happened.
2. **`discharged` is never touched.** The certificate's only power is
   turning a withheld ``unknown`` back into ``violated-over-set``. Pinned
   per obligation, with a POSITIVE CONTROL underneath it: an unsound
   mutant of this very machinery that CAN reach ``discharged``, which the
   same ledger must flag. A zero with no positive control is
   unfalsifiable.
3. **The witness is a MEMBER of the declared set** — a value of the
   declaration's own dtype, not merely a number inside the interval — and
   the predicate is confirmed in the arithmetic that judged the query, so
   an enclosure that straddles a bound confirms nothing.

The ROUTING claims — that the witness decision is
``exactness.certifies_point_witness`` and that both legs carry the answer
into ``exactness.certifies_set_refutation`` — live in
``tests/test_exactness_lift.py``, with the shared-decision pins.
"""

from __future__ import annotations

import dataclasses

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp  # noqa: E402

from stelling import exactness  # noqa: E402
from stelling import propagate as P  # noqa: E402
from stelling.harness import any_array, assert_, assume, trace  # noqa: E402
from stelling.propagate import propagate  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _prop(h, **kw):
    return propagate(trace(h), **kw)


def _without_certificate(h, monkeypatch, **kw):
    """The same run with the certificate's route forced to find nothing —
    which is exactly the pre-certificate behaviour, since the ONLY thing
    the build added to a run is this answer."""
    monkeypatch.setattr(
        exactness, "certifies_point_witness", lambda **k: False
    )
    out = propagate(trace(h), **kw)
    monkeypatch.undo()
    return out


# --- the harnesses ------------------------------------------------------------
#
# Each docstring's claim about the assumed region is an ORACLE result, not
# an argument: `scratchpad/cert/oracle.py` samples the declared set outside
# stelling and reports whether any sampled member satisfies every assume.


def h_inhabited_narrowing():
    """`{x ∈ [0,1] : 2x ≥ 0.5}` is `[0.25, 1]` — INHABITED. The narrowing
    target `y` is a transfer output, so the run is uncertified."""
    x = any_array((), "float64", (0.0, 1.0))
    y = x * 2.0
    assume(y >= 0.5)
    return (assert_(x <= -1.0),)


def h_empty_narrowing():
    """`{x ∈ [0,1] : x - x ≥ 0.5}` is EMPTY — `x - x` is exactly 0 at every
    point while its box is `[-1, 1]`. The certificate must decline."""
    x = any_array((), "float64", (0.0, 1.0))
    y = x - x
    assume(y >= 0.5)
    return (assert_(x <= -1.0),)


def h_inhabited_relational():
    """A RELATIONAL assume: dropped over the boxes, decidable at a point.
    `{x, y ∈ [0,1] : x ≥ y}` is INHABITED."""
    x = any_array((), "float64", (0.0, 1.0))
    y = any_array((), "float64", (0.0, 1.0))
    assume(x >= y)
    return (assert_(x + y >= 100.0),)


def h_empty_relational():
    """The same shape over disjoint boxes: `{x ∈ [0,1], y ∈ [2,3] : x ≥ y}`
    is EMPTY."""
    x = any_array((), "float64", (0.0, 1.0))
    y = any_array((), "float64", (2.0, 3.0))
    assume(x >= y)
    return (assert_(x + y >= 100.0),)


def h_transcendental_slack():
    """`{x ∈ [0,1] : sqrt(x + 1) ≥ 1.2}` is `[0.44, 1]` — INHABITED, and
    the enclosure of `sqrt` at a pinned point has SLACK against 1.2, so a
    definite TRUE over the enclosure is a TRUE at the value."""
    x = any_array((), "float64", (0.0, 1.0))
    assume(jnp.sqrt(x + 1.0) >= 1.2)
    return (assert_(x <= -1.0),)


def h_discharged_under_uncertified():
    """A DISCHARGE under an uncertified narrowing. Nothing here may move."""
    x = any_array((), "float64", (0.0, 1.0))
    y = x * 2.0
    assume(y >= 0.5)
    return (assert_(x >= -5.0),)


def h_mixed():
    """Both faces in one query: a discharge and a violation under the same
    uncertified narrowing, so the per-obligation ledger has something to
    separate."""
    x = any_array((), "float64", (0.0, 1.0))
    y = x * 2.0
    assume(y >= 0.5)
    return (assert_(x >= -5.0), assert_(x <= -1.0))


_RECOVERS = (h_inhabited_narrowing, h_inhabited_relational,
             h_transcendental_slack, h_mixed)
_DECLINES = (h_empty_narrowing, h_empty_relational)


# --- 1. ONE-SIDED -------------------------------------------------------------


@pytest.mark.parametrize("h", _DECLINES, ids=lambda f: f.__name__)
def test_a_failed_certificate_search_changes_nothing_at_all(h, monkeypatch):
    """The one-sidedness pin, byte-for-byte over the WHOLE propagation.

    Not "the status is still unknown" — the whole record, including the
    notes. A failed search is not a finding and must not read as one: no
    "searched and found nothing" note, no changed detail, nothing a
    consumer could mistake for evidence that the region is empty. Failing
    to find a point is not a proof of emptiness, and this is what that
    sentence costs in code.
    """
    with_search = _prop(h)
    without = _without_certificate(h, monkeypatch)
    assert with_search.region_inhabited is False
    assert with_search == without, (
        "a failed certificate search must leave the run exactly as it was"
    )
    assert with_search.obligations[0].status == "unknown"
    assert "WITHHELD from REFUTED" in with_search.obligations[0].detail


def test_a_failed_search_is_not_a_claim_of_emptiness_anywhere_in_the_notes():
    """The same rule stated against the TEXT, because the text is what a
    reader acts on. Nothing on a declining run may say the region is
    empty, was searched, or was shown to be anything."""
    p = _prop(h_empty_narrowing)
    joined = "\n".join(p.notes) + p.obligations[0].detail
    for forbidden in ("CERTIFIED NON-EMPTY", "no witness", "search", "empty region"):
        assert forbidden not in joined, forbidden


# --- 2. `discharged` is never touched -----------------------------------------


def _ledger(before, after):
    """Per-obligation moves between two runs of the SAME query, as
    ``(index, before_status, after_status)`` for every obligation whose
    status differs. Per obligation, never per query: a per-query score
    once turned a measured 24:168 trade in this project into a fake
    216:216."""
    return [
        (i, b.status, a.status)
        for i, (b, a) in enumerate(zip(before.obligations, after.obligations))
        if b.status != a.status
    ]


@pytest.mark.parametrize("h", _RECOVERS + _DECLINES, ids=lambda f: f.__name__)
def test_the_certificate_moves_only_unknown_to_violated(h, monkeypatch):
    """The ledger, per obligation. Every move the certificate makes is
    ``unknown -> violated-over-set``; nothing reaches ``discharged`` and
    nothing leaves it."""
    without = _without_certificate(h, monkeypatch)
    with_search = _prop(h)
    for index, before, after in _ledger(without, with_search):
        assert (before, after) == ("unknown", "violated-over-set"), (
            f"obligation {index} moved {before} -> {after}; the certificate "
            f"may only restore a withheld violation"
        )
    # and the discharges are untouched as a set, which the move list alone
    # would not say if two obligations swapped
    assert (
        {i for i, o in enumerate(without.obligations) if o.status == "discharged"}
        == {i for i, o in enumerate(with_search.obligations) if o.status == "discharged"}
    )


def test_the_ledger_would_see_a_move_toward_discharged__POSITIVE_CONTROL(
    monkeypatch,
):
    """THE POSITIVE CONTROL for the zero above. A zero with no positive
    control is unfalsifiable.

    An unsound mutant of this build's own machinery — one that lets the
    certificate lift a withheld obligation all the way to ``discharged``
    instead of only back to ``violated-over-set`` — is driven through the
    real pipeline, and the same ledger that reports zero on the real build
    must report the move here. If it did not, the zero above would mean
    nothing.
    """
    real = P._withhold_uncertified_refutations

    def two_sided(p):
        # THE MUTANT: reaches `discharged`, which the real build's
        # one-sidedness forbids outright.
        if p.region_inhabited:
            for i, o in enumerate(p.obligations):
                if o.status == "violated-over-set":
                    p.obligations[i] = dataclasses.replace(
                        o, status="discharged", detail="MUTANT"
                    )
        return real(p)

    without = _without_certificate(h_mixed, monkeypatch)
    monkeypatch.setattr(P, "_withhold_uncertified_refutations", two_sided)
    mutated = _prop(h_mixed)
    moves = _ledger(without, mutated)
    assert any(after == "discharged" for _, _, after in moves), (
        "the ledger cannot see a move toward discharged, so its zero on "
        "the real build is not evidence of anything"
    )


def test_a_certified_run_does_not_stamp_a_known_false_assumption():
    """The stamp swap. Both uncertified assumptions say "the conditional
    claim MAY BE vacuous" — true when the walk writes them, before any
    witness exists, and FALSE on a run the certificate then settles. A
    stamped assumption is what a verdict claims to rest on, so a
    known-false one is a disclosure defect whatever the verdict says.
    """
    p = _prop(h_inhabited_narrowing)
    assert p.region_inhabited is True
    assert P.UNCERTIFIED_NARROWING_ASSUMPTION not in p.assumptions
    assert P.UNCERTIFIED_DROP_ASSUMPTION not in p.assumptions
    assert P.REGION_INHABITED_ASSUMPTION in p.assumptions
    # and what REPLACES it says what the claim now rests on, rather than
    # simply dropping the disclosure
    assert "probed point of the declared set" in P.REGION_INHABITED_ASSUMPTION
    assert "What this rests on" in P.REGION_INHABITED_ASSUMPTION

    # the control: a run the certificate declines keeps its uncertified
    # assumption exactly as before, so the swap is the certificate's and
    # not an unconditional deletion
    q = _prop(h_empty_narrowing)
    assert q.region_inhabited is False
    assert P.UNCERTIFIED_NARROWING_ASSUMPTION in q.assumptions
    assert P.REGION_INHABITED_ASSUMPTION not in q.assumptions

    # and the DROP half of the swap, on its own mechanism
    d = _prop(h_inhabited_relational)
    assert d.assume_dropped is True and d.region_inhabited is True
    assert P.UNCERTIFIED_DROP_ASSUMPTION not in d.assumptions
    assert P.REGION_INHABITED_ASSUMPTION in d.assumptions


def test_a_certificate_never_reaches_a_run_with_nothing_withheld():
    """The search does not even run where there is nothing to lift: a
    query whose obligations all DISCHARGED pays nothing and claims
    nothing."""
    p = _prop(h_discharged_under_uncertified)
    assert p.narrowing_uncertified is True  # the withholding WOULD apply...
    assert p.obligations[0].status == "discharged"
    assert p.region_inhabited is False  # ...and no certificate was minted
    assert not any("CERTIFIED NON-EMPTY" in n for n in p.notes)


def test_the_certificate_reaches_the_affine_leg_as_a_LIVE_argument():
    """The second leg's third argument is not a documented-dead constant,
    and this is the query that makes it live.

    `assume(x >= y)` is RELATIONAL — dropped, so the run withholds and
    `coverage.constrained` stays 0, which is the one state in which the
    affine refinement does not decline wholly. `assert_(x - x >= 0.5)` is
    interval-UNDECIDED and affine-VIOLATED, so the refinement is the leg
    that mints the violation and the leg that must decide whether to
    withhold it. The region `{x, y ∈ [-1,1]^3 : x ≥ y}` contains
    `x = y = 0`, so the refutation is owed.

    Measured before the search's gate learned about this leg: UNKNOWN,
    with the certificate never computed — the interval leg had nothing
    withheld of its own, so it did not look.
    """
    from stelling import affine

    def h():
        x = any_array((3,), "float64", (-1.0, 1.0))
        y = any_array((3,), "float64", (-1.0, 1.0))
        assume(x >= y)
        return (assert_(x - x >= 0.5),)

    closed = trace(h)
    p = propagate(closed)
    assert p.assume_dropped is True
    assert p.coverage.constrained == 0  # or the refinement declines wholly
    assert p.obligations[0].status == "unknown"  # the interval leg cannot
    assert p.region_inhabited is True  # ...and the search ran anyway
    r, rep = affine.refine_propagation(closed, p)
    assert r.obligations[0].status == "violated-over-set"
    assert rep.violated == (0,)

    # the empty-region twin: same shape, disjoint boxes, and the affine
    # leg must still withhold. Without this the test above would pass on
    # a refinement that had simply stopped consulting the shared point.
    def h_empty():
        x = any_array((3,), "float64", (-1.0, 1.0))
        y = any_array((3,), "float64", (2.0, 3.0))
        assume(x >= y)
        return (assert_(x - x >= 0.5),)

    ce = trace(h_empty)
    pe = propagate(ce)
    assert pe.region_inhabited is False
    re, repe = affine.refine_propagation(ce, pe)
    assert re.obligations[0].status == "unknown"
    assert repe.violated == ()


def test_the_certificate_can_never_restore_a_branch_scoped_refutation():
    """A structural guarantee, not a coincidence, and worth its own pin.

    The certificate can only fire on a run that narrowed or dropped an
    assume; `_reachability_witnesses` certifies NOTHING on exactly such a
    run (``p.any_constrained or p.assume_dropped`` returns the empty set),
    so every branch-scoped violation stays withheld by the branch pass
    however inhabited the top-level region is. That is what keeps a
    top-level witness from walking around a branch-scoped precondition.
    """
    from jax import lax

    def h():
        x = any_array((2,), "float64", (-1.0, 1.0))

        def yes(v):
            assume(v[0] >= 2.0)  # EMPTY within the branch
            return assert_(v[0] > 5.0)

        def no(v):
            return assert_(v[0] > 5.0)

        return lax.cond(x[0] > 0, yes, no, x)

    p = _prop(h)
    assert p.assume_dropped is True
    assert all(o.status == "unknown" for o in p.obligations)


def test_an_assume_the_probe_walks_around_is_not_witnessed():
    """The static-requirement rule, on the query that motivates it. The
    branch-scoped assume is REQUIRED (it is in the IR) and can never be
    WITNESSED by a probe that takes the other branch, so no certificate is
    issued — whatever the top-level assume does."""
    from jax import lax

    def h():
        x = any_array((2,), "float64", (-1.0, 1.0))
        assume(x >= -0.9)  # top-level, certified, satisfiable

        def yes(v):
            assume(v[0] >= 2.0)  # EMPTY within the branch
            return assert_(v[0] > 5.0)

        def no(v):
            return assert_(v[0] > 5.0)

        return lax.cond(x[0] > 0, yes, no, x)

    p = _prop(h)
    assert p.region_inhabited is False
    assert all(o.status == "unknown" for o in p.obligations)


def test_an_assume_the_probe_walks_INTO_is_witnessed_and_certified():
    """The counter-construction to *"branch-scoped assumes are never
    certified"*, which is what this file, ``SOUNDNESS.md`` and two
    docstrings used to say.

    A probe pins each declaration to a point, which FORCES the cond — and
    forcing it can force it EITHER WAY. Here the query's only assume sits
    inside the branch taken when `x >= 0.5`, and probe 1 (the declared
    box's HIGH corner, `x = 1.0`) walks into that branch, evaluates the
    assume, finds it definitely true and witnesses it. The static
    requirement is then satisfied — the one assume the IR contains is the
    one the walk evaluated — and the certificate fires.

    **The recovery is sound**, and that is the point: at `x = 1.0` the
    program really does take the branch, really does evaluate
    `assume(v >= 0.25)`, and really does satisfy it, so the assumed region
    (as executed) is inhabited and the definite violation over the judged
    set is not vacuous.

    What the static requirement actually guarantees is the narrower
    sentence the docstrings now carry: an assume the probe walked AROUND
    is never certified — required and not witnessed, so the subset test
    fails. The other, independent mechanism (`_reachability_witnesses`
    returning the empty set on any run with `any_constrained or
    assume_dropped`) is what protects branch-scoped VIOLATIONS, and it is
    pinned one test up. Conflating the two overstated both.
    """
    from jax import lax

    def h():
        x = any_array((), "float64", (0.0, 1.0))

        def has_assume(v):
            assume(v >= 0.25)  # SATISFIABLE within the branch
            return v * 2.0

        y = lax.cond(x >= 0.5, has_assume, lambda v: v, x)
        return (assert_(y <= -1.0),)

    c = trace(h)
    # the assume really is branch-scoped: the top-level jaxpr holds none
    assert not [e for e in c.jaxpr.eqns if e.primitive == "stelling_assume"]
    assert len(P._assume_equation_ids(c.jaxpr)) == 1

    p = propagate(c)
    assert p.narrowing_uncertified is True, (
        "a branch invar never inherits exactness, so a branch-scoped "
        "assume narrows an over-approximated intermediate and the run "
        "would withhold — which is what makes the recovery observable"
    )
    assert p.region_inhabited is True
    assert p.obligations[0].status == "violated-over-set"
    assert any(
        "assumed region CERTIFIED NON-EMPTY: probe point 1 " in n
        for n in p.notes
    ), p.notes


def test_a_region_inhabited_only_via_the_UNTAKEN_branch_is_not_recovered():
    """The cost of the same rule, in exactly the shape the old sentence
    claimed it prevented.

    The only assume sits in the branch taken when `x >= 0.5` and is
    UNSATISFIABLE there (`v >= 2` over `[0.5, 1]`). Every `x < 0.5`
    therefore satisfies every assume the program EVALUATES at it — the
    assumed region is `[0, 0.5)`, inhabited — and the assert is definitely
    violated over the whole declared box, so a REFUTED is owed.

    It is not given. The requirement is STATIC, so the assume in the
    branch those points do not take is required and never witnessed:
    measured, 8 of the 16 probes walk the branch WITHOUT the assume and
    record an EMPTY witness map, and the subset test fails on every one.
    A sound refutation, lost to the static requirement. Withholding is the
    safe direction and this is a real price, recorded here rather than
    left to be rediscovered.
    """
    from jax import lax

    def h():
        x = any_array((), "float64", (0.0, 1.0))

        def has_assume(v):
            assume(v >= 2.0)  # EMPTY within the branch
            return v * 2.0

        y = lax.cond(x >= 0.5, has_assume, lambda v: v, x)
        return (assert_(y <= -1.0),)

    c = trace(h)
    required = P._assume_equation_ids(c.jaxpr)
    assert len(required) == 1

    walked_around = 0
    for k in range(P._PROBE_COUNT):
        probe = P._Propagator("constrain", "real")
        probe.pin = k
        try:
            probe.run(c.jaxpr, list(c.consts), [])
        except Exception:  # noqa: BLE001
            continue
        if not probe.assume_witness:
            walked_around += 1
    assert walked_around, (
        "no probe walked around the assume, so this row is not measuring "
        "the static requirement at all"
    )

    p = propagate(c)
    assert p.region_inhabited is False
    assert p.obligations[0].status == "unknown"


# --- 3. the witness is a MEMBER of the declared set ---------------------------


def test_the_witness_is_a_value_of_the_declarations_own_dtype():
    """`_member_bounds`/`_probe_point` are reused rather than re-derived,
    and this is what that buys: an INTEGER declaration is witnessed only by
    integers.

    `n ∈ int32 [0, 10]`, `m = 2n`, `assume(m >= 3)` and `assume(m <= 3)`
    together say `m == 3` — satisfiable for a REAL `n` (n = 1.5), and
    satisfiable for no member of the declared set, every one of which
    makes `m` even. A witness search that probed the interval rather than
    the dtype's values would certify it.
    """
    def h():
        n = any_array((), "int32", (0.0, 10.0))
        m = n * 2
        assume(m >= 3)
        assume(m <= 3)
        return (assert_(n >= 100),)

    p = _prop(h)
    assert p.region_inhabited is False, (
        "no int32 n makes 2n equal 3; a witness here is off the dtype's grid"
    )

    # the control: the same shape at an even bound IS witnessed, so the
    # refusal above is the integer rule and not the search going dark
    def h_ok():
        n = any_array((), "int32", (0.0, 10.0))
        m = n * 2
        assume(m >= 4)
        assume(m <= 4)
        return (assert_(n >= 100),)

    assert _prop(h_ok).region_inhabited is True


def test_a_narrow_float_declaration_is_witnessed_by_a_value_of_its_format():
    """The float half of the same rule — the half a float64-only corpus
    cannot see at all, because float64 is the one format that IS its own
    interval."""
    def h():
        x = any_array((), "float32", (0.0, 1.0))
        y = x * 2.0
        assume(y >= 0.5)
        return (assert_(x <= -1.0),)

    p = _prop(h)
    assert p.region_inhabited is True
    assert p.obligations[0].status == "violated-over-set"


def test_the_witness_check_runs_in_the_arithmetic_that_judged_the_query():
    """The exactness question, on the case that separates the answers.

    At the point `(0.1, 0.2)` the predicate `x0 + x1 >= 0.30000000000000004`
    is TRUE in binary64 and FALSE in ℝ. Under ``semantics="real"`` the
    endpoints are computed in ``Fraction`` and directed-rounded, so the box
    STRADDLES the bound: the predicate is INDETERMINATE and no witness is
    claimed from it. An exact-rational checker would answer FALSE; this
    answers "not established". Weaker, never unsound — and never in
    disagreement with the propagation that judged the query, which is the
    reason for running the check here rather than beside it.
    """
    k = 0.30000000000000004

    def h_at(bound):
        def h():
            # POINT declarations for the two operands, so the declared set
            # pins `a` and `b` to exactly 0.1 and 0.2 and no probe of the
            # grid can wander off the case under test. `z - z` is exactly 0
            # at any point and `[-1, 1]` as a box, so the sum's BOX is far
            # too wide for the assume to certify itself through audit F8 —
            # which is what leaves the certificate as the only route and
            # makes this a test of the certificate's arithmetic.
            a = any_array((), "float64", (0.1, 0.1))
            b = any_array((), "float64", (0.2, 0.2))
            z = any_array((), "float64", (0.0, 1.0))
            assume(a + b + (z - z) >= bound)
            return (assert_(a <= -1.0),)

        return h

    p = _prop(h_at(k))
    assert p.narrowing_uncertified is True  # audit F8 did NOT certify it
    assert p.region_inhabited is False, (
        "the enclosure of 0.1 + 0.2 straddles this bound: TRUE in binary64, "
        "FALSE in R, and INDETERMINATE here — which withholds"
    )
    assert p.obligations[0].status == "unknown"

    # the control, with the bound clear of the whole enclosure: the
    # predicate IS definitely true at the point, the certificate fires,
    # and the withheld refutation comes back. So the refusal above is the
    # straddle and not the shape.
    ok = _prop(h_at(0.29))
    assert ok.narrowing_uncertified is True
    assert ok.region_inhabited is True
    assert ok.obligations[0].status == "violated-over-set"


def test_a_transcendental_is_a_boundary_not_a_gap():
    """Nothing confirms a point exactly through `sqrt`/`sin`/`exp`/`log`,
    and this does not try: the enclosure at a pinned point has width. What
    it still does, soundly, is confirm a predicate with SLACK against that
    width — a definite TRUE over an enclosure is a TRUE at the value. So
    the boundary is the margin, not the primitive."""
    p = _prop(h_transcendental_slack)
    assert p.region_inhabited is True
    assert p.obligations[0].status == "violated-over-set"

    def h_no_slack():
        # a POINT declaration at 0.25, whose square root is exactly 0.5 in
        # ℝ and in binary64 — and whose stelling enclosure is
        # [0x1.fffffffffffffp-2, 0x1.0000000000001p-1], one ulp either
        # side, because the transfer rounds outward unconditionally. The
        # bound sits INSIDE that: no definite TRUE, no witness, however
        # true the predicate happens to be.
        x = any_array((), "float64", (0.25, 0.25))
        assume(jnp.sqrt(x) >= 0.5)
        return (assert_(x <= -1.0),)

    p_ns = _prop(h_no_slack)
    assert p_ns.region_inhabited is False
    assert p_ns.obligations[0].status == "unknown"


def test_the_certificate_speaks_the_dial_the_query_was_judged_on():
    """The witness check runs in the run's OWN semantics, and the two
    dials are NOT ordered — measured, against the sentence that was
    written here first.

    `x` is declared at the point 0.25 and `sqrt(0.25)` is EXACTLY 0.5 in
    binary64, so the ieee transfer encloses it as the point `[0.5, 0.5]`
    and `>= 0.5` is definitely TRUE; the REAL-mode transfer bumps outward
    unconditionally, straddles the bound and certifies nothing. **ieee
    certifies where real does not** — the opposite of the usual direction,
    and sound in both, because the certificate is computed in the same
    arithmetic the obligations are judged in and under ieee the program
    really does compute 0.5. Three corpus rows go the other way
    (`scratchpad/cert/RESULTS_invariant.txt`).
    """
    def h():
        x = any_array((), "float64", (0.25, 0.25))
        assume(jnp.sqrt(x) >= 0.5)
        return (assert_(x <= -1.0),)

    assert _prop(h, semantics="real").region_inhabited is False
    assert _prop(h, semantics="ieee").region_inhabited is True

    # and an EMPTY region is declined on BOTH dials: the dials differ on
    # what they can confirm, never on whether an empty region certifies
    for dial in ("real", "ieee"):
        assert _prop(h_empty_narrowing, semantics=dial).region_inhabited is False
        assert _prop(h_empty_relational, semantics=dial).region_inhabited is False


def test_a_certifying_probe_narrows_nothing():
    """The invariant the reading order rests on.

    The witness answer for each assume is read BEFORE `_assume_constrain`
    can meet anything into the env, and the argument that this suffices is
    that a predicate whose box is `[1, 1]` is definitely true over the
    boxes in force, so its meet with the closed half-space is a NO-OP. If
    that failed, an earlier assume could certify itself AND cut the box a
    later assume is read against, and the later `[1, 1]` would be a
    statement about a box that no longer over-approximates the point.

    Measured across the corpus rather than argued: 148 certifying probe
    runs inspected, 0 narrowed anything
    (`scratchpad/cert/RESULTS_invariant.txt`). Driven here on the two
    rows where a certifying run has more than one assume to get wrong.
    """
    for h in (h_inhabited_narrowing, h_mixed):
        c = trace(h)
        required = P._assume_equation_ids(c.jaxpr)
        certified = 0
        for k in range(P._PROBE_COUNT):
            probe = P._Propagator("constrain", "real")
            probe.pin = k
            try:
                probe.run(c.jaxpr, list(c.consts), [])
            except Exception:  # noqa: BLE001
                continue
            if not exactness.certifies_point_witness(
                required_assumes=required,
                witnessed_assumes=frozenset(
                    key for key, ok in probe.assume_witness.items() if ok
                ),
            ):
                continue
            certified += 1
            assert not [n for n in probe.notes if "narrowed x" in n], (
                f"{h.__name__} probe {k} certified AND narrowed"
            )
            assert probe.narrowing_uncertified is False
        assert certified, f"{h.__name__} certified on no probe at all"


# --- the cap (Gate 2) ---------------------------------------------------------


def test_the_search_is_capped_by_the_DECLARED_size(monkeypatch):
    """The work is bounded by the one quantity the user wrote down and the
    one the cost tracks. Above the cap the search does not run at all —
    silently, like every other declining path here — and the withholding
    the run already carries is unchanged."""
    n = P._CERT_MAX_ELEMENTS + 1

    def h():
        x = any_array((n,), "float64", (0.0, 1.0))
        y = x * 2.0
        assume(y >= 0.5)
        return (assert_(x <= -1.0),)

    assert P._declared_element_count(trace(h).jaxpr) == n
    capped = _prop(h)
    assert capped.region_inhabited is False
    assert capped.obligations[0].status == "unknown"

    # the cap is what did it: raise it and the same query certifies. This
    # is also the measurement of the cap's COST — a sound refutation, lost
    # to a size bound rather than to anything about the region.
    monkeypatch.setattr(P, "_CERT_MAX_ELEMENTS", n)
    lifted = _prop(h)
    assert lifted.region_inhabited is True
    assert lifted.obligations[0].status == "violated-over-set"


def test_the_probe_count_scales_down_with_the_declared_size():
    """The second bound, and the one that made the cap affordable: a
    failing search at the size cap walked the whole 16-point grid and cost
    **469 ms against a 23 ms propagation — 95% of the whole `check()`
    pipeline** (`scratchpad/cert/RESULTS_cap.txt`). Scaling the probe
    count by the declared size brings that to ~95 ms while small
    declarations keep the full grid.

    The FLOOR is 3 and not a fitted number: probes 0, 1 and 2 are the
    declared box's low corner, high corner and midpoint, and across the 17
    corpus rows that witness at all the first witnessing probe is one of
    those three in 17 of 17 (`RESULTS_probe_index.txt`).
    """
    assert P._certificate_probe_count(1) == P._PROBE_COUNT
    assert P._certificate_probe_count(P._CERT_PROBE_BUDGET // P._PROBE_COUNT) \
        == P._PROBE_COUNT
    assert P._certificate_probe_count(P._CERT_MAX_ELEMENTS) == P._CERT_MIN_PROBES
    assert P._certificate_probe_count(10**9) == P._CERT_MIN_PROBES
    # monotone non-increasing in the declared size, which is what "bounded
    # by the declared size" has to mean if it means anything
    counts = [P._certificate_probe_count(n) for n in (1, 4, 16, 64, 256,
                                                      1024, 4096, 65536)]
    assert counts == sorted(counts, reverse=True)


def test_the_search_stops_at_the_first_witness(monkeypatch):
    """The other half of the bound: a successful search stops at the first
    witness rather than walking the grid. One point is the whole claim.
    """
    runs = []
    real = P._Propagator.run

    def counting(self, *a, **kw):
        if self.pin is not None:
            runs.append(self.pin)
        return real(self, *a, **kw)

    monkeypatch.setattr(P._Propagator, "run", counting)
    p = _prop(h_inhabited_narrowing)
    assert p.region_inhabited is True
    assert len(runs) < P._PROBE_COUNT, (
        f"the search ran {len(runs)} probes after finding a witness"
    )

    # and the declining run is what the bounds are FOR: it pays the whole
    # grid its declared size earns (a scalar earns all of it)
    runs.clear()
    assert _prop(h_empty_relational).region_inhabited is False
    assert len(runs) == P._certificate_probe_count(2) == P._PROBE_COUNT
