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


def test_a_certificate_never_reaches_a_run_with_nothing_withheld():
    """The search does not even run where there is nothing to lift: a
    query with no definite violation pays nothing and claims nothing."""
    p = _prop(h_discharged_under_uncertified)
    assert p.narrowing_uncertified is True  # the withholding WOULD apply...
    assert p.obligations[0].status == "discharged"
    assert p.region_inhabited is False  # ...and no certificate was minted
    assert not any("CERTIFIED NON-EMPTY" in n for n in p.notes)


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


def test_the_search_stops_at_the_first_witness(monkeypatch):
    """The other half of the bound: a successful search costs ONE probe
    propagation, not `_PROBE_COUNT` of them. One point is the whole claim.
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

    # and the declining run is what the cap is FOR: it pays the full grid
    runs.clear()
    assert _prop(h_empty_relational).region_inhabited is False
    assert len(runs) == P._PROBE_COUNT
