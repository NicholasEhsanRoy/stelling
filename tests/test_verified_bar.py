# SPDX-FileCopyrightText: 2026 Nicholas Ehsan Roy
# SPDX-License-Identifier: Apache-2.0

"""The scatter VERIFIED bar: it fires, and it fires only where intended.

The bar closes the one direction a wrong SMT emission row could not
self-check. A spurious witness is caught by exact-rational replay; a MISSED
violation would mint a false VERIFIED with nothing downstream to catch it.
So solver-path VERIFIED is withheld on obligations whose EMITTED SLICE
carries a barred primitive, and the refutation path — which is
self-checking — stays open.

Three halves are pinned here, because there turned out to be three. A bar
nobody has seen fire is not a mechanism; a bar that fires on interval-only
verdicts would silently withhold verdicts it was never meant to touch; and a
bar scoped to the whole traced query fires on obligations the emission row
never touched, which is the same over-firing one level finer. That third
case is the reconstruction: this module's original fixture WAS such a case
(solver-decided slice ``['sub','ge']``, the scatter on a different,
interval-decided obligation) and its docstring anticipated needing rebuild.
It is kept, inverted, as the regression — see
``test_a_scatter_OFF_the_decided_slice_withholds_nothing``.

**AND A FOURTH, WHICH IS WHERE THE SCOPE COMES FROM.** Narrowing the bar to
one obligation means answering "which obligation, and what is on it", and the
answer must not be a claim the verdict's own inputs can make about
themselves. A predecessor recorded the per-obligation barred set on
``solvers.ObligationEscalation.barred_on_slice`` and the bar read it;
measured, that lost BOTH immunities the whole-query bar had — an empty tuple
is a positive claim nothing validated, and ``make_solver_verdict`` bound its
escalation to its ``closed`` nowhere, so a scatter-free escalation stamped
against a scatter-bearing query returned VERIFIED where the whole-query bar
returned UNKNOWN. The scope is now DERIVED from ``closed`` (re-slicing the
decided obligations), and both directions are pinned below.

**AND A FIFTH: DERIVING THE CONTENTS DID NOT BIND THE PAIRING.** The repair
above claimed it did, on the reasoning that a mispaired index would not slice
— and `test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query`
agreed, because its mispaired index reaches no real obligation of the wrong
query. Its NEIGHBOUR does: two scatter-bearing queries of the same shape, the
escalation of the one whose scatter is on the decided obligation stamped
against the one whose scatter is elsewhere. That index slices, finds no
barred primitive, and cleared the bar (measured VERIFIED on `caac1ee` and
`45cf526`, UNKNOWN on `8e42934`). An index is not evidence about a query.

**AND A SIXTH: THE SCRIPT HASH IS NOT EVIDENCE ABOUT A SLICE.** The fifth
repair re-emitted the re-derived slice and matched the `smt2_sha256` the
invocation records. Emission IS a pure function of (slice, flavour, timeout);
the converse is what the guard needed, and it is FALSE for the one primitive
under the bar. The static-index `scatter` SET row emits NO line, so an
untouched element aliases the operand's term and `s[1] - x[1] <= 0` emits byte
for byte what `x[1] - x[1] <= 0` emits — same sha, barred sets `('scatter',)`
and `()`. The fixture that was supposed to catch this **built away its own
trigger**: it said "the one difference is WHERE" while also introducing a
fresh input and a different predicate, and it was THAT which made the hashes
differ. Rebuilt to differ only in where the scatter sits, it returns VERIFIED
on `eb1ff86`. The narrowing now also requires `slice_sha256`, the fingerprint
of the slice's own primitives and nesting, which the emitted text cannot
carry — see ``test_the_script_hash_alone_cannot_separate_these_two_slices``,
``test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar`` in both
parametrisations, ``test_the_bar_scope_itself_widens_on_the_colliding_pair``,
and the other direction,
``test_the_correct_pairing_still_narrows_and_the_hash_is_why``.

**AND A SEVENTH, WHICH IS NOT A PROPERTY OF THE BAR AT ALL.** Every repair
above narrows or widens a bar; none of them binds
``make_solver_verdict``'s arguments to one query, and nothing else did
either. The file used to record that as "the cost of scoping the bar" —
that a whole-query bar had backstopped the pairing by accident and a scoped
one could not. That backstop covered scatter-bearing queries ONLY, because
those are the only ones any version of the bar looks at: the identical
mispaired VERIFIED on a REFUTED query rides a query with **no barred
primitive anywhere**, on every build including `8e42934`. So the finding is
not a cost of scoping — it is that the three arguments were never bound.
They are bound now for two of the three, by the QUERY PAIRING GATE
(``escalate`` records the query's content hash; assembly recomputes it and
refuses a mismatch). See
``test_the_pairing_gate_closes_the_SCATTER_FREE_row`` — the row that
settles which statement is true —
``test_the_pairing_gate_refuses_the_mispairing_the_bar_only_narrows``, and
``test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation`` for
the residue. The bar's own mispairing tests satisfy the gate by hand
(``_past_the_pairing_gate``) so that neither mechanism can hide the other's
failure.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

import stelling.verdict as V
from stelling.harness import any_array, assert_, trace
from stelling.preconditions import check


@pytest.fixture(autouse=True)
def _x64():
    """Scope x64 to this module. Setting it at import leaks float64 into every
    later-run module in the same process — measured here the same way the
    suite already documents: it flipped test_transcribe's cross-process
    hash-stability test, which passes alone and fails in-suite."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def _obl_solves(v) -> int:
    sol = v.stamp.solver if v.stamp else None
    sols = sol if isinstance(sol, tuple) else (sol,)
    return len([s for s in sols if s and s.invoked and "widen" not in s.reason])


def _scatter_ON_the_decided_slice():
    """A query whose SOLVER-DECIDED obligation carries `scatter` on its
    emitted slice — the shape the bar exists for.

    Obligation 0 is `s[1] - x[1] <= 0` where `s = x.at[0].set(0.5)`. It is
    exactly true (element 1 is untouched by the write), but intervals are
    correlation-blind and propagate `[-1, 1]`, so it escalates; the emitted
    slice is measured as `['broadcast_in_dim', 'scatter', 'slice', 'squeeze',
    'slice', 'squeeze', 'sub', 'le']` — the `scatter` really is on the slice
    the solver was asked about. Obligation 1 is settled by intervals. Both
    discharge, so the verdict would be VERIFIED but for the bar.

    THIS SCENARIO DEPENDS ON AN IMPRECISION, and says so deliberately: it
    needs the correlation between `s[1]` and `x[1]` to stay invisible to the
    abstraction so that escalation runs and the bar is reached. A refinement
    that recovers that correlation (affine arithmetic across the scatter row
    is the obvious one) will decide obligation 0 without a solver, and this
    test will fail. The repair is a DIFFERENT construction that still forces
    a solver-decided obligation WITH `scatter` on its slice — not deleting
    the assertion, which would leave the bar untested while looking green.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(s[1] - x[1] <= 0.0), assert_(s >= 0.0))


def _scatter_OFF_the_decided_slice():
    """A query that CONTAINS scatter whose solver-decided obligation does not
    touch it — the false bar this module's predecessor asserted.

    Obligation 0 is `y - y >= 0`: undecidable by intervals (correlation-blind
    again) and trivial for SMT, and its emitted slice is measured as
    `['sub', 'ge']` — no scatter anywhere on it. Obligation 1 holds the
    scatter and is settled by intervals, so no emission row was consulted
    about it either. Nothing in this verdict can be wrong because the scatter
    emission is wrong.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0), assert_(s >= 0.0))


def test_the_bar_withholds_a_solver_path_verified_on_a_scatter_slice():
    # Deliberately NOT skipif-guarded on the bar being non-empty. A skip is
    # not a failure, so a guarded test cannot be mutation-proved: emptying
    # VERIFIED_BARRED_PRIMITIVES would silently skip this rather than fail it,
    # which is exactly the false negative the mutation norm exists to catch.
    # When the principal lifts the bar this test fails loudly and is updated
    # deliberately — the same one-identifiable-place discipline as the bar.
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    v = check(_scatter_ON_the_decided_slice,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "intervals settled everything, so the bar never applies and this "
        "test does not exercise the firing direction"
    )
    assert all(o.status == "discharged" for o in v.obligations), (
        "the scenario must reach the bar with every obligation discharged — "
        "otherwise UNKNOWN would prove nothing about the bar"
    )
    assert v.status == "UNKNOWN"
    withheld = [n for n in v.notes if "VERIFIED withheld" in n]
    assert withheld
    assert all("assert #0" in n for n in withheld), (
        f"the note must NAME the obligation whose slice carries the barred "
        f"primitive rather than the query as a whole: {withheld}"
    )


def test_the_scatter_really_is_on_the_decided_slice():
    """ANTI-VACUITY for the test above (Norm C). If the fixture's scatter
    drifted OFF the escalated obligation's slice, the test above would stop
    measuring the bar's scope while still looking green under a fallback.
    Assert the slice itself."""
    from stelling.obligation import (
        DeclinedObligation,
        slice_unknown_obligations,
    )
    from stelling.propagate import interval_env, propagate

    closed = trace(_scatter_ON_the_decided_slice)
    p = propagate(closed)
    env = interval_env(closed)
    sliced = [
        s for s in slice_unknown_obligations(closed, p, env)
        if not isinstance(s, DeclinedObligation)
    ]
    assert sliced, "no obligation was sliced; nothing reaches the solver"
    prims = {str(e.primitive) for s in sliced for e in s.eqns}
    assert prims & V.VERIFIED_BARRED_PRIMITIVES, (
        f"no barred primitive on any escalated slice ({sorted(prims)}) — the "
        f"bar test above is not measuring what it claims"
    )


def test_a_scatter_OFF_the_decided_slice_withholds_nothing():
    """THE DIRECTION THE SLICE-SCOPING CHANGED, and the reconstruction of
    this module's original fixture.

    The bar used to read `_barred_primitives(closed)` — the WHOLE traced
    query — so a scatter anywhere in the jaxpr withheld a VERIFIED resting
    entirely on obligations the scatter emission row never touched. Measured
    on exactly this query: solver-decided slice `['sub','ge']`, verdict
    UNKNOWN. The emission row cannot be wrong about an obligation it was not
    asked, so there was nothing to withhold.
    """
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed = trace(_scatter_OFF_the_decided_slice)
    assert any(str(e.primitive) == "scatter" for e in closed.jaxpr.eqns), (
        "this test is vacuous unless the query really does contain scatter"
    )
    assert V._barred_primitives(closed), (
        "the WHOLE-QUERY barred set is empty on this fixture, so it cannot "
        "distinguish a slice-scoped bar from a whole-query one"
    )
    v = check(_scatter_OFF_the_decided_slice,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "nothing was solver-decided, so the bar was never consulted and this "
        "test does not exercise the scoping"
    )
    assert v.status == "VERIFIED", (
        f"{v.status}: a VERIFIED resting on a scatter-free slice was "
        f"withheld — the bar is scoped to the query again, not to the slice"
    )
    assert not any("VERIFIED withheld" in n for n in v.notes)


# ── the scope's PROVENANCE: its CONTENTS derived, its DOMAIN a precondition ──
#
# Everything below stamps a verdict through `make_solver_verdict` directly
# rather than through `check`, because that is the surface the defects were
# on: it is public, it takes `closed`, `propagation` and `escalation` as three
# independent arguments, and it gates their pairing on semantics, ieee,
# constrained-assume and ledger provenance — but on nothing that ties the
# escalation to the query.
#
# WHAT IS PINNED HERE IS NOT IMMUNITY TO A FORGED ESCALATION, and the section
# heading used to read as if it were ("never read off a record", full stop).
# No barred PRIMITIVE is read off a record — that is real, and it is what the
# deleted `barred_on_slice` field cost. But WHICH obligations the bar is asked
# about does come from the records, so the bar inherits
# `make_solver_verdict`'s documented precondition that its escalation came
# from `escalate()` on this query. Hardening against a violation of it would
# defend nothing: `stelling.verdict.Verdict` is public and a plain frozen
# dataclass, so a caller who can hand-build an `ObligationEscalation` can
# hand-build the VERIFIED and never reach this function. These tests measure
# what a MISPAIRED honest assembly does, and that the domain cannot disagree
# with the discharge about one record — not what an adversary cannot do.

VERSIONS = dict(
    stelling_version="0.0.0", jax_version="0.0.0", precision_config="x64"
)


def _scatter_free():
    """No scatter anywhere, one solver-decided obligation, VERIFIED."""
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0),)


def _two_solver_decided_obligations():
    """BOTH obligations escalate and BOTH discharge, and only #0's slice
    carries scatter. Measured slices: #0 is
    ``['broadcast_in_dim','le','scatter','slice','slice','squeeze','squeeze',
    'sub']`` and #1 is ``['add','le','sub']``.

    #1 is `y + 1.0 - y <= 1.0`: exactly 1.0 in ℝ, but the correlation is
    invisible to intervals ([2,3] − [1,2] = [0,2]), so it escalates instead of
    being settled — which is what makes it a SECOND deciding obligation and
    not just a second obligation."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(s[1] - x[1] <= 0.0), assert_(y + 1.0 - y <= 1.0))


def _stamped(build):
    """(closed, propagation, escalation) for a build, un-assembled."""
    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate

    closed = trace(build)
    prop = propagate(closed)
    return closed, prop, escalate(closed, prop,
                                  SolverConfig(timeout_ms=20000))


def _verdict_status(closed, prop, esc):
    from stelling.solvers import make_solver_verdict

    return make_solver_verdict(closed, prop, esc, **VERSIONS).status


def _past_the_pairing_gate(esc, closed):
    """The mispaired escalation with its recorded query hash OVERWRITTEN to
    the query it is about to be stamped against — i.e. the pairing gate
    deliberately satisfied by hand.

    EVERY BAR MISPAIRING TEST BELOW GOES THROUGH THIS, and it is not a
    weakening: it is what keeps them measuring the BAR. Since the query
    pairing gate landed (`solvers.make_solver_verdict`, the fourth
    `MispairedEscalationError`), a genuinely mispaired assembly does not reach
    the bar at all — it raises. A test that just asserted the raise would have
    stopped measuring `_bar_scope` entirely, which is could-not-fail shape #7:
    a fixture that never reaches the guard's condition. So the gate is
    bypassed HERE, explicitly and in one place, and the bar is measured as the
    SECOND, anti-correlated mechanism it is: the gate keys on the query's
    content hash, the bar on the decided slice's fingerprint and script, and
    neither derives from the other.

    That the gate itself fires on these same fixtures WITHOUT this bypass is
    `test_the_pairing_gate_refuses_the_mispairing_the_bar_only_narrows` and
    `test_the_pairing_gate_closes_the_SCATTER_FREE_row`.
    """
    import dataclasses

    return dataclasses.replace(esc, query_sha256=closed.content_hash())


def test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query():
    """THE MISPAIRING THE WHOLE-QUERY BAR WAS IMMUNE TO, and the reason the
    scope is derived rather than recorded.

    Nothing in `make_solver_verdict` binds `escalation` to `closed`. Stamp a
    genuine scatter-free escalation against a scatter-bearing query and the
    bar must still fire: the whole-query bar it replaced read `closed`
    directly and returned UNKNOWN here (measured on 8e42934), and a
    slice-scoped bar that reads its scope off the escalation returned
    VERIFIED. Narrowing the scope must not cost this.

    THIS FIXTURE IS THE MISPAIRING THAT FAILS SAFE FOR THE WEAKER REASON, and
    for a while it was the only one, which is how a live hole stayed green.
    Its `clean` query has ONE obligation, so the decided index 0 is the only
    index there is and the `dirty` query's obligation #0 carries the scatter
    itself — the bar fires whatever mechanism is asked. The arrangement that
    separates "the index exists here" from "the escalation is about here" is
    `test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar`, and it is
    where the mechanism this test cannot see is measured.
    """
    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    dirty, _, _ = _stamped(_scatter_ON_the_decided_slice)
    clean, prop, esc = _stamped(_scatter_free)
    assert V._barred_primitives(dirty) and not V._barred_primitives(clean), (
        "the two queries do not differ in the whole-query barred set, so this "
        "test cannot distinguish a query-derived scope from a recorded one"
    )
    assert make_solver_verdict(clean, prop, esc, **VERSIONS).status == (
        "VERIFIED"
    ), "the correctly-paired assembly does not VERIFY; the fixture is wrong"

    v = make_solver_verdict(dirty, prop, _past_the_pairing_gate(esc, dirty),
                            **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: an escalation carrying no scatter cleared the bar on a "
        f"query that does — the bar's scope is being read off the escalation "
        f"instead of derived from the query it is stamped against"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def _scatter_ELSEWHERE_same_shape():
    """THE NEAREST NEIGHBOUR of `_scatter_ON_the_decided_slice`, and the
    difference is WHERE THE SCATTER SITS AND NOTHING ELSE.

    Same declared inputs, same two obligations, same predicate on #0, same
    `s >= 0` on #1. The single edit is that #0 reads `x[1]` where
    `_scatter_ON_the_decided_slice` reads `s[1]` — so the scatter is on #1
    alone, and #0's emitted slice carries none.

    THE PREDECESSOR OF THIS FIXTURE BUILT AWAY ITS OWN TRIGGER. It said "the
    one difference is WHERE" while ALSO introducing a fresh scalar input `y`
    and a different predicate (`y - y >= 0`), and it was that difference — not
    the scatter's location — that made the two queries emit different scripts.
    So the test below passed on `eb1ff86`, whose narrowing keyed on the script
    hash alone, while the defect it was written for was live: with the fixture
    differing only in WHERE, `eb1ff86` returns VERIFIED. See
    `test_the_script_hash_alone_cannot_separate_these_two_slices`, which
    measures the byte-equality that makes this the hard case.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(x[1] - x[1] <= 0.0), assert_(s >= 0.0))


def _scatter_ELSEWHERE_different_predicate():
    """The WEAKER neighbour, kept because it separates the two keys.

    Same arrangement as `_scatter_ELSEWHERE_same_shape` — scatter on #1, #0
    solver-decided and scatter-free — but #0 is a different claim over a fresh
    input, so its emitted slice (`['ge','sub']`) emits a DIFFERENT script.
    That difference is enough for the script hash alone, which is why this
    shape was green on `eb1ff86` while its sibling above was not.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(y - y >= 0.0), assert_(s >= 0.0))


def test_the_script_hash_alone_cannot_separate_these_two_slices():
    """THE COLLISION, AS A MEASUREMENT — the premise the bar's key rests on.

    `_evidence_is_about` re-emits the re-derived slice and compares hashes.
    That can only witness WHICH SLICE was emitted from if emission is
    injective, and it is not: the static-index `scatter` SET row appends no
    line, so an element the write did not touch aliases the operand's term.
    Here the scatter-BEARING slice of `_scatter_ON_the_decided_slice`#0 and
    the scatter-FREE slice of `_scatter_ELSEWHERE_same_shape`#0 emit the same
    bytes, and their barred sets differ.

    If this test ever goes red because the two scripts stopped colliding, the
    bar has not been repaired — the fixture has drifted back to the shape that
    could not fail, and `test_a_mispaired_query_that_still_SLICES...` is again
    passing for the weaker reason.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit, slice_fingerprint

    on_closed = trace(_scatter_ON_the_decided_slice)
    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    on_sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    el_sl = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(on_sl, DeclinedObligation)
    assert not isinstance(el_sl, DeclinedObligation)

    assert V._barred_in_eqns(on_sl.eqns) == ("scatter",)
    assert V._barred_in_eqns(el_sl.eqns) == ()
    assert emit(on_sl, "z3", 20000).sha256 == emit(el_sl, "z3", 20000).sha256, (
        "the two slices no longer emit byte-identical scripts, so the script "
        "hash WOULD separate them and the mispairing test below is measuring "
        "the weaker key"
    )
    assert slice_fingerprint(on_sl) != slice_fingerprint(el_sl), (
        "the slice fingerprint does not separate a scatter-bearing slice from "
        "a scatter-free one, so the bar's key distinguishes nothing that the "
        "script hash did not already"
    )


@pytest.mark.parametrize("elsewhere,key", [
    (_scatter_ELSEWHERE_same_shape, "the slice fingerprint"),
    (_scatter_ELSEWHERE_different_predicate, "the script hash"),
])
def test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar(
    elsewhere, key
):
    """THE MISPAIRING THE INDEX-ONLY DOMAIN DID NOT CATCH — and, in its first
    parametrisation, the one the SCRIPT-HASH-ONLY narrowing did not catch
    either.

    `test_a_scatter_free_escalation_cannot_clear_a_scatter_BEARING_query`
    passes on the arrangement where the mispaired index does not reach a real
    obligation of the wrong query. These are its neighbours: BOTH queries
    carry `scatter`, both have two obligations, and the decided index 0 exists
    in both. So the re-slice does not decline — it succeeds, on the wrong
    query, and finds no barred primitive.

    THE TWO PARAMETRISATIONS NEED DIFFERENT KEYS, which is why both are here.

        elsewhere = same shape (scatter's LOCATION is the only difference)
            8e42934 UNKNOWN | caac1ee VERIFIED | 45cf526 VERIFIED
            eb1ff86 VERIFIED | here UNKNOWN
            separated only by `slice_sha256`; the two scripts are byte-equal

        elsewhere = different predicate (a fresh input and another claim)
            8e42934 UNKNOWN | caac1ee VERIFIED | 45cf526 VERIFIED
            eb1ff86 UNKNOWN | here UNKNOWN
            separated by `smt2_sha256` already

    The repair is `verdict._evidence_is_about`: an obligation narrows the bar
    only when a recorded invocation reproduces BOTH the script hash and the
    slice fingerprint from the slice re-derived out of the query being
    stamped.
    """
    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    on_closed, on_prop, on_esc = _stamped(_scatter_ON_the_decided_slice)
    el_closed, _, _ = _stamped(elsewhere)

    # the pair must be indistinguishable to every WEAKER test than the one
    # this pins, or it does not measure the mechanism it claims to
    assert V._barred_primitives(on_closed) == V._barred_primitives(el_closed), (
        "the two queries differ in their whole-query barred set, so a "
        "whole-query bar would already separate them and this test would not "
        "be measuring the slice-scoped one"
    )
    decided = {r.index: r.invocations
               for r in on_esc.records if r.outcome == "discharged"}
    assert set(decided) == {0}, decided
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    again = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(again, DeclinedObligation), (
        "the mispaired index DECLINES to slice out of the wrong query, so "
        "this test is the case the fallback already caught and not the one "
        "it missed"
    )
    assert not (V._barred_in_eqns(again.eqns)), (
        "the mispaired query's obligation #0 carries a barred primitive on "
        "its own slice, so the bar would fire for a reason having nothing to "
        "do with the mispairing"
    )

    assert make_solver_verdict(on_closed, on_prop, on_esc, **VERSIONS).status == (
        "UNKNOWN"
    ), "the correctly-paired assembly does not bar; the fixture is wrong"

    v = make_solver_verdict(el_closed, on_prop,
                            _past_the_pairing_gate(on_esc, el_closed),
                            **VERSIONS)
    assert [o.status for o in v.obligations] == ["discharged"] * len(
        v.obligations
    ), (
        "the mispaired assembly did not reach a would-be VERIFIED, so there "
        "is nothing for the bar to withhold and this test proves nothing"
    )
    assert v.status == "UNKNOWN", (
        f"{v.status}: an escalation produced on a DIFFERENT scatter-bearing "
        f"query cleared the bar. This arrangement is separated by {key}, so "
        f"that is the part of `_evidence_is_about` that has stopped working"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def _scatter_ELSEWHERE_and_actually_REFUTED():
    """`_scatter_ELSEWHERE_same_shape` with its SECOND obligation made FALSE.

    #0 is the byte-colliding `x[1] - x[1] <= 0` again; #1 is `s >= 0.5`, which
    fails at `x = [0, 0, 0]` (then `s = [0.5, 0, 0]`). Checked honestly this
    query is REFUTED — so a VERIFIED stamped on it is not a withheld claim
    becoming available, it is a FALSE claim.
    """
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(x[1] - x[1] <= 0.0), assert_(s >= 0.5))


def test_the_collision_could_mint_a_VERIFIED_on_a_REFUTED_query():
    """WHAT THE COLLISION ACTUALLY COSTS, and the case that settles whether
    the narrowing was a policy slip or a soundness defect.

    The mispairing tests above show the bar clearing on a query the escalation
    is not about. On its own that could be read as harmless: the script hash
    does pin the TEXT, so an `unsat` about that text is an `unsat` about the
    obligation this query's own slice emits. What it does not pin is the rest
    of the verdict — the obligations the mispaired PROPAGATION decided.

    So here the mispaired query's second obligation is FALSE. Measured:

        EL checked honestly                REFUTED (all four builds)
        ON escalation+propagation, EL query
            8e42934  UNKNOWN   (the whole-query bar)
            caac1ee  VERIFIED
            45cf526  VERIFIED
            eb1ff86  VERIFIED   <- a false VERIFIED on a REFUTED query
            here     UNKNOWN

    Since the query pairing gate landed this mispairing no longer reaches the
    bar at all — it raises `MispairedEscalationError` — so the assembly below
    goes through `_past_the_pairing_gate`, which satisfies the gate by hand
    and leaves the bar as the thing being measured. The `here UNKNOWN` row is
    the bar's, and it is the row that would still hold if the gate were
    deleted. The whole-query bar's apparent immunity on this shape was never a
    mechanism: it covered scatter-bearing queries only, and the same false
    VERIFIED rides a scatter-free query on every build — see
    `test_the_pairing_gate_closes_the_SCATTER_FREE_row`.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit
    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    on_closed, on_prop, on_esc = _stamped(_scatter_ON_the_decided_slice)
    el_closed, el_prop, el_esc = _stamped(_scatter_ELSEWHERE_and_actually_REFUTED)

    # the claim that makes this a FALSE verified rather than a withheld one
    assert make_solver_verdict(el_closed, el_prop, el_esc, **VERSIONS).status == (
        "REFUTED"
    ), "the mispaired query is not actually false, so nothing here is a lie"

    # and the collision is what removes the backstop: only the fingerprint
    # separates the two obligation #0 slices
    on_sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    el_sl = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(el_sl, DeclinedObligation)
    assert emit(on_sl, "z3", 20000).sha256 == emit(el_sl, "z3", 20000).sha256, (
        "the two slices no longer emit the same script, so the script hash "
        "already separates them and this test is not measuring the collision"
    )

    v = make_solver_verdict(el_closed, on_prop,
                            _past_the_pairing_gate(on_esc, el_closed),
                            **VERSIONS)
    assert [o.status for o in v.obligations] == ["discharged"] * len(
        v.obligations
    ), "no would-be VERIFIED was reached, so there is nothing to withhold"
    assert v.status == "UNKNOWN", (
        f"{v.status}: a mispaired assembly issued this verdict on a query "
        f"whose honest verdict is REFUTED. The whole-query bar at `8e42934` "
        f"returned UNKNOWN here; the narrowing has removed that backstop for "
        f"the shape whose emitted script does not distinguish the slices"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def _scatter_free_TRUE_two_obligations():
    """NO BARRED PRIMITIVE ANYWHERE, two obligations, both solver-decided,
    both true. The pair below is the same mispairing as the scatter fixtures
    on a query no version of the bar has ever looked at."""
    x = any_array((3,), "float64", (0.0, 1.0))
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(x[1] - x[1] <= 0.0), assert_(y + 1.0 - y <= 1.0))


def _scatter_free_REFUTED_two_obligations():
    """Its twin with #1 made FALSE: `y + 1 - y` is exactly 1.0, never ≤ 0.5.
    Same #0, same declared inputs, no scatter."""
    x = any_array((3,), "float64", (0.0, 1.0))
    y = any_array((), "float64", (1.0, 2.0))
    return (assert_(x[1] - x[1] <= 0.0), assert_(y + 1.0 - y <= 0.5))


def _scatter_ELSEWHERE_identical_decided_slice():
    """`_scatter_ELSEWHERE_and_actually_REFUTED` with obligation #0 the SAME
    EXPRESSION as the query it will be mispaired with — not merely one that
    emits the same bytes. Its #1 is the false `s >= 0.5`."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(x[1] - x[1] <= 0.0), assert_(s >= 0.5))


def _scatter_ELSEWHERE_identical_decided_slice_TRUE():
    """Its honest twin: same #0, and a #1 that holds."""
    x = any_array((3,), "float64", (0.0, 1.0))
    s = x.at[0].set(0.5)
    return (assert_(x[1] - x[1] <= 0.0), assert_(s >= 0.0))


def test_the_pairing_gate_refuses_the_mispairing_the_bar_only_narrows():
    """WHAT THE BAR DOES NOT DO, AND WHAT NOW DOES IT — this test was
    `test_the_LIMIT_of_this_bar_when_the_decided_slice_is_genuinely_the_same`,
    which asserted the PERMISSIVE outcome and instructed its successor to
    rewrite it around whatever started catching the general mispairing. That
    is the query pairing gate, and this is the rewrite.

    THE FRAMING THE OLD VERSION HAD WRONG, corrected because it was measured
    and not argued. It said the false VERIFIED here was "the cost of scoping
    the bar at all" — that the whole-query bar had BACKSTOPPED
    `make_solver_verdict`'s pairing precondition and a narrowed bar could not.
    The backstop was real but it was a COINCIDENCE OF SCOPE, not a mechanism:
    it only ever covered queries that carry a barred primitive. The same false
    VERIFIED, from the same mispairing, is reachable on a query with NO
    SCATTER ANYWHERE — where every version of the bar, whole-query included,
    is silent — on every build including `8e42934`. That is
    `test_the_pairing_gate_closes_the_SCATTER_FREE_row`, and it is why the
    correct statement is not "scoping cost a backstop" but **scoping revealed
    that `make_solver_verdict` never bound its three arguments to one query.**

        mispaired assembly, IDENTICAL decided slice, REFUTED query
            8e42934  UNKNOWN (the whole-query bar, for an unrelated reason)
            caac1ee / 45cf526 / eb1ff86 / f5280cf   VERIFIED
            here     MispairedEscalationError

        the SAME mispairing on a SCATTER-FREE query
            8e42934 / eb1ff86 / f5280cf   VERIFIED  (no bar fires at all)
            here     MispairedEscalationError

    WHAT THE GATE KEYS ON, AND WHY IT IS NOT THE BAR AGAIN: `escalate` records
    `ClosedJaxpr.content_hash()` of the query it ran on, and assembly
    recomputes it from the `closed` it is handed. Here the two queries differ
    only in obligation #1 (`s >= 0.0` vs `s >= 0.5`) — an obligation the bar
    never reads, whose slices are identical by every measure the bar has, and
    the content hash separates them. Asserted below, so a hash that stopped
    covering the whole query would fail here rather than pass quietly.

    THE BAR IS STILL LOAD-BEARING and this test does not replace it: it fires
    on a correctly-paired query whose decided slice carries the unaudited row,
    which no pairing gate can see. The two are anti-correlated by
    construction — whole-query content hash vs per-slice fingerprint plus
    script — and the tests above measure the bar with the gate deliberately
    bypassed (`_past_the_pairing_gate`) so that neither can hide the other's
    failure.
    """
    import dataclasses

    from stelling.obligation import slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit, slice_fingerprint
    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    on_closed, on_prop, on_esc = _stamped(
        _scatter_ELSEWHERE_identical_decided_slice_TRUE)
    el_closed, el_prop, el_esc = _stamped(
        _scatter_ELSEWHERE_identical_decided_slice)

    assert make_solver_verdict(el_closed, el_prop, el_esc, **VERSIONS).status == (
        "REFUTED"
    ), "the mispaired query is not false, so there is no false VERIFIED here"
    assert make_solver_verdict(on_closed, on_prop, on_esc, **VERSIONS).status == (
        "VERIFIED"
    ), (
        "the honest twin does not VERIFY, so the mispaired assembly could not "
        "have minted a VERIFIED either and this test measures nothing"
    )

    # the two decided slices are the same by every measure the BAR has — this
    # is what makes the bar powerless here and the gate necessary
    on_sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    el_sl = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert emit(on_sl, "z3", 20000).sha256 == emit(el_sl, "z3", 20000).sha256
    assert slice_fingerprint(on_sl) == slice_fingerprint(el_sl)
    assert V._barred_in_eqns(on_sl.eqns) == () == V._barred_in_eqns(el_sl.eqns)

    # ... and the QUERY hash is not the same, which is the gate's whole key
    assert on_closed.content_hash() != el_closed.content_hash(), (
        "the two queries have the same content hash, so the pairing gate "
        "cannot separate them and this fixture cannot measure it"
    )
    assert on_esc.query_sha256 == on_closed.content_hash(), (
        "`escalate` did not record the query it ran on, so the gate has "
        "nothing to compare and passes vacuously"
    )

    with pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(el_closed, on_prop, on_esc, **VERSIONS)
    assert "not the same query" in str(exc.value)

    # NON-VACUITY, and it is the half that matters: with the gate satisfied by
    # hand the assembly goes through and returns exactly the false VERIFIED
    # this gate exists to refuse. So the refusal above is the gate firing, not
    # some unrelated guard, and not a blanket refusal of mispaired shapes.
    forged = dataclasses.replace(on_esc, query_sha256=el_closed.content_hash())
    v = make_solver_verdict(el_closed, on_prop, forged, **VERSIONS)
    assert v.status == "VERIFIED", (
        f"{v.status}: with the pairing hash forged to match, the assembly no "
        f"longer reaches the false VERIFIED — so the refusal above is not "
        f"this gate and this test is measuring something else"
    )


def test_the_pairing_gate_closes_the_SCATTER_FREE_row():
    """THE ROW THAT SETTLES WHAT THE WHOLE-QUERY BAR ACTUALLY WAS. Its
    "backstop" covered scatter-bearing queries only, so it was a coincidence
    of scope; the identical false VERIFIED rides a query with no barred
    primitive anywhere, where no version of the bar has ever fired.

    Two queries, neither carrying `scatter`, agreeing on obligation #0 and
    differing on #1 (`y + 1 - y <= 1.0`, true, vs `<= 0.5`, false). Stamp the
    true one's propagation and escalation against the false one and every
    obligation comes back `discharged`: VERIFIED on a REFUTED query, measured
    on `8e42934`, `eb1ff86` and `f5280cf` alike.

    This is the test that makes the P1 claim falsifiable. If the bar were the
    mechanism, this row would be UNKNOWN somewhere in that list; it is
    VERIFIED everywhere, which is why the repair is a pairing gate and not a
    wider bar. `V._barred_primitives` is asserted empty on both queries below
    so the fixture cannot silently acquire a scatter and start passing for the
    bar's reason.
    """
    import dataclasses

    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    on_closed, on_prop, on_esc = _stamped(_scatter_free_TRUE_two_obligations)
    el_closed, el_prop, el_esc = _stamped(_scatter_free_REFUTED_two_obligations)

    assert V._barred_primitives(on_closed) == () == V._barred_primitives(
        el_closed
    ), (
        "a barred primitive has appeared in one of these queries, so the bar "
        "can fire here and this row no longer isolates the pairing gate"
    )
    assert make_solver_verdict(on_closed, on_prop, on_esc, **VERSIONS).status == (
        "VERIFIED"
    ), "the honest twin does not VERIFY; the fixture is wrong"
    assert make_solver_verdict(el_closed, el_prop, el_esc, **VERSIONS).status == (
        "REFUTED"
    ), "the mispaired query is not false, so nothing minted here is a lie"

    with pytest.raises(MispairedEscalationError):
        make_solver_verdict(el_closed, on_prop, on_esc, **VERSIONS)

    # and the same non-vacuity control: forge the hash and the false VERIFIED
    # comes right back, so the gate is the only thing standing here
    forged = dataclasses.replace(on_esc, query_sha256=el_closed.content_hash())
    v = make_solver_verdict(el_closed, on_prop, forged, **VERSIONS)
    assert v.status == "VERIFIED" and not any(
        "VERIFIED withheld" in n for n in v.notes
    ), (
        f"{v.status}: with the pairing hash forged the assembly did not mint "
        f"the false VERIFIED, so this row is being closed by something other "
        f"than the pairing gate"
    )


def test_the_pairing_gate_binds_the_ESCALATION_and_not_the_propagation():
    """THE RESIDUE, STATED AND MEASURED rather than left for the next audit.

    The gate binds two of `make_solver_verdict`'s three arguments: `closed`
    and `escalation`. It does NOT bind `propagation`, and the reason is
    mechanical — `Propagation` is defined in `stelling.propagate`, which this
    repair was required to leave at zero line delta, so there is no field on
    it to record the query in.

    What is left open, exactly: an assembly of (query A, propagation of query
    B, escalation of query A). The obligations come from B, the discharges
    from A by index. Measured below — it assembles, and the gate does not stop
    it. What is NOT left open is the shape that actually mints a false
    VERIFIED out of a cached escalation, because the discharges have to come
    from somewhere and the gate refuses them.

    Kept as a live measurement rather than a comment so that closing it later
    is a test that goes red, not an archaeology exercise. IF THIS TEST FAILS
    because the assembly no longer returns VERIFIED, the residue has been
    closed: say so in `SOUNDNESS.md` and rewrite this around the mechanism
    that closed it, exactly as its predecessor instructed."""
    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    true_closed, true_prop, true_esc = _stamped(_scatter_free_TRUE_two_obligations)
    false_closed, false_prop, false_esc = _stamped(
        _scatter_free_REFUTED_two_obligations)

    assert false_prop != true_prop, "the two propagations are equal; nothing to mix"
    assert make_solver_verdict(
        false_closed, false_prop, false_esc, **VERSIONS
    ).status == "REFUTED", "the false query is not false; the fixture is wrong"

    # THE COVERED DIRECTION: the escalation is the thing that discharges, and
    # it cannot travel to another query.
    with pytest.raises(MispairedEscalationError):
        make_solver_verdict(false_closed, false_prop, true_esc, **VERSIONS)

    # THE RESIDUE, MEASURED: the propagation can. Obligations come from the
    # FALSE query, the discharges from the TRUE query's escalation by index,
    # and the stamp names the TRUE query — a verdict reporting the false
    # query's obligations as discharged under the true query's hash.
    v = make_solver_verdict(true_closed, false_prop, true_esc, **VERSIONS)
    assert v.status == "VERIFIED", (
        f"{v.status}: the mixed-propagation assembly no longer VERIFIES — "
        f"see this docstring's last paragraph before changing this line"
    )
    assert v.stamp.query_content_hash == true_closed.content_hash(), (
        "the stamp does not name the query it was assembled against, which "
        "would be a different defect from the one this test discloses"
    )
    assert [o.source_info for o in v.obligations] == [
        o.source_info for o in false_prop.obligations
    ], (
        "the reported obligations are no longer the mispaired propagation's, "
        "so the misattribution this test measures is not happening"
    )


def test_every_escalate_return_site_records_the_query():
    """THE GATE'S COVERAGE, PINNED AT ITS PRODUCER — a return path out of
    `escalate` that forgot `query_sha256` is an escalation the gate cannot
    check, and the empty default makes that a SILENT hole rather than a loud
    one on every path except the one the gate refuses.

    Pinned by construction, not by listing today's paths: every `return
    Escalation(` in `stelling/solvers.py` must pass `query_sha256`, counted
    off the source, and each REACHABLE path is then driven and asserted to
    carry this query's own hash. A new return site that omits it fails the
    count; a site that passes a WRONG value fails the drives.
    """
    import inspect

    from stelling import solvers as S

    src = inspect.getsource(S.escalate)
    sites = []
    for start in range(len(src)):
        if not src.startswith("return Escalation(", start):
            continue
        i = start + len("return Escalation(")
        depth = 1
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        sites.append(src[start:i])
    assert len(sites) >= 5, (
        f"found {len(sites)} `return Escalation(` site(s) in escalate(); the "
        f"scan has stopped matching the source and this test is vacuous"
    )
    missing = [i for i, body in enumerate(sites) if "query_sha256" not in body]
    assert not missing, (
        f"`return Escalation(` site(s) {missing} in escalate() do not record "
        f"`query_sha256`. An escalation without it cannot be paired to a "
        f"query, and the gate's default-empty refusal only fires once it "
        f"reaches assembly — every path must carry it at birth"
    )

    # ... and the reachable paths, driven
    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate

    def _nothing_unknown():
        x = any_array((), "float64", (1.0, 2.0))
        return assert_(x - x <= 0.0)  # intervals settle it; no escalation work

    for build, label in (
        (_scatter_ON_the_decided_slice, "solver work"),
        (_nothing_unknown, "nothing to escalate"),
    ):
        closed = trace(build)
        esc = escalate(closed, propagate(closed), SolverConfig(timeout_ms=20000))
        assert esc.query_sha256 == closed.content_hash(), (
            f"{label}: escalate() recorded {esc.query_sha256!r} for a query "
            f"hashing to {closed.content_hash()!r}"
        )


class _Unhashable:
    """A `closed` in every respect except that `content_hash()` RAISES —
    which is what `_query_sha256` turns into `""`."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def content_hash(self):
        raise RuntimeError("this query's content hash cannot be taken")


@pytest.mark.parametrize("recorded", ["", "an-honest-looking-hash"])
def test_the_pairing_gate_refuses_an_EMPTY_hash_and_not_only_a_DIFFERENT_one(
    recorded,
):
    """THE LEG WHERE EQUALITY WAS NOT THE PROPERTY WANTED.

    Both sides of the gate come from `_query_sha256`, which returns `""` when
    `ClosedJaxpr.content_hash()` raises. So a `closed` that cannot be hashed
    and an escalation that recorded nothing produced the SAME value, the
    equality test passed, and the gate — whose own field docstring said "the
    gate refuses that too" — refused nothing. Measured on `e35de13`: the
    refusal came from `Stamp.__post_init__` ("stamp field
    'query_content_hash' is empty") one layer later. Loud, but not this gate,
    and not what the docstring claimed.

    Two absences are not a match. Both parametrisations must raise
    `MispairedEscalationError` FROM THE GATE, and the second (a real recorded
    hash against an unhashable query) is the control: it already raised
    before, so a repair that somehow only fixed the first would still be
    visible as an asymmetry here.
    """
    import dataclasses

    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_free)
    assert esc.query_sha256, "the fixture's escalation recorded no hash"
    blinded = _Unhashable(closed)
    from stelling.solvers import _query_sha256

    assert _query_sha256(blinded) == "", (
        "the blinded query still hashes, so this test never reaches the "
        "empty-hash leg it is about"
    )

    with pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(blinded, prop,
                            dataclasses.replace(esc, query_sha256=recorded),
                            **VERSIONS)
    assert "unhashable" in str(exc.value), str(exc.value)
    assert "StampError" not in type(exc.value).__name__


def test_the_pairing_gate_costs_no_additional_hash():
    """THE COST, AS A MECHANISM RATHER THAN AS A TIMING. `make_solver_verdict`
    already took `closed.content_hash()` for the stamp; the gate compares that
    same value, so binding the escalation to the query adds ZERO hashes to
    assembly. Counted, not timed — a timing would be a flaky way to assert a
    structural property, and the structural property is the claim.

    `escalate` pays one hash, once per escalation, which is measured in
    `SOUNDNESS.md` against the solver work it sits beside.
    """
    from stelling import ir
    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)

    calls = []
    real = ir.ClosedJaxpr.content_hash

    def counted(self):
        calls.append(id(self))
        return real(self)

    ir.ClosedJaxpr.content_hash = counted
    try:
        v = make_solver_verdict(closed, prop, esc, **VERSIONS)
    finally:
        ir.ClosedJaxpr.content_hash = real

    assert v.stamp.query_content_hash == real(closed)
    assert len(calls) == 1, (
        f"assembly took {len(calls)} content_hash() call(s); the gate and the "
        f"stamp must share one. Taking it twice is a real cost on every "
        f"verdict and the two could drift"
    )


def test_the_public_path_cannot_mispair_and_the_gate_never_fires_on_it():
    """THE REACHABILITY CLAIM THIS ENTRY RESTS ON, DRIVEN RATHER THAN
    ARGUED. `SOUNDNESS.md` says no recorded verdict is retroactively invalid
    because `stelling.preconditions.check` derives all three artifacts from
    one trace and cannot mispair. That is a claim about code, so it is
    measured here: the public path runs, and the escalation it built carries
    this query's own hash.

    Both directions, because the first alone would also hold if the gate had
    silently stopped being reachable: `check` must produce a verdict at all,
    and the gate must be SATISFIED rather than absent — an escalation whose
    `query_sha256` is empty would pass no gate and produce the same verdict.
    """
    import dataclasses

    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate

    v = check(_scatter_ON_the_decided_slice, vacuity_mode="inputs-only",
              solver_timeout_ms=20000)
    assert v.status in ("UNKNOWN", "VERIFIED", "REFUTED"), v.status
    assert v.stamp.query_content_hash == trace(
        _scatter_ON_the_decided_slice).content_hash(), (
        "`check` stamped a query hash that re-tracing the same harness does "
        "not reproduce, so the pairing gate's key is not stable across the "
        "public path and honest verdicts would start being refused"
    )

    # the gate is satisfied, not absent
    closed = trace(_scatter_ON_the_decided_slice)
    prop = propagate(closed)
    esc = escalate(closed, prop, SolverConfig(timeout_ms=20000))
    assert esc.query_sha256 == closed.content_hash() != ""
    assert dataclasses.replace(esc, query_sha256="").query_sha256 == "", (
        "the field cannot be cleared, so the gate below is not testing what "
        "it thinks"
    )


def test_the_bar_scope_itself_widens_on_the_colliding_pair():
    """The same defect one layer down, at `_bar_scope` rather than at the
    assembled verdict — so that deleting the `_evidence_is_about` call is not
    a one-test edit.

    Not a duplicate of the parametrised test above: that one can only see the
    bar through a VERIFIED that may be withheld for some other reason, and
    this one reads the scope directly. Both must move together, because both
    are downstream of the same call.
    """
    on_closed, _on_prop, on_esc = _stamped(_scatter_ON_the_decided_slice)
    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    decided = {r.index: r.invocations
               for r in on_esc.records if r.outcome == "discharged"}
    assert set(decided) == {0}, decided

    honest, honest_why = V._bar_scope(on_closed, decided)
    assert honest == ("scatter",) and "assert #0" in honest_why, (
        f"the honestly-paired scope is {honest!r}; the fixture is wrong"
    )
    barred, why = V._bar_scope(el_closed, decided)
    assert barred == ("scatter",), (
        f"_bar_scope narrowed to {barred!r} on a query the escalation is not "
        f"about, whose obligation #0 emits the SAME script as the one the "
        f"solver answered. Narrowing is keyed on the emitted text again"
    )
    assert "fell back to the whole query" in why, why


def test_a_wrong_closed_does_NOT_reliably_widen_the_bar():
    """THE COUNTEREXAMPLE TO A SENTENCE THAT STOOD BOLDED IN
    `make_solver_verdict`'s DOCSTRING: *"A wrong ``closed`` widens the bar,
    and the reason is measured rather than assumed."* It does not, and this
    is the measurement that says so.

    On the identical-decided-slice pairing the bar goes SILENT on the wrong
    query: `_bar_scope(wrong, decided)` is `((), '')` — narrowed to nothing,
    empty reason — while `_barred_primitives(wrong)` is `('scatter',)`. The
    query carries the barred primitive and the bar withholds nothing.

    The claim was not merely unproven; it was the third bolded generalisation
    in that paragraph's history, each naming a real mechanism ("the re-slice
    would decline", "the evidence widens it") and then quantifying it over
    every wrong `closed`. The mechanism is real and NARROW: a wrong `closed`
    widens the bar exactly when the evidence check can see the mispairing,
    which the control below shows it sometimes can. Pairing is now the query
    pairing gate's job, not the bar's.

    This test exists so the sentence cannot come back. If the bar ever does
    widen here, the docstring changed something real and should say what.
    """
    on_closed, _on_prop, on_esc = _stamped(
        _scatter_ELSEWHERE_identical_decided_slice_TRUE)
    wrong_closed = trace(_scatter_ELSEWHERE_identical_decided_slice)
    decided = {r.index: r.invocations
               for r in on_esc.records if r.outcome == "discharged"}
    assert set(decided) == {0}, decided

    assert V._barred_primitives(wrong_closed) == ("scatter",), (
        "the wrong query carries no barred primitive, so a silent bar there "
        "is correct and this test measures nothing"
    )
    assert V._bar_scope(wrong_closed, decided) == ((), ""), (
        f"{V._bar_scope(wrong_closed, decided)!r}: the bar now widens on a "
        f"wrong `closed` for this pairing. That is STRONGER than what "
        f"`make_solver_verdict`'s docstring claims — say what changed"
    )

    # THE CONTROL, so this is not read as "a wrong closed never widens": when
    # the escalation's decided slice DOES carry the barred primitive, the
    # evidence check fails on the wrong query and the fallback fires.
    on2_closed, _p2, on2_esc = _stamped(_scatter_ON_the_decided_slice)
    d2 = {r.index: r.invocations
          for r in on2_esc.records if r.outcome == "discharged"}
    assert V._bar_scope(wrong_closed, d2)[0] == ("scatter",), (
        "the widening direction does not happen on ANY pairing, so the "
        "docstring's mechanism is not merely over-generalised but absent"
    )


def test_the_correct_pairing_still_narrows_and_the_hash_is_why():
    """THE OTHER DIRECTION of the test above, and the case that distinguishes
    the repair from a silent revert to the whole-query bar.

    Withholding everything would pass the mispairing test trivially. What must
    also hold is that an HONEST assembly still narrows — and that the things
    permitting it are the measured ones: BOTH the script the solver was
    actually sent and the fingerprint of the slice it came out of, re-derived
    from this query, reproduce what the invocation recorded.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import emit, slice_fingerprint

    closed, _prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    (record,) = [r for r in esc.records if r.outcome == "discharged"]
    sl = slice_obligation(closed, record.index, interval_env(closed))
    assert not isinstance(sl, DeclinedObligation)
    assert record.invocations, "no invocation to check the pairing against"
    matched, fingerprinted = [], []
    for stamp in record.invocations:
        opts = dict(stamp.options or ())
        timeout = opts.get(":timeout") or opts.get(":tlimit")
        assert timeout, (
            f"the {stamp.name} stamp records no timeout under a key "
            f"`_evidence_is_about` reads, so the honest path cannot bind and "
            f"the bar would widen on every correctly-paired verdict"
        )
        matched.append(emit(sl, stamp.name, int(timeout)).sha256
                       == opts.get("smt2_sha256"))
        fingerprinted.append(slice_fingerprint(sl) == opts.get("slice_sha256"))
    assert all(matched), (
        f"re-emitting this query's own slice does not reproduce the hash the "
        f"invocation recorded ({matched}) — emission is no longer a function "
        f"of (slice, solver, timeout), so the pairing check can never say yes "
        f"and the bar has silently become the whole-query one again"
    )
    assert all(fingerprinted), (
        f"the stamp does not carry this slice's fingerprint ({fingerprinted}) "
        f"— the second conjunct of `_evidence_is_about` can never say yes, so "
        f"every correctly-paired verdict on a scatter-bearing query now gets "
        f"the whole-query bar"
    )
    assert V._evidence_is_about(sl, record.invocations)
    assert not V._evidence_is_about(sl, ()), (
        "an obligation with NO recorded invocation binds, so the check is "
        "vacuous"
    )


def test_a_mispaired_PROPAGATION_cannot_empty_the_scope_either():
    """The other mispairing, and the one the recording design was built
    against: a propagation whose obligations are already `discharged` slices
    to NOTHING (the slicer only slices `unknown` ones), so a scope recomputed
    from `(closed, propagation)` would come back empty while every existing
    gate passed.

    The derivation does not take the propagation. It re-slices by INDEX out of
    `closed`, so there is no propagation to mispair — asserted here rather
    than argued, because it was the stated reason for recording.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    settled = dataclasses.replace(prop, obligations=tuple(
        dataclasses.replace(o, status="discharged") for o in prop.obligations
    ))
    v = make_solver_verdict(closed, settled, esc, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: a propagation with nothing left to slice emptied the "
        f"bar's scope — the scope is being recomputed from the propagation "
        f"rather than from the query"
    )


def test_the_fallback_is_the_WHOLE_QUERY_SET_and_never_silence():
    """THE FAIL-CLOSED FALLBACK ITSELF, which nothing measured.

    Every "derive, don't record" argument in this module ends at the same
    sentence: when the derivation cannot be completed the bar drops to the
    whole-query set, which is WIDER, rather than to an empty scope, which is
    silence. Measured before this test existed: replacing both `return
    fallback` statements in `_bar_scope` with `return (), ""` left the full
    suite at 2008 passed, 2 skipped — the fallback the design rests on was
    load-bearing for nothing.

    Both directions are asserted, because the two are different code paths and
    a test that only calls the function cannot see the verdict move:

    * the derivation itself returns the whole-query set for an index that does
      not slice;
    * and a verdict assembled with such a record goes UNKNOWN rather than
      VERIFIED — a stray `discharged` record is the cheapest way for an
      honest-but-mispaired assembly to reach this path.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing; the fallback is unreachable"

    barred, why = V._bar_scope(closed, {99: ()})
    assert barred == whole, (
        f"an index that does not slice returned {barred!r} where the "
        f"whole-query set is {whole!r} — the derivation fell to SILENCE, not "
        f"to the wider bar, and every 'fails closed' sentence about this "
        f"function is false"
    )
    assert "fell back to the whole query" in why

    # ... and it moves a verdict, which the call above cannot show
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "VERIFIED"
    ), "the honest assembly does not VERIFY, so nothing here can be withheld"
    (real,) = [r for r in esc.records if r.outcome == "discharged"]
    strayed = dataclasses.replace(esc, records=esc.records + (
        dataclasses.replace(real, index=99),
    ))
    v = make_solver_verdict(closed, prop, strayed, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: a discharged record naming an obligation this query "
        f"does not have left the bar's scope EMPTY instead of falling back "
        f"to the whole query"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_what_a_stray_index_ACTUALLY_DOES_all_four_of_them():
    """The stray-index enumeration, MEASURED rather than asserted.

    `_bar_scope`'s docstring and `make_solver_verdict`'s block comment both
    enumerate what a stray index does. The enumeration has been wrong twice —
    first "a stray index does not slice" (one behaviour, and the wrong one),
    then a list of three presented as the whole space. There is a fourth: an
    index past the START of the list raises `IndexError` out of
    `slice_obligation` rather than declining, and reaches the whole-query set
    through `_bar_scope`'s outer `except` instead of through a `fallback`
    call. No soundness difference — it is named because "three behaviours"
    was being read as closed and was not, and this test is what makes the
    enumeration a measurement.

    NOT PRESENTED AS EXHAUSTIVE EITHER. What is asserted is that each listed
    behaviour is the one claimed, and that every one of them ends at the
    whole-query set.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env

    closed = trace(_scatter_ON_the_decided_slice)
    env = interval_env(closed)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing barred; this measures nothing"

    # 1. an index intervals decided slices perfectly well
    sl = slice_obligation(closed, 1, env)
    assert not isinstance(sl, DeclinedObligation)
    assert V._barred_in_eqns(sl.eqns) == ("scatter",)
    # 2. a negative index within range is Python indexing
    assert not isinstance(slice_obligation(closed, -1, env), DeclinedObligation)
    # 3. an index matching no assert equation DECLINES
    assert isinstance(slice_obligation(closed, 99, env), DeclinedObligation)
    # 4. and one past the start RAISES rather than declining
    with pytest.raises(IndexError):
        slice_obligation(closed, -3, env)

    for index in (1, -1, 99, -3):
        barred, why = V._bar_scope(closed, {index: ()})
        assert barred == whole, (
            f"index {index} narrowed the bar to {barred!r} instead of "
            f"falling back to the whole-query set {whole!r}"
        )
        assert "fell back to the whole query" in why


def test_the_fallback_also_holds_when_the_derivation_RAISES(monkeypatch):
    """The other `return fallback`, and it is a different statement.

    `_bar_scope` wraps the whole re-derivation in `except Exception` with the
    comment "a bar must never break a verdict, and it must never go quiet
    either". The first half is pinned by construction — an exception cannot
    escape. The second half was pinned by nothing: silencing THIS branch alone
    also left the suite fully green.

    Driven by making the slicer raise, which is the only way to reach a branch
    whose whole purpose is a failure no fixture produces.
    """
    import stelling.obligation as _ob

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(_scatter_OFF_the_decided_slice)
    whole = V._barred_primitives(closed)
    assert whole, "the fixture carries nothing; the fallback is unreachable"
    assert _verdict_status(closed, prop, esc) == "VERIFIED", (
        "the honest assembly does not VERIFY, so nothing here can be withheld"
    )

    def boom(*_a, **_k):
        raise RuntimeError("slicer exploded")

    monkeypatch.setattr(_ob, "slice_obligation", boom)
    barred, why = V._bar_scope(closed, {0: ()})
    assert barred == whole, (
        f"a raising re-derivation returned {barred!r} where the whole-query "
        f"set is {whole!r} — the except branch goes QUIET, so a bar that "
        f"cannot compute its scope silently stops being a bar"
    )
    assert "could not be re-derived" in why
    assert _verdict_status(closed, prop, esc) == "UNKNOWN", (
        "a verdict assembled while the slicer raises came back VERIFIED — "
        "the except branch is not withholding"
    )


# The values a record field of each declared type can be moved to. Keyed on
# the annotation text rather than on the field name, so a field added tomorrow
# is probed by this test without anybody remembering to list it — and an
# annotation not in this map is a LOUD failure below, never a silent skip.
#
# WHAT TWO VALUES BUY DEPENDS ENTIRELY ON THE TYPE, and an earlier version of
# this comment said they were "chosen so that a field whose NON-default value
# would clear the bar is probed with it" as though that were true of all of
# them. For `bool` it is: the two probes EXHAUST the type, so a conjunct on a
# new bool field is caught whichever way it points — that is how the
# `audited_clean` mutant this test was rewritten for is caught. For `str`,
# `int` and the tuples they are a SAMPLE of an unbounded space, and a sample
# cannot pin a channel. Measured on `eb1ff86`: `audit_token: str = ""` plus
# `and r.audit_token != "clean"` in the bar's domain is UNKNOWN at both probe
# values and VERIFIED at `'clean'`, full suite green.
#
# So the channel is pinned by `test_the_bars_domain_cannot_read_a_new_field`,
# which hands the domain a record that HAS no other field; this sweep stays
# for what it does do — it exercises the whole assembly at each probe value,
# which the construction test cannot, and it is the loud failure for a field
# of a type nobody has thought about.
def _field_probes():
    from stelling.verdict import solver_absent

    return {
        "int": (0, 7),
        "bool": (False, True),
        "str": ("", "a value no honest record carries"),
        "tuple[str, ...]": ((), ("a value no honest record carries",)),
        "tuple[SolverStamp, ...]": ((), (solver_absent("probe"),)),
        # a NON-NONE witness is what this probes; the bar reads no field of
        # one, and if that ever changes the sentinel raises here rather than
        # passing quietly
        "Witness | None": (None, object()),
    }


def test_no_record_field_can_narrow_the_bars_domain():
    """A RECORD MUST NOT BE ABLE TO CERTIFY ITS OWN CLEANLINESS — pinned as a
    CHANNEL, not as a list of today's fields.

    The predecessor's `barred_on_slice=()` was a positive claim ("nothing
    barred on my slice") that nothing validated: measured on this fixture, a
    record carrying it earned VERIFIED where the genuine scope was
    `('scatter',)`. The repair deletes the field rather than validating it, so
    what is pinned is the ABSENCE of any such channel.

    THE VERSION OF THIS TEST THAT PINNED A MEMBER LIST COULD NOT FAIL FOR ITS
    OWN DEFECT, and that is why it was rewritten. It reset every
    non-load-bearing field **to its declared default** and accounted for the
    field names — so a new field whose default is inert and whose OTHER value
    clears the bar was invisible to both halves. Measured: adding
    `audited_clean: bool = False` to `ObligationEscalation` and `and not
    r.audited_clean` to the bar's domain, then handing in records with
    `audited_clean=True`, produced VERIFIED where the honest assembly gives
    UNKNOWN — with the full suite green, the field-accounting assertion
    included, because `dataclasses.replace(r, audited_clean=False)` is exactly
    what the test did.

    So each field is now moved AWAY from what it holds, with the values taken
    from its declared TYPE. A field of a type `_field_probes` does not know
    fails this test loudly, which is the only way an enumeration can stay
    honest about the thing it has not thought of.

    THAT REWRITE MOVED THE ENUMERATION FROM NAMES TO TYPES; IT DID NOT PIN THE
    CHANNEL, and the difference is measurable. Two probe values exhaust `bool`
    and merely sample `str`: `audit_token: str = ""` on the record plus
    `and r.audit_token != "clean"` in the bar's domain is UNKNOWN at both
    probes and VERIFIED at `'clean'`, with the full suite green. This test
    still earns its place — it drives the WHOLE assembly at each value, which
    a construction test cannot — but the channel itself is pinned next door,
    by `test_the_bars_domain_cannot_read_a_new_field`, which removes the
    fields rather than guessing their values.

    LOAD-BEARING IS EXACTLY `index` AND `outcome`. `invocations` is read by
    `_evidence_is_about`, but only to PERMIT narrowing — every value of it
    that does not reproduce BOTH the recorded script hash and the recorded
    slice fingerprint widens the bar — so it is probed here like any other
    field and the bar must not move. (This sentence said "a matching script
    hash" and was stale by one hash from the commit that added the second
    conjunct.) An earlier version
    exempted it, which was false in the direction that matters; see
    `test_stripping_invocations_cannot_clear_the_bar` for that drift.
    """
    import dataclasses

    from stelling.solvers import ObligationEscalation, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "UNKNOWN"
    ), "the genuine assembly does not bar; the fixture is wrong"

    assert "barred_on_slice" not in {
        f.name for f in dataclasses.fields(ObligationEscalation)
    }, (
        "a record field named `barred_on_slice` is back. It was deleted "
        "because a recorded scope is a claim the bar cannot check; if it "
        "returns, the bar must not read it"
    )

    probes = _field_probes()
    load_bearing = {"index", "outcome"}
    fields = [f for f in dataclasses.fields(ObligationEscalation)
              if f.name not in load_bearing]
    assert fields, "every field is load-bearing; this test measures nothing"
    unknown = [f.name for f in fields if str(f.type) not in probes]
    assert not unknown, (
        f"record field(s) {unknown} have a type this test does not know how "
        f"to move ({[str(f.type) for f in fields if f.name in unknown]}). It "
        f"CANNOT be skipped: the defect this test exists for is a field whose "
        f"non-default value clears the bar, and an unprobed field is exactly "
        f"that field. Add the type to `_field_probes` with a value no honest "
        f"record carries"
    )

    def stamp(**overrides):
        forged = dataclasses.replace(esc, records=tuple(
            dataclasses.replace(r, **overrides) for r in esc.records
        ))
        return make_solver_verdict(closed, prop, forged, **VERSIONS)

    # one field at a time, at every probe value ...
    for f in fields:
        for value in probes[str(f.type)]:
            v = stamp(**{f.name: value})
            assert [o.status for o in v.obligations] == (
                ["discharged"] * len(v.obligations)
            ), (
                f"setting {f.name}={value!r} un-discharged an obligation, so "
                f"there is no VERIFIED left for the bar to withhold and this "
                f"probe passes for the wrong reason"
            )
            assert v.status == "UNKNOWN", (
                f"{v.status}: a record carrying {f.name}={value!r} cleared "
                f"the bar. The bar's domain is `outcome == OB_DISCHARGED` and "
                f"nothing else may narrow it — a field that does is the "
                f"deleted `barred_on_slice` under another name"
            )
    # ... and all of them at once, in both directions
    for which in (0, 1):
        v = stamp(**{f.name: probes[str(f.type)][which] for f in fields})
        assert v.status == "UNKNOWN", (
            f"{v.status}: moving every non-load-bearing field at once "
            f"cleared the bar"
        )


class _OnlyThreeFields:
    """A record with `index`, `outcome`, `invocations` AND NOTHING ELSE.

    `__slots__` is the whole mechanism: reading any other attribute raises
    `AttributeError`, so a conjunct on a field of ANY type — `str`, `int`, a
    tuple, something not thought of — cannot silently evaluate. This is what
    an enumeration of probe VALUES cannot do, because `str` has more values
    than a test can list."""

    __slots__ = ("index", "outcome", "invocations")

    def __init__(self, index, outcome, invocations=()):
        self.index = index
        self.outcome = outcome
        self.invocations = invocations


class _JustRecords:
    __slots__ = ("records",)

    def __init__(self, records):
        self.records = records


def test_the_bars_domain_cannot_read_a_new_field():
    """PIN THE CHANNEL BY REMOVING IT, not by guessing what would flow down
    it. The bar's domain is `outcome == OB_DISCHARGED`, keyed by `index`, and
    carries `invocations` across for the widening check — three fields, and
    the record it is handed here HAS no fourth.

    THE DEFECT THIS IS FOR IS MEASURED, not imagined. On `eb1ff86`, adding
    `audit_token: str = ""` to `ObligationEscalation` and
    `and r.audit_token != "clean"` to the domain filter produced VERIFIED for
    a record carrying `audit_token='clean'` on the bar's own fixture, with the
    full suite green — including
    `test_no_record_field_can_narrow_the_bars_domain`, whose two `str` probes
    (`''` and `'a value no honest record carries'`) both left the bar firing.
    Two values exhaust `bool`; they sample `str`.

    Under this test the same conjunct raises `AttributeError` inside
    `_bar_domain`, which widens to the sentinel rather than returning a
    domain, and the equality below fails.
    """
    from stelling.solvers import OB_DISCHARGED, _bar_domain

    stamps = (V.solver_absent("probe"),)
    records = (
        _OnlyThreeFields(0, OB_DISCHARGED, stamps),
        _OnlyThreeFields(1, "unknown", ()),
        _OnlyThreeFields(2, OB_DISCHARGED, ()),
    )
    domain = _bar_domain(_JustRecords(records))
    assert domain == {0: stamps, 2: ()}, (
        f"the bar's domain came back {domain!r} from a record carrying "
        f"nothing but `index`, `outcome` and `invocations`. Either the "
        f"domain now reads a fourth field — which is the deleted "
        f"`barred_on_slice` under another name, whatever its type — or its "
        f"membership test is no longer `outcome == OB_DISCHARGED`"
    )
    # and it is not vacuous: the outcome really is what selects
    assert _bar_domain(_JustRecords((_OnlyThreeFields(0, "unknown", stamps),))) == {}


# -- the channel, pinned by OBSERVATION rather than by removal ---------------
#
# `test_the_bars_domain_cannot_read_a_new_field` removes the fields, which is
# the right idea and reaches exactly one access form. Measured on this branch,
# against a `__slots__` record carrying only `index`, `outcome`,
# `invocations`: a plain attribute read is caught, and so is a `@property`;
# `getattr(r, "audit_token", "")`, a `hasattr` guard and `r.__dict__.get(...)`
# are NOT — they evaluate to the default and pass. And it pins the invariant at
# ONE PRODUCER, `_bar_domain`, while the same conjunct written at the CALL SITE
# or one function over does the same job. Six channels were measured on this
# branch, each flipping the bar's UNKNOWN to VERIFIED with the full suite
# green:
#
#   1 `_evidence_is_about` reads `options.get("audited_clean")`  (no new field)
#   2 `make_solver_verdict`, one line after `_bar_domain`: `decided = {}`
#   3 `_bar_domain`, `getattr(r, "audit_token", "") == "clean"`
#   4 `make_solver_verdict`, AFTER the bar is consulted: `barred = ()`
#   5 `_bar_domain`, `type(r.index) is not int`, on an `int` SUBCLASS
#   6 `_bar_scope` reads the token out of the existing `SolverStamp.options`
#   7 `_bar_scope` skips an obligation whose stamp records a chosen `:timeout`
#   8 the SAME conjunct written inside the permitted reader, on a value the
#     sweep does not sample — three spellings, measured on `3e107cf`
#
# CHANNEL 7 NEEDED NO FORGED RECORD AT ALL, which is what separates it from
# 1-6: it is driven by a PUBLIC KEYWORD ARGUMENT. `solver_timeout_ms` is
# carried verbatim into the stamp as `:timeout`, `:timeout` is a member of
# `_EVIDENCE_OPTION_KEYS` (it has to be — it is part of the emitted text and
# therefore part of the hash), and NOTHING constrained what the decision did
# with a whitelisted key's VALUE. Measured on `e35de13`:
#
#   check(build, vacuity_mode="inputs-only", solver_timeout_ms=20000)
#       -> UNKNOWN, "VERIFIED withheld ... assert #0"
#   check(build, vacuity_mode="inputs-only", solver_timeout_ms=31337)
#       -> VERIFIED, no note
#
# on `_scatter_ON_the_decided_slice`, with the full suite byte-identical to
# unmutated HEAD in both columns (2044 / 2 and 2040 / 6).
#
# Why the six mechanisms already here all missed it, each measured:
#   * the read LEDGER below attributed the read to `_evidence_options`, the
#     PROJECTION HELPER, rather than to the function that ASKED for it — so
#     `_bar_scope` calling the permitted projection logged the already-
#     permitted pair, and the ledger saw nothing new in EITHER direction;
#   * `_EVIDENCE_OPTION_KEYS` constrains WHICH KEYS reach the decision and
#     says nothing about what the decision does with their values;
#   * `test_the_narrowing_decision_reads_options_in_one_place` scanned for
#     `.options` / `"options"` / `'options'`, and `_evidence_options(s)`
#     matches none of the three;
#   * the type-identity, field-removal and field-probe tests are untouched:
#     no new field, no new key, no `type()` discrimination.
#
# It is closed by TWO anti-correlated mechanisms, and each has its own
# mutation. STRUCTURALLY, the ledger now attributes an `options` read to the
# function that asked for it (`_PROJECTIONS` below), so a projection call from
# anywhere but the one permitted reader is a new pair — and the source scan
# forbids the call outright in the bar's other functions. BEHAVIOURALLY,
# `test_the_bars_narrowing_does_not_move_with_a_CALLER_SETTABLE_option_value`
# pins that the bar's answer does not move under any value a caller can
# choose, which is the half that reaches a conjunct written INSIDE the one
# permitted reader, where no site-based mechanism can see it.
#
# CHANNEL 8 IS WHAT THAT SPLIT ACTUALLY LEFT OPEN, and it is the reason the
# permitted reader is no longer the DECISION. On `3e107cf` the one function
# allowed to see option values was `_evidence_is_about` — the function that
# decides whether the bar narrows — so both site mechanisms were satisfied by
# construction and the whole defence rested on the sweep. The sweep is a
# SAMPLE, and it can only reach the two of the four whitelisted keys that a
# caller can set. Three line-neutral corruptions, each measured at `3e107cf`
# with the full suite BYTE-IDENTICAL to unmutated in both columns
# (2055 passed / 2 skipped and 2051 / 6):
#
#   8a `_evidence_is_about` stashes its projected budget in a module global;
#      `_bar_scope` zeroes `found` when that global is "30000". No new
#      (function, attribute) pair — the read stays in the one permitted
#      reader — and `_bar_scope`'s body carries no `.options`, no "options"
#      and no `_evidence_options`, so neither scan matches. Driven through the
#      public API on `_scatter_ON_the_decided_slice`:
#
#          solver_timeout_ms=20000 -> UNKNOWN, "VERIFIED withheld ... #0"
#          solver_timeout_ms=29999 -> UNKNOWN, withheld
#          solver_timeout_ms=30000 -> VERIFIED, NO withheld note
#          solver_timeout_ms=30001 -> UNKNOWN, withheld
#          solver_timeout_ms=31337 -> UNKNOWN, withheld
#
#      THE ISOLATING CONTROL is the identical mutation keyed on "31337",
#      which is 2 RED in this file. The only difference is the constant. So
#      the sentence `_CALLER_BUDGETS` used to carry — "an equality on any
#      round or memorable number is hit" — was FALSE, and is corrected there;
#   8b a conjunct on the VALUE of `smt2_sha256` inside `_evidence_is_about`,
#      and
#   8c the same on `slice_sha256`. Both keys are whitelisted, both values are
#      record-supplied, and the sweep varies NEITHER — it varies 2 of the 4
#      keys. Driven with the pairing gate satisfied by hand
#      (`_past_the_pairing_gate`, as every mispairing test here does) and the
#      hash forged to "0"*64: VERIFIED with no withheld note, where the
#      unmutated build returns UNKNOWN with the whole-query fallback. That is
#      channel 1's threat model — a record certifying its own cleanliness —
#      reached through a whitelisted key's VALUE rather than through a new key.
#
# THE REPAIR IS NOT A WIDER SWEEP. `_CALLER_BUDGETS` says why in its own
# comment, and this project has watched "raise the bound" fail on six other
# axes. The narrowing DECISION now consumes a boolean: `_evidence_is_about`
# binds no recorded value and may not call the projection (both scans cover it
# now), and the value zone is one four-line function, `_evidence_reproduces`,
# whose two halves cannot aim — `_evidence_options` never sees the query,
# `_reproduced_evidence` never sees the record. What is left is covered by
# `test_the_evidence_path_cannot_name_a_VALUE`, which is TOTAL over the source
# rather than a sample over values.
#
# `test_the_bar_is_consulted_with_exactly_that_domain` does not reach 2, 4 or
# 6: all three satisfy `len(seen) == 1 and seen[0] == _bar_domain(case)`,
# because a conjunct keyed on an UNPROBED value changes nothing at the values
# it probes. That is the same gap the branch identified inside `_bar_domain`
# and left standing at the call site.
#
# So the pin here is a READ LEDGER. Every record and every invocation stamp
# reaching the assembly is wrapped in an observer that records
# (reading function, attribute name) for EVERY attribute access — plain,
# property, `getattr` with a default, `hasattr`, `__dict__`, all of them go
# through `__getattribute__` — and the whole of `make_solver_verdict` is then
# driven. A read that is not in the allow-list below is a new channel whatever
# it is called, whatever its type, and wherever it is written. Channels 2, 3, 4
# and 6 are all new (function, attribute) pairs. Channel 1 is a new KEY rather
# than a new attribute and is closed by `verdict._EVIDENCE_OPTION_KEYS`;
# channel 5 reads nothing new at all and is closed by the type-invariance test
# after that.
_COMPREHENSIONS = ("<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>")

# PURE PROJECTION HELPERS, WHICH ARE NOT THE READER — CHANNEL 7'S MECHANISM.
# `_evidence_options` exists to project a stamp's option set down to
# `_EVIDENCE_OPTION_KEYS`; it decides nothing. Attributing the read to IT
# rather than to the function that ASKED for it is what let channel 7 through:
# `_bar_scope` calling the permitted projection logged the permitted pair, so
# the ledger's "nothing new" and "nothing dead" were both satisfied by a
# reader in a function that had never read a stamp before. The frame walk
# skips a projection for the same reason it skips a comprehension — neither is
# the site that decided to look. A projection added under a different name is
# not skipped, and therefore logs itself as a new pair, which is the direction
# this must fail in.
_PROJECTIONS = ("_evidence_options",)
_TRANSPARENT_FRAMES = _COMPREHENSIONS + _PROJECTIONS


def _reading_function() -> str:
    """The function that ASKED for the attribute, seen through any
    comprehension frames it went via (a genexpr's own `co_name` would hide
    which function wrote it) and through any pure projection helper it went
    via (see `_PROJECTIONS`)."""
    import sys

    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_name in _TRANSPARENT_FRAMES:
        frame = frame.f_back
    return frame.f_code.co_name if frame is not None else "<unknown>"


class _Watched:
    """A transparent proxy that logs `(kind, reading function, attribute)` for
    every attribute access, then delegates."""

    def __init__(self, inner, log, kind):
        object.__setattr__(self, "_watch", (inner, log, kind))

    def __getattribute__(self, name):
        inner, log, kind = object.__getattribute__(self, "_watch")
        if name == "_watch":
            return (inner, log, kind)
        log.add((kind, _reading_function(), name))
        return getattr(inner, name)


# EXACTLY the reads the assembly makes off a record or an invocation stamp.
# Asserted in both directions: nothing outside it (a new channel), and nothing
# in it that never happens (a padded list would let a real read hide).
_ALLOWED_READS = frozenset({
    # the bar's domain: the discharging predicate, its key, and the stamps
    # carried across for the WIDENING decision
    ("record", "_bar_domain", "outcome"),
    ("record", "_bar_domain", "index"),
    ("record", "_bar_domain", "invocations"),
    # verdict assembly proper — the obligation loop, the notes, the
    # witnesses, the redundancy surface
    ("record", "make_solver_verdict", "index"),
    ("record", "make_solver_verdict", "outcome"),
    ("record", "make_solver_verdict", "detail"),
    ("record", "make_solver_verdict", "notes"),
    ("record", "make_solver_verdict", "witness"),
    ("record", "make_solver_verdict", "answered_by"),
    # the narrowing DECISION, which reads only what it needs to skip a stamp
    # it cannot use — and no option value at all
    ("stamp", "_evidence_is_about", "invoked"),
    ("stamp", "_evidence_is_about", "name"),
    # ... and the one function that holds a recorded VALUE: the flavour it
    # re-emits at, plus `options` projected to `_EVIDENCE_OPTION_KEYS` on the
    # way. Attributed to the function that ASKED, not to the projection: see
    # `_PROJECTIONS`, and channels 7 and 8 in the block comment above.
    ("stamp", "_evidence_reproduces", "name"),
    ("stamp", "_evidence_reproduces", "options"),
})


def _watched_escalation(esc, log):
    import dataclasses

    return dataclasses.replace(esc, records=tuple(
        _Watched(
            dataclasses.replace(r, invocations=tuple(
                _Watched(s, log, "stamp") for s in r.invocations)),
            log, "record",
        )
        for r in esc.records
    ))


def test_nothing_in_the_assembly_reads_a_field_it_is_not_allowed_to():
    """THE CHANNEL, PINNED AT THE SURFACE AND NOT ONLY AT ITS PRODUCER.

    See the block comment above for the six measured channels and for why
    removing fields from a record reaches only one of the access forms. This
    drives the WHOLE of `make_solver_verdict` — the call site included, and
    `_bar_scope` and `_evidence_is_about` with it — and fails on any attribute
    read that is not in `_ALLOWED_READS`.

    Both arrangements are driven, because the bar reaches different code in
    each: the honest one bars through the decided slice's own barred set, the
    mispaired one through the evidence check's fallback, and a conjunct
    written for the second is invisible to the first.
    """
    from stelling.solvers import make_solver_verdict

    log: set = set()

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    v = make_solver_verdict(closed, prop, _watched_escalation(esc, log),
                            **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: the watched assembly does not reach the bar, so this "
        f"test never drives the code the channels live in"
    )

    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    mispaired = _watched_escalation(
        _past_the_pairing_gate(esc, el_closed), log)
    v2 = make_solver_verdict(el_closed, prop, mispaired, **VERSIONS)
    assert v2.status == "UNKNOWN", v2.status

    extra = log - _ALLOWED_READS
    assert not extra, (
        f"the assembly read {sorted(extra)} off a record or an invocation "
        f"stamp. Every entry is a new channel by which a record could certify "
        f"its own cleanliness — the deleted `barred_on_slice` under another "
        f"name, at another site, or through another access form. If the read "
        f"is legitimate, adding it to `_ALLOWED_READS` is the whole cost, and "
        f"it should be argued in the commit rather than done quietly"
    )
    dead = _ALLOWED_READS - log
    assert not dead, (
        f"`_ALLOWED_READS` permits {sorted(dead)}, which this test never "
        f"observed. A permitted read that never happens is padding, and "
        f"padding is where a real read hides"
    )


def test_the_narrowing_decision_may_read_only_these_option_keys():
    """CHANNEL 1: `SolverStamp.options` is a record-carried `dict[str, str]`
    that the narrowing decision reads, and nothing constrained WHICH keys.

    A conjunct on a key no honest stamp carries needs no new field anywhere —
    it is invisible to `test_the_bars_domain_cannot_read_a_new_field`, which
    removes fields, and to `test_no_record_field_can_narrow_the_bars_domain`,
    which probes their values. Measured on this branch:
    `options.get("audited_clean") == "yes"` in `_evidence_is_about` turns the
    mispaired assembly's UNKNOWN into VERIFIED with the full suite green.

    Closed by PROJECTION. `_evidence_options` returns a mapping whose keys are
    a subset of `_EVIDENCE_OPTION_KEYS`, so a key outside that set cannot
    reach the decision at all. Both halves are pinned: the set is asserted
    exactly (widening it is the cost of reopening the channel), and the
    projection is asserted to drop arbitrary keys rather than the ones this
    test happened to think of.
    """
    assert V._EVIDENCE_OPTION_KEYS == frozenset({
        "smt2_sha256", "slice_sha256", ":timeout", ":tlimit",
    }), (
        f"the narrowing decision may now read {sorted(V._EVIDENCE_OPTION_KEYS)} "
        f"out of a stamp's options. Each added key is a value the record "
        f"supplies and the bar believes — state what it is for"
    )

    honest = (("smt2_sha256", "a"), ("slice_sha256", "b"), (":timeout", "1"))
    poison = tuple(
        (name, "yes") for name in (
            "audited_clean", "audit_token", "barred_on_slice", "clean",
            "x" * 200, "", ":timeout ", "SMT2_SHA256",
        )
    )
    projected = V._evidence_options(
        V.SolverStamp(invoked=True, reason="probe", name="z3",
                      version="0", transport="wheel",
                      options=honest + poison))
    assert set(projected) <= V._EVIDENCE_OPTION_KEYS, (
        f"the projection let {sorted(set(projected) - V._EVIDENCE_OPTION_KEYS)} "
        f"through; it is not a whitelist"
    )
    assert projected == dict(honest), projected

    # and it is not vacuous: the keys the decision DOES need survive
    assert V._evidence_options(V.solver_absent("probe")) == {}


def _bar_body(fn) -> str:
    """One of the bar's functions with its comments and docstring removed —
    what the source scans below are entitled to read as CODE."""
    import inspect

    code = "\n".join(
        line for line in inspect.getsource(fn).splitlines()
        if not line.lstrip().startswith("#")
    )
    _, _, after = code.partition('"""')
    _, _, after = after.partition('"""')  # drop the docstring
    return after


def test_the_narrowing_decision_reads_options_in_one_place():
    """CHANNEL 6, and the second, anti-correlated mechanism for it: the token
    smuggled through the EXISTING `SolverStamp.options`, read not by
    `_evidence_is_about` but by `_bar_scope`, which is handed the stamps.

    The read ledger catches that as a new `(function, attribute)` pair. This
    catches it in the source, so a reader added where the ledger's fixture does
    not reach is caught too. `_render_one_solver` renders the whole option set
    for DISPLAY and is outside the bar entirely, which is why the scan is
    scoped to the bar's own functions rather than to the module.

    THE SECOND SCAN IS CHANNEL 7, AND THE FIRST ONE COULD NOT SEE IT. A
    conjunct that calls `_evidence_options(s)` — the PERMITTED projection —
    from inside `_bar_scope` reaches every option value the decision may see
    while matching none of `.options`, `"options"`, `'options'`. Measured on
    `e35de13`: `solver_timeout_ms=31337` turned the bar's UNKNOWN into
    VERIFIED with the full suite green in both columns. So the projection
    itself is scoped: exactly ONE function in the bar may call it, and it is
    the one the ledger names.

    **AND THE PERMITTED CALLER IS NO LONGER THE DECISION.** On `3e107cf` it
    was `_evidence_is_about` — the function that decides whether the bar
    narrows — so a conjunct on a whitelisted key's VALUE was expressible at the
    decision itself and neither of these scans could say a word about it (both
    are about SITES, and that site was permitted). `_evidence_is_about` now
    joins the forbidden list; the permitted caller is `_evidence_reproduces`,
    which returns a boolean and decides nothing else. See channel 8 in the
    block comment above for the three corruptions that measures, and
    `test_the_evidence_path_cannot_name_a_VALUE` for the half that reaches
    inside the permitted caller.
    """
    import inspect

    for fn in (V._bar_scope, V._evidence_is_about, V._barred_in_eqns,
               V._barred_primitives, V._bar_scope_phrase):
        after = _bar_body(fn)
        # every way of reaching the attribute: `x.options`, and `getattr` /
        # `__dict__` by name. The local `recorded = _evidence_options(stamp)`
        # is not a read of the attribute and is not matched.
        reads = [form for form in (".options", '"options"', "'options'")
                 if form in after]
        assert not reads, (
            f"{fn.__name__} reads `options` directly (as {reads}). The "
            f"narrowing decision reads a stamp's option set in ONE place, "
            f"`_evidence_options`, which projects it to "
            f"`_EVIDENCE_OPTION_KEYS`; a second reader reopens the key "
            f"channel without touching either"
        )
    for fn in (V._bar_scope, V._evidence_is_about, V._barred_in_eqns,
               V._barred_primitives, V._bar_scope_phrase):
        assert "_evidence_options" not in _bar_body(fn), (
            f"{fn.__name__} calls `_evidence_options`. That is the permitted "
            f"PROJECTION, not a permission to read: every whitelisted option "
            f"value reaches whoever calls it, and a conjunct on one of those "
            f"VALUES is channel 7 — measured on `e35de13` as a public keyword "
            f"argument (`solver_timeout_ms=31337`) clearing the bar with the "
            f"suite green, and on `3e107cf` again at 30000, a budget the "
            f"sweep does not sample. Exactly one function in the bar may ask "
            f"for a stamp's options, it is NOT the one that decides, and "
            f"`_ALLOWED_READS` names it"
        )
    assert '"options"' in inspect.getsource(V._evidence_options), (
        "the one permitted reader no longer reads options, so the scan above "
        "passes vacuously"
    )
    assert "_evidence_options" in _bar_body(V._evidence_reproduces), (
        "the one permitted CALLER no longer calls the projection, so the "
        "second scan above passes vacuously"
    )
    # ... and the permitted caller is not the decision: if `_bar_scope` ever
    # consulted it directly, the value zone and the narrowing would be one
    # function again and the scan above would be measuring nothing.
    assert "_evidence_reproduces" not in _bar_body(V._bar_scope), (
        "`_bar_scope` calls `_evidence_reproduces` directly. The decision "
        "consults `_evidence_is_about`, which consults the value zone — "
        "collapsing the two puts a recorded value back in the frame that "
        "decides"
    )


# -- channel 8: the value zone's own source, pinned TOTALLY ------------------
#
# The functions a recorded option value can be reached from, and the ones that
# DECIDE. Split because the two carry different rules: the value zone may name
# an attribute (it has to read one), the decision may name nothing at all.
_EVIDENCE_VALUE_ZONE = ("_evidence_options", "_evidence_budget",
                        "_reproduced_evidence", "_evidence_reproduces")
_EVIDENCE_DECISION = ("_evidence_is_about", "_bar_scope")

# The forms that smuggle a value out of the zone without returning it. `8a`
# used `globals().__setitem__(...)`; a `global` statement does the same job.
_SMUGGLERS = ("globals", "vars", "setattr", "locals")


def _fn_body_ast(fn):
    """One function's body as AST, with its docstring dropped — a docstring is
    a `str` constant and every rule below is about constants."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _value_zone_offences(nodes, allow):
    """Every way the three corruptions of channel 8 have to be SPELLED, found
    in a parsed function body. Returns a list of (rule, detail)."""
    import ast

    out = []
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant):
                value = sub.value
                if isinstance(value, (str, bytes)):
                    if value not in allow:
                        out.append(("literal", repr(value)))
                elif not (value is None or value is Ellipsis
                          or isinstance(value, bool)):
                    # NUMBERS TOO, and this is not tidiness. A conjunct on a
                    # DERIVED quantity of the budget dodges the comparison rule
                    # without any comparison at all — `if not (budget % 30000)`
                    # is a `UnaryOp` over a `BinOp` — and 30000 is not a budget
                    # the sweep samples. Measured: that exact line minted the
                    # same false narrowing as 8a. The zone needs no number of
                    # its own, so it may not spell one.
                    out.append(("literal", repr(value)))
            if isinstance(sub, ast.Compare):
                for side in [sub.left, *sub.comparators]:
                    for inner in ast.walk(side):
                        if isinstance(inner, ast.Constant) and isinstance(
                                inner.value, (str, bytes, int, float)):
                            out.append(("compare-literal", repr(inner.value)))
            if isinstance(sub, (ast.Global, ast.Nonlocal)):
                out.append(("global-statement", ", ".join(sub.names)))
            if isinstance(sub, ast.Name) and sub.id in _SMUGGLERS:
                out.append(("smuggler", sub.id))
    return out


def test_the_evidence_path_cannot_name_a_VALUE():
    """CHANNEL 8, AND THE HALF NO SITE-BASED MECHANISM COULD REACH.

    The read ledger and the two source scans are about SITES: which function
    may ask for a stamp's options. They said nothing at all about a conjunct
    written INSIDE the permitted one, and on `3e107cf` the permitted one was
    the decision itself. The only cover was
    `test_the_bars_narrowing_does_not_move_with_a_CALLER_SETTABLE_option_value`,
    which is a SAMPLE over one of the two value axes it can reach: 30000
    survives it, 31337 does not, and the only difference is the constant.

    **So this pin is over the SOURCE, and it is total.** Every one of the three
    measured corruptions has to be spelled one of four ways, and none of the
    four is available to the repaired code:

    * a string literal that is not an attribute name the READ LEDGER already
      permits — `"0" * 64`, `"30000"`, a smuggled option key. The allow-list is
      derived from `_ALLOWED_READS` rather than written out here, so adding a
      literal means adding a ledger entry, which is itself pinned in both
      directions;
    * a comparison against a literal of any type, anywhere in either operand's
      subtree — which is how 8a reads its stashed global and how 8b/8c read a
      hash. The decision functions are covered by this rule too, because 8a
      wrote its conjunct in `_bar_scope`;
    * a `global`/`nonlocal` statement, or a call through `globals()` / `vars()`
      / `setattr()` / `locals()` — 8a's actual spelling was
      `globals().__setitem__("_BSTASH", ...)`, chosen to keep the read inside
      the permitted reader and out of the ledger.

    WHAT IT DOES NOT CLAIM, said because could-not-fail shape #4 is exactly
    "enumerating current members rather than pinning the channel": this
    constrains what may be WRITTEN on the path, not every predicate Python can
    express. A discriminator spelled as a method call on a value —
    `recorded.get(k).startswith(...)`, a hash of it, a length test that dodges
    `ast.Compare` — is not matched. That residue is why the budget sweep below
    is KEPT as corroboration rather than deleted, and why the value zone is
    also made unable to AIM: `_evidence_options` is handed no query and
    `_reproduced_evidence` is handed no record, so neither can compute the
    mapping a false narrowing would have to produce (`::test_the_reproduction_
    is_handed_no_record`).
    """
    allow = {attr for _kind, _fn, attr in _ALLOWED_READS}
    assert "options" in allow and "invoked" in allow and "name" in allow, (
        f"the ledger no longer permits the attribute names the value zone "
        f"reaches for ({sorted(allow)}); this pin's allow-list is derived "
        f"from it and would now forbid an honest read"
    )

    for name in _EVIDENCE_VALUE_ZONE:
        offences = _value_zone_offences(_fn_body_ast(getattr(V, name)), allow)
        assert not offences, (
            f"`{name}` is on the path a recorded option VALUE reaches, and it "
            f"now spells {offences}. Every measured spelling of channel 8 is "
            f"one of these: a literal the record can be compared against, a "
            f"comparison against a literal, or a module global carrying a "
            f"value out of the zone. If the construct is honest, it needs a "
            f"ledger entry or a named constant at module scope — and it "
            f"should be argued in the commit rather than done quietly"
        )
    # The DECISION may not name anything at all — but it does carry f-string
    # message text, so only the comparison and smuggling rules apply.
    for name in _EVIDENCE_DECISION:
        offences = [
            (rule, detail)
            for rule, detail in _value_zone_offences(
                _fn_body_ast(getattr(V, name)), allow)
            if rule != "literal"
        ]
        assert not offences, (
            f"`{name}` DECIDES whether the bar narrows and now spells "
            f"{offences}. A comparison against a literal there is corruption "
            f"8a's shape exactly — it read a stashed budget out of a module "
            f"global and compared it to \"30000\", with every other mechanism "
            f"in this file green"
        )

    # ANTI-VACUITY: the checker must actually catch each of the three
    # corruptions. Measured against their real spellings, parsed rather than
    # described, so a checker that stopped matching fails here.
    import ast
    import textwrap

    for label, src in (
        ("8a-stash", 'def f(o):\n'
                     '    globals().__setitem__("_BSTASH", o.get(":timeout"))\n'),
        ("8a-read", 'def f():\n'
                    '    return () if globals().get("_BSTASH") == "30000" else 1\n'),
        ("8b-hash", 'def f(recorded, script):\n'
                    '    return recorded == "0" * 64 or script == recorded\n'),
        ("8c-slice", 'def f(recorded_slice, script):\n'
                     '    return recorded_slice == "0" * 64\n'),
        ("global-stmt", 'def f(v):\n    global _T\n    _T = v\n'),
        # the DERIVED-quantity spelling, which has no comparison in it at all
        ("8d-derived", 'def f(recorded):\n'
                       '    return not (_evidence_budget(recorded) % 30000)\n'),
    ):
        body = ast.parse(textwrap.dedent(src)).body[0].body
        assert _value_zone_offences(body, allow), (
            f"the source pin does not catch {label}, which was MEASURED green "
            f"through the whole suite in both columns on `3e107cf`. A pin that "
            f"does not catch its own trigger is could-not-fail shape #3"
        )
    # ... and it does not fire on the honest shape it has to permit
    assert not _value_zone_offences(
        ast.parse('def f(s):\n'
                  '    return dict(getattr(s, "options", None) or ())\n'
                  ).body[0].body, allow)


def test_the_reproduction_is_handed_no_record():
    """THE OTHER HALF OF CHANNEL 8: a conjunct inside the value zone can only
    mint a false narrowing if it can AIM.

    `test_the_evidence_path_cannot_name_a_VALUE` constrains what may be
    written; this constrains what could be written to any effect. The
    comparison is `recorded == reproduced`, so minting a narrowing means
    producing, from one side, a mapping equal to the other's:

    * `_evidence_options` is handed a stamp and nothing else. It cannot see the
      slice, so it cannot compute what `_reproduced_evidence` will return;
    * `_reproduced_evidence` is handed the slice, a flavour LABEL and an `int`.
      It cannot see the stamp, so it cannot compute what the record carries.

    Both halves are pinned at the SIGNATURE, because that is the thing that
    would have to change first. A parameter added to either — `stamp`,
    `record`, `invocations`, `recorded` — is the channel reopening, whatever
    the body then does with it.
    """
    import inspect

    assert list(inspect.signature(V._reproduced_evidence).parameters) == [
        "sliced", "flavour", "budget"
    ], (
        f"`_reproduced_evidence` now takes "
        f"{list(inspect.signature(V._reproduced_evidence).parameters)}. It "
        f"re-derives what an honest record MUST carry, and it is safe from a "
        f"conjunct precisely because it cannot see what any record DOES carry "
        f"— a fourth argument is that guarantee going away"
    )
    assert list(inspect.signature(V._evidence_options).parameters) == [
        "stamp"
    ], "the projection now sees more than the stamp; it can aim"

    # and the mirror: neither can produce the other's answer. The projection of
    # an honest stamp is not computable from the slice alone and vice versa —
    # asserted by showing the two DISAGREE when the budget is wrong, which is
    # the only lever either of them has.
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env

    on_closed = trace(_scatter_ON_the_decided_slice)
    sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    assert not isinstance(sl, DeclinedObligation)
    stamp = _stamp_recording(sl, "z3", ":timeout", 20000)
    recorded = V._evidence_options(stamp)
    assert recorded == V._reproduced_evidence(sl, "z3", 20000), (
        "the honest record and the re-derivation disagree; the narrowing "
        "cannot work at all and every assertion below is vacuous"
    )
    assert recorded != V._reproduced_evidence(sl, "z3", 20001), (
        "the re-derivation returns the same mapping at a different budget, so "
        "the budget's influence is not the one `_evidence_budget` documents "
        "and a corrupted budget could not be detected by the comparison"
    )


def test_the_reproduction_comes_from_the_stamps_own_derivation():
    """ONE DERIVATION, PINNED BY SUBSTITUTION RATHER THAN BY AGREEING.

    The bar's check and the record it checks must be built by the same
    function. If the bar re-stated "what an honest stamp's options look like",
    that statement would be a second copy to keep correct — and the two would
    agree right up until emission changed one of them.

    So `_reproduced_evidence` calls `stelling.smt.Script.stamp_options`, the
    method `stelling.solvers` stamps an invocation with. Substituting it moves
    the bar's answer, which is what proves the bar reads THAT and not a copy.
    Reading them both and asserting equality would not: two copies agree too.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import Script

    on_closed = trace(_scatter_ON_the_decided_slice)
    sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    assert not isinstance(sl, DeclinedObligation)

    honest = V._reproduced_evidence(sl, "z3", 20000)
    assert honest, "the re-derivation produced nothing; the substitution below measures nothing"

    real = Script.stamp_options
    try:
        Script.stamp_options = lambda self: real(self) + (("smt2_sha256", "x"),)
        substituted = V._reproduced_evidence(sl, "z3", 20000)
    finally:
        Script.stamp_options = real

    assert substituted != honest, (
        "substituting `Script.stamp_options` did not move the bar's "
        "re-derivation, so the bar is not built from the stamp's own "
        "derivation — it carries a second statement of what a stamp records, "
        "and the two will drift"
    )
    assert V._reproduced_evidence(sl, "z3", 20000) == honest, (
        "the substitution leaked; the assertions after this point are unsound"
    )


# -- channel 7: the VALUE of a whitelisted key ------------------------------
#
# The budgets swept below. A RULE, not a list of the values a conjunct was
# imagined to use: powers of ten and their neighbours, the canonical 20000 and
# the value one above it, a prime in the middle, and the 32-bit ceiling — so a
# threshold anywhere across seven orders of magnitude has values on both sides
# of it.
#
# THE SENTENCE THAT USED TO END THAT PARAGRAPH — "and an equality on any round
# or memorable number is hit" — IS FALSE, and was measured false. 30000 is not
# in this tuple, and neither are 2000, 15000, 25000, 50000 or 60000.
# Corruption 8a keyed on "30000" is 0 RED in this file; the IDENTICAL
# corruption keyed on "31337", which is in the tuple, is 2 RED. The only
# difference is the constant. Both measurements are in the block comment above.
#
# THE TUPLE IS NOT WIDENED IN RESPONSE, and that is the point rather than an
# omission. Adding 30000 would answer a demonstration with the reflex this
# project has watched fail six times on other axes: the next conjunct is
# written on the next unsampled value. What changed instead is that the
# narrowing DECISION no longer holds a value to key on, and the value zone's
# source is pinned totally by
# `test_the_evidence_path_cannot_name_a_VALUE`.
#
# WHAT THIS DOES NOT CLAIM, stated because the fourth could-not-fail shape is
# exactly "enumerating current members rather than pinning the channel": this
# is a SAMPLE of an unbounded value space, and a conjunct keyed on a value
# between two samples survives it — 30000 is the demonstration. It is kept as
# CORROBORATION, not as the defence: it reaches the two of the four
# whitelisted keys a caller can set, at the values it lists, and the
# structural pins reach all four keys at every value.
_CALLER_BUDGETS = (
    1, 9, 10, 99, 100, 999, 1000, 5000, 9999, 10000, 12345, 20000, 20001,
    31337, 65535, 99999, 100000, 999999, 1000000, 2 ** 31 - 1,
)


def _stamp_recording(sliced, flavour, key, budget):
    """An HONEST invocation stamp for `sliced`: the two hashes really are the
    ones emission produces at this flavour and budget, and the budget is
    recorded under the option key that flavour spells it with. Nothing here is
    forged — this is the record `escalate` writes."""
    from stelling.smt import emit

    script = emit(sliced, flavour, budget)
    return V.SolverStamp(
        invoked=True, reason="probe", name=flavour, version="0",
        transport="wheel",
        options=(("smt2_sha256", script.sha256),
                 ("slice_sha256", script.slice_sha256),
                 (key, str(budget))),
    )


def test_the_bars_narrowing_does_not_move_with_a_CALLER_SETTABLE_option_value():
    """CHANNEL 7: A CONJUNCT ON THE **VALUE** OF A WHITELISTED OPTION KEY.

    `_EVIDENCE_OPTION_KEYS` closed which keys reach the decision. It says
    nothing about what the decision does with them, and `:timeout` has to be
    in it — the budget is part of the emitted text, so it is part of the hash
    the narrowing compares. It is also the caller's own `solver_timeout_ms`,
    carried verbatim into the stamp. So this channel needs NO forged record
    and NO new field: it is driven by a public keyword argument. Measured on
    `e35de13` (`_bar_scope`, `_evidence_options(s).get(":timeout") ==
    "31337"`): `check(_scatter_ON_the_decided_slice, vacuity_mode=
    "inputs-only", solver_timeout_ms=31337)` returned VERIFIED with no note
    where 20000 returns UNKNOWN, full suite byte-identical in both columns.

    WHAT IS PINNED IS THE PROPERTY, NOT THE SPELLING: **the bar's answer is a
    function of what the evidence is ABOUT — the slice it re-derives and the
    hashes that slice reproduces — and not of any value the caller chose.**
    Both directions are swept, because a conjunct can widen as easily as
    narrow: on the honest pairing the bar must narrow to the decided slice's
    own barred set at every budget, and on the mispaired one it must fall back
    to the whole query at every budget.

    Read at `_bar_scope` rather than at `check` deliberately, and the reason
    is coverage rather than speed: no solver runs, so the budget can be swept
    over seven orders of magnitude instead of over the three or four values a
    solver-per-case test could afford.
    `test_the_bars_verdict_does_not_move_with_the_SOLVER_TIMEOUT` carries the
    surface arm, where a conjunct outside `_bar_scope` would live.

    Could-not-fail: this is at risk of shape #4 (enumerating members rather
    than pinning the channel), and `_CALLER_BUDGETS` says so in its own
    comment. The structural half of the repair — the read ledger's attribution
    and the source scan — is what covers the values this cannot.
    """
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    on_closed = trace(_scatter_ON_the_decided_slice)
    el_closed = trace(_scatter_ELSEWHERE_same_shape)
    on_sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    assert not isinstance(on_sl, DeclinedObligation)

    # the pair the whole-query bar cannot separate, so a narrowing that moves
    # is really the narrowing moving (the same premise as
    # `test_a_mispaired_query_that_still_SLICES_cannot_clear_the_bar`)
    assert V._barred_primitives(on_closed) == V._barred_primitives(el_closed)
    assert V._barred_in_eqns(on_sl.eqns) == ("scatter",)

    seen_narrow = seen_widen = 0
    for flavour, key in (("z3", ":timeout"), ("cvc5", ":tlimit")):
        for budget in _CALLER_BUDGETS:
            stamp = _stamp_recording(on_sl, flavour, key, budget)

            barred, why = V._bar_scope(on_closed, {0: (stamp,)})
            assert (barred, "assert #0" in why) == (("scatter",), True), (
                f"{flavour} {key}={budget}: the bar answered {barred!r} / "
                f"{why!r} on the query the evidence really is about, where "
                f"every other budget answers ('scatter',). The narrowing "
                f"moved with a value the CALLER chose — `solver_timeout_ms` "
                f"is carried into the stamp verbatim, so this is a public "
                f"keyword argument deciding whether a bar fires"
            )
            assert "fell back" not in why
            seen_narrow += 1

            barred_m, why_m = V._bar_scope(el_closed, {0: (stamp,)})
            assert "fell back to the whole query" in why_m, (
                f"{flavour} {key}={budget}: an escalation produced on one "
                f"query and stamped against another cleared the bar "
                f"({barred_m!r} / {why_m!r}). The evidence check narrowed on "
                f"a caller-chosen value instead of on what the evidence is "
                f"about"
            )
            assert barred_m == V._barred_primitives(el_closed)
            seen_widen += 1

    # ANTI-VACUITY, both halves: the sweep must actually have exercised both
    # directions, and must span more than one decade — a sweep collapsed to
    # one budget would pass every assertion above while measuring nothing.
    assert seen_narrow == seen_widen == 2 * len(_CALLER_BUDGETS) >= 20
    assert max(_CALLER_BUDGETS) // max(min(_CALLER_BUDGETS), 1) >= 10 ** 6, (
        "the budget sweep no longer spans six orders of magnitude; it has "
        "been shrunk back to a list of the values someone thought of"
    )


@pytest.mark.parametrize("budget", [20000, 20001, 31337, 65535])
def test_the_bars_verdict_does_not_move_with_the_SOLVER_TIMEOUT(budget):
    """CHANNEL 7 AT THE SURFACE — the arm that reaches a conjunct written
    anywhere in the assembly rather than inside `_bar_scope`.

    `solver_timeout_ms` is the public keyword argument the channel rides on.
    Whatever it is set to, a solver-path VERIFIED resting on a slice that
    carries the barred primitive must be withheld, and one that does not must
    not be. Every budget here is large enough for these two trivial queries to
    be answered, so a moving verdict is the bar moving and not the solver
    timing out — which the `discharged` assertion below makes explicit.

    Deliberately NOT the same read as the test above: that one drives
    `_bar_scope` with a hand-built stamp and cannot see a conjunct in
    `make_solver_verdict` or `_bar_domain`; this one drives the whole
    pipeline, at the price of a narrower sweep.
    """
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    barred = check(_scatter_ON_the_decided_slice,
                   vacuity_mode="inputs-only", solver_timeout_ms=budget)
    assert all(o.status == "discharged" for o in barred.obligations), (
        f"budget {budget} did not decide every obligation, so an UNKNOWN here "
        f"would not be the bar"
    )
    assert barred.status == "UNKNOWN" and [
        n for n in barred.notes if "VERIFIED withheld" in n and "assert #0" in n
    ], (
        f"solver_timeout_ms={budget}: {barred.status}, notes {barred.notes}. "
        f"A caller-chosen budget decided whether the scatter bar fired"
    )

    clean = check(_scatter_OFF_the_decided_slice,
                  vacuity_mode="inputs-only", solver_timeout_ms=budget)
    assert clean.status == "VERIFIED" and not any(
        "VERIFIED withheld" in n for n in clean.notes
    ), (
        f"solver_timeout_ms={budget}: {clean.status} on a query whose decided "
        f"slice is scatter-free — the bar moved with the budget in the "
        f"WIDENING direction, which the barred arm above cannot see"
    )


class _EqualButNotInt(int):
    """Equal to its value, `type()` is not `int`. Channel 5's exploit."""


class _EqualButNotStr(str):
    """Equal to its value, `type()` is not `str`."""


def test_the_bars_decision_does_not_look_at_the_TYPE_of_a_record_field():
    """CHANNEL 5: a conjunct that reads NO new field and adds NO new key —
    `type(r.index) is not int` in the bar's domain, cleared by handing it an
    `int` SUBCLASS. Measured on this branch: UNKNOWN becomes VERIFIED, full
    suite green, and every field-removal and field-probe test stays green
    because nothing new is read and no value is unusual.

    What is pinned is the property that closes it: **the bar's domain is a
    function of the VALUES of `index` and `outcome`, not of their runtime
    types.** A subclass that compares equal must produce the same domain and
    the same verdict. That covers every type-identity discrimination — `type(…)
    is`, `is not`, `__class__`, a `type(...).__name__` check — rather than the
    one spelling this test's fixture uses.
    """
    import dataclasses

    from stelling.solvers import _bar_domain, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    honest = make_solver_verdict(closed, prop, esc, **VERSIONS)
    assert honest.status == "UNKNOWN", "the genuine assembly does not bar"

    subclassed = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, index=_EqualButNotInt(r.index),
                            outcome=_EqualButNotStr(r.outcome))
        for r in esc.records
    ))
    # the fixture really does present a different TYPE at an equal VALUE
    assert [type(r.index) for r in subclassed.records] != [
        type(r.index) for r in esc.records
    ]
    assert all(r.index == s.index and r.outcome == s.outcome
               for r, s in zip(subclassed.records, esc.records))

    assert _bar_domain(subclassed) == _bar_domain(esc), (
        "the bar's domain moved when the records' fields changed TYPE at the "
        "same VALUE — the domain is discriminating on `type()`, which no "
        "field-removal or value-probe test can see"
    )
    v = make_solver_verdict(closed, prop, subclassed, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: an `int`/`str` subclass carrying the same values as an "
        f"honest record cleared the bar"
    )
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), "the subclass un-discharged an obligation; the probe passes vacuously"


def test_the_bar_is_consulted_with_exactly_that_domain(monkeypatch):
    """The surface half: `make_solver_verdict` must hand `_bar_scope` what
    `_bar_domain` returns and nothing else.

    Without this, the test above pins an invariant at its producer and never
    at the place it matters — a conjunct written at the CALL SITE rather than
    inside `_bar_domain` would leave that test green. Checked on the honest
    assembly and on every `_field_probes` assignment, so a call-site conjunct
    keyed on any probed value moves the captured domain away from the
    computed one.
    """
    import dataclasses

    from stelling.solvers import _bar_domain, make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    seen: list = []
    real = V._bar_scope

    def spy(c, decided):
        seen.append(decided)
        return real(c, decided)

    monkeypatch.setattr(V, "_bar_scope", spy)

    probes = _field_probes()
    load_bearing = {"index", "outcome"}
    from stelling.solvers import ObligationEscalation
    fields = [f for f in dataclasses.fields(ObligationEscalation)
              if f.name not in load_bearing]
    cases = [esc] + [
        dataclasses.replace(esc, records=tuple(
            dataclasses.replace(r, **{f.name: probes[str(f.type)][which]})
            for r in esc.records
        ))
        for f in fields for which in (0, 1)
    ]
    for case in cases:
        seen.clear()
        make_solver_verdict(closed, prop, case, **VERSIONS)
        assert len(seen) == 1, (
            f"the bar was consulted {len(seen)} time(s); a domain that came "
            f"back empty means some conjunct removed every discharged record "
            f"from it between `_bar_domain` and the call"
        )
        assert seen[0] == _bar_domain(case), (
            f"`make_solver_verdict` handed the bar {seen[0]!r} where "
            f"`_bar_domain` returns {_bar_domain(case)!r} — the domain is "
            f"being filtered at the call site, outside the one place the "
            f"channel is pinned"
        )


class _InvocationsThatRaise:
    """A record whose ``invocations`` raises something `_bar_domain`'s INNER
    guard does not catch, so the OUTER `except` is the one that runs.

    Everything else on it is an honest value, so the assembly proceeds
    normally and the only thing that changes is that the bar's domain cannot
    be built."""

    def __init__(self, record):
        self.index = record.index
        self.outcome = record.outcome
        self.detail = record.detail
        self.witness = record.witness
        self.notes = record.notes
        self.answered_by = record.answered_by

    @property
    def invocations(self):
        raise ValueError("this record's stamps cannot be read")


def test_an_UNREADABLE_domain_widens_the_bar_and_the_sentinel_is_why():
    """THE OUTER `except` OF `_bar_domain`, WHICH NOTHING DROVE.

    `_bar_domain` has two guards. The inner `except TypeError` around
    `tuple(r.invocations)` is exercised three times over
    (`test_a_bar_must_never_BREAK_a_verdict_either`). The OUTER one — the one
    that returns `_BAR_DOMAIN_UNREADABLE` — was driven by nothing at all, and
    so neither was the sentinel it returns.

    That left the sentinel's TRUTHINESS unpinned, and truthiness is its whole
    mechanism: the bar branch is guarded on `decided` being non-empty, because
    an empty domain honestly means "no solver decided anything". Measured on
    this branch: `__bool__` returning `False` instead of `True` turns this
    UNKNOWN into VERIFIED, with no withheld note and the full suite green. The
    docstring said the sentinel "widens rather than silencing"; nothing
    checked it.

    Both halves are asserted, because either alone could pass for the wrong
    reason: that the unreadable escalation really does reach the sentinel
    (otherwise the assembly below is measuring an ordinary domain), and that
    the assembled verdict is withheld with the whole-query reason.
    """
    import dataclasses

    from stelling.solvers import (
        _BAR_DOMAIN_UNREADABLE,
        _bar_domain,
        make_solver_verdict,
    )

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    unreadable = dataclasses.replace(esc, records=tuple(
        _InvocationsThatRaise(r) for r in esc.records))

    assert _bar_domain(unreadable) is _BAR_DOMAIN_UNREADABLE, (
        "the record does not reach `_bar_domain`'s outer `except`, so this "
        "test drives the sentinel path in name only"
    )
    v = make_solver_verdict(closed, prop, unreadable, **VERSIONS)
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), (
        "the unreadable record un-discharged an obligation, so there is no "
        "VERIFIED for the bar to withhold and this test passes vacuously"
    )
    assert v.status == "UNKNOWN", (
        f"{v.status}: an escalation whose domain could not be read cleared "
        f"the bar. The sentinel is truthy so the bar branch is ENTERED and "
        f"`_bar_scope` widens to the whole query; a falsy sentinel is spelled "
        f"the same way as an honest empty domain, which SILENCES the bar"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)
    assert any("fell back to the whole query" in n for n in v.notes), (
        "the bar fired but not through the fallback, so the sentinel is not "
        "what produced this UNKNOWN"
    )


class _TwoFaced:
    """A `records` that shows one face to the FIRST reader and another to
    every later one. Ordering cannot defend against this — the domain really
    is read first, it is just read from a different value."""

    def __init__(self, first, later):
        self.first, self.later, self.passes = first, later, 0

    def __iter__(self):
        self.passes += 1
        return iter(self.first if self.passes == 1 else self.later)


@pytest.mark.parametrize("build,expected,withheld", [
    (_scatter_ON_the_decided_slice, "UNKNOWN", True),
    (_scatter_free, "VERIFIED", False),
])
def test_a_ONE_SHOT_records_behaves_EXACTLY_LIKE_THE_TUPLE_it_yields(
    build, expected, withheld
):
    """THE OTHER WAY PAST THE SENTINEL, WHICH DOES NOT GO THROUGH IT AT ALL —
    AND THE COST OF THE FIRST REPAIR FOR IT.

    `_UnreadableBarDomain` defends against a `records` that cannot be READ. It
    says nothing about one that can be read ONCE. `make_solver_verdict` walks
    `escalation.records` five times, and while the bar's domain was built at
    the bar — several passes down — a generator, a `map`, or any consumed
    iterator was exhausted by the obligation loop first. `_bar_domain` then
    returned an HONEST-EMPTY `{}`, which is exactly the value that means "no
    solver decided anything" and skips the bar. Measured on `eb1ff86` and on
    `f5280cf`: VERIFIED, with no withheld note, on the bar's own fixture — a
    silencing path that never touched the sentinel.

    THE FIRST REPAIR WAS ORDERING, AND ITS PRICE WAS UNDISCLOSED. Reading the
    domain first made the bar's fixture fail safe, and the claim recorded for
    it was "a degenerate `records` costs the discharges, never the bar". That
    is broader than it reads. Measured on `e35de13`, on a SCATTER-FREE query —
    one the bar never touches, so nothing about the bar is at stake — a
    one-shot `records` turned an honest VERIFIED into UNKNOWN, and the only
    note it carried was the generic undecided-cause line, which attributes the
    UNKNOWN to "the propagated interval straddling the asserted bound". Not
    silence: a WRONG EXPLANATION of a verdict the argument's shape caused.

    So the repair is now ONE PASS, taken at the top: a degenerate `records`
    behaves exactly like the tuple it yields, and nothing downstream can
    disagree with anything else about what the records are. Both arms are
    parametrised here because they fail in opposite directions — the barred
    query must still bar (with the discharges intact, which the ordering
    repair could not deliver) and the scatter-free query must still VERIFY.

    Could-not-fail: replacing an assertion is where shape #3 (asserting away
    its own trigger) lives, so the ORIGINAL defect is asserted harder than
    before rather than dropped — `withheld` is checked, and the barred arm now
    requires every obligation to discharge, which is what makes its UNKNOWN
    the bar and not an accident.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(build)
    honest = make_solver_verdict(closed, prop, esc, **VERSIONS)
    assert honest.status == expected, (
        f"the genuine assembly is {honest.status}, not {expected}; the "
        f"fixture is wrong and the comparison below measures nothing"
    )

    one_shot = dataclasses.replace(esc, records=iter(tuple(esc.records)))
    v = make_solver_verdict(closed, prop, one_shot, **VERSIONS)
    assert v.status == expected, (
        f"{v.status} where the same records as a tuple give {expected}. A "
        f"`records` that can only be iterated once is being read by some "
        f"passes and not others — which silences the bar when the obligation "
        f"loop wins the race, and costs an honest VERIFIED when the bar does"
    )
    assert [o.status for o in v.obligations] == [
        o.status for o in honest.obligations
    ], (
        f"{[o.status for o in v.obligations]} vs "
        f"{[o.status for o in honest.obligations]}: a one-shot `records` "
        f"changed which obligations were decided"
    )
    assert any("VERIFIED withheld" in n for n in v.notes) is withheld, (
        f"withheld-note presence moved on a one-shot `records`: {v.notes}"
    )


@pytest.mark.parametrize("build,honest", [
    (_scatter_ON_the_decided_slice, "UNKNOWN"),
    (_scatter_free, "VERIFIED"),
])
def test_a_TWO_FACED_records_is_REFUSED_and_not_absorbed(build, honest):
    """THE SHAPE ORDERING CANNOT REACH — AND THE ONE ONE PASS DID NOT CLOSE
    EITHER, WHICH IS WHY THIS IS A REFUSAL NOW.

    Reading the bar's domain FIRST is a defence against a `records` that runs
    OUT. It is no defence at all against one that CHANGES: an iterable that
    yields nothing on its first pass and the real records on every later one
    shows the bar an honest-empty domain — the one value that skips it — and
    the obligation loop a full set of discharging records. The domain really
    was read first; it was just read from a different value. Measured on
    `e35de13`: VERIFIED, every obligation discharged, no withheld note, on the
    bar's own fixture.

    ONE PASS CLOSED THAT ONE AND NOT THE OTHER, AND THE COMMENT SAID
    OTHERWISE. `make_solver_verdict` materialising `records` at the top makes
    every reader see one value, so the bar and the discharges cannot be told
    different things. But one pass at the top IS choosing pass 1, and for a
    two-faced `records` pass 1 is the empty face. Measured on `3e107cf`:

        scatter-free query, `records` empty on pass 1 and real after
            -> VERIFIED becomes UNKNOWN, obligation `unknown`, carrying
               "…the propagated interval straddling the asserted bound"

    which is VERBATIM the defect `SOUNDNESS.md` (2) recorded the one-pass
    repair as closing. It closed it for the ONE-SHOT shape, where pass 1 is
    where the real records are. The scatter-free arm is parametrised here
    because it is the row that separates the two: the bar never fires on it, so
    nothing about the bar is at stake and the whole of what is measured is
    whether the verdict is honest about why it is UNKNOWN.

    So the shape is REFUSED. The ledger is a separate field carried whole and
    is an independent witness that solvers ran; an escalation that says they
    ran and hands over no record of what they answered cannot have come from
    `escalate()`. Refusing beats absorbing, because absorbing produced a wrong
    EXPLANATION rather than silence.

    Could-not-fail: replacing an assertion with an expected exception is where
    shape #3 lives (asserting away the trigger), so the ORIGINAL defect is
    asserted harder rather than dropped — the refusal is checked to be THIS
    gate by its own words, the honest assembly is checked to still produce its
    honest verdict, and the fixture is checked to really carry ledger work,
    without which the gate would not fire and this would measure nothing.
    """
    import dataclasses

    import pytest as _pytest

    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    closed, prop, esc = _stamped(build)
    real = tuple(esc.records)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == honest, (
        f"the genuine assembly is not {honest}; the fixture is wrong and the "
        f"comparison below measures nothing"
    )
    assert esc.ledger.spawns and real, (
        "the fixture carries no ledger work or no records, so the coherence "
        "gate below could not fire and this test passes vacuously"
    )

    two_faced = _TwoFaced((), real)
    with _pytest.raises(MispairedEscalationError) as exc:
        make_solver_verdict(
            closed, prop, dataclasses.replace(esc, records=two_faced),
            **VERSIONS)
    assert two_faced.passes >= 1, "the fixture was never iterated at all"
    assert "incoherent escalation" in str(exc.value) and (
        "came back" in str(exc.value)
    ), (
        f"the assembly refused, but not through the coherence gate: "
        f"{exc.value}. Some other guard is firing, and the shape this test is "
        f"about would come back if that guard moved"
    )


def test_a_degenerate_records_with_NO_ledger_work_is_still_assembled():
    """THE SCOPE OF THE COHERENCE GATE, so it is not read as "empty `records`
    is refused".

    The nothing-to-escalate shape — no records, no spawns, no stamps — is a
    legitimate escalation and every other gate in `make_solver_verdict` exempts
    it explicitly. The coherence gate keys on the LEDGER saying solvers ran, so
    it must not touch that shape: an interval-only propagation assembled
    against an empty escalation still returns its honest verdict.

    Without this, the gate would be a rule about `records` being empty rather
    than about the escalation contradicting itself, and the exemption every
    neighbouring gate carries would be silently gone.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_free)
    empty = dataclasses.replace(
        esc, records=(), notes=(),
        ledger=dataclasses.replace(esc.ledger, spawns=0, stamps=()))
    v = make_solver_verdict(closed, prop, empty, **VERSIONS)
    assert v.status in ("VERIFIED", "UNKNOWN"), v.status
    assert not any("incoherent" in n for n in v.notes)


@pytest.mark.parametrize("build,strip,label", [
    (_scatter_ON_the_decided_slice, 0, "the scatter obligation, alone"),
    (_two_solver_decided_obligations, 0, "the scatter obligation, of two"),
])
def test_stripping_invocations_cannot_clear_the_bar(build, strip, label):
    """ONE CONCEPT, ONE PREDICATE — the regression for the drift that made
    `discharging` and `decided` two different tests over one record.

    `make_solver_verdict` discharged an obligation on
    `record.outcome == OB_DISCHARGED` and took the bar's domain from
    `outcome == OB_DISCHARGED and r.invocations`. The conjunct looks like a
    hardening and behaves like a second definition: empty `invocations` on a
    discharging record and the obligation still discharges — so the VERIFIED
    still stands — while its slice silently leaves the bar's domain.

    Measured on both shapes, against `8e42934` (whole-query bar) as control:

        fixture                         base      drifted   here
        one decided obligation          UNKNOWN   VERIFIED  UNKNOWN
        two, strip the scatter one's    UNKNOWN   VERIFIED  UNKNOWN

    The one-obligation row is the sharper of the two, because it ALSO closes a
    hole the base had: with every record's `invocations` gone the old
    `solver_decided` was False and the bar branch was never entered at all, so
    `8e42934` returned VERIFIED there too. Both rows are UNKNOWN now.

    This is a regression, not a defence. `make_solver_verdict`'s docstring
    states the precondition — the escalation came from `escalate()` on this
    query — and a caller who can hand-edit a record can hand-build a
    `Verdict` and skip this function entirely. What is pinned is that the
    bar's domain and the discharge cannot disagree about the SAME record.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed, prop, esc = _stamped(build)
    assert make_solver_verdict(closed, prop, esc, **VERSIONS).status == (
        "UNKNOWN"
    ), f"{label}: the genuine assembly does not bar; the fixture is wrong"
    assert any(r.index == strip and r.invocations for r in esc.records), (
        f"{label}: record #{strip} carries no invocations to strip, so this "
        f"test cannot see the drift"
    )

    forged = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, invocations=()) if r.index == strip else r
        for r in esc.records
    ))
    v = make_solver_verdict(closed, prop, forged, **VERSIONS)
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), (
        f"{label}: emptying `invocations` un-discharged an obligation, so "
        f"there is no VERIFIED left for the bar to withhold and this test "
        f"passes for the wrong reason"
    )
    assert v.status == "UNKNOWN", (
        f"{label}: {v.status} — a record kept its discharge and left the "
        f"bar's domain, so `decided` and `discharged` are two predicates "
        f"again"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


@pytest.mark.parametrize("invocations,label", [
    ([], "an empty list"),
    (None, "None"),
    (7, "not iterable at all"),
])
def test_a_bar_must_never_BREAK_a_verdict_either(invocations, label):
    """"A bar must never break a verdict" has to cover the whole path FEEDING
    the bar, and at `eb1ff86` it did not.

    `_bar_scope` wraps its own body in `except Exception`. The read that
    builds its domain sat OUTSIDE that `try`, in `make_solver_verdict`, and
    was spelled `decided.get(r.index, ()) + r.invocations` — so a record whose
    `invocations` is a plain `list` raised `TypeError` out of the public
    assembly function. `45cf526` tolerated the same record.

    Both directions are required here. A tolerated shape must produce a
    verdict, and an untolerated one must WIDEN — never raise, and never go
    quiet, because an empty domain is how "no solver decided anything" is
    spelled and that silences the bar.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)
    forged = dataclasses.replace(esc, records=tuple(
        dataclasses.replace(r, invocations=(
            list(r.invocations) if invocations == [] else invocations
        ))
        for r in esc.records
    ))
    v = make_solver_verdict(closed, prop, forged, **VERSIONS)
    assert [o.status for o in v.obligations] == (
        ["discharged"] * len(v.obligations)
    ), f"{label}: nothing left to withhold, so this proves nothing"
    assert v.status == "UNKNOWN", (
        f"{label}: {v.status} — a record shape the bar cannot read cleared it"
    )
    assert any("VERIFIED withheld" in n for n in v.notes)


def test_the_containment_guard_short_circuits_before_any_re_slicing(monkeypatch):
    """SOUNDNESS.md's CONTAINMENT ARGUMENT, pinned at the mechanism it cites.

    The entry says every verdict the slice-scoped bar fires on is one the
    whole-query bar fired on too, and the reason it gives is structural:
    `_bar_scope` evaluates `whole = _barred_primitives(closed)` — the old
    bar's own set — FIRST, and returns an empty scope without re-slicing when
    it is empty. So "the new bar fired" implies "the whole-query set was
    non-empty" implies "the old bar fired", with no dependence on the two
    walks agreeing about anything.

    The weaker argument (a slice's equations come from the query, and both
    roots use the same walk) is also true and is pinned in
    `tests/test_bar_walk_parity.py`. It is not what the containment rests on,
    and the difference is the whole point: it would stop holding the moment
    the roots disagreed, which is the failure mode this repo keeps finding.

    Asserted as ORDER, not as outcome, and deliberately. Deleting the guard
    leaves every verdict in this suite unchanged — the re-derivation on a
    scatter-free query finds nothing on any slice and returns the same empty
    scope — so an outcome assertion cannot see it and the argument would be
    unpinned prose. What is observable is that the slicer is never consulted.
    """
    import stelling.obligation as _ob

    closed = trace(_scatter_free)
    assert not V._barred_primitives(closed), (
        "the fixture's whole-query barred set is non-empty, so the guard "
        "would not short-circuit and this test measures nothing"
    )

    calls = []
    real = _ob.slice_obligation

    def spy(*args, **kwargs):
        calls.append(args[1] if len(args) > 1 else None)
        return real(*args, **kwargs)

    monkeypatch.setattr(_ob, "slice_obligation", spy)
    assert V._bar_scope(closed, {0: ()}) == ((), "")
    assert calls == [], (
        f"`_bar_scope` re-sliced obligation(s) {calls} on a query whose "
        f"whole-query barred set is EMPTY. The early return is gone, so the "
        f"new bar's fired-set is no longer contained in the old one by "
        f"construction — it now depends on the slice walk finding nothing the "
        f"query walk missed. SOUNDNESS.md cites the guard; restore it or "
        f"re-derive the containment."
    )

    # ANTI-VACUITY: the spy must be reachable at all, or `calls == []` holds
    # for the trivial reason that nothing ever calls `slice_obligation`.
    dirty = trace(_scatter_ON_the_decided_slice)
    assert V._barred_primitives(dirty), "the control query carries nothing"
    V._bar_scope(dirty, {0: ()})
    assert calls, (
        "`_bar_scope` did not reach `slice_obligation` even on a "
        "scatter-bearing query, so the assertion above is vacuous"
    )


def test_the_note_names_only_the_obligations_whose_slice_carries_it():
    """FINDING 3: the message must not claim a scope the mechanism lacks.

    With two solver-decided obligations the predecessor built `{where}` from
    ALL of them while `{prims}` was the UNION, and nothing intersected the
    two: it rendered "the emitted slice of assert #0, assert #1 contains
    scatter" while #1's slice was `['add','le','sub']`. The single-deciding-
    obligation fixture above cannot see that, which is why this one exists.
    """
    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_two_solver_decided_obligations)
    # the bar's own predicate, verbatim — not a paraphrase of it
    deciding = [r.index for r in esc.records if r.outcome == "discharged"]
    assert sorted(deciding) == [0, 1], (
        f"the fixture no longer has TWO solver-decided obligations "
        f"({deciding}), so it cannot see the misattribution"
    )
    v = make_solver_verdict(closed, prop, esc, **VERSIONS)
    assert v.status == "UNKNOWN"
    (withheld,) = [n for n in v.notes if "VERIFIED withheld" in n]
    assert "assert #0" in withheld
    assert "assert #1" not in withheld, (
        f"the note names assert #1, whose emitted slice carries no barred "
        f"primitive — `where` is built from every deciding obligation while "
        f"`prims` is the union over them:\n  {withheld}"
    )


def test_the_bar_does_not_touch_interval_only_verdicts():
    """The negative half, and the reason the bar is scoped to the solver path.

    HeatNode's Dirichlet writeback puts `scatter` in the jaxpr, and this
    obligation is discharged by intervals alone. A whole-verdict bar would
    withhold the Richardson flagship for a reason having nothing to do with
    the emission row under audit — the interval transfer is long-standing and
    unchanged by that work.
    """
    pytest.importorskip("maddening")  # jax CI job does not install it
    from maddening.nodes.heat import HeatNode

    node = HeatNode("h", timestep=0.001, n_cells=5, length=1.0,
                    thermal_diffusivity=0.01)

    def row7():
        T = any_array((5,), "float64", (10.0, 100.0))
        st = {"temperature": T}
        full = node.update(st, {}, 0.01)
        half = node.update(node.update(st, {}, 0.005), {}, 0.005)
        scale = 0.0 + 1e-6 * jnp.maximum(jnp.abs(half["temperature"]),
                                         jnp.abs(full["temperature"]))
        return (assert_(scale > 0.0),)

    cj = jax.make_jaxpr(row7)()
    assert any(str(e.primitive) == "scatter" for e in cj.jaxpr.eqns), (
        "this test is vacuous unless the query really does contain scatter"
    )
    v = check(row7, vacuity_mode="inputs-only")
    assert v.status == "VERIFIED"
    assert not any("VERIFIED withheld" in n for n in v.notes)
