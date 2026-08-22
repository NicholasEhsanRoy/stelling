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
``test_the_two_pairing_gates_bind_the_ESCALATION_AND_the_propagation``, which
was the residue and is now the second gate: `Propagation` carries its own
``query_sha256`` since B11 (audit 0.2.0 B6 re-audit UNSOUND-3), so a stranger
propagation degrades to UNKNOWN instead of minting. The bar's own mispairing
tests satisfy BOTH gates by hand (``_past_the_pairing_gate`` and
``_past_the_propagation_gate``) so that no one of the three mechanisms can
hide another's failure.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # zero-dep CI has no jax
import jax.numpy as jnp

import stelling.verdict as V
from stelling.harness import any_array, assert_, assume, trace
from stelling.preconditions import check
from _solver_gate import need_solver  # noqa: E402


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


@need_solver
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
    """ANTI-VACUITY for the test above (docs/norms.md, "A measurement whose result is an ABSENCE needs a positive control"). If the fixture's scatter
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


@need_solver
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


def _past_the_propagation_gate(prop, closed):
    """The mispaired PROPAGATION with its recorded query hash OVERWRITTEN to
    the query it is about to be stamped against — the second pairing gate,
    deliberately satisfied by hand, for exactly the reason its sibling above
    is.

    THERE ARE TWO GATES NOW, AND EVERY BAR MISPAIRING TEST HAS TO GET PAST
    BOTH. `stelling.propagate.Propagation` gained a `query_sha256` in B11
    (audit 0.2.0 B6 re-audit UNSOUND-3) and `make_solver_verdict` refuses a
    propagation that is not the stamped query's — so a mispaired assembly
    now degrades to UNKNOWN before the bar is consulted at all, which is
    could-not-fail shape #7 again: a fixture that never reaches the guard's
    condition. Measured: with only the escalation's hash bypassed, every
    fixture below returned UNKNOWN with `obligations=()` and the
    unpaired-propagation note — and the gate returns before
    `verdict._bar_scope` is called at all, so none of them was reaching the
    bar to be measured by.

    Bypassing it here changes no fixture's OUTCOME: each of these tests keeps
    the assertions it carried on `207faca`, unedited, and passes them. That
    keeps the bar measured as the third, anti-correlated mechanism it is:
    the escalation gate keys on the escalation's recorded hash, the
    propagation gate on the propagation's, and the bar on the decided
    slice's fingerprint and script. None derives from another.

    That the propagation gate itself fires on these same shapes WITHOUT this
    bypass is `tests/test_propagation_identity.py`, one row per consumption
    site.
    """
    import dataclasses

    return dataclasses.replace(prop, query_sha256=closed.content_hash())


@need_solver
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

    v = make_solver_verdict(dirty, _past_the_propagation_gate(prop, dirty),
                            _past_the_pairing_gate(esc, dirty), **VERSIONS)
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


@need_solver
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

    v = make_solver_verdict(el_closed,
                            _past_the_propagation_gate(on_prop, el_closed),
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


@need_solver
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

    v = make_solver_verdict(el_closed,
                            _past_the_propagation_gate(on_prop, el_closed),
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


@need_solver
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
    v = make_solver_verdict(
        el_closed, _past_the_propagation_gate(on_prop, el_closed), forged,
        **VERSIONS)
    assert v.status == "VERIFIED", (
        f"{v.status}: with BOTH pairing hashes forged to match, the assembly "
        f"no longer reaches the false VERIFIED — so the refusal above is not "
        f"this gate and this test is measuring something else"
    )
    # ... AND THE PROPAGATION GATE IS INDEPENDENTLY LOAD-BEARING HERE: forge
    # only the escalation's hash and the assembly still refuses, on the other
    # leg. Neither gate is redundant on this fixture, which is what makes
    # forging both above a bypass rather than a weakening.
    only_esc = make_solver_verdict(el_closed, on_prop, forged, **VERSIONS)
    assert only_esc.status == "UNKNOWN" and only_esc.obligations == (), (
        f"{only_esc.status}: with the ESCALATION hash forged and the "
        f"propagation left a stranger, the assembly reached a verdict about "
        f"this query's obligations anyway"
    )


@need_solver
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
    v = make_solver_verdict(
        el_closed, _past_the_propagation_gate(on_prop, el_closed), forged,
        **VERSIONS)
    assert v.status == "VERIFIED" and not any(
        "VERIFIED withheld" in n for n in v.notes
    ), (
        f"{v.status}: with BOTH pairing hashes forged the assembly did not "
        f"mint the false VERIFIED, so this row is being closed by something "
        f"other than the pairing gates"
    )
    # ... and, as above, the propagation gate alone also refuses this shape
    only_esc = make_solver_verdict(el_closed, on_prop, forged, **VERSIONS)
    assert only_esc.status == "UNKNOWN" and only_esc.obligations == (), (
        f"{only_esc.status}: with the ESCALATION hash forged and the "
        f"propagation left a stranger, the assembly reached a verdict about "
        f"this query's obligations anyway"
    )


@need_solver
def test_the_two_pairing_gates_bind_the_ESCALATION_AND_the_propagation():
    """THE RESIDUE THIS TEST USED TO DISCLOSE IS CLOSED, and this is the
    measurement of the mechanism that closed it (audit 0.2.0 B6 re-audit
    UNSOUND-3, B11).

    **What it said before.** The gate bound two of `make_solver_verdict`'s
    three arguments — `closed` and `escalation` — and NOT `propagation`, "and
    the reason is mechanical: `Propagation` is defined in
    `stelling.propagate`, which this repair was required to leave at zero
    line delta, so there is no field on it to record the query in." That
    scoping constraint was a per-batch one and it no longer binds:
    `Propagation` now carries a `query_sha256`, stamped at `propagate`'s
    single construction site, and every site that consumes a propagation
    against a query checks it (`tests/test_propagation_identity.py`, one row
    per site).

    **What is measured here.** The assembly of (query A, propagation of query
    B, escalation of query A) — the exact residue — now returns UNKNOWN with
    no obligations and the unpaired-propagation reason, where it returned
    VERIFIED on `207faca` and on the released `v0.1.0`.

    **And the two gates are INDEPENDENT, which is the part a single status
    assertion cannot show.** Each is driven with the other satisfied by hand:
    the escalation mispairing still RAISES with the propagation paired
    honestly, and the propagation mispairing still degrades with the
    escalation's hash forged to match. Neither is doing the other's work.
    """
    import dataclasses

    from stelling.solvers import MispairedEscalationError, make_solver_verdict

    true_closed, true_prop, true_esc = _stamped(_scatter_free_TRUE_two_obligations)
    false_closed, false_prop, false_esc = _stamped(
        _scatter_free_REFUTED_two_obligations)

    assert false_prop != true_prop, "the two propagations are equal; nothing to mix"
    assert make_solver_verdict(
        false_closed, false_prop, false_esc, **VERSIONS
    ).status == "REFUTED", "the false query is not false; the fixture is wrong"

    # LEG 1 — the escalation. Driven with the PROPAGATION paired honestly, so
    # the raise cannot be the propagation gate's doing.
    with pytest.raises(MispairedEscalationError):
        make_solver_verdict(false_closed, false_prop, true_esc, **VERSIONS)

    # LEG 2 — the propagation. Driven with the ESCALATION's hash forged to
    # match, so the refusal cannot be the escalation gate's doing.
    forged = dataclasses.replace(
        true_esc, query_sha256=true_closed.content_hash())
    v = make_solver_verdict(true_closed, false_prop, forged, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: the mixed-propagation assembly still reaches a definite "
        f"verdict — this is the residue UNSOUND-3 was about, reopened"
    )
    assert v.obligations == (), (
        "the mispaired propagation's obligations are still reported under "
        "this query's name, which is the misattribution the status alone "
        "does not rule out"
    )
    assert v.stamp.query_content_hash == true_closed.content_hash(), (
        "the stamp does not name the query it was assembled against, which "
        "would be a different defect from the one this test measures"
    )
    assert v.notes and v.notes[0].startswith("unpaired propagation:")

    # NON-VACUITY: with the propagation's identity ALSO forged, the old
    # misattribution comes straight back — so what closed the row above is
    # this gate and not some unrelated guard.
    v2 = make_solver_verdict(
        true_closed, _past_the_propagation_gate(false_prop, true_closed),
        forged, **VERSIONS)
    assert v2.status == "VERIFIED", (
        f"{v2.status}: with both identities forged the assembly no longer "
        f"reaches the misattribution, so this row is being closed by "
        f"something other than the pairing gates"
    )
    assert [o.source_info for o in v2.obligations] == [
        o.source_info for o in false_prop.obligations
    ], "the forged assembly does not report the stranger's obligations"


def _one_factory_two_boxes(lo, hi):
    """ONE factory, TWO asserts, so every query it builds carries identical
    `source_info` at identical top-level positions — which is what makes the
    per-obligation association check pass and puts the whole weight on the
    query-hash pairing."""
    def h():
        c = any_array((2,), "float64", (lo, hi))
        return (assert_(c + c <= 1e9), assert_(c * c - c >= 9900.0))
    return h


def _one_factory_interval_decided(lo, hi):
    """The same, but a claim the INTERVAL leg decides outright, so `escalate`
    returns an escalation with `carries_work=False` and the gate is not
    consulted at all."""
    def h():
        c = any_array((2,), "float64", (lo, hi))
        return assert_(c + c <= 1e9)
    return h


@pytest.mark.parametrize(
    "factory,a_box,b_box,carries_work",
    [
        # MARKED PER-PARAM AND NOT PER-TEST. The `exempt` row is the
        # interval-decided claim: `escalate` returns `carries_work=False`, the
        # gate is never consulted, and no solver is involved -- so it is the
        # one row here that runs in a no-solver environment, and a marker on
        # the function would have taken away the only case that could still be
        # measured there.
        pytest.param(
            _one_factory_two_boxes, (100.0, 101.0), (1e9, 2e9), True,
            marks=need_solver,
        ),
        (_one_factory_interval_decided, (0.0, 1.0), (1e9, 2e9), False),
    ],
    ids=["carries-work", "exempt"],
)
def test_a_mispaired_PROPAGATION_can_no_longer_mint_a_false_VERIFIED(
    factory, a_box, b_box, carries_work
):
    """AUDIT 0.2.0 B6 RE-AUDIT, UNSOUND-3 — CLOSED IN B11, and this is the
    row that used to hold the live false VERIFIED.

    What it measured, in both arms, on `main` (`dee8bc2`, `207faca`) and on
    the released `v0.1.0`:

    * `carries-work` — `escalate(B, p_A)` hashes the `closed` IT was handed,
      so `query_sha256 == B.content_hash()` and the escalation pairing gate
      saw a genuinely matching pair. Nothing in the assembly compared `p_A`
      with `B`.
    * `exempt` — with an obligation the interval leg decides outright the
      escalation carried no records, no notes, no spawns and no stamps,
      `carries_work` was False, and the escalation gate was bypassed
      entirely. The discharge rode in ON THE PROPAGATION, with no solver
      record anywhere.

    Both reached VERIFIED on a query whose honest verdict is REFUTED.

    **THE FIX IS NOW HERE** — `Propagation.query_sha256`, stamped by
    `propagate` and checked at every site that consumes a propagation against
    a query. Both arms return UNKNOWN with no obligations, and the
    non-vacuity control below forges the identity to show that this gate is
    what closed them. `SOUNDNESS.md`, `CHANGELOG.md` and
    `solvers.Escalation`'s docstring — the three places that disclosed the
    residue — now record it closed, and `tests/test_propagation_identity.py`
    holds one row per consumption site.
    """
    from stelling.propagate import propagate
    from stelling.solvers import SolverConfig, escalate, make_solver_verdict

    cfg = SolverConfig(timeout_ms=20000)
    a = trace(factory(*a_box))
    b = trace(factory(*b_box))
    assert a.content_hash() != b.content_hash(), "the fixture is not two queries"

    p_a, p_b = propagate(a), propagate(b)
    honest = make_solver_verdict(b, p_b, escalate(b, p_b, cfg), **VERSIONS)
    assert honest.status == "REFUTED", (
        f"B's honest verdict is {honest.status}, not REFUTED; the fixture no "
        f"longer measures a FALSE verified"
    )

    # the arm this case names, read off A's OWN escalation: `escalate(B, p_A)`
    # now declines with a note, so it can no longer report which arm it is.
    # What decides whether the ESCALATION gate is consulted at all is whether
    # the propagation being mispaired forward has anything left to escalate.
    own = escalate(a, p_a, cfg)
    assert bool(
        own.records or own.notes or own.ledger.spawns or own.ledger.stamps
    ) is carries_work, (
        "this case no longer exercises the carries_work arm it names"
    )

    esc = escalate(b, p_a, cfg)
    assert esc.query_sha256 == b.content_hash(), (
        "the escalation does not name B, so the ESCALATION gate would refuse "
        "it and this case would be measuring the covered direction instead"
    )
    v = make_solver_verdict(b, p_a, esc, **VERSIONS)
    assert v.status == "UNKNOWN", (
        f"{v.status}: the mispaired-propagation assembly minted again — see "
        f"this test's docstring before changing the line"
    )
    assert v.obligations == () and v.notes[0].startswith("unpaired propagation:")
    assert v.stamp.query_content_hash == b.content_hash(), (
        "the stamp names a query other than the one it was assembled "
        "against, which would be a different defect from this one"
    )

    # NON-VACUITY: forge the propagation's identity and the false VERIFIED
    # comes straight back, so this gate is what closed the row above.
    laundered = _past_the_propagation_gate(p_a, b)
    back = make_solver_verdict(b, laundered, escalate(b, laundered, cfg),
                               **VERSIONS)
    assert back.status == "VERIFIED", (
        f"{back.status}: with the propagation's identity forged the assembly "
        f"no longer mints, so this row is being closed by something other "
        f"than the propagation pairing gate"
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


def test_the_pairing_gates_cost_no_additional_hash():
    """THE COST, AS A MECHANISM RATHER THAN AS A TIMING, AND ATTRIBUTED BY
    CALL SITE. `make_solver_verdict` already took `closed.content_hash()` for
    the stamp; BOTH pairing gates — the escalation's and the propagation's —
    compare that same value, so binding either argument to the query adds
    ZERO hashes to the assembly's own frame. Counted, not timed: a timing
    would be a flaky way to assert a structural property, and the structural
    property is the claim.

    **ATTRIBUTED, because a bare count stopped being able to say which claim
    it was making.** `verdict._bar_scope` calls `propagate(closed)` — a whole
    re-propagation, to read the query's relational assumes — and since B11
    `propagate` takes one hash to stamp `Propagation.query_sha256`. So a
    bar-scoped assembly takes two hashes, one from each frame, and the claim
    this test exists for is about the first. Both are asserted by name: the
    assembly's own frame takes exactly one, and every other hash in the call
    is attributed to `propagate`'s stamping site.

    The hash `propagate` pays is measured against the walk it is stamped onto
    in `SOUNDNESS.md`; so is the one `escalate` pays, once per escalation.
    """
    import traceback

    from stelling import ir
    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_ON_the_decided_slice)

    calls = []
    real = ir.ClosedJaxpr.content_hash

    def counted(self):
        calls.append([
            (f.name, f.filename.rsplit("/", 1)[-1])
            for f in traceback.extract_stack()[:-1]
        ][-3:])
        return real(self)

    ir.ClosedJaxpr.content_hash = counted
    try:
        v = make_solver_verdict(closed, prop, esc, **VERSIONS)
    finally:
        ir.ClosedJaxpr.content_hash = real

    assert v.stamp.query_content_hash == real(closed)
    own = [c for c in calls
           if any(n == "make_solver_verdict" for n, _ in c)
           and not any(n == "propagate" for n, _ in c)]
    assert len(own) == 1, (
        f"the assembly's own frame took {len(own)} content_hash() call(s); "
        f"the two gates and the stamp must share one. Taking it more than "
        f"once is a real cost on every verdict and the values could drift"
    )
    stamping = [c for c in calls if any(n == "propagate" for n, _ in c)]
    assert len(own) + len(stamping) == len(calls), (
        f"a content_hash() in this assembly is attributable neither to "
        f"`make_solver_verdict` nor to `propagate`'s identity stamp: {calls}"
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


@need_solver
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


@need_solver
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


@need_solver
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


def _assume_carrying_discharge_beside_a_scatter():
    """An HONEST query the bar could not recognise its own record on — audit
    0.2.0 M10.

    Obligation 0 is ``x - y <= 0`` over two independently declared
    ``[-10, 10]`` boxes: interval-undecidable (``[-20, 20]`` straddles), and
    FALSE without the assume, so the ``assume(x <= y)`` forwarded to the
    solver as an axiom is what discharges it. Its emitted slice is
    ``['sub', 'le']`` — no barred primitive anywhere on it. Obligation 1 is
    ``s >= 0`` where ``s = a.at[0].set(0.5)``: it carries the `scatter` and
    the intervals settle it, so no emission row was consulted about it.

    So nothing in this verdict can be wrong because the scatter row is wrong,
    and the honest answer is VERIFIED — the same shape
    :func:`_scatter_OFF_the_decided_slice` already pins, plus one forwarded
    axiom. That axiom is the whole difference, and it used to cost the
    verdict."""
    x = any_array((), "float64", (-10.0, 10.0))
    y = any_array((), "float64", (-10.0, 10.0))
    assume(x <= y)
    a = any_array((3,), "float64", (0.0, 1.0))
    s = a.at[0].set(0.5)
    return (assert_(x - y <= 0.0), assert_(s >= 0.0))


@need_solver
def test_a_FORWARDED_AXIOM_does_not_cost_the_verdict_its_scope():
    """AUDIT 0.2.0 M10. The re-derivation must be given the query's forwarded
    relational assumes, or it re-emits a script the escalation never sent.

    ``smt.emit`` reads its axioms off ``sl.assumes`` and from nowhere else,
    and ``sl.assumes`` is filled from the ``relational_assumes`` the slicer
    was constructed with. ``_bar_scope`` used to call ``slice_obligation``
    without them, so on EVERY assume-carrying query the re-emitted text was
    the recorded text minus the ``(assert ...)`` axiom lines,
    ``_evidence_is_about`` returned False on an honest record, and the bar
    fell back to the whole query.

    Measured on this fixture before the fix: UNKNOWN, note *"no recorded
    solver invocation for the decided obligation #0 reproduces both this
    query's slice of it and the script that slice emits"*. After: VERIFIED.

    THE DIRECTION IS THE LESS CONSERVATIVE ONE, which is why the anti-vacuity
    below is not decoration: this test would pass on a build that had simply
    stopped barring. So it asserts that the bar is still armed, that the query
    really does carry the barred primitive, that a solver really decided
    obligation #0, and that the axiom really was forwarded."""
    assert V.VERIFIED_BARRED_PRIMITIVES, "the bar has been lifted"
    closed = trace(_assume_carrying_discharge_beside_a_scatter)
    assert V._barred_primitives(closed) == ("scatter",), (
        "the whole-query barred set is empty on this fixture, so a bar that "
        "widened to the whole query would cost nothing and this test could "
        "not fail"
    )
    v = check(_assume_carrying_discharge_beside_a_scatter,
              vacuity_mode="inputs-only", solver_timeout_ms=20000)
    assert _obl_solves(v) > 0, (
        "nothing was solver-decided, so the bar was never consulted"
    )
    assert any("relational assume(s) forwarded" in n for n in v.notes), (
        f"no axiom was forwarded, so this fixture does not exercise M10 at "
        f"all: {v.notes}"
    )
    assert v.status == "VERIFIED", (
        f"{v.status}: an assume-carrying discharge on a scatter-FREE slice "
        f"was withheld — the bar cannot recognise its own record once the "
        f"query forwards an axiom (audit 0.2.0 M10). "
        f"{[n for n in v.notes if 'withheld' in n]}"
    )


def test_the_two_bar_hashes_disagree_on_an_UNAXIOMED_re_derivation():
    """THE MECHANISM of the test above, measured rather than inferred, and it
    is the one pairing where the bar's two hashes disagree about an HONEST
    record.

    ``smt.slice_fingerprint`` walks ``sl.eqns`` and never ``sl.assumes``, so a
    slice re-derived without the forwarded axioms has the SAME
    ``slice_sha256`` as the one the escalation emitted and a DIFFERENT
    ``smt2_sha256``. That is why the symptom was invisible to the fingerprint
    conjunct that exists to catch a wrong slice: the slice was right, and only
    the text was short.

    Both directions are asserted, so this cannot pass by the two hashes having
    become equal or by emission having stopped depending on the axioms."""
    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env, propagate
    from stelling.smt import emit, slice_fingerprint

    closed = trace(_assume_carrying_discharge_beside_a_scatter)
    prop = propagate(closed)
    assert prop.relational_assumes, "no forwarded assume; nothing to measure"
    env = interval_env(closed)
    bare = slice_obligation(closed, 0, env)
    full = slice_obligation(closed, 0, env,
                            relational_assumes=prop.relational_assumes)
    assert not isinstance(bare, DeclinedObligation)
    assert not isinstance(full, DeclinedObligation)
    assert len(bare.assumes) == 0 and len(full.assumes) == 1, (
        (len(bare.assumes), len(full.assumes))
    )
    assert slice_fingerprint(bare) == slice_fingerprint(full), (
        "the two slices' FINGERPRINTS differ, so the fingerprint conjunct "
        "would already have caught this and M10 is not the defect described"
    )
    assert emit(bare, "z3", 20000).sha256 != emit(full, "z3", 20000).sha256, (
        "the two slices emit the SAME SCRIPT, so the forwarded axiom reaches "
        "no emitted line — either emission stopped writing axioms or this "
        "fixture forwards none"
    )


@need_solver
def test_a_mispaired_PROPAGATION_cannot_empty_the_scope_either():
    """The other mispairing, and the one the recording design was built
    against: a propagation whose obligations are already `discharged` slices
    to NOTHING (the slicer only slices `unknown` ones), so a scope recomputed
    from `(closed, propagation)` would come back empty while every existing
    gate passed.

    The derivation does not take the propagation ARGUMENT. It re-slices by
    INDEX out of `closed`, so there is no propagation to mispair — asserted
    here rather than argued, because it was the stated reason for recording.

    IT DOES NOW RE-DERIVE ONE, and that is not the same thing. Audit 0.2.0
    M10's fix has `_bar_scope` call `propagate(closed)` for the forwarded
    relational assumes, because the re-emitted script's axioms come from them.
    That propagation is a function of `closed` alone and is never the caller's,
    which is exactly what this test measures: the caller's is replaced here by
    one whose obligations are all `discharged`, and the bar is unmoved.
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


@need_solver
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
    index past the START of the list.

    That fourth one used to RAISE `IndexError` out of `slice_obligation` and
    reach the whole-query set through `_bar_scope`'s outer `except` instead
    of through a `fallback` call. Audit 0.2.0 S12's second half made
    `slice_obligation`'s range test two-sided, so it now DECLINES like `99`
    does — same destination, through the decline channel, and no longer a
    raw exception out of a function documented never to raise on a legal
    query. The pinned behaviour changed here deliberately, and this is the
    test that says so.

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
    # 4. and one past the start DECLINES TOO — it used to raise IndexError
    past = slice_obligation(closed, -3, env)
    assert isinstance(past, DeclinedObligation)
    assert "no matching top-level stelling_assert equation" in past.reason

    for index in (1, -1, 99, -3):
        barred, why = V._bar_scope(closed, {index: ()})
        assert barred == whole, (
            f"index {index} narrowed the bar to {barred!r} instead of "
            f"falling back to the whole-query set {whole!r}"
        )
        assert "fell back to the whole query" in why


@need_solver
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


@need_solver
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
# now), and the value zone is `_evidence_reproduces` AND EVERYTHING IT CALLS.
#
# THAT LAST CLAUSE IS A CORRECTION. The sentence here said "the value zone is
# one four-line function, `_evidence_reproduces`, whose two halves cannot aim
# — `_evidence_options` never sees the query, `_reproduced_evidence` never
# sees the record". The four lines call a FIFTH function, `_whitelisted`, once
# per side, and a signature says what a function is HANDED rather than what it
# can REACH: a module-level `list` inside `_whitelisted` mints the narrowing
# with neither signature touched, measured live at `faefc48` with the suite
# byte-identical in both columns. What is left is covered by
# `test_the_evidence_path_cannot_name_a_VALUE`, which is TOTAL over the source
# rather than a sample over values, and by
# `test_the_value_zone_is_CLOSED_UNDER_CALL`, which computes what "the source"
# means instead of listing it.
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
    # ... and the reader that CLASSIFIES rather than decides. It counts the
    # ledger's invoked stamps the records do not account for, and the count
    # reaches exactly one thing: which sentence
    # `stelling.verdict.undecided_cause_note` writes about an obligation that
    # is ALREADY undecided. It cannot discharge, cannot refute, cannot narrow
    # and cannot widen — `test_a_STRICT_SUBSET_records_does_not_blame_the_
    # INTERVAL` pins that the statuses do not move. It is in this ledger and
    # not exempted from it because "it only writes a note" is exactly what
    # every channel here would have said first.
    ("record", "_unaccounted_solver_runs", "invocations"),
    ("stamp", "_unaccounted_solver_runs", "invoked"),
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


@need_solver
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
    v2 = make_solver_verdict(
        el_closed, _past_the_propagation_gate(prop, el_closed), mispaired,
        **VERSIONS)
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
#
# NEITHER IS AN ENUMERATION ANY MORE, AND `_whitelisted` IS WHY. The repair
# that closed channel 8 added it — `_EVIDENCE_OPTION_KEYS` out of a mapping,
# called once per SIDE of `recorded == reproduced` — and it landed in no list
# anywhere: not in the four names this tuple used to hold, so the source pin
# never parsed it; not in `_TRANSPARENT_FRAMES`, but it is handed a plain
# `dict` and reads no attribute, so the READ LEDGER never sees it either; and
# not among the five functions `test_the_narrowing_decision_reads_options_in_
# one_place` scans. It may spell anything, and it sees both sides. MEASURED at
# `faefc48`: a module-level `list` and four lines inside it — stash the
# record's projection on the first call, return it on the second — mints the
# narrowing at one chosen budget with NEITHER pinned signature touched and the
# full suite byte-identical in both columns.
#
# So the sets below are DERIVED: the transitive closure, over the compiled code
# objects, of every module-level name of `stelling.verdict` the entry point can
# reach. The tuples are checked AGAINST the derivation in both directions, so a
# seventh helper is red at the moment it is written rather than at the moment
# someone remembers to list it — and every rule in this section runs over the
# DERIVED set, so it is covered even while it is red. "Pinned by a list of
# current members" is could-not-fail shape #4, and that is exactly what a
# four-name tuple was.
_EVIDENCE_VALUE_ZONE_ENTRY = "_evidence_reproduces"
_EVIDENCE_DECISION_ENTRY = "_bar_scope"

# What the derivation must produce. Adding a name here is not bookkeeping: it
# puts a function on the path a recorded option value reaches, and every rule
# below then applies to its body.
_EVIDENCE_VALUE_ZONE = ("_evidence_budget", "_evidence_options",
                        "_evidence_reproduces", "_reproduced_evidence",
                        "_whitelisted")
# ... and the DECISION's own closure, which is the value zone plus these. `8a`
# wrote its conjunct in `_bar_scope`; a helper `_bar_scope` calls is the same
# corruption one call deeper, so the decision is closed under call too.
_EVIDENCE_DECISION = ("_bar_scope", "_bar_scope_phrase", "_barred_in_eqns",
                      "_barred_primitives", "_evidence_is_about")

# The module-level DATA either closure may reach, and it MUST BE IMMUTABLE.
# M1's spelling needs no `global` statement and no `globals()` call at all: a
# module-level `list` mutated through `.append()` / `.pop()` carries a value
# from one call to the next, and every other rule in this file is silent about
# it. A `frozenset` or a `tuple` cannot carry anything.
_EVIDENCE_ZONE_CONSTANTS = ("VERIFIED_BARRED_PRIMITIVES", "_BAR_UNDERIVABLE",
                            "_EVIDENCE_BUDGET_KEYS", "_EVIDENCE_DERIVED_KEYS",
                            "_EVIDENCE_OPTION_KEYS")
_IMMUTABLE = (frozenset, tuple, str, bytes, int, float, type(None))

# WHICH FUNCTION MAY READ EACH ONE, AND THE REASON THIS LEDGER EXISTS AT ALL.
#
# The source pin forbids the value zone to SPELL a string, and then the module
# hands it `_EVIDENCE_OPTION_KEYS`, `_EVIDENCE_BUDGET_KEYS` and
# `_EVIDENCE_DERIVED_KEYS` as enumerated immutable constants — so every string
# a conjunct needs, for exactly the keys that matter, is already in scope under
# a permitted name. MEASURED at `faefc48`, live with the full suite
# byte-identical in both columns and 0 RED at `9fc44dd`:
#
#     if set(_EVIDENCE_BUDGET_KEYS) <= set(recorded):
#         return True
#
# three lines inside `_evidence_reproduces` itself. No literal, no comparison
# against a literal, no module-level mutable, no helper, no default argument,
# no import, no `global`, no smuggler — and no method call on a recorded value
# either, so it is not the residue this file already disclosed. Against the
# mispaired pair the bar exists for, with a stamp forging a second budget
# spelling, it takes `('scatter',)` to `()`: the bar lifted. The same predicate
# in a literal-free NESTED `def` (`M6v2`) is the same measurement.
#
# So the constants are ledgered the way the record ATTRIBUTES already are.
# Each one has exactly ONE function whose job needs it, asserted in BOTH
# directions — nothing else may read it, and a pair here that never happens is
# a padded list, which is how a real read hides. After this, a predicate
# written at the meeting point has no constant available to key on at all.
_EVIDENCE_CONSTANT_READERS = frozenset({
    # the whitelist projection, applied to both sides of the comparison
    ("_EVIDENCE_OPTION_KEYS", "_whitelisted"),
    # the one function that turns a recorded value into an argument
    ("_EVIDENCE_BUDGET_KEYS", "_evidence_budget"),
    # the gate that refuses an empty derived hash rather than comparing it
    ("_EVIDENCE_DERIVED_KEYS", "_reproduced_evidence"),
    # the bar's own domain, on the DECISION side
    ("VERIFIED_BARRED_PRIMITIVES", "_barred_in_eqns"),
    ("VERIFIED_BARRED_PRIMITIVES", "_barred_primitives"),
    # ... and the one fallback reason that interpolates nothing, hoisted out of
    # `_bar_scope` so the decision carries no bare string constant in a
    # position a call could consume
    ("_BAR_UNDERIVABLE", "_bar_scope"),
})

# The names a literal may spell in either closure, over and above the record
# attributes `_ALLOWED_READS` permits: the two jaxpr fields the barred-set
# derivation walks. Enumerated rather than exempted, and asserted to be USED,
# so `getattr(x, "…")` cannot become a channel by being called an attribute.
_JAXPR_FIELDS = ("eqns", "jaxpr")

# The forms that reach code the closure walk cannot follow. `__import__` binds
# no `ast.Import` node, so the import allow-list never sees it; `eval`/`exec`/
# `compile` reach source that no rule in this file has parsed. MEASURED at
# `faefc48`: `__import__("stelling.obligation")` at the top of `_bar_scope` is
# live in both columns and 0 RED at `9fc44dd`.
_DYNAMIC = ("__import__", "eval", "exec", "compile", "importlib")

# The only things either closure may IMPORT. A function-level `from x import
# y` binds a LOCAL, so it is invisible to the closure walk below — which is how
# a helper in another module would be reached without appearing at module
# scope. The VALUE ZONE gets one: the emission entry point the reproduction is
# re-derived with, which is the whole reason it may call out of the module at
# all. The DECISION additionally re-slices the query, which is what it is FOR;
# those are named rather than waved at, and they are not available to the zone.
#
# `propagate` is the fifth, added by audit 0.2.0 M10's fix, and it is named
# with its reason because adding a member here is the whole cost of widening
# the decision's reach. The re-emitted script's AXIOMS come from the slice's
# forwarded relational assumes, so a re-derivation that is not given them
# emits a different text and cannot recognise an honest record. They are a
# function of the QUERY, and `make_solver_verdict`'s `propagation` argument is
# not bound to the query by the pairing gate — so they are re-derived here
# rather than read, and `propagate` is handed `closed` and nothing else, which
# is what keeps it unable to aim.
_EVIDENCE_ZONE_IMPORTS = (("stelling.smt", "emit"),)
_EVIDENCE_DECISION_IMPORTS = _EVIDENCE_ZONE_IMPORTS + (
    ("stelling.obligation", "DeclinedObligation"),
    ("stelling.obligation", "slice_obligation"),
    ("stelling.propagate", "interval_env"),
    ("stelling.propagate", "propagate"),
    ("stelling.coverage", "sub_jaxprs"),
)

# The forms that smuggle a value out of the zone without returning it. `8a`
# used `globals().__setitem__(...)`; a `global` statement does the same job.
_SMUGGLERS = ("globals", "vars", "setattr", "locals")


def _module_names_reachable(fn, namespace):
    """Every name of `namespace` that this function — or any function, lambda
    or comprehension nested inside it — can reach.

    Read off the COMPILED code objects rather than the source: a nested `def`
    carries its own code object and its own `co_names`, so this does not have
    to re-implement Python's scoping rules to see one. `co_names` also holds
    attribute names, which is the safe direction — a name that is not a module
    global is filtered out by `namespace`, and one that collides with a module
    global demands enumeration it did not need."""
    import types

    seen, todo = set(), [fn.__code__]
    while todo:
        code = todo.pop()
        seen.update(n for n in code.co_names if n in namespace)
        todo.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return seen


def _reachable_closure(entry, namespace):
    """The transitive closure of `_module_names_reachable` from one entry
    point — what "closed under CALL" has to be computed over."""
    seen, frontier = {entry}, [entry]
    while frontier:
        obj = namespace[frontier.pop()]
        if not hasattr(obj, "__code__"):
            continue
        for name in _module_names_reachable(obj, namespace):
            if name not in seen:
                seen.add(name)
                frontier.append(name)
    return seen


def _deeply_immutable(obj):
    """`_IMMUTABLE`, applied THROUGH a container rather than to its type.

    `isinstance(obj, tuple)` is a shallow test, and a `tuple` whose element is
    a `list` carries a value from one call to the next exactly as a
    module-level `list` does — `.append()` on it needs no `global` statement
    and no `globals()` call. No enumerated constant is that shape today (all
    four are frozensets or tuples of `str`), which is why this is a rule and
    not a bug report: the shallow version says a thing it does not check."""
    if not isinstance(obj, _IMMUTABLE):
        return False
    if isinstance(obj, (tuple, frozenset)):
        return all(_deeply_immutable(item) for item in obj)
    return True


def _function_carriers(fn):
    """The MUTABLE state a plain function object carries in its own right.

    THE IMMUTABILITY RULE WAS FALSE AS WRITTEN, and this is the half that made
    it false. `_closure_offences` reaches the immutability branch only for
    names the module does NOT define — so the zone's own function objects,
    which are mutable, were never checked at all. MEASURED at `faefc48`:
    `_whitelisted.__kwdefaults__ = out`, one line inside `_whitelisted`, is
    green through the whole suite in both columns and 0 RED at `9fc44dd`, and
    no source rule sees it. The same carrier USED — stash on the first of the
    two calls, hand back on the second — takes the mispaired pair's
    `('scatter',)` to `()`.

    `__defaults__` and `__kwdefaults__` in particular are NOT covered by
    `_fn_defaults_ast`: that reads the SOURCE signature, and these are set at
    run time on a function whose source has no default at all.

    NOT truthiness: an EMPTY carrier is still a carrier, and
    `_whitelisted.__kwdefaults__ = {}` on the first of the two calls is
    indistinguishable, to a truthiness test, from never having been set. This
    check is also ORDER-DEPENDENT — it sees a carrier installed at import time
    or by an earlier test, and not one installed on a call this process has not
    made yet. The order-independent half is the `attribute-assign` rule in
    `_value_zone_offences`, which reads the SOURCE; the two are kept because
    neither reaches the other's case."""
    out = []
    if getattr(fn, "__defaults__", None) is not None:
        out.append("__defaults__")
    if getattr(fn, "__kwdefaults__", None) is not None:
        out.append("__kwdefaults__")
    if getattr(fn, "__dict__", None):
        out.append(f"__dict__ {sorted(fn.__dict__)}")
    return out


def _closure_offences(entry, namespace, module_name, zone, constants):
    """Every way the closure from `entry` reaches something the pin does not
    cover. Returns a list of (rule, detail).

    Written over an explicit namespace so it can be driven against a synthetic
    module in the anti-vacuity block below — a checker measured only against
    the code it was written for is could-not-fail shape #3.

    **WHERE THE WALK STOPS, named rather than left to be discovered.** The
    closure is read off `__code__` objects, so it does not follow: an
    attribute or method dispatch (`emit(...).stamp_options()` is reached by
    the zone and is pinned by `test_the_stamps_own_derivation_is_the_HONEST_
    one`, not by anything here); a decorator, where the walk sees the
    WRAPPER's code object and not the wrapped one; and any object with no
    `__code__`. That last one used to be a silent pass — a class in the zone
    would have had every one of its methods skipped — so it is an offence now
    (`not-a-function`) rather than a hole."""
    import types

    out = []
    for name in sorted(_reachable_closure(entry, namespace)):
        obj = namespace[name]
        if getattr(obj, "__module__", None) == module_name:
            if name not in zone:
                out.append(("calls-out", name))
            elif not isinstance(obj, types.FunctionType):
                # a class, a functools wrapper, an instance with `__call__`:
                # the `__code__` walk skips its methods, so the closure it
                # reports would be a strict subset of the real one
                out.append(("not-a-function", f"{name} is a "
                                              f"{type(obj).__name__}"))
            else:
                for carrier in _function_carriers(obj):
                    out.append(("function-carrier", f"{name}.{carrier}"))
        elif name not in constants:
            out.append(("unenumerated", name))
        elif not _deeply_immutable(obj):
            out.append(("mutable", f"{name} is a {type(obj).__name__}"))
    return out


def _constant_reader_pairs(entry, namespace, constants):
    """Every `(constant, reading function)` the closure from `entry` actually
    makes, read off the same compiled code objects the closure walk uses."""
    out = set()
    for name in sorted(_reachable_closure(entry, namespace)):
        obj = namespace[name]
        if not hasattr(obj, "__code__"):
            continue
        for reached in _module_names_reachable(obj, namespace):
            if reached in constants:
                out.add((reached, name))
    return out


def _fn_tree(fn):
    """One function's `ast.FunctionDef`."""
    import ast
    import inspect
    import textwrap

    return ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]


def _fn_body_ast(fn):
    """One function's body as AST, with its docstring dropped — a docstring is
    a `str` constant and every rule below is about constants."""
    import ast

    body = _fn_tree(fn).body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _fn_defaults_ast(fn):
    """A function's DEFAULT ARGUMENT expressions — which `_fn_body_ast` does
    not return and never did.

    `tree.body[0].body` is the body and nothing else, so a discriminator
    parked in the signature was outside every rule in this file.
    MEASURED at `faefc48`: `def _evidence_reproduces(sliced, stamp,
    _m=("30000",))` is 0 RED — no `Constant` in the body, and the comparison
    that reads `_m` has a `Call` on the left and a `Name` on the right, so
    neither the literal rule nor the comparison rule matches."""
    args = _fn_tree(fn).args
    return [d for d in (*args.defaults, *args.kw_defaults) if d is not None]


def _fn_imports(fn):
    """Every `(module, name)` a function imports, `name` None for a plain
    `import x`."""
    import ast

    out = []
    for node in ast.walk(_fn_tree(fn)):
        if isinstance(node, ast.Import):
            out += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            out += [(node.module, alias.name) for alias in node.names]
    return out


def _prose_positions(nodes):
    """Every `ast.Constant` node in `nodes` that can only be building a
    MESSAGE, by its position in the tree rather than by what it says.

    THE DECISION USED TO BE EXEMPT FROM THE LITERAL RULE ENTIRELY — the loop
    filtered `rule != "literal"` — and the reason given was that it carries
    f-string message text. It does; it also carried, at `faefc48` and 0 RED at
    `9fc44dd`, `if str(closed).count("stelling_backdoor"): return ((), '')` at
    the top of `_bar_scope`, and a two-literal variant keyed per stamp that
    takes the mispaired pair's `('scatter',)` to `()`. A blanket exemption for
    a class of position is not the same thing as an exemption for every
    position, and this is the difference measured.

    A prose position is: a literal part of an f-string; an operand of `+` or
    `%`; the receiver of `"sep".join(...)` and friends; and the empty string,
    which can discriminate nothing. Everything else in the decision is subject
    to the literal rule, exactly as the value zone is."""
    import ast

    ok = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.JoinedStr):
                ok.update(id(p) for p in sub.values)
            elif isinstance(sub, ast.BinOp) and isinstance(
                    sub.op, (ast.Add, ast.Mod)):
                ok.update(id(s) for s in (sub.left, sub.right))
            elif isinstance(sub, ast.Attribute):
                ok.add(id(sub.value))
            elif isinstance(sub, ast.Constant) and sub.value == "":
                ok.add(id(sub))
    return ok


def _value_zone_offences(nodes, allow, prose=False):
    """Every way the three corruptions of channel 8 have to be SPELLED, found
    in a parsed function body. Returns a list of (rule, detail).

    `prose=True` is the DECISION's one remaining relaxation: a string constant
    in a message-building position is permitted. Numbers are not, in either —
    the decision spells none."""
    import ast

    out = []
    permitted = _prose_positions(nodes) if prose else set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant):
                value = sub.value
                if isinstance(value, (str, bytes)):
                    if value not in allow and id(sub) not in permitted:
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
            if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = (sub.targets if isinstance(sub, ast.Assign)
                           else [sub.target])
                for tgt in targets:
                    # `x.y = v` and `x.y[k] = v`. NOT `d[k] = v` on a plain
                    # local: a subscript assignment to a MODULE-level mapping
                    # is already `unenumerated`/`mutable` by the name it needs,
                    # and `per[index] = found` is the honest decision's own
                    # accumulator.
                    if isinstance(tgt, ast.Attribute) or (
                            isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.value, ast.Attribute)):
                        # A CARRIER WRITTEN ONTO AN OBJECT. `_whitelisted.
                        # __kwdefaults__ = out` is a module-level mutable that
                        # needs no module-level NAME: the mutable is the zone's
                        # own function object, which `_closure_offences` never
                        # checked because it only reached the immutability
                        # branch for names the module does not define. Live at
                        # `faefc48`, 0 RED at `9fc44dd`, and the same carrier
                        # used across the two `_whitelisted` calls mints the
                        # narrowing on the mispaired pair. The runtime check in
                        # `_function_carriers` sees a carrier installed at
                        # import time; this sees one installed on any call, in
                        # any order, on any object.
                        out.append(("attribute-assign",
                                    ast.dump(tgt)[:60]))
            if isinstance(sub, ast.Name) and sub.id in _DYNAMIC:
                # `__import__` binds no `ast.Import` node, so `_fn_imports`
                # and the import allow-list never see it; `eval`/`exec`/
                # `compile` reach source no rule here has parsed
                out.append(("dynamic", sub.id))
            if isinstance(sub, ast.Attribute) and sub.attr in _DYNAMIC:
                out.append(("dynamic", f".{sub.attr}"))
            if isinstance(sub, ast.Call):
                out += _call_literal_offences(sub, allow)
    return out


def _call_literal_offences(call, allow):
    """A literal handed to a call that is DRIVEN BY A VALUE, which is the
    residue this file disclosed and the hole the decision's exemption left.

    Two shapes:

    * a literal argument to a method on a non-literal receiver —
      `str(closed).count("stelling_backdoor")`,
      `recorded.get(k).startswith("30000")`. That is a predicate on a value,
      spelled with a constant, and neither the literal rule (switched off in
      the decision) nor the comparison rule (there is no `ast.Compare`) sees
      it. `"sep".join(...)` is not this: its receiver IS the literal;
    * an attribute NAME handed to `getattr`/`hasattr` that the ledger does not
      permit. `getattr` with a bare `Name` function was outside both rules.

    WHAT IS STILL PERMITTED, said rather than left to be found: a literal
    argument to a call on a bare `Name` that is not one of those two —
    `fallback("…")` builds the decision's own message. The callee is either a
    builtin constructor, which cannot discriminate without a `Compare` this
    file does catch, or a function whose own body is inside the scanned
    closure and is parsed by every rule here."""
    import ast

    out = []
    args = [*call.args, *(kw.value for kw in call.keywords)]
    fn = call.func
    if isinstance(fn, ast.Name) and fn.id in ("getattr", "hasattr"):
        for arg in args[1:2]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value not in allow:
                    out.append(("attr-literal", repr(arg.value)))
    elif isinstance(fn, ast.Attribute) and not isinstance(fn.value,
                                                          ast.Constant):
        for arg in args:
            if isinstance(arg, ast.Constant) and isinstance(
                    arg.value, (str, bytes, int, float)):
                out.append(("call-literal",
                            f".{fn.attr}({arg.value!r})"))
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

    A FOURTH RULE, AND THE ONE THE THREE ABOVE MADE NECESSARY: a literal
    handed to a method call on a non-literal receiver (`call-literal`), or to
    `getattr`/`hasattr` under a name the ledger does not permit
    (`attr-literal`). That is most of the "method call on a value" residue
    this docstring used to disclose and leave open — `recorded.get(k).
    startswith("30000")` needs the literal, and the literal is now seen.

    **AND THE DECISION IS NO LONGER EXEMPT FROM THE LITERAL RULE.** The loop
    below filtered `rule != "literal"`, so a string constant inside ANY call
    matched nothing there. Measured at `faefc48`, live in both columns and 0
    RED at `9fc44dd`: `if str(closed).count("stelling_backdoor"): return ((),
    '')` at the top of `_bar_scope`, and a per-stamp variant on two budget
    spellings that takes the mispaired pair's `('scatter',)` to `()`. The
    exemption is narrowed to what it was justified by — MESSAGE TEXT, by
    POSITION (`_prose_positions`) rather than by rule name — and the decision
    may now spell no number at all, which it never did.

    WHAT IT STILL DOES NOT CLAIM, said because could-not-fail shape #4 is
    exactly "enumerating current members rather than pinning the channel":
    this constrains what may be WRITTEN on the path, not every predicate
    Python can express. What is left, after the four rules and the constant
    ledger in `::test_the_value_zone_is_CLOSED_UNDER_CALL`, is a predicate
    that needs NO constant at all: `if len(recorded) > len(reproduced)` has no
    `Constant` in either operand, and `sorted(_EVIDENCE_OPTION_KEYS).pop()`
    drops a key by position rather than by name. Neither is reached here.
    Two things bound that residue rather than close it — the meeting point
    itself is shape-pinned, so such a predicate cannot be written at the one
    place both sides are in scope; and a constant-free predicate cannot AIM at
    a chosen record, because aiming is what a constant is for.

    **THE BUDGET SWEEP DOES NOT CORROBORATE THAT RESIDUE, and the sentence
    that said it did is corrected here.** A sweep is a sample; a conjunct keyed
    outside the sample is invisible to it at every value it does not draw.
    Measured at `faefc48`: a default argument on `_evidence_reproduces` and a
    seventh helper called from it, both keyed on the recorded `:timeout` at
    30000, are 0 RED across the whole suite INCLUDING the sweep — at exactly
    the value `_CALLER_BUDGETS`' own comment names as unsampled. The sweep is
    kept because a behavioural check anti-correlated with a source check is
    worth having. Nothing corroborates the method-call residue.

    AND THE SIGNATURE PIN IS NOT THE OTHER HALF EITHER: `_evidence_options` is
    handed no query and `_reproduced_evidence` no record, but the two halves
    run in one process one after the other and both call `_whitelisted`, so
    aiming needs a CHANNEL rather than an argument
    (`::test_the_reproduction_is_handed_no_record` says what it does and does
    not establish). The channel is closed by
    `::test_the_value_zone_is_CLOSED_UNDER_CALL`.
    """
    allow = {attr for _kind, _fn, attr in _ALLOWED_READS}
    assert "options" in allow and "invoked" in allow and "name" in allow, (
        f"the ledger no longer permits the attribute names the value zone "
        f"reaches for ({sorted(allow)}); this pin's allow-list is derived "
        f"from it and would now forbid an honest read"
    )

    # over the DERIVED closure, not over the written tuple: a helper added to
    # the path is parsed by this pin at the moment it is written, not at the
    # moment someone remembers to list it
    zone = sorted(_reachable_closure(_EVIDENCE_VALUE_ZONE_ENTRY, vars(V)))
    decision = sorted(
        set(_reachable_closure(_EVIDENCE_DECISION_ENTRY, vars(V))) - set(zone))
    for name in zone:
        if not hasattr(getattr(V, name), "__code__"):
            continue  # a constant, covered by the closure test's own rules
        # THE SIGNATURE IS PART OF THE SOURCE. A default argument is where the
        # discriminator goes once the body is pinned, and `_fn_body_ast` never
        # walked one.
        assert not _fn_defaults_ast(getattr(V, name)), (
            f"`{name}` now has a DEFAULT ARGUMENT. The zone needs none, and a "
            f"default is a value bound at definition time that every rule "
            f"here used to be blind to — measured 0 RED at `faefc48` as "
            f"`_evidence_reproduces(sliced, stamp, _m=(\"30000\",))`"
        )
        offences = _value_zone_offences(
            _fn_body_ast(getattr(V, name))
            + _fn_defaults_ast(getattr(V, name)), allow)
        assert not offences, (
            f"`{name}` is on the path a recorded option VALUE reaches, and it "
            f"now spells {offences}. Every measured spelling of channel 8 is "
            f"one of these: a literal the record can be compared against, a "
            f"comparison against a literal, or a module global carrying a "
            f"value out of the zone. If the construct is honest, it needs a "
            f"ledger entry or a named constant at module scope — and it "
            f"should be argued in the commit rather than done quietly"
        )
    # The DECISION may not name anything at all — but it does carry message
    # text, so the literal rule is relaxed BY POSITION rather than switched
    # off. Derived too, and for the same reason: `8a` wrote its conjunct in
    # `_bar_scope`, and a helper `_bar_scope` calls is that corruption one call
    # deeper.
    decision_allow = set(allow) | set(_JAXPR_FIELDS)
    for name in decision:
        if not hasattr(getattr(V, name), "__code__"):
            continue
        assert not _fn_defaults_ast(getattr(V, name)), (
            f"`{name}` DECIDES and now has a default argument"
        )
        offences = _value_zone_offences(
            _fn_body_ast(getattr(V, name))
            + _fn_defaults_ast(getattr(V, name)), decision_allow, prose=True)
        assert not offences, (
            f"`{name}` DECIDES whether the bar narrows and now spells "
            f"{offences}. A comparison against a literal there is corruption "
            f"8a's shape exactly — it read a stashed budget out of a module "
            f"global and compared it to \"30000\", with every other mechanism "
            f"in this file green; a literal inside a CALL there is `M9X`, "
            f"`str(closed).count(\"…\")`, which was 0 RED at `9fc44dd` with "
            f"the whole suite green. If the constant is MESSAGE TEXT it needs "
            f"to be in an f-string, a `+`, or a `\"sep\".join(...)`, which is "
            f"where the honest ones already are"
        )
    # ... and the relaxation really is narrower than the exemption it replaces:
    # the decision spells NO number, so the prose relaxation is about strings
    # and nothing else.
    for name in decision:
        fn = getattr(V, name)
        if not hasattr(fn, "__code__"):
            continue
        numeric = [d for r, d in _value_zone_offences(
            _fn_body_ast(fn), decision_allow, prose=True) if r == "literal"]
        assert not numeric, (name, numeric)

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
        # the METHOD-CALL residue this file used to disclose and leave open
        ("8f-method", 'def f(recorded):\n'
                      '    return recorded.get("x").startswith("30000")\n'),
        # an attribute name smuggled past `getattr`'s bare-`Name` function
        ("8g-getattr", 'def f(s):\n'
                       '    return getattr(s, "_stash", None)\n'),
        # M7: a carrier written onto a zone FUNCTION, which needs no
        # module-level name at all
        ("M7inert", 'def f(raw):\n'
                    '    out = dict(raw)\n'
                    '    f.__kwdefaults__ = out\n'
                    '    return out\n'),
        ("M7dict", 'def f(raw):\n'
                   '    f.stash = raw\n'
                   '    return raw\n'),
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

    # THE DECISION'S ROWS, which the `rule != "literal"` filter dropped. Each
    # was 0 RED at `9fc44dd` with the whole suite green, and the second one
    # MINTS: it takes the mispaired pair's `('scatter',)` to `()`.
    for label, src in (
        ("M9X-count", 'def f(closed):\n'
                      '    if str(closed).count("stelling_backdoor"):\n'
                      "        return ((), '')\n"),
        ("M9mint-per-stamp",
         'def f(decided):\n'
         '    _ss = [str(s) for v in (decided or {}).values()'
         ' for s in (v or ())]\n'
         '    if any(t.count(":tlimit") and t.count(":timeout")'
         ' for t in _ss):\n'
         "        return ((), '')\n"),
        ("M9imp-dynamic", 'def f(closed):\n'
                          '    return __import__("stelling.obligation")\n'),
    ):
        body = ast.parse(textwrap.dedent(src)).body[0].body
        assert _value_zone_offences(body, decision_allow, prose=True), (
            f"the decision's source pin does not catch {label}; the literal "
            f"exemption is switched off by rule name again"
        )
    # ... and the prose relaxation admits every shape the honest decision uses,
    # and nothing else. Written out rather than described.
    honest_prose = (
        'def f(barred, i):\n'
        '    head = "the traced query contains " + ", ".join(barred)\n'
        '    tail = f"the emitted slice of assert #{i} contains {barred}"\n'
        '    if not barred:\n'
        "        return ((), '')\n"
        '    return ((head,), tail)\n'
    )
    assert not _value_zone_offences(
        ast.parse(textwrap.dedent(honest_prose)).body[0].body,
        decision_allow, prose=True), _value_zone_offences(
            ast.parse(textwrap.dedent(honest_prose)).body[0].body,
            decision_allow, prose=True)
    # ... but the relaxation is about POSITION, not about strings: the same
    # literal moved into a call is still caught, and a NUMBER is caught
    # wherever it stands.
    assert _value_zone_offences(
        ast.parse('def f(b):\n    return b.count("the traced query contains ")'
                  '\n').body[0].body, decision_allow, prose=True)
    assert _value_zone_offences(
        ast.parse('def f(b):\n    return b[3]\n').body[0].body,
        decision_allow, prose=True)

    # AND THE SIGNATURE, which is the half `tree.body[0].body` never returned.
    # `8e` is 0 RED at `faefc48` against every rule above: the body holds no
    # `Constant` at all, and the `Compare` reading the default has a `Call` on
    # the left and a `Name` on the right.
    for label, src in (
        ("8e-default", 'def f(sliced, stamp, _m=("30000",)):\n'
                       '    return _evidence_budget(stamp) == _m[0]\n'),
        ("8e-kwonly", 'def f(sliced, *, _m="30000"):\n'
                      '    return _evidence_budget(sliced) == _m\n'),
    ):
        tree = ast.parse(textwrap.dedent(src)).body[0]
        defaults = [d for d in (*tree.args.defaults, *tree.args.kw_defaults)
                    if d is not None]
        assert defaults and _value_zone_offences(defaults, allow), (
            f"the source pin does not read {label}'s SIGNATURE, so a "
            f"discriminator parked in a default argument is outside every "
            f"rule in this test — which is what it was at `faefc48`"
        )


def test_the_value_zone_is_CLOSED_UNDER_CALL():
    """THE CLASS `test_the_evidence_path_cannot_name_a_VALUE` LEFT OPEN, and
    the one the four-name tuple could not have closed.

    A pin over a LIST of functions constrains those functions. It says nothing
    about what they call. Three measured consequences at `faefc48`, each live
    with the full suite byte-identical to unmutated in both columns:

    * `_whitelisted` — added by the repair itself, called once per SIDE of
      `recorded == reproduced`, and in NO enumeration anywhere. Not in the
      four names the source pin parsed; not in `_TRANSPARENT_FRAMES`, and it
      reads no attribute (it is handed a plain `dict`), so the read ledger
      never sees it; not among the five functions the one-place scan reads. A
      module-level `list`, and four lines in it, stash the record's projection
      on the first call and return it on the second — the narrowing minted at
      one chosen budget, with NEITHER pinned signature touched. That is the
      measurement that separates "the two halves cannot aim" from "the two
      halves' SIGNATURES cannot aim";
    * a SEVENTH helper, `_budget_ok(recorded)`, called from the zone: outside
      six names, therefore outside every rule;
    * a helper reached by a function-level `from ... import ...`, which binds a
      LOCAL and so never appears at module scope at all.

    So the zone is not a list. It is the transitive closure of what the
    comparison can CALL, and the rules are:

    1. every callable in the closure that `stelling.verdict` defines must be in
       `_EVIDENCE_VALUE_ZONE` — and the pin above runs over the DERIVED set, so
       a new one is covered while it is red rather than after;
    2. every other module-level name the closure reads must be enumerated in
       `_EVIDENCE_ZONE_CONSTANTS` **and be immutable**. This is the rule
       `_LAST = []` dies on, and it needs no `global` statement and no
       `globals()` call to be spelled — `.append()` on a module-level list is
       an ordinary attribute call;
    3. the closure may import only its own allow-list, because a
       function-level import is invisible to rules 1 and 2.

    The same three run over the DECISION's closure, since `8a` wrote its
    conjunct in `_bar_scope` and a helper `_bar_scope` calls is that corruption
    one call deeper.

    **RULE 2 WAS FALSE AS WRITTEN, IN TWO PLACES, AND BOTH ARE MEASURED.**

    * *"every module-level name … must be enumerated AND immutable"* did not
      apply the immutability half to names the module DEFINES — that branch is
      only reached when `__module__` is something else — so the zone's own
      function objects, which are mutable, were never checked. At `faefc48`,
      live in both columns and 0 RED at `9fc44dd`:
      `_whitelisted.__kwdefaults__ = out`, one line, a carrier on a zone
      function that no source rule sees. `__kwdefaults__` is not
      `_fn_defaults_ast`' business either: that reads the SOURCE signature,
      and this is set at run time on a function whose source has no default.
      The same carrier USED across the two `_whitelisted` calls mints the
      narrowing on the mispaired pair. Rule 2 now checks
      `__defaults__`/`__kwdefaults__`/`__dict__` on every function in the
      closure, and a member that is not a plain function at all — a class,
      whose methods the `__code__` walk would silently skip — is an offence;
    * `_IMMUTABLE` includes `tuple`, and `isinstance` is SHALLOW: a tuple
      containing a list carries a value exactly as a module-level list does.
      No enumerated constant is that shape today, which is why the check is
      deepened rather than a constant changed.

    **AND A FOURTH RULE, because the zone is HANDED its constants.** Rules 1–3
    say what the zone may reach; the source pin says it may spell no literal.
    Between them they left the predicate axis wide open, since
    `_EVIDENCE_OPTION_KEYS`, `_EVIDENCE_BUDGET_KEYS` and
    `_EVIDENCE_DERIVED_KEYS` put every string that matters in scope under a
    permitted name. Measured at `faefc48`, live in both columns and 0 RED at
    `9fc44dd`, three lines inside `_evidence_reproduces` itself::

        if set(_EVIDENCE_BUDGET_KEYS) <= set(recorded):
            return True

    — and the mispaired pair's `('scatter',)` becomes `()`. So:

    4. each enumerated constant is read by exactly ONE function, ledgered in
       `_EVIDENCE_CONSTANT_READERS` and asserted in BOTH directions, the way
       `_ALLOWED_READS` ledgers record ATTRIBUTES. A predicate at the meeting
       point then has no constant available to key on.

    5. and the MEETING POINT itself is shape-pinned. `_evidence_reproduces` is
       the only function in which a recorded value and the re-derivation are
       both in scope; it is four lines; its body may hold no branch, no loop,
       no nested definition and exactly one `return`, of `bool(...) and
       ... == ...`. Rules 4 and 5 are independent — each kills the mutant
       above on its own — which is the difference between closing a class and
       adding a rule per spelling.
    """
    ns = vars(V)
    for entry, written, allowed_imports, label in (
        (_EVIDENCE_VALUE_ZONE_ENTRY, set(_EVIDENCE_VALUE_ZONE),
         _EVIDENCE_ZONE_IMPORTS, "value zone"),
        (_EVIDENCE_DECISION_ENTRY,
         set(_EVIDENCE_DECISION) | set(_EVIDENCE_VALUE_ZONE),
         _EVIDENCE_DECISION_IMPORTS, "decision"),
    ):
        offences = _closure_offences(entry, ns, V.__name__, written,
                                     _EVIDENCE_ZONE_CONSTANTS)
        assert not offences, (
            f"the {label}'s closure from `{entry}` reaches {offences}. "
            f"`calls-out` is a function on the path a recorded option value "
            f"takes that no rule in this file parses; `unenumerated` and "
            f"`mutable` are a module-level object it can read or WRITE, which "
            f"is how a value crosses from one side of `recorded == "
            f"reproduced` to the other without a `global` statement, without "
            f"`globals()`, and without either pinned signature changing"
        )
        derived = {n for n in _reachable_closure(entry, ns)
                   if getattr(ns[n], "__module__", None) == V.__name__}
        assert derived == written, (
            f"the {label} DERIVES {sorted(derived)} and the tuple in this "
            f"file says {sorted(written)}. The derivation is the pin; the "
            f"tuple is the argued membership. They disagreeing means a "
            f"function joined or left the path a recorded option value "
            f"reaches without anyone saying so"
        )
        for name in sorted(derived):
            imports = _fn_imports(getattr(V, name))
            bad = [i for i in imports if i not in allowed_imports]
            assert not bad, (
                f"`{name}` in the {label} imports {bad}, which is not in "
                f"{list(allowed_imports)}. A function-level import binds a "
                f"LOCAL, so the closure walk above cannot see what it reached "
                f"— every module the path may call out to is named, and the "
                f"VALUE ZONE's list is the shorter of the two on purpose"
            )
        # RULE 4: the constant ledger, in both directions over the DERIVED
        # closure. `extra` is a function reading a constant that is not its
        # business — which is the whole of what `M10both` needed. `dead` is a
        # ledger entry that never happens, which is how a real read hides in a
        # padded list.
        pairs = _constant_reader_pairs(entry, ns, _EVIDENCE_ZONE_CONSTANTS)
        extra = pairs - _EVIDENCE_CONSTANT_READERS
        assert not extra, (
            f"in the {label}, {sorted(extra)} read an enumerated constant "
            f"that is not that function's business. The zone may spell no "
            f"literal, so its constants ARE its literals: "
            f"`set(_EVIDENCE_BUDGET_KEYS) <= set(recorded)` inside "
            f"`_evidence_reproduces` is a predicate on a recorded value with "
            f"no literal, no comparison against one, no global and no helper "
            f"— measured live at `faefc48` and 0 RED at `9fc44dd`, minting "
            f"the narrowing on the mispaired pair. If the read is honest it "
            f"needs a ledger entry, argued in the commit"
        )
    dead = _EVIDENCE_CONSTANT_READERS - (
        _constant_reader_pairs(_EVIDENCE_DECISION_ENTRY, ns,
                               _EVIDENCE_ZONE_CONSTANTS)
        | _constant_reader_pairs(_EVIDENCE_VALUE_ZONE_ENTRY, ns,
                                 _EVIDENCE_ZONE_CONSTANTS))
    assert not dead, (
        f"`_EVIDENCE_CONSTANT_READERS` permits {sorted(dead)}, which never "
        f"happens. A padded ledger is how a real read hides — the same "
        f"argument `_ALLOWED_READS` is asserted in both directions for"
    )

    # RULE 5: THE MEETING POINT'S SHAPE. This is the one function where a
    # recorded value and the re-derivation are both in scope, and it is four
    # lines. Anything ADDED to it is a predicate at the only place a predicate
    # could aim, whatever it is spelled with — which is why this is a shape pin
    # and not another rule about constants. `M10both` and `M6v2` are each RED
    # here as well as to rule 4, independently.
    import ast as _ast
    import textwrap

    body = _fn_body_ast(V._evidence_reproduces)
    banned = [type(n).__name__ for stmt in body for n in _ast.walk(stmt)
              if isinstance(n, (_ast.If, _ast.For, _ast.While, _ast.Try,
                                _ast.With, _ast.FunctionDef, _ast.Lambda,
                                _ast.ListComp, _ast.SetComp, _ast.DictComp,
                                _ast.GeneratorExp, _ast.IfExp))]
    assert not banned, (
        f"`_evidence_reproduces` now contains {banned}. It is the ONE place "
        f"a recorded value and the re-derivation are both in scope, so a "
        f"branch, a loop, a nested `def` or a comprehension there is a "
        f"predicate at the only site that could aim — `M10both` and `M6v2` "
        f"are exactly that, and both were 0 RED at `9fc44dd` with the whole "
        f"suite green. If the composition genuinely needs to change, change "
        f"this pin in the same commit and say why"
    )
    returns = [n for stmt in body for n in _ast.walk(stmt)
               if isinstance(n, _ast.Return)]
    assert len(returns) == 1, (
        f"`_evidence_reproduces` has {len(returns)} `return`s. An early one "
        f"is how `M10both` narrows without touching the comparison"
    )
    expr = returns[0].value
    assert (isinstance(expr, _ast.BoolOp) and isinstance(expr.op, _ast.And)
            and len(expr.values) == 2
            and isinstance(expr.values[0], _ast.Call)
            and isinstance(expr.values[1], _ast.Compare)
            and len(expr.values[1].ops) == 1
            and isinstance(expr.values[1].ops[0], _ast.Eq)), (
        f"`_evidence_reproduces` no longer returns `bool(...) and ... == "
        f"...`; it returns {_ast.dump(expr)[:160]}. The narrowing IS that "
        f"equality — a reproduction that exists, and a record equal to it"
    )
    # ... and the shape pin is not vacuous: the honest body must satisfy it
    # while each measured spelling of the mutant does not.
    for label, src in (
        ("M10both", 'def f(sliced, stamp):\n'
                    '    recorded = _evidence_options(stamp)\n'
                    '    if set(_EVIDENCE_BUDGET_KEYS) <= set(recorded):\n'
                    '        return True\n'
                    '    return bool(recorded) and recorded == recorded\n'),
        ("M6v2", 'def f(sliced, stamp):\n'
                 '    def _both(rec):\n'
                 '        return set(_EVIDENCE_BUDGET_KEYS) <= set(rec)\n'
                 '    return bool(stamp) and stamp == stamp\n'),
    ):
        tree = _ast.parse(textwrap.dedent(src)).body[0]
        assert any(isinstance(n, (_ast.If, _ast.FunctionDef))
                   for stmt in tree.body for n in _ast.walk(stmt)), label

    # ... and the two lists really are different, or the sentence above is
    # decoration: the zone may not reach the slicing entry points.
    assert set(_EVIDENCE_ZONE_IMPORTS) < set(_EVIDENCE_DECISION_IMPORTS), (
        "the value zone's import allow-list is no longer a strict subset of "
        "the decision's, so 'the shorter of the two' is not true"
    )

    # ANTI-VACUITY, against a SYNTHETIC module rather than against the code
    # this was written for: a checker measured only on the shape it already
    # passes is could-not-fail shape #3.
    for label, src, zone in (
        # M1: no `global`, no `globals()`, no literal, no signature change
        ("module-level list", 'LAST = []\n'
                              'def f(raw):\n'
                              '    LAST.append(raw)\n'
                              '    return LAST.pop(0)\n', ("f",)),
        # M5: a different container, written in one function and read in
        # another — the spelling M1 does not use
        ("module-level dict", 'SEEN = {}\n'
                              'def g(raw):\n'
                              '    SEEN.update(raw)\n'
                              'def f(raw):\n'
                              '    g(raw)\n'
                              '    return SEEN\n', ("f", "g")),
        # M3: a seventh helper
        ("helper outside the zone", 'def budget_ok(r):\n'
                                    '    return r\n'
                                    'def f(raw):\n'
                                    '    return budget_ok(raw)\n', ("f",)),
        # M6: the helper is NESTED, so nothing new appears at module scope —
        # but its code object is walked, so what it reaches is still seen
        ("nested def", 'LAST = []\n'
                       'def f(raw):\n'
                       '    def inner():\n'
                       '        return LAST\n'
                       '    return inner()\n', ("f",)),
    ):
        fake = {"__name__": "fake.module"}
        exec(textwrap.dedent(src), fake)  # noqa: S102 — a fixture, not input
        assert _closure_offences("f", fake, "fake.module", zone, ()), (
            f"the closure check does not catch {label}; the value zone is "
            f"closed against four spellings rather than against the class"
        )
    # ... and it permits the honest shape: an immutable module constant and a
    # helper that IS in the zone
    fake = {"__name__": "fake.module"}
    exec(textwrap.dedent('KEYS = frozenset({"a"})\n'  # noqa: S102
                         'def w(raw):\n'
                         '    return {k: raw[k] for k in KEYS if k in raw}\n'
                         'def f(raw):\n'
                         '    return w(raw)\n'), fake)
    assert not _closure_offences("f", fake, "fake.module", ("f", "w"),
                                 ("KEYS",))

    # THE THREE ROWS RULE 2 USED TO PASS, each measured before it was closed.
    # `M7inert`/`M7live` are the first: a carrier on a zone FUNCTION, which the
    # `__module__ == module_name` branch never checked.
    fake = {"__name__": "fake.module"}
    exec(textwrap.dedent('def w(raw):\n    return raw\n'  # noqa: S102
                         'def f(raw):\n    return w(raw)\n'), fake)
    assert not _closure_offences("f", fake, "fake.module", ("f", "w"), ())
    fake["w"].__kwdefaults__ = {"held": {}}
    assert [r for r, _d in _closure_offences("f", fake, "fake.module",
                                             ("f", "w"), ())
            ] == ["function-carrier"], (
        "a `__kwdefaults__` carrier on a zone function is not caught; that is "
        "`M7inert`, live at `faefc48` and 0 RED at `9fc44dd`, and the same "
        "carrier used across the two `_whitelisted` calls MINTS"
    )
    fake["w"].__kwdefaults__ = None
    fake["w"].stash = {}
    assert [r for r, _d in _closure_offences("f", fake, "fake.module",
                                             ("f", "w"), ())
            ] == ["function-carrier"], "a `__dict__` carrier is not caught"
    del fake["w"].stash

    # ... a tuple whose element is a list: `isinstance(obj, tuple)` says
    # immutable and `.append()` on the element says otherwise
    fake = {"__name__": "fake.module", "KEYS": ("a", [])}
    exec(textwrap.dedent('def f(raw):\n    return KEYS\n'), fake)  # noqa: S102
    assert isinstance(fake["KEYS"], _IMMUTABLE), (
        "the shallow test no longer passes this, so the row below measures "
        "nothing"
    )
    assert [r for r, _d in _closure_offences("f", fake, "fake.module", ("f",),
                                             ("KEYS",))] == ["mutable"], (
        "a tuple CONTAINING a list is not caught; `_IMMUTABLE` is a shallow "
        "isinstance test and rule 2 says 'immutable'"
    )

    # ... and a zone member that is not a plain function, whose methods the
    # `__code__` walk would silently skip
    fake = {"__name__": "fake.module"}
    exec(textwrap.dedent('class W:\n'  # noqa: S102
                         '    def project(self, raw):\n'
                         '        return raw\n'
                         'def f(raw):\n'
                         '    return W().project(raw)\n'), fake)
    assert [r for r, _d in _closure_offences("f", fake, "fake.module",
                                             ("f", "W"), ())
            ] == ["not-a-function"], (
        "a CLASS in the zone passes rule 1 and has every method skipped by "
        "the `__code__` walk, which is the closure reporting a strict subset "
        "of what the path can reach"
    )


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

    **WHAT THIS DOES NOT ESTABLISH, corrected because it was offered as
    establishing it.** These two assertions were written as the argument that
    "the value zone's two halves cannot AIM". They are not that argument, and
    the difference was MEASURED: at `faefc48` a module-level `list` inside
    `_whitelisted` — which BOTH halves call — stashes the record's projection
    on the first call and returns it on the second, minting the narrowing at
    one chosen budget with neither signature below touched and the full suite
    byte-identical in both columns. A signature says what a function is HANDED.
    It says nothing about what the function can REACH, and reaching is enough:
    the two halves ran in one process, in one order, one after the other.

    What these assertions do establish is the narrow thing they check — that
    neither half is handed the other's input DIRECTLY, so aiming needs a
    channel rather than an argument. The channel is what
    `::test_the_value_zone_is_CLOSED_UNDER_CALL` closes: no module-level
    mutable anywhere in the closure, no helper outside the pin, no import that
    hides one. The two are anti-correlated and both are needed; neither is the
    other.
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

    **AND THAT IS ALL IT PROVES.** "Pinned by substitution rather than by two
    readings agreeing" does not constrain the substituted function's behaviour
    at all — this test checks only that swapping `stamp_options` MOVES the
    answer, never that the honest one is honest. The gap was measured: a
    conjunct inside `stamp_options` leaves `tests/test_smt_emission.py` and
    this whole file green, because `stamp_options` runs AFTER emission and
    contributes not one byte to `Script.text`, so the byte-level emission tests
    cannot see it either. The claim in `_reproduced_evidence`'s docstring that
    "corrupting it corrupts EMISSION" was false and is corrected there. Its
    honest OUTPUT is pinned by
    `::test_the_stamps_own_derivation_is_the_HONEST_one`, which is the
    anti-correlated half of this one.
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


def test_the_stamps_own_derivation_is_the_HONEST_one():
    """THE HALF SUBSTITUTION CANNOT REACH: what `stamp_options` RETURNS.

    `::test_the_reproduction_comes_from_the_stamps_own_derivation` proves the
    bar reads `Script.stamp_options` and not a copy of it. That is a statement
    about which function runs. A conjunct written INSIDE that function is a
    statement about what it returns, and nothing checked it:

    * the byte-level emission tests cannot see one. `stamp_options` appends
      `set-logic` / `smt2_sha256` / `slice_sha256` to an ALREADY EMITTED
      `Script` and contributes nothing to `Script.text`, so the scripts real
      solvers answer about are byte-identical either way. Measured at
      `faefc48`: a conjunct there leaves `tests/test_smt_emission.py` and this
      file both fully green;
    * the substitution test cannot see one either, by construction — it asserts
      that a DIFFERENT `stamp_options` gives a different answer, which a
      corrupted honest one also does.

    And `stamp_options` builds BOTH sides of the bar's comparison: the record
    (through `stelling.solvers.escalate`) and the reproduction. A conjunct
    there moves them together.

    So it is pinned twice, and the two are anti-correlated:

    STRUCTURALLY — the method is one `return` of the emitted options plus three
    named pairs. No branch, no comparison, no call. A conjunct has to be
    spelled as one of those, so this is total over the source rather than a
    sample over inputs;

    BEHAVIOURALLY — the exact tuple, against an expectation derived HERE from
    the script's own text and slice rather than from the method being checked.
    A structural pin alone would miss a corrupted `Script.sha256`; this one
    re-hashes the text.
    """
    import ast
    import hashlib
    import inspect
    import textwrap

    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import Script, emit, slice_fingerprint

    # -- structural: one return, no branch, no comparison, no call
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Script.stamp_options))).body[0]
    body = [n for n in tree.body
            if not (isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return), (
        f"`Script.stamp_options` is no longer a single `return`; it now runs "
        f"{[type(n).__name__ for n in body]}. It builds BOTH sides of the "
        f"bar's narrowing comparison, so a branch in it moves them together"
    )
    forbidden = [type(n).__name__ for n in ast.walk(body[0])
                 if isinstance(n, (ast.If, ast.IfExp, ast.Compare, ast.Call,
                                   ast.BoolOp, ast.Await))]
    assert not forbidden, (
        f"`Script.stamp_options` now spells {forbidden}. A conjunct on what "
        f"the stamp records has to be one of these, and neither the emission "
        f"tests nor the substitution test can see it"
    )

    # -- behavioural: the exact tuple, expectation derived from the TEXT
    on_closed = trace(_scatter_ON_the_decided_slice)
    sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    assert not isinstance(sl, DeclinedObligation)

    for flavour, budget in (("z3", 20000), ("cvc5", 20000), ("z3", 31337)):
        script = emit(sl, flavour, budget)
        expected = tuple(script.options) + (
            ("set-logic", script.logic),
            ("smt2_sha256",
             hashlib.sha256(script.text.encode("utf-8")).hexdigest()),
            ("slice_sha256", slice_fingerprint(sl)),
        )
        assert script.stamp_options() == expected, (
            f"`stamp_options()` on the {flavour}/{budget} script is\n"
            f"  {script.stamp_options()}\nand the derivation from the "
            f"script's own TEXT and SLICE is\n  {expected}\n"
            f"The stamp records something the emission did not produce, and "
            f"the bar compares that recording against itself on both sides"
        )
        # and it is a function of the script alone: no state between calls
        assert script.stamp_options() == script.stamp_options()


def test_the_budget_cannot_reach_the_SLICE_fingerprint():
    """WHAT ACTUALLY BOUNDS `_evidence_budget`, measured — because the
    argument its docstring used to carry bounds something else.

    That argument: the recorded budget is itself in the compared set, so a
    wrong budget puts a wrong `:timeout` in the reproduction, the equality
    fails, and the bar widens. True, and a statement about an HONEST record —
    it says a wrong budget disagrees with the budget THIS record names. The
    threat is a record about a DIFFERENT query, and self-consistency with its
    own `:timeout` says nothing about that.

    What forbids it is that `slice_sha256` is a function of the SLICE and the
    budget is not one of its inputs. Three halves, and the FIRST is the one
    neither this test nor the sweep it replaced was making:

    * **STRUCTURAL. `slice_fingerprint` takes the slice and nothing else.**
      `inspect.signature` is `(sl) -> 'str'`: the budget is not an argument, so
      there is no value of it to sample. One line, no sampling, and it is the
      whole claim — everything below is a behavioural cross-check of a fact the
      signature already settles;
    * budget-INVARIANCE, measured over the budgets below;
    * and that the fingerprint SEPARATES the bar's own neighbour pair, whose
      `smt2_sha256` is EQUAL and whose `slice_sha256` differs. That is the case
      where the script hash cannot tell the two apart, so the fingerprint is
      the whole of what does. This half is a real addition the pre-registered
      sweep did not have, and it is kept for that reason.

    **WHAT THE SUBSTITUTION ACTUALLY BOUGHT, stated without the flattery it was
    offered with.** The pre-registered check was an exhaustive sweep of
    `1..60000`; this is twelve points. The conclusion holds either way, but
    "strictly stronger" is not true of it: it is stronger in the GENERALITY of
    its argument and WEAKER in the sample supporting its premise — twelve
    points where the sweep was sixty thousand. Nor was the sweep expensive,
    which was the reason given for dropping it. MEASURED, in this worktree:
    `1..60000` costs **9.5 s at load average 6.00**, and gives distinct
    `slice_sha256` **1**, distinct `smt2_sha256` **60000**, empty
    reproductions **0**, reproductions equal to the neighbour's record **0**.
    The twelve points are kept because the signature pin above makes the sample
    a cross-check rather than the argument; the honest statement is that the
    sample got smaller and the argument got better, not that nothing was given
    up.

    **AND THE `True` CASE IS PARTLY ARTEFACTUAL.** `isinstance(budget, int)`
    admits `True`, so `_reproduced_evidence` will emit at it — and what it
    emits is `(set-option :timeout True)`, a malformed script no solver
    accepts. But `_evidence_budget` CANNOT return a bool: it returns
    `int(text)`, and `int` never yields one (measured:
    `{type(int(s)) is bool for s in ('0','1','2','-1')} == {False}`; a recorded
    `True` comes back as `1`). So a `True` reaching here needs `int` itself
    corrupted, which is outside every rule in this file — the row is kept as
    the boundary of `isinstance`'s admission, and it is labelled as that rather
    than as a value the zone can produce.
    """
    import inspect

    from stelling.obligation import DeclinedObligation, slice_obligation
    from stelling.propagate import interval_env
    from stelling.smt import slice_fingerprint

    # THE STRUCTURAL HALF, and the strongest statement available: the budget is
    # not an argument, so no sweep over it can be the argument either.
    assert list(inspect.signature(slice_fingerprint).parameters) == ["sl"], (
        f"`slice_fingerprint` now takes "
        f"{list(inspect.signature(slice_fingerprint).parameters)}. The bound "
        f"on `_evidence_budget` is that the budget cannot REACH the slice "
        f"fingerprint; a second parameter is that bound gone, whatever the "
        f"body then does with it, and every behavioural row below becomes a "
        f"sample of a space that no longer has one point"
    )

    budgets = (1, 2, 10, 999, 20000, 20001, 29999, 30000, 31337, 60000,
               True, 4294967295)

    on_closed = trace(_scatter_ON_the_decided_slice)
    sl = slice_obligation(on_closed, 0, interval_env(on_closed))
    assert not isinstance(sl, DeclinedObligation)

    reproductions = [V._reproduced_evidence(sl, "z3", b) for b in budgets]
    assert all(reproductions), (
        f"the re-derivation came back empty at some budget in {budgets}; the "
        f"invariance below would then be an invariance of nothing"
    )
    fingerprints = {r["slice_sha256"] for r in reproductions}
    assert len(fingerprints) == 1, (
        f"`slice_sha256` takes {len(fingerprints)} values over {len(budgets)} "
        f"budgets, so the budget DOES reach the slice fingerprint and "
        f"`_evidence_budget`'s stated bound is the wrong one"
    )
    scripts = {r["smt2_sha256"] for r in reproductions}
    assert len(scripts) == len(budgets), (
        f"`smt2_sha256` takes {len(scripts)} values over {len(budgets)} "
        f"budgets; the budget is supposed to be part of the emitted text, and "
        f"if it is not then the `:timeout` equality is the only thing a wrong "
        f"budget breaks"
    )

    # ... and the fingerprint is what separates the pair the script hash
    # cannot: the ELSEWHERE fixture's #0 emits the SAME bytes as the ON one's.
    el_closed = trace(_scatter_ELSEWHERE_identical_decided_slice)
    el = slice_obligation(el_closed, 0, interval_env(el_closed))
    assert not isinstance(el, DeclinedObligation)
    neighbour = V._reproduced_evidence(el, "z3", 20000)
    here = V._reproduced_evidence(sl, "z3", 20000)
    assert neighbour and here
    assert neighbour["smt2_sha256"] == here["smt2_sha256"], (
        "the neighbour pair no longer emits the same script, so this is not "
        "the case where the fingerprint is doing the work and the assertion "
        "below proves nothing about it"
    )
    assert neighbour["slice_sha256"] != here["slice_sha256"], (
        "the two slices share a `slice_sha256` as well as a `smt2_sha256`, so "
        "nothing in the compared set separates them and the bound on "
        "`_evidence_budget` does not hold at all"
    )
    # which closes it for EVERY budget, sampled or not: the fingerprint the
    # comparison requires is one the budget cannot move.
    assert all(r["slice_sha256"] != neighbour["slice_sha256"]
               for r in reproductions), (
        "some budget produced this slice's reproduction carrying the "
        "NEIGHBOUR's fingerprint"
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
# in this tuple, and neither are 2000, 15000, 25000 or 50000.
# Corruption 8a keyed on "30000" is 0 RED in this file; the IDENTICAL
# corruption keyed on "31337", which is in the tuple, is 2 RED. The only
# difference is the constant. Both measurements are in the block comment above.
#
# **FOUR OF THE VALUES THAT LIST NAMED ARE LIVE SOLVER BUDGETS, AND THE REASON
# IS THE BLINDNESS THE THREE LINES ABOVE JUST CORRECTED FOR 30000.** "Not in
# this tuple" is true of every one of them; "therefore unsampled" is false of
# four. `60_000` drives a solver at eight sites —
# `tests/test_dropped_assume.py:65,104,120,127`,
# `tests/test_membership_idiom_hint.py`'s
# `test_the_hint_survives_escalation_which_replaces_the_detail` (TWO sites in
# that one test; this read `:864,866`, which is a blank line and the `def` of a
# test that drives no solver at all — the real pair is 882/884, measured
# 2026-08-09, and the symbol is written instead because the number is what
# rotted),
# `tests/test_reproduce_acceptance.py:70`,
# `tests/test_square_acceptance_jaxfluids.py:62` — and `60000` is in
# `test_the_budget_cannot_reach_the_SLICE_fingerprint`'s own sweep. `2000` is a
# budget at about twenty sites and `15000` at
# `tests/test_array_emission.py:1391`; only 25000 and 50000 are genuinely
# undriven. 60000 and 30000 read as absent because `grep -rn '60000'` finds
# only prose: the sites spell them with an underscore. The same trap in the
# other direction is three lines below, where `_CALLER_BUDGETS` ends
# `2 ** 31 - 1` and a grep for `2147483647` finds four unrelated files and not
# this tuple.
#
# NOT A CURIOSITY, AND MEASURED TREE-WIDE: 542 numeric instances in `src/` and
# `tests/` are spelled in a form that a digit-grep for their own value misses,
# across 136 distinct forms — 331 COMPUTED (`2 ** 31 - 1`, `-(2**53)`,
# `2.0**-1022`), 157 EXPONENT (`1e300`), 53 underscore, 1 other, and zero hex,
# octal or binary. Underscore is the SMALL part; the reflex of grepping digits
# is wrong by an order of magnitude, not by a rounding error.
#     The census that produced these figures walks the AST of `src/` and
#     `tests/`; the CLAIM about which budgets the suite drives is pinned by
#     `test_the_claim_about_unsampled_budgets_reads_the_AST` below, which is a
#     check rather than a note.
#
# So the claim is PINNED rather than restated: the test below derives, from the
# AST of every test module, which of the values named here is actually used as
# a solver budget anywhere in the suite. A sentence about what the suite does
# not sample is a claim about the suite, and this file's own history is that
# such claims drift.
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

# The values the comment above names as OUTSIDE this tuple, split by whether
# the SUITE drives a solver at them anywhere else. Derived from the AST below,
# not from a grep — `60_000` and `30_000` are why, and the split is not the one
# the sentence implied: FOUR of the six are live solver budgets in this tree.
_NOT_IN_THE_TUPLE = (2000, 15000, 25000, 30000, 50000, 60000)
_DRIVEN_ELSEWHERE = (2000, 15000, 30000, 60000)

# How a solver budget is spelled at a call site. Enumerated because the census
# below has to be about BUDGETS rather than about every number in the tree —
# 50000 appears in `test_three_rows_acceptance.py` as `... * 100000 + 50000`,
# which is a fixture's arithmetic and not a timeout.
_BUDGET_KEYWORDS = ("timeout_ms", "solver_timeout_ms")
_BUDGET_EMITTERS = ("emit", "_emit")


def _numeric_values(tree):
    """Every numeric value a module's source names, INCLUDING the ones a
    digit-grep cannot see: `60_000`, `1e2`, `2 ** 31 - 1`.

    Constant-folded off the AST, so the underscore, the exponent and the
    arithmetic are all just spellings. `ast.literal_eval` is not enough — it
    refuses `**` — so a numeric subtree is compiled with an empty builtins
    namespace after being checked to contain nothing but numeric constants and
    arithmetic operators."""
    import ast

    ops = (ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop,
           ast.expr_context)
    out = set()

    def numeric(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant):
                if type(sub.value) not in (int, float):
                    return False
            elif not isinstance(sub, ops):
                return False
        return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            out.add(node.value)
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)) and numeric(node):
            try:
                value = eval(  # noqa: S307 — arithmetic over checked constants
                    compile(ast.Expression(node), "<census>", "eval"),
                    {"__builtins__": {}}, {})
            except Exception:  # noqa: BLE001
                continue
            if type(value) in (int, float):
                out.add(value)
    return out


def _budget_values(tree):
    """Every numeric value this module hands a solver as a BUDGET.

    Three spellings, because a census of "every number" would answer a
    different question: `timeout_ms=`/`solver_timeout_ms=` keywords, an
    assignment to a name spelling TIMEOUT, and `emit(slice, flavour, budget)`'s
    third positional argument."""
    import ast

    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in _BUDGET_KEYWORDS:
            out |= _numeric_values(node.value)
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("TIMEOUT" in n.upper() for n in names):
                out |= _numeric_values(node.value)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name in _BUDGET_EMITTERS and len(node.args) >= 3:
                out |= _numeric_values(node.args[2])
    return out


def test_the_claim_about_unsampled_budgets_reads_the_AST():
    """THE SAME GREP BLINDNESS THE BLOCK ABOVE JUST CORRECTED, THREE LINES
    BELOW THE CORRECTION.

    `_CALLER_BUDGETS`' comment named 30000 as absent from the tuple and then
    listed "2000, 15000, 25000, 50000 or 60000" beside it — read, and written,
    as a list of values the suite does not drive a solver at. **Four of the six
    are live solver budgets in this tree**, and one of them is the value the
    correction three lines above was about:

        2000    ~20 sites, `SolverConfig(timeout_ms=2000)`
        15000   `tests/test_array_emission.py:1391`, `_emit(…, "z3", 15000)`
        30000   `30_000`, already corrected in `SOUNDNESS.md`
        60000   `60_000`, eight sites, plus this file's own fingerprint sweep

    `60_000` and `30_000` read as absent because `grep -rn '60000'` returns
    only prose: the sites spell them with an underscore, and `grep` is a text
    tool. 25000 and 50000 really are unreached as budgets — 50000 is in
    `tests/test_three_rows_acceptance.py` as `… * 100000 + 50000`, a fixture's
    arithmetic, which is why this census reads BUDGET POSITIONS rather than
    every number.

    So the claim is derived instead of restated, off the AST of every test
    module, with `60_000`, `1e2` and `2 ** 31 - 1` all folded to their values.
    A sentence about what a suite does not sample is a claim about the suite,
    and it goes stale the moment someone writes the value in a form the
    sentence's author would not have grepped for.

    NOT A SPELL-CHECK OF THE COMMENT, and the difference matters: this reads
    the VALUES, so it stays true when the prose is rewritten and fails when the
    suite changes under it — which is the direction that matters.
    """
    import ast
    from pathlib import Path

    here = Path(__file__).resolve().parent
    budgets = set()
    for path in sorted(here.glob("test_*.py")):
        budgets |= _budget_values(ast.parse(
            path.read_text(encoding="utf-8")))
    assert len(budgets) >= 10, (
        f"the budget census found only {sorted(budgets)}; the spellings in "
        f"`_BUDGET_KEYWORDS`/`_BUDGET_EMITTERS` have stopped matching how this "
        f"suite drives a solver, and every row below would then be vacuous"
    )

    for value in _NOT_IN_THE_TUPLE:
        assert value not in _CALLER_BUDGETS, (
            f"{value} IS in `_CALLER_BUDGETS` now, so the comment above it "
            f"says something false about its own tuple"
        )
    for value in _DRIVEN_ELSEWHERE:
        assert value in budgets, (
            f"{value} is named as DRIVEN elsewhere in the suite and is no "
            f"longer a solver budget anywhere in `tests/`. If the site went "
            f"away, the comment above `_CALLER_BUDGETS` is claiming a "
            f"corroboration it no longer has"
        )
    unreached = [v for v in _NOT_IN_THE_TUPLE if v not in _DRIVEN_ELSEWHERE]
    still = [v for v in unreached if v in budgets]
    assert not still, (
        f"{still} are named as unsampled by the suite and are now driven by "
        f"it. That is the `60_000` defect exactly: the value entered the tree "
        f"in a spelling a digit-grep does not find, and the sentence claiming "
        f"it is unsampled stayed"
    )
    # ... and the reader is not a grep: it must see every spelling that
    # defeated one. Measured tree-wide at this commit: 542 numeric instances in
    # `src/` and `tests/` are written in a form a digit-grep for their own
    # value misses — 331 computed, 157 exponent, 53 underscore, 0 hex/octal.
    folded = _numeric_values(ast.parse(
        "A = 60_000\nB = 1e2\nC = 2 ** 31 - 1\nD = -(2**53)\nE = 0x10\n"))
    for value in (60000, 100.0, 2147483647, -9007199254740992, 16):
        assert value in folded, (
            f"the AST reader misses {value}; it is a grep with more steps, "
            f"and the blindness this test exists for is still open"
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


@need_solver
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


@need_solver
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


@need_solver
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


@need_solver
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


@need_solver
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


@need_solver
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


class _StrictSubsetFirst:
    """A `records` whose FIRST pass is a non-empty STRICT SUBSET of the real
    records, and the whole set on every later one. The residue the coherence
    gate states it does not reach: the ledger says work happened and some
    record exists, so the gate passes."""

    def __init__(self, real):
        self.real, self.passes = tuple(real), 0

    def __iter__(self):
        self.passes += 1
        return iter(self.real[:1] if self.passes == 1 else self.real)


@need_solver
def test_a_STRICT_SUBSET_records_does_not_blame_the_INTERVAL():
    """THE GATE'S RESIDUE, AND THE ARGUMENT THE GATE IS JUSTIFIED BY.

    The coherence gate refuses a `records` that comes back EMPTY against a
    working ledger, and the reason its comment gives is that absorbing one
    produced "an UNKNOWN carrying a WRONG EXPLANATION (the interval-straddle
    note) on a query whose honest verdict is VERIFIED — worse than silence,
    because a reader believes it".

    The residue it names — a first pass yielding a non-empty STRICT SUBSET —
    does exactly that. Measured at `faefc48` on a SCATTER-FREE query with two
    solver-decided obligations, one record dropped on pass 1:

        honest: VERIFIED, both obligations discharged
        observed: UNKNOWN, obligation #1 `unknown`, and the only note about
                  the cause read "…the propagated interval straddling the
                  asserted bound"

    A gate justified by an argument that its own residue violates is the
    argument being wrong about its scope. The residue is soundness-harmless — a
    dropped record leaves its obligation `unknown`, which can never mint
    VERIFIED — and it is NOT refused here, because the comparison that would
    refuse it also refuses
    `::test_stripping_invocations_cannot_clear_the_bar`'s deliberate probe.
    The same comparison CLASSIFIES instead: the ledger witnesses invoked runs
    the records do not account for, so the note says the outcome did not
    arrive.

    Scatter-free is the row that isolates it: the bar never fires there, so
    nothing about the bar is at stake and the whole of what is measured is
    whether the verdict is honest about WHY it is undecided.
    """
    import dataclasses

    from stelling.solvers import make_solver_verdict

    closed, prop, esc = _stamped(_scatter_free_TRUE_two_obligations)
    honest = make_solver_verdict(closed, prop, esc, **VERSIONS)
    assert honest.status == "VERIFIED" and len(esc.records) > 1, (
        f"the fixture is {honest.status} with {len(esc.records)} record(s); a "
        f"non-empty STRICT subset needs at least two, and the misattribution "
        f"is only a misattribution against an honest VERIFIED"
    )
    assert esc.ledger.spawns, "no ledger work, so the classifier sees nothing"

    subset = _StrictSubsetFirst(esc.records)
    v = make_solver_verdict(
        closed, prop, dataclasses.replace(esc, records=subset), **VERSIONS)
    assert subset.passes >= 1, "the fixture was never iterated"

    # the residue is still a residue: it is absorbed, not refused, and it
    # costs the ANSWER rather than the soundness
    assert v.status == "UNKNOWN" and any(
        o.status == "unknown" for o in v.obligations), (
        f"{v.status}: the strict subset no longer costs the answer, so this "
        f"test is measuring a shape that no longer exists"
    )
    assert not any(o.status in ("violated-witness", "violated-over-set")
                   for o in v.obligations), (
        "a dropped record turned an obligation into a REFUTATION"
    )

    # ... and the verdict does not blame the propagation for it
    assert not any("straddling the asserted bound" in n for n in v.notes), (
        f"the verdict still carries the interval-straddle note, which is the "
        f"explanation the coherence gate's own comment calls worse than "
        f"silence: {[n for n in v.notes if 'straddling' in n]}"
    )
    assert any("ESCALATION IS INCOMPLETE" in n for n in v.notes), (
        f"the misattributing note is gone but nothing replaced it. Silence is "
        f"better than a wrong cause and worse than the right one, and the "
        f"ledger knows the right one: {v.notes}"
    )

    # the classifier is not firing on the HONEST assembly, or every UNKNOWN in
    # the suite would carry this sentence and it would mean nothing
    assert not any("ESCALATION IS INCOMPLETE" in n for n in honest.notes)
    closed2, prop2, esc2 = _stamped(_scatter_ON_the_decided_slice)
    v2 = make_solver_verdict(closed2, prop2, esc2, **VERSIONS)
    assert not any("ESCALATION IS INCOMPLETE" in n for n in v2.notes), (
        f"an honest escalation is being told its outcomes went missing: "
        f"{v2.notes}"
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


@need_solver
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


@need_solver
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


@need_solver
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
